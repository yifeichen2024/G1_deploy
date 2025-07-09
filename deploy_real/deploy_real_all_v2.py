#!/usr/bin/env python3
"""
Multi-policy ONNX + Loco TorchScript deploy for Unitree G1
=========================================================

  • Motion-mode : 多个 ONNX 动作策略（原始逻辑完全保留）
  • Loco-mode   : TorchScript 行走策略（左摇杆 XY、右摇杆 X 控制）
  • 状态机      : Zero-torque → Default → 等待 A → Stance
  • 热键        : L1/R1 切 Motion   |  L2+Y 进 Loco   |  R2+Y 回 Stance
  • 退出        : SELECT
  • 复位        : X（重回 Default pose）

本脚本仅在原版基础上插入 **LocoTorchPolicy** 与相应调用，
尽量不触碰 Motion 相关代码。
"""

from __future__ import annotations
from typing import List, Union
import argparse, sys, time, datetime as _dt, os

import numpy as np
import yaml
import torch
import onnxruntime

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__LowCmd_,
    unitree_hg_msg_dds__LowState_,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG, LowState_ as LowStateHG
from unitree_sdk2py.utils.crc import CRC

from common.command_helper import (
    create_zero_cmd,
    create_damping_cmd,
    init_cmd_hg,
    MotorMode,
)
from common.rotation_helper import get_gravity_orientation, transform_imu_data
from common.remote_controller import RemoteController, KeyMap
from config import Config

# ════════════════════════════════════════════════════════════════
# 1.  Loco TorchScript policy（新增，Motion 相关代码未动）
# ════════════════════════════════════════════════════════════════
class LocoTorchPolicy:
    """TorchScript 行走策略；返回 *delta-action*（与 Motion 缩放保持一致）"""
    def __init__(self, loco_yaml: str, motion_cfg: Config):
        with open(loco_yaml, "r") as f:
            cfg = yaml.safe_load(f)
        self.cfg        = cfg
        self.motion_cfg = motion_cfg
        self.na  = cfg["num_actions"]
        self.j2m = np.array(cfg["action_joint2motor_idx"], np.int32)

        # 载入 TorchScript
        self.ts = torch.jit.load(cfg["policy_path"])
        self.ts.eval()
        with torch.inference_mode():
            self.ts(torch.zeros(1, cfg["num_obs"], dtype=torch.float32))

        # 一些常量
        self.act_prev = np.zeros(self.na, np.float32)
        self.obs      = np.zeros(cfg["num_obs"], np.float32)
        self.r_vx = np.array(cfg["cmd_range"]["lin_vel_x"], np.float32)
        self.r_vy = np.array(cfg["cmd_range"]["lin_vel_y"], np.float32)
        self.r_vz = np.array(cfg["cmd_range"]["ang_vel_z"], np.float32)

    def _scale_cmd(self, raw: np.ndarray) -> np.ndarray:
        out = np.zeros(3, np.float32)
        for i, rg in enumerate((self.r_vx, self.r_vy, self.r_vz)):
            a, b = rg ; out[i] = (raw[i] - a) / (b - a) * 2 - 1
            out[i] = np.clip(out[i], -1, 1)
        return out

    # --- 核心前向 ----------------------------------------------------------------
    def step(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        ang_vel: np.ndarray,
        gravity: np.ndarray,
        vel_cmd: np.ndarray,
    ) -> np.ndarray:
        cfg = self.cfg
        cmd = self._scale_cmd(vel_cmd) * cfg["cmd_scale"]

        # 构造观测
        self.obs[0:3]      = ang_vel * cfg["ang_vel_scale"]
        self.obs[3:6]      = gravity
        self.obs[6:9]      = cmd
        self.obs[9 : 9+self.na]           = (q[self.j2m] - cfg["default_angles"]) * cfg["dof_pos_scale"]
        self.obs[9+self.na : 9+2*self.na] = dq[self.j2m] * cfg["dof_vel_scale"]
        self.obs[9+2*self.na:]            = self.act_prev

        with torch.inference_mode():
            act = (
                self.ts(torch.from_numpy(self.obs).unsqueeze(0))
                .clip(-100, 100)
                .squeeze()
                .cpu()
                .numpy()
            )
        self.act_prev = act

        # 计算 *目标* 关节角
        tgt = cfg["default_angles"] + act * cfg["action_scale"]

        # 转成与 Motion 相同的 delta-action（便于复用原 _post_process_and_send）
        delta = (tgt - self.motion_cfg.default_angles) / self.motion_cfg.action_scale
        return np.clip(delta, -1.0, 1.0)


# ════════════════════════════════════════════════════════════════
# 2.  原 Controller + Loco 增强
# ════════════════════════════════════════════════════════════════
class Controller:
    """保持原 Motion 逻辑，仅额外插入 Loco-mode 支持"""

    def __init__(self, config: Config, loco_yaml: str):
        self.config = config
        self.remote_controller = RemoteController()
        self.control_dt = config.control_dt

        # ---------- Motion-policy 加载（原逻辑） ----------
        policy_paths: List[str]
        if isinstance(config.policy_paths, str):
            policy_paths = [config.policy_paths]
        else:
            policy_paths = list(config.policy_paths)
        assert len(policy_paths) > 0, "No motion policy paths!"

        self.ort_sessions = [onnxruntime.InferenceSession(p) for p in policy_paths]
        self.input_names  = [s.get_inputs()[0].name for s in self.ort_sessions]
        self.policy_idx   = 0      # 当前 Motion policy
        self.stance_idx   = 0
        self.motion_lens  = list(config.motion_lens)
        self.motion_len   = self.motion_lens[0]

        # ---------- Loco-policy ----------
        self.loco_policy = LocoTorchPolicy(loco_yaml, config)
        self.mode = "motion"       # {"motion", "loco"}

        # ---------- 运行时缓存（保持不变） ----------
        na, hl = config.num_actions, config.history_length
        self.qj  = np.zeros(na, np.float32)
        self.dqj = np.zeros(na, np.float32)
        self.action = np.zeros(na, np.float32)
        self.counter = 0
        self.ref_motion_phase = 0.0
        self.ang_vel_buf = np.zeros(3 * hl, np.float32)
        self.proj_g_buf  = np.zeros(3 * hl, np.float32)
        self.dof_pos_buf = np.zeros(na * hl, np.float32)
        self.dof_vel_buf = np.zeros(na * hl, np.float32)
        self.action_buf  = np.zeros(na * hl, np.float32)
        self.ref_motion_phase_buf = np.zeros(hl, np.float32)

        self.last_button_state = np.zeros(16, np.int32)
        self.last_switch_time  = time.time()
        self.cooldown_time     = 0.5

        # ---------- DDS 初始化（原逻辑） ----------
        self.low_cmd   = unitree_hg_msg_dds__LowCmd_()
        self.low_state = unitree_hg_msg_dds__LowState_()
        self.lowcmd_pub = ChannelPublisher(config.lowcmd_topic, LowCmdHG) ; self.lowcmd_pub.Init()
        self.lowstate_sub = ChannelSubscriber(config.lowstate_topic, LowStateHG)
        self.lowstate_sub.Init(self._lowstate_cb, 10)

        init_cmd_hg(self.low_cmd, 0, MotorMode.PR)
        print("[DEBUG] Waiting for low-state …")
        self._wait_for_low_state()
        print("[DEBUG] DDS connected!")

    # -----------------------------------------------------------------
    # DDS 回调 & helper
    # -----------------------------------------------------------------
    def _lowstate_cb(self, msg: LowStateHG):
        self.low_state = msg
        self.remote_controller.set(msg.wireless_remote)

    def _wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.control_dt)

    def send_cmd(self, cmd: LowCmdHG):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_pub.Write(cmd)

    # -----------------------------------------------------------------
    # 热键检测（扩展 L2+Y / R2+Y）
    # -----------------------------------------------------------------
    def _check_hotkeys(self):
        cur = self.remote_controller.button
        now = time.time()

        # --- Mode 切换 ---
        if (
            cur[KeyMap.L2]
            and cur[KeyMap.Y]
            and self.mode != "loco"
            and now - self.last_switch_time > self.cooldown_time
        ):
            self.mode = "loco"
            self.last_switch_time = now
            print(">>> Enter LOCO-mode")

        if (
            cur[KeyMap.R2]
            and cur[KeyMap.Y]
            and self.mode == "loco"
            and now - self.last_switch_time > self.cooldown_time
        ):
            self.mode = "motion"
            self.policy_idx = self.stance_idx
            self.motion_len = self.motion_lens[self.stance_idx]
            self.ref_motion_phase = 0.0
            self.counter = 0
            self.last_switch_time = now
            print(">>> Back to Motion-stance")

        # --- Motion 内部切换（仅在 motion 模式下允许） ---
        if self.mode == "motion":
            if (
                cur[KeyMap.L1]
                and not self.last_button_state[KeyMap.L1]
                and now - self.last_switch_time > self.cooldown_time
            ):
                self.switch_policy(-1)
                self.last_switch_time = now
            if (
                cur[KeyMap.R1]
                and not self.last_button_state[KeyMap.R1]
                and now - self.last_switch_time > self.cooldown_time
            ):
                self.switch_policy(+1)
                self.last_switch_time = now

        self.last_button_state = cur.copy()

    def switch_policy(self, delta: int):
        if len(self.ort_sessions) == 1:
            return
        self.policy_idx = (self.policy_idx + delta) % len(self.ort_sessions)
        self.motion_len = self.motion_lens[self.policy_idx]
        self.ref_motion_phase = 0.0
        self.counter = 0
        name = (
            self.config.policy_names[self.policy_idx]
            if hasattr(self.config, "policy_names")
            else f"policy{self.policy_idx}"
        )
        print(f"[INFO] Motion → {name}")


    # ----------------------------------------------------------------------------
    # The following three state phases (zero torque, move to default, default pos)
    # are untouched except for button‑based policy switching capability.
    # ----------------------------------------------------------------------------
    def zero_torque_state(self):
        print("Enter zero‑torque state. Press START to continue …")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            self._check_hotkeys()
            time.sleep(self.config.control_dt)

    def move_to_default_pos(self):
        print("Moving to default pos.")
        # move time 5 s
        total_time = 5
        num_step = int(total_time / self.config.control_dt)
        
        dof_idx = self.config.action_joint2motor_idx + self.config.fixed_joint2motor_idx
        kps = self.config.kps + self.config.fixed_kps
        kds = self.config.kds + self.config.fixed_kds
        default_pos = np.concatenate((self.config.default_angles, self.config.fixed_target), axis=0)
        dof_size = len(dof_idx)
        
        # record the current pos
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q
        
        # move to default pos
        for i in range(num_step):
            alpha = i / num_step
            for j in range(dof_size):
                motor_idx = dof_idx[j]
                target_pos = default_pos[j]
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = kps[j]
                self.low_cmd.motor_cmd[motor_idx].kd = kds[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

        for i in range(dof_size):
            motor_idx = dof_idx[i]
            current_q = self.low_state.motor_state[motor_idx].q
            target_q = default_pos[i]
            err = current_q - target_q
            print(f"[DEBUG] Motor {motor_idx:02d}: target={target_q:+.3f}, actual={current_q:+.3f}, err={err:+.3f}")
        if abs(err) < 0.01:
            print("[INFO] Successfully moved to default pos!")
        else:
            print("[INFO] Failed to move to default pos!")

    def default_pos_state(self):
        print("Enter default pos state.")
        print("Waiting for the Button A signal...")
        while self.remote_controller.button[KeyMap.A] != 1:
            for i in range(len(self.config.action_joint2motor_idx)):
                motor_idx = self.config.action_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.default_angles[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            for i in range(len(self.config.fixed_joint2motor_idx)):
                motor_idx = self.config.fixed_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.fixed_target[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.fixed_kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.fixed_kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)
    # -----------------------------------------------------------------
    # 主控制步
    # -----------------------------------------------------------------
    def run(self):
        self._check_hotkeys()

        # --- Loco 模式 ------------------------------------------------
        if self.mode == "loco":
            # 采集状态
            na = self.config.num_actions
            q  = np.zeros(na, np.float32)
            dq = np.zeros(na, np.float32)
            for i, m in enumerate(self.config.action_joint2motor_idx):
                st = self.low_state.motor_state[m]
                q[i], dq[i] = st.q, st.dq
            quat = self.low_state.imu_state.quaternion
            ang_vel = np.array(self.low_state.imu_state.gyroscope, np.float32)
            gravity = get_gravity_orientation(quat)
            vel_cmd = np.array(
                [self.remote_controller.ly, -self.remote_controller.lx, -self.remote_controller.rx],
                np.float32,
            )
            # Loco 推理 → delta-action
            self.action = self.loco_policy.step(q, dq, ang_vel, gravity, vel_cmd)
            # 直接复用原发送逻辑
            self._post_process_and_send()
            time.sleep(self.control_dt)
            return False  # 不退出

        # --- Motion 模式（原始逻辑保持不变） --------------------------
        self.counter += 1
        obs_buf = self._build_observation()          # 原函数，未改
        sess      = self.ort_sessions[self.policy_idx]
        input_n   = self.input_names[self.policy_idx]
        self.action = np.squeeze(
            sess.run(None, {input_n: obs_buf[None]})[0]
        )
        self._post_process_and_send()

        # 动作结束自动回 stance
        if self.policy_idx != self.stance_idx and self.ref_motion_phase >= 1.0:
            print("[INFO] Motion finished → Stance")
            self.policy_idx = self.stance_idx
            self.motion_len = self.motion_lens[self.stance_idx]
            self.ref_motion_phase = 0.0
            self.counter = 0

        time.sleep(self.control_dt)
        return False

    # -----------------------------------------------------------------
    # 以下 _build_observation 及 _post_process_and_send 与原脚本一致
    # -----------------------------------------------------------------
    def _build_observation(self) -> np.ndarray:
        """Wrapped original obs construction code."""
        # (place the exact logic you had inside `run` here)
        
         # Get the current joint position and velocity
        for i in range(len(self.config.action_joint2motor_idx)):
            self.qj[i] = self.low_state.motor_state[self.config.action_joint2motor_idx[i]].q
            self.dqj[i] = self.low_state.motor_state[self.config.action_joint2motor_idx[i]].dq

        # imu_state quaternion: w, x, y, z
        quat = self.low_state.imu_state.quaternion
        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)

        if self.config.imu_type == "torso":
            # h1 and h1_2 imu is on the torso
            # imu data needs to be transformed to the pelvis frame
            waist_yaw = self.low_state.motor_state[self.config.fixed_joint2motor_idx[0]].q
            waist_yaw_omega = self.low_state.motor_state[self.config.fixed_joint2motor_idx[0]].dq
            quat, ang_vel = transform_imu_data(waist_yaw=waist_yaw, waist_yaw_omega=waist_yaw_omega, imu_quat=quat, imu_omega=ang_vel)

        qj = self.qj.copy()    
        dqj = self.dqj.copy()   
    
        projected_gravity = get_gravity_orientation(quat)
        dof_pos = qj * self.config.dof_pos_scale
        dof_vel = dqj * self.config.dof_vel_scale
        base_ang_vel = ang_vel* self.config.ang_vel_scale
        
        motion_time = self.counter * self.control_dt
        if self.policy_idx == self.stance_idx:
            self.ref_motion_phase = 0.0 # No acculumation of stance phase
        else:
            self.ref_motion_phase = motion_time / self.motion_len
            print("[DEBUG] ref_motion_phase", self.ref_motion_phase)
        # self.ref_motion_phase += self.config.ref_motion_phase
        # self.ref_motion_phase = self.ref_motmove_toion_phase % 1 # 循环播放
        num_actions = self.config.num_actions

        history_obs_buf = np.concatenate((self.action_buf, self.ang_vel_buf, self.dof_pos_buf, self.dof_vel_buf, self.proj_g_buf, self.ref_motion_phase_buf), axis=-1, dtype=np.float32)

        try:
            obs_buf = np.concatenate((self.action, base_ang_vel.flatten(), dof_pos, dof_vel, history_obs_buf, projected_gravity, [self.ref_motion_phase]), axis=-1, dtype=np.float32)
        except ValueError as e:
            print(f"Concatenation failed with error: {e}")
            print("Please ensure all arrays have the same number of dimensions (either all 1D or all 2D)")
            raise

        # update history
        self.ang_vel_buf = np.concatenate((base_ang_vel.flatten(), self.ang_vel_buf[:-3]), axis=-1, dtype=np.float32)
        
        self.proj_g_buf = np.concatenate((projected_gravity, self.proj_g_buf[:-3] ), axis=-1, dtype=np.float32)
        self.dof_pos_buf = np.concatenate((dof_pos, self.dof_pos_buf[:-num_actions] ), axis=-1, dtype=np.float32)
        self.dof_vel_buf = np.concatenate((dof_vel, self.dof_vel_buf[:-num_actions] ), axis=-1, dtype=np.float32)
        self.action_buf = np.concatenate((self.action, self.action_buf[:-num_actions] ), axis=-1, dtype=np.float32)
        self.ref_motion_phase_buf = np.concatenate(([self.ref_motion_phase], self.ref_motion_phase_buf[:-1] ), axis=-1, dtype=np.float32)                

        return obs_buf

    def _post_process_and_send(self):
        """Wrapped original action clipping, PD mapping & safety checks."""
        
        
        # Warning for the action threshold
        if np.any(np.abs(self.action) > self.config.action_clip_warn_threshold):
            print(f"[WARN] Action exceeds threshold | max={np.max(np.abs(self.action)):.3f}")

        # action clipping
        # self.action = np.clip(self.action, -self.config.action_clip, self.config.action_clip)

        # transform action to target_dof_pos
        target_dof_pos = self.config.default_angles + self.action * self.config.action_scale

        # Target_position clip 
        target_dof_pos = np.clip(target_dof_pos, self.config.dof_pos_lower_limit, self.config.dof_pos_upper_limit)

        # Build low cmd
        for i in range(len(self.config.action_joint2motor_idx)):
            motor_idx = self.config.action_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = target_dof_pos[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0
            print(f"[MOTOR] idx={motor_idx:02d}  target={target_dof_pos[i]:+.3f}")
        for i in range(len(self.config.fixed_joint2motor_idx)):
            motor_idx = self.config.fixed_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = self.config.fixed_target[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.fixed_kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.fixed_kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0



# ════════════════════════════════════════════════════════════════
# 3.  入口：新增 loco_yaml 参数
# ════════════════════════════════════════════════════════════════
class _Tee:
    def __init__(self, f: str):
        self.term = sys.stdout
        self.log  = open(f, "w", buffering=1)
    def write(self, m): self.term.write(m); self.log.write(m)
    def flush(self): self.term.flush(); self.log.flush()

if __name__ == "__main__":
    pa = argparse.ArgumentParser("Motion + Loco deploy (minimal diff)")
    pa.add_argument("net",        help="network interface (e.g. eth0)")
    pa.add_argument("motion_yaml",help="absolute path of motion-mode YAML")
    pa.add_argument("loco_yaml",  help="absolute path of loco-mode YAML")
    args = pa.parse_args()

    log_f = f"deploy_log_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    sys.stdout = _Tee(log_f)

    # DDS & 配置
    ChannelFactoryInitialize(0, args.net)
    cfg_motion_path = f"deploy_real/configs/{args.motion_yaml}"
    cfg_loco_path = f"deploy_real/configs/{args.loco_yaml}"
    motion_cfg = Config(cfg_motion_path)   # YAML 提供的全是绝对路径
    ctrl       = Controller(motion_cfg, cfg_loco_path)

    # 启动阶段
    ctrl.zero_torque_state()
    ctrl.move_to_default_pos()
    ctrl.default_pos_state()

    print("[RUN]  L1/R1 motion | L2+Y loco | R2+Y stance | X reset | SELECT quit")
    try:
        while True:
            if ctrl.run(): break
    except KeyboardInterrupt:
        pass

    create_damping_cmd(ctrl.low_cmd)
    ctrl.send_cmd(ctrl.low_cmd)
    print("Exit cleanly.")
