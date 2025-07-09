#!/usr/bin/env python3
"""
G1 deploy script – Motion-only
==============================

• 多个 ONNX Motion 策略（L1 / R1 切换）
• 启动序列: Zero-torque → Default pose → 等待 A → Stance
• 热键:
    L1 / R1 : 上 / 下一个动作
    X       : 回到默认位姿
    SELECT  : 退出脚本
"""

from __future__ import annotations
from typing import List, Union
import argparse, sys, os, time, datetime as _dt

import numpy as np
import onnxruntime as ort

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__LowCmd_,
    unitree_hg_msg_dds__LowState_,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG, LowState_ as LowStateHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC

from common.command_helper import (
    create_zero_cmd,
    create_damping_cmd,
    init_cmd_hg,
    MotorMode,
)
from common.rotation_helper import get_gravity_orientation
from common.remote_controller import RemoteController, KeyMap
from config import Config

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ════════════════════════════════════════════════════════════════
# 1. Helper containers
# ════════════════════════════════════════════════════════════════
class PolicyOutput:
    def __init__(self, nj: int):
        self.actions = np.zeros(nj, np.float32)
        self.kps     = np.zeros(nj, np.float32)
        self.kds     = np.zeros(nj, np.float32)


class _State:
    __slots__ = ("q", "dq", "quat", "ang_vel", "gravity_ori")
    def __init__(self, nj: int):
        self.q           = np.zeros(nj, np.float32)
        self.dq          = np.zeros(nj, np.float32)
        self.quat        = np.zeros(4,  np.float32)
        self.ang_vel     = np.zeros(3,  np.float32)
        self.gravity_ori = np.zeros(3,  np.float32)

# ════════════════════════════════════════════════════════════════
# 2.  ONNX Motion policy
# ════════════════════════════════════════════════════════════════
class ONNXMotionPolicy:
    def __init__(self, cfg: Config, onnx_path: str, name: str, motion_len: float):
        self.cfg   = cfg
        self.name  = name
        self.dt    = cfg.control_dt
        self.nj    = cfg.num_actions
        self.len_s = motion_len
        self.step_counter = 0

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name

        hl, na = cfg.history_length, cfg.num_actions
        self.action    = np.zeros(na,    np.float32)
        self.ang_buf   = np.zeros(3*hl,  np.float32)
        self.g_buf     = np.zeros(3*hl,  np.float32)
        self.pos_buf   = np.zeros(na*hl, np.float32)
        self.vel_buf   = np.zeros(na*hl, np.float32)
        self.act_buf   = np.zeros(na*hl, np.float32)
        self.phase_buf = np.zeros(hl,    np.float32)

    def _build_obs(self, st: _State) -> np.ndarray:
        cfg = self.cfg
        ref_phase = 0.0 if self.len_s > 9000 else (self.step_counter * self.dt / self.len_s)

        obs = np.concatenate(
            (
                self.action,
                st.ang_vel             * cfg.ang_vel_scale,
                st.q                   * cfg.dof_pos_scale,
                st.dq                  * cfg.dof_vel_scale,
                self.act_buf, self.ang_buf,
                self.pos_buf, self.vel_buf,
                self.g_buf,  self.phase_buf,
                st.gravity_ori, [ref_phase],
            ),
            dtype=np.float32,
        )

        # 更新历史缓冲
        na = cfg.num_actions
        self.ang_buf   = np.concatenate((st.ang_vel * cfg.ang_vel_scale, self.ang_buf[:-3]))
        self.g_buf     = np.concatenate((st.gravity_ori,                 self.g_buf[:-3]))
        self.pos_buf   = np.concatenate((st.q * cfg.dof_pos_scale,       self.pos_buf[:-na]))
        self.vel_buf   = np.concatenate((st.dq* cfg.dof_vel_scale,       self.vel_buf[:-na]))
        self.act_buf   = np.concatenate((self.action,                    self.act_buf[:-na]))
        self.phase_buf = np.concatenate(([ref_phase],                    self.phase_buf[:-1]))
        return obs

    def step(self, st: _State) -> PolicyOutput:
        self.step_counter += 1
        obs = self._build_obs(st)
        self.action = np.squeeze(self.sess.run(None, {self.in_name: obs[None]})[0])

        tgt = np.clip(
            self.cfg.default_angles + self.action * self.cfg.action_scale,
            self.cfg.dof_pos_lower_limit,
            self.cfg.dof_pos_upper_limit,
        )
        po            = PolicyOutput(self.nj)
        po.actions[:] = tgt
        po.kps[:]     = self.cfg.kps
        po.kds[:]     = self.cfg.kds
        return po

# ════════════════════════════════════════════════════════════════
# 3.  Controller (Motion-only)
# ════════════════════════════════════════════════════════════════
class Controller:
    def __init__(self, cfg: Config):
        self.cfg    = cfg
        self.remote = RemoteController()
        self.dt     = cfg.control_dt
        self.nj     = 29  # G1 全关节

        # ── DDS init ───────────────────────────────────────────
        # self.low_cmd   = unitree_hg_msg_dds__LowCmd_()
        # self.low_state = unitree_hg_msg_dds__LowState_()
        # self.pub = ChannelPublisher(cfg.lowcmd_topic, LowCmdHG); self.pub.Init()
        # self.sub = ChannelSubscriber(cfg.lowstate_topic, LowStateHG)
        # self.sub.Init(self._cb, 10)
        if cfg.msg_type == "hg":
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self.low_state = unitree_hg_msg_dds__LowState_()
            self.mode_pr_ = MotorMode.PR
            self.mode_machine_ = 0

            self.lowcmd_publisher_ = ChannelPublisher(cfg.lowcmd_topic, LowCmdHG)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(cfg.lowstate_topic, LowStateHG)
            self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)
        elif cfg.msg_type == "go":
            self.low_cmd = unitree_go_msg_dds__LowCmd_()
            self.low_state = unitree_go_msg_dds__LowState_()

            self.lowcmd_publisher_ = ChannelPublisher(cfg.lowcmd_topic, LowCmdGo)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(cfg.lowstate_topic, LowStateGo)
            self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)
        else:
            raise ValueError("Invalid msg_type")

        init_cmd_hg(self.low_cmd, 0, MotorMode.PR)
        print("[INFO] Waiting DDS …")
        self._wait_for_low_state()
        print("[INFO] Connected!")

        # ── 载入 ONNX 动作策略 ────────────────────────────────
        self.policies: List[ONNXMotionPolicy] = []
        for i, p in enumerate(cfg.policy_paths):
            name = cfg.policy_names[i] if hasattr(cfg, "policy_names") else f"motion{i}"
            self.policies.append(ONNXMotionPolicy(cfg, p, name, cfg.motion_lens[i]))
        self.idx            = 0               # 当前策略编号
        self.motion_indices = list(range(len(self.policies)))

        # ── 运行时状态 ─────────────────────────────────────────
        self.state        = _State(self.nj)
        self.last_btn     = np.zeros(16, np.int32)
        self.last_switch  = time.time()
        self.cooldown_s   = 0.4

    # ── DDS 回调 ───────────────────────────────────────────────
    def _cb(self, msg: LowStateHG):
        self.low_state = msg
        self.remote.set(msg.wireless_remote)
    # ─────────────────────────────────────────
    # DDS Callbacks (unchanged)
    # ─────────────────────────────────────────
    def LowStateHgHandler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def LowStateGoHandler(self, msg: LowStateGo):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)
    def _wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.dt)

    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
            cmd.crc = CRC().Crc(cmd)
            self.lowcmd_publisher_.Write(cmd)
    # ── 热键处理 (只剩 L1/R1) ─────────────────────────────────
    def _handle_buttons(self):
        btn = self.remote.button
        now = time.time()

        if btn[KeyMap.L1] and not self.last_btn[KeyMap.L1] and now - self.last_switch > self.cooldown_s:
            self.idx = (self.idx - 1) % len(self.motion_indices)
            self.last_switch = now
            print(f"< Motion #{self.idx} >")

        if btn[KeyMap.R1] and not self.last_btn[KeyMap.R1] and now - self.last_switch > self.cooldown_s:
            self.idx = (self.idx + 1) % len(self.motion_indices)
            self.last_switch = now
            print(f"< Motion #{self.idx} >")

        self.last_btn = btn.copy()

    # ── 状态采样 ───────────────────────────────────────────────
    def _fill_state(self):
        for i in range(self.nj):
            self.state.q [i] = self.low_state.motor_state[i].q
            self.state.dq[i] = self.low_state.motor_state[i].dq
        self.state.quat[:]     = self.low_state.imu_state.quaternion
        self.state.ang_vel[:]  = self.low_state.imu_state.gyroscope
        self.state.gravity_ori = get_gravity_orientation(self.state.quat)

    # ── 统一发送 ───────────────────────────────────────────────
    def _send(self, po: PolicyOutput):
        # 动作关节
        for i, m in enumerate(self.cfg.action_joint2motor_idx):
            mc = self.low_cmd.motor_cmd[m]
            mc.q, mc.qd, mc.kp, mc.kd, mc.tau = po.actions[i], 0, po.kps[i], po.kds[i], 0
        # 固定关节
        for j, m in enumerate(self.cfg.fixed_joint2motor_idx):
            mc = self.low_cmd.motor_cmd[m]
            mc.q, mc.qd, mc.kp, mc.kd, mc.tau = (
                self.cfg.fixed_target[j],
                0,
                self.cfg.fixed_kps[j],
                self.cfg.fixed_kds[j],
                0,
            )
        self.low_cmd.crc = CRC().Crc(self.low_cmd)
        self.lowcmd_publisher_.Write(self.low_cmd)

        
    # ── 主循环一步 ─────────────────────────────────────────────
    def loop_once(self) -> bool:
        self._handle_buttons()

        # 系统按键
        if self.remote.button[KeyMap.select]:
            return True
        if self.remote.button[KeyMap.X]:
            print("[STATE] Reset default pose …")
            self.move_to_default_pos()
            return False

        self._fill_state()
        po = self.policies[self.idx].step(self.state)

        # 安全检查
        if not np.isfinite(po.actions).all() or np.max(np.abs(po.actions)) > 15:
            print("[SAFETY] Invalid action → damping")
            create_damping_cmd(self.low_cmd)
            self._send(po)
            return True

        self._send(po)

        # 若非 stance 且动作播放完自动回 stance
        if self.idx != 0:
            pol = self.policies[self.idx]
            if pol.step_counter * pol.dt >= pol.len_s:
                print("[INFO] Motion finished → stance")
                self.idx = 0
                pol.step_counter = 0
        return False

    # ── Boot phases ────────────────────────────────────────────
    def zero_torque_state(self):
        print("[STATE] Zero torque – press START")
        while not self.remote.is_button_pressed(KeyMap.start):
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            self._send(PolicyOutput(self.nj))
            time.sleep(self.dt)

    def move_to_default_pos(self):
        print("[STATE] Moving to default pose (5 s)")
        steps   = int(5 / self.dt)
        dof_idx = self.cfg.action_joint2motor_idx + self.cfg.fixed_joint2motor_idx
        kps     = self.cfg.kps + self.cfg.fixed_kps
        kds     = self.cfg.kds + self.cfg.fixed_kds
        tgt     = np.concatenate((self.cfg.default_angles, self.cfg.fixed_target))
        init    = np.array([self.low_state.motor_state[i].q for i in dof_idx])

        for s in range(steps):
            a = s / steps
            for j, m in enumerate(dof_idx):
                qd = init[j] * (1 - a) + tgt[j] * a
                mc = self.low_cmd.motor_cmd[m]
                mc.q, mc.qd, mc.kp, mc.kd, mc.tau = qd, 0, kps[j], kds[j], 0
            self._send(PolicyOutput(self.nj))
            time.sleep(self.dt)
        print("[INFO] Default pose reached")

    def default_pos_state(self):
        print("[STATE] Holding default pose – press A for stance")
        while not self.remote.is_button_pressed(KeyMap.A):
            for i, m in enumerate(self.cfg.action_joint2motor_idx):
                mc = self.low_cmd.motor_cmd[m]
                mc.q, mc.qd, mc.kp, mc.kd, mc.tau = (
                    self.cfg.default_angles[i],
                    0,
                    self.cfg.kps[i],
                    self.cfg.kds[i],
                    0,
                )
            for i, m in enumerate(self.cfg.fixed_joint2motor_idx):
                mc = self.low_cmd.motor_cmd[m]
                mc.q, mc.qd, mc.kp, mc.kd, mc.tau = (
                    self.cfg.fixed_target[i],
                    0,
                    self.cfg.fixed_kps[i],
                    self.cfg.fixed_kds[i],
                    0,
                )
            self._send(PolicyOutput(self.nj))
            time.sleep(self.dt)

# ════════════════════════════════════════════════════════════════
# 4. Logger + main
# ════════════════════════════════════════════════════════════════
class _Tee:
    def __init__(self, f: str):
        self.term = sys.stdout
        self.log  = open(f, "w", buffering=1)
    def write(self, m): self.term.write(m); self.log.write(m)
    def flush(self): self.term.flush(); self.log.flush()

if __name__ == "__main__":
    pa = argparse.ArgumentParser("Motion-only deploy")
    pa.add_argument("net",         help="network interface (e.g. eth0)")
    pa.add_argument("motion_yaml", help="absolute path of motion-mode YAML")
    args = pa.parse_args()

    log_f = os.path.join(PROJECT_ROOT, f"deploy_log_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    sys.stdout = _Tee(log_f)

    ChannelFactoryInitialize(0, args.net)
    cfg_motion_path = f"deploy_real/configs/{args.motion_yaml}"
    motion_cfg = Config(cfg_motion_path)   # YAML 内提供绝对路径
    ctrl       = Controller(motion_cfg)

    # Boot sequence
    ctrl.zero_torque_state()
    ctrl.move_to_default_pos()
    ctrl.default_pos_state()

    print("[RUN]  L1/R1: switch motion | X reset | SELECT quit")
    try:
        while True:
            if ctrl.loop_once():
                break
            time.sleep(ctrl.dt)
    except KeyboardInterrupt:
        pass

    create_damping_cmd(ctrl.low_cmd)
    ctrl._send(PolicyOutput(ctrl.nj))
    print("Exit cleanly.")
