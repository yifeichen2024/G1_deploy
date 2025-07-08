
# ─────────────────────────────────────────────────────────────
# Imports & basic paths (no pathlib)
# ─────────────────────────────────────────────────────────────
from __future__ import annotations
from typing import List
import os, sys, argparse, time, datetime as _dt

import numpy as np
import yaml
import torch
import onnxruntime as ort

from unitree_sdk2py.core.channel import (
    ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize,
)
from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG, LowState_ as LowStateHG
from unitree_sdk2py.utils.crc import CRC

from common.command_helper import (
    create_zero_cmd, create_damping_cmd, init_cmd_hg, MotorMode,
)
from common.rotation_helper import get_gravity_orientation
from common.remote_controller import RemoteController, KeyMap
from config import Config

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────
# 1.  Shared helper structs
# ─────────────────────────────────────────────────────────────
class PolicyOutput:
    def __init__(self, n_joints: int):
        self.actions = np.zeros(n_joints, np.float32)
        self.kps     = np.zeros(n_joints, np.float32)
        self.kds     = np.zeros(n_joints, np.float32)


class _State:
    """Lightweight container for current robot meta-state."""
    __slots__ = ("q", "dq", "quat", "ang_vel", "gravity_ori", "vel_cmd")
    def __init__(self, n_joints: int):
        self.q = np.zeros(n_joints, np.float32)
        self.dq = np.zeros(n_joints, np.float32)
        self.quat = np.zeros(4, np.float32)
        self.ang_vel = np.zeros(3, np.float32)
        self.gravity_ori = np.zeros(3, np.float32)
        self.vel_cmd = np.zeros(3, np.float32)  # lx, ly, rz

# ─────────────────────────────────────────────────────────────
# 2.  Motion-mode policy (ONNX)
# ─────────────────────────────────────────────────────────────
class ONNXMotionPolicy:
    def __init__(self, cfg: Config, onnx_path: str, name: str, motion_len: float):
        self.name = name
        self.cfg = cfg
        self.motion_len = motion_len
        self.dt = cfg.control_dt
        self.nj = cfg.num_actions
        self.counter = 0

        opt = ort.SessionOptions()
        opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(onnx_path, opt, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name

        hl, na = cfg.history_length, cfg.num_actions
        self.action = np.zeros(na, np.float32)
        self.ang_buf = np.zeros(3 * hl, np.float32)
        self.g_buf   = np.zeros(3 * hl, np.float32)
        self.pos_buf = np.zeros(na * hl, np.float32)
        self.vel_buf = np.zeros(na * hl, np.float32)
        self.act_buf = np.zeros(na * hl, np.float32)
        self.phase_buf = np.zeros(hl, np.float32)

    # ---------------------------------------------------------
    def _build_obs(self, st: _State) -> np.ndarray:
        cfg = self.cfg
        ref_phase = 0.0 if self.motion_len > 9000 else (self.counter * self.dt / self.motion_len)
        obs = np.concatenate(
            (
                self.action,
                st.ang_vel * cfg.ang_vel_scale,
                st.q * cfg.dof_pos_scale,
                st.dq * cfg.dof_vel_scale,
                self.act_buf,
                self.ang_buf,
                self.pos_buf,
                self.vel_buf,
                self.g_buf,
                self.phase_buf,
                st.gravity_ori,
                [ref_phase],
            ),
            dtype=np.float32,
        )
        # update history
        na = cfg.num_actions
        self.ang_buf = np.concatenate((st.ang_vel * cfg.ang_vel_scale, self.ang_buf[:-3]))
        self.g_buf   = np.concatenate((st.gravity_ori, self.g_buf[:-3]))
        self.pos_buf = np.concatenate((st.q * cfg.dof_pos_scale, self.pos_buf[:-na]))
        self.vel_buf = np.concatenate((st.dq * cfg.dof_vel_scale, self.vel_buf[:-na]))
        self.act_buf = np.concatenate((self.action, self.act_buf[:-na]))
        self.phase_buf = np.concatenate(([ref_phase], self.phase_buf[:-1]))
        return obs

    # ---------------------------------------------------------
    def step(self, st: _State) -> PolicyOutput:
        self.counter += 1
        obs = self._build_obs(st)
        self.action = np.squeeze(self.sess.run(None, {self.in_name: obs[None]})[0])
        tgt = np.clip(
            self.cfg.default_angles + self.action * self.cfg.action_scale,
            self.cfg.dof_pos_lower_limit,
            self.cfg.dof_pos_upper_limit,
        )
        po = PolicyOutput(self.nj)
        po.actions[:] = tgt
        po.kps[:] = self.cfg.kps
        po.kds[:] = self.cfg.kds
        return po

# ─────────────────────────────────────────────────────────────
# 3.  Loco-mode policy (TorchScript)
# ─────────────────────────────────────────────────────────────
class LocoTorchPolicy:
    def __init__(self, full_joints: int, loco_yaml: str):
        with open(loco_yaml, "r") as f:
            cfg = yaml.safe_load(f)
        self.cfg = cfg
        self.nj = full_joints
        self.joint2motor = np.array(cfg["joint2motor_idx"], np.int32)
        self.na = cfg["num_actions"]

        # TS model（cfg["policy_path"] 现为绝对路径）
        self.ts = torch.jit.load(cfg["policy_path"])
        self.ts.eval()
        for _ in range(10):
            with torch.inference_mode():
                self.ts(torch.zeros(1, cfg["num_obs"], dtype=torch.float32))

        # constant arrays
        self.kps = np.zeros(full_joints, np.float32)
        self.kds = np.zeros(full_joints, np.float32)
        self.def_ang = np.zeros(full_joints, np.float32)
        for i, m in enumerate(self.joint2motor):
            self.kps[m] = cfg["kps"][i]
            self.kds[m] = cfg["kds"][i]
            self.def_ang[m] = cfg["default_angles"][i]

        self.action_prev = np.zeros(self.na, np.float32)
        self.obs = np.zeros(cfg["num_obs"], np.float32)

        # ranges
        self.range_vx = np.array(cfg["cmd_range"]["lin_vel_x"], np.float32)
        self.range_vy = np.array(cfg["cmd_range"]["lin_vel_y"], np.float32)
        self.range_vz = np.array(cfg["cmd_range"]["ang_vel_z"], np.float32)

    # ---------------------------------------------------------
    def _scale_cmd(self, raw: np.ndarray) -> np.ndarray:
        out = np.zeros(3, np.float32)
        for i, rg in enumerate((self.range_vx, self.range_vy, self.range_vz)):
            a, b = rg
            out[i] = (raw[i] - a) / (b - a) * 2 - 1
            out[i] = np.clip(out[i], -1, 1)
        return out

    # ---------------------------------------------------------
    def step(self, st: _State) -> PolicyOutput:
        cfg = self.cfg
        q_sel = st.q[self.joint2motor]
        dq_sel = st.dq[self.joint2motor]
        cmd_scaled = self._scale_cmd(st.vel_cmd) * cfg["cmd_scale"]

        self.obs[:3] = st.ang_vel * cfg["ang_vel_scale"]
        self.obs[3:6] = st.gravity_ori
        self.obs[6:9] = cmd_scaled
        self.obs[9:9+self.na] = (q_sel - cfg["default_angles"]) * cfg["dof_pos_scale"]
        self.obs[9+self.na:9+2*self.na] = dq_sel * cfg["dof_vel_scale"]
        self.obs[9+2*self.na:9+3*self.na] = self.action_prev

        with torch.inference_mode():
            act = (
                self.ts(torch.from_numpy(self.obs).unsqueeze(0))
                .clip(-100, 100)
                .squeeze()
                .cpu()
                .numpy()
            )
        self.action_prev = act
        full_target = self.def_ang.copy()
        loco = act * cfg["action_scale"] + cfg["default_angles"]
        for i, m in enumerate(self.joint2motor):
            full_target[m] = loco[i]
        po = PolicyOutput(self.nj)
        po.actions[:] = full_target
        po.kps[:] = self.kps
        po.kds[:] = self.kds
        return po
# ────────────────────────────────────────────────────────────────────────
# 4.  Controller (both modes)
# ────────────────────────────────────────────────────────────────────────
class Controller:
    def __init__(self, motion_cfg: Config, loco_yaml: str):
        self.cfg = motion_cfg
        self.remote = RemoteController()
        self.dt = motion_cfg.control_dt
        self.nj = motion_cfg.num_actions

        # DDS ---------------------------------------------------
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = unitree_hg_msg_dds__LowState_()
        self.pub = ChannelPublisher(motion_cfg.lowcmd_topic, LowCmdHG); self.pub.Init()
        self.sub = ChannelSubscriber(motion_cfg.lowstate_topic, LowStateHG)
        self.sub.Init(self._cb, 10)
        init_cmd_hg(self.low_cmd, 0, MotorMode.PR)
        print("[INFO] Waiting DDS …")
        while self.low_state.tick == 0:
            time.sleep(self.dt)
        print("[INFO] Connected!")

        # build policies ---------------------------------------
        self.policies: List[object] = []
        for i, p in enumerate(motion_cfg.policy_paths):
            name = (
                motion_cfg.policy_names[i]
                if hasattr(motion_cfg, "policy_names")
                else f"motion{i}"
            )
            self.policies.append(
                ONNXMotionPolicy(motion_cfg, p, name, motion_cfg.motion_lens[i])
            )
        self.motion_indices = list(range(len(self.policies)))
        self.loco_idx = len(self.policies)
        self.policies.append(LocoTorchPolicy(self.nj, loco_yaml))

        self.idx = 0  # start with stance
        self.state = _State(self.nj)
        self.last_btn = np.zeros(16, np.int32)
        self.last_switch = time.time()
        self.cooldown = 0.35

    # DDS callback --------------------------------------------
    def _cb(self, msg: LowStateHG):
        self.low_state = msg
        self.remote.set(msg.wireless_remote)

    # ---------------------------------------------------------
    def _fill_state(self):
        for i in range(self.nj):
            self.state.q[i] = self.low_state.motor_state[i].q
            self.state.dq[i] = self.low_state.motor_state[i].dq
        self.state.quat[:] = self.low_state.imu_state.quaternion
        self.state.ang_vel[:] = self.low_state.imu_state.gyroscope
        self.state.gravity_ori[:] = get_gravity_orientation(self.state.quat)
        # joystick velocity cmd mapping (left stick xy, right stick x)
        self.state.vel_cmd[:] = [self.remote.ly, -self.remote.lx, -self.remote.rx]

    # ---------------------------------------------------------
    def _send(self, po: PolicyOutput):
        for i in range(self.nj):
            mc = self.low_cmd.motor_cmd[i]
            mc.q, mc.qd, mc.kp, mc.kd, mc.tau = po.actions[i], 0, po.kps[i], po.kds[i], 0
        self.low_cmd.crc = CRC().Crc(self.low_cmd)
        self.pub.Write(self.low_cmd)

    # -----------------------------------------------------------
    def _handle_buttons(self):
        btn = self.remote.button; now = time.time()
        # switch between motion policies only when not in loco
        if self.idx in self.motion_indices:
            if btn[KeyMap.L1] and not self.last_btn[KeyMap.L1] and now - self.last_switch > self.cooldown:
                self.idx = (self.idx - 1) % len(self.motion_indices)
                self.last_switch = now; print("< Motion ←>")
            if btn[KeyMap.R1] and not self.last_btn[KeyMap.R1] and now - self.last_switch > self.cooldown:
                self.idx = (self.idx + 1) % len(self.motion_indices)
                self.last_switch = now; print("< Motion →>")
        # L2+Y → loco
        if btn[KeyMap.L2] and btn[KeyMap.Y] and now - self.last_switch > self.cooldown and self.idx != self.loco_idx:
            self.idx = self.loco_idx; self.last_switch = now; print("< LOCOMODE >")
        # R2+Y → stance
        if btn[KeyMap.R2] and btn[KeyMap.Y] and now - self.last_switch > self.cooldown and self.idx == self.loco_idx:
            self.idx = 0; self.last_switch = now; print("< Back to Stance >")
        self.last_btn = btn.copy()

    # -----------------------------------------------------------
    def loop_once(self) -> bool:
        self._handle_buttons()
        # system buttons
        if self.remote.button[KeyMap.select]:
            return True
        if self.remote.button[KeyMap.X]:
            print("[STATE] Resetting default pose …"); self.move_to_default_pos(); return False
        # fill state & policy step
        self._fill_state(); po = self.policies[self.idx].step(self.state)
        # NaN safety
        if not np.isfinite(po.actions).all() or np.max(np.abs(po.actions)) > 15:
            print("[SAFETY] invalid action -> damping"); create_damping_cmd(self.low_cmd); self._send(po); return True
        self._send(po)
        # auto‑return when motion finished and not loco
        if self.idx not in (0, self.loco_idx):
            pol: ONNXMotionPolicy = self.policies[self.idx]  # type: ignore
            if pol.counter * self.dt >= pol.motion_len:
                print("[INFO] Motion done -> stance"); self.idx = 0; pol.counter = 0
        return False

    # ───────── boot sequence phases─────────
    def zero_torque_state(self):
        print("[STATE] Zero torque – press START")
        while not self.remote.is_button_pressed(KeyMap.start):
            create_zero_cmd(self.low_cmd); self._send(PolicyOutput(self.nj)); time.sleep(self.dt)

    def move_to_default_pos(self):
        print("[STATE] Moving to default pose (5 s)")
        steps = int(5 / self.dt)
        dof_idx = self.cfg.action_joint2motor_idx + self.cfg.fixed_joint2motor_idx
        kps = self.cfg.kps + self.cfg.fixed_kps
        kds = self.cfg.kds + self.cfg.fixed_kds
        tgt = np.concatenate((self.cfg.default_angles, self.cfg.fixed_target))
        init = np.array([self.low_state.motor_state[i].q for i in dof_idx])
        for s in range(steps):
            a = s / steps
            for j, m in enumerate(dof_idx):
                q_d = init[j] * (1 - a) + tgt[j] * a
                mc = self.low_cmd.motor_cmd[m]; 
                mc.q, mc.qd, mc.kp, mc.kd, mc.tau = q_d, 0, kps[j], kds[j], 0
            self._send(PolicyOutput(self.nj)); time.sleep(self.dt)
        print("[INFO] Default pose reached")

    def default_pos_state(self):
        print("[STATE] Holding default pose – press A for stance")
        while not self.remote.is_button_pressed(KeyMap.A):
            for i, m in enumerate(self.cfg.action_joint2motor_idx):
                mc = self.low_cmd.motor_cmd[m]
                mc.q, mc.qd, mc.kp, mc.kd, mc.tau = self.cfg.default_angles[i], 0, self.cfg.kps[i], self.cfg.kds[i], 0
            for i, m in enumerate(self.cfg.fixed_joint2motor_idx):
                mc = self.low_cmd.motor_cmd[m]
                mc.q, mc.qd, mc.kp, mc.kd, mc.tau = self.cfg.fixed_target[i], 0, self.cfg.fixed_kps[i], self.cfg.fixed_kds[i], 0
            self._send(PolicyOutput(self.nj)); time.sleep(self.dt)

# ─────────────────────────────────────────────────────────────
# 5.  Logger &  entry
# ─────────────────────────────────────────────────────────────
class _Tee:
    def __init__(self, f: str):
        self.term = sys.stdout
        self.log = open(f, "w", buffering=1)
    def write(self, m):
        self.term.write(m)
        self.log.write(m)
    def flush(self):
        self.term.flush()
        self.log.flush()


if __name__ == "__main__":
    pa = argparse.ArgumentParser("Motion + Loco deploy")
    pa.add_argument("net", help="network interface (e.g. eth0)")
    pa.add_argument("motion_yaml", help="motion-mode YAML path")
    pa.add_argument("loco_yaml", help="loco-mode YAML path")
    args = pa.parse_args()

    log_f = os.path.join(
        PROJECT_ROOT,
        f"deploy_log_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
    )
    sys.stdout = _Tee(log_f)

    ChannelFactoryInitialize(0, args.net)

    motion_cfg = Config(args.motion_yaml)
    loco_cfg = Config(args.loco_yaml)
    ctrl = Controller(motion_cfg, args.loco_yaml)

    # Boot sequence: zero-torque → default pose → wait for A → stance
    ctrl.zero_torque_state()
    ctrl.move_to_default_pos()
    ctrl.default_pos_state()

    print("[RUN]  L1/R1 motion  |  L2+Y loco  |  R2+Y stance  |  X reset  |  SELECT quit")
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