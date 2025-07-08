#!/usr/bin/env python3
"""
Motion‑mode deploy script for Unitree‑G1
=======================================

Handles **only** ONNX motion policies (stance / wait / horsestance …) exactly
like your original `deploy_real_multi_policy.py`, but wrapped into a slightly
more modular architecture so that we can later plug‑in additional modes (e.g.
loco) without touching this file.

* Hot‑keys*
------------
• **L1 / R1**   cycle through motion policies (prev / next)  
• **SELECT**    quit                 
• **START**     exit zero‑torque / reset controller

The rest of the behaviour (safety walls, PD mapping, logging …) is identical
to your working script.
"""
# Std‑lib & deps
from __future__ import annotations
from pathlib import Path
import argparse, sys, time, datetime as _dt
from typing import List

import numpy as np
import onnxruntime as ort

# Unitree SDK2 & project utils
from unitree_sdk2py.core.channel import (
    ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize,
)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG, LowState_ as LowStateHG
from unitree_sdk2py.utils.crc import CRC

from common.command_helper import create_zero_cmd, create_damping_cmd, init_cmd_hg, MotorMode
from common.rotation_helper import get_gravity_orientation
from common.remote_controller import RemoteController, KeyMap
from config import Config

# ══════════════════════════════════════════════════════════════════════
# 1.  Policy interface (simplified)
# ══════════════════════════════════════════════════════════════════════
class PolicyOutput:
    def __init__(self, n_joints: int):
        self.actions = np.zeros(n_joints, np.float32)
        self.kps     = np.zeros(n_joints, np.float32)
        self.kds     = np.zeros(n_joints, np.float32)


class ONNXMotionPolicy:
    """Wraps a single ONNX motion network & its runtime buffers."""

    def __init__(self, cfg: Config, onnx_path: Path, name: str, motion_len: float):
        self.name = name
        self.cfg = cfg
        self.motion_len = motion_len
        self.n_joints = cfg.num_actions
        self.control_dt = cfg.control_dt

        # ORT session
        sess_opt = ort.SessionOptions()
        sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(onnx_path), sess_opt, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name

        # run‑time buffers (copied from original script)
        na, hist = cfg.num_actions, cfg.history_length
        self.counter = 0
        self.qj = np.zeros(na, np.float32)
        self.dqj = np.zeros(na, np.float32)
        self.action = np.zeros(na, np.float32)
        self.ang_vel_buf = np.zeros(3 * hist, np.float32)
        self.proj_g_buf  = np.zeros(3 * hist, np.float32)
        self.dof_pos_buf = np.zeros(na * hist, np.float32)
        self.dof_vel_buf = np.zeros(na * hist, np.float32)
        self.action_buf  = np.zeros(na * hist, np.float32)
        self.phase_buf   = np.zeros(hist, np.float32)

    # ------------------------------------------------------------
    def _build_obs(self, state) -> np.ndarray:
        cfg = self.cfg
        proj_g = state.gravity_ori
        dof_pos = state.q * cfg.dof_pos_scale
        dof_vel = state.dq * cfg.dof_vel_scale
        ang_vel = state.ang_vel * cfg.ang_vel_scale
        ref_phase = min(self.counter * self.control_dt / self.motion_len, 1.0)

        hist = np.concatenate(
            (
                self.action_buf,
                self.ang_vel_buf,
                self.dof_pos_buf,
                self.dof_vel_buf,
                self.proj_g_buf,
                self.phase_buf,
            ),
            dtype=np.float32,
        )
        obs = np.concatenate(
            (
                self.action,
                ang_vel,
                dof_pos,
                dof_vel,
                hist,
                proj_g,
                [ref_phase],
            ),
            dtype=np.float32,
        )

        # history shift
        na = cfg.num_actions
        self.ang_vel_buf = np.concatenate((ang_vel, self.ang_vel_buf[:-3]))
        self.proj_g_buf  = np.concatenate((proj_g, self.proj_g_buf[:-3]))
        self.dof_pos_buf = np.concatenate((dof_pos, self.dof_pos_buf[:-na]))
        self.dof_vel_buf = np.concatenate((dof_vel, self.dof_vel_buf[:-na]))
        self.action_buf  = np.concatenate((self.action, self.action_buf[:-na]))
        self.phase_buf   = np.concatenate(([ref_phase], self.phase_buf[:-1]))
        return obs, ref_phase

    # ------------------------------------------------------------
    def step(self, state) -> PolicyOutput:
        self.counter += 1
        obs, phase = self._build_obs(state)
        self.action = np.squeeze(self.sess.run(None, {self.input_name: obs[None].astype(np.float32)})[0])
        # scale to target pos
        target = np.clip(
            self.cfg.default_angles + self.action * self.cfg.action_scale,
            self.cfg.dof_pos_lower_limit,
            self.cfg.dof_pos_upper_limit,
        )
        po = PolicyOutput(self.n_joints)
        po.actions[:] = target
        po.kps[:] = self.cfg.kps
        po.kds[:] = self.cfg.kds
        return po


# ══════════════════════════════════════════════════════════════════════
# 2.  Controller (motion‑only)
# ══════════════════════════════════════════════════════════════════════
class Controller:
    def __init__(self, cfg: Config, net: str):
        self.cfg = cfg
        self.remote = RemoteController()
        self.dt = cfg.control_dt
        self.nj = cfg.num_actions

        # DDS ----------------------------------------------------
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = unitree_hg_msg_dds__LowState_()
        self.publisher = ChannelPublisher(cfg.lowcmd_topic, LowCmdHG)
        self.publisher.Init()
        self.subscriber = ChannelSubscriber(cfg.lowstate_topic, LowStateHG)
        self.subscriber.Init(self._on_low_state, 10)
        init_cmd_hg(self.low_cmd, 0, MotorMode.PR)

        print("[INFO] Waiting for low‑state …")
        while self.low_state.tick == 0:
            time.sleep(self.dt)
        print("[INFO] Connected!")

        # Policies ----------------------------------------------
        self.policies: List[ONNXMotionPolicy] = []
        for i, p_path in enumerate(cfg.policy_paths):
            name = cfg.policy_names[i] if hasattr(cfg, "policy_names") else f"motion{i}"
            pol = ONNXMotionPolicy(cfg, Path(p_path), name, cfg.motion_lens[i])
            self.policies.append(pol)
        self.idx = 0
        self.last_btn = np.zeros(16, np.int32)
        self.last_switch = time.time()
        self.cooldown = 0.4

    # DDS callback ---------------------------------------------
    def _on_low_state(self, msg: LowStateHG):
        self.low_state = msg
        self.remote.set(msg.wireless_remote)

    # Utilities -------------------------------------------------
    def _fill_state_struct(self):
        class S:
            pass
        s = S()
        s.q = np.array([m.q for m in self.low_state.motor_state], np.float32)[: self.nj]
        s.dq = np.array([m.dq for m in self.low_state.motor_state], np.float32)[: self.nj]
        quat = self.low_state.imu_state.quaternion
        s.ang_vel = np.array(self.low_state.imu_state.gyroscope, np.float32)
        s.gravity_ori = get_gravity_orientation(quat)
        return s

    def _send(self, po: PolicyOutput):
        for i in range(self.nj):
            mc = self.low_cmd.motor_cmd[i]
            mc.q, mc.qd, mc.kp, mc.kd, mc.tau = po.actions[i], 0, po.kps[i], po.kds[i], 0
        self.low_cmd.crc = CRC().Crc(self.low_cmd)
        self.publisher.Write(self.low_cmd)

    # Hot‑key handler ------------------------------------------
    def _handle_buttons(self):
        btn = self.remote.button
        now = time.time()
        if btn[KeyMap.L1] and not self.last_btn[KeyMap.L1] and now - self.last_switch > self.cooldown:
            self.idx = (self.idx - 1) % len(self.policies)
            self.last_switch = now
            print(f"< Motion ← {self.policies[self.idx].name} >")
        if btn[KeyMap.R1] and not self.last_btn[KeyMap.R1] and now - self.last_switch > self.cooldown:
            self.idx = (self.idx + 1) % len(self.policies)
            self.last_switch = now
            print(f"< Motion → {self.policies[self.idx].name} >")
        self.last_btn = btn.copy()

    # Control loop --------------------------------------------
    def loop_once(self) -> bool:
        self._handle_buttons()
        if self.remote.button[KeyMap.select]:
            return True
        state = self._fill_state_struct()
        po = self.policies[self.idx].step(state)
        # simple NaN guard
        if not np.all(np.isfinite(po.actions)):
            print("[SAFETY] NaN detected – damping!")
            create_damping_cmd(self.low_cmd); self._send(po); return True
        self._send(po)
        return False


# ══════════════════════════════════════════════════════════════════════
# 3.  Logging helper
# ══════════════════════════════════════════════════════════════════════
class _Tee:
    def __init__(self, fname: Path):
        self.terminal = sys.stdout
        self.log = open(fname, "w", buffering=1)
    def write(self, msg):
        self.terminal.write(msg); self.log.write(msg)
    def flush(self):
        self.terminal.flush(); self.log.flush()


# ══════════════════════════════════════════════════════════════════════
# 4.  Entry
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    p = argparse.ArgumentParser("Motion‑only deployment")
    p.add_argument("net", help="network interface (e.g. eth0)")
    p.add_argument("yaml", help="motion config YAML path")
    args = p.parse_args()

    log_f = Path(__file__).with_name(f"motion_log_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    sys.stdout = _Tee(log_f)

    cfg_path = f"deploy_real/configs/{args.yaml}"
    cfg = Config(cfg_path)
    ChannelFactoryInitialize(0, args.net)
    ctrl = Controller(cfg, args.net)

    # Zero‑torque hold until START
    print("[STATE] Zero‑torque … press START to continue")
    while not ctrl.remote.is_button_pressed(KeyMap.start):
        create_zero_cmd(ctrl.low_cmd); ctrl._send(None); time.sleep(ctrl.dt)

    print("[STATE] Running …  L1/R1 cycle, SELECT exit")
    try:
        while True:
            if ctrl.loop_once():
                break
            time.sleep(ctrl.dt)
    except KeyboardInterrupt:
        pass

    create_damping_cmd(ctrl.low_cmd)
    ctrl._send(None)
    print("Exit cleanly.")
