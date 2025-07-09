#!/usr/bin/env python3
"""
G1 locomotion-only deploy script (derived from deploy_real_multi_policy.py)

  • 仅 Locomotion-mode : TorchScript 行走策略
  • Boot 流程、按键框架、日志等保持不变
  • YAML 内提供绝对路径 policy_path 及 loco 专用参数
"""

from __future__ import annotations
from typing import List, Union
import argparse, sys, time, datetime as dt

import numpy as np
import torch
import onnxruntime  # 仅为兼容 Config，实际未使用
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
    create_damping_cmd,
    create_zero_cmd,
    init_cmd_hg,
    MotorMode,
)
from common.rotation_helper import get_gravity_orientation
from common.remote_controller import RemoteController, KeyMap
from config import Config


# ════════════════════════════════════════════════════════════════
# Loco helper
# ════════════════════════════════════════════════════════════════
class LocoPolicy:
    """TorchScript Locomotion policy wrapper."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.nj  = cfg.num_actions
        self.j2m = np.array(cfg.action_joint2motor_idx, np.int32)

        # ── TorchScript 模型 ───────────────────────────────
        self.ts = torch.jit.load(cfg.policy_path)
        self.ts.eval()
        # warm-up
        with torch.inference_mode():
            self.ts(torch.zeros(1, cfg.num_obs, dtype=torch.float32))

        # ── 常量缓存 ───────────────────────────────────────
        self.kps = np.zeros(self.nj, np.float32)
        self.kds = np.zeros(self.nj, np.float32)
        self.def_ang = np.zeros(self.nj, np.float32)
        for i, m in enumerate(self.j2m):
            self.kps[i] = cfg.kps[i]
            self.kds[i] = cfg.kds[i]
            self.def_ang[i] = cfg.default_angles[i]

        # buffers
        self.action_prev = np.zeros(self.nj, np.float32)
        self.obs         = np.zeros(cfg.num_obs, np.float32)

        # joystick cmd ranges
        self.r_lin_x = np.array(cfg.cmd_range["lin_vel_x"], np.float32)
        self.r_lin_y = np.array(cfg.cmd_range["lin_vel_y"], np.float32)
        self.r_ang_z = np.array(cfg.cmd_range["ang_vel_z"], np.float32)

    # ── helper ────────────────────────────────────────────────
    def _scale_cmd(self, raw: np.ndarray) -> np.ndarray:
        out = np.zeros(3, np.float32)
        for i, rg in enumerate((self.r_lin_x, self.r_lin_y, self.r_ang_z)):
            a, b = rg
            out[i] = (raw[i] - a) / (b - a) * 2 - 1       # → [-1,1]
            out[i] = np.clip(out[i], -1, 1)
        return out * self.cfg.cmd_scale

    # ── policy forward ────────────────────────────────────────
    def step(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        ang_vel: np.ndarray,
        g_ori: np.ndarray,
        vel_cmd: np.ndarray,
    ) -> np.ndarray:
        na = self.nj
        cfg = self.cfg
        cmd_sc = self._scale_cmd(vel_cmd)

        self.obs[0:3]          = ang_vel * cfg.ang_vel_scale
        self.obs[3:6]          = g_ori
        self.obs[6:9]          = cmd_sc
        self.obs[9 : 9 + na]   = (q - cfg.default_angles) * cfg.dof_pos_scale
        self.obs[9 + na : 9 + 2 * na] = dq * cfg.dof_vel_scale
        self.obs[9 + 2 * na :] = self.action_prev

        with torch.inference_mode():
            act = (
                self.ts(torch.from_numpy(self.obs).unsqueeze(0))
                .clip(-100, 100)
                .squeeze()
                .cpu()
                .numpy()
            )
        self.action_prev = act
        tgt = np.clip(
            self.def_ang + act * cfg.action_scale,
            cfg.dof_pos_lower_limit,
            cfg.dof_pos_upper_limit,
        )
        return tgt


# ════════════════════════════════════════════════════════════════
# Controller (外形保持不变)
# ════════════════════════════════════════════════════════════════
class Controller:
    """Locomotion-only runtime controller."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.dt     = config.control_dt
        self.remote = RemoteController()

        # ── Policy ──────────────────────────────────────────
        self.policy = LocoPolicy(config)

        # ── DDS init ───────────────────────────────────────
        self.low_cmd   = unitree_hg_msg_dds__LowCmd_()
        self.low_state = unitree_hg_msg_dds__LowState_()
        self.pub = ChannelPublisher(config.lowcmd_topic, LowCmdHG); self.pub.Init()
        self.sub = ChannelSubscriber(config.lowstate_topic, LowStateHG)
        self.sub.Init(self._cb_low_state, 10)

        init_cmd_hg(self.low_cmd, 0, MotorMode.PR)
        print("[INFO] Waiting for DDS …")
        self._wait_for_state()
        print("[INFO] Connected!")

        # buffers (保留原接口字段，便于向下兼容)
        self.qj  = np.zeros(config.num_actions, np.float32)
        self.dqj = np.zeros(config.num_actions, np.float32)

    # ── DDS callback ───────────────────────────────────────────
    def _cb_low_state(self, msg: LowStateHG):
        self.low_state = msg
        self.remote.set(msg.wireless_remote)

    def _wait_for_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.dt)

    # ── Boot phases: 与旧版完全一致 ───────────────────────────
    def zero_torque_state(self):
        print("[STATE] Zero torque – press START")
        while not self.remote.is_button_pressed(KeyMap.start):
            create_zero_cmd(self.low_cmd)
            self._send_cmd()
            time.sleep(self.dt)

    def move_to_default_pos(self):
        print("[STATE] Moving to default pose (5 s)")
        steps   = int(5 / self.dt)
        dof_idx = self.config.action_joint2motor_idx + self.config.fixed_joint2motor_idx
        kps     = self.config.kps + self.config.fixed_kps
        kds     = self.config.kds + self.config.fixed_kds
        tgt     = np.concatenate((self.config.default_angles, self.config.fixed_target))
        init    = np.array([self.low_state.motor_state[i].q for i in dof_idx])
        for s in range(steps):
            a = s / steps
            for j, m in enumerate(dof_idx):
                qd = init[j] * (1 - a) + tgt[j] * a
                mc = self.low_cmd.motor_cmd[m]
                mc.q, mc.qd, mc.kp, mc.kd, mc.tau = qd, 0, kps[j], kds[j], 0
            self._send_cmd()
            time.sleep(self.dt)
        print("[INFO] Default pose reached")

    def default_pos_state(self):
        print("[STATE] Holding default pose – press A to start loco")
        while not self.remote.is_button_pressed(KeyMap.A):
            for i, m in enumerate(self.config.action_joint2motor_idx):
                mc = self.low_cmd.motor_cmd[m]
                mc.q, mc.qd, mc.kp, mc.kd, mc.tau = (
                    self.config.default_angles[i],
                    0,
                    self.config.kps[i],
                    self.config.kds[i],
                    0,
                )
            for i, m in enumerate(self.config.fixed_joint2motor_idx):
                mc = self.low_cmd.motor_cmd[m]
                mc.q, mc.qd, mc.kp, mc.kd, mc.tau = (
                    self.config.fixed_target[i],
                    0,
                    self.config.fixed_kps[i],
                    self.config.fixed_kds[i],
                    0,
                )
            self._send_cmd()
            time.sleep(self.dt)

    # ── 主循环一步 ───────────────────────────────────────────
    def run(self) -> bool:
        """返回 True → 需要退出"""
        # 系统按键
        if self.remote.button[KeyMap.select]:
            return True
        if self.remote.button[KeyMap.X]:
            print("[STATE] Reset → default pose")
            self.move_to_default_pos()
            return False

        # ── 填充状态 ────────────────────────────────────
        for i in range(self.config.num_actions):
            self.qj[i]  = self.low_state.motor_state[self.config.action_joint2motor_idx[i]].q
            self.dqj[i] = self.low_state.motor_state[self.config.action_joint2motor_idx[i]].dq

        quat      = self.low_state.imu_state.quaternion
        ang_vel   = np.array(self.low_state.imu_state.gyroscope, np.float32)
        gravity_o = get_gravity_orientation(quat)
        # 左摇杆 XY, 右摇杆 X
        vel_cmd   = np.array([self.remote.ly, -self.remote.lx, -self.remote.rx], np.float32)

        # ── 策略前向 ────────────────────────────────────
        tgt = self.policy.step(
            q=self.qj,
            dq=self.dqj,
            ang_vel=ang_vel,
            g_ori=gravity_o,
            vel_cmd=vel_cmd,
        )

        # ── 安全检查 ────────────────────────────────────
        if not np.isfinite(tgt).all() or np.max(np.abs(tgt)) > 15:
            print("[SAFETY] invalid tgt → damping")
            create_damping_cmd(self.low_cmd)
            self._send_cmd()
            return True

        # ── 写指令 & 发送 ───────────────────────────────
        for i, m in enumerate(self.config.action_joint2motor_idx):
            mc = self.low_cmd.motor_cmd[m]
            mc.q, mc.qd, mc.kp, mc.kd, mc.tau = tgt[i], 0, self.policy.kps[i], self.policy.kds[i], 0
        for j, m in enumerate(self.config.fixed_joint2motor_idx):
            mc = self.low_cmd.motor_cmd[m]
            mc.q, mc.qd, mc.kp, mc.kd, mc.tau = (
                self.config.fixed_target[j],
                0,
                self.config.fixed_kps[j],
                self.config.fixed_kds[j],
                0,
            )
        self._send_cmd()
        return False

    # ── 内部统一发送 ───────────────────────────────────────
    def _send_cmd(self):
        self.low_cmd.crc = CRC().Crc(self.low_cmd)
        self.pub.Write(self.low_cmd)


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════
class _Logger:
    def __init__(self, fname: str):
        self.term = sys.stdout
        self.log  = open(fname, "w", buffering=1)
    def write(self, m): self.term.write(m); self.log.write(m)
    def flush(self): self.term.flush(); self.log.flush()


if __name__ == "__main__":
    pa = argparse.ArgumentParser("G1 Loco-only deploy")
    pa.add_argument("net",    help="network interface, e.g. eth0")
    pa.add_argument("config", help="absolute path to Loco YAML")
    args = pa.parse_args()

    log_name = f"deploy_log_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    sys.stdout = _Logger(log_name)

    # DDS 初始化
    ChannelFactoryInitialize(0, args.net)

    # YAML 绝对路径直接传入 Config
    cfg = Config(args.config)

    ctrl = Controller(cfg)

    # Boot phases
    ctrl.zero_torque_state()
    ctrl.move_to_default_pos()
    ctrl.default_pos_state()

    print("[RUN]  Left stick XY + Right-stick X 控制速度 | X reset | SELECT quit")
    try:
        while True:
            if ctrl.run():
                break
            time.sleep(ctrl.dt)
    except KeyboardInterrupt:
        pass

    create_damping_cmd(ctrl.low_cmd)
    ctrl._send_cmd()
    print("Exit cleanly.")
