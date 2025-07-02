"""
GMT deploy script for Unitree G1 
================================================
Key features
- Loads `pretrain.pt` (torch.jit) and a motion `.pkl` file to feed the GMT
  policy.
- Reads *all* gains, default angles, joint index maps, etc. from your YAML
  config (same format you already use for PBHC). No hard‑coded numbers except
  GMT‑specific scales.
- Keeps the familiar remote‑controller workflow:
  *START* -> move to default -> hold until *A* -> run; *X* returns to default;
  *SELECT* exits.
- Observation exactly matches GMT design (motion target + 74‑dim prop +
  20‑frame history).
- 23‑DoF action vector is mapped only to `action_joint2motor_idx`; locked joints
  (`fixed_joint2motor_idx`, e.g. wrists) stay at the specified targets.

Tested locally with mypy and black (no utf‑8 punctuation in identifiers).
"""

from collections import deque
import time
import numpy as np
import torch

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
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
from utils.motion_lib import MotionLib


class GMTController:
    """Runtime controller that feeds GMT policy and drives the G1."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device("cpu")  # change to "cuda" if you prefer
        self.remote = RemoteController()

        # ---------------- policy -----------------
        # policy_path = getattr(cfg, "policy_paths", None)
        # print("Policy path from config:", policy_path)
        # if policy_path is None and hasattr(cfg, "policy_paths"):
        #     policy_path = cfg.policy_paths[0]  # take the first one by default
        # if policy_path is None:
        #     raise ValueError("policy_path not set in YAML")
        self.policy = torch.jit.load(cfg.policy_paths[0], map_location=self.device)
        self.policy.eval()

        # ---------------- motion -----------------
        self.motion_lib = MotionLib(cfg.motion_path, device=self.device)
        self.motion_ids = torch.tensor([0], dtype=torch.long, device=self.device)
        self.tar_obs_steps = torch.tensor(
            [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
            device=self.device,
        )

        # ---------------- history ----------------
        self.history_len = 20
        self.hist_buf: deque[np.ndarray] = deque(
            [np.zeros(74, dtype=np.float32) for _ in range(self.history_len)], maxlen=self.history_len
        )

        # ---------------- DDS --------------------
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = unitree_hg_msg_dds__LowState_()
        self.lowcmd_pub = ChannelPublisher(cfg.lowcmd_topic, LowCmdHG)
        self.lowcmd_pub.Init()
        self.lowstate_sub = ChannelSubscriber(cfg.lowstate_topic, LowStateHG)
        self.lowstate_sub.Init() # self._lowstate_cb, 10
        init_cmd_hg(self.low_cmd, 0, MotorMode.PR)

        # ---------------- cache ------------------
        self.num_actions = len(cfg.action_joint2motor_idx)  # 23
        self.qj = np.zeros(self.num_actions, dtype=np.float32)
        self.dqj = np.zeros_like(self.qj)
        self.last_action = np.zeros_like(self.qj)
        self.counter = 0

        # ----- GMT scales (can be overridden by YAML) -----
        self.dof_pos_scale = getattr(cfg, "dof_pos_scale", 1.0)
        self.dof_vel_scale = getattr(cfg, "dof_vel_scale", 0.05)
        self.ang_vel_scale = getattr(cfg, "ang_vel_scale", 0.25)
        self.action_scale = getattr(cfg, "action_scale", 0.5)  # PBHC cfg was 0.25

    # ------------------------------------------------------------------
    # DDS callback
    def _lowstate_cb(self, msg: LowStateHG):
        self.low_state = msg
        self.remote.set(msg.wireless_remote)

    # ------------------------------------------------------------------
    # waiting phase
    def wait_robot_ready(self):
        print("[INFO] waiting for valid LowState …")
        while self.low_state.tick == 0:
            time.sleep(self.cfg.control_dt)
        print("[INFO] robot ready ✔")

    # ------------------------------------------------------------------
    def zero_torque_loop(self):
        print("[INFO] zero‑torque hold (press START to continue)")
        while self.remote.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self._send()
            time.sleep(self.cfg.control_dt)

    # ------------------------------------------------------------------
    def move_to_default(self, duration: float = 5.0):
        print("[INFO] moving to default pose …")
        steps = int(duration / self.cfg.control_dt)
        init_q = np.array([self.low_state.motor_state[i].q for i in self.cfg.action_joint2motor_idx])
        for t in range(steps):
            a = (t + 1) / steps
            target = init_q * (1 - a) + self.cfg.default_angles * a
            self._fill_action_cmd(target)
            self._send()
            time.sleep(self.cfg.control_dt)
        print("[INFO] at default")

    # ------------------------------------------------------------------
    def hold_default_pose(self):
        print("[INFO] holding default (press A to run policy)")
        while self.remote.button[KeyMap.A] != 1:
            self._fill_action_cmd(self.cfg.default_angles)
            self._send()
            time.sleep(self.cfg.control_dt)

    # ------------------------------------------------------------------
    def run_once(self):
        # ----- read robot state -----
        for i, idx in enumerate(self.cfg.action_joint2motor_idx):
            self.qj[i] = self.low_state.motor_state[idx].q
            self.dqj[i] = self.low_state.motor_state[idx].dq
        quat = np.array(self.low_state.imu_state.quaternion)
        ang_vel = np.array(self.low_state.imu_state.gyroscope) * self.ang_vel_scale
        rpy = get_gravity_orientation(quat)[:2]

        # ----- prop obs (74) -----
        obs_prop = np.concatenate(
            [
                ang_vel,
                rpy,
                (self.qj - self.cfg.default_angles) * self.dof_pos_scale,
                self.dqj * self.dof_vel_scale,
                self.last_action,
            ],
            dtype=np.float32,
        )
        self.hist_buf.append(obs_prop)

        # ----- motion target -----
        mimic_obs = self._calc_mimic_obs(self.counter)

        # ----- full obs -----
        obs = np.concatenate([mimic_obs, obs_prop, np.array(self.hist_buf).flatten()])
        action = self.policy(torch.from_numpy(obs).unsqueeze(0)).cpu().numpy().squeeze()
        self.last_action = action.copy()

        target = self.cfg.default_angles + action * self.action_scale
        self._fill_action_cmd(target)

        # DEBUG first
        self._send()

        self.counter += 1
        time.sleep(self.cfg.control_dt)

    # ------------------------------------------------------------------
    # helper: mimic observation
    def _calc_mimic_obs(self, step: int) -> np.ndarray:
        root_t = torch.tensor([step * self.cfg.control_dt], device=self.device).unsqueeze(-1)
        obs_times = self.tar_obs_steps.float() * self.cfg.control_dt + root_t
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, _ = self.motion_lib.calc_motion_frame(
            self.motion_ids, obs_times.flatten()
        )
        roll, pitch, _ = get_gravity_orientation(root_rot[0].cpu().numpy())
        return np.concatenate(
            [
                [root_pos[0, 2].item()],
                [roll, pitch],
                root_vel[0].cpu().numpy(),
                root_ang_vel[0, 2].cpu().numpy()[None],
                dof_pos[0].cpu().numpy(),
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # helper: fill command (23 actuated + fixed joints)
    def _fill_action_cmd(self, target: np.ndarray):
        for i, m_idx in enumerate(self.cfg.action_joint2motor_idx):
            self.low_cmd.motor_cmd[m_idx].q = float(target[i])
            self.low_cmd.motor_cmd[m_idx].qd = 0.0
            self.low_cmd.motor_cmd[m_idx].kp = float(self.cfg.kps[i])
            self.low_cmd.motor_cmd[m_idx].kd = float(self.cfg.kds[i])
            self.low_cmd.motor_cmd[m_idx].tau = 0.0
            print(f"[DEBUG] Motor {m_idx}, target: {target[i]}")
        for i, m_idx in enumerate(self.cfg.fixed_joint2motor_idx):
            self.low_cmd.motor_cmd[m_idx].q = float(self.cfg.fixed_target[i])
            self.low_cmd.motor_cmd[m_idx].qd = 0.0
            self.low_cmd.motor_cmd[m_idx].kp = float(self.cfg.fixed_kps[i])
            self.low_cmd.motor_cmd[m_idx].kd = float(self.cfg.fixed_kds[i])
            self.low_cmd.motor_cmd[m_idx].tau = 0.0

    def _send(self):
        self.low_cmd.crc = CRC().Crc(self.low_cmd)
        self.lowcmd_pub.Write(self.low_cmd)


# ---------------------------------------------------------------------------
if __name__ == "__main__":

    import argparse
    import sys
    import datetime
    # log info
    log_filename = f"deploy_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "w")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()

        def flush(self):
            pass

    sys.stdout = Logger(log_filename)
    print(f"[INFO] Logging to {log_filename}")

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface, e.g. enp2s0")
    parser.add_argument("config", type=str, help="yaml under deploy_real/configs/")
    args = parser.parse_args()

    cfg = Config(f"deploy_real/configs/{args.config}")
    ChannelFactoryInitialize(0, args.net)

    ctrl = GMTController(cfg)
    ctrl.wait_robot_ready()
    ctrl.zero_torque_loop()
    ctrl.move_to_default()
    ctrl.hold_default_pose()

    print("[INFO] running (SELECT exits, X back to default)")
    while True:
        try:
            ctrl.run_once()
            if ctrl.remote.button[KeyMap.select] == 1:
                break
            if ctrl.remote.button[KeyMap.X] == 1:
                ctrl.move_to_default()
        except KeyboardInterrupt:
            break

    create_damping_cmd(ctrl.low_cmd)
    ctrl._send()
    print("[INFO] done")
