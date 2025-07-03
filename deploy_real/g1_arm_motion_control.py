import argparse
import time
import sys
import os
from pathlib import Path
from typing import List
import select

import numpy as np

from unitree_sdk2py.core.channel import (
    ChannelPublisher,
    ChannelSubscriber,
    ChannelFactoryInitialize,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

# -----------------------------------------------------------------------------
# 1. Robot Joint Index Map  (Unitree G1 29 DOF)
# -----------------------------------------------------------------------------

class G1JointIndex:
    # legs (0–11) – not controlled here
    LeftHipPitch = 0
    LeftHipRoll = 1
    LeftHipYaw = 2
    LeftKnee = 3
    LeftAnklePitch = 4
    LeftAnkleRoll = 5
    RightHipPitch = 6
    RightHipRoll = 7
    RightHipYaw = 8
    RightKnee = 9
    RightAnklePitch = 10
    RightAnkleRoll = 11

    # waist (we will use all 3)
    WaistYaw = 12
    WaistRoll = 13
    WaistPitch = 14

    # left arm
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20
    LeftWristYaw = 21

    # right arm
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27
    RightWristYaw = 28

    kNotUsedJoint = 29  # SDK‑enable flag bit

# -----------------------------------------------------------------------------
# 2. High‑level Config – **edit这里即可快速调整**
# -----------------------------------------------------------------------------

class Config:
    """All tunable parameters collected在一起，方便统一修改/加载 YAML。"""

    control_dt: float = 0.02  # 20 ms outer loop

    # ------- 2.1 17 DOF we want to RECORD & PLAY (arms + 3‑axis waist) -------
    action_joint2motor_idx: List[int] = [
        # left arm 7
        G1JointIndex.LeftShoulderPitch,
        G1JointIndex.LeftShoulderRoll,
        G1JointIndex.LeftShoulderYaw,
        G1JointIndex.LeftElbow,
        G1JointIndex.LeftWristRoll,
        G1JointIndex.LeftWristPitch,
        G1JointIndex.LeftWristYaw,
        # right arm 7
        G1JointIndex.RightShoulderPitch,
        G1JointIndex.RightShoulderRoll,
        G1JointIndex.RightShoulderYaw,
        G1JointIndex.RightElbow,
        G1JointIndex.RightWristRoll,
        G1JointIndex.RightWristPitch,
        G1JointIndex.RightWristYaw,
        # waist 3
        G1JointIndex.WaistYaw,
        G1JointIndex.WaistRoll,
        G1JointIndex.WaistPitch,
    ]

    # ------- 2.2 joints to KEEP FIXED during record/play (e.g. wrists) -------
    fixed_joint2motor_idx: List[int] = [
        G1JointIndex.LeftWristRoll,
        G1JointIndex.LeftWristPitch,
        G1JointIndex.LeftWristYaw,
        G1JointIndex.RightWristRoll,
        G1JointIndex.RightWristPitch,
        G1JointIndex.RightWristYaw,
    ]
    fixed_target: List[float] = [
        -0.05,
        0.12,
        -0.03,
        -0.16,
        0.12,
        -0.02,
    ]
    fixed_kps: List[float] = [60, 60, 60, 60, 60, 60]
    fixed_kds: List[float] = [1, 1, 1, 1, 1, 1]

    # ------- 2.3 Per‑joint PD – play/default & record (soft) 17dof ---------
    # 下标与 action_joint2motor_idx 对齐
    kps_play: List[float] = [
        100, 100, 50, 50, 100, 100, 50,  # left
        100, 100, 50, 50, 100, 100, 50,  # right
        400, 400, 400,  # waist yaw/roll/pitch
    ]
    kds_play: List[float] = [
        2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2,
        5, 5, 5,
    ]

    # 录制时减小增益，便于拖拽
    kps_record: List[float] = [kp * 0.1 for kp in kps_play]
    kds_record: List[float] = [kd * 0.1 for kd in kds_play]

    # ------- 2.4 Default pose (17dof order) -------
    default_angles: List[float] = [
        0.2, 0.2, 0.0, 0.9, 0.0, 0.0, 0.0,  # left arm
        0.2, -0.2, 0.0, 0.9, 0.0, 0.0, 0.0,  # right arm
        0.0, 0.0, 0.0,  # waist
    ]

    # security clipping
    action_clip_warn_threshold: float = 1.5  # rad

cfg = Config()

# -----------------------------------------------------------------------------
# 3. Trajectory helpers  (only save 17 DOF)
# -----------------------------------------------------------------------------

def save_traj(path: Path, ts: List[float], q: List[np.ndarray]):
    np.savez_compressed(path, t=np.asarray(ts), q=np.vstack(q))
    print(f"[TRAJ] Saved {len(ts)} frames to {path}")

def load_traj(path: Path, speed: float):
    d = np.load(path)
    t, q = d["t"], d["q"]
    t = (t - t[0]) / speed  # rescale time
    return t, q

# -----------------------------------------------------------------------------
# 4. Controller class
# -----------------------------------------------------------------------------

class ArmTrajCtrl:
    def __init__(self):
        self.pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self.sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.crc = CRC()
        self.low_state: LowState_ | None = None
        self.first_state = False
        self.low_cmd = LowCmd_()

    # ---------------- DDS ----------------
    def init_dds(self):
        self.pub.Init()
        self.sub.Init(self._cb, 10)
        print("[DDS] Waiting for state...")
        while not self.first_state:
            time.sleep(0.1)
        print("[DDS] Ready.")

    def _cb(self, msg: LowState_):
        self.low_state = msg
        self.first_state = True

    # ---------------- Helpers ----------------
    def _set_arm_enable(self, en: bool):
        self.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = 1 if en else 0

    def _flush(self):
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.pub.Write(self.low_cmd)

    def _apply_fixed(self):
        for i, m in enumerate(cfg.fixed_joint2motor_idx):
            self.low_cmd.motor_cmd[m].q = cfg.fixed_target[i]
            self.low_cmd.motor_cmd[m].dq = 0
            self.low_cmd.motor_cmd[m].kp = cfg.fixed_kps[i]
            self.low_cmd.motor_cmd[m].kd = cfg.fixed_kds[i]
            self.low_cmd.motor_cmd[m].tau = 0

    # ---------------- Default pose ----------------
    def goto_default(self, duration=3.0, stiff=True):
        if self.low_state is None:
            raise RuntimeError("no lowstate yet")
        steps = int(duration / cfg.control_dt)
        cur = np.array([self.low_state.motor_state[j].q for j in cfg.action_joint2motor_idx])
        target = np.asarray(cfg.default_angles)
        for s in range(steps):
            r = (s + 1) / steps
            q_cmd = (1 - r) * cur + r * target
            for k, m in enumerate(cfg.action_joint2motor_idx):
                self.low_cmd.motor_cmd[m].q = float(q_cmd[k])
                self.low_cmd.motor_cmd[m].dq = 0
                kp_arr = cfg.kps_play if stiff else cfg.kps_record
                kd_arr = cfg.kds_play if stiff else cfg.kds_record
                self.low_cmd.motor_cmd[m].kp = kp_arr[k]
                self.low_cmd.motor_cmd[m].kd = kd_arr[k]
                self.low_cmd.motor_cmd[m].tau = 0
            self._apply_fixed()
            self._set_arm_enable(True)
            self._flush()
            time.sleep(cfg.control_dt)

    # ---------------- Record ----------------
    def record(self, out: Path):
        # set soft PD for action joints, fixed joints locked
        print("[RECORD] Prepare. Press ENTER to start, ENTER to stop…")
        input()
        ts, qs = [], []
        t0 = time.time()
        try:
            while True:
                # check stop
                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    if sys.stdin.readline().strip() == "":
                        break
                # send soft command (follow current position each loop)
                for k, m in enumerate(cfg.action_joint2motor_idx):
                    cur_q = self.low_state.motor_state[m].q
                    self.low_cmd.motor_cmd[m].q = cur_q
                    self.low_cmd.motor_cmd[m].dq = 0
                    self.low_cmd.motor_cmd[m].kp = cfg.kps_record[k]
                    self.low_cmd.motor_cmd[m].kd = cfg.kds_record[k]
                    self.low_cmd.motor_cmd[m].tau = 0
                self._apply_fixed()
                self._set_arm_enable(True)
                self._flush()
                # log
                ts.append(time.time() - t0)
                qs.append([self.low_state.motor_state[m].q for m in cfg.action_joint2motor_idx])
                time.sleep(cfg.control_dt)
        except KeyboardInterrupt:
            pass
        save_traj(out, ts, qs)
        print("[RECORD] Done & saved. Going default…")
        self.goto_default()

    # ---------------- Play ----------------
    def play(self, infile: Path, speed: float):
        t, q = load_traj(infile, speed)
        # align first frame (0.5 s)
        self._interpolate(q[0], 0.5)
        start = time.time()
        idx = 0
        try:
            while idx < len(t):
                elapsed = time.time() - start
                while idx < len(t) and t[idx] <= elapsed:
                    idx += 1
                if idx >= len(t):
                    break
                q_now = q[idx]
                for k, m in enumerate(cfg.action_joint2motor_idx):
                    self.low_cmd.motor_cmd[m].q = float(q_now[k])
                    self.low_cmd.motor_cmd[m].dq = 0
                    self.low_cmd.motor_cmd[m].kp = cfg.kps_play[k]
                    self.low_cmd.motor_cmd[m].kd = cfg.kds_play[k]
                    self.low_cmd.motor_cmd[m].tau = 0
                self._apply_fixed()
                self._set_arm_enable(True)
                self._flush()
                time.sleep(cfg.control_dt)
        except KeyboardInterrupt:
            print("[PLAY] Interrupted.")
        self.goto_default()

    # small helper
    def _interpolate(self, target_q: np.ndarray, duration: float):
        cur = np.array([self.low_state.motor_state[m].q for m in cfg.action_joint2motor_idx])
        steps = int(duration / cfg.control_dt)
        for s in range(steps):
            r = (s + 1) / steps
            q_cmd = (1 - r) * cur + r * target_q
            for k, m in enumerate(cfg.action_joint2motor_idx):
                self.low_cmd.motor_cmd[m].q = float(q_cmd[k])
                self.low_cmd.motor_cmd[m].dq = 0
                self.low_cmd.motor_cmd[m].kp = cfg.kps_play[k]
                self.low_cmd.motor_cmd[m].kd = cfg.kds_play[k]
                self.low_cmd.motor_cmd[m].tau = 0
            self._apply_fixed()
            self._set_arm_enable(True)
            self._flush()
            time.sleep(cfg.control_dt)

# -----------------------------------------------------------------------------
# 5. CLI
# -----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser("G1 upper‑body record/play tool", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("mode", choices=["record", "play"], help="Operating mode")
    p.add_argument("file", type=str, help="trajectory npz path")
    p.add_argument("--speed", type=float, default=1.0, help="playback speed scale")
    p.add_argument("--ip", type=str, default=None, help="DDS IP if not localhost")
    args = p.parse_args()

    if args.ip:
        ChannelFactoryInitialize(0, args.ip)
    else:
        ChannelFactoryInitialize(0)

    ctrl = ArmTrajCtrl()
    ctrl.init_dds()
    ctrl.goto_default()

    if args.mode == "record":
        out = Path(args.file).expanduser().absolute()
        out.parent.mkdir(parents=True, exist_ok=True)
        ctrl.record(out)
    else:
        ctrl.play(Path(args.file).expanduser().absolute(), args.speed)

    print("[MAIN] Exit.")

if __name__ == "__main__":
    main()
