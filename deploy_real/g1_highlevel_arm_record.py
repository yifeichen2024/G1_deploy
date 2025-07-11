#!/usr/bin/env python3
# G1 Workspace Motion Recorder with IK Conversion

import time
import sys
import numpy as np
import yaml
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from common.remote_controller import RemoteController, KeyMap

import pinocchio as pin
from g1_arm_IK import G1_29_ArmIK

# -----------------------------------------------------------------------------
# G1 Joint Index (copy from your CustomRecorder)
# -----------------------------------------------------------------------------
class G1JointIndex:
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    WaistYaw = 12
    WaistRoll = 13
    WaistPitch = 14
    LeftWristRoll = 19
    LeftWristPitch = 20
    LeftWristYaw = 21
    RightWristRoll = 26
    RightWristPitch = 27
    RightWristYaw = 28
    kNotUsedJoint = 29

# -----------------------------------------------------------------------------
# Config loader (reuse your existing)
# -----------------------------------------------------------------------------
class Config: pass

def load_cfg(yaml_path="deploy_real/configs/config_high_level.yaml") -> Config:
    with open(yaml_path, 'r') as f:
        d = yaml.safe_load(f)
    cfg = Config()
    for k, v in d.items():
        setattr(cfg, k, np.array(v) if isinstance(v, list) else v)
    cfg.kps_record = cfg.kps_play * 1
    cfg.kds_record = cfg.kds_play * 1
    return cfg

cfg = load_cfg()

# -----------------------------------------------------------------------------
# Workspace Recorder using Arm IK for conversion
# -----------------------------------------------------------------------------
class WorkspaceRecorder:
    def __init__(self):
        self.recording = False
        self.low_state = None
        self.crc = CRC()
        self.first_state = False
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.current_target_q = cfg.default_angles.copy()
        self.remote = RemoteController()
        self.t_record_start = time.time()
        self.record_times = []
        self.record_qs = []
        self.record_Mf_L = []
        self.record_Mf_R = []

        # IK solver instance
        self.arm_ik = G1_29_ArmIK(Unit_Test=False, Visualization=False)

    def Init(self):
        self.arm_pub = ChannelPublisher("rt/arm_sdk", LowCmd_); self.arm_pub.Init()
        self.state_sub = ChannelSubscriber("rt/lowstate", LowState_); self.state_sub.Init(self.cb, 10)

    def cb(self, msg: LowState_):
        self.low_state = msg
        if not self.first_state:
            self.first_state = True
        self.remote.set(msg.wireless_remote)

    def move_to_default(self, duration=5.0):
        cur = np.array([self.low_state.motor_state[m].q for m in cfg.action_joints])
        steps = int(duration / cfg.control_dt)
        for i in range(steps):
            r = (i + 1) / steps
            self.current_target_q = (1 - r) * cur + r * cfg.default_angles
            time.sleep(cfg.control_dt)
        self.current_target_q = cfg.default_angles.copy()



    def forward_kin(self, q: np.ndarray):
        # Numeric forward kinematics to get end-effector transforms
        pin.forwardKinematics(self.arm_ik.reduced_robot.model, self.arm_ik.reduced_robot.data, q)
        pin.updateFramePlacements(self.arm_ik.reduced_robot.model, self.arm_ik.reduced_robot.data)
        Mf_L = self.arm_ik.reduced_robot.data.oMf[self.arm_ik.L_hand_id]
        Mf_R = self.arm_ik.reduced_robot.data.oMf[self.arm_ik.R_hand_id]
        return Mf_L, Mf_R

    def Loop(self):
        if self.low_state is None:
            return

        # Build stiffness arrays
        kps_record, kds_record = [], []
        for idx, joint in enumerate(cfg.action_joints):
            factor = cfg.stiffness_factor_waist_rp if joint in [G1JointIndex.WaistRoll, G1JointIndex.WaistPitch] else cfg.stiffness_factor
            kps_record.append(cfg.kps_play[idx] * factor)
            kds_record.append(cfg.kds_play[idx] * factor)

        kp_arr, kd_arr = (kps_record, kds_record) if self.recording else (cfg.kps_play, cfg.kds_play)

        # Send LowCmd
        for i, m in enumerate(cfg.action_joints):
            self.low_cmd.motor_cmd[m].q = float(self.current_target_q[i])
            self.low_cmd.motor_cmd[m].dq = 0.0
            self.low_cmd.motor_cmd[m].kp = float(kp_arr[i])
            self.low_cmd.motor_cmd[m].kd = float(kd_arr[i])
            self.low_cmd.motor_cmd[m].tau = 0.0
        for i, m in enumerate(cfg.fixed_joints):
            self.low_cmd.motor_cmd[m].q = float(cfg.fixed_target[i])
            self.low_cmd.motor_cmd[m].dq = 0.0
            self.low_cmd.motor_cmd[m].kp = float(cfg.fixed_kps[i])
            self.low_cmd.motor_cmd[m].kd = float(cfg.fixed_kds[i])
            self.low_cmd.motor_cmd[m].tau = 0.0

        self.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = 1
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.arm_pub.Write(self.low_cmd)

        # Record workspace poses
        if self.recording:
            t = time.time() - self.t_record_start
            q = np.array([self.low_state.motor_state[m].q for m in cfg.action_joints])
            Mf_L, Mf_R = self.forward_kin(q)
            self.record_times.append(t)
            self.record_qs.append(q.copy())
            self.record_Mf_L.append(Mf_L.homogeneous.copy())
            self.record_Mf_R.append(Mf_R.homogeneous.copy())

    def Start(self):
        # self.thread = RecurrentThread(interval=cfg.control_dt, target=self.Loop, name="control")
        # while not self.first_state: 
        #     time.sleep(0.1)
        # self.thread.Start()

        self.thread = RecurrentThread(interval=cfg.control_dt, target=self.Loop, name="control")
        while not self.first_state: 
            time.sleep(0.1)
        self.thread.Start()

        # —— 启动后先做一次平滑过渡到默认姿态 —— #
        print("[INIT] 平滑过渡到默认姿态…")
        # 临时进入“录制”模式，使用低增益
        self.recording = True
        self.move_to_default(duration=3.0)
        self.recording = False


    def Run(self):
        # self.thread = RecurrentThread(interval=cfg.control_dt, target=self.Loop, name="control")
        # while not self.first_state:
        #     time.sleep(0.1)
        # self.thread.Start()
        self.move_to_default()
        print("[RECORD] Press A to start, B to stop/save, Y to finish, X to reset")
        seg_id = 1
        while True:
            if not self.recording and self.remote.button[KeyMap.A] == 1:
                self.recording = True
                self.t_record_start = time.time()
                self.record_times.clear()
                self.record_qs.clear()
                self.record_Mf_L.clear()
                self.record_Mf_R.clear()
                print(f"[RECORD] ▶ Start segment {seg_id}")

            elif self.recording and self.remote.button[KeyMap.B] == 1:
                self.recording = False
                # Save segment with both joint and workspace data
                np.savez_compressed(
                    f"segment_ws_{seg_id:02d}.npz",
                    t=np.array(self.record_times),
                    q=np.vstack(self.record_qs),
                    Mf_L=np.stack(self.record_Mf_L),
                    Mf_R=np.stack(self.record_Mf_R)
                )
                print(f"[SAVE] segment_ws_{seg_id:02d}.npz saved")
                seg_id += 1

            elif self.remote.button[KeyMap.Y] == 1:
                print("[RECORD] Finish and enter damping mode")
                for i in cfg.action_joints:
                    self.low_cmd.motor_cmd[i].kp = 0
                    self.low_cmd.motor_cmd[i].kd = 1
                self.arm_pub.Write(self.low_cmd)

                print("Press X to return to default")
                while True:
                    key = self.low_state.wireless_remote[2]
                    if self.remote.button[KeyMap.X] == 1:  # X
                        self.move_to_default(6.0)
                        print("[RECORD] Done. Ctrl+C to exit.")
                        return

            time.sleep(0.05)

if __name__ == "__main__":
    ChannelFactoryInitialize(0, sys.argv[1] if len(sys.argv) > 1 else None)
    recorder = WorkspaceRecorder()
    recorder.Init()
    recorder.Start()
    recorder.Run()
