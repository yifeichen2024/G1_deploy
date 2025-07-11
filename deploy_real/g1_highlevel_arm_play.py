#!/usr/bin/env python3
# G1 Workspace Motion Recorder with IK Conversion and Quick Replay

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
from scipy.spatial.transform import Slerp, Rotation

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
    # optional: transition duration for replay prep
    cfg.replay_transition_duration = 2.0
    return cfg

cfg = load_cfg()

# -----------------------------------------------------------------------------
# Workspace Recorder using Arm IK for conversion and replay
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
        # buffers for last segment
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
        cur = self.current_target_q.copy()
        steps = int(duration / cfg.control_dt)
        for i in range(steps):
            r = (i + 1) / steps
            self.current_target_q = (1 - r) * cur + r * cfg.default_angles
            time.sleep(cfg.control_dt)
        self.current_target_q = cfg.default_angles.copy()

    def forward_kin(self, q: np.ndarray):
        pin.forwardKinematics(self.arm_ik.reduced_robot.model, self.arm_ik.reduced_robot.data, q)
        pin.updateFramePlacements(self.arm_ik.reduced_robot.model, self.arm_ik.reduced_robot.data)
        Mf_L = self.arm_ik.reduced_robot.data.oMf[self.arm_ik.L_hand_id]
        Mf_R = self.arm_ik.reduced_robot.data.oMf[self.arm_ik.R_hand_id]
        return Mf_L, Mf_R

    def workspace_interpolation(self, start_L_h, start_R_h, end_L_h, end_R_h, steps):
        pA_L = start_L_h[:3, 3]; pB_L = end_L_h[:3, 3]
        pA_R = start_R_h[:3, 3]; pB_R = end_R_h[:3, 3]
        inter_p_L = np.linspace(pA_L, pB_L, steps)
        inter_p_R = np.linspace(pA_R, pB_R, steps)
        rot_seq_L = Rotation.from_matrix([start_L_h[:3, :3], end_L_h[:3, :3]])
        rot_seq_R = Rotation.from_matrix([start_R_h[:3, :3], end_R_h[:3, :3]])
        slerp_L = Slerp([0, 1], rot_seq_L)
        slerp_R = Slerp([0, 1], rot_seq_R)
        times = np.linspace(0, 1, steps)
        mats = []
        for i, t in enumerate(times):
            R_L = slerp_L(t).as_matrix()
            R_R = slerp_R(t).as_matrix()
            MfL = np.vstack([np.hstack([R_L, inter_p_L[i].reshape(3, 1)]), [0, 0, 0, 1]])
            MfR = np.vstack([np.hstack([R_R, inter_p_R[i].reshape(3, 1)]), [0, 0, 0, 1]])
            mats.append((MfL, MfR))
        return mats

    def prepare_replay(self):
        if not self.record_Mf_L:
            print("[REPLAY] No segment recorded.")
            return
        print("[REPLAY] Preparing transition to first frame...")
        cur_q = self.current_target_q.copy()
        cur_Mf_L, cur_Mf_R = self.forward_kin(cur_q)
        first_L = self.record_Mf_L[0]
        first_R = self.record_Mf_R[0]
        steps = int(cfg.replay_transition_duration / cfg.control_dt)
        seq = self.workspace_interpolation(cur_Mf_L.homogeneous, cur_Mf_R.homogeneous,
                                          first_L, first_R, steps)
        for MfL_h, MfR_h in seq:
            sol_q, _ = self.arm_ik.solve_ik(MfL_h, MfR_h, current_lr_arm_motor_q=self.current_target_q)
            self.current_target_q = sol_q
            time.sleep(cfg.control_dt)
        print("[REPLAY] Ready at first frame.")

    def do_replay(self):
        if not self.record_Mf_L:
            print("[REPLAY] No segment recorded.")
            return
        print("[REPLAY] Starting replay...")
        for MfL_h, MfR_h in zip(self.record_Mf_L, self.record_Mf_R):
            sol_q, _ = self.arm_ik.solve_ik(MfL_h, MfR_h, current_lr_arm_motor_q=self.current_target_q)
            self.current_target_q = sol_q
            time.sleep(cfg.control_dt)
        print("[REPLAY] Replay finished. Transitioning to default.")
        self.move_to_default(6)

    def Loop(self):
        if self.low_state is None:
            return
        # build stiffness arrays
        kps, kds = [], []
        for idx, joint in enumerate(cfg.action_joints):
            factor = cfg.stiffness_factor_waist_rp if joint in [G1JointIndex.WaistRoll, G1JointIndex.WaistPitch] else cfg.stiffness_factor
            kps.append(cfg.kps_play[idx] * factor)
            kds.append(cfg.kds_play[idx] * factor)
        kp_arr, kd_arr = (kps, kds) if self.recording else (cfg.kps_play, cfg.kds_play)
        # send command
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
        # record if needed
        if self.recording:
            t = time.time() - self.t_record_start
            q = np.array([self.low_state.motor_state[m].q for m in cfg.action_joints])
            Mf_L, Mf_R = self.forward_kin(q)
            self.record_times.append(t)
            self.record_qs.append(q.copy())
            self.record_Mf_L.append(Mf_L.homogeneous.copy())
            self.record_Mf_R.append(Mf_R.homogeneous.copy())

    def Start(self):
        self.thread = RecurrentThread(interval=cfg.control_dt, target=self.Loop, name="control")
        while not self.first_state:
            time.sleep(0.5)
            print("[INIT] Waiting for first state...")
        self.thread.Start()
        print("[INIT] Smooth transition to default...")
        self.recording = True
        self.move_to_default(duration=3.0)
        self.recording = False

    def Run(self):
        self.move_to_default()
        print("[RECORD/REPLAY] A:start record, B:stop/save, Y:finish, X:reset, UP:prep replay, DOWN:start replay")
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
                np.savez_compressed(
                    f"segment_ws_{seg_id:02d}.npz",
                    t=np.array(self.record_times),
                    q=np.vstack(self.record_qs),
                    Mf_L=np.stack(self.record_Mf_L),
                    Mf_R=np.stack(self.record_Mf_R)
                )
                print(f"[SAVE] segment_ws_{seg_id:02d}.npz saved")
                seg_id += 1

            elif self.remote.button[KeyMap.UP] == 1:
                self.prepare_replay()

            elif self.remote.button[KeyMap.DOWN] == 1:
                self.do_replay()

            elif self.remote.button[KeyMap.Y] == 1:
                print("[RECORD] Finish and enter damping mode")
                for i in cfg.action_joints:
                    self.low_cmd.motor_cmd[i].kp = 0
                    self.low_cmd.motor_cmd[i].kd = 1
                self.arm_pub.Write(self.low_cmd)
                print("Press X to return to default")
                while True:
                    if self.remote.button[KeyMap.X] == 1:
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
