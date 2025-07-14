#!/usr/bin/env python3
# G1 FK/IK and Trajectory Test Script

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
# G1 Joint Index
# -----------------------------------------------------------------------------
class G1JointIndex:
    # only arm joints
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    WaistYaw = 12
    WaistRoll = 13
    WaistPitch = 14
    RightElbow = 25
    LeftWristRoll = 19
    LeftWristPitch = 20
    LeftWristYaw = 21
    RightWristRoll = 26
    RightWristPitch = 27
    RightWristYaw = 28

# -----------------------------------------------------------------------------
# Load config for action_joints, control_dt, default_angles
# -----------------------------------------------------------------------------
class Config: pass

def load_cfg(path="deploy_real/configs/config_high_level.yaml"):
    with open(path, 'r') as f:
        d = yaml.safe_load(f)
    cfg = Config()
    for k, v in d.items():
        setattr(cfg, k, np.array(v) if isinstance(v, list) else v)
    return cfg

cfg = load_cfg()

# -----------------------------------------------------------------------------
# Test Controller
# -----------------------------------------------------------------------------
class FKIKTester:
    def __init__(self):
        self.low_state = None
        self.crc = CRC()
        self.first_state = False
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.remote = RemoteController()
        # IK solver
        self.ik = G1_29_ArmIK(Unit_Test=False, Visualization=False)

    def Init(self):
        self.cmd_pub = ChannelPublisher("rt/arm_sdk", LowCmd_); self.cmd_pub.Init()
        self.state_sub = ChannelSubscriber("rt/lowstate", LowState_); self.state_sub.Init(self.cb, 10)

    def cb(self, msg: LowState_):
        self.low_state = msg
        if not self.first_state:
            self.first_state = True
        self.remote.set(msg.wireless_remote)

    def forward_kin(self, q):
        pin.forwardKinematics(self.ik.reduced_robot.model, self.ik.reduced_robot.data, q)
        pin.updateFramePlacements(self.ik.reduced_robot.model, self.ik.reduced_robot.data)
        Mf = self.ik.reduced_robot.data.oMf[self.ik.L_hand_id]
        return Mf

    def send_joint(self, q):
        for i, m in enumerate(cfg.action_joints):
            self.low_cmd.motor_cmd[m].q = float(q[i])
            self.low_cmd.motor_cmd[m].dq = 0.0
            self.low_cmd.motor_cmd[m].kp = float(cfg.kps_play[i])
            self.low_cmd.motor_cmd[m].kd = float(cfg.kds_play[i])
            self.low_cmd.motor_cmd[m].tau = 0.0
        for i, m in enumerate(cfg.fixed_joints):
            self.low_cmd.motor_cmd[m].q = float(cfg.fixed_target[i])
            self.low_cmd.motor_cmd[m].dq = 0.0
            self.low_cmd.motor_cmd[m].kp = float(cfg.fixed_kps[i])
            self.low_cmd.motor_cmd[m].kd = float(cfg.fixed_kds[i])
            self.low_cmd.motor_cmd[m].tau = 0.0
        
        # self.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = 1
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.cmd_pub.Write(self.low_cmd)

    def Loop(self):
        if not self.first_state:
            return

    def Start(self):
        self.thread = RecurrentThread(interval=cfg.control_dt, target=self.Loop, name="control")
        while not self.first_state:
            time.sleep(0.5)
            print("[INIT] Waiting for first state...")
        self.thread.Start()
        print("[INIT] Smooth transition to default...")

    def Run(self):
        while True:
            # FK test: L1 prints current EE
            if self.remote.button[KeyMap.L1] == 1:
                q = np.array([self.low_state.motor_state[m].q for m in cfg.action_joints])
                Mf = self.forward_kin(q)
                print(f"[FK] L EE pos: {Mf.translation}, rot:\n{Mf.rotation}")
                time.sleep(0.3)

            # IK test: L2 moves EE +5cm in z
            if self.remote.button[KeyMap.L2] == 1:
                q0 = np.array([self.low_state.motor_state[m].q for m in cfg.action_joints])
                Mf0 = self.forward_kin(q0)
                print(f"[FK] L EE pos: {Mf0.translation}, rot:\n{Mf0.rotation}")
                T_target = pin.SE3(Mf0.rotation, Mf0.translation + np.array([0,0,0.05]))
                steps = 50
                # interpolate and command
                for t in np.linspace(0,1,steps):
                    p = Mf0.translation*(1-t) + T_target.translation*t
                    R = Mf0.rotation
                    Mf_h = np.vstack([np.hstack([R, p.reshape(3,1)]), [0,0,0,1]])
                    print(f"[FK] Current workspace status: {Mf_h}, target status: {T_target}")
                    q_cmd, _ = self.ik.solve_ik(Mf_h, Mf_h, current_lr_arm_motor_q=q0)
                    print(f"[IK] Solved joint space cmd: {q_cmd}")
                    # comment out for debug 
                    self.send_joint(q_cmd)  
                    time.sleep(cfg.control_dt)
                print("[IK] Move +5cm in Z done.")
                time.sleep(0.3)

            # Trajectory test: START draws circle
            if self.remote.button[KeyMap.start] == 1:
                q0 = np.array([self.low_state.motor_state[m].q for m in cfg.action_joints])
                Mf0 = self.forward_kin(q0)
                center = Mf0.translation.copy()
                radius = 0.05
                steps = 100
                for i in range(steps):
                    angle = 2*np.pi*i/steps
                    p = center + np.array([radius*np.cos(angle), radius*np.sin(angle), 0])
                    R = Mf0.rotation
                    Mf_h = np.vstack([np.hstack([R, p.reshape(3,1)]), [0,0,0,1]])
                    print(f"[FK] Current workspace status: {Mf_h}, target status: {T_target}")
                    q_cmd, _ = self.ik.solve_ik(Mf_h, Mf_h, current_lr_arm_motor_q=q0)
                    print(f"[IK] Solved joint space cmd: {q_cmd}")
                    # comment out for debug
                    # self.send_joint(q_cmd)  
                    time.sleep(cfg.control_dt)
                print("[TRAJ] Circle done.")
                time.sleep(0.3)
    

if __name__ == "__main__":
    ChannelFactoryInitialize(0, sys.argv[1] if len(sys.argv)>1 else None)
    tester = FKIKTester()
    tester.Init()
    tester.Start()
    tester.Run()
