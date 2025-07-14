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



class FKIKTester:
    def __init__(self):
        self.low_state = None
        self.crc = CRC()
        self.first_state = False
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.remote = RemoteController()
        self.ik = G1_29_ArmIK(Unit_Test=False, Visualization=False)
        # threaded control target
        self.current_target_q = cfg.default_angles.copy()
        self.mode = 'idle'
        self.ik_plan = []
        self.traj_plan = []
        self.plan_step = 0

    def Init(self):
        self.cmd_pub = ChannelPublisher("rt/arm_sdk", LowCmd_); self.cmd_pub.Init()
        self.state_sub = ChannelSubscriber("rt/lowstate", LowState_); self.state_sub.Init(self.cb, 10)

    def cb(self, msg: LowState_):
        self.low_state = msg
        if not self.first_state:
            self.first_state = True
        self.remote.set(msg.wireless_remote)

    def send_joint(self, q):
        for i, m in enumerate(cfg.action_joints):
            cmd = self.low_cmd.motor_cmd[m]
            cmd.q = float(q[i])
            cmd.dq = 0.0
            cmd.kp = float(cfg.kps_play[i])
            cmd.kd = float(cfg.kds_play[i])
            cmd.tau = 0.0
        for i, m in enumerate(cfg.fixed_joints):
            cmd = self.low_cmd.motor_cmd[m]
            cmd.q = float(cfg.fixed_target[i])
            cmd.dq = 0.0
            cmd.kp = float(cfg.fixed_kps[i])
            cmd.kd = float(cfg.fixed_kds[i])
            cmd.tau = 0.0
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.cmd_pub.Write(self.low_cmd)

    def move_to_default(self, duration=3.0):
        cur = np.array([self.low_state.motor_state[m].q for m in cfg.action_joints])
        steps = int(duration / cfg.control_dt)
        for i in range(steps):
            r = (i + 1) / steps
            self.current_target_q = (1 - r) * cur + r * cfg.default_angles
            self.send_joint(self.current_target_q)
            time.sleep(cfg.control_dt)
        self.current_target_q = cfg.default_angles.copy()

    def forward_kin(self, q):
        pin.forwardKinematics(self.ik.reduced_robot.model, self.ik.reduced_robot.data, q)
        pin.updateFramePlacements(self.ik.reduced_robot.model, self.ik.reduced_robot.data)
        Mf_L = self.ik.reduced_robot.data.oMf[self.ik.L_hand_id]
        Mf_R = self.ik.reduced_robot.data.oMf[self.ik.R_hand_id]
        return Mf_L, Mf_R
    
    def Start(self):
        while not self.first_state:
            time.sleep(0.1)
            print("[INIT] Waiting for first state...")
        # start recurrent control thread
        self.thread = RecurrentThread(interval=cfg.control_dt, target=self.Loop, name="control")
        self.thread.Start()
        print("[INIT] Smooth transition to default...")
        # self.move_to_default(duration=3.0)

    def Loop(self):
        if not self.first_state:
            return
        
        # execute IK plan
        if self.mode == 'ik' and self.plan_step < len(self.ik_plan):
            self.current_target_q = self.ik_plan[self.plan_step]
            self.plan_step += 1
            if self.plan_step >= len(self.ik_plan):
                self.mode = 'idle'
        # execute trajectory plan
        elif self.mode == 'traj' and self.plan_step < len(self.traj_plan):
            self.current_target_q = self.traj_plan[self.plan_step]
            self.plan_step += 1
            if self.plan_step >= len(self.traj_plan):
                self.mode = 'idle'
        
        # maintain current target each cycle
        self.send_joint(self.current_target_q)

    def prepare_ik(self):
        q0 = np.array([self.low_state.motor_state[m].q for m in cfg.action_joints])
        Mf0_L, Mf0_R = self.forward_kin(q0)

        T_target_L = pin.SE3(Mf0_L.rotation, Mf0_L.translation + np.array([0,0,0.05]))
        T_target_R = pin.SE3(Mf0_R.rotation, Mf0_R.translation + np.array([0,0,0.05]))
        steps = 50
        self.ik_plan = []
        for t in np.linspace(0,1,steps):
            p_L = Mf0_L.translation * (1-t) + T_target_L.translation * t
            p_R = Mf0_R.translation * (1-t) + T_target_R.translation * t
            Mf_h_L = pin.SE3(Mf0_L.rotation, p_L)
            Mf_h_R = pin.SE3(Mf0_R.rotation, p_R)
            q_cmd, _ = self.ik.solve_ik(Mf_h_L.homogeneous, Mf_h_R.homogeneous, current_lr_arm_motor_q=q0)
            print(f"[IK] Current joint: {q_cmd}")
            print(f"[IK] Mf0_L translation: {Mf_h_L.translation}, Mf0_R translation: {Mf_h_R.translation}")
            self.ik_plan.append(q_cmd)
        self.plan_step = 0
        self.mode = 'ik'

    def circular_offset(t, r=0.04, T=5.0):
        """给定时间 t返回在 y-z 平面的小圆形偏移 (dy, dz)。"""
        from math import sin, cos, tau 
        theta = (t % T) / T * tau      # 映射到 [0, 2π)
        return np.array([0.0, r * cos(theta), r * sin(theta)])

    def prepare_traj(self):
        q0 = np.array([self.low_state.motor_state[m].q for m in cfg.action_joints])

        Mf0_L, Mf0_R = self.forward_kin(q0)

        T_target_L = pin.SE3(Mf0_L.rotation, Mf0_L.translation + np.array([0,0,0.05]))
        T_target_R = pin.SE3(Mf0_R.rotation, Mf0_R.translation + np.array([0,0,0.05]))

        L_center = np.array([0.30, +0.20, 0.15])
        R_center = np.array([0.30, -0.20, 0.15])
        q_id = pin.Quaternion(1, 0, 0, 0)
        t0 = time.time()

        radius = 0.04
        steps = 100
        self.traj_plan = []
        for i in range(steps):
            t = time.time() - t0
            offset =self. circular_offset(t)
            L_target = pin.SE3(q_id, L_center + offset)
            R_target = pin.SE3(q_id, R_center + offset)
            
            q_cmd, _ = self.ik.solve_ik(L_target.homogeneous, R_target.homogeneous, current_lr_arm_motor_q=q0)
            print(f"[IK] current joint: {q_cmd}")
            print(f"[IK] Mf0_L translation: {L_target.translation}, Mf0_R translation: {R_target.translation}")
            self.traj_plan.append(q_cmd)
        self.plan_step = 0
        self.mode = 'traj'

    def Run(self):
        while True:
            if self.remote.button[KeyMap.L1] == 1:
                q = np.array([self.low_state.motor_state[m].q for m in cfg.action_joints])
                Mf0_L, Mf0_R = self.forward_kin(q)
                print(f"[FK] Current EE: {q}")
                print(f"[FK] Mf0_L translation: {Mf0_L.translation}, Mf0_R: {Mf0_R.translation}")
                time.sleep(0.3)
            if self.remote.button[KeyMap.L2] == 1:
                print("[IK] Preparing +5cm Z move...")
                self.prepare_ik()
                q = np.array([self.low_state.motor_state[m].q for m in cfg.action_joints])
                Mf0_L, Mf0_R = self.forward_kin(q)
                print(f"[FK] Current EE: {q}")
                print(f"[FK] Mf0_L translation: {Mf0_L.translation}, Mf0_R: {Mf0_R.translation}")
                time.sleep(0.3)
            if self.remote.button[KeyMap.start] == 1:
                print("[TRAJ] Preparing circle trajectory...")
                self.prepare_traj()
                q = np.array([self.low_state.motor_state[m].q for m in cfg.action_joints])
                Mf0_L, Mf0_R = self.forward_kin(q)
                print(f"[FK] Current EE: {q}")
                print(f"[FK] Mf0_L translation: {Mf0_L.translation}, Mf0_R: {Mf0_R.translation}")
                time.sleep(0.3)

if __name__ == "__main__":
    ChannelFactoryInitialize(0, sys.argv[1] if len(sys.argv)>1 else None)
    tester = FKIKTester()
    tester.Init()
    tester.Start()
    tester.Run()
