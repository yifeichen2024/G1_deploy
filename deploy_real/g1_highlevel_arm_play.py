import time
import sys
from pathlib import Path
import numpy as np
import select 

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
# TODO RecurrentThread 的报错修复
# TODO 功能按键绑定
# TODO 执行多个任务
# ------------------- G1 Joint Map (same as official) -------------------- #
class G1JointIndex:
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
    
    WaistYaw = 12
    WaistRoll = 13
    WaistPitch = 14

    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20
    LeftWristYaw = 21

    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27
    RightWristYaw = 28

    kNotUsedJoint = 29

# ---------------------------- Config ----------------------------------- #
class Config:
    control_dt = 0.02  # 20 ms

    action_joints = [
        G1JointIndex.LeftShoulderPitch, 
        G1JointIndex.LeftShoulderRoll, 
        G1JointIndex.LeftShoulderYaw,
        G1JointIndex.LeftElbow, 
        G1JointIndex.LeftWristRoll, 
        G1JointIndex.LeftWristPitch, 
        G1JointIndex.LeftWristYaw,

        G1JointIndex.RightShoulderPitch, 
        G1JointIndex.RightShoulderRoll, 
        G1JointIndex.RightShoulderYaw,
        G1JointIndex.RightElbow, 
        G1JointIndex.RightWristRoll, 
        G1JointIndex.RightWristPitch, 
        G1JointIndex.RightWristYaw,

        G1JointIndex.WaistYaw, 
        G1JointIndex.WaistRoll, 
        G1JointIndex.WaistPitch,
    ]

    # fixed wrists (lock during play)
    fixed_joints = [
        G1JointIndex.LeftWristRoll, 
        G1JointIndex.LeftWristPitch, 
        G1JointIndex.LeftWristYaw,

        G1JointIndex.RightWristRoll, 
        G1JointIndex.RightWristPitch, 
        G1JointIndex.RightWristYaw,
    ]
    fixed_target = np.array([-0.05, 0.12, -0.03, -0.16, 0.12, -0.02])
    fixed_kps = np.array([60, 60, 60, 60, 60, 60])
    fixed_kds = np.array([1, 1, 1, 1, 1, 1])

    # PD for play (17 DOF)
    kps = np.array([100, 100, 50, 50, 100, 100, 50, 
                    100, 100, 50, 50, 100, 100, 50, 
                    400, 400, 400])
    
    kds = np.array([2, 2, 2, 2, 2, 2, 2, 
                    2, 2, 2, 2, 2, 2, 2,
                    5, 5, 5])

    default_angles = np.array([
        0.2, 0.2, 0.0, 0.9, 0.0, 0.0, 0.0,
        0.2, -0.2, 0.0, 0.9, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
    ])
    
cfg = Config()

# ---------------------------- Player Class ----------------------------- #
class CustomPlayer:
    def __init__(self, traj_path: str, speed: float):
        data = np.load(traj_path)
        self.t_arr = (data['t'] - data['t'][0]) / speed  # scale time
        self.q_arr = data['q']                           # shape [N,17]
        self.idx = 0
        self.start_time = None

        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = None
        self.first_state = False
        self.crc = CRC()

    # ---------------- DDS Init ---------------- #
    def Init(self):
        self.pub = ChannelPublisher('rt/arm_sdk', LowCmd_)
        self.pub.Init()
        self.sub = ChannelSubscriber('rt/lowstate', LowState_)
        self.sub.Init(self.LowStateHandler, 10)

    def LowStateHandler(self, msg: LowState_):
        self.low_state = msg
        if not self.first_state:
            self.first_state = True

    # ---------------- Start control thread ---------------- #
    def Start(self):
        self.thread = RecurrentThread(interval=cfg.control_dt, target=self.ControlLoop, name='control')
        while not self.first_state:
            time.sleep(0.1)
        print('[DDS] State ready, starting playback…')
        self.start_time = time.time()
        self.thread.Start()

    # ---------------- Control loop ---------------- #
    def ControlLoop(self):
        elapsed = time.time() - self.start_time

        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            if sys.stdin.readline().strip() == "":
                print("[PLAY] Interrupted by user, returning to default pose.")
                # self.publish_default()
                self.move_to_default()
                return


        # advance index
        while self.idx < len(self.t_arr) and self.t_arr[self.idx] <= elapsed:
            self.idx += 1
        if self.idx >= len(self.t_arr):
            # finished → hold default pose & disable thread
            # self.publish_default()
            self.move_to_default()
            # self.thread.Stop()
            print('[PLAY] Trajectory finished.')
            return
        q_target = self.q_arr[self.idx]
        # build cmd
        for k, m in enumerate(cfg.action_joints):
            self.low_cmd.motor_cmd[m].q = float(q_target[k])
            self.low_cmd.motor_cmd[m].dq = 0.0
            self.low_cmd.motor_cmd[m].kp = float(cfg.kps[k])
            self.low_cmd.motor_cmd[m].kd = float(cfg.kds[k])
            self.low_cmd.motor_cmd[m].tau = 0.0
        # lock wrists
        for i, m in enumerate(cfg.fixed_joints):
            self.low_cmd.motor_cmd[m].q = float(cfg.fixed_target[i])
            self.low_cmd.motor_cmd[m].dq = 0.0
            self.low_cmd.motor_cmd[m].kp = float(cfg.fixed_kps[i])
            self.low_cmd.motor_cmd[m].kd = float(cfg.fixed_kds[i])
            self.low_cmd.motor_cmd[m].tau = 0.0
        # enable SDK
        self.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = 1
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.pub.Write(self.low_cmd)

    # ---------------- Helper to publish default pose ---------------- #
    def move_to_default(self, duration=5.0):
        current_q = np.array([
            self.low_state.motor_state[m].q for m in cfg.action_joints
        ])
        steps = int(duration / cfg.control_dt)
        for step in range(steps):
            ratio = (step + 1) / steps
            q_cmd = (1 - ratio) * current_q + ratio * cfg.default_angles
            for k, m in enumerate(cfg.action_joints):
                self.low_cmd.motor_cmd[m].q = float(q_cmd[k])
                self.low_cmd.motor_cmd[m].dq = 0.0
                self.low_cmd.motor_cmd[m].kp = float(cfg.kps[k])
                self.low_cmd.motor_cmd[m].kd = float(cfg.kds[k])
                self.low_cmd.motor_cmd[m].tau = 0.0
            for i, m in enumerate(cfg.fixed_joints):
                self.low_cmd.motor_cmd[m].q = float(cfg.fixed_target[i])
                self.low_cmd.motor_cmd[m].dq = 0.0
                self.low_cmd.motor_cmd[m].kp = float(cfg.fixed_kps[i])
                self.low_cmd.motor_cmd[m].kd = float(cfg.fixed_kds[i])
                self.low_cmd.motor_cmd[m].tau = 0.0
            self.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = 1
            self.low_cmd.crc = self.crc.Crc(self.low_cmd)
            self.pub.Write(self.low_cmd)
            time.sleep(cfg.control_dt)
    
    def publish_default(self):
        for k, m in enumerate(cfg.action_joints):
            self.low_cmd.motor_cmd[m].q = float(cfg.default_angles[k])
            self.low_cmd.motor_cmd[m].dq = 0.0
            self.low_cmd.motor_cmd[m].kp = float(cfg.kps[k])
            self.low_cmd.motor_cmd[m].kd = float(cfg.kds[k])
            self.low_cmd.motor_cmd[m].tau = 0.0
        for i, m in enumerate(cfg.fixed_joints):
            self.low_cmd.motor_cmd[m].q = float(cfg.fixed_target[i])
            self.low_cmd.motor_cmd[m].dq = 0.0
            self.low_cmd.motor_cmd[m].kp = float(cfg.fixed_kps[i])
            self.low_cmd.motor_cmd[m].kd = float(cfg.fixed_kds[i])
            self.low_cmd.motor_cmd[m].tau = 0.0
        self.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = 1
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.pub.Write(self.low_cmd)

# ----------------------------- Main ------------------------------------- #
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python g1_arm_play.py <traj.npz> [speed] [robot_ip]')
        sys.exit(0)

    traj_file = Path(sys.argv[1]).expanduser()
    speed = float(sys.argv[2]) if len(sys.argv) >= 3 else 1.0

    if len(sys.argv) >= 4:
        ChannelFactoryInitialize(0, sys.argv[3])
    else:
        ChannelFactoryInitialize(0)

    player = CustomPlayer(str(traj_file), speed)
    player.Init()
    player.Start()

    # keep main thread alive until playback thread ends
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print('Keyboard interrupt, exit.')
