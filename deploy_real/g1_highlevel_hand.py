import sys
import tty
import termios
import threading
import time
import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_

# DDS 话题
kTopicDex3LeftCommand  = "rt/dex3/left/cmd"
kTopicDex3RightCommand = "rt/dex3/right/cmd"
kTopicDex3LeftState    = "rt/dex3/left/state"
kTopicDex3RightState   = "rt/dex3/right/state"

# 手势预设关节值
from enum import Enum, IntEnum
class HandGesture(Enum):
    DEFAULT = "DEFAULT"
    RELEASE = "RELEASE"
    GRIP    = "GRIP"

GESTURE_Q = {
    HandGesture.DEFAULT: (
        np.array([-0.016,  0.657,  1.486, -1.578, -1.796, -1.663, -1.729]),
        np.array([-0.005, -0.916, -1.597,  1.56,   1.704,  1.553,  1.727]),
    ),
    HandGesture.RELEASE: (
        np.array([-0.016, -0.718,  0.49,  -0.727, -1.033, -0.932, -0.606]),
        np.array([-0.005, -0.916, -1.597,  1.56,   1.704,  1.553,  1.727]),
    ),
    HandGesture.GRIP: (
        np.array([-0.029,  0.426,  0.492, -0.809, -1.025, -1.071, -0.617]),
        np.array([-0.005, -0.916, -1.597,  1.56,   1.704,  1.553,  1.727]),
    ),
}

# Joint index 枚举
class Dex3_1_Left_JointIndex(IntEnum):
    kLeftHandThumb0   = 0
    kLeftHandThumb1   = 1
    kLeftHandThumb2   = 2
    kLeftHandMiddle0  = 3
    kLeftHandMiddle1  = 4
    kLeftHandIndex0   = 5
    kLeftHandIndex1   = 6

class Dex3_1_Right_JointIndex(IntEnum):
    kRightHandThumb0  = 0
    kRightHandThumb1  = 1
    kRightHandThumb2  = 2
    kRightHandIndex0  = 3
    kRightHandIndex1  = 4
    kRightHandMiddle0 = 5
    kRightHandMiddle1 = 6

# 构造 motor_mode
class _RIS_Mode:
    def __init__(self, id=0, status=0x01, timeout=0):
        self.id      = id     & 0x0F
        self.status  = status & 0x07
        self.timeout = timeout& 0x01

    def to_uint8(self):
        m = 0
        m |=  self.id
        m |= (self.status << 4)
        m |= (self.timeout<<7)
        return m

def _getch():
    """Unix 单字符读取"""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

class Dex3GestureController:
    def __init__(self, fps=50.0):
        # initial by the higher level program.
        # ChannelFactoryInitialize(0)

        self.fps = fps

        # 发布器
        self.left_pub  = ChannelPublisher(kTopicDex3LeftCommand,  HandCmd_)
        self.right_pub = ChannelPublisher(kTopicDex3RightCommand, HandCmd_)
        self.left_pub.Init()
        self.right_pub.Init()

        # 订阅器
        self.left_sub  = ChannelSubscriber(kTopicDex3LeftState,  HandState_)
        self.right_sub = ChannelSubscriber(kTopicDex3RightState, HandState_)
        self.left_sub.Init()
        self.right_sub.Init()

        # 存放最新状态
        self.left_state  = np.zeros(len(Dex3_1_Left_JointIndex),  dtype=float)
        self.right_state = np.zeros(len(Dex3_1_Right_JointIndex), dtype=float)

        # 构造消息模板
        self.left_msg  = unitree_hg_msg_dds__HandCmd_()
        self.right_msg = unitree_hg_msg_dds__HandCmd_()
        self._init_msg(self.left_msg,  Dex3_1_Left_JointIndex)
        self._init_msg(self.right_msg, Dex3_1_Right_JointIndex)

        print("[DDS] Hand Publisher & Subscriber ready.")
        self.running = True

        # 启动订阅状态线程
        self.sub_thread = threading.Thread(target=self._state_loop, daemon=True)
        self.sub_thread.start()

        self.current_gesture = np.array(self.left_state[:]), np.array(self.right_state[:])
        self.left_q_target, self.right_q_target = self.current_gesture
        self.kp_left,self.kd_left = 1.0, 0.2
        self.kp_right, self.kd_right = 1.0, 0.2
        # 启动发布命令线程
        self.pub_thread = threading.Thread(target=self._publish_loop, daemon=True)
        self.pub_thread.start()
        

    def _init_msg(self, msg, JointIndex):
        for j in JointIndex:
            ris = _RIS_Mode(id=j, status=0x01, timeout=0)
            m = msg.motor_cmd[j]
            m.mode = ris.to_uint8()
            m.q    = 0.0
            m.dq   = 0.0
            m.tau  = 0.0
            m.kp   = 1.5
            m.kd   = 0.2

    def _state_loop(self):
        """持续读取 HandState_ 并更新 self.left_state/right_state"""
        while self.running:
            lmsg = self.left_sub.Read()
            rmsg = self.right_sub.Read()
            if lmsg is not None:
                for idx, j in enumerate(Dex3_1_Left_JointIndex):
                    self.left_state[idx] = lmsg.motor_state[j].q
            if rmsg is not None:
                for idx, j in enumerate(Dex3_1_Right_JointIndex):
                    self.right_state[idx] = rmsg.motor_state[j].q
            time.sleep(1.0 / (self.fps * 2))  # 以更高频率读状态

    def _publish_loop(self):
        """持续发布当前 target，并打印最新状态"""
        interval = 1.0 / self.fps
        while self.running:
            # 更新命令 q
            for idx, j in enumerate(Dex3_1_Left_JointIndex):
                self.left_msg.motor_cmd[j].q = float(self.left_q_target[idx])
                self.left_msg.motor_cmd[j].kp = self.kp_left
                self.left_msg.motor_cmd[j].kd = self.kd_left
            for idx, j in enumerate(Dex3_1_Right_JointIndex):
                self.right_msg.motor_cmd[j].q = float(self.right_q_target[idx])
                self.right_msg.motor_cmd[j].kp = self.kp_right
                self.right_msg.motor_cmd[j].kd = self.kd_right

            # 发布
            self.left_pub.Write(self.left_msg)
            self.right_pub.Write(self.right_msg)

            # 打印状态（可选）
            # print(f"[STATE] L: {np.round(self.left_state,3)} | R: {np.round(self.right_state,3)}", end="\r")

            time.sleep(interval)

    def zero_torque(self):
        for id in Dex3_1_Left_JointIndex:
            # self.left_msg.motor_cmd[id].kp = 0.0
            # self.left_msg.motor_cmd[id].kd = 0.0
            self.kp_left, self.kd_left = 0.0, 0.0
        for id in Dex3_1_Right_JointIndex:
            # self.right_msg.motor_cmd[id].kp = 0.0
            # self.right_msg.motor_cmd[id].kd = 0.0
            self.kp_right, self.kd_right = 0.0, 0.0

    def damping(self):
        for id in Dex3_1_Left_JointIndex:
            # self.left_msg.motor_cmd[id].kp = 0.0
            # self.left_msg.motor_cmd[id].kd = 0.2
            self.kp_left, self.kd_left = 0.0, 0.2
        for id in Dex3_1_Right_JointIndex:
            # self.right_msg.motor_cmd[id].kp = 0.0
            # self.right_msg.motor_cmd[id].kd = 0.2
            self.kp_right, self.kd_right = 0.0, 0.2

    def motion(self):
        for id in Dex3_1_Left_JointIndex:
            # self.left_msg.motor_cmd[id].kp = 0.0
            # self.left_msg.motor_cmd[id].kd = 0.2
            self.kp_left, self.kd_left = 2.0, 0.5
        for id in Dex3_1_Right_JointIndex:
            # self.right_msg.motor_cmd[id].kp = 0.0
            # self.right_msg.motor_cmd[id].kd = 0.2
            self.kp_right, self.kd_right = 2.0, 0.5

    def switch_gesture(self, gesture: HandGesture):
        if gesture not in GESTURE_Q:
            print(f"[WARN] 未知手势: {gesture}")
            return
        self.current_gesture = gesture
        self.left_q_target, self.right_q_target = GESTURE_Q[gesture]
        self.motion()
        print(f"\n[INFO] 切换到手势 {gesture.name}")

    def run(self):
        print("按键切换手势： z(zero torque) p(print state) d(DEFAULT)  r(RELEASE)  g(GRIP)  q(退出)")
        while True:
            ch = _getch().lower()
            if ch == 'z':
                self.left_q_target = self.left_state
                self.right_q_target = self.right_state
                self.zero_torque()
            if ch == 'p':
                print(f"[STATE] L: {np.round(self.left_state,3)} | R: {np.round(self.right_state,3)}", end="\r")

                
            if ch == 'd':
                self.switch_gesture(HandGesture.DEFAULT)
                print(f"[STATE] L: {np.round(self.left_state,3)} | R: {np.round(self.right_state,3)}", end="\r")
            elif ch == 'r':
                self.switch_gesture(HandGesture.RELEASE)
                print(f"[STATE] L: {np.round(self.left_state,3)} | R: {np.round(self.right_state,3)}", end="\r")

            elif ch == 'g':
                self.switch_gesture(HandGesture.GRIP)
                print(f"[STATE] L: {np.round(self.left_state,3)} | R: {np.round(self.right_state,3)}", end="\r")

            elif ch == 'q':
                print("\n退出程序...")
                print(f"[STATE] L: {np.round(self.left_state,3)} | R: {np.round(self.right_state,3)}", end="\r")
                self.damping()
                self.zero_torque()
                time.sleep(0.5)
                self.running = False
                break

if __name__ == "__main__":
    controller = Dex3GestureController(fps=30.0)
    controller.run()
