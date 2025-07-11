import time
import numpy as np
from multiprocessing import Array
from enum import IntEnum
import threading

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_

from common.remote_controller import RemoteController, KeyMap

Dex3_Num_Motors = 7
kTopicDex3LeftCommand = "rt/dex3/left/cmd"
kTopicDex3RightCommand = "rt/dex3/right/cmd"
kTopicDex3LeftState = "rt/dex3/left/state"
kTopicDex3RightState = "rt/dex3/right/state"

class Dex3_1_Left_JointIndex(IntEnum):
    kLeftHandThumb0 = 0
    kLeftHandThumb1 = 1
    kLeftHandThumb2 = 2
    kLeftHandMiddle0 = 3
    kLeftHandMiddle1 = 4
    kLeftHandIndex0 = 5
    kLeftHandIndex1 = 6

class Dex3_1_Right_JointIndex(IntEnum):
    kRightHandThumb0 = 0
    kRightHandThumb1 = 1
    kRightHandThumb2 = 2
    kRightHandIndex0 = 3
    kRightHandIndex1 = 4
    kRightHandMiddle0 = 5
    kRightHandMiddle1 = 6

class Dex3_1_Controller:
    def __init__(self, fps=100.0):
        self.fps = fps
        ChannelFactoryInitialize(0)

        # Initialize DDS channels
        self.left_pub = ChannelPublisher(kTopicDex3LeftCommand, HandCmd_)
        self.right_pub = ChannelPublisher(kTopicDex3RightCommand, HandCmd_)
        self.left_pub.Init()
        self.right_pub.Init()
        
        self.left_sub = ChannelSubscriber(kTopicDex3LeftState, HandState_)
        self.right_sub = ChannelSubscriber(kTopicDex3RightState, HandState_)
        self.left_sub.Init()
        self.right_sub.Init()

        self.left_state = Array('d', Dex3_Num_Motors, lock=True)
        self.right_state = Array('d', Dex3_Num_Motors, lock=True)

        self.target_left_q = np.zeros(Dex3_Num_Motors)
        self.target_right_q = np.zeros(Dex3_Num_Motors)
        self.recording = True
        # Create DDS messages
        self.left_msg = unitree_hg_msg_dds__HandCmd_()
        self.right_msg = unitree_hg_msg_dds__HandCmd_()
        self._init_hand_msg()

        # Start threads
        threading.Thread(target=self._recv_state_loop, daemon=True).start()
        threading.Thread(target=self._control_loop, daemon=True).start()

        print("Dex3_1_Controller initialized.")

    def _init_hand_msg(self):
        for id in Dex3_1_Left_JointIndex:
            self.left_msg.motor_cmd[id].mode = 0x01
            self.left_msg.motor_cmd[id].kp = 1.0
            self.left_msg.motor_cmd[id].kd = 0.2

        for id in Dex3_1_Right_JointIndex:
            self.right_msg.motor_cmd[id].mode = 0x01
            self.right_msg.motor_cmd[id].kp = 0.0
            self.right_msg.motor_cmd[id].kd = 0.0

    def _recv_state_loop(self):
        while True:
            left_msg = self.left_sub.Read()
            right_msg = self.right_sub.Read()
            if left_msg:
                for i, id in enumerate(Dex3_1_Left_JointIndex):
                    self.left_state[i] = left_msg.motor_state[id].q
            if right_msg:
                for i, id in enumerate(Dex3_1_Right_JointIndex):
                    self.right_state[i] = right_msg.motor_state[id].q
            time.sleep(0.005)

    def _control_loop(self):
        while True:
            if self.recording:
                kp_left, kd_left = 0.0, 0.0
                kp_right, kd_right = 0.0, 0.0
            else:
                kp_left, kd_left = 1.0, 0.2
                kp_right, kd_right = 1.0, 0.2

            for i, id in enumerate(Dex3_1_Left_JointIndex):
                self.left_msg.motor_cmd[id].q = self.target_left_q[i]
                self.left_msg.motor_cmd[id].kp = kp_left
                self.left_msg.motor_cmd[id].kd = kd_left
            for i, id in enumerate(Dex3_1_Right_JointIndex):
                self.right_msg.motor_cmd[id].q = self.target_right_q[i]
                self.left_msg.motor_cmd[id].kp = kp_right
                self.left_msg.motor_cmd[id].kd = kd_right
            
            # load the message first comment out this one 
            self.send_cmd()
            time.sleep(1 / self.fps)

    def zero_torque(self):
        for id in Dex3_1_Left_JointIndex:
            self.left_msg.motor_cmd[id].mode = 0x01
            self.left_msg.motor_cmd[id].kp = 0.0
            self.left_msg.motor_cmd[id].kd = 0.0

        for id in Dex3_1_Right_JointIndex:
            self.right_msg.motor_cmd[id].mode = 0x01
            self.right_msg.motor_cmd[id].kp = 0.0
            self.right_msg.motor_cmd[id].kd = 0.0
        
        self.send_cmd()

    def _interpolate_motion(self, target_left_q, target_right_q, duration=2.0):
        """平滑插值从当前状态到目标状态"""
        current_left_q, current_right_q = self.get_current_q()
        steps = int(duration * self.fps)
        
        for t in range(steps):
            alpha = (t + 1) / steps
            interp_left_q  = (1 - alpha) * current_left_q  + alpha * target_left_q
            interp_right_q = (1 - alpha) * current_right_q + alpha * target_right_q
            self.set_target_q(interp_left_q, interp_right_q)
            time.sleep(1 / self.fps)

    def send_cmd(self):
        self.left_pub.Write(self.left_msg)
        self.right_pub.Write(self.right_msg)
        
    def set_target_q(self, left_q: np.ndarray, right_q: np.ndarray):
        """外部接口：设置目标角度"""
        assert left_q.shape[0] == Dex3_Num_Motors
        assert right_q.shape[0] == Dex3_Num_Motors
        self.target_left_q = left_q.copy()
        self.target_right_q = right_q.copy()

    def get_current_q(self):
        """外部接口：获取当前关节角度状态"""
        with self.left_state.get_lock(), self.right_state.get_lock():
            return np.array(self.left_state[:]), np.array(self.right_state[:])

    def move_to_default(self, duration=2.0):
        """从当前状态移动到默认手势（如张开）"""
        default_left_q = np.array([-0.2, 0.2, 0.2, -1.0, -0.3, -1.0, -0.3])
        default_right_q = np.array([-0.2, 0.2, 0.2, -1.0, -0.3, -1.0, -0.3])
        self._interpolate_motion(default_left_q, default_right_q, duration)

    def release_motion(self, duration=2.0):
        """释放动作（张开手）"""
        release_left_q = np.array([-0.2, 0.2, 0.2, -1.0, -0.3, -1.0, -0.3])
        release_right_q = np.array([-0.2, 0.2, 0.2, -1.0, -0.3, -1.0, -0.3])
        self._interpolate_motion(release_left_q, release_right_q, duration)

    def grip_motion(self, duration=2.0):
        """抓握动作（目标姿态）"""
        grip_left_q = np.array([-0.2, 0.2, 0.2, -1.0, -0.3, -1.0, -0.3])
        grip_right_q = np.array([-0.2, 0.2, 0.2, -1.0, -0.3, -1.0, -0.3])
        self._interpolate_motion(grip_left_q, grip_right_q, duration)

# if __name__ == "__main__":
#     import numpy as np

#     controller = Dex3_1_Controller()

#     print("Waiting for DDS state feedback...")
#     time.sleep(1)

#     print("Sending example target pose...")
#     # TODO record and play and hold
#     try:
#         while True:
#             left_q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
#             right_q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
#             controller.set_target_q(left_q, right_q)

#             lq, rq = controller.get_current_q()
#             print(f"Current Left Q: {lq.round(3)} \nCurrent Right Q: {rq.round(3)}")
#             time.sleep(0.5)
#     except KeyboardInterrupt:
#         controller.zero_torque()
    

if __name__ == "__main__":
    import time
    import numpy as np

    controller = Dex3_1_Controller()
    print("Waiting for DDS state feedback...")
    time.sleep(1.0)

    # 1. 定义动作序列：描述 + 对应方法
    sequence = [
        ("Move to default position", controller.move_to_default(duration=3.0)),
        ("Grip motion", controller.grip_motion(1.0)),
        ("Release motion", controller.release_motion(1.0)),
        ("Zero torque and exit", controller.zero_torque)
    ]

    print("Action sequence loaded. 按 Enter 执行下一步，输入 'q' + Enter 提前退出。")

    for desc, action in sequence:
        user_input = input(f"\n即将执行：{desc} -> 按 Enter 开始; 或输入 q + Enter 退出：")
        if user_input.lower() == 'q':
            print("提前退出，启用 zero_torque 并结束。")
            controller.zero_torque()
            break

        print(f"--- {desc} ---")
        # 如果动作有持续时间参数，也可以在这里传入，比如 move_to_default(2.0)
        controller.zero_torque()
        # action()
        # 等待动作执行完毕（根据你的动作内部默认 duration）
        # 如果需要打印当前关节状态，可以：
        lq, rq = controller.get_current_q()
        print(f"当前关节状态：\n  Left:  {lq.round(3)}\n  Right: {rq.round(3)}")

    print("动作序列结束。")