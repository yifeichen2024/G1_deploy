import numpy as np
import time
import torch
from typing import Union 

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC

from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_hg, init_cmd_go, MotorMode
from common.rotation_helper import get_gravity_orientation, transform_imu_data
from common.command_helper import (
    create_damping_cmd, create_zero_cmd,
    init_cmd_hg, init_cmd_go,                          # === MOD
    MotorMode
)
from common.rotation_helper import get_gravity_orientation, transform_imu_data
from common.remote_controller import RemoteController, KeyMap
from common.utils import scale_values
from config import Config


class Controller:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.remote_controller = RemoteController()
        print("[DEBUG]: Start to load the policy...")
        self.policy = torch.jit.load(config.policy_path)
        print("[DEBUG]: Load successful!")

        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        self.obs = np.zeros(config.num_obs, dtype=np.float32)
        self.cmd = np.array(config.cmd_init, dtype=np.float32)
        self.counter = 0

        # === MOD: 
        if config.msg_type == "hg":
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self.low_state = unitree_hg_msg_dds__LowState_()
            self.mode_pr_ = MotorMode.PR
            self.mode_machine_ = 0

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
            self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)

        elif config.msg_type == "go":
            self.low_cmd = unitree_go_msg_dds__LowCmd_()
            self.low_state = unitree_go_msg_dds__LowState_()

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
            self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)

        else:
            raise ValueError("Invalid msg_type in config (expect 'hg' or 'go').")

        # === MOD: 等待 DDS 数据
        self.wait_for_low_state()

        # === MOD: 初始化低层指令
        if config.msg_type == "hg":
            init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        else:  # go
            init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)

        # cmd 范围映射工具
        self.range_velx = np.array(config.cmd_range["lin_vel_x"], dtype=np.float32)
        self.range_vely = np.array(config.cmd_range["lin_vel_y"], dtype=np.float32)
        self.range_velz = np.array(config.cmd_range["ang_vel_z"], dtype=np.float32)
    
    # === MOD: 分别处理 HG / GO 状态回调
    def LowStateHgHandler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def LowStateGoHandler(self, msg: LowStateGo):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    # === MOD: send_cmd 与完整版保持一致，支持两种消息类型
    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        print("Enter zero torque state. Waiting for start signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def move_to_default_pos(self):
        print("Moving to default pos.")
        total_time = 8
        num_step = int(total_time / self.config.control_dt)

        joint_idxs = self.config.action_joint2motor_idx
        kps = self.config.kps
        kds = self.config.kds
        default_pos = self.config.default_angles
        dof_size = len(joint_idxs)

        init_dof_pos = np.zeros(len(joint_idxs), dtype=np.float32)
        for i in range(len(joint_idxs)):
            init_dof_pos[i] = self.low_state.motor_state[joint_idxs[i]].q

        for i in range(num_step):
            alpha = i / num_step
            for j in range(len(joint_idxs)):
                idx = joint_idxs[j]

                self.low_cmd.motor_cmd[idx].q = init_dof_pos[j] * (1 - alpha) + default_pos[j]
                self.low_cmd.motor_cmd[idx].qd = 0
                self.low_cmd.motor_cmd[idx].kp = kps[j]
                self.low_cmd.motor_cmd[idx].kd = kds[j]
                self.low_cmd.motor_cmd[idx].tau = 0

        self.send_cmd(self.low_cmd)
        time.sleep(self.config.control_dt)

    def default_pos_state(self):
        print("Enter default pos state.")
        print("Waiting for the Button A signal...")
        while self.remote_controller.button[KeyMap.A] != 1:
            for i in range(len(self.config.action_joint2motor_idx)):
                motor_idx = self.config.action_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.default_angles[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            for i in range(len(self.config.fixed_joint2motor_idx)):
                motor_idx = self.config.fixed_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.fixed_target[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.fixed_kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.fixed_kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def run(self):
        joint_idxs = self.config.action_joint2motor_idx
        for i in range(self.config.num_actions):
            self.qj[i] = self.low_state.motor_state[joint_idxs[i]].q
            self.dqj[i] = self.low_state.motor_state[joint_idxs[i]].dq

        quat = self.low_state.imu_state.quaternion
        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)

        if self.config.imu_type == "torso":
            waist_yaw = self.low_state.motor_state[14].q  # adjust index if needed
            waist_yaw_omega = self.low_state.motor_state[14].dq
            quat, ang_vel = transform_imu_data(waist_yaw, waist_yaw_omega, quat, ang_vel)

        gravity_orientation = get_gravity_orientation(quat)
        cmd = np.array([self.remote_controller.ly, -self.remote_controller.lx, -self.remote_controller.rx])
        self.cmd = scale_values(cmd, [self.range_velx, self.range_vely, self.range_velz]) * self.config.cmd_scale

        qj_obs = (self.qj - self.config.default_angles[:self.config.num_actions]) * self.config.dof_pos_scale
        dqj_obs = self.dqj * self.config.dof_vel_scale
        ang_vel = ang_vel.squeeze() * self.config.ang_vel_scale

        self.obs[:3] = ang_vel
        self.obs[3:6] = gravity_orientation
        self.obs[6:9] = self.cmd
        self.obs[9:9 + self.config.num_actions] = qj_obs
        self.obs[9 + self.config.num_actions:9 + 2 * self.config.num_actions] = dqj_obs
        self.obs[9 + 2 * self.config.num_actions:9 + 3 * self.config.num_actions] = self.action

        obs_tensor = torch.from_numpy(self.obs.reshape(1, -1).astype(np.float32))
        self.action = self.policy(obs_tensor).detach().numpy().squeeze()

        target_pos = self.action * self.config.action_scale + self.config.default_angles # 从 action 转换到目标位置
        
        print(f"[DEBUG] target_pos shape: {len(target_pos)}, target_pos: {target_pos}")
        # action_reorder = self.config.default_angles.copy()
        for i in range(len(self.config.action_joint2motor_idx)):
                motor_idx = self.config.action_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = target_pos[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
                print(f"[MOTOR] idx={motor_idx:02d}  target={target_pos[i]:+.3f}")

        self.send_cmd(self.low_cmd)
        time.sleep(self.config.control_dt)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument("net", type=str, help="network interface")
    parser.add_argument("config", type=str, help="config yaml in configs folder")
    args = parser.parse_args()

    config_path = f"deploy_real/configs/{args.config}"
    config = Config(config_path)
    ChannelFactoryInitialize(0, args.net)

    controller = Controller(config)
    controller.zero_torque_state()
    controller.move_to_default_pos()
    controller.default_pos_state()
    
    while True:
        try:
            controller.run()
            if controller.remote_controller.button[KeyMap.select] == 1:
                break
        except KeyboardInterrupt:
            break

    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print("Exit")
