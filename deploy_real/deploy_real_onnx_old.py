from legged_gym import LEGGED_GYM_ROOT_DIR
from typing import Union
import numpy as np
import time
import torch

# import onnx
import onnxruntime


from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC

from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_hg, init_cmd_go, MotorMode
from common.rotation_helper import get_gravity_orientation, transform_imu_data
from common.remote_controller import RemoteController, KeyMap
from config import Config


class Controller:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.remote_controller = RemoteController()

        # Initialize the policy network
        # pytorch script
        # self.policy = torch.jit.load(config.policy_path)

        # onnx
        self.ort_session = onnxruntime.InferenceSession(config.policy_path)
        self.input_name = self.ort_session.get_inputs()[0].name


        # Initializing process variables
        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        # self.target_dof_pos = config.default_angles.copy()
        # self.obs = np.zeros(config.num_obs, dtype=np.float32)
        # self.cmd = np.array([0.0, 0, 0])
        self.counter = 0


        self.target_dof_pos = config.default_angles.copy()
        self.ref_motion_phase = 0
        self.ang_vel_buf = np.zeros(3 * config.history_length, dtype=np.float32)
        self.proj_g_buf = np.zeros(3 * config.history_length, dtype=np.float32)
        self.dof_pos_buf = np.zeros(config.num_actions * config.history_length, dtype=np.float32)
        self.dof_vel_buf = np.zeros(config.num_actions * config.history_length, dtype=np.float32)
        self.action_buf = np.zeros(config.num_actions * config.history_length, dtype=np.float32)
        self.ref_motion_phase_buf = np.zeros(1 * config.history_length, dtype=np.float32)


        if config.msg_type == "hg":
            # g1 and h1_2 use the hg msg type
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self.low_state = unitree_hg_msg_dds__LowState_()
            self.mode_pr_ = MotorMode.PR
            self.mode_machine_ = 0

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
            self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)

        elif config.msg_type == "go":
            # h1 uses the go msg type
            self.low_cmd = unitree_go_msg_dds__LowCmd_()
            self.low_state = unitree_go_msg_dds__LowState_()

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
            self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)

        else:
            raise ValueError("Invalid msg_type")

        # wait for the subscriber to receive data
        self.wait_for_low_state()

        # Initialize the command msg
        if config.msg_type == "hg":
            init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        elif config.msg_type == "go":
            init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)

    def LowStateHgHandler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def LowStateGoHandler(self, msg: LowStateGo):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def move_to_default_pos(self):
        print("Moving to default pos.")
        # move time 2s
        total_time = 5
        num_step = int(total_time / self.config.control_dt)
        
        dof_idx = self.config.action_joint2motor_idx + self.config.fixed_joint2motor_idx
        kps = self.config.kps + self.config.fixed_kps
        kds = self.config.kds + self.config.fixed_kds
        default_pos = np.concatenate((self.config.default_angles, self.config.fixed_target), axis=0)
        dof_size = len(dof_idx)
        
        # record the current pos
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q
            
        # move to default pos
        for i in range(num_step):
            alpha = i / num_step
            for j in range(dof_size):
                motor_idx = dof_idx[j]
                target_pos = default_pos[j]
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = kps[j]
                self.low_cmd.motor_cmd[motor_idx].kd = kds[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            
            # print("======= CHECK motor_cmd before send =======")
            # for i in range(23):  # motor 0 ~ 22
            #     cmd = self.low_cmd.motor_cmd[i]
            #     print(f"motor {i:02d}: q={cmd.q:.3f}, kp={cmd.kp:.1f}, kd={cmd.kd:.1f}, tau={cmd.tau:.1f}")

            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

            # === 实时打印当前误差 ===
            print(f"[STEP {i+1}/{num_step}] Joint position errors:")
            for j in range(dof_size):  # 打印前6个关节作为代表
                motor_idx = dof_idx[j]
                current_q = self.low_state.motor_state[motor_idx].q
                target_q = self.low_cmd.motor_cmd[motor_idx].q
                err = current_q - target_q
                print(f"  motor {motor_idx:02d}: target={target_q:+.3f}, actual={current_q:+.3f}, error={err:+.3f}")
        # # === check whether final position reached ===
        # final_pos_error = []
        # for i in range(dof_size):
        #     motor_idx = dof_idx[i]
        #     current_pos = self.low_state.motor_state[motor_idx].q
        #     target_pos = default_pos[i]
        #     error = abs(current_pos - target_pos)
        #     final_pos_error.append(error)

        # max_error = max(final_pos_error)
        # avg_error = sum(final_pos_error) / len(final_pos_error)

        # print(f"[CHECK] Default position check complete:")
        # print(f"  Max joint position error: {max_error:.5f} rad")
        # print(f"  Avg joint position error: {avg_error:.5f} rad")

        # tolerance = 0.05  # radians, adjust if needed
        # if max_error > tolerance:
        #     print(f"[WARN] ❌ Robot did NOT reach default position within {tolerance} rad tolerance.")
        # else:
        #     print(f"[INFO] ✅ Robot reached default position within {tolerance} rad tolerance.")


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
        self.counter += 1
        # Get the current joint position and velocity
        for i in range(len(self.config.action_joint2motor_idx)):
            self.qj[i] = self.low_state.motor_state[self.config.action_joint2motor_idx[i]].q
            self.dqj[i] = self.low_state.motor_state[self.config.action_joint2motor_idx[i]].dq

        # imu_state quaternion: w, x, y, z
        quat = self.low_state.imu_state.quaternion
        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)

        if self.config.imu_type == "torso":
            # h1 and h1_2 imu is on the torso
            # imu data needs to be transformed to the pelvis frame
            waist_yaw = self.low_state.motor_state[self.config.fixed_joint2motor_idx[0]].q
            waist_yaw_omega = self.low_state.motor_state[self.config.fixed_joint2motor_idx[0]].dq
            quat, ang_vel = transform_imu_data(waist_yaw=waist_yaw, waist_yaw_omega=waist_yaw_omega, imu_quat=quat, imu_omega=ang_vel)

        # create observation
        # gravity_orientation = get_gravity_orientation(quat)

        # qj_obs = self.qj.copy()
        # dqj_obs = self.dqj.copy()

        # qj_obs = (qj_obs - self.config.default_angles) * self.config.dof_pos_scale
        # dqj_obs = dqj_obs * self.config.dof_vel_scale
        # ang_vel = ang_vel * self.config.ang_vel_scale

        # period = 0.8
        # count = self.counter * self.config.control_dt
        # phase = count % period / period
        # sin_phase = np.sin(2 * np.pi * phase)
        # cos_phase = np.cos(2 * np.pi * phase)

        # self.cmd[0] = self.remote_controller.ly
        # self.cmd[1] = self.remote_controller.lx * -1
        # self.cmd[2] = self.remote_controller.rx * -1

        # num_actions = self.config.num_actions

        # self.obs[:3] = ang_vel
        # self.obs[3:6] = gravity_orientation
        # self.obs[6:9] = self.cmd * self.config.cmd_scale * self.config.max_cmd
        # self.obs[9 : 9 + num_actions] = qj_obs
        # self.obs[9 + num_actions : 9 + num_actions * 2] = dqj_obs
        # self.obs[9 + num_actions * 2 : 9 + num_actions * 3] = self.action
        # self.obs[9 + num_actions * 3] = sin_phase
        # self.obs[9 + num_actions * 3 + 1] = cos_phase

        # Get the action from the policy network
        # obs_tensor = torch.from_numpy(self.obs).unsqueeze(0)

        
        qj = self.qj.copy()    
        dqj = self.dqj.copy()   
    


        projected_gravity = get_gravity_orientation(quat)
        dof_pos = qj * self.config.dof_pos_scale
        dof_vel = dqj * self.config.dof_vel_scale
        base_ang_vel = ang_vel* self.config.ang_vel_scale
        
        self.ref_motion_phase += self.config.ref_motion_phase
        num_actions = self.config.num_actions

        # DEBUG 
        # print("Shapes of arrays to concatenate:")
        # print(f"self.action shape: {np.array(self.action).shape}")
        # print(f"base_ang_vel shape: {np.array(base_ang_vel).shape}")
        # print(f"dof_pos shape: {np.array(dof_pos).shape}")
        # print(f"dof_vel shape: {np.array(dof_vel).shape}")
        # # print(f"history_obs_buf shape: {np.array(history_obs_buf).shape}")
        # print(f"projected_gravity shape: {np.array(projected_gravity).shape}")
        # print(f"[self.ref_motion_phase] shape: {np.array([self.ref_motion_phase]).shape}")

        history_obs_buf = np.concatenate((self.action_buf, self.ang_vel_buf, self.dof_pos_buf, self.dof_vel_buf, self.proj_g_buf, self.ref_motion_phase_buf), axis=-1, dtype=np.float32)
        
        # print(f"history_obs_buf shape: {np.array(history_obs_buf).shape}")

        try:
            obs_buf = np.concatenate((self.action, base_ang_vel.flatten(), dof_pos, dof_vel, history_obs_buf, projected_gravity, [self.ref_motion_phase]), axis=-1, dtype=np.float32)
        except ValueError as e:
            print(f"Concatenation failed with error: {e}")
            print("Please ensure all arrays have the same number of dimensions (either all 1D or all 2D)")
            raise
        # obs_buf = np.concatenate((self.action, base_ang_vel, dof_pos, dof_vel, history_obs_buf, projected_gravity, [self.ref_motion_phase]), axis=-1, dtype=np.float32)
            


        # history_obs_buf = np.concatenate((self.action_buf, self.ang_vel_buf, self.dof_pos_buf, self.dof_vel_buf, self.proj_g_buf, self.ref_motion_phase_buf), axis=-1, dtype=np.float32)
                        
        # obs_buf = np.concatenate((self.action, base_ang_vel, dof_pos, dof_vel, history_obs_buf, projected_gravity, [self.ref_motion_phase]), axis=-1, dtype=np.float32)
            

        # update history
        self.ang_vel_buf = np.concatenate((base_ang_vel.flatten(), self.ang_vel_buf[:-3]), axis=-1, dtype=np.float32)
        
        self.proj_g_buf = np.concatenate((projected_gravity, self.proj_g_buf[:-3] ), axis=-1, dtype=np.float32)
        self.dof_pos_buf = np.concatenate((dof_pos, self.dof_pos_buf[:-num_actions] ), axis=-1, dtype=np.float32)
        self.dof_vel_buf = np.concatenate((dof_vel, self.dof_vel_buf[:-num_actions] ), axis=-1, dtype=np.float32)
        self.action_buf = np.concatenate((self.action, self.action_buf[:-num_actions] ), axis=-1, dtype=np.float32)
        self.ref_motion_phase_buf = np.concatenate(([self.ref_motion_phase], self.ref_motion_phase_buf[:-1] ), axis=-1, dtype=np.float32)                
        
        
        
        obs_tensor = torch.from_numpy(obs_buf).unsqueeze(0).cpu().numpy()
        self.action = np.squeeze(self.ort_session.run(None, {self.input_name: obs_tensor})[0])


        # self.action = self.policy(obs_tensor).detach().numpy().squeeze()
        
        # transform action to target_dof_pos
        target_dof_pos = self.config.default_angles + self.action * self.config.action_scale

        # Build low cmd
        for i in range(len(self.config.action_joint2motor_idx)):
            motor_idx = self.config.action_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = target_dof_pos[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0
            print(f"[DEBUG]: motor_idx {self.config.action_joint2motor_idx[i]} Action cmd: ", self.low_cmd.motor_cmd[self.config.action_joint2motor_idx[i]].q) 
        for i in range(len(self.config.fixed_joint2motor_idx)):
            motor_idx = self.config.fixed_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = self.config.fixed_target[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.fixed_kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.fixed_kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0
            
            # print("\n")
        # send the command
        # self.send_cmd(self.low_cmd)
        
        time.sleep(self.config.control_dt)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface")
    parser.add_argument("config", type=str, help="config file name in the configs folder", default="g1_29dof_PBHC.yaml")
    args = parser.parse_args()

    # Load config
    # config_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_real/configs/{args.config}"
    config_path = f"deploy_real/configs/{args.config}"
    config = Config(config_path)

    # Initialize DDS communication
    ChannelFactoryInitialize(0, args.net)

    controller = Controller(config)

    # Enter the zero torque state, press the start key to continue executing
    controller.zero_torque_state()

    # Move to the default position
    controller.move_to_default_pos()

    # Enter the default position state, press the A key to continue executing
    controller.default_pos_state()

    while True:
        try:
            controller.run()
            # Press the select key to exit
            if controller.remote_controller.button[KeyMap.select] == 1:
                break
        except KeyboardInterrupt:
            break
    # Enter the damping state
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print("Exit")
