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
        self.ort_session = onnxruntime.InferenceSession(config.policy_paths[1])
        self.input_name = self.ort_session.get_inputs()[0].name

        # Initializing process variables
        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        # self.target_dof_pos = config.default_angles.copy()
        # self.obs = np.zeros(config.num_obs, dtype=np.float32)
        # self.cmd = np.array([0.0, 0, 0])
        self.counter = 0
        # motion lens 
        
        self.motion_len = config.motion_lens[1]
        self.control_dt = config.control_dt


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

        print("[DEBUG]: Waiting for the lower state info...")
        # wait for the subscriber to receive data
        self.wait_for_low_state()
        print("[DEBUG]: Lower state get!")

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
        # move time 5 s
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
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

        for i in range(dof_size):
            motor_idx = dof_idx[i]
            current_q = self.low_state.motor_state[motor_idx].q
            target_q = default_pos[i]
            err = current_q - target_q
            print(f"[DEBUG] Motor {motor_idx:02d}: target={target_q:+.3f}, actual={current_q:+.3f}, err={err:+.3f}")
        if abs(err) < 0.01:
            print("Successfully moved to default pos!")
        else:
            print("Failed to move to default pos!")

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
        
        # self.ref_motion_phase += self.config.ref_motion_phase
        motion_time = self.counter * self.control_dt
        self.ref_motion_phase = motion_time / self.motion_len
        print(f"[DEBUG] ref_motion phase: {self.ref_motion_phase}, timesteps: {self.counter}" )
        # self.ref_motion_phase = self.ref_motion_phase % 1 # Prevention
        num_actions = self.config.num_actions
        
        if self.ref_motion_phase >= 1.0:
            print("[INFO] Motion finished.")
            self.move_to_default_pos()
            return True

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
        
        
        # Get ready for the observation input 
        obs_tensor = torch.from_numpy(obs_buf).unsqueeze(0).cpu().numpy()
        # Get the action from the policy network
        self.action = np.squeeze(self.ort_session.run(None, {self.input_name: obs_tensor})[0])

        # Warning for the action threshold
        if np.any(np.abs(self.action) > self.config.action_clip_warn_threshold):
            print(f"[WARNING] Action exceeds warning threshold: {self.action}")

        # action clipping
        # self.action = np.clip(self.action, -self.config.action_clip, self.config.action_clip)
        
        # self.action = self.policy(obs_tensor).detach().numpy().squeeze()
        
        # transform action to target_dof_pos
        target_dof_pos = self.config.default_angles + self.action * self.config.action_scale

        # Target_position clip 
        target_dof_pos = np.clip(target_dof_pos, self.config.dof_pos_lower_limit, self.config.dof_pos_upper_limit)

        # Build low cmd
        for i in range(len(self.config.action_joint2motor_idx)):
            motor_idx = self.config.action_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = target_dof_pos[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0
            print(f"[DEBUG]: Motor {motor_idx} position: {target_dof_pos[i]}")
        for i in range(len(self.config.fixed_joint2motor_idx)):
            motor_idx = self.config.fixed_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = self.config.fixed_target[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.fixed_kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.fixed_kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        # 实时速度超限报警
        dqj_abs = np.abs(self.dqj)
        if np.any(dqj_abs > self.config.dof_vel_limit):
            print(f"[ERROR] Joint velocity exceeds limit! DQJ: {self.dqj}, Limit: {self.config.dof_vel_limit}")

        for i in range(len(self.config.action_joint2motor_idx)):
            motor_idx = self.config.action_joint2motor_idx[i]
            measured_tau = self.low_state.motor_state[motor_idx].tau_est  
            if np.abs(measured_tau) > self.config.dof_effort_limit[i]:
                print(f"[ERROR] Torque exceeds limit! Motor {motor_idx}, Tau: {measured_tau:.3f}, Limit: {self.config.dof_effort_limit[i]}")

                # # send damping mode for protection
                # create_damping_cmd(self.low_cmd)
                # self.send_cmd(self.low_cmd)
                # time.sleep(self.config.control_dt)

                # # Directly return
                # return True
            
        # send the command
        self.send_cmd(self.low_cmd)

        time.sleep(self.config.control_dt)


if __name__ == "__main__":
    import argparse
    import sys
    import datetime
    # log info
    log_filename = f"deploy_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "w")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()

        def flush(self):
            pass

    sys.stdout = Logger(log_filename)
    print(f"[INFO] Logging to {log_filename}")


    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface")
    parser.add_argument("config", type=str, help="config file name in the configs folder", default="g1_29dof_PBHC.yaml")
    args = parser.parse_args()

    # Load config
    config_path = f"deploy_real/configs/{args.config}"
    config = Config(config_path)

    # Initialize DDS communication
    # ChannelFactoryInitialize(0, 'enp2s0') # DEBUG 
    ChannelFactoryInitialize(0, args.net)

    print("[DEBUG]: Start to initialize the controller!")
    controller = Controller(config)
    print("[DEBUG]: Controller initialized.")
    # Enter the zero torque state, press the start key to continue executing
    controller.zero_torque_state()

    # Move to the default position
    controller.move_to_default_pos()

    # Enter the default position state, press the A key to continue executing
    controller.default_pos_state()
    print("Ready to run the movements")
    while True:
        try:
            
            if controller.run(): 
                break
            # Press the select key to exit
            if controller.remote_controller.button[KeyMap.select] == 1:
                break

            # 按 X 重置动作到默认位置
            if controller.remote_controller.button[KeyMap.X] == 1:
                print("[INFO] Resetting to default position")
                controller.move_to_default_pos()
                
        except KeyboardInterrupt:
            break
    # Enter the damping state
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print("Exit")
