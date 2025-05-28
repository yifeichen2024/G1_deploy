# deploy_real.py
from legged_gym import LEGGED_GYM_ROOT_DIR
from typing import Union, Tuple, Dict, Any
import numpy as np
import time
import torch

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber # Removed duplicate ChannelFactoryInitialize
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

        self.policy = torch.jit.load(config.policy_path)
        
        if config.num_actions != 23: # Example check, adjust if your DOF count is different but consistent
            print(
                f"Warning: config.num_actions ({config.num_actions}) is not 23. "
                f"Ensure this matches the policy's expected DOF count for dof_pos/vel/actions."
            )

        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32) 
        self.target_dof_pos = config.default_angles.copy()
        
        self.cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32) # Use float for consistency
        self.counter = 0

        # Define dimensions for actor observation based on logs
        self.actor_base_ang_vel_dim = 3
        self.actor_projected_gravity_dim = 3
        self.actor_dof_pos_dim = config.num_actions 
        self.actor_dof_vel_dim = config.num_actions 
        self.actor_actions_dim = config.num_actions
        self.actor_ref_motion_phase_dim = 1

        self.current_actor_frame_obs_dim = (
            self.actor_base_ang_vel_dim +
            self.actor_projected_gravity_dim +
            self.actor_dof_pos_dim +
            self.actor_dof_vel_dim +
            self.actor_actions_dim +
            self.actor_ref_motion_phase_dim
        ) 
        # For num_actions = 23, this should be 3+3+23+23+23+1 = 76

        # Expected history length from logs (304 / 76 = 4)
        self.actor_history_length = 4 
        self.actor_history_dim = self.actor_history_length * self.current_actor_frame_obs_dim 
        # For 76 * 4 = 304

        self.total_actor_obs_dim = self.current_actor_frame_obs_dim + self.actor_history_dim
        # For 76 + 304 = 380
        
        # self.obs will hold the final 380-dimensional vector
        # Ensure config.num_obs (if used from file) matches self.total_actor_obs_dim
        if hasattr(config, 'num_obs') and config.num_obs != self.total_actor_obs_dim:
            print(f"Warning: config.num_obs ({config.num_obs}) does not match calculated total_actor_obs_dim ({self.total_actor_obs_dim}). Using calculated value.")
        self.obs = np.zeros(self.total_actor_obs_dim, dtype=np.float32)
        
        self.actor_obs_history_buffer = np.zeros(
            (self.actor_history_length, self.current_actor_frame_obs_dim), dtype=np.float32
        )
        self.is_first_frame_actor_history = True

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
            raise ValueError("Invalid msg_type")

        self.wait_for_low_state()

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
        print("Waiting for connection to robot...")
        while self.low_state.tick == 0: # Ensure low_state is initialized
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal (Remote Controller Start Button)...")
        while self.remote_controller.button.get(KeyMap.start, 0) != 1: # Use .get for safety
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def move_to_default_pos(self):
        print("Moving to default pos.")
        total_time = 2.0 # Ensure float
        num_step = int(total_time / self.config.control_dt)
        
        dof_idx = self.config.leg_joint2motor_idx + self.config.arm_waist_joint2motor_idx
        kps = self.config.kps + self.config.arm_waist_kps
        kds = self.config.kds + self.config.arm_waist_kds
        # Ensure default_angles and arm_waist_target are numpy arrays for concatenation
        default_pos_legs = np.array(self.config.default_angles, dtype=np.float32)
        default_pos_arm_waist = np.array(self.config.arm_waist_target, dtype=np.float32)
        default_pos_full = np.concatenate((default_pos_legs, default_pos_arm_waist), axis=0)
        dof_size = len(dof_idx)
        
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        # Ensure low_state has been received before accessing motor_state
        if self.low_state.tick == 0:
            print("Warning: Low state not received before move_to_default_pos. Waiting briefly.")
            self.wait_for_low_state() # Ensure we have a valid low_state

        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q
        
        for i in range(num_step):
            alpha = float(i) / num_step # Ensure float division
            for j in range(dof_size):
                motor_idx_in_cmd = dof_idx[j] # motor_idx is the actual index for low_cmd.motor_cmd
                target_joint_pos = default_pos_full[j]
                self.low_cmd.motor_cmd[motor_idx_in_cmd].q = init_dof_pos[j] * (1 - alpha) + target_joint_pos * alpha
                self.low_cmd.motor_cmd[motor_idx_in_cmd].qd = 0.0
                self.low_cmd.motor_cmd[motor_idx_in_cmd].kp = float(kps[j])
                self.low_cmd.motor_cmd[motor_idx_in_cmd].kd = float(kds[j])
                self.low_cmd.motor_cmd[motor_idx_in_cmd].tau = 0.0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def default_pos_state(self):
        print("Enter default pos state.")
        print("Waiting for the Button A signal (Remote Controller A Button)...")
        while self.remote_controller.button.get(KeyMap.A, 0) != 1:
            for i in range(len(self.config.leg_joint2motor_idx)):
                motor_idx = self.config.leg_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = float(self.config.default_angles[i])
                self.low_cmd.motor_cmd[motor_idx].qd = 0.0
                self.low_cmd.motor_cmd[motor_idx].kp = float(self.config.kps[i])
                self.low_cmd.motor_cmd[motor_idx].kd = float(self.config.kds[i])
                self.low_cmd.motor_cmd[motor_idx].tau = 0.0
            for i in range(len(self.config.arm_waist_joint2motor_idx)):
                motor_idx = self.config.arm_waist_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = float(self.config.arm_waist_target[i])
                self.low_cmd.motor_cmd[motor_idx].qd = 0.0
                self.low_cmd.motor_cmd[motor_idx].kp = float(self.config.arm_waist_kps[i])
                self.low_cmd.motor_cmd[motor_idx].kd = float(self.config.arm_waist_kds[i])
                self.low_cmd.motor_cmd[motor_idx].tau = 0.0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def _prepare_actor_observation(
        self,
        current_qj: np.ndarray,
        current_dqj: np.ndarray,
        raw_imu_quat: np.ndarray, # Assuming [w,x,y,z] or [x,y,z,w] based on helper needs
        raw_imu_ang_vel: np.ndarray, # Assuming 1D array of 3 elements
        previous_action: np.ndarray,
        timestep_count: int,
        waist_q_for_imu_transform: Union[float, None] = None,
        waist_dq_for_imu_transform: Union[float, None] = None
    ) -> np.ndarray:
        """
        Prepares the 380-dimensional observation vector for the actor policy.
        """
        
        processed_quat = np.array(raw_imu_quat, dtype=np.float32) # Ensure copy and type
        processed_ang_vel = np.array(raw_imu_ang_vel, dtype=np.float32)

        if self.config.imu_type == "torso":
            if waist_q_for_imu_transform is None or waist_dq_for_imu_transform is None:
                raise ValueError("Waist q/dq needed for torso IMU transformation but not provided.")
            # Ensure transform_imu_data input/output formats are handled correctly
            # For example, if imu_quat needs to be [w,x,y,z]
            # And gyroscope is [gx, gy, gz]
            transformed_quat, transformed_ang_vel = transform_imu_data(
                waist_yaw=waist_q_for_imu_transform,
                waist_yaw_omega=waist_dq_for_imu_transform,
                imu_quat=processed_quat, # Pass as is, ensure helper handles order
                imu_omega=processed_ang_vel # Pass as is
            )
            processed_quat = np.array(transformed_quat, dtype=np.float32)
            processed_ang_vel = np.array(transformed_ang_vel, dtype=np.float32).flatten()
        
        # 1. Base Angular Velocity (scaled)
        obs_base_ang_vel = processed_ang_vel * self.config.ang_vel_scale
        
        # 2. Projected Gravity
        # Ensure processed_quat is in the order expected by get_gravity_orientation (e.g. [w,x,y,z])
        obs_projected_gravity = get_gravity_orientation(processed_quat)

        # 3. DOF Positions (scaled)
        # Ensure current_qj and default_angles are aligned and have correct num_actions length
        obs_dof_pos = (current_qj - np.array(self.config.default_angles, dtype=np.float32)) * self.config.dof_pos_scale

        # 4. DOF Velocities (scaled)
        obs_dof_vel = current_dqj * self.config.dof_vel_scale

        # 5. Actions (previous actions from policy)
        obs_actions = previous_action.copy() 

        # 6. Reference Motion Phase (1 dim)
        period = self.config.get('phase_period', 0.8) # Get from config or use default
        current_time_in_period = (timestep_count * self.config.control_dt) % period
        phase = current_time_in_period / period
        obs_ref_motion_phase = np.array([np.sin(2 * np.pi * phase)], dtype=np.float32)

        # Concatenate current frame observations
        current_actor_frame_obs_list = [
            obs_base_ang_vel.flatten(), # Ensure 1D
            obs_projected_gravity.flatten(), # Ensure 1D
            obs_dof_pos.flatten(),
            obs_dof_vel.flatten(),
            obs_actions.flatten(),
            obs_ref_motion_phase.flatten()
        ]
        current_actor_frame_obs = np.concatenate(current_actor_frame_obs_list)

        if current_actor_frame_obs.shape[0] != self.current_actor_frame_obs_dim:
             raise ValueError(
                f"Mismatch in current_actor_frame_obs dimension! "
                f"Expected {self.current_actor_frame_obs_dim}, Got {current_actor_frame_obs.shape[0]}"
            )

        # Update history buffer
        if self.is_first_frame_actor_history:
            for i in range(self.actor_history_length):
                self.actor_obs_history_buffer[i, :] = current_actor_frame_obs
            self.is_first_frame_actor_history = False
        else:
            self.actor_obs_history_buffer = np.roll(self.actor_obs_history_buffer, shift=-1, axis=0)
            self.actor_obs_history_buffer[-1, :] = current_actor_frame_obs
        
        history_actor_flat = self.actor_obs_history_buffer.flatten()

        # Assemble final observation for the policy
        final_observation = np.concatenate((current_actor_frame_obs, history_actor_flat))
        
        if final_observation.shape[0] != self.total_actor_obs_dim:
            raise ValueError(
                f"Mismatch in final observation dimension! "
                f"Expected {self.total_actor_obs_dim}, Got {final_observation.shape[0]}"
            )
        return final_observation

    def run(self):
        self.counter += 1
        
        # 1. Get raw sensor data
        for i in range(len(self.config.leg_joint2motor_idx)): # Assuming this covers all policy-controlled DOFs
            self.qj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].q
            self.dqj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].dq

        # IMU data: ensure correct format [w,x,y,z] or [x,y,z,w] as per your helpers
        # Unitree typically provides quaternion as [w, x, y, z]
        # Gyroscope is typically [x, y, z]
        raw_imu_quat = np.array(self.low_state.imu_state.quaternion, dtype=np.float32) 
        raw_imu_ang_vel = np.array(self.low_state.imu_state.gyroscope, dtype=np.float32)

        waist_q_param = None
        waist_dq_param = None
        if self.config.imu_type == "torso":
            # Ensure these indices are correct and arm_waist_joint2motor_idx is not empty
            if self.config.arm_waist_joint2motor_idx:
                waist_q_param = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].q
                waist_dq_param = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].dq
            else:
                print("Warning: imu_type is 'torso' but arm_waist_joint2motor_idx is not configured for waist joint.")


        # 2. Prepare observation vector using the new function
        self.obs[:] = self._prepare_actor_observation(
            current_qj=self.qj,
            current_dqj=self.dqj,
            raw_imu_quat=raw_imu_quat,
            raw_imu_ang_vel=raw_imu_ang_vel,
            previous_action=self.action, # self.action stores the action from the *previous* step
            timestep_count=self.counter,
            waist_q_for_imu_transform=waist_q_param,
            waist_dq_for_imu_transform=waist_dq_param
        )
        
        # 3. Get user commands (not part of the 380-dim policy input as per logs)
        self.cmd[0] = self.remote_controller.ly
        self.cmd[1] = self.remote_controller.lx * -1.0 
        self.cmd[2] = self.remote_controller.rx * -1.0

        # 4. Get action from policy
        obs_tensor = torch.from_numpy(self.obs).unsqueeze(0) # .to(device) if using GPU
        
        # Update self.action with the *current* policy output. 
        # This will be used as 'previous_action' in the next call to _prepare_actor_observation.
        self.action = self.policy(obs_tensor).detach().cpu().numpy().squeeze()
        
        # 5. Transform action to target_dof_pos for commanding the robot
        # Ensure default_angles and action_scale are numpy arrays of correct size (num_actions)
        current_default_angles = np.array(self.config.default_angles, dtype=np.float32)
        current_action_scale = np.array(self.config.action_scale, dtype=np.float32)
        if current_default_angles.shape[0] != self.config.num_actions or \
           current_action_scale.shape[0] != self.config.num_actions:
            raise ValueError("Mismatch in dimensions of default_angles or action_scale with num_actions")

        target_dof_pos_legs = current_default_angles + self.action * current_action_scale

        # 6. Build and send low_cmd
        for i in range(len(self.config.leg_joint2motor_idx)):
            motor_idx = self.config.leg_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = float(target_dof_pos_legs[i])
            self.low_cmd.motor_cmd[motor_idx].qd = 0.0 
            self.low_cmd.motor_cmd[motor_idx].kp = float(self.config.kps[i])
            self.low_cmd.motor_cmd[motor_idx].kd = float(self.config.kds[i])
            self.low_cmd.motor_cmd[motor_idx].tau = 0.0

        # Arm/waist control can remain as is, or be driven by policy if included in num_actions
        for i in range(len(self.config.arm_waist_joint2motor_idx)):
            motor_idx = self.config.arm_waist_joint2motor_idx[i]
            # If arms/waist are part of the policy, their target should come from self.action too.
            # Otherwise, they use fixed targets:
            self.low_cmd.motor_cmd[motor_idx].q = float(self.config.arm_waist_target[i]) 
            self.low_cmd.motor_cmd[motor_idx].qd = 0.0
            self.low_cmd.motor_cmd[motor_idx].kp = float(self.config.arm_waist_kps[i])
            self.low_cmd.motor_cmd[motor_idx].kd = float(self.config.arm_waist_kds[i])
            self.low_cmd.motor_cmd[motor_idx].tau = 0.0

        self.send_cmd(self.low_cmd)
        time.sleep(self.config.control_dt)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface")
    # Changed default for config to allow it to be optional if you prefer a hardcoded default
    parser.add_argument("--config", type=str, help="config file name in the configs folder", default="g1.yaml") 
    args = parser.parse_args()

    config_file_name = args.config
    if not config_file_name.endswith(".yaml"): # Basic check
        config_file_name += ".yaml"

    #config_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_real/configs/{config_file_name}"
    config_path = "/home/zy/unitree_rl_gym/deploy/deploy_real/configs/g1.yaml"
    print(f"Loading configuration from: {config_path}")
    try:
        config = Config(config_path)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        print(f"Please ensure '{config_file_name}' exists in 'legged_gym/deploy/deploy_real/configs/'")
        exit(1)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        exit(1)


    ChannelFactoryInitialize(0, args.net)
    print("Channel Factory Initialized.")

    controller = Controller(config)

    controller.zero_torque_state()
    controller.move_to_default_pos()
    controller.default_pos_state()

    print("Starting main control loop...")
    try:
        while True:
            controller.run()
            if controller.remote_controller.button.get(KeyMap.select, 0) == 1: # Use .get for safety
                print("Select button pressed. Exiting loop.")
                break
    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Exiting.")
    except Exception as e:
        print(f"An error occurred in the main loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Entering damping state...")
        create_damping_cmd(controller.low_cmd) # Ensure controller.low_cmd is accessible and valid
        controller.send_cmd(controller.low_cmd)
        print("Exited.")