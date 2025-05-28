import numpy as np
from typing import Union # For type hinting if used more broadly

# Assuming these helper functions are accessible in the same scope or imported
# from common.rotation_helper import get_gravity_orientation, transform_imu_data


def _prepare_actor_observation(
    self,
    current_qj: np.ndarray,
    current_dqj: np.ndarray,
    raw_imu_quat: np.ndarray, # Expected order [w,x,y,z] or [x,y,z,w] based on helper needs
    raw_imu_ang_vel: np.ndarray, # Expected 1D array of 3 elements [gx, gy, gz]
    previous_action: np.ndarray,
    timestep_count: int,
    waist_q_for_imu_transform: Union[float, None] = None,
    waist_dq_for_imu_transform: Union[float, None] = None
) -> np.ndarray:
    """
    Prepares the observation vector for the actor policy, matching the target shape.
    This method assumes it's part of the Controller class and has access to:
    - self.config (for scaling factors, imu_type, default_angles, etc.)
    - self.current_actor_frame_obs_dim (e.g., 76)
    - self.actor_history_length (e.g., 4)
    - self.actor_obs_history_buffer (numpy array for storing history)
    - self.is_first_frame_actor_history (boolean flag)
    - self.total_actor_obs_dim (e.g., 380)
    """
    
    # Ensure inputs are numpy arrays of the correct type (can be done earlier too)
    processed_quat = np.array(raw_imu_quat, dtype=np.float32)
    processed_ang_vel = np.array(raw_imu_ang_vel, dtype=np.float32).flatten() # Ensure 1D

    if self.config.imu_type == "torso":
        if waist_q_for_imu_transform is None or waist_dq_for_imu_transform is None:
            # Consider raising an error or logging if data is essential and missing
            print("Warning: Waist q/dq needed for torso IMU transformation but not provided to _prepare_actor_observation.")
            # Fallback or error based on how critical this is
        else:
            # Placeholder for actual transform_imu_data function call
            # from common.rotation_helper import transform_imu_data
            transformed_quat, transformed_ang_vel = transform_imu_data(
                waist_yaw=waist_q_for_imu_transform,
                waist_yaw_omega=waist_dq_for_imu_transform,
                imu_quat=processed_quat, # Pass as is, ensure helper handles order
                imu_omega=processed_ang_vel # Pass as is
            )
            processed_quat = np.array(transformed_quat, dtype=np.float32)
            processed_ang_vel = np.array(transformed_ang_vel, dtype=np.float32).flatten()
    
    # 1. Base Angular Velocity (scaled)
    # Ensure self.config.ang_vel_scale is defined
    obs_base_ang_vel = processed_ang_vel * self.config.ang_vel_scale
    
    # 2. Projected Gravity
    # Placeholder for actual get_gravity_orientation function call
    # from common.rotation_helper import get_gravity_orientation
    # Ensure processed_quat is in the order expected by get_gravity_orientation (e.g. [w,x,y,z])
    obs_projected_gravity = get_gravity_orientation(processed_quat)

    # 3. DOF Positions (scaled)
    # Ensure self.config.default_angles and self.config.dof_pos_scale are defined
    # And current_qj matches the length of self.config.default_angles (num_actions)
    default_angles_arr = np.array(self.config.default_angles, dtype=np.float32)
    obs_dof_pos = (current_qj - default_angles_arr) * self.config.dof_pos_scale

    # 4. DOF Velocities (scaled)
    # Ensure self.config.dof_vel_scale is defined
    obs_dof_vel = current_dqj * self.config.dof_vel_scale

    # 5. Actions (previous actions from policy)
    # previous_action is passed in, should be of size num_actions
    obs_actions = previous_action.copy() 

    # 6. Reference Motion Phase (1 dim)
    # Ensure self.config.control_dt is defined.
    # 'phase_period' can be a field in self.config or a default value.
    period = getattr(self.config, 'phase_period', 0.8) 
    current_time_in_period = (timestep_count * self.config.control_dt) % period
    phase = current_time_in_period / period
    obs_ref_motion_phase = np.array([np.sin(2 * np.pi * phase)], dtype=np.float32)

    # Concatenate current frame observations into a single 1D array
    current_actor_frame_obs_list = [
        obs_base_ang_vel.flatten(), 
        obs_projected_gravity.flatten(),
        obs_dof_pos.flatten(),
        obs_dof_vel.flatten(),
        obs_actions.flatten(),
        obs_ref_motion_phase.flatten()
    ]
    current_actor_frame_obs = np.concatenate(current_actor_frame_obs_list)

    # Verify dimension of the current frame observation
    if current_actor_frame_obs.shape[0] != self.current_actor_frame_obs_dim:
            raise ValueError(
            f"Dimension mismatch for current_actor_frame_obs. "
            f"Expected {self.current_actor_frame_obs_dim}, Got {current_actor_frame_obs.shape[0]}"
        )

    # Update history buffer
    # self.actor_obs_history_buffer should be initialized in __init__
    # self.is_first_frame_actor_history should be initialized in __init__
    if self.is_first_frame_actor_history:
        for i in range(self.actor_history_length): # actor_history_length e.g., 4
            self.actor_obs_history_buffer[i, :] = current_actor_frame_obs
        self.is_first_frame_actor_history = False
    else:
        # Shift buffer: oldest observation moves towards index 0, newest at the end
        self.actor_obs_history_buffer = np.roll(self.actor_obs_history_buffer, shift=-1, axis=0)
        self.actor_obs_history_buffer[-1, :] = current_actor_frame_obs
    
    history_actor_flat = self.actor_obs_history_buffer.flatten()

    # Assemble final observation for the policy by concatenating current obs and its history
    final_observation = np.concatenate((current_actor_frame_obs, history_actor_flat))
    
    # Verify dimension of the final observation
    if final_observation.shape[0] != self.total_actor_obs_dim: # total_actor_obs_dim e.g., 380
        raise ValueError(
            f"Dimension mismatch for final_observation. "
            f"Expected {self.total_actor_obs_dim}, Got {final_observation.shape[0]}"
        )
        
    return final_observation

# ... (other parts of the Controller class)