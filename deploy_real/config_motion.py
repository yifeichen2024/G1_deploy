from legged_gym import LEGGED_GYM_ROOT_DIR
import numpy as np
import yaml


class Config:
    def __init__(self, file_path) -> None:
        with open(file_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

            self.control_dt = config["control_dt"]

            self.msg_type = config["msg_type"]
            self.imu_type = config["imu_type"]

            self.weak_motor = []
            if "weak_motor" in config:
                self.weak_motor = config["weak_motor"]

            self.lowcmd_topic = config["lowcmd_topic"]
            self.lowstate_topic = config["lowstate_topic"]

            self.policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

            self.action_joint2motor_idx = config["action_joint2motor_idx"]
            self.kps = config["kps"]
            self.kds = config["kds"]
            self.default_angles = np.array(config["default_angles"], dtype=np.float32)

            self.fixed_joint2motor_idx = config["fixed_joint2motor_idx"]
            self.fixed_kps = config["fixed_kps"]
            self.fixed_kds = config["fixed_kds"]
            self.fixed_target = np.array(config["fixed_target"], dtype=np.float32)

            self.ang_vel_scale = config["ang_vel_scale"]
            self.dof_pos_scale = config["dof_pos_scale"]
            self.dof_vel_scale = config["dof_vel_scale"]
            self.action_scale = config["action_scale"]
            self.cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
            self.max_cmd = np.array(config["max_cmd"], dtype=np.float32)

            self.num_actions = config["num_actions"]
            self.num_obs = config["num_obs"]
            

            self.history_length = config["history_length"]
            self.ref_motion_phase = config["ref_motion_phase"]

            # safty check part
            self.action_clip = config["action_clip"]
            self.action_scale = config["action_scale"]
            self.debug = config["debug"]
            self.action_clip_warn_threshold = config["action_clip_warn_threshold"]

            self.dof_pos_lower_limit = config["dof_pos_lower_limit"]
            self.dof_pos_upper_limit = config["dof_pos_upper_limit"]
            self.dof_vel_limit = config["dof_vel_limit"]
            self.dof_effort_limit = config["dof_effort_limit"]

            