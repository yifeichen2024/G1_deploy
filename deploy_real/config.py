from legged_gym import LEGGED_GYM_ROOT_DIR
import numpy as np
import yaml


class Config:
    def __init__(self, file_path) -> None:
        with open(file_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

            self.control_dt = config["control_dt"]
            if "motion_lens" in config:
                self.motion_lens = config["motion_lens"]
            if "policy_names" in config:
                self.policy_names = config["policy_names"]

            if "motion_path" in config:
                self.motion_path = config["motion_path"]    
            
            if "msg_type" in config:
                self.msg_type = config["msg_type"]
            else:
                self.msg_type = "hg"
            # self.msg_type = config["msg_type"]
            
            if "imu_type" in config:
                self.imu_type = config["imu_type"]
            # self.imu_type = config["imu_type"]

            self.weak_motor = []
            if "weak_motor" in config:
                self.weak_motor = config["weak_motor"]

            self.lowcmd_topic = config["lowcmd_topic"]
            self.lowstate_topic = config["lowstate_topic"]
            # if "cmd_range" in config:
            #     self.cmd_range = config["cmd_range"]

            if "cmd_range" in config:
                self.cmd_range = config["cmd_range"]
                # self.range_velx = np.array(self.cmd_range['lin_vel_x'], dtype=np.float32)
                # self.range_vely = np.array(self.cmd_range['lin_vel_y'], dtype=np.float32)
                # self.range_velz = np.array(self.cmd_range['ang_vel_z'], dtype=np.float32)
            # --------- 1. 支持多 / 单 policy 两种写法 ----------
            if "policy_paths" in config:                  # 多策略
                raw_paths = config["policy_paths"]
                # 把 {LEGGED_GYM_ROOT_DIR} 展开，保持老逻辑
                self.policy_paths = [
                    p.replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
                    for p in raw_paths
                ]
                assert isinstance(raw_paths, list) and len(raw_paths) > 0, \
                    "policy_paths 必须是非空列表"
            elif "policy_path" in config:                                         # 兼容旧写法
                raw_paths = [config["policy_path"]]
                self.policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)


            self.action_joint2motor_idx = config["action_joint2motor_idx"]
            self.kps = config["kps"]
            self.kds = config["kds"]
            
            self.default_angles = np.array(config["default_angles"], dtype=np.float32)
            
            if "fixed_joint2motor_idx" in config:
                self.fixed_joint2motor_idx = config["fixed_joint2motor_idx"]
            else:
                self.fixed_joint2motor_idx = []
            # self.fixed_joint2motor_idx = config["fixed_joint2motor_idx"]
            if "fixed_kps" in config:
                self.fixed_kps = config["fixed_kps"]
            else:
                self.fixed_kps = []
            
            # self.fixed_kps = config["fixed_kps"]

            if "fixed_kds" in config:
                self.fixed_kds = config["fixed_kds"]
            else:
                self.fixed_kds = []
            # self.fixed_kds = config["fixed_kds"]
            
            self.fixed_target = np.array(config["fixed_target"], dtype=np.float32)
            # if "fxied_target" in config:
            #     self.fixed_target = np.array(config["fixed_target"], dtype=np.float32)
            # else:
            #     self.fixed_target = np.zeros(self.fixed_target, dtype=np.float32)
            
            self.ang_vel_scale = config["ang_vel_scale"]
            self.dof_pos_scale = config["dof_pos_scale"]
            self.dof_vel_scale = config["dof_vel_scale"]
            self.action_scale = config["action_scale"]
            self.cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
            self.max_cmd = np.array(config["max_cmd"], dtype=np.float32)

            self.num_actions = config["num_actions"]
            self.num_obs = config["num_obs"]
            
            if "history_length" in config:
                self.history_length = config["history_length"]
            if "ref_motion_phase" in config:
                self.ref_motion_phase = config["ref_motion_phase"]
            # self.history_length = config["history_length"]
            # self.ref_motion_phase = config["ref_motion_phase"]

            # safty check part
            self.action_clip = config["action_clip"]
            self.action_scale = config["action_scale"]
            # self.debut = config["debug"]
            
            self.action_clip_warn_threshold = config["action_clip_warn_threshold"]

            self.dof_pos_lower_limit = config["dof_pos_lower_limit"]
            self.dof_pos_upper_limit = config["dof_pos_upper_limit"]
            self.dof_vel_limit = config["dof_vel_limit"]
            self.dof_effort_limit = config["dof_effort_limit"]

            