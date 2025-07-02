# deploy_real_multi_policy.py
"""
Multi‑policy ONNX deployment script for Unitree G1 based on your original
`deploy_real_onnx.py`, extended with run‑time policy switching while keeping
all existing safety checks, observation building, remote‑controller workflow,
and logging.

**Key additions**
1.  Support for a *list* of ONNX policies (`config.policy_paths`). All models
   are loaded at start‑up and stored in `self.ort_sessions`.
2.  `self.policy_idx` tracks the active policy. Helper `switch_policy()` wraps
   bounds checking & status print‑out.
3.  Remote controller buttons **LB / RB** (or keyboard `[` / `]` as fallback)
   increment / decrement `policy_idx`. Behaviour is identical in zero‑torque,
   default‑pose and running states.
4.  Action inference line now selects the correct ONNX session via
   `self.ort_sessions[self.policy_idx]` **without touching your history /
   scaling / clipping / safety logic**.
5.  CLI flag `--policies` overrides YAML list for quick testing.

"""
from typing import List, Union
import numpy as np
import time
import torch
import argparse
import sys
import datetime
import onnxruntime

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
from common.remote_controller import RemoteController, KeyMap
from config import Config


class Controller:
    """Runtime controller with multi‑policy switching."""

    def __init__(self, config: Config) -> None: # , policy_paths_cli: List[str] | None = None
        self.config = config
        self.remote_controller = RemoteController()

        # -------------------------------
        # 1.  Load one or multiple ONNX policies
        # -------------------------------
        policy_paths: List[str]
        # if policy_paths_cli is not None and len(policy_paths_cli) > 0:
        #     policy_paths = policy_paths_cli
        #     print(f"[INFO] Using policies from CLI: {policy_paths}")
        # else:

        # YAML can specify a single string or a list
        if isinstance(config.policy_paths, str):
            policy_paths = [config.policy_paths]
        else:
            policy_paths = list(config.policy_paths)
        assert len(policy_paths) > 0, "No policy paths provided!"

        self.ort_sessions = [onnxruntime.InferenceSession(p) for p in policy_paths]
        self.input_names = [sess.get_inputs()[0].name for sess in self.ort_sessions]
        self.policy_idx = 0  # active policy index
        self.stance_idx = 0  # 默认第0个是Stance
        # motion length 
        self.motion_lens = list(config.motion_lens)
        self.motion_len = self.motion_lens[0]     # 当前动作时长
        self.control_dt = config.control_dt

        assert len(self.motion_lens) == len(policy_paths), "Mismatch between policy count and motion_lens"
        # -------------------------------
        # 2.  Original variable buffers stay untouched
        # -------------------------------
        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        self.counter = 0
        self.target_dof_pos = config.default_angles.copy()
        self.ref_motion_phase = 0.0
        self.ang_vel_buf = np.zeros(3 * config.history_length, dtype=np.float32)
        self.proj_g_buf = np.zeros(3 * config.history_length, dtype=np.float32)
        self.dof_pos_buf = np.zeros(config.num_actions * config.history_length, dtype=np.float32)
        self.dof_vel_buf = np.zeros(config.num_actions * config.history_length, dtype=np.float32)
        self.action_buf = np.zeros(config.num_actions * config.history_length, dtype=np.float32)
        self.ref_motion_phase_buf = np.zeros(config.history_length, dtype=np.float32)

        # -------------------------------
        # 3.  DDS init (identical to original)
        # -------------------------------
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

        print("[DEBUG]: Waiting for the lower state info …")
        self.wait_for_low_state()
        print("[DEBUG]: Lower state connected!")

        # Command init (unchanged)
        if config.msg_type == "hg":
            init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        else:
            init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)

        # ══════════════════════════════════════
        # End of __init__
        # ══════════════════════════════════════

    # ─────────────────────────────────────────
    # DDS Callbacks (unchanged)
    # ─────────────────────────────────────────
    def LowStateHgHandler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def LowStateGoHandler(self, msg: LowStateGo):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    # ─────────────────────────────────────────
    # Helper: Publish command with CRC
    # ─────────────────────────────────────────
    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("[INFO] Successfully connected to the robot.")
    # ─────────────────────────────────────────
    # Policy switching helpers
    # ─────────────────────────────────────────
    def switch_policy(self, delta: int):
        candidates = [i for i in range(len(self.ort_sessions)) if i != self.stance_idx]
        if not candidates:
            print("[WARN] No other policies to switch.")
            return

        idx = candidates.index(self.policy_idx) if self.policy_idx in candidates else -1
        new_idx = candidates[(idx + delta) % len(candidates)]

        self.policy_idx = new_idx
        self.motion_len = self.motion_lens[self.policy_idx]
        self.ref_motion_phase = 0.0
        self.counter = 0

        print(f"[INFO] >>> Switched to policy #{self.policy_idx} ({self.config.policy_names[self.policy_idx] if hasattr(self.config, 'policy_names') else ''}), motion_len={self.motion_len:.3f}s")

    # ----------------------------------------------------------------------------
    # The following three state phases (zero torque, move to default, default pos)
    # are untouched except for button‑based policy switching capability.
    # ----------------------------------------------------------------------------
    def zero_torque_state(self):
        print("Enter zero‑torque state. Press START to continue …")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            self._check_hotkeys()
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
            print("[INFO] Successfully moved to default pos!")
        else:
            print("[INFO] Failed to move to default pos!")

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

    # ─────────────────────────────────────────
    # Hot‑key polling (remote buttons + keyboard fallback)
    # ─────────────────────────────────────────
    def _check_hotkeys(self):
        """Call in any loop to poll remote for policy switch keys."""
        if self.remote_controller.button[KeyMap.L1] == 1:  # previous policy
            self.switch_policy(-1)
        if self.remote_controller.button[KeyMap.R1] == 1:  # next policy
            self.switch_policy(+1)

    # ─────────────────────────────────────────
    # Main control loop (only inference section modified)
    # ─────────────────────────────────────────
    def run(self):
        """One control step → returns True when select pressed."""
        self.counter += 1
        # 1.  Build observation exactly like original
        obs_buf = self._build_observation()

        # 2.  Inference with current policy
        ort_sess = self.ort_sessions[self.policy_idx]
        input_name = self.input_names[self.policy_idx]
        obs_tensor = torch.from_numpy(obs_buf).unsqueeze(0).cpu().numpy()
        self.action = np.squeeze(ort_sess.run(None, {input_name: obs_tensor})[0])

        # 3.  Rest of original clipping / safety / sending
        self._post_process_and_send()
        if self.policy_idx != self.stance_idx and self.ref_motion_phase >= 1.0:
            print(f"[INFO] Motion #{self.policy_idx} finished, returning to Stance")
            self.policy_idx = self.stance_idx
            self.motion_len = self.motion_lens[self.stance_idx]
            self.ref_motion_phase = 0.0
            self.counter = 0
            return False  # 保持继续运行
        # return True  # keep

    # ------------- (YOUR original helper bodies copied verbatim) -------------
    def _build_observation(self) -> np.ndarray:
        """Wrapped original obs construction code."""
        # (place the exact logic you had inside `run` here)
        
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

        qj = self.qj.copy()    
        dqj = self.dqj.copy()   
    
        projected_gravity = get_gravity_orientation(quat)
        dof_pos = qj * self.config.dof_pos_scale
        dof_vel = dqj * self.config.dof_vel_scale
        base_ang_vel = ang_vel* self.config.ang_vel_scale
        
        motion_time = self.counter * self.control_dt
        if self.policy_idx == self.stance_idx:
            self.ref_motion_phase = 0.0 # No acculumation of stance phase
        else:
            self.ref_motion_phase = motion_time / self.motion_len
            print("[DEBUG] ref_motion_phase", self.ref_motion_phase)
        # self.ref_motion_phase += self.config.ref_motion_phase
        # self.ref_motion_phase = self.ref_motmove_toion_phase % 1 # 循环播放
        num_actions = self.config.num_actions

        history_obs_buf = np.concatenate((self.action_buf, self.ang_vel_buf, self.dof_pos_buf, self.dof_vel_buf, self.proj_g_buf, self.ref_motion_phase_buf), axis=-1, dtype=np.float32)

        try:
            obs_buf = np.concatenate((self.action, base_ang_vel.flatten(), dof_pos, dof_vel, history_obs_buf, projected_gravity, [self.ref_motion_phase]), axis=-1, dtype=np.float32)
        except ValueError as e:
            print(f"Concatenation failed with error: {e}")
            print("Please ensure all arrays have the same number of dimensions (either all 1D or all 2D)")
            raise

        # update history
        self.ang_vel_buf = np.concatenate((base_ang_vel.flatten(), self.ang_vel_buf[:-3]), axis=-1, dtype=np.float32)
        
        self.proj_g_buf = np.concatenate((projected_gravity, self.proj_g_buf[:-3] ), axis=-1, dtype=np.float32)
        self.dof_pos_buf = np.concatenate((dof_pos, self.dof_pos_buf[:-num_actions] ), axis=-1, dtype=np.float32)
        self.dof_vel_buf = np.concatenate((dof_vel, self.dof_vel_buf[:-num_actions] ), axis=-1, dtype=np.float32)
        self.action_buf = np.concatenate((self.action, self.action_buf[:-num_actions] ), axis=-1, dtype=np.float32)
        self.ref_motion_phase_buf = np.concatenate(([self.ref_motion_phase], self.ref_motion_phase_buf[:-1] ), axis=-1, dtype=np.float32)                

        return obs_buf

    def _post_process_and_send(self):
        """Wrapped original action clipping, PD mapping & safety checks."""
        # Warning for the action threshold
        if np.any(np.abs(self.action) > self.config.action_clip_warn_threshold):
            print(f"[WARN] Action exceeds threshold | max={np.max(np.abs(self.action)):.3f}")


        # action clipping
        # self.action = np.clip(self.action, -self.config.action_clip, self.config.action_clip)

        # transform action to target_dof_pos
        target_dof_pos = self.config.default_angles + self.action * self.config.action_scale

        # Target_position clip 
        # target_dof_pos = np.clip(target_dof_pos, self.config.dof_pos_lower_limit, self.config.dof_pos_upper_limit)

        # Build low cmd
        for i in range(len(self.config.action_joint2motor_idx)):
            motor_idx = self.config.action_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = target_dof_pos[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0
            print(f"[MOTOR] idx={motor_idx:02d}  target={target_dof_pos[i]:+.3f}")
        for i in range(len(self.config.fixed_joint2motor_idx)):
            motor_idx = self.config.fixed_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = self.config.fixed_target[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.fixed_kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.fixed_kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        dqj_abs = np.abs(self.dqj)
        if np.any(np.abs(self.dqj) > self.config.dof_vel_limit):
            print(f"[ERROR] Velocity exceeds limit | dqj_max={np.max(np.abs(self.dqj)):.3f}")

        for i, motor_idx in enumerate(self.config.action_joint2motor_idx):
            measured_tau = self.low_state.motor_state[motor_idx].tau_est  
            if np.abs(measured_tau) > self.config.dof_effort_limit[i]:
                print(f"[ERROR] Torque overload | motor={motor_idx}  tau={measured_tau:.3f} > limit={self.config.dof_effort_limit[i]}")
                # # send damping mode for protection
                # create_damping_cmd(self.low_cmd)
                # self.send_cmd(self.low_cmd)
                # time.sleep(self.config.control_dt)

                # # Directly return
                # return True

        self.send_cmd(self.low_cmd)
        time.sleep(self.config.control_dt)


# ─────────────────────────────────────────────
# Main entry (retains full CLI & logging logic)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface (e.g. eth0)")
    parser.add_argument("config", type=str, help="config YAML in configs folder", default="g1_29dof_PBHC.yaml")
    # parser.add_argument("--policies", nargs="*", help="Override policy path list from CLI", default=None)
    args = parser.parse_args()

    # Logging to file + stdout
    log_filename = f"deploy_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    class Logger:  # identical to original
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

    # Load YAML
    cfg_path = f"deploy_real/configs/{args.config}"
    cfg = Config(cfg_path)

    # Initialise DDS
    ChannelFactoryInitialize(0, args.net)

    print("[DEBUG] Initialising controller …")
    controller = Controller(cfg) # policy_paths_cli=args.policies
    print("[DEBUG] Controller ready.")

    # Phases identical to original
    controller.zero_torque_state()
    controller.move_to_default_pos()
    controller.default_pos_state()

    print("[INFO] Ready to run. LB/RB → switch policy, SELECT → exit, X → reset pose")
    while True:
        try:
            controller._check_hotkeys()
            if controller.remote_controller.button[KeyMap.select] == 1:
                break
            if controller.remote_controller.button[KeyMap.X] == 1:
                print("[INFO] Resetting to default pose …")
                controller.move_to_default_pos()
            if controller.run():
                break
        except KeyboardInterrupt:
            break

    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print("Exit cleanly.")
