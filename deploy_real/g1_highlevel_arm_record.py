import time
import sys
import select
import numpy as np

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread

# -----------------------------------------------------------------------------
# G1 Joint Index (official mapping)
# -----------------------------------------------------------------------------
class G1JointIndex:
    LeftHipPitch = 0
    LeftHipRoll = 1
    LeftHipYaw = 2
    LeftKnee = 3
    LeftAnklePitch = 4
    LeftAnkleRoll = 5

    RightHipPitch = 6
    RightHipRoll = 7
    RightHipYaw = 8
    RightKnee = 9
    RightAnklePitch = 10
    RightAnkleRoll = 11

    WaistYaw = 12
    WaistRoll = 13
    WaistPitch = 14

    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20
    LeftWristYaw = 21

    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27
    RightWristYaw = 28

    kNotUsedJoint = 29

# -----------------------------------------------------------------------------
# Configuration (17 DOF record, per‑joint PD, fixed wrists)
# -----------------------------------------------------------------------------

class Config:
    control_dt = 0.02  # 20 ms

    # action joints (17 DOF: arms + waist)
    action_joints = [
        # left arm 7
        G1JointIndex.LeftShoulderPitch,
        G1JointIndex.LeftShoulderRoll,
        G1JointIndex.LeftShoulderYaw,
        G1JointIndex.LeftElbow,
        G1JointIndex.LeftWristRoll,
        G1JointIndex.LeftWristPitch,
        G1JointIndex.LeftWristYaw,
        # right arm 7
        G1JointIndex.RightShoulderPitch,
        G1JointIndex.RightShoulderRoll,
        G1JointIndex.RightShoulderYaw,
        G1JointIndex.RightElbow,
        G1JointIndex.RightWristRoll,
        G1JointIndex.RightWristPitch,
        G1JointIndex.RightWristYaw,
        # waist 3
        G1JointIndex.WaistYaw,
        G1JointIndex.WaistRoll,
        G1JointIndex.WaistPitch,
    ]

    # joints to lock during record (wrists)
    fixed_joints = [
        G1JointIndex.LeftWristRoll,
        G1JointIndex.LeftWristPitch,
        G1JointIndex.LeftWristYaw,
        G1JointIndex.RightWristRoll,
        G1JointIndex.RightWristPitch,
        G1JointIndex.RightWristYaw,
    ]
    fixed_target = np.array([-0.05, 0.12, -0.03, -0.16, 0.12, -0.02])
    fixed_kps = np.array([60, 60, 60, 60, 60, 60])
    fixed_kds = np.array([1, 1, 1, 1, 1, 1])

    # per‑joint PD (play/default) 17 DOF
    kps_play = np.array([
        100, 100, 50, 50, 100, 100, 50,
        100, 100, 50, 50, 100, 100, 50,
        400, 400, 400,
    ])
    kds_play = np.array([
        2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2,
        5, 5, 5,
    ])

    # record PD (10 % stiff)
    kps_record = 0.5 * kps_play
    kds_record = 0.5 * kds_play

    # default pose (17 DOF order)
    default_angles = np.array([
        0.2, 0.2, 0.0, 0.9, 0.0, 0.0, 0.0,
        0.2, -0.2, 0.0, 0.9, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
    ])

cfg = Config()

# -----------------------------------------------------------------------------
# Custom recording class (follow official SDK skeleton)
# -----------------------------------------------------------------------------

class CustomRecorder:
    def __init__(self):
        # timing
        self.time_ = 0.0
        self.recording = False
        self.record_buffer_t = []  # timestamps
        self.record_buffer_q = []  # ndarray 17

        # SDK objects
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = None
        self.first_state = False
        self.crc = CRC()

        # publishers/subscribers init later

    # -----------------------------------------------------------------
    # DDS init (exactly same pattern as template)
    # -----------------------------------------------------------------
    def Init(self):
        self.arm_pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self.arm_pub.Init()

        self.state_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.state_sub.Init(self.LowStateHandler, 10)

    def Start(self):
        # control loop thread
        self.thread = RecurrentThread(interval=cfg.control_dt, target=self.Loop, name="control")
        while not self.first_state:
            time.sleep(0.1)
        self.thread.Start()

    # -----------------------------------------------------------------
    # State callback
    # -----------------------------------------------------------------
    def LowStateHandler(self, msg: LowState_):
        self.low_state = msg
        if not self.first_state:
            self.first_state = True

    # -----------------------------------------------------------------
    # Main loop (send PD & optionally log)
    # -----------------------------------------------------------------
    def Loop(self):
        # if robot state missing, skip
        if self.low_state is None:
            return

        # build command each cycle
        if self.recording:
            kp_arr, kd_arr = cfg.kps_record, cfg.kds_record
        else:
            kp_arr, kd_arr = cfg.kps_play, cfg.kds_play

        # set action‑joint commands
        for idx, motor in enumerate(cfg.action_joints):
            target_q = self.current_target_q[idx]
            self.low_cmd.motor_cmd[motor].q = float(target_q)
            self.low_cmd.motor_cmd[motor].dq = 0.0
            self.low_cmd.motor_cmd[motor].kp = float(kp_arr[idx])
            self.low_cmd.motor_cmd[motor].kd = float(kd_arr[idx])
            self.low_cmd.motor_cmd[motor].tau = 0.0

        # lock fixed joints
        for i, motor in enumerate(cfg.fixed_joints):
            self.low_cmd.motor_cmd[motor].q = float(cfg.fixed_target[i])
            self.low_cmd.motor_cmd[motor].dq = 0.0
            self.low_cmd.motor_cmd[motor].kp = float(cfg.fixed_kps[i])
            self.low_cmd.motor_cmd[motor].kd = float(cfg.fixed_kds[i])
            self.low_cmd.motor_cmd[motor].tau = 0.0

        # enable SDK
        self.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = 1
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.arm_pub.Write(self.low_cmd)

        # log if recording
        if self.recording:
            t_now = time.time() - self.t_record_start
            q_snapshot = np.array([
                self.low_state.motor_state[m].q for m in cfg.action_joints
            ])
            self.record_buffer_t.append(t_now)
            self.record_buffer_q.append(q_snapshot)

    # -----------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------
    def move_to_default(self, duration=3.0):
        # interpolate from current to default
        current_q = np.array([
            self.low_state.motor_state[m].q for m in cfg.action_joints
        ])
        steps = int(duration / cfg.control_dt)
        for step in range(steps):
            ratio = (step + 1) / steps
            self.current_target_q = (1 - ratio) * current_q + ratio * cfg.default_angles
            time.sleep(cfg.control_dt)
        # hold default afterwards
        self.current_target_q = cfg.default_angles.copy()

    # -----------------------------------------------------------------
    # Public workflow
    # -----------------------------------------------------------------
    def Run(self):
        # 1. Go to default (stiff)
        self.current_target_q = cfg.default_angles.copy()
        self.move_to_default()

        # 2. Wait user start
        print("[RECORD] Press ENTER to start recording waypoints; ENTER again to stop.")
        input()
        self.recording = True
        self.t_record_start = time.time()
        print("[RECORD] Recording... (ENTER to stop)")

        # Wait for ENTER or Ctrl-C
        try:
            while True:
                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    if sys.stdin.readline().strip() == "":
                        break
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass

        self.recording = False
        print("[RECORD] Stopped. Saving file…")
        # 3. Save
        traj_path = input("Enter file path to save (.npz): ").strip()
        if traj_path == "":
            traj_path = "upper_body_traj.npz"
        np.savez_compressed(traj_path,
                            t=np.array(self.record_buffer_t),
                            q=np.vstack(self.record_buffer_q))
        print(f"[RECORD] Saved {len(self.record_buffer_t)} frames to {traj_path}")

        # 4. Return to default (stiff)
        self.current_target_q = cfg.default_angles.copy()
        self.move_to_default()
        print("Done. Ctrl‑C to exit.")

# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("WARNING: Ensure no obstacles around the robot. Press ENTER to continue …")
    input()

    # If user passes robot IP as argv[1]
    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    recorder = CustomRecorder()
    recorder.Init()
    recorder.Start()
    recorder.Run()

    # keep main thread alive
    while True:
        time.sleep(1)
