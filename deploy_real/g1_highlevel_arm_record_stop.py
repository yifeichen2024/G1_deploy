# G1 Arm Motion Recorder with Looping Segments and Damping Mode
import time
import sys
import numpy as np
import yaml
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from common.remote_controller import RemoteController, KeyMap
# -----------------------------------------------------------------------------
# G1 Joint Index
# -----------------------------------------------------------------------------
class G1JointIndex:
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    WaistYaw = 12
    WaistRoll = 13
    WaistPitch = 14
    LeftWristRoll = 19
    LeftWristPitch = 20
    LeftWristYaw = 21
    RightWristRoll = 26
    RightWristPitch = 27
    RightWristYaw = 28
    kNotUsedJoint = 29

# -----------------------------------------------------------------------------
# Config loader
# -----------------------------------------------------------------------------
class Config: pass

def load_cfg(yaml_path="deploy_real/configs/config_high_level.yaml") -> Config:
    with open(yaml_path, 'r') as f:
        d = yaml.safe_load(f)
    cfg = Config()
    for k, v in d.items():
        setattr(cfg, k, np.array(v) if isinstance(v, list) else v)
    cfg.kps_record = cfg.kps_play * 1
    cfg.kds_record = cfg.kds_play * 1
    return cfg

cfg = load_cfg()

# -----------------------------------------------------------------------------
class CustomRecorder:
    def __init__(self):
        self.recording = False
        self.low_state = None
        self.crc = CRC()
        self.first_state = False
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.current_target_q = cfg.default_angles.copy()
        self.remote = RemoteController()
        self.t_record_start = time.time()
        self.record_buffer_t = []
        self.record_buffer_q = []

    def Init(self):
        self.arm_pub = ChannelPublisher("rt/arm_sdk", LowCmd_); self.arm_pub.Init()
        self.state_sub = ChannelSubscriber("rt/lowstate", LowState_); self.state_sub.Init(self.cb, 10)

    def cb(self, msg: LowState_):
        self.low_state = msg
        if not self.first_state:
            self.first_state = True
        self.remote.set(msg.wireless_remote)


    def Start(self):
        # self.thread = RecurrentThread(interval=cfg.control_dt, target=self.Loop, name="control")
        # while not self.first_state: 
        #     time.sleep(0.1)
        # self.thread.Start()

        self.thread = RecurrentThread(interval=cfg.control_dt, target=self.Loop, name="control")
        while not self.first_state: 
            time.sleep(0.1)
        self.thread.Start()

        # —— 启动后先做一次平滑过渡到默认姿态 —— #
        print("[INIT] 平滑过渡到默认姿态…")
        # 临时进入“录制”模式，使用低增益
        self.recording = True # This will let got to zero first. 
        self.move_to_default(duration=3.0)
        self.recording = False
    def Loop(self):
        if self.low_state is None: 
            return
        
        # # —— 1) 如果在录制模式，就把目标设成当前真实角度，使 PD 锁当前位置 —— #
        # actual_q = np.array([self.low_state.motor_state[m].q for m in cfg.action_joints])
        # if self.recording:
        #     self.current_target_q = actual_q.copy()

        kps_record: list[float] = []
        kds_record: list[float] = []
        for idx, joint in enumerate(cfg.action_joints):
            if joint in [G1JointIndex.WaistRoll, G1JointIndex.WaistPitch]:
                kps_record.append(cfg.kps_play[idx] * cfg.stiffness_factor_waist_rp)
                kds_record.append(cfg.kds_play[idx] * cfg.stiffness_factor_waist_rp)
            else:
                kps_record.append(cfg.kps_play[idx] * cfg.stiffness_factor)
                kds_record.append(cfg.kds_play[idx] * cfg.stiffness_factor)

        # build command each cycle
        if self.recording:
            kp_arr, kd_arr = kps_record, kds_record
        else:
            kp_arr, kd_arr = cfg.kps_play, cfg.kds_play

        for i, m in enumerate(cfg.action_joints):
            self.low_cmd.motor_cmd[m].q = float(self.current_target_q[i])
            self.low_cmd.motor_cmd[m].dq = 0.0
            self.low_cmd.motor_cmd[m].kp = float(kp_arr[i])
            self.low_cmd.motor_cmd[m].kd = float(kd_arr[i])
            self.low_cmd.motor_cmd[m].tau = 0.0

        for i, m in enumerate(cfg.fixed_joints):
            self.low_cmd.motor_cmd[m].q = float(cfg.fixed_target[i])
            self.low_cmd.motor_cmd[m].dq = 0.0
            self.low_cmd.motor_cmd[m].kp = float(cfg.fixed_kps[i])
            self.low_cmd.motor_cmd[m].kd = float(cfg.fixed_kds[i])
            self.low_cmd.motor_cmd[m].tau = 0.0

        self.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = 1
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.arm_pub.Write(self.low_cmd)

        if self.recording:
            t = time.time() - self.t_record_start
            q = np.array([self.low_state.motor_state[m].q for m in cfg.action_joints])
            self.record_buffer_t.append(t)
            self.record_buffer_q.append(q)

    def move_to_default(self, duration=5.0):
        cur = np.array([self.low_state.motor_state[m].q for m in cfg.action_joints])
        steps = int(duration / cfg.control_dt)
        for i in range(steps):
            r = (i + 1) / steps
            self.current_target_q = (1 - r) * cur + r * cfg.default_angles
            time.sleep(cfg.control_dt)
        self.current_target_q = cfg.default_angles.copy()

    def Run(self):
        self.move_to_default()
        print("[RECORD] Press A to start, B to stop/save, Y to finish, X to reset")
        seg_id = 1

        while True:
            key = self.low_state.wireless_remote[2]
            if not self.recording and self.remote.button[KeyMap.A] == 1:  # A
                self.recording = True
                self.t_record_start = time.time()
                self.record_buffer_t = []
                self.record_buffer_q = []
                print(f"[RECORD] ▶ Start segment {seg_id}")

            elif self.recording and self.remote.button[KeyMap.B] == 1:  # B
                self.recording = False
                self.current_target_q = self.record_buffer_q[-1]
                np.savez_compressed(f"segment_{seg_id:02d}.npz",
                    t=np.array(self.record_buffer_t),
                    q=np.vstack(self.record_buffer_q))
                print(f"[SAVE] segment_{seg_id:02d}.npz saved")
                seg_id += 1

            elif (self.remote.button[KeyMap.Y] == 1):  # Y
                print("[RECORD] Finish and enter damping mode")
                for i in cfg.action_joints:
                    self.low_cmd.motor_cmd[i].kp = 0
                    self.low_cmd.motor_cmd[i].kd = 1
                self.arm_pub.Write(self.low_cmd)

                print("Press X to return to default")
                while True:
                    key = self.low_state.wireless_remote[2]
                    if self.remote.button[KeyMap.X] == 1:  # X
                        self.move_to_default(6.0)
                        print("[RECORD] Done. Ctrl+C to exit.")
                        return
            time.sleep(0.05)

if __name__ == "__main__":
    print("Ensure no obstacle. Press ENTER..."); input()
    ChannelFactoryInitialize(0, sys.argv[1] if len(sys.argv) > 1 else None)
    recorder = CustomRecorder()
    recorder.Init()
    recorder.Start()
    recorder.Run()
