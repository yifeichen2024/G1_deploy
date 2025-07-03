import time
import sys
import select
from pathlib import Path
from typing import Dict, List
import numpy as np
import yaml

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread

from common.remote_controller import RemoteController, KeyMap
# ------------------- G1 Joint Map (official) -------------------- #
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
# ------------------- Config -------------------- #

# cfg = Config()

class Config: pass
def load_cfg(yaml_path="deploy_real/configs/config_high_level.yaml") -> Config:
    with open(yaml_path, 'r') as f:
        d = yaml.safe_load(f)
    cfg = Config()
    for k, v in d.items():
        setattr(cfg, k, np.array(v) if isinstance(v, list) else v)

    cfg.kps_record = cfg.kps_play * 0.1
    cfg.kds_record = cfg.kds_play * 0.1
    return cfg

cfg = load_cfg()  

# ------------------- Helper to load all motions -------------------- #

def build_motion_bank(paths: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
    bank = {}
    for idx, p in enumerate(paths, 1):
        data = np.load(p)
        bank[str(idx)] = {
            "name": Path(p).stem,
            "t": data["t"] - data["t"][0],
            "q": data["q"],
        }
    return bank

# ------------------- Player -------------------- #
class Player:
    def __init__(self, motion_bank: Dict[str, Dict[str, np.ndarray]], speed: float = 1.0):
        self.bank = motion_bank
        self.speed = speed
        self.current_key = None  # current playing 
        self.start_time = None
        self.idx = 0

        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state: LowState_ | None = None
        self.first_state = False
        self.crc = CRC()
        self.remote = RemoteController()
        self.btn2motion = {                       # 映射表
            KeyMap.up   : '1',
            KeyMap.right: '2',
            KeyMap.down : '3',
            KeyMap.left : '4',
            KeyMap.Y :    '5',
            KeyMap.B :    '6',
        }
        self.pending_key = None

    # -------- DDS --------
    def init_dds(self):
        self.pub = ChannelPublisher("rt/arm_sdk", LowCmd_); self.pub.Init()
        self.sub = ChannelSubscriber("rt/lowstate", LowState_); self.sub.Init(self.cb, 10)
        while not self.first_state:
            time.sleep(0.1)
        # print("[DDS] Ready & in default pose. Ender the number of the motion you want to play and press ENTER。")
        print("[DDS] Ready & in default pose.")
        print("Direction/Y/B play directly; keyboard number press A to play; X to stop\n")
    def cb(self, msg: LowState_):
        self.low_state = msg
        self.first_state = True
        # Remote controller
        self.remote.set(msg.wireless_remote)

    # -------- Thread loop --------
    def start_thread(self):
        self.thread = RecurrentThread(interval=cfg.control_dt, target=self.loop, name="control")
        self.thread.Start()

    def _start_motion(self, motion_id: str):
        if motion_id not in self.bank:
            print(f"[WARN] Motion {motion_id} not loaded.")
            return
        print(f"[PLAY] Start motion {motion_id}: {self.bank[motion_id]['name']}")
        self.current_key = motion_id
        self.start_time = time.time()
        self.idx = 0

    def loop(self):
        # keep default motion
        if self.current_key is None:                        # ---- idle (default) ----
            self.send_pose(cfg.default_angles)

            # --------- 2-a 手柄 6 个键直接触发 ---------
            for btn, motion_id in self.btn2motion.items():
                if self.remote.button[btn] == 1:
                    self._start_motion(motion_id)
                    return                                  # 本帧结束

            # --------- 2-b 键盘输入数字 -------------
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.readline().strip()
                if key in self.bank:
                    self.pending_key = key
                    print(f"[INPUT] Select {key} : {self.bank[key]['name']}, press A to confirm.")
                else:
                    print(f"[WARN] Invalid number {key}")

            # --------- 2-c  A 键确认播放 -------------
            if (self.pending_key is not None) and (self.remote.button[KeyMap.A] == 1):
                self._start_motion(self.pending_key)
                self.pending_key = None
                return

            # --------- 2-d  X 键重置 -------------
            if self.remote.button[KeyMap.X] == 1:
                print("[PLAY] Reset to default pose")
                self.move_to_default()
            return

        # ---- playing ----
        if self.remote.button[KeyMap.X] == 1:               
            print("[PLAY] Emergency stop!")
            self.move_to_default()
            self.current_key = None
            self.pending_key = None
            return
        
        # currently playing.
        motion = self.bank[self.current_key]
        t_arr = motion["t"] / self.speed
        q_arr = motion["q"]

        elapsed = time.time() - self.start_time
        while self.idx < len(t_arr) and t_arr[self.idx] <= elapsed:
            self.idx += 1
        if self.idx >= len(t_arr):
            print("[PLAY] Motion finished. Returning default…")
            self.move_to_default()
            self.current_key = None
            return
        self.send_pose(q_arr[self.idx])

    # -------- Core send pose --------
    def send_pose(self, q_target: np.ndarray):
        for k, m in enumerate(cfg.action_joints):
            self.low_cmd.motor_cmd[m].q = float(q_target[k])
            self.low_cmd.motor_cmd[m].dq = 0.0
            self.low_cmd.motor_cmd[m].kp = float(cfg.kps_play[k])
            self.low_cmd.motor_cmd[m].kd = float(cfg.kds_play[k])
            self.low_cmd.motor_cmd[m].tau = 0.0
        for i, m in enumerate(cfg.fixed_joints):
            self.low_cmd.motor_cmd[m].q = float(cfg.fixed_target[i])
            self.low_cmd.motor_cmd[m].dq = 0.0
            self.low_cmd.motor_cmd[m].kp = float(cfg.fixed_kps[i])
            self.low_cmd.motor_cmd[m].kd = float(cfg.fixed_kds[i])
            self.low_cmd.motor_cmd[m].tau = 0.0
        self.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = 1
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.pub.Write(self.low_cmd)

    # -------- Smooth return --------
    def move_to_default(self, duration: float = 3.0):
        if self.low_state is None:
            return
        cur = np.array([self.low_state.motor_state[m].q for m in cfg.action_joints])
        steps = int(duration / cfg.control_dt)
        for s in range(steps):
            r = (s + 1) / steps
            q_cmd = (1 - r) * cur + r * cfg.default_angles
            self.send_pose(q_cmd)
            time.sleep(cfg.control_dt)

# ------------------- MAIN -------------------- #
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python g1_highlevel_motion_controller.py <traj1.npz> <traj2.npz> ... [--ip IP] [--speed S]")
        sys.exit(0)

    speed = 1.0
    ip = None
    traj_paths = []
    for arg in sys.argv[1:]:
        if arg.startswith("--speed"):
            speed = float(arg.split("=")[-1])
        elif arg.startswith("--ip"):
            ip = arg.split("=")[-1]
        else:
            traj_paths.append(arg)

    if ip:
        ChannelFactoryInitialize(0, ip)
    else:
        ChannelFactoryInitialize(0)

    motion_bank = build_motion_bank(traj_paths)
    if not motion_bank:
        print("[ERROR] No traj files provided!"); sys.exit(0)

    print("[PLAY] Play list:")
    for k, v in motion_bank.items():
        print(f"  {k}: {v['name']}")
    print("[PLAY] Enter the number and press ENTER to run. CTRL + C to end the program.\n")

    player = Player(motion_bank, speed)
    player.init_dds()
    player.start_thread()

    # keep python alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("KeyboardInterrupt, moving to default…")
        player.move_to_default(3.0)
        print("Exit.")
