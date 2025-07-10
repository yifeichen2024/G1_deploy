import time
import sys
import select
import threading
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

# ------------------- G1 Joint Map (官方) -------------------- #
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

# ------------------- 配置加载 -------------------- #
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

# ------------------- 构建动作库 -------------------- #
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

        # 状态机：IDLE, PLAYING, HOLDING
        self.state = "IDLE"
        self.current_key = None
        self.start_time = None
        self.idx = 0
        self.last_q = None

        # DDS 相关
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state: LowState_ | None = None
        self.first_state = False
        self.crc = CRC()

        # 遥控器
        self.remote = RemoteController()

    # -------- DDS 初始化 --------
    def init_dds(self):
        self.pub = ChannelPublisher("rt/arm_sdk", LowCmd_); self.pub.Init()
        self.sub = ChannelSubscriber("rt/lowstate", LowState_); self.sub.Init(self.cb, 10)
        while not self.first_state:
            time.sleep(0.1)
        print("[DDS] Ready & in default pose. 输入动作编号并回车开始播放。")

    def cb(self, msg: LowState_):
        self.low_state = msg
        self.first_state = True

    # -------- 启动控制循环线程 --------
    def start_thread(self):
        self.thread = RecurrentThread(interval=cfg.control_dt, target=self.loop, name="control")
        self.thread.Start()

    # -------- 主循环 --------
    def loop(self):
        # 先更新遥控器状态
        self.remote.update()

        if self.state == "HOLDING":
            # 保持最后一帧姿态
            self.send_pose(self.last_q)
            # 按 B 键归位
            if self.remote.button[KeyMap.B] == 1:
                print("[HOLD] B pressed -> 返回 default")
                threading.Thread(target=self._async_return_default, daemon=True).start()
                self.state = "IDLE"
                return
            # 支持直接按数字键跳下一个动作
            for k in self.bank:
                map_attr = f"NUM_{k}"
                if hasattr(KeyMap, map_attr) and self.remote.button[getattr(KeyMap, map_attr)] == 1:
                    print(f"[HOLD] 按下 {k} -> 直接播放下一个动作")
                    self._start_motion(k)
                    return
            return

        if self.state == "IDLE":
            # 保持 default
            self.send_pose(cfg.default_angles)
            # 标准输入也可触发
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.readline().strip()
                if key in self.bank:
                    self._start_motion(key)
            return

        if self.state == "PLAYING":
            motion = self.bank[self.current_key]
            t_arr = motion["t"] / self.speed
            q_arr = motion["q"]

            elapsed = time.time() - self.start_time
            while self.idx < len(t_arr) and t_arr[self.idx] <= elapsed:
                self.idx += 1

            if self.idx >= len(t_arr):
                print("[PLAY] 动作播放完毕，切换到 HOLDING")
                self.last_q = q_arr[-1]
                self.state = "HOLDING"
            else:
                self.send_pose(q_arr[self.idx])

    # -------- 开始播放动作 --------
    def _start_motion(self, key: str):
        print(f"[PLAY] Start motion {key}: {self.bank[key]['name']}")
        self.current_key = key
        self.start_time = time.time()
        self.idx = 0
        self.state = "PLAYING"

    # -------- 异步归位到 default --------
    def _async_return_default(self, duration: float = 3.0):
        if self.low_state is None:
            return
        cur = np.array([self.low_state.motor_state[m].q for m in cfg.action_joints])
        steps = int(duration / cfg.control_dt)
        for s in range(steps):
            r = (s + 1) / steps
            q_cmd = (1 - r) * cur + r * cfg.default_angles
            self.send_pose(q_cmd)
            time.sleep(cfg.control_dt)

    # -------- 发送关节指令 --------
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

# ------------------- 主入口 -------------------- #
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python g1_arm_play.py <traj1.npz> <traj2.npz> ... [--ip IP] [--speed S]")
        sys.exit(0)

    speed = 1.0
    ip = None
    traj_paths = []
    for arg in sys.argv[1:]:
        if arg.startswith("--speed"):
            speed = float(arg.split("=", 1)[-1])
        elif arg.startswith("--ip"):
            ip = arg.split("=", 1)[-1]
        else:
            traj_paths.append(arg)

    if ip:
        ChannelFactoryInitialize(0, ip)
    else:
        ChannelFactoryInitialize(0)

    motion_bank = build_motion_bank(traj_paths)
    if not motion_bank:
        print("[ERROR] No traj files provided!")
        sys.exit(0)

    print("[PLAY] Play list:")
    for k, v in motion_bank.items():
        print(f"  {k}: {v['name']}")
    print("[PLAY] Enter the number and press ENTER to run. CTRL + C to end the program.\n")

    player = Player(motion_bank, speed)
    player.init_dds()
    player.start_thread()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("KeyboardInterrupt, moving to default…")
        player._async_return_default(3.0)
        print("Exit.")
