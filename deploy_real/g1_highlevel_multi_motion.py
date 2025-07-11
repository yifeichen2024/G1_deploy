import time
import sys
import select
import threading
from pathlib import Path
from typing import Dict, List
import numpy as np
import yaml
import os

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

        # 状态机：IDLE, PLAYING, HOLDING, RETURNING
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
        self.remote.set(msg.wireless_remote)

    # -------- 启动控制循环线程 --------
    def start_thread(self):
        self.thread = RecurrentThread(interval=cfg.control_dt, target=self.loop, name="control")
        self.thread.Start()

    # -------- 主循环 --------
    def loop(self):
        # —— 1. 任何时候按 SELECT 键就锁定当前位置并退出 —— #
        if self.remote.button[KeyMap.select] == 1:
            print("[LOOP] SELECT pressed → lock position and exit")
            # 根据当前状态拿到 q_target
            if self.state == "PLAYING":
                motion = self.bank[self.current_key]
                idx = min(self.idx, len(motion["q"]) - 1)
                q_target = motion["q"][idx]
            elif self.state == "HOLDING" and self.last_q is not None:
                q_target = self.last_q
            else:
                q_target = cfg.default_angles
            # 发送一次锁定姿态
            self.send_pose(q_target)
            # 直接退出整个进程
            os._exit(0)


        # 如果正在归位，跳过所有 send_pose
        if self.state == "RETURNING":
            return

        # HOLDING：保持最后一帧，等待 B 或数字键
        if self.state == "HOLDING":
            self.send_pose(self.last_q)
            if self.remote.button[KeyMap.B] == 1:
                print("[HOLD] B pressed -> 归位中…")
                self.state = "RETURNING"
                threading.Thread(target=self._async_return_default, daemon=True).start()
                return
            for k in self.bank:
                map_attr = f"NUM_{k}"
                if hasattr(KeyMap, map_attr) and self.remote.button[getattr(KeyMap, map_attr)] == 1:
                    print(f"[HOLD] 按下 {k} -> 直接播放下一个动作")
                    self._start_motion(k)
                    return
            return

        # IDLE：保持 default，等待数字键或标准输入
        if self.state == "IDLE":
            self.send_pose(cfg.default_angles)
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.readline().strip()
                if key in self.bank:
                    self._start_motion(key)
            return

        # PLAYING：播放动作帧
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
            self.state = "IDLE"
            return
        cur = np.array([self.low_state.motor_state[m].q for m in cfg.action_joints])
        steps = int(duration / cfg.control_dt)
        for s in range(steps):
            r = (s + 1) / steps
            q_cmd = (1 - r) * cur + r * cfg.default_angles
            self.send_pose(q_cmd)
            try:
                time.sleep(cfg.control_dt)
            except KeyboardInterrupt:
                break
        # 归位完毕，切回 IDLE
        self.state = "IDLE"

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
    traj_paths: List[str] = []
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
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("KeyboardInterrupt, moving to default…")
        # 切到 RETURNING，暂停 loop 中的 send_pose
        player.state = "RETURNING"
        try:
            player._async_return_default(5.0)
        except KeyboardInterrupt:
            pass
        print("Exit.")


