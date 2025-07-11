import time
import sys
import select
import threading
from pathlib import Path
from typing import Dict, List, Optional, Callable
import numpy as np
import yaml
import os

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread

from common.remote_controller import RemoteController, KeyMap
# TODO 插值不够平滑,开启时刻插值不够平滑，会导致连接断开等问题。
# ------------------- G1 Joint Map (官方) -------------------- #
class G1JointIndex:
    LeftHipPitch = 0; LeftHipRoll = 1; LeftHipYaw = 2; LeftKnee = 3
    LeftAnklePitch = 4; LeftAnkleRoll = 5
    RightHipPitch = 6; RightHipRoll = 7; RightHipYaw = 8; RightKnee = 9
    RightAnklePitch = 10; RightAnkleRoll = 11
    WaistYaw = 12; WaistRoll = 13; WaistPitch = 14
    LeftShoulderPitch = 15; LeftShoulderRoll = 16; LeftShoulderYaw = 17
    LeftElbow = 18; LeftWristRoll = 19; LeftWristPitch = 20; LeftWristYaw = 21
    RightShoulderPitch = 22; RightShoulderRoll = 23; RightShoulderYaw = 24
    RightElbow = 25; RightWristRoll = 26; RightWristPitch = 27; RightWristYaw = 28
    kNotUsedJoint = 29

# ------------------- 配置加载 -------------------- #
class Config: pass

def load_cfg(yaml_path="deploy_real/configs/config_high_level.yaml") -> Config:
    with open(yaml_path, 'r') as f:
        d = yaml.safe_load(f)
    cfg = Config()
    for k, v in d.items():
        setattr(cfg, k, np.array(v) if isinstance(v, list) else v)
    # 为平滑过渡准备更小的 kp/kd
    cfg.kps_record = cfg.kps_play * 0.1
    cfg.kds_record = cfg.kds_play * 0.1
    # 新增：过渡用时（秒），可根据需要调整或从 yaml 配置读取
    cfg.transition_duration = 2
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

        # 状态机：IDLE, TRANSITION, PLAYING, HOLDING, RETURNING
        self.state = "IDLE"
        self.current_key: Optional[str] = None
        self.start_time: Optional[float] = None
        self.idx = 0
        self.last_q: Optional[np.ndarray] = None

        # DDS 相关
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state: Optional[LowState_] = None
        self.first_state = False
        self.crc = CRC()

        # 遥控器
        self.remote = RemoteController()

    # -------- DDS 初始化 --------
    def init_dds(self):
        self.pub = ChannelPublisher("rt/arm_sdk", LowCmd_); self.pub.Init()
        self.sub = ChannelSubscriber("rt/lowstate", LowState_); self.sub.Init(self.cb, 10)
        # 等待首帧状态
        while not self.first_state:
            time.sleep(0.5)
        # 首次平滑过渡到默认姿态
        # TODO 从当前状态平滑过渡到default状态非常不平滑，会震动DDS报错连接错误等问题。这个插值还是有问题。需要换方法。减少acc和jerk
        # TODO 中间动作是连续的插值还比较平滑，就是程序启动到default状态的时候会有震动的问题。进而会导致整个dds连接有问题。
        # 1752263190.284963 [0]     python: ddsi_udp_conn_write to udp/239.255.0.1:7401 failed with retcode -1
        # 1752263190.291338 [0]     python: ddsi_udp_conn_write to udp/239.255.0.1:7401 failed with retcode -1
        print("[INIT] 平滑过渡到默认姿态…")
        self._start_transition(
            target_q=cfg.default_angles,
            duration=cfg.transition_duration,
            on_complete=lambda: print("[DDS] Ready & in default pose. 输入动作编号并回车开始播放。")
        )

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
            if self.state == "PLAYING":
                motion = self.bank[self.current_key]
                idx = min(self.idx, len(motion["q"]) - 1)
                q_target = motion["q"][idx]
            elif self.state == "HOLDING" and self.last_q is not None:
                q_target = self.last_q
            else:
                q_target = cfg.default_angles
            self.send_pose(q_target)
            os._exit(0)

        # 过渡中跳过其它 send
        if self.state in ("TRANSITION", "RETURNING"):
            return

        # HOLDING：保持最后一帧，等待 B 或数字键
        if self.state == "HOLDING":
            self.send_pose(self.last_q)
            # if self.remote.button[KeyMap.B] == 1:
            #     print("[HOLD] B pressed -> 归位中…")
            #     self.state = "RETURNING"
            #     threading.Thread(
            #         target=self._async_return_default,
            #         daemon=True
            #     ).start()
            #     return
            #                  
            # TODO Button B is not working.
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                inp = sys.stdin.readline().strip()
                if self.remote.button[KeyMap.B] == 1 or inp.lower() == "b":
                    print("[HOLD] B pressed -> 归位中…")
                    self.state = "RETURNING"
                    threading.Thread(
                        target=self._async_return_default,
                        daemon=True
                    ).start()
                    return
                # 如果输入的是数字，对应某个动作编号
                if inp in self.bank:
                    print(f"[HOLD] 输入 {inp} -> 播放下一动作（平滑过渡）")
                    self._start_motion(inp)
                    return
            return

        # IDLE：保持 default，等待数字键或标准输入
        if self.state == "IDLE":
            # 这里 send_pose 也仅用于姿态补发，实际启动动作走过渡
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
            # 更新 idx
            while self.idx < len(t_arr) and t_arr[self.idx] <= elapsed:
                self.idx += 1

            if self.idx >= len(t_arr):
                print("[PLAY] 动作播放完毕，切换到 HOLDING")
                self.last_q = q_arr[-1]
                self.state = "HOLDING"
            else:
                self.send_pose(q_arr[self.idx])

    # -------- 开始播放动作：先平滑过渡 --------
    def _start_motion(self, key: str):
        motion = self.bank[key]
        target_q = motion["q"][0]
        print(f"[TRANSITION] 从当前位置平滑过渡到动作 {key}: {motion['name']}")
        self.state = "TRANSITION"
        # 插值完成后自动进入 PLAYING
        def on_complete():
            print(f"[PLAY] 过渡完成，开始播放 {key}")
            self.current_key = key
            self.start_time = time.time()
            self.idx = 0
            self.state = "PLAYING"
        self._start_transition(
            target_q=target_q,
            duration=3,
            on_complete=on_complete
        )

    # -------- 通用异步插值函数 --------
    def _start_transition(self,
                          target_q: np.ndarray,
                          duration: float,
                          on_complete: Callable[[], None]):
        def _worker():
            if self.low_state is None:
                on_complete()
                return
            # 读当前真实关节角度
            cur_q = np.array([self.low_state.motor_state[m].q for m in cfg.action_joints])
            steps = max(1, int(duration / cfg.control_dt))
            for s in range(steps):
                r = (s + 1) / steps
                q_cmd = (1 - r) * cur_q + r * target_q
                self.send_pose(q_cmd)
                time.sleep(cfg.control_dt*5)
            on_complete()

        threading.Thread(target=_worker, daemon=True).start()

    # -------- 异步归位到 default --------
    def _async_return_default(self, duration: float = 3.0):
        # RETURNING 状态下使用更长的 duration
        target = cfg.default_angles
        def on_complete():
            self.state = "IDLE"
            print("[RETURN] 回到默认姿态，IDLE")
        self._start_transition(target_q=target, duration=duration, on_complete=on_complete)

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

        # 必填：激活消息发送
        self.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = 1
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.pub.Write(self.low_cmd)

# ------------------- 主入口 -------------------- #
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python g1_arm_play.py <traj1.npz> <traj2.npz> ... [--ip IP] [--speed S]")
        sys.exit(0)

    speed = 1.0; ip = None; traj_paths: List[str] = []
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
        player.state = "RETURNING"
        try:
            player._async_return_default(5.0)
        except KeyboardInterrupt:
            pass
        print("Exit.")
