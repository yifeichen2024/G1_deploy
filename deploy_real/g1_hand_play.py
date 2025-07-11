#!/usr/bin/env python3
import time
import sys
import argparse
import numpy as np
from pathlib import Path

from g1_highlevel_hand import Dex3_1_Controller
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

def main():
    p = argparse.ArgumentParser(description="Dex3 Hand Playback")
    p.add_argument("traj", type=str, help="Trajectory file (.npz)")
    p.add_argument("--speed", "-s", type=float, default=1.0, help="Playback speed (1.0=原速)")
    p.add_argument("--ip",        type=str, default="enp2s0", help="Robot 网络接口（可选）")
    args = p.parse_args()

    # # DDS 初始化
    # if args.ip:
    #     ChannelFactoryInitialize(0, args.ip)
    # else:
    #     ChannelFactoryInitialize(0)

    # 创建控制器
    ctrl = Dex3_1_Controller(fps=100.0)
    # 等几帧让订阅稳定
    time.sleep(0.2)

    # 加载轨迹
    traj_path = Path(args.traj)
    if not traj_path.exists():
        print(f"[ERROR] 文件不存在: {traj_path}")
        sys.exit(1)
    data = np.load(traj_path)
    t_arr = (data["t"] - data["t"][0]) / args.speed   # 时间归一化 + 速度缩放
    q_arr = data["q"]                                 # 每帧 6 维：[3 left | 3 right]

    print(f"[INFO] Loaded '{traj_path.name}'  共 {len(t_arr)} 帧，播放速度={args.speed}×")
    print("按 ENTER 开始播放…")
    input()

    # 1) 平滑到默认手势
    ctrl.move_to_default(duration=1.0)

    # 2) 等待 ENTER 再次按下才真正开始计时
    print("按 ENTER 确认开始播放轨迹")
    input()
    t0 = time.time()

    # 3) 回放
    for i, ti in enumerate(t_arr):
        left_q  = q_arr[i, :7]
        right_q = q_arr[i, 7:]
        print(f"[DEBUG] left: {left_q}, right: {right_q}")
        ctrl.set_target_q(left_q, right_q)
        ctrl.send_cmd()
        
        # 计算到下一帧的 sleep
        if i < len(t_arr)-1:
            next_t = t_arr[i+1]
            now   = time.time() - t0
            to_sleep = next_t - now
            if to_sleep > 0:
                time.sleep(to_sleep)

    print("[INFO] 播放结束，当前保持在最后一帧姿态")

    # 4) 等待 ENTER 再平滑回默认手势
    print("按 ENTER 平滑回到默认手势…")
    input()
    ctrl.move_to_default(duration=1.0)

    print("Done. Exit.")

if __name__ == "__main__":
    main()
