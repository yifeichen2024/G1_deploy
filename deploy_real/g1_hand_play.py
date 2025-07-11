#!/usr/bin/env python3
import time
import sys
import select
import numpy as np
from pathlib import Path

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from g1_highlevel_hand import Dex3_1_Controller

def main():
    # ---------------- 参数解析 ----------------
    # 用法: python hand_player.py [--ip IP] [--speed S] hand_segment_01.npz hand_segment_02.npz ...
    args = sys.argv[1:]
    ip = None
    speed = 1.0
    traj_paths = []
    for arg in args:
        if arg.startswith("--ip="):
            ip = arg.split("=",1)[1]
        elif arg.startswith("--speed="):
            speed = float(arg.split("=",1)[1])
        else:
            traj_paths.append(arg)

    if not traj_paths:
        print("Usage: python hand_player.py [--ip=IP] [--speed=S] <traj1.npz> [<traj2.npz> ...]")
        sys.exit(1)

    # 初始化 DDS
    if ip:
        ChannelFactoryInitialize(0, ip)
    else:
        ChannelFactoryInitialize(0)

    # 加载轨迹
    trajs = []
    for p in traj_paths:
        data = np.load(p)
        t = data["t"] - data["t"][0]
        q = data["q"]
        trajs.append({
            "name": Path(p).stem,
            "t": t,
            "q": q
        })

    print("=== Hand Player ===")
    for i, t in enumerate(trajs,1):
        print(f"  {i}. {t['name']} ({len(t['t'])} frames)")
    print(f"\n播放速度: {speed}x")
    print("按 ENTER 开始第 1 段播放；播放过程中按 Ctrl-C 可紧急退出。")
    print("播放完毕后可继续 ENTER 播放下一个，输入 Y+ENTER 提前退出。\n")

    # ---------------- 控制器 & 状态订阅 ----------------
    ctrl = Dex3_1_Controller(fps=100.0)
    time.sleep(0.5)  # 等几帧，让状态订阅稳定

    segment = 0
    try:
        while segment < len(trajs):
            input(f"[SEG {segment+1}] 按 ENTER 开始播放 “{trajs[segment]['name']}”…")
            print(f"[SEG {segment+1}] Start playing…")
            t_arr = trajs[segment]["t"] / speed
            q_arr = trajs[segment]["q"]
            start = time.time()
            idx = 0

            # 回放
            while True:
                now = time.time() - start
                # 快进到当前帧
                while idx < len(t_arr) and t_arr[idx] <= now:
                    idx += 1
                if idx >= len(t_arr):
                    break
                # 发送命令
                frame = q_arr[idx]
                left_q  = frame[:7]
                right_q = frame[7:]
                ctrl.set_target_q(left_q, right_q)
                time.sleep(1.0 / ctrl.fps)

            # 播放结束：保持最后一帧
            last = q_arr[-1]
            ctrl.set_target_q(last[:7], last[7:])
            print(f"[SEG {segment+1}] 播放完毕，保持最后姿态。\n")

            # 选择下一步
            ans = input("ENTER 播放下一段；输入 Y+ENTER 提前退出：").strip().lower()
            if ans == 'y':
                break
            segment += 1

    except KeyboardInterrupt:
        print("\n[!] KeyboardInterrupt，中断播放。")

    # ---------------- 阻尼 & 平滑复位 ----------------
    input("输入任意键并 ENTER 进入阻尼（零扭矩）模式…")
    ctrl.zero_torque()
    print("[DAMPING] 手部保持零扭矩中…")

    input("再次 ENTER 平滑回默认手势…")
    ctrl.move_to_default(duration=2.0)
    print("已恢复默认，退出。")

if __name__ == "__main__":
    main()
