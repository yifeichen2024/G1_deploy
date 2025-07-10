#!/usr/bin/env python3
import time
import sys
import select
import numpy as np
from pathlib import Path

# 假设你的 Dex3_1_Controller 定义在 common/dex3_controller.py
from g1_highlevel_hand import Dex3_1_Controller
from unitree_sdk2py.utils.crc import CRC

def main():
    # 1) 初始化手部控制器
    ctrl = Dex3_1_Controller(fps=100.0)
    time.sleep(0.5)  # 等几帧，让订阅器先跑起来
    
    print("=== Hand Recorder ===")
    print("按 ENTER 开始/停止每段录制。")
    print("录制完一段后会自动保存为 hand_segment_XX.npz。")
    print("全部录制完成后，输入 Y（然后 ENTER）退出录制，进入阻尼模式。")
    print("阻尼完成后，输入 X（然后 ENTER）再平滑回默认手势。\n")
    
    segment_idx = 1
    try:
        while True:
            # 等待 ENTER 开始一段录制
            input(f"[SEG {segment_idx}] 按 ENTER 开始录制…")
            print(f"[SEG {segment_idx}] Recording… 再次 ENTER 停止")
            t0 = time.time()
            t_buf = []
            q_buf = []
            # 连续采样
            while True:
                # 检查键盘
                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    if sys.stdin.readline():
                        break
                # 读取当前手部关节
                left_q, right_q = ctrl.get_current_q()
                q_buf.append(np.concatenate([left_q, right_q]))
                t_buf.append(time.time() - t0)
                time.sleep(1.0 / ctrl.fps)
            # 保存这一段
            fname = Path(f"hand_segment_{segment_idx:02d}.npz")
            np.savez_compressed(str(fname),
                                t=np.array(t_buf),
                                q=np.vstack(q_buf))
            print(f"[SEG {segment_idx}] Saved {fname} ({len(t_buf)} frames)\n")
            segment_idx += 1

            # 询问是否继续
            ans = input("按 ENTER 继续下一段，或输入 Y 退出录制：").strip().lower()
            if ans == 'y':
                break

    except KeyboardInterrupt:
        print("\n[!] keyboard interrupt, finishing…")

    # 2) 进入阻尼模式
    input("输入任意键并 ENTER 进入阻尼模式…")
    print("[DAMPING] zero torque on hands")
    ctrl.zero_torque()

    # 3) 平滑回默认手势
    input("阻尼中…输入任意键并 ENTER 平滑回默认手势…")
    print("[DEFAULT] moving to default hand pose")
    ctrl.move_to_default(duration=2.0)

    print("Done. 退出。")

if __name__ == '__main__':
    main()