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
    ctrl = Dex3_1_Controller(fps=100.0)
    time.sleep(0.5)  # 等几帧，让状态订阅稳定

    print("=== Hand Recorder ===")
    print("按 ENTER 开始/停止每段录制；")
    print("录制过程中手部处于零扭矩模式，方便手动摆姿势。")
    print("录完一段会保存为 hand_segment_XX.npz。")
    print("完成所有录制后，输入 Y（然后 ENTER）退出；")
    print("然后 ENTER 进入阻尼(零扭矩)，再 ENTER 平滑回默认。\n")

    segment_idx = 1
    try:
        while True:
            # 等待 ENTER 开始
            input(f"[SEG {segment_idx}] 按 ENTER 开始录制…")
            print(f"[SEG {segment_idx}] 开始录制，按 ENTER 停止")

            # 切换到录制模式：零扭矩
            ctrl.recording = True

            t0 = time.time()
            t_buf = []
            q_buf = []

            # 采样循环
            while True:
                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    if sys.stdin.readline():
                        break
                left_q, right_q = ctrl.get_current_q()
                left_q = left_q.round(3)
                right_q = right_q.round(3)
                # print(f"Current Left Q: {left_q} \nCurrent Right Q: {right_q}")
                t_buf.append(time.time() - t0)
                q_buf.append(np.concatenate([left_q, right_q]))
                time.sleep(1.0 / ctrl.fps)

            # 停止录制
            ctrl.recording = False
            print(f"[SEG {segment_idx}] 录制停止，存盘…")

            # 保存
            fname = Path(f"hand_segment_{segment_idx:02d}.npz")
            np.savez_compressed(
                str(fname),
                t=np.array(t_buf),
                q=np.vstack(q_buf),
            )
            print(f"[SEG {segment_idx}] 已保存 {fname} 共 {len(t_buf)} 帧\n")
            segment_idx += 1

            ans = input("按 ENTER 继续下一段，或输入 Y (ENTER) 退出录制：").strip().lower()
            if ans == 'y':
                break

    except KeyboardInterrupt:
        print("\n[!] KeyboardInterrupt，中断录制。")

    # 阻尼(零扭矩)
    input("输入任意键并 ENTER 进入阻尼（零扭矩）模式…")
    ctrl.zero_torque()
    print("[DAMPING] 手部零扭矩保持中…")

    # 平滑回默认手势
    input("再次 ENTER 平滑回默认手势…")
    ctrl.move_to_default(duration=2.0)
    print("已恢复默认手势，退出。")

if __name__ == '__main__':
    # # 如果需要指定 IP 就放在 argv[1]
    # # e.g. python hand_recorder.py enp2s0
    # from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    # if len(sys.argv) > 1:
    #     ChannelFactoryInitialize(0, sys.argv[1])
    # else:
    #     ChannelFactoryInitialize(0)

    main()
