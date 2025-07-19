#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chacha_motion.py ─ Python 3.8 safe
"""

import asyncio
import sys
from functools import partial

from chacha_voice_assistant_controller import ChaChaVoiceAssistantController
# from g1_high_level_controller import G1HighlevelArmController
# from unitree_sdk2py.core.channel import ChannelFactoryInitialize


async def main() -> None:
    # # ---------- G1 控制器初始化 ----------
    # ChannelFactoryInitialize(0, sys.argv[1] if len(sys.argv) > 1 else None)

    # ctrl = G1HighlevelArmController()
    # ctrl.start()                                  # 启动控制线程

    # loop = asyncio.get_running_loop()             # 3.8 可用

    # # 将阻塞式轮询放入默认线程池
    # loop.run_in_executor(                         # ctrl 语音按钮轮询
    #     None,
    #     ctrl.remote_poll_audio,
    # )

    # ---------- 语音助手初始化 ----------
    assistant = ChaChaVoiceAssistantController(
        debug=True,
        network_interface="eth0",
        interaction_timeout=120,
        force_audio_mode=None,                    # "g1" / "pyaudio" / None
    )

    # loop.run_in_executor(                         # assistant 自己的按钮轮询
    #     None,
    #     assistant.remote_poll,
    #     ctrl.remote.button,
    # )

    # 阻塞直到助手退出或超时
    await assistant.run()


if __name__ == "__main__":
    # Python 3.8 入口
    asyncio.run(main())
