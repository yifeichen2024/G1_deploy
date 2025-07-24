import sys
import time
import threading
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from deploy_real.common.remote_controller import RemoteController, KeyMap
from wav import read_wav

class AudioRemotePlayer:
    def __init__(self, net_interface, wav_path):
        # 初始化 DDS 通信
        ChannelFactoryInitialize(0, net_interface)
        print("[DDS] Initialized")

        # 初始化音频客户端
        self.audio = AudioClient()
        self.audio.SetTimeout(10.0)
        self.audio.Init()
        print("[AudioClient] Initialized")
        ret = self.audio.GetVolume()
        print(f"[DEBUG] Volume: {ret}")
        
        # 遥控器订阅 LowState
        self.remote = RemoteController()
        self.sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub.Init(self._lowstate_callback, 10)
        print("[Remote] Subscribed to rt/lowstate")

        # 读取 WAV 数据
        self.pcm_list, self.sample_rate, self.num_channels, self.valid = read_wav(wav_path)
        if not self.valid or self.sample_rate != 16000 or self.num_channels != 1:
            raise RuntimeError("WAV must be 16kHz mono PCM")

        # 0.1s 每块，提高响应速度
        self.chunk_size = self.sample_rate * self.num_channels * 2 // 100
        self.sleep_time = 0.1

        self.playing = False
        self.stop_requested = False
        self.name = "remote_music"
        self.prev_buttons = [0] * len(self.remote.button)
        self.play_thread = None

    def _lowstate_callback(self, msg):
        # 在低层状态回调中更新遥控器按键值
        self.remote.set(msg.wireless_remote)

    def _play_loop(self):
        pcm_bytes = bytes(self.pcm_list)
        offset = 0
        total = len(pcm_bytes)
        # 唯一流 ID
        stream_id = str(int(time.time() * 1000))
        while offset < total and not self.stop_requested:
            end = offset + self.chunk_size
            chunk = pcm_bytes[offset:end]
            ret, _ = self.audio.PlayStream(self.name, stream_id, chunk)
            if ret != 0:
                print(f"[ERROR] chunk send failed, code={ret}")
                break
            offset = end
            time.sleep(self.sleep_time)
        # 停止播放
        self.audio.PlayStop(self.name)
        self.playing = False
        self.stop_requested = False

    def run(self):
        print("[READY] Press UP to play, DOWN to stop.")
        try:
            while True:
                buttons = self.remote.button
                # 上升沿检测
                def pressed(k): return buttons[k] == 1 and self.prev_buttons[k] == 0

                if pressed(KeyMap.up) and not self.playing:
                    print("[AUDIO] Start playing...")
                    self.playing = True
                    self.stop_requested = False
                    self.play_thread = threading.Thread(target=self._play_loop, daemon=True)
                    self.play_thread.start()

                if pressed(KeyMap.down) and self.playing:
                    print("[AUDIO] Stop requested.")
                    self.stop_requested = True
                    
                if pressed(KeyMap.left):
                    current = self.audio.GetVolume()[1]["volume"]
                    new_volume = max(0, current - 10)
                    self.audio.SetVolume(new_volume)
                    print(f"[AUDIO] Volume Down: {new_volume}")

                if pressed(KeyMap.right):
                    current = self.audio.GetVolume()[1]["volume"]
                    new_volume = min(100, current + 10)
                    self.audio.SetVolume(new_volume)
                    print(f"[AUDIO] Volume Up: {new_volume}")
                    
                self.prev_buttons[:] = buttons[:]
                time.sleep(0.02)
        except KeyboardInterrupt:
            self.stop_requested = True
            self.audio.PlayStop(self.name)
            print("\n[EXIT] Audio stopped.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: python3 {sys.argv[0]} <network_interface> <wav_file_path>")
        sys.exit(1)

    app = AudioRemotePlayer(sys.argv[1], sys.argv[2])
    app.run()
