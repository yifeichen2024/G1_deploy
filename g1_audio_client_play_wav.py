import sys
import time
import threading
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from deploy_real.common.remote_controller import RemoteController, KeyMap
from wav import read_wav, play_pcm_stream

class AudioRemotePlayer:
    def __init__(self, net_interface, wav_path):
        # Init DDS
        ChannelFactoryInitialize(0, net_interface)
        print("[DDS] Initialized")

        # Init audio
        self.audio = AudioClient()
        self.audio.SetTimeout(10.0)
        self.audio.Init()
        print("[AudioClient] Initialized")
        ret = self.audio.GetVolume()
        print(f"[DEBUG] Volume: {ret}")
        
        # Init remote controller + subscriber
        self.remote = RemoteController()
        self.low_state_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.low_state_sub.Init(self._lowstate_callback, 10)
        print("[Remote] Subscribed to rt/lowstate")

        # Load WAV
        self.pcm_list, self.sample_rate, self.num_channels, self.valid = read_wav(wav_path)
        if not self.valid or self.sample_rate != 16000 or self.num_channels != 1:
            raise RuntimeError("WAV must be 16kHz mono")

        self.playing = False
        self.stop_requested = False
        self.name = "remote_music"
        self.prev_buttons = [0] * 16
        self.play_thread = None

    def _lowstate_callback(self, msg):
        self.remote.set(msg.wireless_remote)

    def _play_loop(self):
        self.audio.PlayStop(self.name)  # Ensure clean start
        self.stop_requested = False
        frame_size = 640  # Send 640 bytes at a time
        for i in range(0, len(self.pcm_list), frame_size):
            if self.stop_requested:
                print("[AUDIO] Interrupted")
                break
            chunk = self.pcm_list[i:i+frame_size]
            self.audio.PlayAudioStream(self.name, chunk)
            time.sleep(0.02)  # Match audio rate
        self.playing = False
        self.stop_requested = False

    def run(self):
        print("[READY] Press UP to play, DOWN to stop.")
        try:
            while True:
                buttons = self.remote.button

                def pressed(key):
                    return buttons[key] == 1 and self.prev_buttons[key] == 0

                if pressed(KeyMap.up) and not self.playing:
                    print("[AUDIO] Playing…")
                    self.playing = True
                    self.play_thread = threading.Thread(target=self._play_loop, daemon=True)
                    self.play_thread.start()

                if pressed(KeyMap.down) and self.playing:
                    print("[AUDIO] Stopping…")
                    self.stop_requested = True
                    self.audio.PlayStop(self.name)

                self.prev_buttons = buttons[:]
                time.sleep(0.02)
        except KeyboardInterrupt:
            self.audio.PlayStop(self.name)
            print("\n[EXIT] Stopped")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: python3 {sys.argv[0]} <network_interface> <wav_file_path>")
        sys.exit(1)

    app = AudioRemotePlayer(sys.argv[1], sys.argv[2])
    app.run()
