import sys
import time
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from deploy_real.common.remote_controller import RemoteController, KeyMap
from wav import read_wav


class AudioRemotePlayer:
    def __init__(self, net_interface, wav_path):
        # Init DDS
        ChannelFactoryInitialize(0, net_interface)
        print("[DDS] Initialized")

        # Init audio
        self.audio = AudioClient()
        self.audio.SetTimeout(10.0)
        ret = self.audioClient.GetVolume()
        print("[DEBUG] GetVolume: ",ret)
        ret = self.audioClient.SetVolume(60.0)
        print("[DEBUG] SetVolume: ",ret)
        self.audio.Init()
        print("[AudioClient] Initialized")

        # Init remote controller + subscriber
        self.remote = RemoteController()
        self.low_state_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.low_state_sub.Init(self._lowstate_callback, 10)
        print("[Remote] Subscribed to rt/lowstate")

        # Load WAV
        self.wav_data, self.sample_rate, self.num_channels, self.valid = read_wav(wav_path)
        if not self.valid or self.sample_rate != 16000 or self.num_channels != 1:
            raise RuntimeError("WAV must be 16kHz mono")

        self.audio.SetAudio("music", self.wav_data)
        self.playing = False
        self.name = "music"

        self.prev_buttons = [0] * 16  # Enough for all buttons

    def _lowstate_callback(self, msg):
        self.remote.set(msg.wireless_remote)

    def run(self):
        print("[READY] Press UP to play, DOWN to stop.")
        try:
            while True:
                buttons = self.remote.button

                def pressed(key):
                    return buttons[key] == 1 and self.prev_buttons[key] == 0

                if pressed(KeyMap.up):
                    print("[AUDIO] Play")
                    self.audio.Play(self.name)
                    self.playing = True

                if pressed(KeyMap.down):
                    print("[AUDIO] Stop")
                    self.audio.PlayStop(self.name)
                    self.playing = False

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
