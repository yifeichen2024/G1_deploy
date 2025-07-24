import sys
import time
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
from unitree_sdk2py.utils.thread import RecurrentThread
from deploy_real.common.remote_controller import RemoteController, KeyMap
from wav import read_wav

class AudioPlayerWithRemote:
    def __init__(self, net_interface, wav_path):
        # Initialize DDS and audio client
        ChannelFactoryInitialize(0, net_interface)
        self.audioClient = AudioClient()
        self.audioClient.SetTimeout(10.0)
        self.audioClient.Init()

        ret = self.audio_client.GetVolume()
        print("debug GetVolume: ",ret)

        # Initialize remote controller
        self.remote = RemoteController()
        self.prev_buttons = [0] * len(self.remote.button)

        # Load WAV file
        self.pcm_list, self.sample_rate, self.num_channels, self.is_ok = read_wav(wav_path)
        print(f"[DEBUG] Read success: {self.is_ok}")
        print(f"[DEBUG] Sample rate: {self.sample_rate} Hz")
        print(f"[DEBUG] Channels: {self.num_channels}")
        print(f"[DEBUG] PCM byte length: {len(self.pcm_list)}")

        if not self.is_ok or self.sample_rate != 16000 or self.num_channels != 1:
            print("[ERROR] WAV format must be 16kHz mono")
            raise SystemExit

        self.name = "remote_music"
        self.playing = False
        self.volume = 80.0  # Default volume (1.0 = 100%)

    def poll_remote(self):
        buttons = self.remote.button

        # Only trigger on rising edge (i.e., when button goes from 0 to 1)
        def pressed_once(key):
            return buttons[key] == 1 and self.prev_buttons[key] == 0

        if pressed_once(KeyMap.up) and not self.playing:
            print("[AUDIO] Start playing")
            self.audioClient.SetAudio(self.name, self.pcm_list)
            self.audioClient.Play(self.name)
            self.playing = True

        if pressed_once(KeyMap.down) and self.playing:
            print("[AUDIO] Stop playing")
            self.audioClient.PlayStop(self.name)
            self.playing = False

        # Adjust volume up (right)
        if pressed_once(KeyMap.right):
            self.volume = min(100.0, self.volume + 10)
            self.audioClient.SetVolume(self.volume)
            print(f"[AUDIO] Volume increased to {self.volume:.1f}")

        # Adjust volume down (left)
        if pressed_once(KeyMap.left):
            self.volume = max(0.0, self.volume - 10)

            self.audioClient.SetVolume(self.volume)
            print(f"[AUDIO] Volume decreased to {self.volume:.1f}")

        # Safe exit with select
        if pressed_once(KeyMap.select):
            print("[AUDIO] Stopping and exiting.")
            self.audioClient.PlayStop(self.name)
            sys.exit(0)

        self.prev_buttons = buttons[:]

def main():
    if len(sys.argv) < 3:
        print(f"Usage: python3 {sys.argv[0]} <network_interface> <wav_file_path>")
        sys.exit(1)

    player = AudioPlayerWithRemote(sys.argv[1], sys.argv[2])
    print("[INIT] Ready. Use Up to Play, Down to Stop, Left/Right to Adjust Volume.")

    try:
        while True:
            player.poll_remote()
            time.sleep(0.02)
    except KeyboardInterrupt:
        player.audioClient.PlayStop(player.name)
        print("\n[EXIT] Audio playback stopped.")

if __name__ == "__main__":
    main()
