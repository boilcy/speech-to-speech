import sys
import os

print(sys.path)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sounddevice as sd

print(sd.query_devices())
sd.default.device = 0, 25

from src.utils.misc import test_audio_devices
test_audio_devices()
