import sys
import os

print(sys.path)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.misc import test_audio_devices
test_audio_devices()