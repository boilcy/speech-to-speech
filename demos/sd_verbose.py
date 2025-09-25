import sounddevice as sd
import ctypes.util

print("sounddevice version:", sd.__version__)
print("lib used:", sd._libname)
print("ctypes found:", ctypes.util.find_library("portaudio"))

host_apis = sd.query_hostapis()
print("---host apis---")
for i, api in enumerate(host_apis):
    print(f"{i}: {api['name']}")

devices = sd.query_devices()
print("devices:", devices)

default_device = sd.default.device
print("default devices:", default_device)
