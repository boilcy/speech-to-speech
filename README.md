# Speech To Speech

code base: https://github.com/huggingface/speech-to-speech

Simplify to only use whisper + kokoro + SmolVLM-256M

disable compile mode under windows platform, due to triton support

install pytorch before ready for nano super 



python3 -m pip install --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
    torch==2.8.0 torchvision==0.23.0 onnxruntime-gpu-1.23.0 pyceres-2.5 pycolmap-3.13.0.dev0 pymeshlab-2025.7 torchaudio-2.8.0


