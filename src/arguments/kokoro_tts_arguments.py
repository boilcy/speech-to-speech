from dataclasses import dataclass, field


@dataclass
class KokoroTTSHandlerArguments:
    tts_model_name: str = field(
        default="hexgrad/Kokoro-82M-v1.1-zh",
        metadata={
            "help": "The pretrained TTS model to use. Default is 'hexgrad/Kokoro-82M-v1.1-zh'."
        },
    )
    tts_device: str = field(
        default="cuda",
        metadata={
            "help": "The device type on which the model will run. Default is 'cuda' for GPU acceleration."
        },
    )
    tts_torch_dtype: str = field(
        default="float16",
        metadata={
            "help": "The PyTorch data type for the model and input tensors. One of `float32` (full-precision), `float16` or `bfloat16` (both half-precision)."
        },
    )
    tts_compile_mode: str = field(
        default=None,
        metadata={
            "help": "Compile mode for torch compile (Linux only). Either 'default', 'reduce-overhead' and 'max-autotune'. Default is None (no compilation)"
        },
    )

    use_default_voice: bool = field(
        default=True,
        metadata={"help": "Whether to use the default list of speakers or not."},
    )
