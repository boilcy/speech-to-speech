import torch
from loguru import logger
from transformers import AutoModelForSpeechSeq2Seq

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "./models/whisper-large-v3-turbo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

dummy_input = torch.randn(
    (1, model.config.num_mel_bins, 3000),
    dtype=torch_dtype,
    device=device,
)
compile_mode = (
    None  # default, reduce-overhead, max-autotune, max-autotune-no-cudagraphs
)

gen_kwargs = {
    "max_new_tokens": 128,
    "num_beams": 1,
    "return_timestamps": False,
    "task": "transcribe",
}
if compile_mode:
    model.generation_config.cache_implementation = "static"
    model.forward = torch.compile(model.forward, mode=compile_mode, fullgraph=True)

if compile_mode not in (None, "default"):
    # generating more tokens than previously will trigger CUDA graphs capture
    # one should warmup with a number of generated tokens above max tokens targeted for subsequent generation
    # hence, having min_new_tokens < max_new_tokens in the future doesn't make sense
    warmup_gen_kwargs = {
        "min_new_tokens": gen_kwargs[
            "max_new_tokens"
        ],  # Yes, assign max_new_tokens to min_new_tokens
        "max_new_tokens": gen_kwargs["max_new_tokens"],
        **gen_kwargs,
    }
else:
    warmup_gen_kwargs = gen_kwargs

if device == "cuda:0":
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_event.record()

for _ in range(2):
    _ = model.generate(dummy_input, max_new_tokens=128)

if device == "cuda:0":
    end_event.record()
    torch.cuda.synchronize()

    logger.info(f"Warmed up! time: {start_event.elapsed_time(end_event) * 1e-3:.3f} s")
