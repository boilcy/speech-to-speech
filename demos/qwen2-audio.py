from io import BytesIO
from urllib.request import urlopen
import librosa
from transformers import AutoModelForCausalLM, AutoProcessor

# Use Qwen2.5-Omni-3B instead of local Qwen2-Audio
model_name = "Qwen/Qwen2.5-Omni-3B"

# Load processor
processor = AutoProcessor.from_pretrained(model_name)

# Load model with 4-bit quantization (requires bitsandbytes installed)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # automatically place on GPU
    load_in_4bit=True,  # enable 4-bit quantization
)

# Example conversation (still using audio URLs, but Qwen2.5-Omni is multimodal)
conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "audio",
                "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/guess_age_gender.wav",
            },
        ],
    },
    {"role": "assistant", "content": "Yes, the speaker is female and in her twenties."},
    {
        "role": "user",
        "content": [
            {
                "type": "audio",
                "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/translate_to_chinese.wav",
            },
        ],
    },
]

# Convert conversation into text prompt
text = processor.apply_chat_template(
    conversation, add_generation_prompt=True, tokenize=False
)

# Collect audio inputs
audios = []
for message in conversation:
    if isinstance(message["content"], list):
        for ele in message["content"]:
            if ele["type"] == "audio":
                wav, _ = librosa.load(
                    BytesIO(urlopen(ele["audio_url"]).read()),
                    sr=processor.feature_extractor.sampling_rate,
                )
                audios.append(wav)

# Preprocess inputs
inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True).to(
    model.device
)

# Generate response
generate_ids = model.generate(**inputs, max_new_tokens=256)
generate_ids = generate_ids[:, inputs.input_ids.size(1) :]  # strip input tokens

response = processor.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]

print(response)
