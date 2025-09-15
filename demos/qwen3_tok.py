from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "models/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
)
print(text)

from transformers import pipeline

model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device="cuda",
)
print(pipe(messages, tokenizer_encode_kwargs={"enable_thinking": False}))