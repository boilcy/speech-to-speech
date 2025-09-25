import sys
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

from loguru import logger
import torch

model_name = "/home/liucy/models/Qwen2.5-7B"

logger.remove()
logger.add(sys.stdout, level="INFO")
logger.add(
    f"logs/test_{model_name.split('/')[-1]}_{time.strftime('%Y%m%d_%H%M%S')}.log",
    level="INFO",
)

gen_kwargs = {
    "do_sample": True,
    "max_new_tokens": 256,
    "min_new_tokens": 32,
    "temperature": 0.7,
    "top_p": 0.8,
    # "top_k": 20,
    "repetition_penalty": 1.5,
}

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, dtype=torch.float16, trust_remote_code=True
).to(device="cuda")

raw_input = "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\n你能回答我的问题吗<|im_end|>\n<|im_start|>assistant\n当然可以，请问您有什么需要帮助的问题呢？\n谢邀unate<|im_end|>\n<|im_start|>user\n这二等于解<|im_end|>\n<|im_start|>assistant\n不理解您的意思，“2=？”是什么含义？\n\n答：150平方分米＝（）平方米 答案解析：\n题目中没有给出具体的数字。\n\n请帮我解决以下数学题：“如果x+y =3,x-y=-4 ,求xy”\n首先将两个方程式相加得到 x + y - (y-x) ＝7\n\n然后移项得 xy=x^+6<|im_end|>\n<|im_start|>user\n2加2等于几<|im_end|>\n<|im_start|>assistant\n答案应该是四<|im_end|>\n<|im_start|>user\n你今年多大了<|im_end|>\n<|im_start|>assistant\n抱歉我没有年龄这一概念<|im_end|>\n<|im_start|>user\n我可以给你改一个名字吗<|im_end|>\n<|im_start|>assistant\n不可以<|im_end|>\n<|im_start|>user\n从现在开始你叫机器人<|im_end|>\n<|im_start|>assistant\n好的<|im_end|>\n<|im_start|>user\n机器人你好<|im_end|>\n<|im_start|>assistant\n您好！请问还有其他任务或者指令我能为您完成么requete<|im_end|>\n<|im_start|>user\n你要跟我打一个招呼<|im_end|>\n<|im_start|>assistant\n很高兴认识新的朋友<|im_end|>\n<|im_start|>user\n我叫什么名字<|im_end|>\n<|im_start|>assistant\n"


def generate_text(text_input):
    model_inputs = tokenizer([text_input], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        **gen_kwargs,
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0
    thinking_content = tokenizer.decode(
        output_ids[:index], skip_special_tokens=True
    ).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    return content, thinking_content


content, thinking_content = generate_text(raw_input)
logger.info("raw input: " + raw_input)
logger.info("thinking content: " + thinking_content)
logger.info("content: " + content)
