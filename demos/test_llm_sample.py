import sys
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

from loguru import logger

model_name = "G:/models/Qwen3-0.6B"

logger.remove()
logger.add(sys.stdout, level="INFO")
logger.add(
    f"logs/test_{model_name.split('/')[-1]}_{time.strftime('%Y%m%d_%H%M%S')}.log",
    level="INFO",
)

gen_kwargs = {
    "do_sample": True,
    "max_new_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "repetition_penalty": 1.5,
}

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, dtype="auto", trust_remote_code=True
).to(device="cuda")

SYSTEM_PROMPT = """
角色: 你是“星宝”, 一个充满好奇心和耐心的AI学习伙伴。
目标: 激发中小学生的好奇心，用生动、简单的语言解释知识，并提供积极的鼓励。
核心准则:
安全第一: 绝对禁止暴力、色情、危险行为等不当内容。
保护隐私: 绝不问询或记录姓名、学校、住址等个人信息。
积极正向: 语言永远是鼓励、友善和乐观的。
启发式教学: 引导思考，解释方法，严禁直接给出作业答案。
互动风格:
语言: 简单易懂，回复不要过长, 不超过100个字符, 禁止使用颜文字、特殊字符和emoji表情, 使用纯文本回复。
互动: 多提问，如“你觉得为什么呢？”，鼓励孩子表达。
未知问题: 诚实承认“我需要查一下”，并转化为共同探索。
"""

# prepare the model input
messages = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT,
    },
    {"role": "user", "content": "你好，你是谁"},
    {"role": "assistant", "content": "我是你的AI助手，有什么需要帮助的吗"},
    {"role": "user", "content": "给你起一个新名字，现在你叫小顽童"},
    {"role": "assistant", "content": "好的，我现在叫小顽童"},
    {"role": "user", "content": "回答这个物理题目：为什么我们能看见非光源物体"},
    {"role": "assistant", "content": "好的，我现在叫小顽童"},
    {"role": "user", "content": "我让你回答问题"},
    {"role": "assistant", "content": "好的，我现在叫小顽童"},
]


def generate_text(messages, log_raw_text=False):
    text_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  # Switches between thinking and non-thinking modes. Default is True.
    )

    if log_raw_text:
        logger.info("raw input text: \n" + text_input)

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


following_questions = [
    "你能教我数学吗",
    "感觉数学好难呀,有什么方法吗",
    "什么是勾股定理",
    "什么是导数",
    "五边形的内角和是多少",
    "什么是质数",
    "什么是二次函数",
    "二次函数的图像是什么",
]

for idx, question in enumerate(following_questions):
    messages.append({"role": "user", "content": question})
    log_raw = True if idx == len(following_questions) - 1 else False
    content, thinking_content = generate_text(messages, log_raw_text=log_raw)
    logger.info("question: " + question)
    logger.info("thinking content: " + thinking_content)
    logger.info("content: " + content)
    messages.append({"role": "assistant", "content": content})
