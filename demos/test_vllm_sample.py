"""
vLLM 测试脚本 - 使用 OpenAI API 兼容接口

使用前请先启动 vLLM 服务:
python -m vllm.entrypoints.openai.api_server \
    --model G:/models/Qwen3-0.6B \
    --served-model-name Qwen3-0.6B \
    --host localhost \
    --port 8000 \
    --trust-remote-code
"""

import sys
import time
from openai import OpenAI
import re
from loguru import logger

# Unicode ranges for CJK Unified Ideographs (Basic + Ext A–H) and Compat Ideographs
_HAN_RANGES = (
    r"\u3400-\u4DBF"  # CJK Unified Ideographs Extension A
    r"\u4E00-\u9FFF"  # CJK Unified Ideographs (Basic)
    r"\uF900-\uFAFF"  # CJK Compatibility Ideographs
    r"\U0002F800-\U0002FA1F"  # CJK Compatibility Ideographs Supplement
    r"\U00020000-\U0002A6DF"  # Extension B
    r"\U0002A700-\U0002B73F"  # Extension C
    r"\U0002B740-\U0002B81F"  # Extension D
    r"\U0002B820-\U0002CEAF"  # Extension E
    r"\U0002CEB0-\U0002EBEF"  # Extension F
    r"\U00030000-\U0003134F"  # Extension G
    r"\U00031350-\U000323AF"  # Extension H
)

# Optional: include variation selectors to preserve ideographic variants
_VARIATION_SELECTORS = r"\uFE00-\uFE0F\U000E0100-\U000E01EF"

# Common Chinese punctuation (tweak as needed)
_CN_PUNCT = "，。！？、；：—「」『』（）〔〕【】《》〈〉·…．“”‘’︰、～"


def keep_chinese_stdlib(
    text: str,
    keep_punct: bool = False,
    keep_space: bool = False,
    keep_variation_selectors: bool = True,
) -> str:
    """
    Keep only Chinese characters (CJK Unified Ideographs; Ext A–H; Compatibility),
    optionally keeping common CJK punctuation, whitespace, and variation selectors.
    """
    allowed_ranges = _HAN_RANGES
    if keep_variation_selectors:
        allowed_ranges += _VARIATION_SELECTORS

    allowed_class = f"{allowed_ranges}"
    if keep_punct:
        allowed_class += re.escape(_CN_PUNCT)
    if keep_space:
        allowed_class += r"\s"

    allowed_class += r"[0-9]"

    # Remove everything NOT in the allowed set
    return re.sub(rf"[^{allowed_class}]+", "", text)


# vLLM 服务配置
model_name = "/root/models/Llama-3-8B-Instruct/"  # 这是在 vLLM 服务中注册的模型名
vllm_api_base = "http://127.0.0.1:8000/v1"  # vLLM 服务地址
vllm_api_key = "EMPTY"  # vLLM 不需要真实的 API 密钥

logger.remove()
logger.add(sys.stdout, level="INFO")
logger.add(f"logs/test_{model_name}_{time.strftime('%Y%m%d_%H%M%S')}.log", level="INFO")

# OpenAI API 生成参数
gen_kwargs = {
    "max_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.8,
    "frequency_penalty": 1.2,
}

# 初始化 OpenAI 客户端连接到 vLLM 服务
client = OpenAI(
    api_key=vllm_api_key,
    base_url=vllm_api_base,
)

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
    """
    使用 vLLM 的 OpenAI API 兼容接口生成文本

    Args:
        messages: 对话消息列表
        log_raw_text: 是否记录原始输入文本

    Returns:
        tuple: (生成的内容, 思考内容)
    """
    if log_raw_text:
        logger.info("messages input: \n" + str(messages))

    try:
        # 使用 OpenAI API 格式调用 vLLM 服务
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            **gen_kwargs,
        )

        # 提取生成的内容
        content = response.choices[0].message.content

        # vLLM 通过 OpenAI API 不直接支持思考模式
        # 这里返回空字符串作为思考内容
        thinking_content = ""

        return content, thinking_content

    except Exception as e:
        logger.error(f"生成文本时发生错误: {e}")
        return "抱歉，我现在无法回答这个问题。", ""


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

# 测试 vLLM 服务连接
logger.info("开始测试 vLLM 服务连接...")
logger.info(f"服务地址: {vllm_api_base}")
logger.info(f"模型名称: {model_name}")

for idx, question in enumerate(following_questions):
    logger.info(f"处理问题 {idx + 1}/{len(following_questions)}: {question}")

    messages.append({"role": "user", "content": question})
    log_raw = True if idx == len(following_questions) - 1 else False

    start_time = time.time()
    content, thinking_content = generate_text(messages, log_raw_text=log_raw)
    logger.info(f"raw content: {content}")
    content = keep_chinese_stdlib(content, keep_punct=True, keep_space=True)
    end_time = time.time()

    logger.info(f"问题: {question}")
    logger.info(f"思考内容: {thinking_content}")
    logger.info(f"回答: {content}")
    logger.info(f"生成时间: {end_time - start_time:.2f}秒")
    logger.info("=" * 50)

    messages.append({"role": "assistant", "content": content})

logger.info("所有问题处理完成!")
