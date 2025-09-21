from threading import Thread
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    TextIteratorStreamer,
)
import torch

from loguru import logger
from nltk import sent_tokenize

from src.llm.chat import Chat
from src.base import BaseHandler

WHISPER_LANGUAGE_TO_LLM_LANGUAGE = {
    "en": "english",
    "fr": "french",
    "es": "spanish",
    "zh": "chinese",
    "ja": "japanese",
    "ko": "korean",
    "hi": "hindi",
    "de": "german",
    "pt": "portuguese",
    "pl": "polish",
    "it": "italian",
    "nl": "dutch",
}


class LanguageModelHandler(BaseHandler):
    """
    Handles the language model part.
    """

    def setup(
        self,
        model_name="microsoft/Phi-3-mini-4k-instruct",
        device="cuda",
        torch_dtype="float16",
        gen_kwargs={},
        user_role="user",
        chat_size=1,
        init_chat_role=None,
        init_chat_prompt="You are a helpful AI assistant.",
        max_tokens=None,
        preserve_recent=1,
    ):
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype)

        logger.info(
            f"Loading model {model_name} on {device} with torch_dtype {torch_dtype}"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch_dtype, trust_remote_code=True
        ).to(device)
        self.pipe = pipeline(
            "text-generation", model=self.model, tokenizer=self.tokenizer, device=device
        )
        self.streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        # Filter out None values from gen_kwargs to avoid overriding model defaults
        filtered_gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
        self.gen_kwargs = {
            "streamer": self.streamer,
            "return_full_text": False,
            **filtered_gen_kwargs,
        }
        # using tokenizer_encode_kwargs require always input chats instead of text
        if "Qwen3" in model_name:
            self.gen_kwargs["tokenizer_encode_kwargs"] = {
                "enable_thinking": False,
            }

        # 使用改进的Chat类
        self.chat = Chat(
            max_history_pairs=chat_size,
            max_tokens=max_tokens,
            preserve_system=True,
            preserve_recent=preserve_recent
        )
        
        # 设置系统消息
        if init_chat_role:
            if not init_chat_prompt:
                raise ValueError(
                    "An initial prompt needs to be specified when setting init_chat_role."
                )
            self.chat.init_chat({"role": init_chat_role, "content": init_chat_prompt})
        elif init_chat_prompt:
            self.chat.init_chat({"role": "system", "content": init_chat_prompt})
        self.user_role = user_role

        self.warmup()

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")

        dummy_input_text = "Repeat the word 'home'."
        dummy_chat = [{"role": self.user_role, "content": dummy_input_text}]
        warmup_gen_kwargs = {
            **self.gen_kwargs,
        }
        if "min_new_tokens" not in warmup_gen_kwargs:
            warmup_gen_kwargs["min_new_tokens"] = 1
        if "max_new_tokens" not in warmup_gen_kwargs:
            warmup_gen_kwargs["max_new_tokens"] = 10

        n_steps = 2

        if self.device == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()

        for _ in range(n_steps):
            thread = Thread(
                target=self.pipe, args=(dummy_chat,), kwargs=warmup_gen_kwargs
            )
            thread.start()
            for _ in self.streamer:
                pass

        if self.device == "cuda":
            end_event.record()
            torch.cuda.synchronize()

            logger.info(
                f"{self.__class__.__name__}:  warmed up! time: {start_event.elapsed_time(end_event) * 1e-3:.3f} s"
            )

    def process(self, prompt):
        logger.debug("infering language model...")
        language_code = None
        if isinstance(prompt, tuple):
            prompt, language_code = prompt
            if language_code[-5:] == "-auto":
                language_code = language_code[:-5]
                prompt = (
                    f"Please reply to my message in {WHISPER_LANGUAGE_TO_LLM_LANGUAGE[language_code] if language_code in WHISPER_LANGUAGE_TO_LLM_LANGUAGE else 'chinese'}. "
                    + prompt
                )

        self.chat.append({"role": self.user_role, "content": prompt})
        chat_list = self.chat.to_list()
        logger.debug(f"Feeding chat list: {chat_list}")
        thread = Thread(
            target=self.pipe, args=(chat_list,), kwargs=self.gen_kwargs
        )
        thread.start()
        if self.device == "mps":
            generated_text = ""
            for new_text in self.streamer:
                generated_text += new_text
            printable_text = generated_text
            torch.mps.empty_cache()
        else:
            generated_text, printable_text = "", ""
            for new_text in self.streamer:
                generated_text += new_text
                printable_text += new_text
                sentences = sent_tokenize(printable_text)
                if len(sentences) > 1:
                    yield (sentences[0], language_code)
                    printable_text = new_text

        self.chat.append({"role": "assistant", "content": generated_text})

        # don't forget last sentence
        yield (printable_text, language_code)
