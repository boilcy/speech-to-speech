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
        self.gen_kwargs = {
            "streamer": self.streamer,
            "return_full_text": False,
            **gen_kwargs,
        }
        # using tokenizer_encode_kwargs require always input chats instead of text
        if "Qwen3" in model_name:
            self.gen_kwargs["tokenizer_encode_kwargs"] = {
                "enable_thinking": False,
            }

        self.chat = Chat(chat_size)
        if init_chat_role:
            if not init_chat_prompt:
                raise ValueError(
                    "An initial promt needs to be specified when setting init_chat_role."
                )
            self.chat.init_chat({"role": init_chat_role, "content": init_chat_prompt})
        self.user_role = user_role

        self.warmup()

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")

        dummy_input_text = "Repeat the word 'home'."
        dummy_chat = [{"role": self.user_role, "content": dummy_input_text}]
        warmup_gen_kwargs = {
            "min_new_tokens": self.gen_kwargs["min_new_tokens"],
            "max_new_tokens": self.gen_kwargs["max_new_tokens"],
            **self.gen_kwargs,
        }

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
        thread = Thread(
            target=self.pipe, args=(self.chat.to_list(),), kwargs=self.gen_kwargs
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
