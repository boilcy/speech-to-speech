from loguru import logger
import time

from nltk import sent_tokenize
from openai import OpenAI

from src.base import BaseHandler
from src.llm.chat import Chat


WHISPER_LANGUAGE_TO_LLM_LANGUAGE = {
    "en": "english",
    "fr": "french",
    "es": "spanish",
    "zh": "chinese",
    "ja": "japanese",
    "ko": "korean",
}


class OpenApiModelHandler(BaseHandler):
    """
    Handles the language model part.
    """

    def setup(
        self,
        model_name="deepseek-chat",
        device="cuda",
        gen_kwargs={},
        base_url=None,
        api_key=None,
        stream=False,
        user_role="user",
        chat_size=1,
        init_chat_role="system",
        init_chat_prompt="You are a helpful AI assistant.",
    ):
        self.model_name = model_name
        self.stream = stream
        self.chat = Chat(
            max_history_pairs=chat_size,
            max_tokens=2048,
            preserve_system=True,
            preserve_recent=1,
        )
        if init_chat_role:
            if not init_chat_prompt:
                raise ValueError(
                    "An initial promt needs to be specified when setting init_chat_role."
                )
            self.chat.init_chat({"role": init_chat_role, "content": init_chat_prompt})
        self.user_role = user_role
        self.client = OpenAI(api_key=api_key, base_url=base_url)

        self.warmup()

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")
        start = time.time()
        warmup_messages = [{"role": "user", "content": "Hello"}]
        logger.debug(f"Warmup messages: {warmup_messages}")
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=warmup_messages,
            stream=self.stream,
        )
        if self.stream:
            # Consume the stream for warmup
            warmup_response = ""
            for chunk in response:
                warmup_response += chunk.choices[0].delta.content or ""
        else:
            warmup_response = response.choices[0].message.content
        logger.debug(f"Warmup response: '{warmup_response}'")
        end = time.time()
        logger.info(
            f"{self.__class__.__name__}:  warmed up! time: {(end - start):.3f} s"
        )

    def process(self, prompt):
        logger.debug("call api language model...")

        # Log original input
        logger.debug(f"Raw input prompt: {prompt}")

        language_code = None
        if isinstance(prompt, tuple):
            prompt, language_code = prompt
            logger.debug(
                f"Tuple input detected - prompt: '{prompt}', language_code: '{language_code}'"
            )
            if language_code[-5:] == "-auto":
                language_code = language_code[:-5]
                prompt = (
                    f"Please reply to my message in {WHISPER_LANGUAGE_TO_LLM_LANGUAGE[language_code]}. "
                    + prompt
                )
                logger.debug(
                    f"Language auto-detection processed - final prompt: '{prompt}', language_code: '{language_code}'"
                )

        self.chat.append({"role": self.user_role, "content": prompt})
        # Log the messages being sent to the model
        messages = self.chat.to_list()
        logger.info(f"Messages sent to model: {messages}")

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=self.stream,
        )
        if self.stream:
            logger.debug("Processing streaming response...")
            generated_text, printable_text = "", ""
            chunk_count = 0
            for chunk in response:
                chunk_count += 1
                new_text = chunk.choices[0].delta.content or ""
                logger.debug(
                    f"Chunk {chunk_count}: '{new_text}', '{chunk.choices[0].logprobs}'"
                )
                generated_text += new_text
                printable_text += new_text
                sentences = sent_tokenize(printable_text)
                if len(sentences) > 1:
                    printable_text = new_text
                    yield sentences[0], language_code
            self.chat.append({"role": "assistant", "content": generated_text})
            # don't forget last sentence
            logger.debug(f"Yielding final sentence: '{printable_text}'")
            yield printable_text, language_code
        else:
            logger.debug("Processing non-streaming response...")
            generated_text = response.choices[0].message.content
            logger.debug(f"Raw output text (complete): '{generated_text}'")
            self.chat.append({"role": "assistant", "content": generated_text})
            logger.debug(f"Yielding response: '{generated_text}'")
            yield generated_text, language_code
