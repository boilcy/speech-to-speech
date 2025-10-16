from src.base import BaseHandler

IDLE = "idle"
LISTENING = "listening"

class WakeupMiddleware(BaseHandler):
    def setup(self, wakeup_word: str | list[str] = "你好", end_word: str | list[str] = "再见"):
        self.wakeup_word = wakeup_word
        self.end_word = end_word
        self.mode = IDLE

    def process(self, transcript: str):
        if self.mode == LISTENING:
            if self.end_word in transcript:
                self.mode = IDLE
            else:
                yield transcript
        elif self.mode == IDLE:
            if self.wakeup_word in transcript:
                self.mode = LISTENING