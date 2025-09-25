from loguru import logger
import queue
import threading
from abc import ABC, abstractmethod
from time import perf_counter
from collections import deque

SENTINEL = object()


class BaseHandler(ABC):
    """
    Base class for pipeline handlers.
    """

    def __init__(
        self,
        stop_event: threading.Event,
        input_queue: queue.Queue,
        output_queue: queue.Queue,
        min_debug_time: float = 0.001,
        queue_timeout: float = 60,
        setup_args: tuple = (),
        setup_kwargs: dict = {},
    ):
        self.stop_event = stop_event
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.min_debug_time = min_debug_time
        self.queue_timeout = queue_timeout
        self._times = deque(maxlen=1000)
        self.items_processed = 0

        logger.debug(
            f"Setup {self.__class__.__name__} with args: {setup_args} and kwargs: {setup_kwargs}"
        )

        _setup_kwargs = {k: v for k, v in setup_kwargs.items() if v is not None}

        self.setup(*setup_args, **_setup_kwargs)

    def setup(self, *setup_args, **setup_kwargs):
        pass

    @abstractmethod
    def process(self, item):
        """
        Subclasses transaction logic.
        """
        raise NotImplementedError

    def run(self):
        try:
            while not self.stop_event.is_set():
                try:
                    item = self.input_queue.get(timeout=self.queue_timeout)
                except queue.Empty:
                    continue

                if item is SENTINEL:
                    logger.debug("Sentinel received, stopping.")
                    break

                start_time = perf_counter()
                try:
                    for output in self.process(item):
                        self.output_queue.put(output)
                except Exception as e:
                    logger.error(
                        f"{self.__class__.__name__}: Error processing item: {item}. Error: {e}",
                        exc_info=True,
                    )

                duration = perf_counter() - start_time
                self._times.append(duration)
                self.items_processed += 1

                if duration > self.min_debug_time:
                    logger.debug(
                        f"{self.__class__.__name__}: Processed item in {duration:.4f} s"
                    )

        finally:
            logger.info("Running cleanup...")
            self.cleanup()
            logger.info("Putting sentinel in output queue.")
            self.output_queue.put(SENTINEL)

    @property
    def average_time(self) -> float:
        """计算最近处理时间的平均值。"""
        if not self._times:
            return 0.0
        return sum(self._times) / len(self._times)

    def cleanup(self):
        pass
