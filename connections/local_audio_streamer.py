import threading
import numpy as np
import sounddevice as sd
import librosa
from loguru import logger
from queue import Queue

# --- Constants ---
TARGET_SAMPLE_RATE = 16000
# Minimum audio chunk size in samples, required by models like VAD
VAD_MIN_CHUNK_SAMPLES = 512
# Data type for audio processing
AUDIO_DTYPE_FLOAT32 = np.float32
AUDIO_DTYPE_INT16 = np.int16


class LocalAudioStreamer:
    def __init__(
        self,
        input_queue: Queue,
        output_queue: Queue,
        chunk_size: int = 512,
        device: str = None,  # Input, Output
        echo_suppression_delay: float = 0.2,
        echo_suppression_factor: float = 0.8,
    ):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.chunk_size = chunk_size
        self.device_str = device
        self.echo_suppression_delay = echo_suppression_delay
        self.min_gain = 1.0 - echo_suppression_factor

        self.stop_event = threading.Event()
        self.is_playing = threading.Event()
        self.last_play_time = 0.0

        # Buffer for incoming audio before it's chunked
        self._input_buffer = np.array([], dtype=AUDIO_DTYPE_FLOAT32)

        # Device properties that will be configured in run()
        self._input_device_idx = None
        self._output_device_idx = None
        self._input_device_sr = None
        self._needs_resampling = False

    def _setup_devices(self):
        """Queries and sets up the audio devices."""
        logger.debug("Querying audio devices...")
        devices = sd.query_devices()
        logger.info(f"Available audio devices:\n{devices}")

        if self.device_str:
            try:
                self._input_device_idx, self._output_device_idx = map(
                    int, self.device_str.split(",")
                )
            except (ValueError, IndexError):
                logger.warning(
                    f"Invalid device string '{self.device_str}'. Using default devices."
                )
                self._input_device_idx, self._output_device_idx = sd.default.device
        else:
            self._input_device_idx, self._output_device_idx = sd.default.device

        sd.default.device = self._input_device_idx, self._output_device_idx

        input_device_info = devices[self._input_device_idx]
        output_device_info = devices[self._output_device_idx]

        self._input_device_sr = int(input_device_info["default_samplerate"])
        self._needs_resampling = self._input_device_sr != TARGET_SAMPLE_RATE

        logger.info(
            f"Using Input Device : '{input_device_info['name']}' (Index: {self._input_device_idx})"
        )
        logger.info(
            f"Using Output Device: '{output_device_info['name']}' (Index: {self._output_device_idx})"
        )
        logger.info(f"Input device sample rate: {self._input_device_sr} Hz")
        if self._needs_resampling:
            logger.info(
                f"Input resampling enabled: {self._input_device_sr}Hz -> {TARGET_SAMPLE_RATE}Hz"
            )

    def _input_callback(
        self, indata: np.ndarray, frames: int, time, status: sd.CallbackFlags
    ):
        """Callback function for the audio input stream."""
        if status:
            logger.warning(f"Input stream status: {status}")

        current_time = time.currentTime
        gain = 1.0

        # --- Simplified Echo Suppression Logic ---
        if self.is_playing.is_set():
            gain = self.min_gain
            logger.debug(f"Playback active. Suppressing input with gain: {gain:.2f}")
        elif self.last_play_time > 0:
            time_since_last_play = current_time - self.last_play_time
            # Ensure time_since_last_play is non-negative and within reasonable bounds
            if 0 <= time_since_last_play < self.echo_suppression_delay:
                # Ramp gain up from min_gain to 1.0 over the delay period
                ramp_progress = time_since_last_play / self.echo_suppression_delay
                gain = self.min_gain + (1.0 - self.min_gain) * ramp_progress
                logger.debug(
                    f"Post-playback suppression. Time since play: {time_since_last_play:.2f}s. "
                    f"Ramping gain to: {gain:.2f}"
                )
            elif time_since_last_play < 0:
                # Handle negative time difference (clock issues or callback timing)
                logger.debug(
                    f"Negative time difference detected: {time_since_last_play:.2f}s. Using min_gain."
                )
                gain = self.min_gain

        if gain < 1.0:
            indata *= gain

        # --- Resampling and Buffering ---
        try:
            # Use only the first channel and ensure it's float32
            input_audio = indata[:, 0].astype(AUDIO_DTYPE_FLOAT32)

            if self._needs_resampling:
                resampled_audio = librosa.resample(
                    y=input_audio,
                    orig_sr=self._input_device_sr,
                    target_sr=TARGET_SAMPLE_RATE,
                )
            else:
                resampled_audio = input_audio

            self._input_buffer = np.concatenate([self._input_buffer, resampled_audio])

            # --- Chunking and Queuing ---
            while len(self._input_buffer) >= VAD_MIN_CHUNK_SAMPLES:
                chunk = self._input_buffer[:VAD_MIN_CHUNK_SAMPLES]
                self._input_buffer = self._input_buffer[VAD_MIN_CHUNK_SAMPLES:]

                # Convert to int16 for the processing pipeline
                chunk_int16 = (chunk * 32767).astype(AUDIO_DTYPE_INT16)
                self.input_queue.put(chunk_int16)

        except Exception as e:
            logger.error(f"Error in input callback: {e}")

    def _output_callback(
        self, outdata: np.ndarray, frames: int, time, status: sd.CallbackFlags
    ):
        """Callback function for the audio output stream."""
        if status:
            logger.warning(f"Output stream status: {status}")

        try:
            if not self.output_queue.empty():
                audio_chunk = self.output_queue.get_nowait()

                # Set playback flag and update timestamp
                if not self.is_playing.is_set():
                    logger.debug("Playback started.")
                    self.is_playing.set()
                self.last_play_time = time.currentTime

                # Ensure chunk fits into outdata buffer
                chunk_len = len(audio_chunk)
                if chunk_len < frames:
                    # Pad with silence if chunk is too short
                    outdata[:chunk_len, 0] = audio_chunk
                    outdata[chunk_len:, 0] = 0
                else:
                    # Truncate if chunk is too long
                    outdata[:, 0] = audio_chunk[:frames]

            else:
                # Fill with silence if queue is empty
                outdata.fill(0)
                if self.is_playing.is_set():
                    logger.debug(
                        f"Output queue empty. Playback stopped. "
                        f"Echo suppression will ramp down for {self.echo_suppression_delay}s."
                    )
                    self.is_playing.clear()

        except Exception as e:
            logger.error(f"Error in output callback: {e}")
            outdata.fill(0)
            if self.is_playing.is_set():
                self.is_playing.clear()

    def run(self):
        """Starts the audio input and output streams."""
        self._setup_devices()

        # Use smaller blocksize for input to reduce latency
        input_blocksize = min(self.chunk_size, 1024)

        logger.info("Starting audio streams...")
        logger.info(
            f"  Input  -> device={self._input_device_idx}, rate={self._input_device_sr}Hz, blocksize={input_blocksize}"
        )
        logger.info(
            f"  Output -> device={self._output_device_idx}, rate={TARGET_SAMPLE_RATE}Hz, blocksize={self.chunk_size}"
        )

        try:
            with (
                sd.InputStream(
                    device=self._input_device_idx,
                    samplerate=self._input_device_sr,
                    dtype=AUDIO_DTYPE_FLOAT32,
                    channels=1,
                    callback=self._input_callback,
                    blocksize=input_blocksize,
                    latency="low",
                ),
                sd.OutputStream(
                    device=self._output_device_idx,
                    samplerate=TARGET_SAMPLE_RATE,
                    dtype=AUDIO_DTYPE_INT16,
                    channels=1,
                    callback=self._output_callback,
                    blocksize=self.chunk_size,
                    latency="low",
                ),
            ):
                # Wait until stop_event is set, avoiding a busy-wait loop
                self.stop_event.wait()

        except sd.PortAudioError as e:
            logger.error(
                f"PortAudio error: {e}. Check if the audio devices are available and drivers are installed."
            )
        except Exception as e:
            logger.error(f"An unexpected error occurred in the audio stream: {e}")
        finally:
            logger.info("Audio streams stopped.")

    def stop(self):
        """Signals the audio streams to stop."""
        logger.info("Stopping audio streamer...")
        self.stop_event.set()
