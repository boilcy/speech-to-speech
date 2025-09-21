import threading
import sounddevice as sd
import numpy as np
import librosa
import time
from loguru import logger


class LocalAudioStreamer:
    def __init__(
        self,
        input_queue,
        output_queue,
        list_play_chunk_size=512,
        sounddevice_device=None,
        echo_suppression_delay=0.2,  # 回声抑制延迟（秒）
        echo_suppression_factor=0.8,  # 回声抑制强度
    ):
        self.list_play_chunk_size = list_play_chunk_size

        self.stop_event = threading.Event()
        self.input_queue = input_queue
        self.output_queue = output_queue

        self.sounddevice_device = sounddevice_device
        
        # Buffer for accumulating resampled audio to meet minimum chunk size
        self.audio_buffer = np.array([], dtype=np.float32)
        self.min_chunk_samples = 512  # Minimum samples required by VAD model
        
        # Echo suppression parameters
        self.echo_suppression_delay = echo_suppression_delay
        self.echo_suppression_factor = echo_suppression_factor
        self.is_playing = threading.Event()  # Flag to indicate if we're currently playing audio
        self.last_play_time = 0  # Timestamp of last audio playback
        
        # Output audio history for echo cancellation
        self.output_history_buffer = np.array([], dtype=np.float32)
        self.max_history_samples = int(16000 * echo_suppression_delay * 2)  # 2x delay buffer
        
        # Processing lock to prevent race conditions
        self.processing_lock = threading.Lock()

    def run(self):
        devices = sd.query_devices()
        logger.info("Available devices:")
        logger.info(devices)

        input_device_index = (
            sd.default.device[0]
            if self.sounddevice_device is None
            else int(self.sounddevice_device.split(",")[0])
        )
        output_device_index = (
            sd.default.device[1]
            if self.sounddevice_device is None
            else int(self.sounddevice_device.split(",")[1])
        )

        sd.default.device = input_device_index, output_device_index

        default_input_device = devices[input_device_index]["name"]
        default_output_device = devices[output_device_index]["name"]

        default_input_sample_rate = int(
            devices[input_device_index]["default_samplerate"]
        )
        target_sample_rate = 16000
        
        default_output_sample_rate = int(
            devices[output_device_index]["default_samplerate"]
        )

        logger.info(
            f"Using input/output device: {default_input_device}/{default_output_device}"
        )
        logger.info(
            f"Input Device sample rate: {default_input_sample_rate}Hz, target: {target_sample_rate}Hz"
        )

        # Calculate resampling ratio if needed for input (from stream rate to target 16kHz)
        needs_input_resampling = default_input_sample_rate != target_sample_rate
        if needs_input_resampling:
            input_resample_ratio = target_sample_rate / default_input_sample_rate
            logger.info(f"Input resampling enabled: {default_input_sample_rate}Hz -> {target_sample_rate}Hz")
        else:
            input_resample_ratio = 1

        logger.info(f"Using separate streams due to different sample rates: input {default_input_sample_rate}Hz, output {default_output_sample_rate}Hz")
        
        def input_callback(indata, frames, time, status):
            if status:
                logger.warning(f"Input audio callback status: {status}")
            
            logger.debug(f"Input audio callback: {indata}")
            
            # 回声抑制：如果正在播放音频或刚播放完，暂时抑制输入
            current_time = time.inputBufferAdcTime if hasattr(time, 'inputBufferAdcTime') else time.time()
            time_since_last_play = current_time - self.last_play_time
            
            # 如果正在播放或播放结束后的抑制延迟期内，降低输入音频或跳过处理
            if self.is_playing.is_set() or time_since_last_play < self.echo_suppression_delay:
                # 在回声抑制期间，要么完全跳过，要么大幅降低音量
                if time_since_last_play < self.echo_suppression_delay * 0.5:
                    # 完全抑制前半段时间
                    return
                # 后半段时间逐渐恢复
                suppression_factor = (time_since_last_play - self.echo_suppression_delay * 0.5) / (self.echo_suppression_delay * 0.5)
                suppression_factor = min(1.0, max(0.0, suppression_factor)) * (1 - self.echo_suppression_factor)
            else:
                suppression_factor = 1.0
            
            # Handle input audio (recording)
            if indata is not None and len(indata) > 0:
                with self.processing_lock:
                    try:
                        # Convert input data to float32 for processing
                        input_audio = indata[:, 0].astype(np.float32) * suppression_factor
                        
                        # Resample input audio if needed (from device rate to 16kHz)
                        if needs_input_resampling:
                            resampled_input = librosa.resample(
                                input_audio, 
                                orig_sr=default_input_sample_rate, 
                                target_sr=target_sample_rate
                            )
                        else:
                            resampled_input = input_audio
                        
                        # Add to buffer and process when we have enough samples
                        self.audio_buffer = np.concatenate([self.audio_buffer, resampled_input])
                        
                        # Process chunks of minimum size
                        while len(self.audio_buffer) >= self.min_chunk_samples:
                            chunk = self.audio_buffer[:self.min_chunk_samples]
                            self.audio_buffer = self.audio_buffer[self.min_chunk_samples:]
                            
                            # Convert to int16 for pipeline consistency
                            chunk_int16 = (chunk * 32767).astype(np.int16)
                            
                            # 只在非抑制期间发送到队列
                            if suppression_factor > 0.3:  # 只有在抑制因子足够高时才处理
                                self.input_queue.put(chunk_int16)
                                
                    except Exception as e:
                        logger.error(f"Error in input callback: {e}")
        
        def output_callback(outdata, frames, time, status):
            if status:
                logger.warning(f"Output audio callback status: {status}")
                
            if self.output_queue.empty():
                outdata[:] = 0
                self.is_playing.clear()  # 标记不在播放
            else:
                try:
                    with self.processing_lock:
                        audio_data = self.output_queue.get_nowait()
                        
                        # 确保数据格式正确
                        if len(audio_data.shape) == 1:
                            # 单声道数据，添加维度
                            audio_output = audio_data[:frames] if len(audio_data) >= frames else np.pad(audio_data, (0, frames - len(audio_data)))
                        else:
                            # 多声道数据，取第一个声道
                            audio_output = audio_data[:frames, 0] if len(audio_data) >= frames else np.pad(audio_data[:, 0], (0, frames - len(audio_data)))
                        
                        outdata[:, 0] = audio_output
                        
                        # 更新播放状态和时间戳
                        self.is_playing.set()
                        self.last_play_time = time.outputBufferDacTime if hasattr(time, 'outputBufferDacTime') else time.time()
                        
                        # 更新输出历史缓冲区用于回声消除
                        output_float = audio_output.astype(np.float32) / 32767.0 if audio_output.dtype == np.int16 else audio_output.astype(np.float32)
                        self.output_history_buffer = np.concatenate([self.output_history_buffer, output_float])
                        
                        # 限制历史缓冲区大小
                        if len(self.output_history_buffer) > self.max_history_samples:
                            self.output_history_buffer = self.output_history_buffer[-self.max_history_samples:]
                            
                except Exception as e:
                    logger.error(f"Error in output callback: {e}")
                    outdata[:] = 0
                    self.is_playing.clear()

        
        # 优化缓冲区设置以减少延迟和overflow
        input_blocksize = min(self.list_play_chunk_size, 1024)  # 减小输入块大小
        output_blocksize = self.list_play_chunk_size
        
        logger.info(f"Audio stream settings:")
        logger.info(f"  Input: device={input_device_index}, rate={default_input_sample_rate}Hz, blocksize={input_blocksize}")
        logger.info(f"  Output: device={output_device_index}, rate=16000Hz, blocksize={output_blocksize}")
        logger.info(f"  Echo suppression: delay={self.echo_suppression_delay}s, factor={self.echo_suppression_factor}")
        
        # Create separate input and output streams
        with sd.InputStream(
            device=input_device_index,
            samplerate=default_input_sample_rate,
            dtype=np.float32,
            channels=1,
            callback=input_callback,
            blocksize=input_blocksize,
            latency='low',  # 低延迟模式
        ), sd.OutputStream(
            device=output_device_index,
            samplerate=16000,
            dtype="int16",
            channels=1,
            callback=output_callback,
            blocksize=output_blocksize,
            latency='low',  # 低延迟模式
        ):
            logger.info("Starting separate input and output audio streams")
            while not self.stop_event.is_set():
                time.sleep(0.001)
            logger.info("Stopping separate audio streams")
