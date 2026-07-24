"""Microphone capture."""

import threading
import wave
from pathlib import Path
from typing import Optional

import pyaudio

from hackaton import config

FORMAT = pyaudio.paInt16


class AudioRecorder:
    """Records microphone input on a background thread and writes a WAV file."""

    def __init__(self, output_path=None, sample_rate=config.SAMPLE_RATE,
                 channels=config.CHANNELS, chunk=config.CHUNK):
        self.output_path = Path(output_path or config.RECORDING_PATH)
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk = chunk

        self.is_recording = False
        self.frames = []
        self._thread = None

    def start(self):
        """Begin recording in a separate thread."""
        self.frames = []
        self.is_recording = True
        self._thread = threading.Thread(target=self._record)
        self._thread.start()

    def stop(self) -> Optional[Path]:
        """Stop recording and save the WAV file.

        :return: The path of the saved file, or None if saving failed.
        """
        self.is_recording = False
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        return self._save()

    def _record(self):
        pa = pyaudio.PyAudio()
        try:
            stream = pa.open(format=FORMAT,
                             channels=self.channels,
                             rate=self.sample_rate,
                             input=True,
                             frames_per_buffer=self.chunk)
            while self.is_recording:
                data = stream.read(self.chunk, exception_on_overflow=False)
                self.frames.append(data)
            stream.stop_stream()
            stream.close()
        except Exception as e:
            print(f"Recording Error: {e}")
        finally:
            pa.terminate()

    def _save(self) -> Optional[Path]:
        try:
            with wave.open(str(self.output_path), 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self._sample_width())
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(self.frames))
            return self.output_path
        except Exception as e:
            print(f"Error saving WAV: {e}")
            return None

    @staticmethod
    def _sample_width() -> int:
        pa = pyaudio.PyAudio()
        try:
            return pa.get_sample_size(FORMAT)
        finally:
            pa.terminate()
