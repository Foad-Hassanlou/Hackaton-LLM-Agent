"""Audio capture, transcription and speech synthesis."""

from hackaton.audio.recorder import AudioRecorder
from hackaton.audio.transcriber import transcribe_wav
from hackaton.audio.tts import TTSManager

__all__ = ["AudioRecorder", "transcribe_wav", "TTSManager"]
