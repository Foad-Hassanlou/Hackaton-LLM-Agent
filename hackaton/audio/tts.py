"""Text-to-speech playback."""

import numpy as np
import sounddevice as sd
from openai import OpenAI

from hackaton import config


class TTSManager:
    """Manages text-to-speech playback using the synchronous OpenAI client."""

    def __init__(self, client: OpenAI, model: str = config.TTS_MODEL,
                 voice: str = config.TTS_VOICE, sample_rate: int = config.TTS_SAMPLE_RATE):
        self.client = client
        self.model = model
        self.voice = voice
        self.sample_rate = sample_rate
        self.is_running = True

    def text_to_speech(self, text: str):
        """Convert text to speech and play it back, blocking until done."""
        try:
            tts_response = self.client.audio.speech.create(
                model=self.model,
                voice=self.voice,
                input=text,
                response_format="pcm"
            )
            audio_data = tts_response.content
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            sd.play(audio_array, samplerate=self.sample_rate)
            sd.wait()
        except Exception as e:
            print(f"TTS Error: {e}")

    def read_text(self, text: str):
        """Run text_to_speech if the manager is active."""
        if self.is_running:
            self.text_to_speech(text)

    def stop(self):
        """Stop the TTS manager."""
        self.is_running = False
