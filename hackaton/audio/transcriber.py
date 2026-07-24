"""Speech-to-text."""

import speech_recognition as sr

from hackaton import config


def transcribe_wav(wav_path, language: str = config.STT_LANGUAGE) -> str:
    """Transcribe a WAV file with Google's recogniser.

    Returns a human-readable error message instead of raising, because the
    result is shown directly in the transcript area.
    """
    recognizer = sr.Recognizer()
    with sr.AudioFile(str(wav_path)) as source:
        audio_data = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio_data, language=language)
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError:
        return "Speech recognition service error"
