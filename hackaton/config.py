"""Central configuration: environment, paths, constants and client factories.

Nothing here talks to the network or the filesystem at import time; the
`build_*` helpers are called explicitly from `hackaton.app.main`.
"""

import os
from dataclasses import dataclass
from getpass import getpass
from pathlib import Path
from typing import Optional

from autogen import LLMConfig
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# -----------------------------------------------------------------------------
# Paths (resolved against the repository root so the app can be started from
# any working directory)
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ASSETS_DIR = PROJECT_ROOT / "assets"

CSV_PATH = DATA_DIR / "final_data.csv"
CHROMA_DB_PATH = PROJECT_ROOT / ".chroma_keyword_db"
GRAPH_IMAGE = ASSETS_DIR / "graph.png"
QRCODE_IMAGE = ASSETS_DIR / "QRCode.png"
RECORDING_PATH = PROJECT_ROOT / "output.wav"

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o")
TTS_MODEL = os.getenv("TTS_MODEL", "gpt-4o-mini-tts")
TTS_VOICE = "nova"
TTS_SAMPLE_RATE = 24000

# -----------------------------------------------------------------------------
# Audio capture / speech recognition
# -----------------------------------------------------------------------------
SAMPLE_RATE = 44100
CHANNELS = 1
CHUNK = 1024
STT_LANGUAGE = "fa-IR"

# -----------------------------------------------------------------------------
# Agent conversation
# -----------------------------------------------------------------------------
MAX_ROUNDS = 7

# -----------------------------------------------------------------------------
# Product categories
#
# The three spellings below are NOT interchangeable: the CSV column for laptops
# uses a plain space ("لپ تاپ") while the UI label uses a zero-width non-joiner
# ("لپ‌تاپ"). Keep both literals exactly as they are.
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Category:
    key: str          # internal identifier, also used in ChromaDB metadata
    csv_column: str   # column name inside data/final_data.csv
    ui_label: str     # label shown in the category combo box


CATEGORIES = (
    Category("car", "ماشین", "خودرو"),
    Category("laptop", "لپ تاپ", "لپ‌تاپ"),
    Category("phone", "تلفن همراه", "تلفن همراه"),
)

UI_LABELS = [category.ui_label for category in CATEGORIES]

# -----------------------------------------------------------------------------
# Metis / OpenAI credentials
# -----------------------------------------------------------------------------
METIS_BASE_URL = os.getenv("METIS_BASE_URL")


def get_metis_api_key() -> Optional[str]:
    """Return the Metis API key, prompting for it once if it is not set."""
    api_key = os.getenv("METIS_API_KEY")
    if not api_key:
        os.environ["METIS_API_KEY"] = getpass("Paste your Metis API Key: ")
        api_key = os.getenv("METIS_API_KEY")
    return api_key


def build_openai_client() -> OpenAI:
    """Create the synchronous OpenAI client used for text-to-speech."""
    return OpenAI(api_key=get_metis_api_key(), base_url=METIS_BASE_URL)


def build_llm_config() -> LLMConfig:
    """Create the LLM configuration shared by every agent."""
    return LLMConfig(
        api_type="openai",
        model=GPT_MODEL,
        api_key=get_metis_api_key(),
        base_url=METIS_BASE_URL,
    )
