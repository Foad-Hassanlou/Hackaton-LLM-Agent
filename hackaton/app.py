"""Application entry point: builds every component and starts the GUI."""

import sys

from PyQt5.QtWidgets import QApplication

from hackaton import config
from hackaton.agents import build_agent_team
from hackaton.audio import TTSManager
from hackaton.data import ProductCatalog
from hackaton.search import KeywordSearcher
from hackaton.ui import VoiceRecorder


def main():
    client = config.build_openai_client()

    catalog = ProductCatalog()
    searcher = KeywordSearcher(catalog)
    team = build_agent_team(searcher)
    tts_manager = TTSManager(client)

    app = QApplication(sys.argv)
    window = VoiceRecorder(catalog=catalog, team=team, tts_manager=tts_manager)
    window.show()
    sys.exit(app.exec_())
