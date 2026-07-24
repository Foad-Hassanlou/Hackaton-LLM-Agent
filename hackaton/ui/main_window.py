"""The main application window."""

import time

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QTextEdit, QVBoxLayout, QWidget
)

from hackaton import config
from hackaton.agents import AgentTeam
from hackaton.audio import AudioRecorder, TTSManager, transcribe_wav
from hackaton.data import ProductCatalog
from hackaton.ui import styles
from hackaton.ui.windows import DataTableWindow, GraphWindow, QRCodeWindow

USER_SPEAKER = "User"
ASSISTANT_SPEAKER = "Assistant"


class VoiceRecorder(QWidget):
    """Records a Persian query, runs the agent pipeline and speaks the answer."""

    def __init__(self, catalog: ProductCatalog, team: AgentTeam, tts_manager: TTSManager):
        super().__init__()
        self.catalog = catalog
        self.team = team
        self.tts_manager = tts_manager
        self.recorder = AudioRecorder()

        self.last_text = ""   # Store last transcription
        self.messages = []    # Store all messages as (speaker, message) tuples

        self.setWindowTitle("Hackaton")
        self.setFixedSize(400, 750)
        self.setup_ui()

    # -------------------------------------------------------------------------
    # UI construction
    # -------------------------------------------------------------------------
    def setup_ui(self):
        # Main vertical layout
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        layout.addLayout(self._build_top_row())

        # Status label with rounded corners
        self.status_label = QLabel("Ready to record")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet(styles.STATUS_LABEL)
        layout.addWidget(self.status_label)

        # Record/Stop button
        self.record_button = QPushButton("Start Recording 🎤")
        self.record_button.setFixedHeight(60)
        self.record_button.clicked.connect(self.toggle_recording)
        self.record_button.setStyleSheet(styles.RECORD_BUTTON)
        layout.addWidget(self.record_button)

        # Scrollable text area for transcript
        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        self.text_area.setStyleSheet(styles.TEXT_AREA)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.text_area)
        layout.addWidget(scroll)

        # Read last message button
        self.read_button = QPushButton("🔊 Read Last Message")
        self.read_button.setFixedHeight(40)
        self.read_button.clicked.connect(self.read_last_message)
        self.read_button.setStyleSheet(styles.READ_BUTTON)
        layout.addWidget(self.read_button)

        self.setLayout(layout)

    def _build_top_row(self) -> QHBoxLayout:
        """Graph button, category selector and data button."""
        btn_row = QHBoxLayout()

        self.graph_button = QPushButton("📈 Show Graph")
        self.graph_button.setFixedHeight(40)
        self.graph_button.clicked.connect(self.open_graph_window)
        self.graph_button.setStyleSheet(styles.GRAPH_BUTTON)
        btn_row.addWidget(self.graph_button)

        # Category selector combo box (centered)
        self.type_combo = QComboBox()
        self.type_combo.addItems(config.UI_LABELS)
        self.type_combo.setStyleSheet(styles.TYPE_COMBO)
        btn_row.addStretch()
        btn_row.addWidget(self.type_combo, alignment=Qt.AlignCenter)
        btn_row.addStretch()

        self.show_data_button = QPushButton("📊 Display Data")
        self.show_data_button.setFixedHeight(40)
        self.show_data_button.clicked.connect(self.open_data_window)
        self.show_data_button.setStyleSheet(styles.DATA_BUTTON)
        btn_row.addWidget(self.show_data_button)

        return btn_row

    # -------------------------------------------------------------------------
    # Recording flow
    # -------------------------------------------------------------------------
    def toggle_recording(self):
        if not self.recorder.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        # Update UI to recording
        self.recorder.start()
        self.record_button.setText("Stop Recording ✋")
        self.status_label.setText("Recording...")

    def stop_recording(self):
        wav_path = self.recorder.stop()
        if wav_path is None:
            self.status_label.setText("Error saving audio")
            self.record_button.setText("Start Recording 🎤")
            return

        self.status_label.setText("Processing...")
        QApplication.processEvents()

        # Transcribe audio
        text = transcribe_wav(wav_path)
        self.last_text = text
        self.append_message(USER_SPEAKER, text)

        self.status_label.setText("Please Wait to response")
        self.record_button.setText("Processing ...")

        # Run agent conversation
        try:
            self.run_agent_conversation(text)
        except Exception as e:
            print(f"Agent error: {e}")

        self.status_label.setText("Done! Ready for next recording.")
        self.record_button.setText("Start Recording 🎤")

    def run_agent_conversation(self, user_message: str):
        """Run the agent group chat and render every score_agent reply."""
        for entry in self.team.run(user_message):
            if entry.get("name") != "score_agent":
                continue

            self.show_qrcode()
            res = entry["content"]
            if res != "None":
                self.append_message(ASSISTANT_SPEAKER, res)
                self.speak(res)
            elif res is not None:
                self.append_message(ASSISTANT_SPEAKER, res)

    def speak(self, text: str):
        """Read a reply aloud, waiting for playback to finish."""
        try:
            print("TTS: Processing...")
            self.tts_manager.read_text(text)
            time.sleep(3)  # Wait for playback to complete
            print("TTS: Done!")
        except Exception as e:
            print(f"Error during TTS: {e}")

    def read_last_message(self):
        """Read the last assistant's message aloud and update the status label."""
        if not self.messages:
            return

        for speaker, message in reversed(self.messages):
            if speaker == ASSISTANT_SPEAKER:
                self.status_label.setText("🔊 TTS: Processing...")
                QApplication.processEvents()
                self.tts_manager.read_text(message)
                time.sleep(3)
                self.status_label.setText("✅ TTS: Done!")
                break

    # -------------------------------------------------------------------------
    # Transcript & windows
    # -------------------------------------------------------------------------
    def append_message(self, speaker: str, message: str):
        """Add a message to the transcript area and to the history."""
        if speaker == USER_SPEAKER:
            self.text_area.append(f"<div align='right'>🌟 {message}</div>")
        else:
            self.text_area.append(f"<div align='left'>🤖 {message}</div>")
        self.messages.append((speaker, message))

    def open_graph_window(self):
        # Kept on self so the window is not garbage collected
        self.graph_window = GraphWindow()
        self.graph_window.show()

    def open_data_window(self):
        category_label = self.type_combo.currentText()
        df = self.catalog.frame_for_ui_label(category_label)
        self.data_window = DataTableWindow(df, category_label)
        self.data_window.show()

    def show_qrcode(self):
        self.qr_window = QRCodeWindow()
        self.qr_window.show()
