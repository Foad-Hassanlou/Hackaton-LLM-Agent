import sys
import threading
import wave
import pyaudio
import speech_recognition as sr
import pandas as pd
import ast
import time
import os
from dotenv import load_dotenv
from getpass import getpass
import asyncio
import numpy as np
import sounddevice as sd
from openai import AsyncOpenAI
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QLabel, QTextEdit, QScrollArea, QHBoxLayout, QComboBox,
    QTableWidget, QTableWidgetItem
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer
from pygame.time import delay
import chromadb
from chromadb.utils import embedding_functions
import json
import re
from autogen import ConversableAgent, LLMConfig
from autogen.agentchat import initiate_group_chat
from autogen.agentchat.group import AgentTarget, RevertToUserTarget, ReplyResult, ContextVariables
from autogen.agentchat.group.patterns import AutoPattern
from autogen.agentchat.group import OnContextCondition, ExpressionContextCondition, ContextExpression
import typing
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Retrieve Metis API configuration from environment
metis_api_key = os.getenv("METIS_API_KEY")
base_url = os.getenv("METIS_BASE_URL")

if not metis_api_key:
    os.environ["METIS_API_KEY"] = getpass("Paste your Metis API Key: ")
    metis_api_key = os.getenv("METIS_API_KEY")

# Initialize synchronous OpenAI client
client = OpenAI(api_key=metis_api_key, base_url=base_url)

# CSV path (using a consistent path)
CSV_PATH = 'final_data.csv'  # Adjust if necessary

# Load CSV for GUI purposes
final_data_df = pd.read_csv(CSV_PATH)
for col in final_data_df.columns:
    final_data_df[col] = final_data_df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else {})
car_df = pd.DataFrame(final_data_df['ماشین'].tolist())
laptop_df = pd.DataFrame(final_data_df['لپ تاپ'].tolist())
phone_df = pd.DataFrame(final_data_df['تلفن همراه'].tolist())

# Initialize TTS client
tts_client = AsyncOpenAI()

# -----------------------------------------------------------------------------
# KeywordSearcher class: provides normal_search (pure-Python) and pro_search
# -----------------------------------------------------------------------------
class KeywordSearcher:
    """
    A class that implements two types of keyword search over a dataset:
      - normal_search: a simple, pure-Python keyword count search
      - pro_search: a ChromaDB-backed keyword search with filtering

    Data is loaded from 'final_data.csv' and organized into three categories:
      - 'ماشین' (car)
      - 'لپ تاپ' (laptop)
      - 'تلفن همراه' (phone)

    Each search method returns matching documents along with their category
    and original row index in the CSV.
    """
    def __init__(self, csv_path='final_data.csv', db_path='.chroma_keyword_db'):
        # Load CSV into DataFrame
        self.df = pd.read_csv(csv_path)

        # Convert any stringified dicts back to Python dicts
        for col in self.df.columns:
            self.df[col] = self.df[col].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else {}
            )

        # Create separate DataFrames for each category
        self.car_df = pd.DataFrame(self.df['ماشین'].tolist())
        self.laptop_df = pd.DataFrame(self.df['لپ تاپ'].tolist())
        self.phone_df = pd.DataFrame(self.df['تلفن همراه'].tolist())

        # Prepare flat lists of documents and metadata
        self.documents = []
        self.metadatas = []
        self.ids = []
        for category, cat_df in [('car', self.car_df),
                                 ('laptop', self.laptop_df),
                                 ('phone', self.phone_df)]:
            for row_idx, row in cat_df.iterrows():
                # Represent each row as a single concatenated string for searching
                doc_text = ' '.join(f"{k}: {v}" for k, v in row.items())
                self.documents.append(doc_text)
                # Store metadata including category and original CSV row index
                self.metadatas.append({'category': category, 'row_idx': row_idx})
                # Create a unique ID combining category and row index
                self.ids.append(f"{category}_{row_idx}")

        # Initialize ChromaDB client for pro search
        self.client = chromadb.PersistentClient(path=db_path)
        # Embeddings are not used in keyword matching, but required by API
        dummy_ef = embedding_functions.DefaultEmbeddingFunction()
        # Ensure fresh collection: delete existing data if any
        try:
            self.client.delete_collection(name='pure_keyword_search')
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name='pure_keyword_search',
            embedding_function=dummy_ef
        )

        # Clear any existing documents just in case
        try:
            self.collection.delete(where={})
        except Exception:
            pass

        # Add documents to the ChromaDB collection with row_idx metadata
        self.collection.add(
            documents=self.documents,
            metadatas=self.metadatas,
            ids=self.ids
        )

    def normal_search(self, query, k=5):
        """
        Perform a simple keyword-count search over the documents.

        :param query: The search query string
        :param k: Number of top matches to return
        :return: List of tuples (score, category, row_idx, document)
        """
        query = query.strip().lower()
        results = []
        # Score each document by counting occurrences of query words
        for doc, meta in zip(self.documents, self.metadatas):
            doc_lower = doc.lower()
            score = sum(1 for word in query.split() if word in doc_lower)
            if score > 0:
                results.append((score, meta['category'], meta['row_idx'], doc))
        # Sort by descending score and return top k
        results.sort(key=lambda x: x[0], reverse=True)
        return results[:k]

    def pro_search(self, query, k=5, category=None):
        """
        Perform a ChromaDB-backed keyword search using filter conditions.

        :param query: The search query string
        :param k: Number of top matches to return
        :param category: Optional category filter ('car', 'laptop', 'phone')
        :return: List of tuples (document, category, row_idx)
        """
        # Build filter for document text
        where_doc = {"$contains": query}
        # Prepare query args
        query_args = {
            'where_document': where_doc,
            'limit': k,
            'include': ['documents', 'metadatas']
        }
        # Add category filter if provided
        if category:
            query_args['where'] = {'category': category}

        results = self.collection.get(**query_args)
        # Include row_idx in output
        output = []
        for doc, meta in zip(results['documents'], results['metadatas']):
            output.append((doc, meta['category'], meta['row_idx']))
        return output

# -----------------------------------------------------------------------------
# Instantiate the KeywordSearcher and define search functions for agents
# -----------------------------------------------------------------------------
searcher = KeywordSearcher(csv_path='final_data.csv')

def search_products(query: str, k: int = 5):
    """
    Wrapper for ConversableAgent function: uses KeywordSearcher.normal_search
    to retrieve up to k matching products for the given Persian query string.
    Returns a list of dicts with fields: category, row_idx, score, and doc.
    """
    raw_results = searcher.normal_search(query, k=k)
    return [
        {
            "category": category,
            "row_idx": row_idx,
            "score": score,
            "doc": doc
        }
        for score, category, row_idx, doc in raw_results
    ]

def pro_search_products(query: str, k: int = 5):
    """
    Wrapper for ConversableAgent function: uses KeywordSearcher.pro_search
    to retrieve up to k matching products for the given Persian query string.
    Returns a list of dicts with fields: category, row_idx, score, and doc.
    """
    raw_results = searcher.pro_search(query, k=k)
    return [
        {
            "category": category,
            "row_idx": row_idx,
            "score": 1,  # Dummy score for pro_search
            "doc": doc
        }
        for doc, category, row_idx in raw_results
    ]

def perform_search(query: str, context: ContextVariables):
    """
    Function for search_agent: calls search_products and constructs a message
    with the query and raw results as a JSON string. Updates context_vars
    with the number of results found.
    """
    results = search_products(query)
    context_vars.data['num_results'] = len(results)  # Update global context_vars
    results_str = json.dumps(results, ensure_ascii=False)
    message = f'For query "{query}", raw results: {results_str}'
    return message


def handle_results(message: str, context: ContextVariables) -> ReplyResult:
    """
    Tool for check_agent: decides the next step based on the number of results
    stored in context_vars.data['num_results'] and returns a ReplyResult with the
    appropriate message and target agent.
    """
    num_results = context_vars.data['num_results']  # Use global context_vars

    # Try to match the expected message format from perform_search
    match = re.search(r'For query "(.*?)", raw results: (.*)', message)
    if not match:
        try:
            data = json.loads(message)
            if isinstance(data, dict) and 'query' in data and 'results' in data:
                query = data['query']
                results_str = json.dumps(data['results'], ensure_ascii=False)
            else:
                return ReplyResult(
                    message="Error: Could not parse the message.",
                    target=None,
                    context_variables=context_vars  # Pass context_vars
                )
        except json.JSONDecodeError:
            return ReplyResult(
                message="Error: Could not parse the message.",
                target=None,
                context_variables=context_vars  # Pass context_vars
            )
    else:
        query = match.group(1)
        results_str = match.group(2)

    if num_results == 5:
        return ReplyResult(
            message=f"To score_agent: Here are the results: {results_str}",
            target=AgentTarget(score_agent),
            context_variables=context_vars  # Pass context_vars
        )
    elif 1 <= num_results <= 4:
        pro_results = pro_search_products(query, k=10)
        results = json.loads(results_str)
        all_results = results + pro_results
        unique_results = {}
        for res in all_results:
            row_idx = res['row_idx']
            if row_idx not in unique_results or unique_results[row_idx]['score'] < res['score']:
                unique_results[row_idx] = res
        sorted_results = sorted(unique_results.values(), key=lambda x: x['score'], reverse=True)
        top5 = sorted_results[:5]
        top5_str = json.dumps(top5, ensure_ascii=False)
        return ReplyResult(
            message=f"To score_agent: Here are the combined results: {top5_str}",
            target=AgentTarget(score_agent),
            context_variables=context_vars  # Pass context_vars
        )
    else:  # num_results == 0
        return ReplyResult(
            message=f"To custom_search_agent: Please modify the query: {query}",
            target=AgentTarget(custom_search_agent),
            context_variables=context_vars  # Pass context_vars
        )

# -----------------------------------------------------------------------------
# Autogen agents setup
# -----------------------------------------------------------------------------
# Shared context with initial stage and placeholder for number of results
context_vars = ContextVariables(data={"stage": "search", "num_results": 0})

# Configure the LLM
llm_config = LLMConfig(
    api_type="openai",
    model="gpt-4o",
    api_key=metis_api_key,
    base_url=base_url
)

with llm_config:
    user_agent = ConversableAgent(name="user", human_input_mode="ALWAYS")

    search_agent = ConversableAgent(
        name="search_agent",
        system_message="""
        You are the search agent. Your sole responsibility is to handle messages from the user or custom_search_agent and return the exact output of perform_search without any modification.
        - For user messages, extract meaningful Persian keywords (e.g., "ماشین", "سبز") and call perform_search.
        - For messages from custom_search_agent like 'To search_agent: Try searching with modified query: [query]', extract the query and call perform_search.
        - Return the EXACT output of perform_search without parsing, formatting, summarizing, or altering it in any way.
        """,
        functions=[perform_search]
    )

    check_agent = ConversableAgent(
        name="check_agent",
        system_message="""
            You are the check agent. When you receive a message from the search_agent, call handle_results with the entire message and the context. Output the exact message returned by handle_results.
            """,
        functions=[handle_results]
    )

    custom_search_agent = ConversableAgent(
        name="custom_search_agent",
        system_message="""
        You are the custom search agent. You respond to messages like 'To custom_search_agent: Modify the query: [query]'.
        Modify the query by removing the last word (e.g., "ماشین سفید با کارکرد زیر 50 هزار" becomes "ماشین سفید با کارکرد زیر 50"), and send 'To search_agent: Try searching with modified query: [modified_query]' to the group.
        """,
    )

    score_agent = ConversableAgent(
        name="score_agent",
        system_message="""
            You are the score agent. You receive messages like 'To score_agent: Here are the results: [results]' or 'To score_agent: Here are the combined results: [results]'.

            First, identify the type of item(s) the user is evaluating (e.g., car, laptop, house). This is either provided explicitly or inferred from the content of the docs.

            Parse the results and sort them by score in descending order. For each result, extract the row_idx and doc.

            Present each result in Persian as follows:
            'آگهی [row_idx]: مزایا: [یک مزیت کوتاه بر اساس کالا و متن doc، مثل قیمت پایین یا سالم بودن]. معایب: [یک عیب کوتاه، مثل کارکرد بالا یا مدل قدیمی].'

            At the beginning of the output, state:
            'کالا: [نام کالا یا کالاها]' in Persian. If more than one item type, list them separated by "،".

            Use simple Persian words, avoid complex symbols, and keep the output concise for text-to-speech clarity.

            List all 5 results in order, then say:

            Example output:
            کالا: خودرو
            آگهی ۱: مزایا: قیمت پایین. معایب: مدل قدیمی.
            آگهی ۲: مزایا: وضعیت خوب. معایب: کارکرد بالا.
            ...
        """,
    )

search_agent.handoffs.set_after_work(AgentTarget(check_agent))

# Build the conversation pattern with all agents
pattern = AutoPattern(
    initial_agent=search_agent,
    agents=[search_agent, check_agent, custom_search_agent, score_agent],
    user_agent=user_agent,
    context_variables=context_vars,
    group_manager_args={"llm_config": llm_config}
)

# -----------------------------------------------------------------------------
# TTSManager class for text-to-speech functionality
# -----------------------------------------------------------------------------
class TTSManager:
    """Manages text-to-speech functionality using synchronous OpenAI client."""
    def __init__(self):
        self.is_running = True

    def text_to_speech(self, text: str):
        """Convert text to speech using OpenAI TTS."""
        try:
            # Use synchronous client to call the TTS endpoint
            tts_response = client.audio.speech.create(
                model="gpt-4o-mini-tts",  # Assuming this is the intended TTS model
                voice="nova",
                input=text,
                response_format="pcm"
            )
            # Extract audio data from response
            audio_data = tts_response.content
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            sd.play(audio_array, samplerate=24000)
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

# -----------------------------------------------------------------------------
# VoiceRecorder class with integrated agent system
# -----------------------------------------------------------------------------
class VoiceRecorder(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hackaton")
        self.setFixedSize(400, 750)
        self.last_text = ""  # Store last transcription
        self.messages = []  # Store all messages as (speaker, message) tuples
        self.tts_manager = TTSManager()  # Initialize TTS manager
        self.setup_ui()
        self.is_recording = False
        self.frames = []

    def setup_ui(self):
        # Main vertical layout
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Top button row: Graph, type selector, display data
        btn_row = QHBoxLayout()

        # Graph button
        self.graph_button = QPushButton("📈 Show Graph")
        self.graph_button.setFixedHeight(40)
        self.graph_button.clicked.connect(self.open_graph_window)
        self.graph_button.setStyleSheet(
            "QPushButton { font-size: 14px; color: #ffffff; background: #56CCF2; "
            "border-radius: 20px; padding: 8px; }"
            "QPushButton:hover { background: #2F80ED; }"
        )
        btn_row.addWidget(self.graph_button)

        # Category selector combo box (centered)
        self.type_combo = QComboBox()
        self.type_combo.addItems(['خودرو', 'لپ‌تاپ', 'تلفن همراه'])
        self.type_combo.setStyleSheet(
            "QComboBox { font-size: 14px; padding: 5px; border-radius: 20px; text-align: center; }"
        )
        btn_row.addStretch()
        btn_row.addWidget(self.type_combo, alignment=Qt.AlignCenter)
        btn_row.addStretch()

        # Display data button
        self.show_data_button = QPushButton("📊 Display Data")
        self.show_data_button.setFixedHeight(40)
        self.show_data_button.clicked.connect(self.open_data_window)
        self.show_data_button.setStyleSheet(
            "QPushButton { font-size: 14px; color: #ffffff; background: #6FCF97; "
            "border-radius: 20px; padding: 8px; }"
            "QPushButton:hover { background: #27AE60; }"
        )
        btn_row.addWidget(self.show_data_button)

        layout.addLayout(btn_row)

        # Status label with rounded corners
        self.status_label = QLabel("Ready to record")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet(
            "font-size: 16px; color: #ffffff; background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, "
            "stop:0 #ff7e5f, stop:1 #feb47b); border-radius: 30px; padding: 5px;"
        )
        layout.addWidget(self.status_label)

        # Record/Stop button
        self.record_button = QPushButton("Start Recording 🎤")
        self.record_button.setFixedHeight(60)
        self.record_button.clicked.connect(self.toggle_recording)
        self.record_button.setStyleSheet(
            "QPushButton { font-size: 18px; color: #ffffff; background: #6a82fb; "
            "border-radius: 30px; padding: 10px; }"
            "QPushButton:hover { background: #fc5c7d; }"
        )
        layout.addWidget(self.record_button)

        # Scrollable text area for transcript
        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        self.text_area.setStyleSheet(
            "font-size: 14px; background: #f7f7f7; border-radius: 10px; padding: 10px;"
        )
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.text_area)
        layout.addWidget(scroll)

        # Read last message button
        self.read_button = QPushButton("🔊 Read Last Message")
        self.read_button.setFixedHeight(40)
        self.read_button.clicked.connect(self.read_last_message)
        self.read_button.setStyleSheet(
            "QPushButton { font-size: 14px; color: #ffffff; background: #F2994A; "
            "border-radius: 20px; padding: 8px; }"
            "QPushButton:hover { background: #EB5757; }"
        )
        layout.addWidget(self.read_button)

        self.setLayout(layout)

    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        # Update UI to recording
        self.is_recording = True
        self.record_button.setText("Stop Recording ✋")
        self.status_label.setText("Recording...")

        # Start audio stream in a separate thread
        self.frames = []
        self.record_thread = threading.Thread(target=self.record)
        self.record_thread.start()

    def record(self):
        pa = pyaudio.PyAudio()
        try:
            stream = pa.open(format=pyaudio.paInt16,
                             channels=1,
                             rate=44100,
                             input=True,
                             frames_per_buffer=1024)
            while self.is_recording:
                data = stream.read(1024, exception_on_overflow=False)
                self.frames.append(data)
            stream.stop_stream()
            stream.close()
        except Exception as e:
            print(f"Recording Error: {e}")
        finally:
            pa.terminate()

    def stop_recording(self):
        self.is_recording = False
        self.record_thread.join()

        # Save to WAV
        try:
            filename = "output.wav"
            wf = wave.open(filename, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
            wf.setframerate(44100)
            wf.writeframes(b''.join(self.frames))
            wf.close()
        except Exception as e:
            print(f"Error saving WAV: {e}")
            self.status_label.setText("Error saving audio")
            self.record_button.setText("Start Recording 🎤")
            return

        self.status_label.setText("Processing...")
        QApplication.processEvents()

        # Transcribe audio
        text = self.transcribe(filename)
        self.last_text = text
        self.text_area.append(f"<div align='right'>🌟 {text}</div>")
        self.messages.append(("User", text))  # Fix tuple format

        self.status_label.setText("Please Wait to response")
        self.record_button.setText("Processing ...")

        # Run agent conversation
        try:
            response = self.run_agent_conversation(text)
        except Exception as e:
            pass

        self.status_label.setText("Done! Ready for next recording.")
        self.record_button.setText("Start Recording 🎤")

    def transcribe(self, wav_path):
        # Use SpeechRecognition to convert speech to text
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio_data, language="fa-IR")
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError:
            return "Speech recognition service error"

    def run_agent_conversation(self, user_message):
        # Run the agent conversation with the transcribed text
        result, final_context, last_agent = initiate_group_chat(
            pattern=pattern,
            messages=[f"User: {user_message}"],
            max_rounds=7
        )

        for entry in result.chat_history:
            if entry.get("name") == "score_agent":
                self.Show_QRCode()
                res = entry["content"]
                if (res != "None"):
                    self.text_area.append(f"<div align='left'>🤖 {res}</div>")
                    self.messages.append(("Assistant", entry["content"]))

                    # Initialize TTS manager
                    manager = TTSManager()
                    # Read the test text
                    try:
                        print("TTS: Processing...")
                        manager.read_text(res)
                        time.sleep(3)  # Wait for playback to complete
                        print("TTS: Done!")
                    except Exception as e:
                        print(f"Error during TTS: {e}")
                    # Clean up
                    manager.stop()

                elif (res != None):
                    self.text_area.append(f"<div align='left'>🤖 {res}</div>")
                    self.messages.append(("Assistant", entry["content"]))
                else:
                    pass

    def open_graph_window(self):
        # Create and show new window for graph image sized 1500x320
        self.graph_window = QWidget()
        self.graph_window.setWindowTitle("Graph Display")
        self.graph_window.setFixedSize(1520, 320)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        # Load and display the graph image at target size
        label = QLabel()
        pixmap = QPixmap("graph.png")  # Ensure 'graph.png' is in working directory
        label.setPixmap(pixmap.scaled(1510, 310, Qt.IgnoreAspectRatio, Qt.SmoothTransformation))
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

        self.graph_window.setLayout(layout)
        self.graph_window.show()

    def open_data_window(self):
        # Create a window to display data for selected category
        category = self.type_combo.currentText()
        # Choose the correct DataFrame based on category
        if category == 'خودرو':
            df = car_df
        elif category == 'لپ‌تاپ':
            df = laptop_df
        elif category == 'تلفن همراه':
            df = phone_df
        else:
            df = pd.DataFrame()

        # Convert DataFrame rows to list of dicts
        entries = df.to_dict('records')

        self.data_window = QWidget()
        self.data_window.setWindowTitle(f"Data: {category}")
        self.data_window.setFixedSize(800, 600)
        layout = QVBoxLayout()

        if entries:
            headers = list(entries[0].keys())
            table = QTableWidget(len(entries), len(headers))
            table.setHorizontalHeaderLabels(headers)
            table.setStyleSheet("QTableWidget { font-size: 12px; }")
            # Populate the table with values from each entry
            for row_idx, entry in enumerate(entries):
                for col_idx, key in enumerate(headers):
                    value = str(entry.get(key, ""))
                    table.setItem(row_idx, col_idx, QTableWidgetItem(value))
            layout.addWidget(table)
        else:
            # Show a friendly message if there's no data
            no_label = QLabel("No data available for this category.")
            no_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(no_label)

        self.data_window.setLayout(layout)
        self.data_window.show()

    def read_last_message(self):
        """Read the last assistant's message aloud using TTS and update status label."""
        if not self.messages:
            return

        # Find the last assistant's message
        for speaker, message in reversed(self.messages):
            if speaker == "Assistant":
                self.status_label.setText("🔊 TTS: Processing...")
                QApplication.processEvents()
                self.tts_manager.read_text(message)
                time.sleep(3)
                self.status_label.setText("✅ TTS: Done!")
                break

    def Show_QRCode(self):
        self.qr_window = QWidget()
        self.qr_window.setWindowTitle("QRCode")
        self.qr_window.setFixedSize(260, 260)

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setAlignment(Qt.AlignCenter)

        label = QLabel()
        pixmap = QPixmap("QRCode.png") \
            .scaled(240, 240, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignCenter)

        layout.addWidget(label)
        self.qr_window.setLayout(layout)

        self.qr_window.show()

# -----------------------------------------------------------------------------
# Main execution block
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    manager = TTSManager()
    app = QApplication(sys.argv)
    window = VoiceRecorder()
    window.show()
    sys.exit(app.exec_())