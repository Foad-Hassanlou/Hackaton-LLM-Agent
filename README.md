# Persian Voice Search Application

This is a PyQt5-based desktop application for Persian voice search, integrating speech recognition, text-to-speech (TTS), and product search functionalities. It allows users to search for products (cars, laptops, phones) using Persian voice commands, processes queries with AI agents, and displays results in a GUI with TTS feedback.

## Features
- **Voice Input**: Records and transcribes Persian speech using `SpeechRecognition`.
- **Product Search**: Searches products in a CSV dataset using `KeywordSearcher` with simple and ChromaDB-backed search.
- **AI Agents**: Uses `autogen` for multi-agent conversation to process queries and refine results.
- **TTS Output**: Reads results aloud in Persian using OpenAI's TTS.
- **GUI**: Displays results, data tables, graphs, and QR codes with PyQt5.
- **Categories**: Supports searching for cars (خودرو), laptops (لپ‌تاپ), and phones (تلفن همراه).

## Prerequisites
- Python 3.8+
- Install dependencies from `requirements.txt`:
  ```bash
  pip install -r requirements.txt
