# Persian Voice Search App

This desktop application, developed with PyQt5, enables voice-driven product search in Persian. It combines speech recognition, text-to-speech (TTS), and intelligent AI processing to allow users to search for products like cars, laptops, and mobile phones using spoken commands. Results are displayed in a user-friendly interface and can be read aloud using TTS.

## Key Features

- **Voice Recognition**: Captures and transcribes Persian speech using the `SpeechRecognition` library.
- **Product Search**: Performs keyword-based searches on a CSV dataset using `KeywordSearcher`, with support for both basic and ChromaDB-enhanced modes.
- **AI Agent Integration**: Uses `autogen` for multi-agent communication to interpret and refine user queries.
- **Text-to-Speech (TTS)**: Delivers search results audibly in Persian using OpenAI’s TTS.
- **Graphical Interface**: Built with PyQt5 to show results, data tables, graphs, and QR codes.
- **Search Categories**: Supports خودرو (cars), لپ‌تاپ (laptops), and تلفن همراه (mobile phones).

## Requirements

- Python 3.8 or higher
- Install dependencies with:

  ```bash
  pip install -r requirements.txt
