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
  ```

## Setup

1. Copy the example environment file and fill in your credentials:

   ```bash
   cp _.env.example .env
   ```

   | Variable | Purpose | Default |
   | --- | --- | --- |
   | `METIS_API_KEY` | API key for chat + TTS. Prompted for interactively if missing. | — |
   | `METIS_BASE_URL` | OpenAI-compatible endpoint. | — |
   | `GPT_MODEL` | Model used by the agents. | `gpt-4o` |
   | `TTS_MODEL` | Model used for speech synthesis. | `gpt-4o-mini-tts` |

2. Run the app:

   ```bash
   python Hackaton.py
   ```

## Project Structure

```
Hackaton.py                     # launcher
hackaton/
  config.py                     # env vars, paths, models, category table
  app.py                        # wires everything together and starts the GUI
  data/catalog.py               # ProductCatalog — parses final_data.csv once
  search/keyword_searcher.py    # plain keyword search + ChromaDB search
  agents/
    prompts.py                  # agent system messages
    tools.py                    # perform_search / handle_results tools
    team.py                     # builds the AutoGen group chat
  audio/
    recorder.py                 # microphone capture -> WAV
    transcriber.py              # WAV -> Persian text
    tts.py                      # text -> speech playback
  ui/
    main_window.py              # VoiceRecorder window
    windows.py                  # graph / data table / QR code pop-ups
    styles.py                   # Qt stylesheets
```

### How a query flows

`VoiceRecorder` records audio → `transcribe_wav` turns it into Persian text → `AgentTeam.run` starts the group chat, where `search_agent` calls `perform_search`, `check_agent` decides via `handle_results` whether to widen the search (`custom_search_agent`) or rank the hits (`score_agent`) → the reply is shown in the transcript and spoken by `TTSManager`.

## Data

`final_data.csv` holds one column per category (`ماشین`, `لپ تاپ`, `تلفن همراه`), each cell being a Python dict describing a single advertisement. On startup the file is parsed once into per-category tables that feed both the data viewer and the search index. The ChromaDB collection is rebuilt from scratch in `.chroma_keyword_db` on every run.
