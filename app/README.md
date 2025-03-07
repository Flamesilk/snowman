# Voice Assistant with PipeCat and Daily

A simple voice assistant implementation using PipeCat and Daily.co for real-time audio processing.

## Features

- Always-on voice assistant that can be activated with wake words
- Natural conversation with AI using Google's Gemini model
- Speech recognition using Silero (local, no API key needed)
- Text-to-speech using Silero (local, no API key needed)
- Wake word detection using Picovoice Porcupine
- Handles interruptions and turn detection

## Setup

1. Run the setup script to create a virtual environment, install dependencies, and create your .env file:
   ```
   python setup.py
   ```

   The setup script will:
   - Create a virtual environment in the `venv` directory
   - Install all required dependencies in the virtual environment
   - Create a `.env` file from the template
   - Provide instructions for obtaining API keys

2. **Activate the virtual environment** (this step is required):
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     source venv/bin/activate
     ```

   You'll know the environment is activated when you see `(venv)` at the beginning of your command prompt, like this:
   ```
   (venv) $
   ```

   ⚠️ **Important for macOS users**: You must use `source venv/bin/activate` to activate the environment. Simply running the script without activation will not work.

3. Edit the `.env` file with your API keys for:
   - Daily.co (for WebRTC)
   - Google (for Gemini AI model)
   - Picovoice Porcupine (for wake word detection)

   Note: By default, the assistant uses Silero for both speech recognition and text-to-speech, which runs locally and doesn't require API keys.

## Running the Voice Assistant

Make sure your virtual environment is activated (you should see `(venv)` in your prompt) before running these commands:

### 1. Simple Assistant (No Wake Word)

This version is always listening and doesn't require wake word detection:

```
python simple_assistant.py
```

### 2. Wake Word Assistant

This version starts in standby mode and activates when it hears a wake word:

```
python wake_word_assistant.py
```

The assistant will start in standby mode, listening for wake words. Once activated with a wake word (default: "hey assistant" or "computer"), it will begin a conversation and respond to your queries.

## Configuration

You can customize the following in the `.env` file:

### Required Configuration
- `DAILY_ROOM_URL`: Your Daily.co room URL
- `GOOGLE_API_KEY`: Your Google API key for Gemini
- `PORCUPINE_ACCESS_KEY`: Your Picovoice Porcupine access key (for wake word detection)

### Speech Recognition Options
- By default, the assistant uses Silero (local, no API key needed)
- To use Deepgram (cloud-based) instead:
  - Set `USE_SILERO_ASR=false`
  - Add your `DEEPGRAM_API_KEY`

### Text-to-Speech Options
- By default, the assistant uses Silero (local, no API key needed)
- You can customize the Silero voice with `SILERO_VOICE` (e.g., "en_0", "en_1")
- To use Cartesia (cloud-based) instead:
  - Set `USE_SILERO_TTS=false`
  - Add your `CARTESIA_API_KEY` and `CARTESIA_VOICE_ID`

### Wake Word Options
- `WAKE_KEYWORDS`: Comma-separated list of wake words to activate the assistant

## Architecture

The voice assistant uses:
- PipeCat for pipeline orchestration
- Daily for WebRTC audio transport
- Google Gemini for AI conversation
- Silero for speech recognition (local)
- Silero for text-to-speech (local)
- Picovoice Porcupine for wake word detection

## About the Services

### Google Gemini
Google's Gemini is a powerful AI model that can understand and generate natural language. It powers the conversation capabilities of the assistant.

### Silero (Speech Recognition and Text-to-Speech)
Silero is an open-source toolkit that provides speech recognition and text-to-speech capabilities. It runs completely locally on your machine, requiring no API keys or internet connection for these functions. This makes it:
- Free to use with no usage limits
- Privacy-friendly (your voice data stays on your device)
- Works offline
- Lower latency than cloud solutions

The trade-off is that it may have slightly lower accuracy than cloud-based solutions.

### Picovoice Porcupine
Porcupine is a wake word detection engine that can recognize custom wake words. It offers a free tier with limited usage.
