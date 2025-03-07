# Simple Local Voice Assistant

A powerful, privacy-focused voice assistant that runs entirely on your local machine. It uses Google's Gemini model for conversation, Whisper for speech recognition, Edge TTS for voice synthesis, and Porcupine for wake word detection.

## Features

- 🎙️ **Wake Word Detection**: Uses Porcupine to detect wake words like "computer" or "alexa"
- 🗣️ **Speech Recognition**: Uses OpenAI's Whisper for accurate, local speech-to-text
- 💬 **Conversation**: Powered by Google's Gemini model for natural, context-aware responses
- 🔊 **Text-to-Speech**: Uses Microsoft Edge TTS for high-quality voice synthesis
- 🌐 **Multilingual**: Supports both English and Chinese (see CHINESE_SUPPORT.md)
- 🔒 **Privacy-Focused**: All core functionality runs locally on your machine

## Setup

1. Clone this repository
2. Run the setup script:
   ```
   python setup.py
   ```
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`
4. Copy `.env.template` to `.env` and add your Google API key

## Running the Assistant

Basic usage:
```
python simple_local_assistant.py
```

With Chinese language support:
```
python simple_local_assistant.py --language chinese
```

## How It Works

1. **Wake Word Detection**: Porcupine listens for the wake word in a low-power mode
2. **Speech Recognition**: When activated, Whisper converts your speech to text
   - Uses the base model (~140MB) for good accuracy and performance
   - Runs entirely locally - no internet connection needed for speech recognition
   - Automatically uses GPU if available
3. **Conversation**: Your text is sent to Gemini, which generates a natural response
4. **Voice Synthesis**: The response is converted to speech using Edge TTS

## Configuration

Edit the `.env` file to customize:
- `GOOGLE_API_KEY`: Your Google API key for Gemini
- `WAKE_WORD`: Choose between "computer", "alexa", etc.
- `TTS_VOICE`: The Edge TTS voice to use
- `CHINESE_TTS_VOICE`: The Chinese voice (if using Chinese mode)

## Dependencies

Core components:
- OpenAI Whisper: Speech recognition (requires ffmpeg)
- Google Gemini: Conversation AI
- Microsoft Edge TTS: Text-to-speech
- Porcupine: Wake word detection
- PyAudio: Audio input/output
- FFmpeg: Required for audio processing (installed automatically on macOS/Linux)

System requirements:
- Python 3.8 or higher
- FFmpeg (installed automatically by setup script on macOS/Linux)
- PortAudio (for PyAudio, installed automatically on macOS)

The setup script will attempt to install FFmpeg automatically:
- On macOS: Using Homebrew
- On Linux: Using apt-get or yum
- On Windows: Manual installation required (see setup instructions)

## Troubleshooting

### Audio Issues
The assistant uses a multi-layered approach for audio playback:
1. System audio player (afplay on macOS, etc.)
2. PyAudio fallback with librosa/soundfile
3. Text fallback if audio fails

If you experience audio issues:
- Check your system's default audio devices
- Try adjusting PyAudio settings (SAMPLE_RATE, CHUNK_SIZE, FORMAT)
- Make sure no other applications are blocking the audio device

### Speech Recognition Issues
- Speak clearly and in a quiet environment
- For better accuracy, you can modify the code to use a larger Whisper model
- Check that the Whisper model was downloaded correctly during setup

### Wake Word Detection
- Make sure your microphone is working and selected as the default input device
- Try speaking the wake word clearly and at a normal volume
- Adjust your microphone's input level if needed

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
