# Snowman

A modern voice assistant that combines state-of-the-art AI models with efficient local processing for a seamless conversational experience. Snowman demonstrates how to build a capable voice assistant by leveraging the best of both worlds: local processing for speed-critical components and cloud APIs for intelligent responses.

## Motivation

Since the AI revolution began, I've been waiting for smart speakers to deliver truly intelligent conversations - but they've consistently fallen short of expectations. What I really wanted was simple: a portable device that sits at my dining table where family and friends can casually throw questions at it and get thoughtful, real-time answers without reaching for phones or typing queries. Snowman is built to be that knowledgeable companion, making AI-powered conversations as natural as talking to someone at the table.

## Features

- **Wake Word Detection**: Uses Porcupine for reliable wake word detection
- **Voice Activity Detection**: Implements Cobra VAD for accurate speech detection
- **Speech Recognition**: Employs faster-whisper for local, offline speech-to-text conversion
- **Natural Language Understanding**: Utilizes Google's Gemini for AI-powered responses
- **Text-to-Speech**: Uses Edge TTS for high-quality, local voice synthesis
- **Multilingual Support**: Handles both English and Chinese conversations seamlessly
- **Web Search Integration**: Optional Tavily search integration for real-time information
- **Interruption Handling**: Supports interrupting the assistant while speaking
- **Audio Monitoring**: Real-time audio level monitoring and visualization (in debug mode)

## Key Technical Implementations

### Core Components

- **Audio Processing**: Uses `pvrecorder` for high-quality audio capture and processing
- **Wake Word System**: Implements Porcupine wake word detection for hands-free activation
- **Speech Recognition**: Integrates faster-whisper for efficient, local speech recognition
- **Language Model**: Leverages Gemini for context-aware, natural language responses
- **Voice Synthesis**: Implements Edge TTS for fast, high-quality speech synthesis

### Advanced Features

- **Conversation Management**:
  - Intelligent conversation flow with automatic language detection
  - Timeout handling for user inactivity
  - Session statistics tracking

- **Audio Handling**:
  - Real-time voice activity detection
  - Dynamic audio level monitoring
  - Cross-platform audio playback support

- **Performance Optimization**:
  - Asynchronous audio processing
  - Efficient resource management
  - Optimized model loading and inference

## Requirements

- Python 3.8+
- Required API Keys:
  - Google API Key (for Gemini)
  - PicoVoice Access Key (for voice activity and wake word detection)
  - Tavily API Key (for web search)

## Installation on Raspberry Pi

1. Ensure you have the following:
   - A Raspberry Pi (4 or newer recommended)
   - SSH access to your Raspberry Pi
   - A USB sound card or compatible audio HAT/pHAT is recommended for audio recording (For example, [this one](https://www.amazon.com/dp/B09T33LJQM?ref_=ppx_hzsearch_conn_dt_b_fed_asin_title_1&th=1))

2. Copy `.env.example` to `.env` file in the `local` directory, and edit it with your API keys.

3. Run the installation script:
```bash
./local/install.sh -h <pi_hostname/ip> -u <username> [-p <port>]
```
Example:
```bash
./local/install.sh -h 192.168.1.100 -u pi
```

The script will:
- Set up the required system environment
- Install all dependencies
- Configure audio devices
- Create and start a systemd service
- Set appropriate permissions

After installation, you can manage the service using:
```bash
# Start the service
sudo systemctl start voice-assistant.service

# Check status
sudo systemctl status voice-assistant.service

# View logs
sudo journalctl -u voice-assistant.service -f
```

## Test on PC

1. Create and activate a virtual environment:
```bash
cd local
python3 -m venv venv
source venv/bin/activate
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your environment variables as described above.

4. Run the assistant:
```bash
python local/simple_local_assistant.py
```

## License

MIT
