#!/usr/bin/env python3
"""
Simple Local Voice Assistant

A lightweight voice assistant implementation that uses:
- PvRecorder for audio recording
- Porcupine for wake word detection
- Whisper for speech recognition (local, no API key needed)
- Google's Gemini for AI responses
- Edge TTS for text-to-speech (local, no API key needed)

This implementation runs entirely locally except for Gemini API calls,
with no dependency on PipeCat or Daily.
"""

import os
import sys
import time
import queue
import threading
import argparse
import wave
import json
import numpy as np
import google.generativeai as genai
import asyncio
import edge_tts
import io
import tempfile
import subprocess
from dotenv import load_dotenv
import pvporcupine
from pvrecorder import PvRecorder
from tavily import TavilyClient

# Use faster-whisper for speech recognition
from faster_whisper import WhisperModel

# Constants
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 512
SILENCE_THRESHOLD = 0.025
SILENCE_DURATION = 1.5
INITIAL_SILENCE_DURATION = 0.4  # Shorter silence duration for initial question after wake word
INTERRUPTION_THRESHOLD = 0.1
INTERRUPTION_MIN_CHUNKS = 3
DEBUG_AUDIO = True
MAX_RECORDING_TIME = 20
USE_FIXED_THRESHOLDS = True
USE_MANUAL_RECORDING = False
ENABLE_INTERRUPTION = False
USE_EDGE_TTS = True
CONVERSATION_TIMEOUT = 30  # Timeout in seconds for no user input

# Sound effect paths
SOUND_EFFECTS = {
    "wake": "sounds/wake.mp3",  # Sound played when wake word is detected
    "start_listening": "sounds/start_listening.m4a",  # Sound played when starting to listen
    "start_transcribe": "sounds/start_transcribe.mp3",  # Sound played before starting transcription
    "pre_response": "sounds/pre_response.mp3",  # Sound played before getting AI response
    "goodbye_en": "sounds/goodbye_en.m4a",  # English goodbye message
    "goodbye_zh": "sounds/goodbye_zh.m4a",  # Chinese goodbye message
    "not_understood_en": "sounds/not_understood_en.m4a",  # English not understood message
    "not_understood_zh": "sounds/not_understood_zh.m4a",  # Chinese not understood message
}

# Pre-recorded messages mapping
PRE_RECORDED_MESSAGES = {
    "goodbye": {
        "english": "goodbye_en",
        "chinese": "goodbye_zh"
    },
    "not_understood": {
        "english": "not_understood_en",
        "chinese": "not_understood_zh"
    }
}

# Search-related constants
ENABLE_SEARCH = True

# TTS Voice settings
EDGE_TTS_VOICES = {
    "english": "en-US-JennyNeural",
    "chinese": "zh-CN-XiaoxiaoNeural",
    "others": "en-US-EmmaMultilingualNeural",
    # Add more languages as needed:
    # "japanese": "ja-JP-KeitaNeural",
    # "korean": "ko-KR-InJoonNeural",
    # "spanish": "es-ES-AlvaroNeural",
    # "french": "fr-FR-HenriNeural",
}

# Default voices
EDGE_TTS_VOICE = EDGE_TTS_VOICES["english"]
CHINESE_EDGE_TTS_VOICE = EDGE_TTS_VOICES["chinese"]

# Wake word settings
DEFAULT_WAKE_KEYWORDS = ["computer", "alexa", "hey siri", "jarvis"]
END_CONVERSATION_PHRASES = ["goodbye", "bye", "end conversation", "stop listening", "thank you", "thanks"]
CHINESE_END_CONVERSATION_PHRASES = [
    # Simplified Chinese
    "再见", "拜拜", "结束对话", "谢谢", "谢谢你",
    # Traditional Chinese
    "再見", "拜拜", "結束對話", "謝謝", "謝謝你",
]
LANGUAGE = "english"

# System prompt for Gemini to generate concise responses
SYSTEM_PROMPT = """
You are a friendly and witty voice assistant. Please provide concise, direct answers optimized for text-to-speech conversion:

1. Keep responses brief but engaging
2. Use simple language and short sentences
3. Avoid special characters, emojis, or symbols
4. Don't use markdown formatting, code blocks, or technical syntax
5. Don't include URLs or links
6. Avoid parentheses, brackets, or other text decorations
7. Write numbers as words for better speech synthesis
8. Use natural, conversational language
9. Limit response to 1-2 sentences when possible
10. Be charming but not over-the-top silly

Important: You should detect the language of the user's input and respond in the same language.
For Chinese input, respond in Simplified Chinese.
For English input, respond in English.
For other languages, try to respond in the same language if possible, otherwise use English.
"""

class SimpleLocalAssistant:
    def __init__(self, use_wake_word=True, debug=False, language="english"):
        """Initialize the voice assistant"""
        # Load environment variables
        load_dotenv()

        # Set debug mode
        global DEBUG_AUDIO
        DEBUG_AUDIO = debug
        if DEBUG_AUDIO:
            print("🔍 Debug mode enabled - will print audio volume levels")

        # Set initial language (will be updated based on speech detection)
        self.language = language.lower()
        print(f"🌐 Initial language set to: {self.language}")

        # Check required environment variables
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        if not self.google_api_key:
            print("❌ GOOGLE_API_KEY is required in .env file")
            sys.exit(1)

        print(f"Using Google API key: {self.google_api_key[:5]}...{self.google_api_key[-5:]}")

        # Set thresholds
        if USE_FIXED_THRESHOLDS:
            self.silence_threshold = SILENCE_THRESHOLD
            self.interruption_threshold = INTERRUPTION_THRESHOLD
            print(f"Using fixed thresholds - Silence: {self.silence_threshold}, Interruption: {self.interruption_threshold}")
        else:
            # Calibrate microphone to set thresholds
            print("Calibrating microphone for dynamic thresholds...")
            self.silence_threshold, self.interruption_threshold = self.calibrate_microphone()
            print(f"Using calibrated thresholds - Silence: {self.silence_threshold}, Interruption: {self.interruption_threshold}")

        # Initialize speech recognition
        self.init_speech_recognition()

        # Initialize TTS
        self.use_silero_tts = False
        self.init_edge_tts()

        # Initialize wake word detection if enabled
        self.use_wake_word = use_wake_word
        if self.use_wake_word:
            self.init_porcupine()

        # Initialize Gemini model
        self.init_gemini()

        # Initialize chat session
        self.init_chat_session()

        # Initialize search APIs if enabled
        if ENABLE_SEARCH:
            self.init_search_apis()

        # Initialize timing stats
        self.search_times = []  # Add search timing stats

        # State variables
        self.is_listening = False
        self.is_speaking = False
        self.should_exit = False
        self.audio_queue = queue.Queue()

        print("✅ Voice assistant initialized and ready")

    def calibrate_microphone(self):
        """Measure ambient noise and calibrate thresholds using PvRecorder"""
        print("🎙️ Calibrating microphone (please be quiet)...")

        recorder = None
        try:
            recorder = PvRecorder(device_index=-1, frame_length=CHUNK_SIZE)
            recorder.start()

            # Collect ambient noise samples
            ambient_levels = []
            calibration_time = 2  # seconds
            samples_to_collect = int(calibration_time * SAMPLE_RATE / CHUNK_SIZE)

            for _ in range(samples_to_collect):
                try:
                    pcm = recorder.read()
                    audio_data = np.array(pcm, dtype=np.int16)
                    volume_norm = np.abs(audio_data).mean() / 32768.0
                    ambient_levels.append(volume_norm)
                except Exception as e:
                    print(f"Error during calibration: {e}")

        finally:
            if recorder is not None:
                recorder.stop()
                recorder.delete()

        # Calculate thresholds based on ambient noise
        if ambient_levels:
            avg_ambient = sum(ambient_levels) / len(ambient_levels)
            max_ambient = max(ambient_levels)

            # Set thresholds relative to ambient noise
            silence_threshold = max(avg_ambient * 1.2, 0.003)
            interruption_threshold = max(max_ambient * 2, 0.01)

            print(f"Ambient noise level: {avg_ambient:.4f}")
            print(f"Silence threshold set to: {silence_threshold:.4f}")
            print(f"Interruption threshold set to: {interruption_threshold:.4f}")

            return silence_threshold, interruption_threshold
        else:
            # Fallback to default values
            print("⚠️ Calibration failed, using default thresholds")
            return SILENCE_THRESHOLD, INTERRUPTION_THRESHOLD

    def init_speech_recognition(self):
        """Initialize Whisper speech recognition model for both English and Chinese"""
        print("Loading Whisper ASR model...")
        try:
            # Use small model for better accuracy while maintaining reasonable performance
            model_size = "small"
            # Use CPU for better stability
            device = "cpu"
            compute_type = "int8"

            print(f"Using faster-whisper on {device} with compute type {compute_type}")

            try:
                # First try to load from local cache only
                print("Attempting to load model from local cache...")
                self.whisper_model = WhisperModel(
                    model_size,
                    device=device,
                    compute_type=compute_type,
                    cpu_threads=2,
                    num_workers=1,
                    download_root="models",
                    local_files_only=True
                )
            except Exception as cache_error:
                print(f"Model not found in cache, downloading {model_size} model (this may take a while)...")
                # If local load fails, download the model
                self.whisper_model = WhisperModel(
                    model_size,
                    device=device,
                    compute_type=compute_type,
                    cpu_threads=2,
                    num_workers=1,
                    download_root="models",
                    local_files_only=False  # Allow downloading
                )

            print(f"✅ Whisper {model_size} model loaded on {device}")
        except Exception as e:
            print(f"❌ Error loading Whisper model: {e}")
            print("Speech recognition may not work properly.")
            self.whisper_model = None

    def init_edge_tts(self):
        """Initialize Edge TTS"""
        try:
            # Get voice from environment or use default based on language
            if self.language == "chinese":
                self.edge_tts_voice = os.getenv("CHINESE_EDGE_TTS_VOICE", CHINESE_EDGE_TTS_VOICE)
            else:
                self.edge_tts_voice = os.getenv("EDGE_TTS_VOICE", EDGE_TTS_VOICE)

            # List available voices
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            voices = loop.run_until_complete(edge_tts.list_voices())
            loop.close()

            # Check if the voice is available
            voice_names = [voice["ShortName"] for voice in voices]
            if self.edge_tts_voice not in voice_names:
                print(f"⚠️ Voice '{self.edge_tts_voice}' not found. Available voices for {self.language}:")
                for voice in voices:
                    if (self.language == "chinese" and voice["Locale"].startswith("zh-")) or \
                       (self.language == "english" and voice["Locale"].startswith("en-")):
                        print(f"  - {voice['ShortName']} ({voice['Locale']})")

                # Set default fallback voice based on language
                if self.language == "chinese":
                    self.edge_tts_voice = CHINESE_EDGE_TTS_VOICE
                else:
                    self.edge_tts_voice = EDGE_TTS_VOICE
                print(f"Using fallback voice: {self.edge_tts_voice}")

            print(f"✅ Edge TTS initialized with voice: {self.edge_tts_voice}")
        except Exception as e:
            print(f"❌ Error initializing Edge TTS: {e}")
            sys.exit(1)

    def init_porcupine(self):
        """Initialize Porcupine wake word detection"""
        access_key = os.getenv("PORCUPINE_ACCESS_KEY")
        if not access_key:
            print("❌ PORCUPINE_ACCESS_KEY is required in .env file for wake word detection")
            sys.exit(1)

        # Check for custom wake word file
        custom_keyword_path = os.getenv("CUSTOM_KEYWORD_PATH")
        if custom_keyword_path and os.path.exists(custom_keyword_path):
            print(f"🔍 Using custom wake word from: {custom_keyword_path}")
            try:
                self.porcupine = pvporcupine.create(
                    access_key=access_key,
                    keyword_paths=[custom_keyword_path]
                )
                self.keywords = ["hey snowman"]  # For display purposes
                print(f"✅ Porcupine initialized with custom wake word: {self.keywords[0]}")
                print(f"Sample rate: {self.porcupine.sample_rate}")
                print(f"Frame length: {self.porcupine.frame_length}")
                return
            except Exception as e:
                print(f"❌ Failed to initialize Porcupine with custom wake word: {str(e)}")
                print("Falling back to default keywords...")

        # If no custom wake word or failed to load it, use default keywords
        wake_keywords_str = os.getenv("WAKE_KEYWORDS", ",".join(DEFAULT_WAKE_KEYWORDS))
        requested_keywords = [kw.strip() for kw in wake_keywords_str.split(",")]
        print(f"Requested wake words: {requested_keywords}")

        # Filter to only use available default keywords
        available_keywords = [
            "picovoice", "ok google", "hey google", "hey barista", "terminator",
            "americano", "grasshopper", "porcupine", "pico clock", "grapefruit",
            "bumblebee", "computer", "alexa", "hey siri", "jarvis", "blueberry"
        ]

        # Find keywords that are both requested and available
        self.keywords = [kw for kw in requested_keywords if kw.lower() in [k.lower() for k in available_keywords]]
        print(f"Valid wake words found: {self.keywords}")

        # If no valid keywords, use default ones
        if not self.keywords:
            print(f"⚠️ No valid wake keywords found in '{wake_keywords_str}'. Using defaults: {DEFAULT_WAKE_KEYWORDS}")
            self.keywords = DEFAULT_WAKE_KEYWORDS

        print(f"🔍 Attempting to initialize Porcupine with keywords: {self.keywords}")
        try:
            self.porcupine = pvporcupine.create(
                access_key=access_key,
                keywords=self.keywords
            )
            print(f"✅ Porcupine initialized with wake words: {', '.join(self.keywords)}")
            print(f"Sample rate: {self.porcupine.sample_rate}")
            print(f"Frame length: {self.porcupine.frame_length}")
        except Exception as e:
            print(f"❌ Failed to initialize Porcupine: {str(e)}")
            print("\nAvailable default keywords are:")
            for kw in available_keywords:
                print(f"  - {kw}")
            print("\nPlease:")
            print("1. Verify your access key at https://console.picovoice.ai/")
            print("2. Make sure you're using keywords from the list above")
            print("3. Check that there are no spaces after commas in WAKE_KEYWORDS")
            sys.exit(1)

    def init_chat_session(self):
        """Initialize a new chat session with the system prompt"""
        try:
            # Create a new chat session with the language-aware system prompt
            self.chat_session = self.model.start_chat(
                history=[
                    {"role": "user", "parts": [SYSTEM_PROMPT]},
                    {"role": "model", "parts": ["I'll keep my responses concise and adapt to the user's language."]}
                ]
            )
            print("🔄 Started new chat session with language adaptation")
        except Exception as e:
            print(f"❌ Error initializing chat session: {e}")
            sys.exit(1)

    def init_gemini(self):
        """Initialize Google Gemini model"""
        print("Initializing Gemini model...")
        try:
            if not self.google_api_key:
                print("❌ Google API key not found. Please set GOOGLE_API_KEY in your .env file.")
                sys.exit(1)

            # Configure the Gemini API
            genai.configure(api_key=self.google_api_key)

            # Set up the model
            self.generation_config = {
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 1024,
            }

            # Create the model
            self.model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                generation_config=self.generation_config
            )

            print("✅ Gemini Flash model initialized")
        except Exception as e:
            print(f"❌ Error initializing Gemini model: {e}")
            sys.exit(1)

    def init_search_apis(self):
        """Initialize search API configurations"""
        try:
            # Initialize Tavily client
            self.tavily_api_key = os.getenv("TAVILY_API_KEY")
            if self.tavily_api_key:
                self.tavily_client = TavilyClient(api_key=self.tavily_api_key)

                # Test the API with a simple query
                try:
                    test_params = {
                        "query": "test",
                        "search_depth": "basic",
                        "max_results": 1
                    }
                    print("🔍 Testing Tavily API connection...")
                    test_result = self.tavily_client.search(**test_params)
                    if test_result:
                        print("✅ Tavily Search API test successful")
                    print("✅ Tavily Search API initialized")
                except Exception as test_error:
                    print(f"❌ Tavily API test failed: {test_error}")
                    if hasattr(test_error, 'response'):
                        try:
                            error_details = test_error.response.json()
                        except:
                            error_details = test_error.response.text if hasattr(test_error.response, 'text') else str(test_error)
                        print(f"Error details: {error_details}")
                    raise Exception(f"Tavily API test failed: {test_error}")
            else:
                print("⚠️ TAVILY_API_KEY not found in .env file")
                print("Search functionality will be limited")
        except Exception as e:
            print(f"⚠️ Error initializing search APIs: {e}")
            print("Search functionality may be limited")
            self.tavily_client = None  # Ensure client is None if initialization fails

    def play_sound_effect(self, effect_name):
        """Play a sound effect from the sounds directory"""
        if effect_name not in SOUND_EFFECTS:
            print(f"⚠️ Sound effect {effect_name} not found")
            return

        sound_path = os.path.join(os.path.dirname(__file__), SOUND_EFFECTS[effect_name])
        if not os.path.exists(sound_path):
            print(f"⚠️ Sound file not found: {sound_path}")
            return

        try:
            if sys.platform == "darwin":  # macOS
                subprocess.run(["afplay", sound_path], check=True)
            elif sys.platform == "win32":  # Windows
                subprocess.run(["start", sound_path], shell=True, check=True)
            elif sys.platform.startswith("linux"):  # Linux
                for player in ["mpg123", "mpg321", "mplayer", "play"]:
                    try:
                        subprocess.run([player, sound_path], check=True)
                        break
                    except (subprocess.SubprocessError, FileNotFoundError):
                        continue
        except Exception as e:
            print(f"⚠️ Error playing sound effect: {e}")

    def listen_for_wake_word(self):
        """Listen for wake word using Porcupine and capture any following speech"""
        print("👂 Listening for wake word...")

        recorder = None
        try:
            # Initialize recorder with frame length 512 (works for both wake word and speech)
            recorder = PvRecorder(device_index=-1, frame_length=self.porcupine.frame_length)
            recorder.start()

            print(f"Using audio device: {recorder.selected_device}")
            print(f"Wake word frame length: {self.porcupine.frame_length}")
            print(f"Audio chunk size: {CHUNK_SIZE}")
            print(f"Silence threshold: {SILENCE_THRESHOLD}")

            # Buffer to store audio frames after wake word
            post_wake_frames = []
            is_collecting = False
            silent_chunks = 0
            required_silent_chunks = int(INITIAL_SILENCE_DURATION * SAMPLE_RATE / CHUNK_SIZE)
            speech_detected = False
            collection_start_time = None

            # Create a small circular buffer to keep the last few frames before wake word
            pre_wake_buffer = []
            PRE_WAKE_BUFFER_SIZE = 20  # Increased buffer size to capture more context

            while not self.should_exit:
                try:
                    pcm = recorder.read()

                    if not is_collecting:
                        # Keep the last few frames in a circular buffer
                        pre_wake_buffer.append(np.array(pcm, dtype=np.int16).tobytes())
                        if len(pre_wake_buffer) > PRE_WAKE_BUFFER_SIZE:
                            pre_wake_buffer.pop(0)

                        # Process for wake word detection
                        keyword_index = self.porcupine.process(pcm)
                        if keyword_index >= 0:
                            print("🎯 Wake word detected!")

                            is_collecting = True
                            post_wake_frames = []

                            # Keep a few frames from before the wake word to catch early speech
                            post_wake_frames.extend(pre_wake_buffer[-10:])  # Keep last 5 frames
                            pre_wake_buffer = []

                            # Start collecting immediately without delay
                            silent_chunks = 0
                            speech_detected = False
                            collection_start_time = time.time()
                            print("🎤 Started collecting speech after wake word...")
                            continue
                    else:
                        # Collecting speech after wake word
                        # Add the frame to our buffer
                        post_wake_frames.append(np.array(pcm, dtype=np.int16).tobytes())

                        # Calculate volume
                        audio_data = np.array(pcm, dtype=np.int16)
                        volume_norm = np.abs(audio_data).mean() / 32768.0

                        # Print volume for debugging
                        if DEBUG_AUDIO:
                            print(f"Post-wake volume: {volume_norm:.4f} (threshold: {SILENCE_THRESHOLD * 0.6})")

                        # Use an even lower threshold for initial speech detection
                        current_threshold = SILENCE_THRESHOLD * 0.6  # Lower threshold to 60% of silence threshold
                        if volume_norm > current_threshold:
                            if not speech_detected:
                                print(f"Speech detected after wake word! (volume: {volume_norm:.4f})")
                            speech_detected = True
                            silent_chunks = 0
                        else:
                            silent_chunks += 1
                            if DEBUG_AUDIO and silent_chunks > 0:
                                print(f"Silent chunk {silent_chunks}/{required_silent_chunks} (volume: {volume_norm:.4f})")

                        # Calculate elapsed time
                        elapsed_time = time.time() - collection_start_time if collection_start_time else 0

                        # Calculate required silent chunks based on whether speech was detected
                        # Use shorter silence duration for initial question
                        required_silent_chunks = int(INITIAL_SILENCE_DURATION * SAMPLE_RATE / CHUNK_SIZE)

                        # Stop collecting if:
                        # 1. We've detected speech and then silence (normal end)
                        # 2. We've hit the maximum recording time
                        # 3. We've collected too much audio without any speech
                        if ((speech_detected and silent_chunks >= required_silent_chunks) or
                            elapsed_time >= MAX_RECORDING_TIME or
                            (not speech_detected and elapsed_time >= 1.0)):  # Increased timeout to 3 seconds

                            print(f"Stopping collection after {elapsed_time:.1f} seconds")
                            print(f"Speech detected: {speech_detected}")
                            print(f"Silent chunks: {silent_chunks}")
                            print(f"Frames collected: {len(post_wake_frames)}")

                            # If we detected speech, process it
                            if len(post_wake_frames) > 0:
                                if not speech_detected:
                                    # Only play wake sound if no speech was detected
                                    self.play_sound_effect("wake")
                                print("Processing collected audio...")
                                audio_data = b''.join(post_wake_frames)
                                self.handle_conversation(initial_audio=audio_data)
                            else:
                                print("No audio frames collected, starting normal conversation...")
                                # Play wake sound when starting normal conversation
                                self.play_sound_effect("wake")
                                self.handle_conversation()

                            # Reset and continue listening for wake word
                            is_collecting = False
                            post_wake_frames = []
                            pre_wake_buffer = []  # Clear the pre-wake buffer
                            print("👂 Listening for wake word...")

                except Exception as e:
                    print(f"Error processing audio: {e}")
                    import traceback
                    traceback.print_exc()
                    break

        except Exception as e:
            print(f"Error in wake word detection: {e}")
            import traceback
            traceback.print_exc()

        finally:
            if recorder is not None:
                recorder.stop()
                recorder.delete()

    def record_audio(self):
        """Record audio from microphone until silence is detected"""
        if USE_MANUAL_RECORDING:
            return self.record_audio_manual()
        else:
            return self.record_audio_auto()

    def record_audio_manual(self):
        """Record audio with manual control (press Enter to start/stop)"""
        print("🎤 Press Enter to start recording...")
        input()  # Wait for Enter key

        # Play start listening sound
        self.play_sound_effect("start_listening")

        print("🎤 Recording... (press Enter to stop)")
        self.is_listening = True

        frames = []
        recorder = None

        # Create a thread to wait for Enter key
        stop_recording = threading.Event()

        def wait_for_enter():
            input()  # Wait for Enter key
            stop_recording.set()

        input_thread = threading.Thread(target=wait_for_enter)
        input_thread.daemon = True
        input_thread.start()

        # Record until Enter is pressed or timeout
        start_time = time.time()
        try:
            recorder = PvRecorder(device_index=-1, frame_length=CHUNK_SIZE)
            recorder.start()

            while not stop_recording.is_set() and time.time() - start_time < MAX_RECORDING_TIME:
                try:
                    pcm = recorder.read()
                    frames.append(np.array(pcm, dtype=np.int16).tobytes())

                    # Print volume for debugging
                    if DEBUG_AUDIO and len(frames) % 20 == 0:  # Only print every 20 frames
                        audio_data = np.array(pcm, dtype=np.int16)
                        volume_norm = np.abs(audio_data).mean() / 32768.0
                        print(f"Recording volume: {volume_norm:.4f}")
                except Exception as e:
                    print(f"Error reading audio: {e}")
                    break

        finally:
            if recorder is not None:
                recorder.stop()
                recorder.delete()
            self.is_listening = False

            # Print recording stats
            elapsed_time = time.time() - start_time
            print(f"Recording finished after {elapsed_time:.1f} seconds")

        return b''.join(frames)

    def record_audio_auto(self):
        """Record audio from microphone until silence is detected (automatic)"""
        print("🎤 Listening... (automatic mode)")

        # Play start listening sound
        self.play_sound_effect("start_listening")

        # Optionally recalibrate if not using fixed thresholds
        if not USE_FIXED_THRESHOLDS:
            print("Recalibrating thresholds before recording...")
            self.silence_threshold, self.interruption_threshold = self.calibrate_microphone()

        self.is_listening = True
        frames = []
        silent_chunks = 0
        required_silent_chunks = int(SILENCE_DURATION * SAMPLE_RATE / CHUNK_SIZE)

        # Variables to track speech activity
        speech_detected = False
        max_volume = 0.0
        interruption_chunks = 0

        # Initialize PvRecorder
        recorder = None
        try:
            recorder = PvRecorder(device_index=-1, frame_length=CHUNK_SIZE)
            recorder.start()

            print(f"Using audio device: {recorder.selected_device}")

            # Add timeout mechanism
            start_time = time.time()
            max_chunks = int(MAX_RECORDING_TIME * SAMPLE_RATE / CHUNK_SIZE)
            chunk_count = 0

            # Wait a moment before starting to record
            time.sleep(0.1)

            while not self.should_exit and self.is_listening:
                try:
                    # Check for timeout
                    if chunk_count >= max_chunks:
                        print(f"⚠️ Recording timeout after {MAX_RECORDING_TIME} seconds")
                        break

                    # Get audio frame from PvRecorder
                    pcm = recorder.read()
                    frames.append(np.array(pcm, dtype=np.int16).tobytes())
                    chunk_count += 1

                    # Calculate volume using numpy
                    audio_data = np.array(pcm, dtype=np.int16)
                    volume_norm = np.abs(audio_data).mean() / 32768.0
                    max_volume = max(max_volume, volume_norm)

                    # Print volume for debugging
                    if DEBUG_AUDIO:
                        if chunk_count % 10 == 0:  # Only print every 10 frames
                            print(f"Recording volume: {volume_norm:.4f} (max: {max_volume:.4f}, threshold: {SILENCE_THRESHOLD})")

                    # Detect speech
                    if volume_norm > SILENCE_THRESHOLD:
                        speech_detected = True
                        silent_chunks = 0

                        # Check for interruption if enabled and speech was detected
                        if ENABLE_INTERRUPTION and speech_detected:
                            if volume_norm > INTERRUPTION_THRESHOLD:
                                interruption_chunks += 1
                                if interruption_chunks >= INTERRUPTION_MIN_CHUNKS:
                                    print(f"🔊 Interruption detected (volume: {volume_norm:.4f})")
                                    break
                            else:
                                interruption_chunks = 0
                    else:
                        silent_chunks += 1
                        interruption_chunks = 0  # Reset interruption counter during silence
                        # Only end recording if we've detected some speech first
                        if speech_detected and silent_chunks >= required_silent_chunks:
                            if DEBUG_AUDIO:
                                print(f"Silence detected for {SILENCE_DURATION}s after speech, stopping recording")
                                print(f"Final max volume: {max_volume:.4f}")
                            break

                except Exception as e:
                    print(f"Error reading audio: {e}")
                    break

        finally:
            # Clean up PvRecorder
            if recorder is not None:
                recorder.stop()
                recorder.delete()
            self.is_listening = False

            # Print recording stats
            elapsed_time = time.time() - start_time
            print(f"Recording finished after {elapsed_time:.1f} seconds")
            print(f"Max volume detected: {max_volume:.4f}")

        # If we didn't detect any speech, return empty data
        if not speech_detected:
            print("⚠️ No speech detected, please try again")
            return b''

        return b''.join(frames)

    def transcribe_audio(self, audio_data):
        """Convert audio to text using Whisper with automatic language detection"""
        temp_wav = None
        try:
            # Save audio to temporary file using a proper temporary file
            temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
            with wave.open(temp_wav, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)  # 16-bit audio = 2 bytes
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio_data)

            # Check if Whisper model is available
            if not hasattr(self, 'whisper_model') or self.whisper_model is None:
                print("❌ Whisper model not available")
                return ""

            try:
                # Transcribe with Whisper - with automatic language detection
                print("Starting Whisper transcription...")
                print(f"Audio file size: {len(audio_data)} bytes")
                print(f"Audio duration: {len(audio_data) / (SAMPLE_RATE * 2):.2f} seconds")

                # Use faster-whisper API with conservative settings
                segments, info = self.whisper_model.transcribe(
                    temp_wav,
                    beam_size=1,
                    best_of=1,
                    temperature=0.0,
                    condition_on_previous_text=False,
                    no_speech_threshold=0.6,
                    compression_ratio_threshold=2.4,
                    log_prob_threshold=-1.0,
                    vad_filter=True,
                    vad_parameters=dict(
                        min_silence_duration_ms=1000,
                        speech_pad_ms=200,
                        threshold=0.4
                    )
                )

                # Get the detected language
                detected_lang = info.language
                lang_probability = info.language_probability
                print(f"🌐 Detected language: {detected_lang} (probability: {lang_probability:.2f})")

                # Update the assistant's language setting based on detection
                if lang_probability > 0.5:  # Only update if confidence is high enough
                    if detected_lang == "zh":
                        self.language = "chinese"
                    elif detected_lang == "en":
                        self.language = "english"
                    # For other languages, keep current setting but adapt the response

                # Get the transcription text
                transcription = " ".join([segment.text for segment in segments]).strip()
                print(f"🎯 Whisper transcription: {transcription}")
                return transcription
            except Exception as e:
                print(f"❌ Error during Whisper transcription: {e}")
                import traceback
                traceback.print_exc()
                return ""

        except Exception as e:
            print(f"❌ Error preparing audio for transcription: {e}")
            import traceback
            traceback.print_exc()
            return ""
        finally:
            # Clean up temporary file
            if temp_wav and os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except Exception as e:
                    print(f"Warning: Could not remove temporary file {temp_wav}: {e}")
                    pass

    def perform_search(self, query, search_type="general"):
        """
        Perform a search using Tavily API
        search_type can be: "general" or "news"
        """
        if not hasattr(self, 'tavily_client') or self.tavily_client is None:
            raise Exception("Tavily client not initialized")

        if self.language == 'chinese':
            query += " (提供简明扼要的回答，适合口语交流)"
        else:
            query += " (provide a concise answer, suitable for casual conversation)"

        # Maximum number of retries
        max_retries = 2
        retry_count = 0
        last_error = None

        while retry_count <= max_retries:
            try:
                # Convert search type to Tavily parameters
                search_params = {
                    "query": query,
                    "search_depth": "basic",
                    "topic": search_type,  # Use topic parameter for news vs general searches
                    "include_answer": "basic",
                    "include_raw_content": False,
                    "include_images": False,
                    "max_results": 3,
                    "language": 'zh' if self.language == 'chinese' else 'en'
                }

                # For news searches, we can also specify the time range
                if search_type == "news":
                    search_params["time_range"] = "day"  # Get very recent news

                # Log search parameters for debugging
                print(f"🔍 Tavily search parameters (attempt {retry_count + 1}/{max_retries + 1}): {json.dumps(search_params, ensure_ascii=False, indent=2)}")

                # Perform the search
                results = self.tavily_client.search(**search_params)
                # print(f"🔍 Tavily raw response: {json.dumps(results, ensure_ascii=False, indent=2)}")

                # Extract and format the results
                if results.get("answer"):
                    return results["answer"]
                else:
                    formatted_results = []
                    for result in results.get("results", [])[:3]:
                        title = result.get("title", "")
                        snippet = result.get("snippet", "")
                        # For news, include the published date if available
                        if search_type == "news" and result.get("published_date"):
                            formatted_results.append(f"{title} ({result['published_date']}): {snippet}")
                        else:
                            formatted_results.append(f"{title}: {snippet}")

                    if self.language == "chinese":
                        formatted_answer = f"以下是关于'{query}'的搜索结果：\n" + "\n".join(formatted_results)
                    else:
                        formatted_answer = f"Here are the search results for '{query}':\n" + "\n".join(formatted_results)

                    return formatted_answer[:500]  # Limit response length

            except Exception as e:
                last_error = e
                error_details = str(e)

                # Get detailed error information if available
                if hasattr(e, 'response'):
                    try:
                        error_details = e.response.json()
                    except:
                        error_details = e.response.text if hasattr(e.response, 'text') else str(e)

                print(f"❌ Tavily search error (attempt {retry_count + 1}/{max_retries + 1}): {error_details}")

                # Check if we should retry
                if retry_count < max_retries:
                    retry_count += 1
                    print(f"Retrying search (attempt {retry_count + 1}/{max_retries + 1})...")
                    time.sleep(1)  # Wait a second before retrying
                    continue
                else:
                    # If all retries failed, raise the last error
                    raise Exception(f"Search failed after {max_retries + 1} attempts: {error_details}")

        # If we get here, all retries failed
        if self.language == "chinese":
            return f"抱歉，搜索时出现错误。错误信息：{str(last_error)}"
        else:
            return f"Sorry, there was an error performing the search. Error: {str(last_error)}"

    def get_ai_response(self, user_input):
        """Get response from Gemini AI using chat history"""
        try:
            print(f"🧠 Processing: '{user_input}'")

            # Combined decision and response prompt
            decision_prompt = {
                "english": f"""You must respond with a valid JSON object and nothing else.
                Analyze this query: "{user_input}"

                RESPOND WITH ONLY A JSON OBJECT IN THIS EXACT FORMAT:
                {{
                    "need_search": true/false,
                    "response_text": "your response here",
                    "reason": "your reason here"
                }}

                Rules:

                1. Please make a careful decision about if search is needed based on the conversation history, and set need_search accordingly.
                   - Set need_search=true if the query needs a search.
                   - Set need_search=false if the query is a general question that can be answered with the current knowledge.

                2. For response_text:
                   - If need_search=true: Write a brief acknowledgment
                   - If need_search=false: Write the complete answer

                3. Keep reason brief and clear

                4. For search_query:
                   - If need_search=true: Write the search query for Tavily, considering the conversation history, not just the current query.
                   - If need_search=false: Do not include search_query

                IMPORTANT:
                - Use proper JSON formatting with double quotes
                - Do not include any text outside the JSON object
                - Do not include any markdown or formatting
                - Do not include line breaks in strings""",

                "chinese": f"""你必须只返回一个有效的JSON对象，不要包含任何其他内容。
                分析这个问题："{user_input}"

                只返回以下格式的JSON对象：
                {{
                    "need_search": true/false,
                    "response_text": "你的回应",
                    "reason": "原因说明",
                    "search_query": "搜索查询"
                }}

                规则:

                1. 结合对话历史，决定当前是否需要搜索，来设置 need_search
                    - 如果需要搜索，设置 need_search=true
                    - 如果不需要搜索，设置 need_search=false

                2. response_text内容：
                   - 如果need_search=true：写一个简短的确认信息
                   - 如果need_search=false：写出完整答案

                3. reason保持简短明确

                4. search_query内容：
                    - 如果need_search=true：写出搜索查询的问题，供Tavily搜索使用，考虑对话历史，不仅仅是当前查询。
                    - 如果need_search=false：不要包含search_query

                重要提示：
                - 使用正确的JSON格式和双引号
                - 不要在JSON对象外包含任何文本
                - 不要包含任何markdown或格式化
                - 字符串中不要包含换行符"""
            }

            # Get structured response from LLM
            response = self.chat_session.send_message(
                decision_prompt[self.language],
                stream=False
            )

            try:
                # Clean up the response text to ensure valid JSON
                response_text = response.text.strip()
                # Remove any potential markdown code block markers
                response_text = response_text.replace('```json', '').replace('```', '')
                # Remove any leading/trailing whitespace or newlines
                response_text = response_text.strip()

                # Parse the JSON response
                result = json.loads(response_text)

                # Validate required fields
                required_fields = ['need_search', 'response_text', 'reason']
                if not all(field in result for field in required_fields):
                    raise ValueError("Missing required fields in JSON response")

                print(f"🤔 Decision: need_search={result['need_search']}, reason={result['reason']}")

                # If search is needed, start it immediately
                search_results = None
                if ENABLE_SEARCH and result['need_search']:
                    print("🔍 Starting web search...")

                    # Start search in a separate thread
                    search_query = result.get('search_query', user_input)
                    search_thread = threading.Thread(target=lambda: self._perform_search(search_query))
                    search_thread.start()

                    # Speak acknowledgment while search is running
                    self.speak_text(result['response_text'])

                    # Wait for search to complete
                    search_thread.join()

                    # Get search results from the thread
                    search_results = getattr(search_thread, 'search_results', None)
                    search_error = getattr(search_thread, 'search_error', None)

                    if search_error:
                        raise search_error

                    print(f"🔍 Search results: {search_results}")

                    if search_results:
                        # Return search results directly without processing
                        return search_results
                    else:
                        # Fallback if search failed
                        if self.language == "chinese":
                            return "抱歉，搜索结果获取失败。"
                        else:
                            return "Sorry, I couldn't retrieve the search results."
                else:
                    # Return the direct response
                    return result['response_text']

            except (json.JSONDecodeError, ValueError) as e:
                print(f"⚠️ Error parsing LLM response: {e}")
                print(f"Raw response: {response.text}")

                # Fallback: Try to extract a usable response
                if self.language == "chinese":
                    return "抱歉，我遇到了一个处理错误。让我重新组织语言。"
                else:
                    return "I apologize, I encountered a processing error. Let me rephrase that."

        except Exception as e:
            print(f"❌ Error getting AI response: {e}")
            import traceback
            traceback.print_exc()
            return "I'm sorry, I encountered an error processing your request."

    def _perform_search(self, query):
        """Helper method to perform search in a separate thread"""
        try:
            # Determine if it's a news query
            search_type = "general"
            # search_type = "news" if any(word in query.lower() for word in ["news", "新闻", "最新", "最近"]) else "general"
            # if search_type == "news":
            #     print("📰 Using news search")

            # Perform the search with timing
            search_start = time.time()
            search_results = self.perform_search(query, search_type)
            search_time = time.time() - search_start
            self.search_times.append(search_time)
            print(f"🔍 Search completed in {search_time:.2f} seconds")

            # Store results in the thread object
            threading.current_thread().search_results = search_results
        except Exception as e:
            # Store error in the thread object
            threading.current_thread().search_error = e

    def speak_text(self, text):
        """Convert text to speech and play it"""
        self.speak_text_edge(text)

    def speak_text_edge(self, text):
        """Convert text to speech using Edge TTS and play it"""
        try:
            print(f"🔊 Speaking: '{text}'")
            self.is_speaking = True

            # Select voice based on current language
            if self.language == "chinese":
                voice = os.getenv("CHINESE_EDGE_TTS_VOICE", CHINESE_EDGE_TTS_VOICE)
            else:
                voice = os.getenv("EDGE_TTS_VOICE", EDGE_TTS_VOICE)

            # Update the voice if it changed
            if voice != self.edge_tts_voice:
                self.edge_tts_voice = voice
                print(f"Switched TTS voice to: {self.edge_tts_voice}")

            # Create a temporary file for the audio
            temp_file = "temp_tts.mp3"

            # Create a new event loop for async operations
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Function to generate speech with Edge TTS and save to file
            async def generate_speech():
                communicate = edge_tts.Communicate(text, self.edge_tts_voice)
                await communicate.save(temp_file)

            # Generate speech and save to file
            loop.run_until_complete(generate_speech())
            loop.close()

            # Try different methods to play the audio file
            played_successfully = False

            # Method 1: Use system player (most reliable)
            try:
                if sys.platform == "darwin":  # macOS
                    subprocess.run(["afplay", temp_file], check=True)
                    played_successfully = True
                elif sys.platform == "win32":  # Windows
                    subprocess.run(["start", temp_file], shell=True, check=True)
                    played_successfully = True
                elif sys.platform.startswith("linux"):  # Linux
                    for player in ["mpg123", "mpg321", "mplayer", "play"]:
                        try:
                            subprocess.run([player, temp_file], check=True)
                            played_successfully = True
                            break
                        except (subprocess.SubprocessError, FileNotFoundError):
                            continue
            except Exception as e:
                print(f"System audio player error: {e}")

            # Method 2: Last resort - print the text if audio failed
            if not played_successfully:
                print(f"⚠️ Audio playback failed. Text response: {text}")

            # Clean up the temporary file
            try:
                os.remove(temp_file)
            except:
                pass

            self.is_speaking = False
        except Exception as e:
            print(f"❌ Error speaking text with Edge TTS: {e}")
            import traceback
            traceback.print_exc()
            self.is_speaking = False

    def play_pre_recorded_message(self, message_type):
        """Play a pre-recorded message based on type and current language"""
        if message_type in PRE_RECORDED_MESSAGES:
            sound_effect = PRE_RECORDED_MESSAGES[message_type][self.language]
            self.play_sound_effect(sound_effect)
        else:
            print(f"⚠️ No pre-recorded message found for: {message_type}")

    def calculate_session_stats(self):
        """Calculate session statistics"""
        self.session_duration = time.time() - self.session_start_time

        # Calculate Whisper stats
        self.whisper_avg_time = sum(self.whisper_times) / len(self.whisper_times) if self.whisper_times else 0
        self.whisper_fastest = min(self.whisper_times) if self.whisper_times else 0
        self.whisper_slowest = max(self.whisper_times) if self.whisper_times else 0

        # Calculate Gemini stats
        self.gemini_avg_time = sum(self.gemini_times) / len(self.gemini_times) if self.gemini_times else 0
        self.gemini_fastest = min(self.gemini_times) if self.gemini_times else 0
        self.gemini_slowest = max(self.gemini_times) if self.gemini_times else 0

        # Calculate Edge TTS stats
        self.edge_tts_avg_time = sum(self.edge_tts_times) / len(self.edge_tts_times) if self.edge_tts_times else 0
        self.edge_tts_fastest = min(self.edge_tts_times) if self.edge_tts_times else 0
        self.edge_tts_slowest = max(self.edge_tts_times) if self.edge_tts_times else 0

    def handle_conversation(self, initial_audio=None):
        """Handle a complete conversation turn"""
        # Start a conversation loop
        in_conversation = True

        # Initialize session statistics
        self.session_start_time = time.time()
        self.conversation_turns = 0
        self.languages_detected = set()

        # Initialize timing stats
        self.whisper_times = []
        self.gemini_times = []
        self.edge_tts_times = []
        self.search_times = []  # Reset search times for new session

        # Initialize last activity timestamp
        last_activity_time = time.time()

        # Process initial audio if provided
        if initial_audio:
            try:
                # Convert speech to text
                whisper_start = time.time()
                user_input = self.transcribe_audio(initial_audio)
                whisper_time = time.time() - whisper_start
                self.whisper_times.append(whisper_time)

                if user_input:
                    print(f"🎤 You said: '{user_input}'")
                    self.conversation_turns += 1
                    last_activity_time = time.time()

                    # Process the input and get AI response
                    gemini_start = time.time()
                    ai_response = self.get_ai_response(user_input)
                    gemini_time = time.time() - gemini_start
                    self.gemini_times.append(gemini_time)

                    if ai_response:
                        # Speak the response
                        edge_tts_start = time.time()
                        self.speak_text(ai_response)
                        edge_tts_time = time.time() - edge_tts_start
                        self.edge_tts_times.append(edge_tts_time)
                        last_activity_time = time.time()
            except Exception as e:
                print(f"Error processing initial audio: {e}")

        # Continue with normal conversation loop
        while in_conversation and not self.should_exit:
            try:
                # Check for timeout
                if time.time() - last_activity_time > CONVERSATION_TIMEOUT:
                    print(f"\n⏰ No activity detected for {CONVERSATION_TIMEOUT} seconds")
                    self.play_pre_recorded_message("goodbye")

                    # Calculate and print session statistics
                    self.calculate_session_stats()
                    self.print_session_stats()
                    return

                # Record user's speech
                audio_data = self.record_audio()

                if not audio_data or len(audio_data) == 0:
                    print("⚠️ No audio data recorded")
                    continue

                # Update last activity timestamp when we get valid audio input
                last_activity_time = time.time()

                # Play sound before starting transcription
                self.play_sound_effect("start_transcribe")

                # Convert speech to text
                whisper_start = time.time()
                user_input = self.transcribe_audio(audio_data)
                whisper_time = time.time() - whisper_start
                self.whisper_times.append(whisper_time)

                if not user_input:
                    self.play_pre_recorded_message("not_understood")
                    continue

                print(f"🎤 You said: '{user_input}'")
                self.conversation_turns += 1

                # Check if the user wants to end the conversation
                end_phrases = END_CONVERSATION_PHRASES
                if self.language == "chinese":
                    end_phrases = CHINESE_END_CONVERSATION_PHRASES + END_CONVERSATION_PHRASES

                # For English phrases, use case-insensitive comparison
                # For Chinese phrases, use exact match
                should_end = any(
                    (phrase in user_input.lower() if all(ord(c) < 128 for c in phrase) else phrase in user_input)
                    for phrase in end_phrases
                )

                if should_end:
                    print("🔚 Ending conversation")
                    self.play_pre_recorded_message("goodbye")

                    # Calculate and print session statistics
                    self.calculate_session_stats()
                    self.print_session_stats()
                    return

                # Play sound before getting AI response
                self.play_sound_effect("pre_response")

                # Get AI response
                try:
                    gemini_start = time.time()
                    ai_response = self.get_ai_response(user_input)
                    gemini_time = time.time() - gemini_start
                    self.gemini_times.append(gemini_time)

                    if ai_response:
                        # Speak the response
                        edge_tts_start = time.time()
                        self.speak_text(ai_response)
                        edge_tts_time = time.time() - edge_tts_start
                        self.edge_tts_times.append(edge_tts_time)

                        # Update last activity time after AI response
                        last_activity_time = time.time()

                        print("👂 Continuing conversation... (say 'goodbye' to end)")
                    else:
                        # Fallback response if AI fails
                        if self.language == "chinese":
                            self.speak_text("抱歉，我无法生成回应。请再试一次。")
                        else:
                            self.speak_text("Sorry, I couldn't generate a response. Please try again.")
                        # Update last activity time even for fallback response
                        last_activity_time = time.time()
                except Exception as e:
                    print(f"❌ Error getting or speaking AI response: {e}")
                    import traceback
                    traceback.print_exc()
                    # Provide a fallback response
                    if self.language == "chinese":
                        self.speak_text("抱歉，出现了一个错误。请再试一次。")
                    else:
                        self.speak_text("Sorry, there was an error. Please try again.")
                    # Update last activity time even for error response
                    last_activity_time = time.time()
            except Exception as e:
                print(f"❌ Error in conversation handling: {e}")
                import traceback
                traceback.print_exc()
                # Try to recover and continue listening
                try:
                    if self.language == "chinese":
                        self.speak_text("抱歉，出现了一个错误。我将继续聆听。")
                    else:
                        self.speak_text("Sorry, there was an error. I'll continue listening.")
                    # Update last activity time for recovery response
                    last_activity_time = time.time()
                except:
                    print("❌ Could not recover from error")
                    in_conversation = False

    def run(self):
        """Run the voice assistant"""
        try:
            if self.use_wake_word:
                try:
                    self.listen_for_wake_word()
                except Exception as e:
                    print(f"❌ Error in wake word detection: {e}")
                    import traceback
                    traceback.print_exc()
                    print("Attempting to restart wake word detection...")
                    time.sleep(1)
                    try:
                        self.listen_for_wake_word()
                    except:
                        print("❌ Failed to restart wake word detection")
            else:
                print("Starting voice assistant in continuous mode (no wake word)...")
                while not self.should_exit:
                    try:
                        self.handle_conversation()
                    except Exception as e:
                        print(f"❌ Error in conversation: {e}")
                        import traceback
                        traceback.print_exc()
                        print("Continuing to next conversation...")
                        time.sleep(1)
        except KeyboardInterrupt:
            print("\nExiting voice assistant...")
        except Exception as e:
            print(f"❌ Fatal error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        self.should_exit = True
        if self.use_wake_word and hasattr(self, 'porcupine'):
            self.porcupine.delete()
        if hasattr(self, 'audio'):
            self.audio.terminate()
        print("✅ Voice assistant resources cleaned up")

    def print_session_stats(self):
        """Print session statistics"""
        print("\n📊 Session Statistics:")
        print("==================================================")
        print(f"Session duration: {self.session_duration:.2f} seconds")
        print(f"Total conversation turns: {self.conversation_turns}")

        print("\nSpeech-to-Text (Whisper):")
        print(f"  Average time: {self.whisper_avg_time:.2f} seconds")
        print(f"  Fastest: {self.whisper_fastest:.2f}s")
        print(f"  Slowest: {self.whisper_slowest:.2f}s")

        print("\nLLM Response (Gemini):")
        print(f"  Average time: {self.gemini_avg_time:.2f} seconds")
        print(f"  Fastest: {self.gemini_fastest:.2f}s")
        print(f"  Slowest: {self.gemini_slowest:.2f}s")

        if self.search_times:  # Add search statistics
            search_avg = sum(self.search_times) / len(self.search_times)
            search_fastest = min(self.search_times)
            search_slowest = max(self.search_times)
            print("\nWeb Search (Tavily):")
            print(f"  Average time: {search_avg:.2f} seconds")
            print(f"  Fastest: {search_fastest:.2f}s")
            print(f"  Slowest: {search_slowest:.2f}s")
            print(f"  Total searches: {len(self.search_times)}")

        print("\nText-to-Speech (Edge TTS):")
        print(f"  Average time: {self.edge_tts_avg_time:.2f} seconds")
        print(f"  Fastest: {self.edge_tts_fastest:.2f}s")
        print(f"  Slowest: {self.edge_tts_slowest:.2f}s")

        if hasattr(self, 'languages_detected') and self.languages_detected:
            print("\nLanguages detected: " + ", ".join(self.languages_detected))
        print("==================================================")

def main():
    parser = argparse.ArgumentParser(description="Simple Local Voice Assistant")
    parser.add_argument(
        "--no-wake-word",
        action="store_true",
        help="Run in continuous mode without wake word detection"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode to print audio volume levels"
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Use manual recording mode (press Enter to start/stop)"
    )
    parser.add_argument(
        "--disable-interruption",
        action="store_true",
        help="Disable interruption detection (enabled by default)"
    )
    parser.add_argument(
        "--language",
        choices=["english", "chinese"],
        default="english",
        help="Set the language (english or chinese)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Conversation timeout in seconds (default: 30)"
    )
    args = parser.parse_args()

    # Set global options
    global USE_MANUAL_RECORDING, ENABLE_INTERRUPTION, CONVERSATION_TIMEOUT
    USE_MANUAL_RECORDING = args.manual
    ENABLE_INTERRUPTION = not args.disable_interruption
    CONVERSATION_TIMEOUT = args.timeout

    assistant = SimpleLocalAssistant(
        use_wake_word=not args.no_wake_word,
        debug=args.debug,
        language=args.language
    )
    assistant.run()

if __name__ == "__main__":
    main()
