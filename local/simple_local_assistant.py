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
import tempfile
import subprocess
from dotenv import load_dotenv
import pvporcupine
from pvrecorder import PvRecorder
from tavily import TavilyClient
from cobra_vad import CobraVAD

# Import prompts from the prompts module (absolute import)
from prompts import SYSTEM_PROMPT, CHAT_PROMPTS

# Use faster-whisper for speech recognition
from faster_whisper import WhisperModel

# Constants
SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_LENGTH = 512
SILENCE_THRESHOLD = 0.025
SILENCE_DURATION = 1.5
INITIAL_SILENCE_DURATION = 0.4  # Shorter silence duration for initial question after wake word
INTERRUPTION_THRESHOLD = 0.1
INTERRUPTION_MIN_CHUNKS = 3
DEBUG_AUDIO = True
USE_FIXED_THRESHOLDS = True
USE_MANUAL_RECORDING = False
ENABLE_INTERRUPTION = False
USE_EDGE_TTS = True

# Timeout settings
UTTERANCE_TIMEOUT = 30.0  # Maximum time to wait for a single utterance (in seconds)
INACTIVITY_TIMEOUT = 30.0  # Time to wait for next user input before ending conversation (in seconds)

# VAD settings
VAD_THRESHOLD = 0.6  # Voice probability threshold for Cobra VAD

# Sound effect paths
SOUND_EFFECTS = {
    "wake": "sounds/wake_chime.wav",  # Shorter wake sound
    "start_listening": "sounds/start_listening.wav",  # Sound played when starting to listen
    "start_transcribe": "sounds/start_transcribe.wav",  # Sound played before starting transcription
    "pre_response": "sounds/pre_response.wav",  # Sound played before getting AI response
    "goodbye_en": "sounds/goodbye_en.wav",  # English goodbye message
    "goodbye_zh": "sounds/goodbye_zh.wav",  # Chinese goodbye message
    "not_understood_en": "sounds/not_understood_en.wav",  # English not understood message
    "not_understood_zh": "sounds/not_understood_zh.wav",  # Chinese not understood message
}

# Pre-recorded messages mapping
PRE_RECORDED_MESSAGES = {
    "goodbye": {
        "english": "goodbye_en",
        "chinese": "goodbye_zh",
        "others": "goodbye_en"
    },
    "not_understood": {
        "english": "not_understood_en",
        "chinese": "not_understood_zh",
        "others": "not_understood_en"
    }
}

# Search-related constants
ENABLE_SEARCH = True

# TTS Voice settings
EDGE_TTS_VOICES = {
    "english": "en-US-AvaMultilingualNeural",
    "chinese": "zh-CN-XiaoxiaoNeural",
    "others": "en-US-AvaMultilingualNeural",  # For languages other than English and Chinese
}

# Default voices
ENGLISH_EDGE_TTS_VOICE = EDGE_TTS_VOICES["english"]
CHINESE_EDGE_TTS_VOICE = EDGE_TTS_VOICES["chinese"]
OTHER_EDGE_TTS_VOICE = EDGE_TTS_VOICES["others"]

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

class SimpleLocalAssistant:
    def __init__(self, debug=False):
        """Initialize the voice assistant"""
        # Load environment variables
        load_dotenv()

        # Set debug mode
        global DEBUG_AUDIO
        DEBUG_AUDIO = debug
        if DEBUG_AUDIO:
            print("🔍 Debug mode enabled - will print audio volume levels")

        # Set initial language
        self.language = "english"
        print(f"🌐 Initial language set to: {self.language}")

        # Check required environment variables
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        if not self.google_api_key:
            print("❌ GOOGLE_API_KEY is required in .env file")
            sys.exit(1)

        # Initialize Cobra VAD
        self.init_cobra_vad()

        # Initialize speech recognition
        self.init_speech_recognition()

        # Initialize TTS
        self.init_edge_tts()

        # Initialize wake word detection
        self.init_porcupine()

        # Initialize Gemini model
        self.init_gemini()

        # Initialize chat session
        self.init_chat_session()

        # Initialize search APIs if enabled
        if ENABLE_SEARCH:
            self.init_search_apis()

        # Initialize timing stats
        self.stt_times = []
        self.llm_times = []
        self.tts_times = []
        self.search_times = []  # Reset search times for new session

        # State variables
        self.is_listening = False
        self.is_speaking = False
        self.should_exit = False
        self.audio_queue = queue.Queue()

        print("✅ Voice assistant initialized and ready")

    def init_cobra_vad(self):
        """Initialize Cobra VAD for speech detection"""
        try:
            # Get access key from environment
            access_key = os.getenv("PICOVOICE_ACCESS_KEY")
            if not access_key:
                print("❌ PICOVOICE_ACCESS_KEY is required in .env file for Cobra VAD")
                sys.exit(1)

            # Create Cobra VAD instance
            self.cobra_vad = CobraVAD(
                access_key=access_key,
                threshold=VAD_THRESHOLD,
                debug=DEBUG_AUDIO
            )

            print(f"✅ Cobra VAD initialized with threshold: {VAD_THRESHOLD}")
        except Exception as e:
            print(f"❌ Error initializing Cobra VAD: {e}")
            sys.exit(1)

    def calibrate_microphone(self):
        """Measure ambient noise and calibrate thresholds using PvRecorder"""
        print("🎙️ Calibrating microphone (please be quiet)...")

        recorder = None
        try:
            recorder = PvRecorder(device_index=self.audio_device_index, frame_length=FRAME_LENGTH)
            print(f"Using audio device: {recorder.selected_device}")
            recorder.start()

            # Collect ambient noise samples
            ambient_levels = []
            calibration_time = 2  # seconds
            samples_to_collect = int(calibration_time * SAMPLE_RATE / FRAME_LENGTH)

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
            # Use base model for faster loading and less memory usage
            model_size = "tiny"
            device = "auto"
            compute_type = "int8"

            print(f"Using faster-whisper {model_size} model on {device} with compute type {compute_type}")

            try:
                # First try to load from local cache only
                print("Attempting to load model from local cache...")
                self.whisper_model = WhisperModel(
                    model_size,
                    device=device,
                    compute_type=compute_type,
                    cpu_threads=2,  # Reduced for Raspberry Pi
                    num_workers=2,
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
                    cpu_threads=2,  # Reduced for Raspberry Pi
                    num_workers=2,
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
            elif self.language == "english":
                self.edge_tts_voice = os.getenv("ENGLISH_EDGE_TTS_VOICE", ENGLISH_EDGE_TTS_VOICE)
            else:
                self.edge_tts_voice = os.getenv("OTHER_EDGE_TTS_VOICE", OTHER_EDGE_TTS_VOICE)

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
                    self.edge_tts_voice = ENGLISH_EDGE_TTS_VOICE
                print(f"Using fallback voice: {self.edge_tts_voice}")

            print(f"✅ Edge TTS initialized with voice: {self.edge_tts_voice}")
        except Exception as e:
            print(f"❌ Error initializing Edge TTS: {e}")
            sys.exit(1)

    def init_porcupine(self):
        """Initialize Porcupine wake word detection"""
        access_key = os.getenv("PICOVOICE_ACCESS_KEY")
        if not access_key:
            print("❌ PICOVOICE_ACCESS_KEY is required in .env file for wake word detection")
            sys.exit(1)

        # Get audio device index from environment
        try:
            self.audio_device_index = int(os.getenv("AUDIO_DEVICE_INDEX", -1))
            if DEBUG_AUDIO:
                print(f"Using audio device index {self.audio_device_index} from environment")
        except ValueError:
            print("⚠️ Invalid AUDIO_DEVICE_INDEX in environment, using default")
            self.audio_device_index = -1

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
        """Initialize Google Gemini model with family-friendly safety settings"""
        print("Initializing Gemini model...")
        try:
            if not self.google_api_key:
                print("❌ Google API key not found. Please set GOOGLE_API_KEY in your .env file.")
                sys.exit(1)

            # Configure the Gemini API
            genai.configure(api_key=self.google_api_key)

            # Set up safety settings for family-friendly content
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]

            # Set up the model configuration
            self.generation_config = {
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 1024,
            }

            # Create the model with safety settings
            self.model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                generation_config=self.generation_config,
                safety_settings=safety_settings
            )

            print("✅ Gemini Flash model initialized with family-friendly safety settings")
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

    def play_sound_effect(self, effect_name, blocking=False):
        """
        Play a sound effect from the sounds directory
        Args:
            effect_name: Name of the sound effect to play
            blocking: Whether to wait for the sound to finish playing
        """
        if effect_name not in SOUND_EFFECTS:
            print(f"⚠️ Sound effect {effect_name} not found")
            return

        sound_path = os.path.join(os.path.dirname(__file__), SOUND_EFFECTS[effect_name])
        if not os.path.exists(sound_path):
            print(f"⚠️ Sound file not found: {sound_path}")
            return

        try:
            # For Raspberry Pi/Linux
            if sys.platform.startswith('linux'):
                try:
                    cmd = ["aplay", sound_path]
                    if blocking:
                        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    else:
                        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    return

                except Exception as e:
                    print(f"⚠️ ALSA playback failed: {e}")
                    # Try mpg123 as fallback
                    try:
                        cmd = ["mpg123", "-q", sound_path]
                        if blocking:
                            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        else:
                            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        return
                    except Exception as e2:
                        print(f"⚠️ mpg123 playback failed: {e2}")

            # Platform-specific playback methods (keep existing code for other platforms)
            elif sys.platform == "darwin":  # macOS
                try:
                    if blocking:
                        subprocess.run(["afplay", sound_path], check=True)
                    else:
                        subprocess.Popen(["afplay", sound_path])
                    return
                except Exception as e:
                    print(f"⚠️ macOS audio playback failed: {e}")

            elif sys.platform == "win32":  # Windows
                try:
                    if blocking:
                        subprocess.run(["wmplayer", sound_path, "/close"], check=True)
                    else:
                        subprocess.Popen(["wmplayer", sound_path, "/close"])
                    return
                except Exception as e:
                    print(f"⚠️ Windows Media Player failed: {e}")
                    try:
                        if blocking:
                            subprocess.run(["start", sound_path], shell=True, check=True)
                        else:
                            subprocess.Popen(["start", sound_path], shell=True)
                        return
                    except Exception as e2:
                        print(f"⚠️ Windows shell playback failed: {e2}")

            # Fallback methods for all platforms
            print("⚠️ Trying fallback audio players...")
            for player in ["aplay", "mpg123", "mpg321", "mplayer", "ffplay"]:
                try:
                    if blocking:
                        subprocess.run([player, sound_path], check=True,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    else:
                        subprocess.Popen([player, sound_path],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    return
                except (subprocess.SubprocessError, FileNotFoundError):
                    continue

            print(f"⚠️ All playback methods failed for: {sound_path}")

        except Exception as e:
            print(f"⚠️ Error playing sound effect: {e}")
            import traceback
            traceback.print_exc()

    def listen_for_wake_word(self):
        """Listen for wake word using Porcupine and capture any following speech"""
        print("👂 Listening for wake word...")

        recorder = None
        try:
            # Initialize recorder with frame length 512 (works for both wake word and speech)
            recorder = PvRecorder(device_index=self.audio_device_index, frame_length=self.porcupine.frame_length)
            recorder.start()

            print(f"Using audio device: {recorder.selected_device}")
            print(f"Wake word frame length: {self.porcupine.frame_length}")

            while not self.should_exit:
                try:
                    pcm = recorder.read()

                    # Process for wake word detection
                    keyword_index = self.porcupine.process(pcm)
                    if keyword_index >= 0:
                        detected_word = self.keywords[keyword_index]
                        print(f"🎯 Wake word detected: '{detected_word}'!")

                        # Start VAD monitoring and play start_listening sound
                        self.cobra_vad.start_monitoring()
                        self.play_sound_effect("start_listening")

                        # Start conversation with Cobra VAD already monitoring
                        self.handle_conversation()

                        # After conversation, go back to listening for wake word
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
        """Record audio from microphone using Cobra VAD"""
        print("\n" + "-"*50)
        print("🎤 Listening with Cobra VAD...")

        self.is_listening = True

        try:
            # Ensure VAD monitoring is started and play sound
            if not self.cobra_vad.is_monitoring:
                print("Starting VAD monitoring...")
                self.cobra_vad.start_monitoring()
                self.play_sound_effect("start_listening")
                # Add a small delay after starting monitoring
                time.sleep(0.2)

            # Clear any residual audio before starting new recording
            self.cobra_vad.clear_audio_buffer()

            # Get the next speech segment with timeout
            audio_data = self.cobra_vad.get_next_audio(timeout=UTTERANCE_TIMEOUT)

            # Print recording stats
            if audio_data:
                duration = len(audio_data) / (SAMPLE_RATE * 2)  # 16-bit audio at 16kHz
                print(f"Recording finished after {duration:.1f} seconds")
            else:
                print(f"⚠️ No speech detected within {UTTERANCE_TIMEOUT} seconds")

            return audio_data or b''

        except Exception as e:
            print(f"❌ Error recording audio with Cobra VAD: {e}")
            return b''
        finally:
            # Don't stop monitoring here - let the conversation handler manage the VAD state
            self.is_listening = False

    def transcribe_audio(self, audio_data):
        """Transcribe audio using Whisper"""
        if not self.whisper_model:
            print("❌ Whisper model not initialized")
            return None

        try:
            print("🎤 Transcribing audio...")

            # Start timing STT
            stt_start = time.time()

            # Create a temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                # Write WAV header and audio data
                with wave.open(temp_wav.name, 'wb') as wf:
                    wf.setnchannels(1)  # Mono
                    wf.setsampwidth(2)  # 16-bit audio
                    wf.setframerate(SAMPLE_RATE)
                    wf.writeframes(audio_data)

                # Use beam_size=5 and best_of=5 for better accuracy
                # Set language=None to enable language detection
                segments, info = self.whisper_model.transcribe(
                    temp_wav.name,
                    beam_size=5,
                    best_of=5,
                    language=None,
                    task="transcribe",
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=1000)
                )

                # Calculate STT time and add to stats
                stt_time = time.time() - stt_start
                self.stt_times.append(stt_time)
                print(f"🕒 STT took {stt_time:.2f} seconds")

                # Clean up the temporary file
                try:
                    os.unlink(temp_wav.name)
                except:
                    pass  # Ignore cleanup errors

                # Get the detected language and its probability
                detected_lang = info.language
                lang_prob = info.language_probability

                print(f"🔍 Detected language: {detected_lang} (probability: {lang_prob:.2f})")

                # Update assistant language based on detected language
                if lang_prob > 0.5:  # Only update if confidence is high enough
                    if detected_lang == "zh":
                        self.language = "chinese"
                        print("🇨🇳 Switching to Chinese mode")
                    elif detected_lang == "en":
                        self.language = "english"
                        print("🇺🇸 Switching to English mode")
                    else:
                        self.language = "others"
                        print(f"🌐 Using multilingual voice for {detected_lang}")

                # Combine all segments into one text
                text = " ".join([segment.text for segment in segments]).strip()

                if not text:
                    print("❌ No speech detected in audio")
                    return None

                print(f"✅ Transcribed text: {text}")
                return text

        except Exception as e:
            print(f"❌ Error transcribing audio: {e}")
            import traceback
            traceback.print_exc()
            return None

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
            decision_prompt = CHAT_PROMPTS[self.language].format(query=user_input)

            # Start timing here - just before the LLM call
            llm_start = time.time()

            # Get structured response from LLM
            response = self.chat_session.send_message(
                decision_prompt,
                stream=False
            )

            # End timing here - right after the LLM call
            llm_time = time.time() - llm_start
            if hasattr(self, 'llm_times'):
                self.llm_times.append(llm_time)
            print(f"🕒 LLM response took {llm_time:.2f} seconds")

            try:
                # Print raw response in a pretty format
                print("\n🔍 LLM Response:\n")
                # Split the response into lines and print each line
                for line in response.text.splitlines():
                    print(line)
                print()

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
        try:
            # Pause VAD monitoring while speaking
            if hasattr(self, 'cobra_vad') and self.cobra_vad.is_monitoring:
                self.cobra_vad.pause_monitoring()

            # Speak the text and get timing
            tts_time = self.speak_text_edge(text)
            if tts_time is not None:
                self.tts_times.append(tts_time)

        finally:
            # Clear any audio that might have accumulated during speaking
            if hasattr(self, 'cobra_vad'):
                self.cobra_vad.clear_audio_buffer()
                # Resume VAD monitoring after speaking
                if self.cobra_vad.is_monitoring:
                    self.cobra_vad.resume_monitoring()
                    self.play_sound_effect("start_listening")

    def speak_text_edge(self, text):
        """Convert text to speech using Edge TTS and play it"""
        try:
            print(f"🔊 Speaking: '{text}'")
            self.is_speaking = True

            # Select voice based on current language
            if self.language == "chinese":
                voice = os.getenv("CHINESE_EDGE_TTS_VOICE", CHINESE_EDGE_TTS_VOICE)
            elif self.language == "english":
                voice = os.getenv("ENGLISH_EDGE_TTS_VOICE", ENGLISH_EDGE_TTS_VOICE)
            else:
                voice = os.getenv("OTHER_EDGE_TTS_VOICE", OTHER_EDGE_TTS_VOICE)

            # Update the voice if it changed
            if voice != self.edge_tts_voice:
                self.edge_tts_voice = voice
                print(f"Switched TTS voice to: {self.edge_tts_voice}")

            # Create a temporary file with a unique name
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
                temp_file = temp_audio.name

            # Create a new event loop for async operations
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Start timing TTS generation
            tts_start = time.time()

            # Function to generate speech with Edge TTS and save to file
            async def generate_speech():
                communicate = edge_tts.Communicate(text, self.edge_tts_voice)
                await communicate.save(temp_file)

            # Generate speech and save to file
            loop.run_until_complete(generate_speech())
            loop.close()

            # Calculate TTS generation time
            tts_time = time.time() - tts_start

            # Try different methods to play the audio file based on platform
            played_successfully = False

            # Platform-specific playback methods
            if sys.platform == "darwin":  # macOS
                try:
                    subprocess.run(["afplay", temp_file], check=True)
                    played_successfully = True
                except Exception as e:
                    print(f"⚠️ macOS audio playback failed: {e}")

            elif sys.platform == "win32":  # Windows
                try:
                    os.startfile(temp_file)  # Native Windows audio playback
                    time.sleep(0.1)  # Small delay to ensure playback starts
                    played_successfully = True
                except Exception as e:
                    print(f"⚠️ Windows audio playback failed: {e}")
                    try:
                        # Fallback to Windows Media Player CLI
                        subprocess.run(["wmplayer", temp_file], check=True)
                        played_successfully = True
                    except Exception as e2:
                        print(f"⚠️ Windows Media Player fallback failed: {e2}")

            elif sys.platform.startswith("linux"):  # Linux/Raspberry Pi
                # For Raspberry Pi, try ALSA first
                try:
                    # Use the configured audio device
                    alsa_device = f"hw:{self.audio_device_index},0" if self.audio_device_index >= 0 else "default"

                    # Try to set maximum volume for the device
                    try:
                        subprocess.run(["amixer", "-c", str(max(0, self.audio_device_index)), "sset", "PCM", "100%"],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    except Exception as e:
                        print(f"⚠️ Could not set volume: {e}")

                    # Try mpg123 with specific ALSA device first (since Edge TTS outputs MP3)
                    try:
                        cmd = ["mpg123", "-a", alsa_device, "-q", temp_file]
                        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        played_successfully = True
                    except Exception as e:
                        print(f"⚠️ mpg123 playback failed: {e}")

                        # If mpg123 fails, try converting to WAV and using aplay
                        try:
                            # Convert MP3 to WAV using ffmpeg
                            wav_file = temp_file.replace('.mp3', '.wav')
                            subprocess.run(["ffmpeg", "-i", temp_file, "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", wav_file],
                                         check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                            # Play WAV with aplay
                            subprocess.run(["aplay", "-D", alsa_device, wav_file],
                                         check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            played_successfully = True

                            # Clean up WAV file
                            try:
                                os.remove(wav_file)
                            except:
                                pass
                        except Exception as e2:
                            print(f"⚠️ WAV conversion and aplay failed: {e2}")

                except Exception as e:
                    print(f"⚠️ ALSA playback failed: {e}")

            # If platform-specific methods failed, try generic fallbacks
            if not played_successfully:
                print("⚠️ Trying fallback audio players...")
                for player in ["mpg123", "mpg321", "mplayer", "ffplay"]:
                    try:
                        subprocess.run([player, temp_file], check=True,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        played_successfully = True
                        break
                    except (subprocess.SubprocessError, FileNotFoundError):
                        continue

            # Method 2: Last resort - print the text if audio failed
            if not played_successfully:
                print(f"⚠️ Audio playback failed. Text response: {text}")

            # Clean up the temporary file
            try:
                os.remove(temp_file)
            except:
                pass

            self.is_speaking = False

            # Return only the TTS generation time
            return tts_time

        except Exception as e:
            print(f"❌ Error speaking text with Edge TTS: {e}")
            import traceback
            traceback.print_exc()
            self.is_speaking = False
            return None

    def play_pre_recorded_message(self, message_type):
        """Play a pre-recorded message based on type and current language"""
        try:
            # Pause VAD monitoring while playing message
            if hasattr(self, 'cobra_vad') and self.cobra_vad.is_monitoring:
                self.cobra_vad.pause_monitoring()

            if message_type in PRE_RECORDED_MESSAGES:
                sound_effect = PRE_RECORDED_MESSAGES[message_type][self.language]
                # Use blocking mode for pre-recorded messages to ensure they complete
                self.play_sound_effect(sound_effect, blocking=True)
            else:
                print(f"⚠️ No pre-recorded message found for: {message_type}")
        finally:
            # Resume VAD monitoring after message is fully played
            if hasattr(self, 'cobra_vad') and self.cobra_vad.is_monitoring:
                self.cobra_vad.resume_monitoring()

    def calculate_session_stats(self):
        """Calculate session statistics"""
        self.session_duration = time.time() - self.session_start_time

        # Calculate STT stats
        self.stt_avg_time = sum(self.stt_times) / len(self.stt_times) if self.stt_times else 0
        self.stt_fastest = min(self.stt_times) if self.stt_times else 0
        self.stt_slowest = max(self.stt_times) if self.stt_times else 0

        # Calculate LLM stats
        self.llm_avg_time = sum(self.llm_times) / len(self.llm_times) if self.llm_times else 0
        self.llm_fastest = min(self.llm_times) if self.llm_times else 0
        self.llm_slowest = max(self.llm_times) if self.llm_times else 0

        # Calculate TTS stats
        self.tts_avg_time = sum(self.tts_times) / len(self.tts_times) if self.tts_times else 0
        self.tts_fastest = min(self.tts_times) if self.tts_times else 0
        self.tts_slowest = max(self.tts_times) if self.tts_times else 0

    def handle_conversation(self):
        """Handle a complete conversation turn"""
        # Start a conversation loop
        in_conversation = True

        # Initialize session statistics
        self.session_start_time = time.time()
        self.conversation_turns = 0
        self.languages_detected = set()

        # Initialize timing stats
        self.stt_times = []
        self.llm_times = []
        self.tts_times = []
        self.search_times = []

        # Initialize last activity timestamp
        last_activity_time = time.time()

        try:
            # Start VAD monitoring at the beginning of conversation
            if not self.cobra_vad.is_monitoring:
                print("Starting VAD monitoring for conversation...")
                self.cobra_vad.start_monitoring()
                self.play_sound_effect("start_listening")

            # Continue with normal conversation loop
            while in_conversation and not self.should_exit:
                try:
                    # Check for timeout
                    if time.time() - last_activity_time > INACTIVITY_TIMEOUT:
                        print(f"\n⏰ No activity detected for {INACTIVITY_TIMEOUT} seconds")
                        self.play_pre_recorded_message("goodbye")

                        # Calculate and print session statistics
                        self.calculate_session_stats()
                        self.print_session_stats()
                        return

                    # Ensure VAD is monitoring before recording
                    if not self.cobra_vad.is_monitoring:
                        print("Restarting VAD monitoring...")
                        self.cobra_vad.start_monitoring()
                        self.play_sound_effect("start_listening")
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
                    user_input = self.transcribe_audio(audio_data)

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
                    # self.play_sound_effect("pre_response")

                    # Get AI response
                    try:
                        ai_response = self.get_ai_response(user_input)

                        if ai_response:
                            # Speak the response (timing is handled in speak_text)
                            self.speak_text(ai_response)

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

        finally:
            # Stop VAD monitoring when conversation ends
            if self.cobra_vad.is_monitoring:
                print("Stopping VAD monitoring at end of conversation...")
                self.cobra_vad.stop_monitoring()

    def print_session_stats(self):
        """Print session statistics"""
        print("\n📊 Session Statistics:")
        print("==================================================")
        print(f"Session duration: {self.session_duration:.2f} seconds")
        print(f"Total conversation turns: {self.conversation_turns}")

        print("\nSpeech-to-Text (faster-whisper transcription):")
        print(f"  Average time: {self.stt_avg_time:.2f} seconds")
        print(f"  Fastest: {self.stt_fastest:.2f}s")
        print(f"  Slowest: {self.stt_slowest:.2f}s")

        print("\nLLM Response (Gemini response):")
        print(f"  Average time: {self.llm_avg_time:.2f} seconds")
        print(f"  Fastest: {self.llm_fastest:.2f}s")
        print(f"  Slowest: {self.llm_slowest:.2f}s")

        if self.search_times:  # Add search statistics
            search_avg = sum(self.search_times) / len(self.search_times)
            search_fastest = min(self.search_times)
            search_slowest = max(self.search_times)
            print("\nWeb Search (Tavily):")
            print(f"  Average time: {search_avg:.2f} seconds")
            print(f"  Fastest: {search_fastest:.2f}s")
            print(f"  Slowest: {search_slowest:.2f}s")
            print(f"  Total searches: {len(self.search_times)}")

        print("\nText-to-Speech (Edge TTS generation):")
        print(f"  Average time: {self.tts_avg_time:.2f} seconds")
        print(f"  Fastest: {self.tts_fastest:.2f}s")
        print(f"  Slowest: {self.tts_slowest:.2f}s")

        if hasattr(self, 'languages_detected') and self.languages_detected:
            print("\nLanguages detected: " + ", ".join(self.languages_detected))
        print("==================================================")

    def run(self):
        """Run the voice assistant"""
        try:
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
        if hasattr(self, 'porcupine'):
            self.porcupine.delete()
        if hasattr(self, 'cobra_vad'):
            self.cobra_vad.cleanup()
        if hasattr(self, 'audio'):
            self.audio.terminate()
        print("✅ Voice assistant resources cleaned up")

def main():
    parser = argparse.ArgumentParser(description="Simple Local Voice Assistant")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode to print audio volume levels"
    )
    args = parser.parse_args()

    # Set global debug mode
    global DEBUG_AUDIO
    DEBUG_AUDIO = args.debug

    assistant = SimpleLocalAssistant(
        debug=args.debug
    )
    assistant.run()

if __name__ == "__main__":
    main()
