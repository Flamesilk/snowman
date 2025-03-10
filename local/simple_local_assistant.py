#!/usr/bin/env python3
"""
Simple Local Voice Assistant

A lightweight voice assistant implementation that uses:
- PyAudio for direct microphone access
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
import pyaudio
import google.generativeai as genai
import asyncio
import edge_tts
import io
import tempfile
import subprocess
from dotenv import load_dotenv
import pvporcupine
from pvrecorder import PvRecorder

# Use faster-whisper for speech recognition
from faster_whisper import WhisperModel

# Constants
SAMPLE_RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK_SIZE = 512
SILENCE_THRESHOLD = 0.01
SILENCE_DURATION = 1.5
INTERRUPTION_THRESHOLD = 0.03
INTERRUPTION_MIN_CHUNKS = 3
DEBUG_AUDIO = True
MAX_RECORDING_TIME = 20
USE_FIXED_THRESHOLDS = True
USE_MANUAL_RECORDING = False
ENABLE_INTERRUPTION = True
USE_EDGE_TTS = True

# TTS Voice settings
EDGE_TTS_VOICES = {
    "english": "en-US-ChristopherNeural",  # Male voice
    "chinese": "zh-CN-XiaoxiaoNeural",        # Female voice
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
CHINESE_END_CONVERSATION_PHRASES = ["再见", "拜拜", "结束对话", "停止聆听", "谢谢"]
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

        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()

        # Set thresholds
        if USE_FIXED_THRESHOLDS:
            self.silence_threshold = SILENCE_THRESHOLD
            self.interruption_threshold = INTERRUPTION_THRESHOLD
            print(f"Using fixed thresholds - Silence: {self.silence_threshold}, Interruption: {self.interruption_threshold}")
        else:
            # Calibrate microphone
            self.silence_threshold, self.interruption_threshold = self.calibrate_microphone()

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

        # State variables
        self.is_listening = False
        self.is_speaking = False
        self.should_exit = False
        self.audio_queue = queue.Queue()

        print("✅ Voice assistant initialized and ready")

    def calibrate_microphone(self):
        """Measure ambient noise and calibrate thresholds"""
        print("🎙️ Calibrating microphone (please be quiet)...")

        # Open audio stream
        stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )

        # Collect ambient noise samples
        ambient_levels = []
        calibration_time = 2  # seconds
        samples_to_collect = int(calibration_time * SAMPLE_RATE / CHUNK_SIZE)

        for _ in range(samples_to_collect):
            try:
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)
                volume_norm = np.abs(audio_data).mean() / 32768.0
                ambient_levels.append(volume_norm)
            except Exception as e:
                print(f"Error during calibration: {e}")

        # Close the stream
        stream.stop_stream()
        stream.close()

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
            # Use base model for better balance of performance and accuracy
            model_size = "base"
            # Use CPU for better stability
            device = "cpu"
            compute_type = "int8"

            print(f"Using faster-whisper on {device} with compute type {compute_type}")

            # Load the Whisper model with conservative settings
            self.whisper_model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
                cpu_threads=2,  # Reduce CPU threads for stability
                num_workers=1,  # Single worker thread
                download_root="models",  # Specify model storage directory
                local_files_only=True  # Use cached model if available
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

    def listen_for_wake_word(self):
        """Listen for wake word using Porcupine with PvRecorder"""
        print(f"Listening for wake words: {', '.join(self.keywords)}...")

        recorder = None
        try:
            # Initialize PvRecorder
            recorder = PvRecorder(
                device_index=-1,  # default audio device
                frame_length=self.porcupine.frame_length
            )
            recorder.start()

            print(f"Using audio device: {recorder.selected_device}")

            while not self.should_exit:
                try:
                    pcm = recorder.read()
                    keyword_index = self.porcupine.process(pcm)

                    if keyword_index >= 0:
                        detected_keyword = self.keywords[keyword_index]
                        print(f"🎤 Wake word detected: '{detected_keyword}'")

                        # Stop recording before handling conversation
                        recorder.stop()

                        try:
                            # Create a new chat session for each wake word detection
                            self.init_chat_session()

                            # Handle the conversation
                            self.handle_conversation()
                        except Exception as e:
                            print(f"❌ Error handling conversation after wake word: {e}")
                            import traceback
                            traceback.print_exc()

                        # Resume recording after conversation
                        if not recorder.is_recording:
                            recorder.start()

                        print(f"Listening for wake words: {', '.join(self.keywords)}...")
                except Exception as e:
                    print(f"❌ Error processing audio for wake word: {e}")
                    # Brief pause to avoid tight error loop
                    time.sleep(0.1)
        except Exception as e:
            print(f"❌ Error in wake word listener: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if recorder is not None:
                try:
                    recorder.stop()
                    recorder.delete()
                except:
                    pass

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

        print("🎤 Recording... (press Enter to stop)")
        self.is_listening = True

        # Open audio stream
        stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )

        frames = []

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
            while not stop_recording.is_set() and time.time() - start_time < MAX_RECORDING_TIME:
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                frames.append(data)

                # Print volume for debugging
                if DEBUG_AUDIO and len(frames) % 20 == 0:  # Only print every 20 frames
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    volume_norm = np.abs(audio_data).mean() / 32768.0
                    print(f"Recording volume: {volume_norm:.4f}")
        finally:
            stream.stop_stream()
            stream.close()
            self.is_listening = False

            # Print recording stats
            elapsed_time = time.time() - start_time
            print(f"Recording finished after {elapsed_time:.1f} seconds")

        return b''.join(frames)

    def record_audio_auto(self):
        """Record audio from microphone until silence is detected (automatic)"""
        print("🎤 Listening... (automatic mode)")
        self.is_listening = True

        # Open audio stream with settings matching Porcupine
        stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )

        frames = []
        silent_chunks = 0
        required_silent_chunks = int(SILENCE_DURATION * SAMPLE_RATE / CHUNK_SIZE)

        # Variables to track speech activity
        speech_detected = False
        max_volume = 0.0

        # Add timeout mechanism
        start_time = time.time()
        max_chunks = int(MAX_RECORDING_TIME * SAMPLE_RATE / CHUNK_SIZE)
        chunk_count = 0

        # Wait a moment before starting to record
        time.sleep(0.1)

        try:
            while not self.should_exit and self.is_listening:
                try:
                    # Check for timeout
                    if chunk_count >= max_chunks:
                        print(f"⚠️ Recording timeout after {MAX_RECORDING_TIME} seconds")
                        break

                    data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                    frames.append(data)
                    chunk_count += 1

                    # Check audio volume
                    audio_data = np.frombuffer(data, dtype=np.int16)
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
                    else:
                        silent_chunks += 1
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
            stream.stop_stream()
            stream.close()
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
                wf.setsampwidth(self.audio.get_sample_size(FORMAT))
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

    def get_ai_response(self, user_input):
        """Get response from Gemini AI using chat history"""
        try:
            print(f"🧠 Processing: '{user_input}'")

            # Set a timeout for the API call
            start_time = time.time()

            # Send the message to the ongoing chat session
            response = self.chat_session.send_message(user_input)

            elapsed_time = time.time() - start_time
            print(f"Gemini response received in {elapsed_time:.2f} seconds")

            return response.text
        except Exception as e:
            print(f"❌ Error getting AI response: {e}")
            import traceback
            traceback.print_exc()
            return "I'm sorry, I encountered an error processing your request."

    # def sanitize_text_for_speech(self, text):
    #     """Clean up text before sending to TTS by removing emojis, brackets and special characters"""
    #     import re

    #     # Remove emojis and other special unicode characters
    #     text = text.encode('ascii', 'ignore').decode('ascii')

    #     # Remove text within brackets (including the brackets)
    #     text = re.sub(r'\[.*?\]', '', text)
    #     text = re.sub(r'\(.*?\)', '', text)

    #     # Remove markdown code blocks and backticks
    #     text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    #     text = re.sub(r'`.*?`', '', text)

    #     # Remove URLs
    #     text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

    #     # Remove multiple spaces and clean up whitespace
    #     text = ' '.join(text.split())

    #     return text.strip()

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

            # Method 2: Fallback to PyAudio if system player failed
            if not played_successfully:
                try:
                    print("Falling back to PyAudio for playback...")
                    # Use a different audio library to read the MP3 file
                    import soundfile as sf
                    import librosa

                    # Convert MP3 to WAV for easier handling
                    y, sr = librosa.load(temp_file, sr=None)
                    temp_wav = "temp_tts.wav"
                    sf.write(temp_wav, y, sr)

                    # Play using PyAudio
                    with wave.open(temp_wav, 'rb') as wf:
                        # Open stream
                        stream = self.audio.open(
                            format=self.audio.get_format_from_width(wf.getsampwidth()),
                            channels=wf.getnchannels(),
                            rate=wf.getframerate(),
                            output=True
                        )

                        # Read data
                        data = wf.readframes(1024)

                        # Play
                        while data:
                            stream.write(data)
                            data = wf.readframes(1024)

                        # Close
                        stream.stop_stream()
                        stream.close()

                    # Clean up temporary WAV file
                    try:
                        os.remove(temp_wav)
                    except:
                        pass

                    played_successfully = True
                except Exception as e:
                    print(f"PyAudio fallback error: {e}")

            # Method 3: Last resort - print the text if audio failed
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

    def handle_conversation(self):
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

        while in_conversation and not self.should_exit:
            try:
                # Record user's speech
                audio_data = self.record_audio()

                if not audio_data or len(audio_data) == 0:
                    print("⚠️ No audio data recorded")
                    continue

                # Convert speech to text
                whisper_start = time.time()
                user_input = self.transcribe_audio(audio_data)
                whisper_time = time.time() - whisper_start
                self.whisper_times.append(whisper_time)

                if not user_input:
                    if self.language == "chinese":
                        self.speak_text("我没听清楚。")
                    else:
                        self.speak_text("I didn't catch that.")
                    continue

                print(f"🎤 You said: '{user_input}'")
                self.conversation_turns += 1

                # Check if the user wants to end the conversation
                end_phrases = END_CONVERSATION_PHRASES
                if self.language == "chinese":
                    end_phrases = CHINESE_END_CONVERSATION_PHRASES + END_CONVERSATION_PHRASES

                if any(phrase in user_input.lower() for phrase in end_phrases):
                    print("🔚 Ending conversation")

                    # Calculate final statistics
                    self.session_duration = time.time() - self.session_start_time
                    self.whisper_avg_time = sum(self.whisper_times) / len(self.whisper_times) if self.whisper_times else 0
                    self.whisper_fastest = min(self.whisper_times) if self.whisper_times else 0
                    self.whisper_slowest = max(self.whisper_times) if self.whisper_times else 0

                    self.gemini_avg_time = sum(self.gemini_times) / len(self.gemini_times) if self.gemini_times else 0
                    self.gemini_fastest = min(self.gemini_times) if self.gemini_times else 0
                    self.gemini_slowest = max(self.gemini_times) if self.gemini_times else 0

                    self.edge_tts_avg_time = sum(self.edge_tts_times) / len(self.edge_tts_times) if self.edge_tts_times else 0
                    self.edge_tts_fastest = min(self.edge_tts_times) if self.edge_tts_times else 0
                    self.edge_tts_slowest = max(self.edge_tts_times) if self.edge_tts_times else 0

                    # Print session statistics
                    self.print_session_stats()

                    if self.language == "chinese":
                        self.speak_text("再见！")
                    else:
                        self.speak_text("Goodbye!")
                    in_conversation = False
                    return

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

                        print("👂 Continuing conversation... (say 'goodbye' to end)")
                    else:
                        # Fallback response if AI fails
                        if self.language == "chinese":
                            self.speak_text("抱歉，我无法生成回应。请再试一次。")
                        else:
                            self.speak_text("Sorry, I couldn't generate a response. Please try again.")
                except Exception as e:
                    print(f"❌ Error getting or speaking AI response: {e}")
                    import traceback
                    traceback.print_exc()
                    # Provide a fallback response
                    if self.language == "chinese":
                        self.speak_text("抱歉，出现了一个错误。请再试一次。")
                    else:
                        self.speak_text("Sorry, there was an error. Please try again.")
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

        print("\nText-to-Speech (Edge TTS):")
        print(f"  Average time: {self.edge_tts_avg_time:.2f} seconds")
        print(f"  Fastest: {self.edge_tts_fastest:.2f}s")
        print(f"  Slowest: {self.edge_tts_slowest:.2f}s")

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
    args = parser.parse_args()

    # Set global options
    global USE_MANUAL_RECORDING, ENABLE_INTERRUPTION
    USE_MANUAL_RECORDING = args.manual
    ENABLE_INTERRUPTION = not args.disable_interruption

    assistant = SimpleLocalAssistant(
        use_wake_word=not args.no_wake_word,
        debug=args.debug,
        language=args.language
    )
    assistant.run()

if __name__ == "__main__":
    main()
