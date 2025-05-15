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
from audio_monitor import AudioMonitor, SAMPLE_RATE, FRAME_LENGTH

# Import prompts from the prompts module (absolute import)
from prompts import SYSTEM_PROMPT, CHAT_PROMPTS

# Use faster-whisper for speech recognition
from faster_whisper import WhisperModel

# Constants
DEBUG_AUDIO = True
USE_FIXED_THRESHOLDS = True
USE_MANUAL_RECORDING = False
ENABLE_INTERRUPTION = False
USE_EDGE_TTS = True

# Timeout settings
UTTERANCE_TIMEOUT = 30.0  # Maximum time to wait for a single utterance (in seconds)
INACTIVITY_TIMEOUT = 30.0  # Time to wait for next user input before ending conversation (in seconds)

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

# End conversation phrases
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

        # Initialize audio monitor (combines VAD and wake word detection)
        self.audio_monitor = AudioMonitor(debug=DEBUG_AUDIO)

        # Initialize speech recognition
        self.init_speech_recognition()

        # Initialize TTS
        self.init_edge_tts()

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
        self.interrupted = False  # New flag for tracking interruption state
        self.playback_process = None  # Store the playback process for interruption

        print("✅ Voice assistant initialized and ready")

    def init_speech_recognition(self):
        """Initialize Whisper speech recognition model for both English and Chinese"""
        print("Loading Whisper ASR model...")
        try:
            # Use base model for faster loading and less memory usage
            model_size = "small" #"tiny"
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

    def init_chat_session(self):
        """Initialize chat session with Gemini"""
        try:
            # Create a new chat session with system prompt
            self.chat_session = self.model.start_chat(
                history=[
                    {"role": "user", "parts": [SYSTEM_PROMPT]},
                    {"role": "model", "parts": ["Understood. I am ready to assist you."]}
                ]
            )
            print("✅ Chat session initialized with system prompt")
        except Exception as e:
            print(f"❌ Error initializing chat session: {e}")
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
        """Listen for wake word using AudioMonitor"""
        print("👂 Listening for wake word...")

        try:
            while not self.should_exit:
                # Ensure monitoring is started
                if not self.audio_monitor.is_monitoring:
                    print("Starting audio monitoring for wake word detection...")
                    self.audio_monitor.start_monitoring()

                try:
                    # Wait for wake word detection
                    wake_word = self.audio_monitor.get_next_wake_word()
                    if wake_word:
                        print(f"🎯 Wake word detected: '{wake_word}'!")

                        # Start conversation
                        self.handle_conversation()

                        # After conversation, restart monitoring and go back to listening for wake word
                        if not self.should_exit:
                            print("Starting audio monitoring for wake word detection...")
                            self.audio_monitor.start_monitoring()
                            print("👂 Listening for wake word...")

                except Exception as e:
                    print(f"Error processing audio: {e}")
                    import traceback
                    traceback.print_exc()
                    # Try to restart monitoring
                    if not self.should_exit:
                        print("Attempting to restart audio monitoring...")
                        self.audio_monitor.stop_monitoring()
                        time.sleep(1)
                        self.audio_monitor.start_monitoring()

        except Exception as e:
            print(f"Error in wake word detection: {e}")
            import traceback
            traceback.print_exc()

        finally:
            self.audio_monitor.stop_monitoring()

    def record_audio(self):
        """Record audio from microphone using AudioMonitor"""
        print("\n" + "-"*50)
        print("🎤 Listening with VAD...")

        self.is_listening = True

        try:
            # Ensure monitoring is started and play sound
            if not self.audio_monitor.is_monitoring:
                print("Starting audio monitoring...")
                self.audio_monitor.start_monitoring()
                self.play_sound_effect("start_listening")
                # Add a small delay after starting monitoring
                time.sleep(0.2)

            # Clear any residual audio before starting new recording
            self.audio_monitor.clear_audio_buffer()

            # Get the next speech segment with timeout
            audio_data = self.audio_monitor.get_next_audio(timeout=UTTERANCE_TIMEOUT)

            # Print recording stats
            if audio_data:
                duration = len(audio_data) / (SAMPLE_RATE * 2)  # 16-bit audio at 16kHz
                print(f"Recording finished after {duration:.1f} seconds")
            else:
                print(f"⚠️ No speech detected within {UTTERANCE_TIMEOUT} seconds")

            return audio_data or b''

        except Exception as e:
            print(f"❌ Error recording audio: {e}")
            return b''
        finally:
            self.is_listening = False

    def transcribe_audio(self, audio_data):
        """Transcribe audio using Whisper"""
        if not self.whisper_model:
            print("❌ Whisper model not initialized")
            return None

        try:
            print("🎤 Transcribing audio...")

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
                if hasattr(self, 'stt_times'):
                    self.stt_times.append(stt_time)

                # Clean up the temporary file
                try:
                    os.unlink(temp_wav.name)
                except:
                    pass  # Ignore cleanup errors

                # Get the detected language and its probability
                detected_lang = info.language
                lang_prob = info.language_probability
                # print(f"🔍 Detected language: {detected_lang} (probability: {lang_prob:.2f})")

                # Update assistant language based on detected language
                if lang_prob > 0.5:  # Only update if confidence is high enough
                    if detected_lang == "zh":
                        self.language = "chinese"
                        lang_symbol = "🇨🇳"
                    elif detected_lang == "en":
                        self.language = "english"
                        lang_symbol = "🇺🇸"
                    else:
                        self.language = "others"
                        lang_symbol = "🌐"

                # Combine all segments into one text
                text = " ".join([segment.text for segment in segments]).strip()

                if not text:
                    print("❌ No speech detected in audio")
                    return None

                print(f"🎤 Transcribed text ({stt_time:.2f} seconds, {lang_symbol}): {text}")
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
            query += " (用中文提供简明扼要的回答，适合口语交流)"
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
                    "language": 'zh_CN' if self.language == 'chinese' else 'en'
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
                    print(f"🔍 Tavily answer: {results['answer']}")
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
                        formatted_answer = f"以下是搜索结果：\n" + "\n".join(formatted_results)
                    else:
                        formatted_answer = f"Here are the search results:\n" + "\n".join(formatted_results)
                    print(f"🔍 Formatted results: {formatted_answer}")

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

            # Use English prompts for any language other than Chinese
            prompt_language = "chinese" if self.language == "chinese" else "english"

            # Combined decision and response prompt
            decision_prompt = CHAT_PROMPTS[prompt_language].format(query=user_input)

            llm_start = time.time()

            # Get structured response from LLM
            response = self.chat_session.send_message(
                decision_prompt,
                stream=False
            )

            llm_time = time.time() - llm_start
            if hasattr(self, 'llm_times'):
                self.llm_times.append(llm_time)

            try:
                # Print raw response in a pretty format
                print(f"\n🔍 LLM Response ({llm_time:.2f} seconds):\n")
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

                    if search_results:
                        # Speak the search results immediately
                        self.speak_text(search_results)
                        return None  # Return None to indicate we've already spoken the response
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
            # Pause monitoring while speaking
            if self.audio_monitor.is_monitoring:
                self.audio_monitor.pause_monitoring()

            # Speak the text and get timing
            tts_time = self.speak_text_edge(text)
            if tts_time is not None:
                self.tts_times.append(tts_time)

        finally:
            # Clear any audio that might have accumulated during speaking
            self.audio_monitor.clear_audio_buffer()
            # Resume monitoring after speaking
            if self.audio_monitor.is_monitoring:
                self.audio_monitor.resume_monitoring()
                self.play_sound_effect("start_listening")

    def speak_text_edge(self, text):
        """Convert text to speech using Edge TTS and play it"""
        try:
            print(f"🔊 Speaking: '{text}'")
            self.is_speaking = True
            self.interrupted = False  # Reset interruption flag

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

            # Resume monitoring to detect wake words during playback
            # But keep VAD paused to avoid processing voice activity
            if self.audio_monitor.is_monitoring:
                self.audio_monitor.resume_monitoring()

            # Try different methods to play the audio file based on platform
            played_successfully = False

            # Platform-specific playback methods
            if sys.platform == "darwin":  # macOS
                try:
                    self.playback_process = subprocess.Popen(["afplay", temp_file])
                    while self.playback_process.poll() is None and not self.interrupted:
                        # Check for wake word while playing
                        wake_word = self.audio_monitor.get_next_wake_word(timeout=0.1)
                        if wake_word:
                            print(f"\n🎯 Wake word detected during playback: '{wake_word}'!")
                            self.interrupted = True
                            self.playback_process.terminate()
                            break
                    played_successfully = True
                except Exception as e:
                    print(f"⚠️ macOS audio playback failed: {e}")

            elif sys.platform == "win32":  # Windows
                try:
                    self.playback_process = subprocess.Popen(["wmplayer", temp_file, "/close"])
                    while self.playback_process.poll() is None and not self.interrupted:
                        wake_word = self.audio_monitor.get_next_wake_word(timeout=0.1)
                        if wake_word:
                            print(f"\n🎯 Wake word detected during playback: '{wake_word}'!")
                            self.interrupted = True
                            self.playback_process.terminate()
                            break
                    played_successfully = True
                except Exception as e:
                    print(f"⚠️ Windows Media Player failed: {e}")
                    try:
                        self.playback_process = subprocess.Popen(["start", temp_file], shell=True)
                        while self.playback_process.poll() is None and not self.interrupted:
                            wake_word = self.audio_monitor.get_next_wake_word(timeout=0.1)
                            if wake_word:
                                print(f"\n🎯 Wake word detected during playback: '{wake_word}'!")
                                self.interrupted = True
                                self.playback_process.terminate()
                                break
                        played_successfully = True
                    except Exception as e2:
                        print(f"⚠️ Windows shell playback failed: {e2}")

            elif sys.platform.startswith("linux"):  # Linux/Raspberry Pi
                # For Raspberry Pi, try ALSA first
                try:
                    # Use the configured audio device
                    alsa_device = f"hw:{self.audio_monitor.device_index},0" if self.audio_monitor.device_index >= 0 else "default"

                    # Try to set maximum volume for the device
                    try:
                        subprocess.run(["amixer", "-c", str(max(0, self.audio_monitor.device_index)), "sset", "PCM", "100%"],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    except Exception as e:
                        print(f"⚠️ Could not set volume: {e}")

                    # Try mpg123 with specific ALSA device first (since Edge TTS outputs MP3)
                    try:
                        self.playback_process = subprocess.Popen(
                            ["mpg123", "-a", alsa_device, "-q", temp_file],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL
                        )
                        while self.playback_process.poll() is None and not self.interrupted:
                            wake_word = self.audio_monitor.get_next_wake_word(timeout=0.1)
                            if wake_word:
                                print(f"\n🎯 Wake word detected during playback: '{wake_word}'!")
                                self.interrupted = True
                                self.playback_process.terminate()
                                break
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
                            self.playback_process = subprocess.Popen(
                                ["aplay", "-D", alsa_device, wav_file],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL
                            )
                            while self.playback_process.poll() is None and not self.interrupted:
                                wake_word = self.audio_monitor.get_next_wake_word(timeout=0.1)
                                if wake_word:
                                    print(f"\n🎯 Wake word detected during playback: '{wake_word}'!")
                                    self.interrupted = True
                                    self.playback_process.terminate()
                                    break
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
                        self.playback_process = subprocess.Popen(
                            [player, temp_file],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL
                        )
                        while self.playback_process.poll() is None and not self.interrupted:
                            wake_word = self.audio_monitor.get_next_wake_word(timeout=0.1)
                            if wake_word:
                                print(f"\n🎯 Wake word detected during playback: '{wake_word}'!")
                                self.interrupted = True
                                self.playback_process.terminate()
                                break
                        played_successfully = True
                        break
                    except (subprocess.SubprocessError, FileNotFoundError):
                        continue

            # Clean up the temporary file
            try:
                os.remove(temp_file)
            except:
                pass

            self.is_speaking = False
            self.playback_process = None

            # Return only the TTS generation time
            return tts_time

        except Exception as e:
            print(f"❌ Error speaking text with Edge TTS: {e}")
            import traceback
            traceback.print_exc()
            self.is_speaking = False
            self.playback_process = None
            return None

    def play_pre_recorded_message(self, message_type):
        """Play a pre-recorded message based on type and current language"""
        try:
            # Pause monitoring while playing message
            if self.audio_monitor.is_monitoring:
                self.audio_monitor.pause_monitoring()

            if message_type in PRE_RECORDED_MESSAGES:
                sound_effect = PRE_RECORDED_MESSAGES[message_type][self.language]
                # Use blocking mode for pre-recorded messages to ensure they complete
                self.play_sound_effect(sound_effect, blocking=True)
            else:
                print(f"⚠️ No pre-recorded message found for: {message_type}")
        finally:
            # Resume monitoring after message is fully played
            if self.audio_monitor.is_monitoring:
                self.audio_monitor.resume_monitoring()

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
            # Start audio monitoring at the beginning of conversation
            if not self.audio_monitor.is_monitoring:
                print("Starting audio monitoring for conversation...")
                self.audio_monitor.start_monitoring()
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

                    # Ensure monitoring is started before recording
                    if not self.audio_monitor.is_monitoring:
                        print("Restarting audio monitoring...")
                        self.audio_monitor.start_monitoring()
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
            # Stop audio monitoring when conversation ends
            if self.audio_monitor.is_monitoring:
                print("Stopping audio monitoring at end of conversation...")
                self.audio_monitor.stop_monitoring()

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
        if hasattr(self, 'audio_monitor'):
            self.audio_monitor.cleanup()
        if hasattr(self, 'whisper_model'):
            del self.whisper_model
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
