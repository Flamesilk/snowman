#!/usr/bin/env python3
"""
Wake Word Voice Assistant using PipeCat, Daily, and Porcupine

This script implements a voice assistant that:
- Uses wake word detection to activate (Porcupine)
- Uses PipeCat for pipeline orchestration
- Uses Daily for WebRTC audio transport
- Processes speech using Silero (local, no API key needed)
- Generates responses using Google's Gemini model
- Converts responses to speech using Silero (local, no API key needed)
- Handles interruptions and turn detection
"""

import os
import asyncio
import logging
import signal
import time
from enum import Enum
from dotenv import load_dotenv

# PipeCat imports
from pipecat.frames.frames import (
    AudioFrame,
    TextFrame,
    TranscriptionFrame,
    TranscriptionState
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.pipeline.runner import PipelineRunner
from pipecat.services.google import GoogleLLMService
from pipecat.services.silero import SileroASRService, SileroTTSService
from pipecat.transports.services.daily import DailyParams, DailyTransport

# Wake word detection
import pvporcupine
from pvrecorder import PvRecorder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("wake_word_assistant")

# Load environment variables
load_dotenv()

# Assistant states
class AssistantState(Enum):
    STANDBY = 0    # Listening for wake word
    ACTIVE = 1     # Actively listening to user
    THINKING = 2   # Processing user input
    SPEAKING = 3   # Speaking response to user

class WakeWordAssistant:
    """
    Voice assistant with wake word detection using PipeCat and Porcupine.
    """

    def __init__(self):
        """Initialize the voice assistant."""
        # Validate required environment variables
        required_vars = [
            "DAILY_ROOM_URL", "GOOGLE_API_KEY", "PICOVOICE_ACCESS_KEY"
        ]

        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

        # Get configuration from environment
        self.daily_room_url = os.getenv("DAILY_ROOM_URL")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.porcupine_access_key = os.getenv("PICOVOICE_ACCESS_KEY")
        self.wake_keywords = os.getenv("WAKE_KEYWORDS", "hey_assistant,computer").split(",")

        # Speech and TTS configuration
        self.use_silero_asr = os.getenv("USE_SILERO_ASR", "true").lower() == "true"
        self.use_silero_tts = os.getenv("USE_SILERO_TTS", "true").lower() == "true"
        self.silero_voice = os.getenv("SILERO_VOICE", "en_0")

        # Import services based on configuration if not using Silero
        if not self.use_silero_asr:
            from pipecat.services.deepgram import DeepgramASRService
            self.ASRService = DeepgramASRService
            if not os.getenv("DEEPGRAM_API_KEY"):
                logger.warning("Missing DEEPGRAM_API_KEY environment variable, falling back to Silero")
                self.use_silero_asr = True
                self.ASRService = SileroASRService
        else:
            self.ASRService = SileroASRService

        if not self.use_silero_tts:
            from pipecat.services.cartesia import CartesiaTTSService
            self.TTSService = CartesiaTTSService
            if not os.getenv("CARTESIA_API_KEY") or not os.getenv("CARTESIA_VOICE_ID"):
                logger.warning("Missing CARTESIA_API_KEY or CARTESIA_VOICE_ID environment variables, falling back to Silero")
                self.use_silero_tts = True
                self.TTSService = SileroTTSService
        else:
            self.TTSService = SileroTTSService

        # Initialize state variables
        self.state = AssistantState.STANDBY
        self.runner = None
        self.transport = None
        self.wake_word_detector = None
        self.recorder = None
        self.conversation_task = None
        self.is_running = False
        self.inactivity_timeout = 60  # seconds
        self.last_activity_time = time.time()

        # Log the configuration
        logger.info(f"Speech recognition: {'Silero (local)' if self.use_silero_asr else 'Deepgram (cloud)'}")
        logger.info(f"Text-to-speech: {'Silero (local)' if self.use_silero_tts else 'Cartesia (cloud)'}")
        if self.use_silero_tts:
            logger.info(f"Silero voice: {self.silero_voice}")

    async def setup(self):
        """Set up the voice assistant components."""
        logger.info("Setting up wake word assistant...")

        # Initialize PipeCat runner
        self.runner = PipelineRunner()

        # Set up Daily transport
        self.transport = DailyTransport(
            room_url=self.daily_room_url,
            token="",  # leave empty as noted in the example
            bot_name="Wake Word Assistant",
            params=DailyParams(
                audio_in_enabled=True,
                audio_out_enabled=True
            )
        )

        # Set up wake word detection
        try:
            # Create Porcupine instance for wake word detection
            self.wake_word_detector = pvporcupine.create(
                access_key=self.porcupine_access_key,
                keywords=self.wake_keywords
            )

            # Create audio recorder
            self.recorder = PvRecorder(
                device_index=-1,  # default microphone
                frame_length=self.wake_word_detector.frame_length
            )

            logger.info(f"Wake word detection set up with keywords: {self.wake_keywords}")
        except Exception as e:
            logger.error(f"Failed to set up wake word detection: {e}")
            raise

        # Register event handlers
        @self.transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            """Handle when a participant joins the session."""
            participant_name = participant.get("info", {}).get("userName", "")
            logger.info(f"Participant joined: {participant_name}")

            # Create a welcome message
            welcome_message = f"Hello {participant_name}! I'm your voice assistant powered by Google Gemini with Silero voice technology. Say my wake word to start a conversation."

            # Create a pipeline for TTS only
            tts_pipeline = Pipeline([
                self._create_tts_service(),
                self.transport.output()
            ])

            # Create and run a task for the welcome message
            welcome_task = PipelineTask(tts_pipeline)
            await welcome_task.queue_frame(TextFrame(welcome_message))
            await self.runner.run(welcome_task)

            # Start listening for wake words
            await self.start_standby_mode()

        @self.transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            """Handle when a participant leaves the session."""
            logger.info(f"Participant left: {participant.get('info', {}).get('userName', '')}")
            if self.conversation_task:
                await self.conversation_task.cancel()
            self.is_running = False

        logger.info("Wake word assistant setup complete")

    def _create_asr_service(self):
        """Create the appropriate ASR service based on configuration."""
        if self.use_silero_asr:
            logger.info("Using Silero for speech recognition (local, no API key needed)")
            return self.ASRService(
                language="en",
                interim_results=True
            )
        else:
            logger.info("Using Deepgram for speech recognition (cloud-based)")
            return self.ASRService(
                api_key=os.getenv("DEEPGRAM_API_KEY"),
                model="nova-2",
                language="en-US",
                smart_format=True,
                diarize=False,
                interim_results=True,
                endpointing=True,
                vad_events=True
            )

    def _create_tts_service(self):
        """Create the appropriate TTS service based on configuration."""
        if self.use_silero_tts:
            logger.info(f"Using Silero for text-to-speech (local, no API key needed) with voice: {self.silero_voice}")
            return self.TTSService(
                voice=self.silero_voice
            )
        else:
            logger.info("Using Cartesia for text-to-speech (cloud-based)")
            return self.TTSService(
                api_key=os.getenv("CARTESIA_API_KEY"),
                voice_id=os.getenv("CARTESIA_VOICE_ID")
            )

    async def start_standby_mode(self):
        """Start standby mode, listening for wake words."""
        logger.info("Starting standby mode, listening for wake words...")
        self.state = AssistantState.STANDBY
        self.is_running = True

        try:
            self.recorder.start()

            while self.is_running and self.state == AssistantState.STANDBY:
                # Process audio frame for wake word detection
                pcm = self.recorder.read()
                keyword_index = self.wake_word_detector.process(pcm)

                # If wake word detected, activate the assistant
                if keyword_index >= 0:
                    detected_keyword = self.wake_keywords[keyword_index]
                    logger.info(f"Wake word detected: {detected_keyword}")
                    await self.activate_assistant()
                    break

                # Small sleep to prevent CPU hogging
                await asyncio.sleep(0.01)

        except Exception as e:
            logger.error(f"Error in standby mode: {e}")
        finally:
            if self.recorder and self.recorder.is_recording:
                self.recorder.stop()

    async def activate_assistant(self):
        """Activate the assistant to start a conversation."""
        logger.info("Activating assistant...")
        self.state = AssistantState.ACTIVE
        self.last_activity_time = time.time()

        # Create the conversation pipeline
        conversation_pipeline = Pipeline([
            # Input from Daily transport
            self.transport.input(),

            # Speech recognition
            self._create_asr_service(),

            # LLM processing with Google Gemini
            GoogleLLMService(
                api_key=self.google_api_key,
                model="gemini-1.5-pro",
                system_prompt=(
                    "You are a helpful voice assistant. Keep your responses concise and conversational. "
                    "If you don't know something, just say so. Don't make up information."
                )
            ),

            # Text-to-speech
            self._create_tts_service(),

            # Output to Daily transport
            self.transport.output()
        ])

        # Create the conversation task
        self.conversation_task = PipelineTask(conversation_pipeline)

        # Add a greeting
        await self.conversation_task.queue_frame(TextFrame("I'm listening. How can I help you?"))

        # Register frame handlers for state management
        @self.conversation_task.frame_handler(TranscriptionFrame)
        async def on_transcription(frame: TranscriptionFrame):
            """Handle transcription frames to detect user activity and turns."""
            if frame.state == TranscriptionState.COMPLETE:
                logger.info(f"User said: {frame.text}")
                self.state = AssistantState.THINKING
                self.last_activity_time = time.time()
            elif frame.state == TranscriptionState.INTERIM:
                # Update activity time when user is speaking
                self.last_activity_time = time.time()

        @self.conversation_task.frame_handler(TextFrame)
        async def on_assistant_response(frame: TextFrame):
            """Handle assistant responses."""
            logger.info(f"Assistant response: {frame.text}")
            self.state = AssistantState.SPEAKING
            self.last_activity_time = time.time()

        @self.conversation_task.frame_handler(AudioFrame)
        async def on_audio_frame(frame: AudioFrame):
            """Handle audio frames to detect when assistant stops speaking."""
            if self.state == AssistantState.SPEAKING and frame.is_last:
                logger.info("Assistant finished speaking")
                self.state = AssistantState.ACTIVE

                # Check for inactivity timeout
                current_time = time.time()
                if current_time - self.last_activity_time > self.inactivity_timeout:
                    logger.info(f"Inactivity timeout ({self.inactivity_timeout}s) reached, returning to standby mode")
                    await self.return_to_standby()

        # Run the conversation task
        try:
            await self.runner.run(self.conversation_task)
        except Exception as e:
            logger.error(f"Error in conversation: {e}")
        finally:
            self.conversation_task = None

            # Return to standby mode if we're not already there
            if self.state != AssistantState.STANDBY:
                await self.return_to_standby()

    async def return_to_standby(self):
        """Return to standby mode after a conversation."""
        logger.info("Returning to standby mode")

        # Cancel the conversation task if it exists
        if self.conversation_task:
            await self.conversation_task.cancel()
            self.conversation_task = None

        # Create a pipeline for TTS only to inform the user
        tts_pipeline = Pipeline([
            self._create_tts_service(),
            self.transport.output()
        ])

        # Create and run a task for the standby message
        standby_task = PipelineTask(tts_pipeline)
        await standby_task.queue_frame(TextFrame("I'll be here if you need me. Just say my wake word."))
        await self.runner.run(standby_task)

        # Start standby mode
        await self.start_standby_mode()

    async def run(self):
        """Run the voice assistant."""
        logger.info("Starting wake word assistant...")

        # Set up signal handlers for graceful shutdown
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))

        try:
            # Set up the assistant
            await self.setup()

            # Create a task that just keeps the transport running
            transport_task = PipelineTask(Pipeline([self.transport.keepalive()]))
            await self.runner.run(transport_task)

        except Exception as e:
            logger.error(f"Error running wake word assistant: {e}")
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Shut down the voice assistant."""
        logger.info("Shutting down wake word assistant...")
        self.is_running = False

        # Stop the recorder if it's running
        if self.recorder and self.recorder.is_recording:
            self.recorder.stop()

        # Delete Porcupine instance
        if self.wake_word_detector:
            self.wake_word_detector.delete()

        # Delete recorder instance
        if self.recorder:
            self.recorder.delete()

        # Cancel conversation task if it exists
        if self.conversation_task:
            await self.conversation_task.cancel()

        # Leave the Daily room
        if self.transport:
            await self.transport.leave()

        logger.info("Wake word assistant shut down complete")


async def main():
    """Main function to run the wake word assistant."""
    try:
        assistant = WakeWordAssistant()
        await assistant.run()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
