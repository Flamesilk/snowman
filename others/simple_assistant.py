#!/usr/bin/env python3
"""
Simple Voice Assistant using PipeCat and Daily

This script implements a simplified voice assistant that:
- Uses PipeCat for pipeline orchestration
- Uses Daily for WebRTC audio transport
- Processes speech using Silero (local, no API key needed)
- Generates responses using Google's Gemini model
- Converts responses to speech using Silero (local, no API key needed)
"""

import os
import asyncio
import logging
from dotenv import load_dotenv

# PipeCat imports
from pipecat.frames.frames import TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.pipeline.runner import PipelineRunner
from pipecat.services.google import GoogleLLMService
from pipecat.services.silero import SileroASRService, SileroTTSService
from pipecat.transports.services.daily import DailyParams, DailyTransport

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("simple_assistant")

# Load environment variables
load_dotenv()

async def main():
    """Main function to run the simple voice assistant."""
    logger.info("Starting simple voice assistant...")

    # Validate required environment variables
    required_vars = [
        "DAILY_ROOM_URL", "GOOGLE_API_KEY"
    ]

    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        return

    # Get configuration from environment
    daily_room_url = os.getenv("DAILY_ROOM_URL")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    use_silero_asr = os.getenv("USE_SILERO_ASR", "true").lower() == "true"
    use_silero_tts = os.getenv("USE_SILERO_TTS", "true").lower() == "true"
    silero_voice = os.getenv("SILERO_VOICE", "en_0")

    # Import services based on configuration if not using Silero
    if not use_silero_asr:
        from pipecat.services.deepgram import DeepgramASRService
        if not os.getenv("DEEPGRAM_API_KEY"):
            logger.error("Missing DEEPGRAM_API_KEY environment variable, falling back to Silero")
            use_silero_asr = True

    if not use_silero_tts:
        from pipecat.services.cartesia import CartesiaTTSService
        if not os.getenv("CARTESIA_API_KEY") or not os.getenv("CARTESIA_VOICE_ID"):
            logger.error("Missing CARTESIA_API_KEY or CARTESIA_VOICE_ID environment variables, falling back to Silero")
            use_silero_tts = True

    try:
        # Initialize PipeCat runner
        runner = PipelineRunner()

        # Set up Daily transport
        transport = DailyTransport(
            room_url=daily_room_url,
            token="",  # leave empty as noted in the example
            bot_name="Simple Voice Assistant",
            params=DailyParams(
                audio_in_enabled=True,
                audio_out_enabled=True
            )
        )

        # Create pipeline components
        pipeline_components = [
            # Input from Daily transport
            transport.input(),
        ]

        # Add speech recognition (ASR) service
        if use_silero_asr:
            logger.info("Using Silero for speech recognition (local, no API key needed)")
            pipeline_components.append(
                SileroASRService(
                    language="en",
                    interim_results=True
                )
            )
        else:
            logger.info("Using Deepgram for speech recognition (cloud-based)")
            pipeline_components.append(
                DeepgramASRService(
                    api_key=os.getenv("DEEPGRAM_API_KEY"),
                    model="nova-2",
                    language="en-US",
                    smart_format=True,
                    diarize=False,
                    interim_results=True,
                    endpointing=True,
                    vad_events=True
                )
            )

        # Add LLM service (Google Gemini)
        logger.info("Using Google Gemini for AI conversation")
        pipeline_components.append(
            GoogleLLMService(
                api_key=google_api_key,
                model="gemini-1.5-pro",
                system_prompt=(
                    "You are a helpful voice assistant. Keep your responses concise and conversational. "
                    "If you don't know something, just say so. Don't make up information."
                )
            )
        )

        # Add text-to-speech (TTS) service
        if use_silero_tts:
            logger.info(f"Using Silero for text-to-speech (local, no API key needed) with voice: {silero_voice}")
            pipeline_components.append(
                SileroTTSService(
                    voice=silero_voice
                )
            )
        else:
            logger.info("Using Cartesia for text-to-speech (cloud-based)")
            pipeline_components.append(
                CartesiaTTSService(
                    api_key=os.getenv("CARTESIA_API_KEY"),
                    voice_id=os.getenv("CARTESIA_VOICE_ID")
                )
            )

        # Add output to Daily transport
        pipeline_components.append(transport.output())

        # Create the conversation pipeline
        conversation_pipeline = Pipeline(pipeline_components)

        # Create the conversation task
        conversation_task = PipelineTask(conversation_pipeline)

        # Register event handlers for the Daily transport
        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            """Handle when a participant joins the session."""
            participant_name = participant.get("info", {}).get("userName", "")
            logger.info(f"Participant joined: {participant_name}")

            # Queue a welcome message
            welcome_message = f"Hello {participant_name}! I'm your voice assistant powered by Google Gemini with Silero voice technology. How can I help you today?"
            await conversation_task.queue_frame(TextFrame(welcome_message))

        @transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            """Handle when a participant leaves the session."""
            logger.info(f"Participant left: {participant.get('info', {}).get('userName', '')}")
            await conversation_task.cancel()

        # Run the conversation task
        logger.info("Starting conversation pipeline...")
        await runner.run(conversation_task)

    except Exception as e:
        logger.error(f"Error running simple voice assistant: {e}")
    finally:
        logger.info("Simple voice assistant shutting down...")
        if 'transport' in locals():
            await transport.leave()
        logger.info("Simple voice assistant shut down complete")


if __name__ == "__main__":
    asyncio.run(main())
