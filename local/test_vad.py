#!/usr/bin/env python3
"""
Test script for Picovoice Cobra Voice Activity Detection (VAD)
This script provides two test modes:
1. Continuous monitoring: Shows real-time voice probability
2. Single recording: Records individual speech segments
"""

import argparse
import os
import struct
import time
import wave
from datetime import datetime
from threading import Thread

import numpy as np
import pvcobra
from pvrecorder import PvRecorder
from cobra_vad import CobraVAD

# Try to load dotenv if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Constants
SAMPLE_RATE = 16000
FRAME_LENGTH = 512  # Default frame length for PvRecorder


class CobraVADDemo:
    def __init__(self, access_key, audio_device_index=-1, show_audio_devices=False,
                 threshold=0.5, save_audio=False):
        """
        Initialize the Cobra VAD demo

        Args:
            access_key (str): Picovoice access key
            audio_device_index (int): Audio device index to use (-1 for default)
            show_audio_devices (bool): Whether to show available audio devices
            threshold (float): Voice probability threshold (0.0 to 1.0)
            save_audio (bool): Whether to save audio when voice is detected
        """
        self.access_key = access_key
        self.audio_device_index = audio_device_index
        self.threshold = threshold
        self.save_audio = save_audio

        # Initialize Cobra
        self.cobra = pvcobra.create(access_key=self.access_key)

        # Show audio devices if requested
        if show_audio_devices:
            self._show_audio_devices()

        # Initialize recorder
        self.recorder = None
        self.is_recording = False

        # For saving audio
        self.voice_audio_buffer = []
        self.is_voice_active = False
        self.last_voice_end_time = None
        self.voice_timeout = 1.0  # seconds

    def _show_audio_devices(self):
        """Show available audio devices"""
        devices = PvRecorder.get_available_devices()
        print("Available audio devices:")
        for i, device in enumerate(devices):
            print(f"[{i}] {device}")

    def start(self):
        """Start recording and VAD processing"""
        try:
            self.recorder = PvRecorder(
                device_index=self.audio_device_index,
                frame_length=FRAME_LENGTH
            )

            print(f"Using device: {self.recorder.selected_device}")

            self.recorder.start()
            self.is_recording = True

            print("Listening... (Press Ctrl+C to exit)")

            # Start a separate thread for saving audio if enabled
            if self.save_audio:
                save_thread = Thread(target=self._save_voice_audio_thread)
                save_thread.daemon = True
                save_thread.start()

            # Main processing loop
            while self.is_recording:
                pcm = self.recorder.read()
                voice_probability = self.cobra.process(pcm)

                # Determine if voice is active based on threshold
                is_voice = voice_probability >= self.threshold

                # Print voice probability with visual indicator
                bar_length = int(voice_probability * 50)
                bar = '█' * bar_length + '░' * (50 - bar_length)
                status = "VOICE" if is_voice else "NOISE"
                print(f"\rProb: {voice_probability:.4f} [{bar}] {status}", end='', flush=True)

                # Handle voice activity for saving audio
                if self.save_audio:
                    if is_voice:
                        self.is_voice_active = True
                        self.last_voice_end_time = None
                        self.voice_audio_buffer.append(pcm)
                    elif self.is_voice_active:
                        # Keep recording for a short time after voice stops
                        if self.last_voice_end_time is None:
                            self.last_voice_end_time = time.time()

                        # Add frame to buffer during timeout period
                        self.voice_audio_buffer.append(pcm)

                        # Check if timeout has elapsed
                        if time.time() - self.last_voice_end_time > self.voice_timeout:
                            # Signal to save the audio
                            self.is_voice_active = False

        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.is_recording = False
            if self.recorder is not None:
                self.recorder.stop()
                self.recorder.delete()
            self.cobra.delete()
            print("\nCleaned up resources")

    def _save_voice_audio_thread(self):
        """Thread function to save voice audio when detected"""
        while self.is_recording:
            # Check if we have a completed voice segment to save
            if not self.is_voice_active and self.voice_audio_buffer:
                # Create a timestamp for the filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"voice_{timestamp}.wav"

                print(f"\nSaving voice audio to {filename}...")

                # Convert buffer to audio data
                audio_data = []
                for frame in self.voice_audio_buffer:
                    audio_data.extend(frame)

                # Save as WAV file
                with wave.open(filename, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # 16-bit audio
                    wf.setframerate(SAMPLE_RATE)
                    wf.writeframes(struct.pack('h' * len(audio_data), *audio_data))

                print(f"Saved {len(self.voice_audio_buffer)} frames ({len(audio_data) / SAMPLE_RATE:.2f} seconds)")

                # Clear the buffer
                self.voice_audio_buffer = []

            # Sleep to avoid consuming too much CPU
            time.sleep(0.1)


def test_single_recording(access_key, threshold=0.5, save=True):
    """Test CobraVAD in single recording mode"""
    print("\nTesting CobraVAD Single Recording Mode")
    print("======================================")

    vad = CobraVAD(
        access_key=access_key,
        threshold=threshold,
        debug=True
    )

    try:
        while True:
            print("\nListening for speech...")
            audio_data = vad.get_next_audio(timeout=10.0)
            if audio_data:
                print(f"Recorded {len(audio_data) / (SAMPLE_RATE * 2):.2f} seconds of audio")

                if save:
                    filename = vad.save_audio_to_file(audio_data)
                    if filename:
                        print(f"Saved audio to {filename}")

            print("Press Ctrl+C to exit or wait for next recording...")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        vad.cleanup()


def main():
    parser = argparse.ArgumentParser(description='Cobra VAD Demo')
    parser.add_argument('--access_key', help='Picovoice access key')
    parser.add_argument('--audio_device_index', type=int, default=-1,
                        help='Index of audio device to use (-1 for default)')
    parser.add_argument('--show_audio_devices', action='store_true',
                        help='Show available audio devices and exit')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Voice probability threshold (0.0 to 1.0)')
    parser.add_argument('--save_audio', action='store_true',
                        help='Save audio when voice is detected')
    parser.add_argument('--mode', choices=['continuous', 'single'], default='continuous',
                        help='Test mode: continuous monitoring or single recording')

    args = parser.parse_args()

    # Get access key from command line or environment
    access_key = args.access_key
    if not access_key:
        access_key = os.getenv("PORCUPINE_ACCESS_KEY")
        if not access_key:
            print("Error: Access key is required. Provide it with --access_key or set PORCUPINE_ACCESS_KEY environment variable.")
            return

    if args.mode == 'single':
        test_single_recording(access_key, threshold=args.threshold, save=args.save_audio)
    else:
        demo = CobraVADDemo(
            access_key=access_key,
            audio_device_index=args.audio_device_index,
            show_audio_devices=args.show_audio_devices,
            threshold=args.threshold,
            save_audio=args.save_audio
        )

        if args.show_audio_devices:
            # Already shown in constructor
            return

        demo.start()


if __name__ == "__main__":
    main()
