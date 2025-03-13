#!/usr/bin/env python3
"""
Cobra VAD Integration Module for Voice Assistant

This module provides a CobraVAD class that can be integrated into the voice assistant
to improve speech detection using Picovoice Cobra.
"""

import os
import time
import numpy as np
import pvcobra
from pvrecorder import PvRecorder
from threading import Thread, Event
from queue import Queue
import wave
import struct

# Constants
SAMPLE_RATE = 16000
FRAME_LENGTH = 512  # Default frame length for PvRecorder
DEFAULT_THRESHOLD = 0.5  # Default voice probability threshold
SILENCE_DURATION = 2.0  # Seconds of silence to consider speech ended
BUFFER_PADDING = 0.5  # Seconds of audio to keep before speech starts

# New duration threshold
MIN_VOICE_DURATION = 0.3  # Minimum duration (in seconds) of voice activity to be considered speech


class CobraVAD:
    """
    Voice Activity Detection using Picovoice Cobra

    This class provides methods to detect voice activity in audio streams
    using Picovoice Cobra, which is more accurate than simple volume thresholds.
    """

    def __init__(self, access_key=None, threshold=DEFAULT_THRESHOLD, debug=False):
        """
        Initialize the Cobra VAD

        Args:
            access_key (str): Picovoice access key (can be loaded from env var PORCUPINE_ACCESS_KEY)
            threshold (float): Voice probability threshold (0.0 to 1.0)
            debug (bool): Whether to print debug information
        """
        # Get access key from environment if not provided
        if access_key is None:
            access_key = os.getenv("PORCUPINE_ACCESS_KEY")
            if not access_key:
                raise ValueError("Picovoice access key is required. Set PORCUPINE_ACCESS_KEY environment variable or pass access_key parameter.")

        self.access_key = access_key
        self.threshold = threshold
        self.debug = debug

        # Initialize Cobra
        self.cobra = pvcobra.create(access_key=self.access_key)

        # Initialize recorder
        self.recorder = None
        self.is_monitoring = False
        self.is_listening = False

        # Audio buffer and state
        self.audio_buffer = []
        self.pre_voice_buffer = []  # Buffer to keep audio before voice is detected
        self.is_voice_active = False
        self.last_voice_end_time = None
        self.voice_timeout = SILENCE_DURATION
        self.pre_buffer_size = int(BUFFER_PADDING * SAMPLE_RATE / FRAME_LENGTH)

        # Voice duration tracking
        self.voice_start_time = None
        self.current_voice_duration = 0
        self.min_voice_frames = int(MIN_VOICE_DURATION * SAMPLE_RATE / FRAME_LENGTH)
        self.voice_frames_count = 0

        # For continuous monitoring
        self.monitor_thread = None
        self.stop_event = Event()
        self.audio_queue = Queue()
        self.pause_event = Event()  # Add pause event

        if self.debug:
            print(f"Cobra VAD initialized with threshold: {self.threshold}")
            print(f"Frame length: {FRAME_LENGTH}")
            print(f"Pre-buffer size: {self.pre_buffer_size} frames ({BUFFER_PADDING} seconds)")
            print(f"Minimum voice duration: {MIN_VOICE_DURATION} seconds ({self.min_voice_frames} frames)")
            print(f"Silence duration for end: {SILENCE_DURATION} seconds")

    def pause_monitoring(self):
        """Temporarily pause voice monitoring"""
        if self.is_monitoring:
            self.pause_event.set()
            if self.debug:
                print("Paused voice monitoring")

    def resume_monitoring(self):
        """Resume voice monitoring after pause"""
        if self.is_monitoring:
            self.pause_event.clear()
            if self.debug:
                print("Resumed voice monitoring")

    def start_monitoring(self, device_index=-1):
        """
        Start continuous monitoring for voice activity

        Args:
            device_index (int): Audio device index to use (-1 for default)

        Returns:
            bool: True if monitoring started successfully
        """
        if self.is_monitoring:
            if self.debug:
                print("Already monitoring")
            return False

        self.stop_event.clear()
        self.is_monitoring = True

        # Start monitoring thread
        self.monitor_thread = Thread(target=self._monitor_thread, args=(device_index,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

        if self.debug:
            print("Started voice activity monitoring")

        return True

    def stop_monitoring(self):
        """Stop continuous monitoring"""
        if not self.is_monitoring:
            return

        self.stop_event.set()
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)

        self.is_monitoring = False

        if self.debug:
            print("Stopped voice activity monitoring")

    def _monitor_thread(self, device_index):
        """Thread function for continuous monitoring"""
        try:
            self.recorder = PvRecorder(
                device_index=device_index,
                frame_length=FRAME_LENGTH
            )

            if self.debug:
                print(f"Using audio device: {self.recorder.selected_device}")

            self.recorder.start()

            # Initialize buffers
            self.pre_voice_buffer = []
            self.audio_buffer = []
            self.is_voice_active = False
            self.last_voice_end_time = None
            self.voice_frames_count = 0

            # Main monitoring loop
            while not self.stop_event.is_set():
                # Check if monitoring is paused
                if self.pause_event.is_set():
                    time.sleep(0.1)  # Short sleep when paused
                    continue

                pcm = self.recorder.read()
                voice_probability = self.cobra.process(pcm)

                # Determine if voice is active based on threshold
                is_voice = voice_probability >= self.threshold

                if self.debug:
                    # Print voice probability with visual indicator
                    bar_length = int(voice_probability * 30)
                    bar = '█' * bar_length + '░' * (30 - bar_length)
                    status = "VOICE" if is_voice else "NOISE"
                    print(f"\rProb: {voice_probability:.4f} [{bar}] {status} | Voice frames: {self.voice_frames_count}/{self.min_voice_frames}", end='', flush=True)

                # Handle pre-voice buffer (circular buffer)
                if not self.is_voice_active:
                    self.pre_voice_buffer.append(pcm)
                    if len(self.pre_voice_buffer) > self.pre_buffer_size:
                        self.pre_voice_buffer.pop(0)

                # Handle voice activity
                if is_voice:
                    if not self.is_voice_active:
                        # Store the current frame while counting up to min_voice_frames
                        self.pre_voice_buffer.append(pcm)
                        self.voice_frames_count += 1

                        if self.voice_frames_count >= self.min_voice_frames:
                            # Voice activity confirmed
                            self.is_voice_active = True
                            self.is_listening = True
                            self.voice_start_time = time.time()

                            # Add pre-voice buffer to audio buffer (includes the initial voice frames)
                            self.audio_buffer = list(self.pre_voice_buffer)
                            self.pre_voice_buffer = []

                            if self.debug:
                                print("\nVoice activity started")
                    else:
                        # Continue recording voice
                        self.voice_frames_count += 1
                        self.last_voice_end_time = None
                        self.audio_buffer.append(pcm)
                elif self.is_voice_active:
                    # Voice was active but now silent
                    if self.last_voice_end_time is None:
                        self.last_voice_end_time = time.time()

                    # Add frame to buffer during timeout period
                    self.audio_buffer.append(pcm)

                    # Check if silence has persisted long enough using SILENCE_DURATION
                    if time.time() - self.last_voice_end_time > SILENCE_DURATION:
                        # Voice activity ended
                        if self.debug:
                            duration = time.time() - self.voice_start_time if self.voice_start_time else 0
                            print(f"\nVoice activity ended after {duration:.2f} seconds")

                        # Put the complete audio segment in the queue
                        audio_data = self._buffer_to_audio_data(self.audio_buffer)
                        self.audio_queue.put(audio_data)

                        # Reset state
                        self.is_voice_active = False
                        self.is_listening = False
                        self.audio_buffer = []
                        self.voice_start_time = None
                        self.voice_frames_count = 0
                else:
                    # No voice activity
                    self.voice_frames_count = 0

        except Exception as e:
            if self.debug:
                print(f"\nError in monitoring thread: {e}")
        finally:
            if self.recorder is not None:
                self.recorder.stop()
                self.recorder.delete()
                self.recorder = None

    def _buffer_to_audio_data(self, buffer):
        """Convert buffer of frames to audio data bytes"""
        audio_data = []
        for frame in buffer:
            audio_data.extend(frame)

        # Convert to bytes
        return struct.pack('h' * len(audio_data), *audio_data)

    def get_next_audio(self, timeout=None):
        """
        Get the next detected speech audio segment

        Args:
            timeout (float): Maximum time to wait for audio (None = wait forever)

        Returns:
            bytes: Audio data as bytes, or None if timeout
        """
        try:
            return self.audio_queue.get(timeout=timeout)
        except:
            return None

    # def record_audio(self, device_index=-1, max_duration=20.0):
    #     """
    #     Record a single audio segment with voice activity detection

    #     This method starts recording and returns when voice activity is detected
    #     and then ends, or when max_duration is reached.

    #     Args:
    #         device_index (int): Audio device index to use (-1 for default)
    #         max_duration (float): Maximum recording duration in seconds

    #     Returns:
    #         bytes: Audio data as bytes, or empty bytes if no speech detected
    #     """
    #     if self.is_monitoring:
    #         if self.debug:
    #             print("Already monitoring")
    #         return b''

    #     try:
    #         self.recorder = PvRecorder(
    #             device_index=device_index,
    #             frame_length=FRAME_LENGTH
    #         )

    #         if self.debug:
    #             print(f"Using audio device: {self.recorder.selected_device}")
    #             print("Listening for speech...")

    #         self.recorder.start()
    #         self.is_monitoring = True
    #         self.is_listening = True

    #         # Initialize buffers and state
    #         self.pre_voice_buffer = []
    #         self.audio_buffer = []
    #         self.is_voice_active = False
    #         self.last_voice_end_time = None

    #         # For timeout
    #         start_time = time.time()
    #         speech_detected = False

    #         # Main recording loop
    #         while self.is_monitoring:
    #             # Check for timeout
    #             if time.time() - start_time > max_duration:
    #                 if self.debug:
    #                     print(f"\nRecording timeout after {max_duration} seconds")
    #                 break

    #             pcm = self.recorder.read()
    #             voice_probability = self.cobra.process(pcm)

    #             # Determine if voice is active based on threshold
    #             is_voice = voice_probability >= self.threshold

    #             if self.debug:
    #                 # Print voice probability with visual indicator
    #                 bar_length = int(voice_probability * 30)
    #                 bar = '█' * bar_length + '░' * (30 - bar_length)
    #                 status = "VOICE" if is_voice else "NOISE"
    #                 print(f"\rProb: {voice_probability:.4f} [{bar}] {status}", end='', flush=True)

    #             # Handle pre-voice buffer (circular buffer)
    #             if not self.is_voice_active:
    #                 self.pre_voice_buffer.append(pcm)
    #                 if len(self.pre_voice_buffer) > self.pre_buffer_size:
    #                     self.pre_voice_buffer.pop(0)

    #             # Handle voice activity
    #             if is_voice:
    #                 if not self.is_voice_active:
    #                     # Store the current frame while counting up to min_voice_frames
    #                     self.pre_voice_buffer.append(pcm)
    #                     self.voice_frames_count += 1

    #                     if self.voice_frames_count >= self.min_voice_frames:
    #                         # Voice activity confirmed
    #                         self.is_voice_active = True
    #                         self.is_listening = True
    #                         self.voice_start_time = time.time()

    #                         # Add pre-voice buffer to audio buffer (includes the initial voice frames)
    #                         self.audio_buffer = list(self.pre_voice_buffer)
    #                         self.pre_voice_buffer = []

    #                         if self.debug:
    #                             print("\nVoice activity started")

    #                 # Reset silence timer
    #                 self.last_voice_end_time = None

    #                 # Add frame to audio buffer
    #                 self.audio_buffer.append(pcm)
    #             elif self.is_voice_active:
    #                 # Voice was active but now silent
    #                 if self.last_voice_end_time is None:
    #                     self.last_voice_end_time = time.time()

    #                 # Add frame to buffer during timeout period
    #                 self.audio_buffer.append(pcm)

    #                 # Check if silence timeout has elapsed
    #                 if time.time() - self.last_voice_end_time > self.voice_timeout:
    #                     # Voice activity ended
    #                     if self.debug:
    #                         print(f"\nVoice activity ended after {len(self.audio_buffer)} frames")
    #                     break

    #         # Convert buffer to audio data
    #         if speech_detected and self.audio_buffer:
    #             audio_data = self._buffer_to_audio_data(self.audio_buffer)
    #             if self.debug:
    #                 duration = len(audio_data) / (SAMPLE_RATE * 2)  # 16-bit = 2 bytes per sample
    #                 print(f"Recorded {duration:.2f} seconds of audio")
    #             return audio_data
    #         else:
    #             if self.debug:
    #                 print("No speech detected")
    #             return b''

    #     except Exception as e:
    #         if self.debug:
    #             print(f"\nError recording audio: {e}")
    #         return b''
    #     finally:
    #         self.is_monitoring = False
    #         self.is_listening = False
    #         if self.recorder is not None:
    #             self.recorder.stop()
    #             self.recorder.delete()
    #             self.recorder = None

    # def save_audio_to_file(self, audio_data, filename=None):
    #     """
    #     Save audio data to a WAV file

    #     Args:
    #         audio_data (bytes): Audio data to save
    #         filename (str): Filename to save to (default: generated based on timestamp)

    #     Returns:
    #         str: Path to saved file
    #     """
    #     if not audio_data:
    #         return None

    #     if filename is None:
    #         import datetime
    #         timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    #         filename = f"voice_{timestamp}.wav"

    #     with wave.open(filename, 'wb') as wf:
    #         wf.setnchannels(1)
    #         wf.setsampwidth(2)  # 16-bit audio
    #         wf.setframerate(SAMPLE_RATE)
    #         wf.writeframes(audio_data)

    #     if self.debug:
    #         print(f"Saved audio to {filename}")

    #     return filename

    def cleanup(self):
        """Clean up resources"""
        self.stop_monitoring()

        if hasattr(self, 'cobra') and self.cobra:
            self.cobra.delete()
            self.cobra = None

        if self.debug:
            print("Cleaned up Cobra VAD resources")
