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
import alsaaudio

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

    # Constants for visualization
    METER_WIDTH = 50
    TARGET_VOLUME = 500
    AMPLIFICATION_FACTOR = 5.0

    # First, check available devices
    devices = PvRecorder.get_available_devices()
    print("\nAvailable audio devices:")
    for i, device in enumerate(devices):
        print(f"[{i}] {device}")

    # Try to find the best recording device
    device_index = -1

    # First priority: Look for Raspiaudio device (for Raspberry Pi)
    for i, device in enumerate(devices):
        if "googlevoicehat" in device.lower():
            device_index = i
            print(f"\nFound Raspiaudio device at index {i}: {device}")
            break

    # Second priority: Look for a real input device (skip "monitor" devices)
    if device_index == -1:
        for i, device in enumerate(devices):
            # Skip monitor/loopback devices
            if "monitor" not in device.lower():
                device_index = i
                print(f"\nFound input device at index {i}: {device}")
                break

    # Last resort: Use index 1 if it exists
    if device_index == -1 and len(devices) > 1:
        device_index = 1
        print(f"\nUsing default input device at index 1: {devices[1]}")

    # If still no device found, use default
    if device_index == -1:
        device_index = -1
        print("\nNo specific input device found, using system default")

    vad = CobraVAD(
        access_key=access_key,
        threshold=threshold,
        debug=True
    )

    try:
        while True:
            print("\nListening for speech...")
            print("VAD meter below (0.0 to 1.0):")
            print("V = Voice probability  | = Threshold (0.5)  N = Noise")
            print(f"Using audio device index: {device_index}")

            # Initialize tracking
            volume_history = []
            vad_history = []  # Track VAD values
            peak_hold = 0
            max_vad = 0.0  # Track maximum VAD value seen

            # Start recording and show real-time VAD
            recorder = PvRecorder(device_index=device_index, frame_length=FRAME_LENGTH)
            recorder.start()

            try:
                while True:
                    pcm = recorder.read()
                    # Calculate volume
                    volume_norm = np.abs(np.array(pcm)).mean()
                    volume_history.append(volume_norm)
                    if len(volume_history) > 5:
                        volume_history.pop(0)
                    avg_volume = sum(volume_history) / len(volume_history)

                    # Get voice probability
                    voice_prob = vad.cobra.process(pcm)
                    vad_history.append(voice_prob)
                    if len(vad_history) > 10:  # Keep last 10 VAD values
                        vad_history.pop(0)
                    max_vad = max(max_vad, voice_prob)  # Track maximum VAD value

                    # Create visualization
                    # Voice probability meter (0.0 to 1.0 range)
                    vad_pos = int(voice_prob * METER_WIDTH)
                    threshold_pos = int(threshold * METER_WIDTH)

                    # Build VAD meter
                    vad_meter = ["N"] * METER_WIDTH  # Start with noise indicators
                    # Fill voice probability
                    for j in range(min(vad_pos, METER_WIDTH)):
                        vad_meter[j] = "V"
                    # Add threshold marker
                    if 0 <= threshold_pos < METER_WIDTH:
                        vad_meter[threshold_pos] = "|"

                    # Show status and values
                    status = "VOICE DETECTED" if voice_prob >= threshold else "silence"
                    avg_vad = sum(vad_history) / len(vad_history) if vad_history else 0

                    # Volume indicators - adjusted thresholds for lower input levels
                    vol_status = "TOO LOW" if avg_volume < 30 else "GOOD" if avg_volume < 1000 else "TOO HIGH"
                    # More granular volume meter
                    vol_meter = "▁" * min(int(avg_volume / 10), 50)  # Adjusted scale for lower volumes

                    # Print status (using multiple lines with ANSI escape codes)
                    print(f"\rVAD: 0.0 [{''.join(vad_meter)}] 1.0  {voice_prob:.3f} (avg: {avg_vad:.3f}, max: {max_vad:.3f})", end='')
                    print(f"\nVol: {avg_volume:6.0f} [{vol_meter}] {vol_status} {'[SPEAKING]' if voice_prob > threshold else ''}", end='\033[F')

                    # Check for voice detection
                    if voice_prob >= threshold:
                        print("\n\nVoice detected! Recording...")
                        break

            except KeyboardInterrupt:
                print("\nStopping...")
                recorder.stop()
                recorder.delete()
                break

            # Get the voice audio
            audio_data = vad.get_next_audio(timeout=10.0)

            if audio_data:
                # Convert to numpy array for processing
                audio_np = np.frombuffer(audio_data, dtype=np.int16)
                duration = len(audio_np) / (SAMPLE_RATE * 2)  # 2 bytes per sample
                print(f"\nRecorded {duration:.2f} seconds of audio")

                # Calculate and show volume stats
                volume = np.abs(audio_np).mean()
                max_vol = np.abs(audio_np).max()
                print(f"Average volume: {volume:.0f}")
                print(f"Peak volume: {max_vol:.0f}")

                if save:
                    # Apply amplification before saving
                    print(f"\nApplying {AMPLIFICATION_FACTOR}x amplification...")
                    audio_np = audio_np.astype(np.float32) * AMPLIFICATION_FACTOR
                    audio_np = np.clip(audio_np, -32768, 32767)
                    audio_np = audio_np.astype(np.int16)

                    # Save amplified audio
                    filename = f"vad_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
                    with wave.open(filename, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(SAMPLE_RATE)
                        wf.writeframes(audio_np.tobytes())
                    print(f"Saved amplified audio to {filename}")

                    # Play back the recording
                    try:
                        print("\nPlaying back the recording...")
                        speaker = alsaaudio.PCM(
                            alsaaudio.PCM_PLAYBACK,
                            channels=1,
                            rate=SAMPLE_RATE,
                            format=alsaaudio.PCM_FORMAT_S16_LE,
                            periodsize=FRAME_LENGTH,
                            device='hw:1,0'
                        )

                        with wave.open(filename, 'rb') as wf:
                            data = wf.readframes(FRAME_LENGTH)
                            while data:
                                speaker.write(data)
                                data = wf.readframes(FRAME_LENGTH)
                        print("Playback finished")
                    except Exception as e:
                        print(f"Playback failed: {e}")

            print("\nPress Ctrl+C to exit or wait for next recording...")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        vad.cleanup()


def get_alsa_devices():
    """Get ALSA devices using arecord"""
    try:
        import subprocess
        result = subprocess.run(['arecord', '-l'], capture_output=True, text=True)
        return result.stdout
    except Exception as e:
        print(f"Error getting ALSA devices: {e}")
        return ""


def test_mic(device_index=-1):
    """Test microphone recording using PvRecorder"""
    FRAME_LENGTH = 512  # Default frame length for PvRecorder
    RECORD_SECONDS = 5
    WAVE_OUTPUT_FILENAME = "test_pvrecorder.wav"
    TARGET_VOLUME = 500  # Adjusted for PvRecorder's different scale
    METER_WIDTH = 50
    SAMPLE_RATE = 16000  # PvRecorder's fixed sample rate
    AMPLIFICATION_FACTOR = 5.0  # Increase this to make playback louder

    print("Starting microphone test with PvRecorder...")

    # Set ALSA environment variables for Raspiaudio
    os.environ['AUDIODEV'] = 'hw:1,0'  # Card 1, device 0
    os.environ['ALSA_CARD'] = 'sndrpigooglevoi'

    # Try to set maximum playback volume
    try:
        print("\nSetting maximum playback volume...")
        # Try different mixer names that might control playback
        mixer_names = ['PCM', 'Speaker', 'Master', 'Playback']
        for mixer_name in mixer_names:
            try:
                mixer = alsaaudio.Mixer(mixer_name, cardindex=1)
                current_vol = mixer.getvolume()
                print(f"Found {mixer_name} mixer, current volume: {current_vol}")
                mixer.setvolume(100)
                print(f"Set {mixer_name} volume to maximum")
            except Exception as e:
                print(f"Could not set {mixer_name} volume: {e}")
    except Exception as e:
        print(f"Error setting playback volume: {e}")

    # First check ALSA devices
    print("\nChecking ALSA devices:")
    alsa_devices = get_alsa_devices()
    print(alsa_devices)

    # Try to get ALSA device index
    try:
        import subprocess
        result = subprocess.run(['arecord', '-L'], capture_output=True, text=True)
        alsa_list = result.stdout.split('\n')
        print("\nALSA device list:")
        print(result.stdout)
    except Exception as e:
        print(f"Error getting detailed ALSA devices: {e}")

    # Show available PvRecorder devices
    devices = PvRecorder.get_available_devices()
    print("\nAvailable PvRecorder devices:")
    for i, device in enumerate(devices):
        print(f"[{i}] {device}")

    # Try each device with ALSA environment set
    print("\nTesting devices with ALSA configuration...")
    working_index = -1
    for i, device in enumerate(devices):
        try:
            print(f"\nTrying device {i}: {device}")
            test_recorder = PvRecorder(device_index=i, frame_length=FRAME_LENGTH)
            test_recorder.start()

            # Try to read a few frames to ensure it's working
            test_frames = []
            max_test_volume = 0
            for _ in range(10):  # Test for a short period
                pcm = test_recorder.read()
                volume = np.abs(np.array(pcm)).mean()
                max_test_volume = max(max_test_volume, volume)
                test_frames.extend(pcm)

            test_recorder.stop()
            test_recorder.delete()

            print(f"Device {i} works! Max volume: {max_test_volume:.0f}")
            if max_test_volume > 0:  # Only consider devices that actually capture audio
                working_index = i
                break
        except Exception as e:
            print(f"Device {i} failed: {e}")
            continue

    # Use the working device or try default
    if working_index != -1:
        selected_index = working_index
        print(f"\nUsing working device at index {selected_index}")
    else:
        print("\nNo working device found with good audio levels, trying default...")
        selected_index = 1  # Try the second device as it's often the actual input

    print(f"\nTrying to use device index: {selected_index}")

    # Initialize recorder
    try:
        recorder = PvRecorder(
            device_index=selected_index,
            frame_length=FRAME_LENGTH
        )
        print(f"Successfully initialized recorder with device: {recorder.selected_device}")
        print(f"Sample Rate: {SAMPLE_RATE} Hz (fixed), Frame Length: {FRAME_LENGTH}")
    except Exception as e:
        print(f"Error initializing recorder with selected device: {e}")
        print("Trying with explicit ALSA device...")
        try:
            # Try using ALSA device name directly
            recorder = PvRecorder(device_index=1, frame_length=FRAME_LENGTH)
            print(f"Successfully initialized recorder with ALSA device")
        except Exception as e2:
            print(f"Error initializing with ALSA device: {e2}")
            print("Falling back to default device...")
            recorder = PvRecorder(device_index=-1, frame_length=FRAME_LENGTH)
        print(f"Using device: {recorder.selected_device}")

    try:
        # Start recording
        print("\nStarting recording...")
        recorder.start()
        print("Recording started successfully")

        frames = []
        max_volume = 0
        min_volume = float('inf')
        volume_history = []  # For moving average
        peak_hold = 0       # For peak hold indicator
        peak_hold_time = 0  # For peak hold decay

        print("\nRecording for 5 seconds (Ctrl+C to stop)...")
        print("Volume meter below (more # = louder):")
        print("P = Peak hold  ! = Current peak  | = Target level")

        for i in range(0, int(RECORD_SECONDS * (SAMPLE_RATE / FRAME_LENGTH))):
            pcm = recorder.read()
            frames.extend(pcm)

            # Calculate volume (scale adjusted for 16-bit audio)
            volume_norm = np.abs(np.array(pcm)).mean()
            volume_history.append(volume_norm)
            if len(volume_history) > 5:  # Keep last 5 samples for moving average
                volume_history.pop(0)
            avg_volume = sum(volume_history) / len(volume_history)

            # Update peak hold
            if volume_norm > peak_hold:
                peak_hold = volume_norm
                peak_hold_time = i
            elif i - peak_hold_time > 50:  # Decay peak hold after ~1 second
                peak_hold = max(peak_hold * 0.95, avg_volume)  # Gradual decay

            # Track volume range
            if volume_norm > 0:  # Only track non-zero volumes
                max_volume = max(max_volume, volume_norm)
                min_volume = min(min_volume, volume_norm)

            # Print volume meter with adjusted scale and info
            meter_pos = int((avg_volume / TARGET_VOLUME) * METER_WIDTH)
            peak_pos = int((peak_hold / TARGET_VOLUME) * METER_WIDTH)
            target_pos = METER_WIDTH // 2  # Middle of the meter

            # Build meter string
            meter = ["░"] * METER_WIDTH
            # Fill up to current volume
            for j in range(min(meter_pos, METER_WIDTH)):
                meter[j] = "█"
            # Add peak hold indicator
            if 0 <= peak_pos < METER_WIDTH:
                meter[peak_pos] = "P"
            # Add target level indicator
            if 0 <= target_pos < METER_WIDTH:
                meter[target_pos] = "|"

            # Show if we're close to clipping
            peak = "!" if volume_norm > 30000 else " "  # Adjusted for 16-bit audio

            print(f"\rVolume: [{''.join(meter)}] {volume_norm:8.0f} {peak}", end='', flush=True)

        print("\n* Recording finished")
        if max_volume > 0 and min_volume < float('inf'):
            print(f"Volume range: {min_volume:.0f} - {max_volume:.0f}")
            print(f"Dynamic range: {(max_volume/min_volume):.1f}x")
        else:
            print("No valid audio detected")

        # Convert to audio data (ensure values are within 16-bit range)
        audio_data = np.array(frames, dtype=np.float32)

        # Apply amplification
        print(f"\nApplying {AMPLIFICATION_FACTOR}x amplification...")
        audio_data = audio_data * AMPLIFICATION_FACTOR

        # Clip to prevent distortion
        audio_data = np.clip(audio_data, -32768, 32767)
        audio_data = audio_data.astype(np.int16)
        audio_bytes = audio_data.tobytes()

        # Save as WAV file
        with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
            wf.setnchannels(1)  # PvRecorder is mono
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(SAMPLE_RATE)  # PvRecorder's fixed rate
            wf.writeframes(audio_bytes)

        print(f"Saved to {WAVE_OUTPUT_FILENAME}")

        # Try to play back the recording
        try:
            print("\nTrying to play back the recording...")
            speaker = alsaaudio.PCM(
                alsaaudio.PCM_PLAYBACK,
                channels=1,  # Mono
                rate=SAMPLE_RATE,
                format=alsaaudio.PCM_FORMAT_S16_LE,  # 16-bit
                periodsize=FRAME_LENGTH,
                device='hw:1,0'
            )

            # Set PCM playback volume to maximum
            try:
                speaker.setvolume(100)
                print("Set PCM playback volume to maximum")
            except Exception as e:
                print(f"Could not set PCM volume: {e}")

            with wave.open(WAVE_OUTPUT_FILENAME, 'rb') as wf:
                data = wf.readframes(FRAME_LENGTH)
                while data:
                    speaker.write(data)
                    data = wf.readframes(FRAME_LENGTH)

            print("Playback finished")
        except Exception as e:
            print(f"Playback failed: {e}")

    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"\nError during recording: {e}")
    finally:
        try:
            recorder.stop()
            recorder.delete()
            print("Recorder cleaned up successfully")
        except Exception as e:
            print(f"Error during cleanup: {e}")


def main():
    parser = argparse.ArgumentParser(description='Cobra VAD Demo')
    parser.add_argument('--access_key', help='Picovoice access key')
    parser.add_argument('--audio_device_index', type=int, default=None,
                        help='Index of audio device to use (-1 for default)')
    parser.add_argument('--show_audio_devices', action='store_true',
                        help='Show available audio devices and exit')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Voice probability threshold (0.0 to 1.0)')
    parser.add_argument('--save_audio', action='store_true',
                        help='Save audio when voice is detected')
    parser.add_argument('--mode', choices=['continuous', 'single', 'mic'], default='continuous',
                        help='Test mode: continuous monitoring, single recording, or basic mic test')

    args = parser.parse_args()

    # Try to load environment variables first
    try:
        load_dotenv()
    except Exception as e:
        print(f"Error loading .env file: {e}")

    # Get access key with priority: command line > env var
    access_key = args.access_key or os.getenv("PICOVOICE_ACCESS_KEY")

    # Get audio device index with priority: command line > env var > auto-detect
    audio_device_index = args.audio_device_index
    if audio_device_index is None:
        try:
            env_device_index = os.getenv("AUDIO_DEVICE_INDEX")
            if env_device_index is not None:
                audio_device_index = int(env_device_index)
                print(f"Using audio device index {audio_device_index} from environment")
        except ValueError as e:
            print(f"Warning: Invalid AUDIO_DEVICE_INDEX in environment: {e}")
            audio_device_index = -1

    # Final check for access key
    if not access_key:
        print("Error: PICOVOICE_ACCESS_KEY is required. You can provide it in one of these ways:")
        print("1. Set PICOVOICE_ACCESS_KEY in your .env file")
        print("2. Set PICOVOICE_ACCESS_KEY environment variable")
        print("3. Use --access_key command line argument")
        return

    if args.mode == 'single':
        test_single_recording(access_key, threshold=args.threshold, save=args.save_audio)
    elif args.mode == 'mic':
        test_mic(device_index=audio_device_index)
    else:
        demo = CobraVADDemo(
            access_key=access_key,
            audio_device_index=audio_device_index,
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
