#!/usr/bin/env python3
"""
Simple ALSA-based microphone test for Raspiaudio MIC+ V3
"""

import alsaaudio
import numpy as np
import wave
import time
import subprocess

def set_alsa_volume():
    """Try different methods to set the volume"""
    try:
        # Try using amixer to set capture volume
        print("\nTrying to set capture volume...")

        # List all controls
        result = subprocess.run(['amixer', '-c', '1', 'contents'],
                              capture_output=True, text=True)
        print("Available controls and settings:")
        print(result.stdout)

        # Try different amixer commands known to work with Raspiaudio
        commands = [
            ['amixer', '-c', '1', 'set', 'Mic', '100%'],
            ['amixer', '-c', '1', 'set', 'Capture', '100%'],
            ['amixer', '-c', '1', 'set', 'Input', '100%'],
            ['amixer', '-c', '1', 'set', 'ADC', '100%'],
            ['amixer', '-c', '1', 'set', 'Digital', '100%'],
            # Try setting specific dB values
            ['amixer', '-c', '1', 'set', 'Capture', '--', '30dB'],
            ['amixer', '-c', '1', 'set', 'Mic', '--', '30dB']
        ]

        for cmd in commands:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                print(f"Tried command '{' '.join(cmd)}': {'success' if result.returncode == 0 else 'failed'}")
            except Exception as e:
                print(f"Command failed: {' '.join(cmd)}")
                continue

    except Exception as e:
        print(f"Error setting volume: {e}")

def test_mic():
    # ALSA parameters for Raspiaudio MIC+ V3
    CHANNELS = 2
    RATE = 48000
    FORMAT = alsaaudio.PCM_FORMAT_S32_LE
    PERIOD_SIZE = 1024
    RECORD_SECONDS = 5
    INITIAL_GAIN = 50
    TARGET_VOLUME = 3500000  # Lowered target volume based on observed levels
    MIN_VOLUME = 100000     # Noise floor threshold
    MAX_GAIN = 80          # Reduced maximum gain to prevent noise amplification
    AGC_SPEED = 0.05       # Slower AGC for more stable gain
    METER_WIDTH = 50       # Width of the volume meter

    print("Starting ALSA microphone test...")
    print(f"Using: hw:1,0 (Raspiaudio MIC+ V3)")
    print(f"Format: {FORMAT}, Rate: {RATE}, Channels: {CHANNELS}")
    print(f"Initial Gain: {INITIAL_GAIN}, Target Volume: {TARGET_VOLUME}")
    print(f"Noise Floor: {MIN_VOLUME}, Max Gain: {MAX_GAIN}")

    try:
        # Open the PCM device for recording
        inp = alsaaudio.PCM(
            alsaaudio.PCM_CAPTURE,
            channels=CHANNELS,
            rate=RATE,
            format=FORMAT,
            periodsize=PERIOD_SIZE,
            device='hw:1,0'
        )

        print("\nRecording started...")
        print("Volume meter below (more # = louder):")
        print("P = Peak hold  ! = Current peak  | = Target level")

        frames = []
        max_volume = 0
        min_volume = float('inf')
        current_gain = INITIAL_GAIN
        volume_history = []  # For moving average
        peak_hold = 0       # For peak hold indicator
        peak_hold_time = 0  # For peak hold decay

        for i in range(0, int(RATE / PERIOD_SIZE * RECORD_SECONDS)):
            length, data = inp.read()
            if length > 0:
                # Convert data to numpy array
                audio_data = np.frombuffer(data, dtype=np.int32)

                # Calculate current volume before gain
                raw_volume = np.linalg.norm(audio_data) / len(audio_data)

                # Update gain using AGC only if above noise floor
                if raw_volume > MIN_VOLUME:
                    volume_ratio = TARGET_VOLUME / raw_volume
                    target_gain = current_gain * volume_ratio
                    # Limit maximum gain
                    target_gain = min(target_gain, MAX_GAIN)
                    # Smooth gain changes
                    current_gain = current_gain * (1 - AGC_SPEED) + target_gain * AGC_SPEED

                # Apply gain with clipping protection
                audio_data = np.clip(audio_data * current_gain, -2**31, 2**31 - 1).astype(np.int32)
                frames.append(audio_data.tobytes())

                # Calculate and show volume
                volume_norm = np.linalg.norm(audio_data) / len(audio_data)
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

                max_volume = max(max_volume, volume_norm)
                min_volume = min(min_volume, volume_norm) if volume_norm > 0 else min_volume

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
                peak = "!" if volume_norm > 2**30 else " "

                print(f"\rVolume: [{''.join(meter)}] {volume_norm:8.0f} Gain: {current_gain:3.1f}x {peak}",
                      end='', flush=True)

        print(f"\nRecording finished")
        print(f"Volume range: {min_volume:.0f} - {max_volume:.0f}")
        print(f"Dynamic range: {(max_volume/min_volume):.1f}x")
        print(f"Final gain: {current_gain:.1f}x")

        # Save as WAV file
        with wave.open('test_alsa.wav', 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(4)  # 32-bit audio
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))

        print("Saved to test_alsa.wav")

        # Try to play back the recording
        try:
            print("\nTrying to play back the recording...")
            speaker = alsaaudio.PCM(
                alsaaudio.PCM_PLAYBACK,
                channels=CHANNELS,
                rate=RATE,
                format=FORMAT,
                periodsize=PERIOD_SIZE,
                device='hw:1,0'
            )

            with wave.open('test_alsa.wav', 'rb') as wf:
                data = wf.readframes(PERIOD_SIZE)
                while data:
                    speaker.write(data)
                    data = wf.readframes(PERIOD_SIZE)

            print("Playback finished")
        except Exception as e:
            print(f"Playback failed: {e}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_mic()
