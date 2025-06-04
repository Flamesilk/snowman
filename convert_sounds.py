#!/usr/bin/env python3
"""
Convert sound effects to WAV format for better compatibility with Raspberry Pi's ALSA.
This script should be run on macOS/development machine before deploying to Pi.
Automatically scans the sounds directory and converts all audio files to WAV format.
"""

import os
import subprocess
import json
from pathlib import Path

# Audio file extensions to convert
AUDIO_EXTENSIONS = {'.mp3', '.m4a', '.aac', '.ogg', '.flac'}

def find_audio_files(directory):
    """Find all audio files in the given directory and its subdirectories"""
    audio_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in AUDIO_EXTENSIONS):
                full_path = os.path.join(root, file)
                audio_files.append(full_path)
    return audio_files

def convert_to_wav(input_path, output_path):
    """Convert audio file to WAV format using ffmpeg"""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Convert to WAV using ffmpeg
        subprocess.run([
            "ffmpeg", "-y",  # -y to overwrite output files
            "-i", input_path,
            "-acodec", "pcm_s16le",  # 16-bit PCM
            "-ar", "44100",  # 44.1kHz sample rate
            "-ac", "2",      # Stereo
            output_path
        ], check=True, capture_output=True)

        print(f"✅ Converted: {input_path} -> {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Conversion failed for {input_path}")
        print(f"Error: {e.stderr.decode()}")
        return False
    except Exception as e:
        print(f"❌ Error converting {input_path}: {e}")
        return False

def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sounds_dir = os.path.join(script_dir, 'sounds')

    # Check if sounds directory exists
    if not os.path.exists(sounds_dir):
        print(f"❌ Sounds directory not found: {sounds_dir}")
        return

    print(f"🔍 Scanning directory: {sounds_dir}")

    # Find all audio files
    audio_files = find_audio_files(sounds_dir)
    if not audio_files:
        print("❌ No audio files found to convert")
        return

    print(f"📝 Found {len(audio_files)} audio files to convert")

    # New sound effects mapping
    new_sound_effects = {}

    # Convert each audio file
    for input_path in audio_files:
        # Generate output path with .wav extension
        rel_path = os.path.relpath(input_path, script_dir)
        output_path = os.path.splitext(input_path)[0] + '.wav'

        # Convert the file
        if convert_to_wav(input_path, output_path):
            # Add to mapping using the relative path
            effect_name = os.path.splitext(os.path.basename(input_path))[0]
            new_sound_effects[effect_name] = os.path.splitext(rel_path)[0] + '.wav'

    # Create updated SOUND_EFFECTS mapping
    print("\n📝 Updated SOUND_EFFECTS mapping:")
    print("SOUND_EFFECTS = {")
    for effect_name, path in sorted(new_sound_effects.items()):
        print(f'    "{effect_name}": "{path}",')
    print("}")

    # Save the new mapping to a JSON file for reference
    mapping_file = os.path.join(script_dir, 'sound_effects_wav.json')
    with open(mapping_file, 'w') as f:
        json.dump(new_sound_effects, f, indent=4)
    print(f"\n💾 Saved mapping to: {mapping_file}")

    print("\n✨ Conversion complete!")
    print(f"Converted {len(new_sound_effects)} audio files to WAV format")
    print("\nYou can now:")
    print("1. Update SOUND_EFFECTS in simple_local_assistant.py with the new mapping")
    print("2. Copy the WAV files to your Raspberry Pi")
    print("3. Delete the original audio files if no longer needed")

if __name__ == "__main__":
    main()
