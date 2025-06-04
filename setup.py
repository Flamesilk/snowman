#!/usr/bin/env python3
"""
Setup script for the Voice Assistant

This script helps users set up the voice assistant by:
1. Creating a virtual environment
2. Installing required dependencies
3. Creating a .env file from the template
4. Providing instructions for obtaining API keys
"""

import os
import sys
import subprocess
import shutil
import venv
from pathlib import Path
import platform

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f" {text} ".center(80, "="))
    print("=" * 80 + "\n")

def print_step(step_num, text):
    """Print a formatted step."""
    print(f"\n[Step {step_num}] {text}")

def create_virtual_environment():
    """Create a virtual environment if it doesn't exist."""
    if not os.path.exists("venv"):
        print("Creating virtual environment...")
        subprocess.check_call([sys.executable, "-m", "venv", "venv"])
    else:
        print("Virtual environment already exists.")

def get_python_executable():
    """Get the path to the Python executable in the virtual environment."""
    if platform.system() == "Windows":
        python_executable = os.path.join("venv", "Scripts", "python.exe")
    else:
        python_executable = os.path.join("venv", "bin", "python")

    if not os.path.exists(python_executable):
        print(f"Error: Python executable not found at {python_executable}")
        print("The virtual environment may not have been created correctly.")
        sys.exit(1)

    return python_executable

def check_ffmpeg():
    """Check if ffmpeg is installed and install it if necessary."""
    print("Checking for ffmpeg...")
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        print("✅ ffmpeg is already installed")
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        print("ffmpeg not found. Attempting to install...")
        try:
            if platform.system() == "Darwin":  # macOS
                subprocess.check_call(["brew", "install", "ffmpeg"])
            elif platform.system() == "Linux":
                # Try apt-get first (Debian/Ubuntu)
                try:
                    subprocess.check_call(["sudo", "apt-get", "update"])
                    subprocess.check_call(["sudo", "apt-get", "install", "-y", "ffmpeg"])
                except:
                    # Try yum (RHEL/CentOS)
                    try:
                        subprocess.check_call(["sudo", "yum", "install", "-y", "ffmpeg"])
                    except:
                        print("❌ Could not install ffmpeg automatically.")
                        print("Please install ffmpeg manually:")
                        print("  - On Ubuntu/Debian: sudo apt-get install ffmpeg")
                        print("  - On RHEL/CentOS: sudo yum install ffmpeg")
                        return False
            elif platform.system() == "Windows":
                print("❌ Please install ffmpeg manually on Windows:")
                print("1. Download from https://www.gyan.dev/ffmpeg/builds/")
                print("2. Extract the archive")
                print("3. Add the bin folder to your system PATH")
                return False
            print("✅ ffmpeg installed successfully")
            return True
        except Exception as e:
            print(f"❌ Error installing ffmpeg: {e}")
            print("Please install ffmpeg manually:")
            print("  - macOS: brew install ffmpeg")
            print("  - Linux: sudo apt-get install ffmpeg")
            print("  - Windows: https://www.gyan.dev/ffmpeg/builds/")
            return False

def install_dependencies(python_executable):
    """Install dependencies using the virtual environment's pip."""
    print("Installing dependencies...")

    # Check for ffmpeg first
    if not check_ffmpeg():
        print("⚠️ Warning: ffmpeg is required for audio processing")
        if not input("Continue with setup anyway? (y/n): ").lower().startswith('y'):
            sys.exit(1)

    # Install PyAudio with specific instructions for different platforms
    if platform.system() == "Darwin":  # macOS
        print("Installing portaudio for macOS...")
        try:
            subprocess.check_call(["brew", "install", "portaudio"])
        except (subprocess.SubprocessError, FileNotFoundError):
            print("Warning: Failed to install portaudio with Homebrew.")
            print("If you encounter issues with PyAudio, please install portaudio manually:")
            print("  brew install portaudio")

    # Install basic requirements
    subprocess.check_call([python_executable, "-m", "pip", "install", "--upgrade", "pip"])
    print("Installing requirements from requirements_local.txt...")
    subprocess.check_call([python_executable, "-m", "pip", "install", "-r", "requirements_local.txt"])

    # Initialize faster-whisper model for all languages
    print("Downloading Whisper base model...")
    subprocess.check_call([
        python_executable, "-c",
        "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8', cpu_threads=4, num_workers=1)"
    ])

def create_env_file():
    """Create a .env file from the template."""
    print_step(3, "Creating .env file from template")

    env_template_path = Path(".env.template")
    env_path = Path(".env")

    if not env_template_path.exists():
        print(f"❌ Template file {env_template_path} not found")
        return False

    if env_path.exists():
        overwrite = input("A .env file already exists. Overwrite? (y/n): ").lower() == 'y'
        if not overwrite:
            print("Skipping .env file creation")
            return True

    try:
        shutil.copy(env_template_path, env_path)
        print(f"✅ Created {env_path}")
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

    return True

def provide_api_instructions():
    """Provide instructions for obtaining API keys."""
    print_step(4, "API Key Instructions")

    print("""
You need to obtain API keys for the required services:

1. Daily.co API Key and Room URL (Required):
   - Sign up at https://dashboard.daily.co/
   - Create a new room and copy the room URL
   - Get your API key from the dashboard

2. Google API Key for Gemini (Required):
   - Sign up at https://makersuite.google.com/
   - Create an API key in your account settings

3. Picovoice Porcupine Access Key (Required for wake word detection):
   - Sign up at https://console.picovoice.ai/
   - Create a new access key

Note: By default, the assistant uses Silero for both speech recognition and
text-to-speech, which runs locally and doesn't require API keys.

Optional cloud-based alternatives (not required):
   - Deepgram for speech recognition: https://console.deepgram.com/
   - Cartesia for text-to-speech: https://cartesia.ai/
    """)

def main():
    """Main function to run the setup script."""
    print_header("Voice Assistant Setup")

    # Check if we're in the right directory
    if not Path("requirements_local.txt").exists():
        print("❌ This script must be run from the app directory containing requirements_local.txt")
        return

    # Create virtual environment
    create_virtual_environment()

    # Get Python executable
    python_executable = get_python_executable()

    # Install dependencies
    install_dependencies(python_executable)

    # Create .env file
    if not create_env_file():
        return

    # Provide API instructions
    provide_api_instructions()

    # Final instructions
    print_header("Setup Complete")
    print("""
To run the voice assistant:

1. Activate the virtual environment:
   - On Windows: venv\\Scripts\\activate
   - On macOS/Linux: source venv/bin/activate

   You'll know the environment is activated when you see (venv) at the beginning of your command prompt.

2. Edit the .env file with your API keys for:
   - Daily.co (DAILY_API_KEY and DAILY_ROOM_URL)
   - Google Gemini (GOOGLE_API_KEY)
   - Picovoice Porcupine (PICOVOICE_ACCESS_KEY)

3. Choose one of the following scripts to run:

   - Simple assistant (no wake word):
     python simple_assistant.py

   - Full assistant with wake word detection:
     python wake_word_assistant.py

Note: The assistant uses Silero for speech recognition and text-to-speech by default,
which runs locally and doesn't require API keys.

For more information, see the README.md file.
    """)

if __name__ == "__main__":
    main()
