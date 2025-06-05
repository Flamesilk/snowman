#!/usr/bin/env python3
"""
Raspberry Pi Audio Volume Test Script

This script tests different methods to control audio volume on Raspberry Pi.
Use this to determine which method works best for your specific setup.
"""

import subprocess
import sys
import time
import os


def run_command(cmd, timeout=5):
    """Run a command and return success status and output"""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)


def list_audio_devices():
    """List available audio devices"""
    print("🔍 Listing available audio devices:")
    print("\n1. ALSA Playback Devices:")
    success, stdout, stderr = run_command(["aplay", "-l"])
    if success:
        print(stdout)
    else:
        print(f"❌ Failed to list ALSA devices: {stderr}")

    print("\n2. ALSA Cards:")
    success, stdout, stderr = run_command(["cat", "/proc/asound/cards"])
    if success:
        print(stdout)
    else:
        print(f"❌ Failed to list sound cards: {stderr}")

    print("\n3. PulseAudio Sinks (if available):")
    success, stdout, stderr = run_command(["pactl", "list", "short", "sinks"])
    if success:
        print(stdout)
    else:
        print(f"❌ PulseAudio not available or failed: {stderr}")


def list_mixer_controls(card_index=None):
    """List available mixer controls for a card"""
    print(f"\n🎛️ Listing mixer controls for card {card_index if card_index is not None else 'default'}:")

    if card_index is not None:
        cmd = ["amixer", "-c", str(card_index), "scontrols"]
    else:
        cmd = ["amixer", "scontrols"]

    success, stdout, stderr = run_command(cmd)
    if success:
        print(stdout)
        return stdout
    else:
        print(f"❌ Failed to list mixer controls: {stderr}")
        return ""


def test_volume_method_alsa(card_index, control, volume):
    """Test ALSA volume control method"""
    print(f"\n🧪 Testing ALSA: Card {card_index}, Control '{control}', Volume {volume}%")

    if card_index is not None:
        cmd = ["amixer", "-c", str(card_index), "sset", control, f"{volume}%"]
    else:
        cmd = ["amixer", "sset", control, f"{volume}%"]

    success, stdout, stderr = run_command(cmd)
    if success:
        print(f"✅ Success! Volume set to {volume}% using {control}")
        print(f"Output: {stdout.strip()}")
        return True
    else:
        print(f"❌ Failed: {stderr}")
        return False


def test_volume_method_pulseaudio(volume):
    """Test PulseAudio volume control method"""
    print(f"\n🧪 Testing PulseAudio: Volume {volume}%")

    cmd = ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{volume}%"]
    success, stdout, stderr = run_command(cmd)
    if success:
        print(f"✅ Success! Volume set to {volume}% using PulseAudio")
        return True
    else:
        print(f"❌ Failed: {stderr}")
        return False


def get_current_volume():
    """Get current volume levels"""
    print("\n📊 Current Volume Levels:")

    # Try to get ALSA mixer info
    controls = ["PCM", "Master", "Speaker", "Headphone", "Playbook"]
    for control in controls:
        cmd = ["amixer", "sget", control]
        success, stdout, stderr = run_command(cmd)
        if success and "%" in stdout:
            print(f"{control}: {stdout}")
        elif success:
            print(f"{control}: Available but no volume info")

    # Try PulseAudio
    cmd = ["pactl", "get-sink-volume", "@DEFAULT_SINK@"]
    success, stdout, stderr = run_command(cmd)
    if success:
        print(f"PulseAudio Default Sink: {stdout.strip()}")


def play_test_sound():
    """Play a test sound if available"""
    print("\n🔊 Testing audio playback...")

    # Try to play a simple beep or tone
    test_methods = [
        ["speaker-test", "-t", "sine", "-f", "1000", "-l", "1"],
        ["aplay", "/usr/share/sounds/alsa/Front_Left.wav"],
        ["paplay", "/usr/share/sounds/alsa/Front_Left.wav"],
    ]

    for method in test_methods:
        print(f"Trying: {' '.join(method)}")
        success, stdout, stderr = run_command(method, timeout=10)
        if success:
            print("✅ Audio test successful!")
            return True
        else:
            print(f"❌ Failed: {stderr}")

    print("⚠️ No test audio method worked")
    return False


def interactive_volume_test():
    """Interactive volume testing"""
    print("\n🎮 Interactive Volume Testing")
    print("This will test different volume levels and methods.")

    # Get available cards
    success, stdout, stderr = run_command(["aplay", "-l"])
    cards = []
    if success:
        for line in stdout.split('\n'):
            if line.startswith('card '):
                card_num = line.split(':')[0].split(' ')[1]
                cards.append(int(card_num))

    if not cards:
        cards = [0]  # Default to card 0

    print(f"Available cards: {cards}")

    # Test different methods
    working_methods = []

    for card in cards:
        print(f"\n--- Testing Card {card} ---")

        # Get mixer controls for this card
        controls_output = list_mixer_controls(card)
        available_controls = []
        for line in controls_output.split('\n'):
            if "'" in line:
                control = line.split("'")[1]
                available_controls.append(control)

        print(f"Available controls: {available_controls}")

        # Test common controls
        test_controls = ["PCM", "Master", "Speaker", "Headphone", "Playback"]
        for control in test_controls:
            if control in available_controls:
                # Test volume levels 30%, 70%, 50%
                for volume in [30, 70, 50]:
                    success = test_volume_method_alsa(card, control, volume)
                    if success:
                        working_methods.append(f"Card {card}, Control {control}")
                        time.sleep(1)  # Brief pause between tests
                        break

    # Test PulseAudio
    for volume in [30, 70, 50]:
        success = test_volume_method_pulseaudio(volume)
        if success:
            working_methods.append("PulseAudio")
            break

    print(f"\n✅ Working volume control methods:")
    for method in working_methods:
        print(f"  - {method}")

    return working_methods


def main():
    print("🎵 Raspberry Pi Volume Control Test Script")
    print("=" * 50)

    if not sys.platform.startswith('linux'):
        print("❌ This script is designed for Linux/Raspberry Pi")
        sys.exit(1)

    while True:
        print("\nChoose an option:")
        print("1. List audio devices")
        print("2. List mixer controls")
        print("3. Get current volume")
        print("4. Test specific volume setting")
        print("5. Play test sound")
        print("6. Run interactive volume test")
        print("7. Exit")

        choice = input("\nEnter your choice (1-7): ").strip()

        if choice == "1":
            list_audio_devices()

        elif choice == "2":
            card_input = input("Enter card index (or press Enter for default): ").strip()
            card_index = int(card_input) if card_input.isdigit() else None
            list_mixer_controls(card_index)

        elif choice == "3":
            get_current_volume()

        elif choice == "4":
            print("\nManual volume test:")
            card_input = input("Enter card index (or press Enter for default): ").strip()
            card_index = int(card_input) if card_input.isdigit() else None

            control = input("Enter control name (e.g., PCM, Master): ").strip()
            volume_input = input("Enter volume percentage (0-100): ").strip()

            try:
                volume = int(volume_input)
                if 0 <= volume <= 100:
                    test_volume_method_alsa(card_index, control, volume)
                else:
                    print("❌ Volume must be between 0 and 100")
            except ValueError:
                print("❌ Invalid volume value")

        elif choice == "5":
            play_test_sound()

        elif choice == "6":
            working_methods = interactive_volume_test()
            if working_methods:
                print(f"\n🎉 Found {len(working_methods)} working method(s)!")
                print("You can use any of these methods in your voice assistant.")
            else:
                print("\n😞 No working volume control methods found.")
                print("Check your audio configuration and try different devices.")

        elif choice == "7":
            print("👋 Goodbye!")
            break

        else:
            print("❌ Invalid choice. Please enter 1-7.")


if __name__ == "__main__":
    main()
