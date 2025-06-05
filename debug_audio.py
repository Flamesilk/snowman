#!/usr/bin/env python3
"""
Raspberry Pi Audio Device Debug Script

This script helps debug audio device issues with PvRecorder and other audio systems.
"""

import subprocess
import sys
import os
from pvrecorder import PvRecorder


def run_command(cmd):
    """Run a command and return output"""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)


def check_audio_permissions():
    """Check if user has audio permissions"""
    print("🔍 Checking audio permissions...")

    # Check if user is in audio group
    success, stdout, stderr = run_command(["groups"])
    if success:
        groups = stdout.strip()
        print(f"User groups: {groups}")
        if "audio" in groups:
            print("✅ User is in audio group")
        else:
            print("❌ User is NOT in audio group")
            print("Fix with: sudo usermod -a -G audio $USER")
            print("Then log out and back in")

    # Check audio device permissions
    audio_devices = ["/dev/snd/", "/dev/dsp", "/dev/audio"]
    for device in audio_devices:
        if os.path.exists(device):
            print(f"📁 {device} exists")
            success, stdout, stderr = run_command(["ls", "-la", device])
            if success:
                print(f"   Permissions: {stdout}")


def list_alsa_devices():
    """List ALSA audio devices"""
    print("\n🎤 ALSA Recording Devices:")
    success, stdout, stderr = run_command(["arecord", "-l"])
    if success:
        print(stdout)
        return stdout
    else:
        print(f"❌ Failed: {stderr}")
        return ""


def list_pvrecorder_devices():
    """List PvRecorder available devices"""
    print("\n🎙️ PvRecorder Available Devices:")
    try:
        devices = PvRecorder.get_available_devices()
        for i, device in enumerate(devices):
            print(f"  Device {i}: {device}")
        return devices
    except Exception as e:
        print(f"❌ Error getting PvRecorder devices: {e}")
        return []


def test_pvrecorder_device(device_index):
    """Test a specific PvRecorder device"""
    print(f"\n🧪 Testing PvRecorder device {device_index}...")

    recorder = None
    try:
        # Try to create and start recorder
        recorder = PvRecorder(device_index=device_index, frame_length=512)
        print(f"✅ Created recorder for device {device_index}: {recorder.selected_device}")

        # Try to start recording
        recorder.start()
        print("✅ Started recording")

        # Try to read a few frames
        for i in range(5):
            try:
                pcm = recorder.read()
                print(f"✅ Read frame {i+1}: {len(pcm)} samples")
            except Exception as e:
                print(f"❌ Error reading frame {i+1}: {e}")
                break

        return True

    except Exception as e:
        print(f"❌ Error testing device {device_index}: {e}")
        return False
    finally:
        if recorder is not None:
            try:
                recorder.stop()
                recorder.delete()
            except:
                pass


def check_audio_processes():
    """Check what processes are using audio"""
    print("\n🔍 Checking processes using audio devices...")

    success, stdout, stderr = run_command(["lsof", "/dev/snd/*"])
    if success and stdout:
        print("Processes using audio:")
        print(stdout)
    else:
        print("No processes found using audio devices")

    # Check for PulseAudio
    success, stdout, stderr = run_command(["pgrep", "-f", "pulseaudio"])
    if success and stdout:
        print("📢 PulseAudio is running")
        # Try to kill it to free up audio devices
        print("Attempting to stop PulseAudio...")
        run_command(["pulseaudio", "--kill"])
    else:
        print("📢 PulseAudio is not running")


def test_alsa_recording():
    """Test ALSA recording directly"""
    print("\n🎤 Testing ALSA recording...")

    # Try to record for 1 second
    cmd = ["arecord", "-d", "1", "-f", "cd", "/tmp/test_recording.wav"]
    print(f"Running: {' '.join(cmd)}")

    success, stdout, stderr = run_command(cmd)
    if success:
        print("✅ ALSA recording successful")
        # Check if file was created
        if os.path.exists("/tmp/test_recording.wav"):
            print("✅ Recording file created")
            os.remove("/tmp/test_recording.wav")
        return True
    else:
        print(f"❌ ALSA recording failed: {stderr}")
        return False


def suggest_fixes():
    """Suggest potential fixes"""
    print("\n🔧 Potential Fixes:")
    print("1. Add user to audio group:")
    print("   sudo usermod -a -G audio $USER")
    print("   (then log out and back in)")
    print()
    print("2. Stop conflicting audio services:")
    print("   sudo systemctl stop pulseaudio")
    print("   pulseaudio --kill")
    print()
    print("3. Check if audio device is being used:")
    print("   lsof /dev/snd/*")
    print()
    print("4. Restart audio services:")
    print("   sudo systemctl restart alsa-state")
    print("   sudo alsactl restore")
    print()
    print("5. Check audio configuration:")
    print("   cat /proc/asound/cards")
    print("   amixer scontrols")
    print()
    print("6. Set specific audio device in .env:")
    print("   AUDIO_DEVICE_INDEX=1")


def main():
    print("🎵 Raspberry Pi Audio Debug Script")
    print("=" * 50)

    # Check basic system info
    print(f"Platform: {sys.platform}")
    print(f"User: {os.getenv('USER', 'unknown')}")

    # Run all checks
    check_audio_permissions()
    alsa_output = list_alsa_devices()
    pvrecorder_devices = list_pvrecorder_devices()
    check_audio_processes()

    # Test ALSA
    alsa_works = test_alsa_recording()

    # Test PvRecorder devices
    working_devices = []
    if pvrecorder_devices:
        for i in range(len(pvrecorder_devices)):
            if test_pvrecorder_device(i):
                working_devices.append(i)

    # Summary
    print("\n📊 Summary:")
    print(f"ALSA recording works: {alsa_works}")
    print(f"Working PvRecorder devices: {working_devices}")

    if working_devices:
        print(f"\n✅ Recommended AUDIO_DEVICE_INDEX: {working_devices[0]}")
        print(f"Add this to your .env file:")
        print(f"AUDIO_DEVICE_INDEX={working_devices[0]}")
    else:
        print("\n❌ No working PvRecorder devices found")
        suggest_fixes()


if __name__ == "__main__":
    main()
