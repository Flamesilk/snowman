#!/bin/bash

# Exit on error and print commands as they are executed
set -e
set -x

# Default values
PI_HOST=""
PI_USER=""
PI_PORT="22"
PROJECT_DIR="~/voice-assistant"
ENV_FILE="local/.env"

# Print usage
usage() {
    echo "Usage: $0 -h <pi_hostname/ip> -u <username> [-p <port>]"
    echo "Example: $0 -h 192.168.86.76 -u pi"
    exit 1
}

# Parse command line arguments
while getopts "h:u:p:" opt; do
    case $opt in
        h) PI_HOST="$OPTARG" ;;
        u) PI_USER="$OPTARG" ;;
        p) PI_PORT="$OPTARG" ;;
        *) usage ;;
    esac
done

# Check required parameters
if [ -z "$PI_HOST" ] || [ -z "$PI_USER" ]; then
    usage
fi

echo "🚀 Starting Voice Assistant Redeployment..."
echo "Host: $PI_HOST"
echo "User: $PI_USER"
echo "Port: $PI_PORT"

# Function to run command on Pi
run_on_pi() {
    ssh -p "$PI_PORT" "$PI_USER@$PI_HOST" "$1"
}

# Function to copy file to Pi
copy_to_pi() {
    local src="$1"
    local dest="$2"
    if [ -d "$src" ]; then
        # If source is a directory, use -r flag
        scp -P "$PI_PORT" -r "$src" "$PI_USER@$PI_HOST:$dest"
    else
        # If source is a file, copy normally
        scp -P "$PI_PORT" "$src" "$PI_USER@$PI_HOST:$dest"
    fi
}

echo "📂 Creating project directory on Pi (if it doesn't exist)..."
run_on_pi "mkdir -p $PROJECT_DIR"
run_on_pi "mkdir -p $PROJECT_DIR/sounds"

echo "🔊 Setting up ALSA configuration..."
run_on_pi "sudo usermod -a -G audio $PI_USER"
run_on_pi "sudo usermod -a -G pulse-access $PI_USER"

# Create ALSA configuration
run_on_pi "sudo tee /etc/asound.conf << 'EOL'
pcm.!default {
    type asym
    playback.pcm {
        type plug
        slave.pcm \"hw:0,0\"
    }
    capture.pcm {
        type plug
        slave.pcm \"hw:0,0\"
    }
}
EOL"

# Set audio device permissions
run_on_pi "sudo chmod 666 /dev/snd/*"

echo "📦 Copying project files..."
# Create temporary directory for files
TEMP_DIR=$(mktemp -d)
mkdir -p "$TEMP_DIR/sounds"

# Copy required files to temp directory
cp local/simple_local_assistant.py "$TEMP_DIR/"
cp local/requirements.txt "$TEMP_DIR/"
cp local/cobra_vad.py "$TEMP_DIR/"  # Copy the Cobra VAD module
cp local/test_vad.py "$TEMP_DIR/"  # Copy VAD test script
cp local/test_alsa.py "$TEMP_DIR/"  # Copy ALSA test script

# Copy only WAV files from sounds directory
echo "🔊 Copying WAV files from sounds directory..."
find local/sounds -name "*.wav" -exec cp {} "$TEMP_DIR/sounds/" \;
wav_count=$(find "$TEMP_DIR/sounds" -name "*.wav" | wc -l)
echo "📝 Found and copied $wav_count WAV files"

# Copy Raspberry Pi specific wake word file
echo "🔍 Looking for Raspberry Pi wake word file..."
raspberry_ppn=$(find local -type f -name "*raspberry-pi*.ppn" | head -n 1)
if [ -n "$raspberry_ppn" ]; then
    echo "📝 Found Raspberry Pi wake word file: $(basename "$raspberry_ppn")"
    cp "$raspberry_ppn" "$TEMP_DIR/"
    copy_to_pi "$TEMP_DIR/$(basename "$raspberry_ppn")" "$PROJECT_DIR/"
else
    echo "⚠️ No Raspberry Pi specific wake word file found"
fi

# Check if .env exists on Pi before copying
if [ -f "$ENV_FILE" ]; then
    if run_on_pi "[ -f $PROJECT_DIR/.env ]"; then
        echo "⚠️ .env file already exists on Pi, skipping..."
    else
        echo "📝 No .env file found on Pi, copying local .env..."
        cp "$ENV_FILE" "$TEMP_DIR/.env"
    fi
fi

echo "📤 Transferring files to Raspberry Pi..."
# Copy files to Pi
copy_to_pi "$TEMP_DIR/simple_local_assistant.py" "$PROJECT_DIR/"
copy_to_pi "$TEMP_DIR/requirements.txt" "$PROJECT_DIR/"
copy_to_pi "$TEMP_DIR/cobra_vad.py" "$PROJECT_DIR/"
copy_to_pi "$TEMP_DIR/test_vad.py" "$PROJECT_DIR/"  # Copy VAD test script
copy_to_pi "$TEMP_DIR/test_alsa.py" "$PROJECT_DIR/"  # Copy ALSA test script
copy_to_pi "$TEMP_DIR/sounds/" "$PROJECT_DIR/"  # Copy only WAV files

# Copy .env file only if it exists in temp directory (meaning it was needed)
if [ -f "$TEMP_DIR/.env" ]; then
    copy_to_pi "$TEMP_DIR/.env" "$PROJECT_DIR/"
fi

# Copy service file
echo "🔧 Copying service file..."
copy_to_pi "local/voice-assistant.service" "/etc/systemd/system/"

# Set permissions and reload systemd
echo "🔄 Setting up service..."
run_on_pi "sudo chmod 644 /etc/systemd/system/voice-assistant.service && sudo systemctl daemon-reload && sudo systemctl enable voice-assistant.service"

# Cleanup temporary directory
rm -rf "$TEMP_DIR"

echo "✅ Redeployment completed successfully!"
echo ""
echo "To restart the voice assistant service:"
echo "sudo systemctl restart voice-assistant.service"
echo ""
echo "To check status:"
echo "sudo systemctl status voice-assistant.service"
echo ""
echo "To view logs:"
echo "sudo journalctl -u voice-assistant.service -f"
