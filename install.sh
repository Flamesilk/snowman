#!/bin/bash

# Exit on error and print commands as they are executed
set -e
set -x

# Default values
PI_HOST=""
PI_USER=""
PI_PORT="22"
PROJECT_DIR="~/voice-assistant"
ENV_FILE=".env.pi"

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

echo "🚀 Starting Voice Assistant Installation..."
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

echo "📂 Creating project directory on Pi..."
run_on_pi "mkdir -p $PROJECT_DIR"

echo "📦 Copying project files..."
# Create temporary directory for files
TEMP_DIR=$(mktemp -d)

# Copy required files to temp directory
cp local/simple_local_assistant.py "$TEMP_DIR/"
cp local/requirements.txt "$TEMP_DIR/"
cp local/cobra_vad.py "$TEMP_DIR/"  # Copy the Cobra VAD module
cp -r local/sounds "$TEMP_DIR/sounds"  # Copy sounds directory

# Copy .ppn files if they exist
for ppn_file in local/*.ppn; do
    if [ -f "$ppn_file" ]; then
        echo "📝 Found wake word file: $ppn_file"
        cp "$ppn_file" "$TEMP_DIR/"
    fi
done

# Copy .env file if it exists
if [ -f "$ENV_FILE" ]; then
    echo "📝 Found .env.pi file, copying it as .env..."
    cp "$ENV_FILE" "$TEMP_DIR/.env"
    USING_LOCAL_ENV=true
else
    echo "⚠️ No .env.pi file found at $ENV_FILE"
    echo "You will be prompted to enter API keys during installation"
    USING_LOCAL_ENV=false
fi

# Create remote installation script
cat > "$TEMP_DIR/remote_install.sh" << 'EOL'
#!/bin/bash
set -e
set -x

PROJECT_DIR="$HOME/voice-assistant"

# Function to wait for apt locks to be released
wait_for_apt() {
    local max_attempts=60  # Maximum number of attempts (10 minutes total)
    local attempt=1

    echo "Waiting for other package managers to finish..."
    while true; do
        if ! sudo lsof /var/lib/dpkg/lock-frontend >/dev/null 2>&1 && \
           ! sudo lsof /var/lib/dpkg/lock >/dev/null 2>&1 && \
           ! sudo lsof /var/lib/apt/lists/lock >/dev/null 2>&1; then
            break
        fi

        if [ $attempt -ge $max_attempts ]; then
            echo "Timeout waiting for package manager locks"
            echo "Attempting to identify processes holding the locks..."

            # Show processes holding the locks
            echo "Processes holding dpkg/apt locks:"
            sudo lsof /var/lib/dpkg/lock-frontend 2>/dev/null || true
            sudo lsof /var/lib/dpkg/lock 2>/dev/null || true
            sudo lsof /var/lib/apt/lists/lock 2>/dev/null || true

            echo "You may need to manually fix the issue with:"
            echo "sudo rm /var/lib/apt/lists/lock"
            echo "sudo rm /var/lib/dpkg/lock"
            echo "sudo rm /var/lib/dpkg/lock-frontend"
            exit 1
        fi

        # Show which process is holding the lock
        local lock_process
        lock_process=$(sudo lsof /var/lib/dpkg/lock-frontend 2>/dev/null | tail -n 1)
        if [ ! -z "$lock_process" ]; then
            echo "Lock is held by: $lock_process"
        fi

        echo "Waiting for package manager to be available... Attempt $attempt/$max_attempts"
        sleep 10
        attempt=$((attempt + 1))
    done
}

echo "🌍 Configuring system locale..."

# First, try to kill any stuck apt/dpkg processes
echo "Checking for stuck apt/dpkg processes..."
if pgrep -a apt >/dev/null || pgrep -a dpkg >/dev/null; then
    echo "Found running apt/dpkg processes. Attempting to finish them..."
    sudo pkill -f apt
    sudo pkill -f dpkg
    sleep 5
fi

# Remove potentially problematic lock files
echo "Cleaning up lock files..."
sudo rm -f /var/lib/apt/lists/lock
sudo rm -f /var/lib/dpkg/lock
sudo rm -f /var/lib/dpkg/lock-frontend

# Fix interrupted dpkg
echo "Fixing any interrupted dpkg configurations..."
sudo dpkg --configure -a

# Wait for any remaining package operations to complete
wait_for_apt

# Ensure package system is in a consistent state
echo "Fixing any broken installations..."
sudo apt-get install -f
wait_for_apt

# Configure and generate locales
sudo apt-get update
wait_for_apt
sudo apt-get install -y locales
wait_for_apt

sudo sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen
sudo locale-gen
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

# Add locale settings to .profile if they don't exist
if ! grep -q "export LANG=en_US.UTF-8" ~/.profile; then
    echo "export LANG=en_US.UTF-8" >> ~/.profile
fi
if ! grep -q "export LC_ALL=en_US.UTF-8" ~/.profile; then
    echo "export LC_ALL=en_US.UTF-8" >> ~/.profile
fi

echo "📦 Updating system packages..."
wait_for_apt
sudo DEBIAN_FRONTEND=noninteractive apt-get update
wait_for_apt
sudo DEBIAN_FRONTEND=noninteractive apt-get upgrade -y
wait_for_apt

echo "📥 Installing system dependencies..."
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    git \
    cmake \
    build-essential \
    libatlas-base-dev \
    wget \
    curl \
    mpg123 \
    alsa-utils \
    ffmpeg \
    libavcodec-extra \
    libportaudio2 \
    portaudio19-dev

# Configure audio
sudo usermod -a -G audio $USER

# Test audio setup
echo "🔊 Testing audio configuration..."
# List audio devices
echo "Audio playback devices:"
aplay -l || true
echo "Audio recording devices:"
arecord -l || true

# Try to set default volume
echo "Setting default audio volume..."
amixer sset 'PCM' 100% || true
amixer sset 'Master' 100% || true

echo "🐍 Setting up Python environment..."
python3 -m venv $PROJECT_DIR/venv
source $PROJECT_DIR/venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install wheel
pip install -r $PROJECT_DIR/requirements.txt

# Create necessary directories
mkdir -p $PROJECT_DIR/logs
mkdir -p $PROJECT_DIR/sounds

# Setup environment variables
if [ -f "$PROJECT_DIR/.env" ]; then
    echo "🔑 Using existing .env file..."
    chmod 600 "$PROJECT_DIR/.env"
else
    echo "🔑 Setting up environment variables..."
    read -p "Enter your Porcupine Access Key: " porcupine_key
    read -p "Enter your Google API Key: " google_key
    read -p "Enter your Tavily API Key (or press enter to skip): " tavily_key

    echo "PICOVOICE_ACCESS_KEY=$porcupine_key" > $PROJECT_DIR/.env
    echo "GOOGLE_API_KEY=$google_key" >> $PROJECT_DIR/.env
    [ ! -z "$tavily_key" ] && echo "TAVILY_API_KEY=$tavily_key" >> $PROJECT_DIR/.env
    chmod 600 $PROJECT_DIR/.env
fi

# Create and verify systemd service file
echo "🔧 Setting up systemd service..."
SERVICE_FILE="/etc/systemd/system/voice-assistant.service"

# Create systemd service file
sudo tee $SERVICE_FILE << EOFS
[Unit]
Description=Voice Assistant Service
After=network.target pulseaudio.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR
Environment=PYTHONPATH=$PROJECT_DIR
Environment=LANG=en_US.UTF-8
Environment=LC_ALL=en_US.UTF-8
EnvironmentFile=$PROJECT_DIR/.env
ExecStart=$PROJECT_DIR/venv/bin/python3 $PROJECT_DIR/simple_local_assistant.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOFS

# Verify service file exists and has correct permissions
if [ ! -f "$SERVICE_FILE" ]; then
    echo "❌ Failed to create service file!"
    exit 1
fi

echo "📄 Service file contents:"
cat "$SERVICE_FILE"

# Set correct permissions
sudo chmod 644 "$SERVICE_FILE"

# Reload systemd and enable service
echo "🔄 Reloading systemd and enabling service..."
sudo systemctl daemon-reload
sudo systemctl enable voice-assistant.service

# Verify service is enabled
if ! systemctl is-enabled voice-assistant.service >/dev/null 2>&1; then
    echo "❌ Failed to enable voice-assistant service!"
    exit 1
fi

# Start the service
echo "🚀 Starting voice assistant service..."
sudo systemctl start voice-assistant.service

# Verify service is running
if ! systemctl is-active voice-assistant.service >/dev/null 2>&1; then
    echo "⚠️ Warning: Service enabled but failed to start. Check status with:"
    echo "sudo systemctl status voice-assistant.service"
fi

echo "✅ Service file created, enabled, and started successfully"

echo "✅ Installation complete!"
echo ""
echo "To start the service:"
echo "sudo systemctl start voice-assistant.service"
echo ""
echo "To check status:"
echo "sudo systemctl status voice-assistant.service"
echo ""
echo "To view logs:"
echo "sudo journalctl -u voice-assistant.service -f"
echo ""
echo "To list audio devices:"
echo "1. List ALSA recording devices:"
echo "   arecord -l"
echo ""
echo "2. List ALSA playback devices:"
echo "   aplay -l"
echo ""
echo "3. List all PulseAudio devices:"
echo "   pactl list short sources    # Recording devices"
echo "   pactl list short sinks      # Playback devices"
echo ""
echo "4. List all audio devices (detailed):"
echo "   cat /proc/asound/cards"
echo ""
echo "To test audio:"
echo "1. Test microphone recording:"
echo "   arecord -d 5 -f cd test.wav     # Record 5 seconds"
echo "   aplay test.wav                   # Play it back"
echo ""
echo "2. Test speakers:"
echo "   speaker-test -t wav -c 2"
EOL

# Make remote install script executable
chmod +x "$TEMP_DIR/remote_install.sh"

echo "📤 Transferring files to Raspberry Pi..."
# Copy files to Pi
copy_to_pi "$TEMP_DIR/simple_local_assistant.py" "$PROJECT_DIR/"
copy_to_pi "$TEMP_DIR/requirements.txt" "$PROJECT_DIR/"
copy_to_pi "$TEMP_DIR/remote_install.sh" "$PROJECT_DIR/"
copy_to_pi "$TEMP_DIR/sounds" "$PROJECT_DIR/"  # Copy sounds directory
copy_to_pi "$TEMP_DIR/cobra_vad.py" "$PROJECT_DIR/"
# Copy any .ppn files that were found
for ppn_file in "$TEMP_DIR"/*.ppn; do
    if [ -f "$ppn_file" ]; then
        copy_to_pi "$ppn_file" "$PROJECT_DIR/"
    fi
done
if [ "$USING_LOCAL_ENV" = true ]; then
    copy_to_pi "$TEMP_DIR/.env" "$PROJECT_DIR/"
fi

echo "🔧 Running installation on Raspberry Pi..."
# Run remote installation script
run_on_pi "chmod +x $PROJECT_DIR/remote_install.sh && $PROJECT_DIR/remote_install.sh"

# Cleanup temporary directory
rm -rf "$TEMP_DIR"

echo "✅ Installation completed successfully!"
if [ "$USING_LOCAL_ENV" = true ]; then
    echo "✨ Using API keys from .env.pi file"
else
    echo "⚠️ Remember to enter your API keys when prompted"
fi
echo ""
echo "Your voice assistant has been installed on the Raspberry Pi."
echo "You can manage it using these commands:"
echo ""
echo "Start: sudo systemctl start voice-assistant.service"
echo "Stop:  sudo systemctl stop voice-assistant.service"
echo "Status: sudo systemctl status voice-assistant.service"
echo "Logs:  sudo journalctl -u voice-assistant.service -f"
