#!/bin/bash

# Exit on error
set -e

echo "Installing Voice Assistant dependencies..."

# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install system dependencies
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    portaudio19-dev \
    mpg123 \
    git

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create logs directory
mkdir -p logs

# Copy service file to systemd
sudo cp voice-assistant.service /etc/systemd/system/
echo "Please edit /etc/systemd/system/voice-assistant.service to set your API keys"

# Reload systemd and enable service
sudo systemctl daemon-reload
sudo systemctl enable voice-assistant.service

echo "Installation complete!"
echo "Please edit the service file to add your API keys:"
echo "sudo nano /etc/systemd/system/voice-assistant.service"
echo ""
echo "Then start the service with:"
echo "sudo systemctl start voice-assistant.service"
echo ""
echo "To check status:"
echo "sudo systemctl status voice-assistant.service"
echo ""
echo "To view logs:"
echo "journalctl -u voice-assistant.service -f"
