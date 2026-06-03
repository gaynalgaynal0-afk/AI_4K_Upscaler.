#!/usr/bin/env bash
set -e

# Install ffmpeg if not present
if ! command -v ffmpeg &> /dev/null; then
    apt-get update -qq && apt-get install -y ffmpeg
fi

# Install Python deps
pip install -r requirements.txt --quiet

# Run bot
python bot.py
