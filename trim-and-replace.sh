#!/bin/bash

# Check if a file path is provided as an argument
if [ -z "$1" ]; then
  echo "Usage: $0 <audio_file>"
  exit 1
fi

INPUT_FILE="$1"
OUTPUT_FILE="/tmp/trim-output.opus"

# Run the Python script
/home/daniel/programas/detect-speech/detect-speech.py "$INPUT_FILE"

# Check if the output file was created
if [ -f "$OUTPUT_FILE" ]; then
  # Overwrite the original file with the new one
  mv "$OUTPUT_FILE" "$INPUT_FILE"
  echo "Original file overwritten with trimmed audio."
else
  echo "Output file not created. Original file remains unchanged."
fi
