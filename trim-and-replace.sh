#!/bin/bash

TRIM_START_ONLY_FLAG=""
TRIM_END_ONLY_FLAG=""
INPUT_FILE=""

for arg in "$@"; do
  if [ "$arg" == "--trim-start-only" ] || [ "$arg" == "-s" ]; then
    TRIM_START_ONLY_FLAG="--trim-start-only"
  elif [ "$arg" == "--trim-end-only" ] || [ "$arg" == "-e" ]; then
    TRIM_END_ONLY_FLAG="--trim-end-only"
  else
    INPUT_FILE="$arg"
  fi
done

# Check if a file path is provided as an argument
if [ -z "$INPUT_FILE" ]; then
  echo "Usage: $0 [-s|--trim-start-only] [-e|--trim-end-only] <audio_file>"
  exit 1
fi

OUTPUT_FILE=$(mktemp --suffix=".opus")

# Trap to ensure the temporary file is removed on exit
trap "rm -f \"$OUTPUT_FILE\"" EXIT

# Run the Python script with the conditional flag
/home/daniel/programas/detect-speech/detect-speech.py "$INPUT_FILE" $TRIM_START_ONLY_FLAG $TRIM_START_ONLY_FLAG --output "$OUTPUT_FILE"

# Check if the output file was created
if [ -f "$OUTPUT_FILE" ]; then
  # Overwrite the original file with the new one
  mv "$OUTPUT_FILE" "$INPUT_FILE"
  echo "Original file overwritten with trimmed audio."
else
  echo "Output file not created. Original file remains unchanged."
fi
