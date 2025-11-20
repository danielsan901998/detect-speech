#!/home/daniel/.venvs/detect-speech/bin/python3
import sys
import torch
import torchaudio

def format_timestamp(
    seconds: float, always_include_hours: bool = False, decimal_marker: str = "."
):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return (
        f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
    )

def find_speech_timestamps(audio_path: str, min_duration_s: float):
    """
    Detects non-speech segments longer than 0.4 seconds in an audio file and prints their timestamps.
    """
    # torch.hub.load can be slow, so let the user know what's happening
    #print("Loading Silero VAD model, this may take a moment...", file=sys.stderr)
    
    # Using force_reload=True as recommended by Silero VAD repo to get latest version
    try:
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=False,
                                  verbose=False,
                                  onnx=False)
    except Exception as e:
        print(f"Failed to load Silero VAD model: {e}", file=sys.stderr)
        print("Please check your internet connection and torch/torchaudio installation.", file=sys.stderr)
        sys.exit(1)


    (get_speech_timestamps,
     _, # save_audio
     read_audio,
     *_) = utils

    # Silero VAD expects 16kHz mono audio
    SAMPLING_RATE = 16000
    
    try:
        # read_audio function from utils resamples and converts to mono
        wav = read_audio(audio_path, sampling_rate=SAMPLING_RATE)
    except Exception as e:
        print(f"Failed to read audio file {audio_path}: {e}", file=sys.stderr)
        sys.exit(1)
    
    # get_speech_timestamps returns a list of dicts with 'start' and 'end' samples
    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=SAMPLING_RATE)

    total_samples = len(wav)
    min_duration_samples = SAMPLING_RATE * min_duration_s

    non_speech_segments = []
    
    # Start from the beginning of the audio
    current_pos = 0

    if speech_timestamps:
        for segment in speech_timestamps:
            start_speech = segment['start']
            end_speech = segment['end']
            
            # The gap between current_pos and start_speech is a non-speech segment
            if start_speech - current_pos > min_duration_samples:
                non_speech_segments.append({'start': current_pos, 'end': start_speech})
            
            current_pos = end_speech
    
    # The rest of the audio from the last speech end to total_samples
    if total_samples - current_pos > min_duration_samples:
        non_speech_segments.append({'start': current_pos, 'end': total_samples})

    if not non_speech_segments:
        print(f"No non-speech segments longer than {min_duration_s}s found.", file=sys.stderr)
        return

    print(f"Non-speech segments longer than {min_duration_s}s found:")
    for i, segment in enumerate(non_speech_segments):
        start_time = segment['start'] / SAMPLING_RATE
        end_time = segment['end'] / SAMPLING_RATE
        print(f"  {i+1}: [{format_timestamp(start_time)}] --> [{format_timestamp(end_time)}]")

if __name__ == "__main__":
    min_non_speech_duration_s = 1.0 # Default value

    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <audio_file> [min_non_speech_duration_s]", file=sys.stderr)
        print(f"  min_non_speech_duration_s (optional): Minimum duration in seconds for non-speech segments to be reported. Default is {min_non_speech_duration_s}s.", file=sys.stderr)
        sys.exit(1)

    audio_file = sys.argv[1]
    
    if len(sys.argv) >= 3:
        try:
            min_non_speech_duration_s = float(sys.argv[2])
            if min_non_speech_duration_s < 0:
                raise ValueError("Duration cannot be negative.")
        except ValueError:
            print(f"Error: Invalid duration '{sys.argv[2]}'. Please provide a positive number for min_non_speech_duration_s.", file=sys.stderr)
            sys.exit(1)

    find_speech_timestamps(audio_file, min_non_speech_duration_s)
