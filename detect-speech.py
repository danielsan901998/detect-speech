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

def find_speech_timestamps(audio_path: str):
    """
    Detects speech activity in an audio file and prints the timestamps of speech segments.
    """
    # torch.hub.load can be slow, so let the user know what's happening
    #print("Loading Silero VAD model, this may take a moment...", file=sys.stderr)
    
    # Using force_reload=True as recommended by Silero VAD repo to get latest version
    try:
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=False,
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

    if not speech_timestamps:
        print("No speech detected in the audio.", file=sys.stderr)
        return

    print("Speech segments found:")
    for i, segment in enumerate(speech_timestamps):
        start_time = segment['start'] / SAMPLING_RATE
        end_time = segment['end'] / SAMPLING_RATE
        print(f"  {i+1}: [{format_timestamp(start_time)}] --> [{format_timestamp(end_time)}]")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <audio_file>", file=sys.stderr)
        sys.exit(1)

    audio_file = sys.argv[1]
    find_speech_timestamps(audio_file)
