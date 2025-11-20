#!/home/daniel/.venvs/detect-speech/bin/python3
import sys
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
import subprocess

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

def trim_audio_based_on_speech(audio_path: str, output_path: str):
    """
    Detects speech in an audio file, trims silence from the beginning and end,
    and saves the result to a new file using ffmpeg.
    """
    # torch.hub.load can be slow, so let the user know what's happening
    #print("Loading Silero VAD model, this may take a moment...", file=sys.stderr)
    
    try:
        model = load_silero_vad()
    except Exception as e:
        print(f"Failed to load Silero VAD model: {e}", file=sys.stderr)
        sys.exit(1)

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
        print("No speech detected. Not creating an output file.", file=sys.stderr)
        return

    start_sample = speech_timestamps[0]['start']
    end_sample = speech_timestamps[-1]['end']
    
    start_time = start_sample / SAMPLING_RATE
    end_time = end_sample / SAMPLING_RATE

    print(f"Detected speech from {format_timestamp(start_time)} to {format_timestamp(end_time)}.", file=sys.stderr)
    print(f"Trimming audio and saving to {output_path}...", file=sys.stderr)

    command = [
        'ffmpeg',
        '-hide_banner', '-nostdin',
        '-i', audio_path,
        '-ss', str(start_time),
        '-to', str(end_time),
        '-c', 'copy',
        '-y', # Overwrite output file
        output_path
    ]
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Successfully created {output_path}.", file=sys.stderr)
        if result.stderr:
            print("ffmpeg output:", file=sys.stderr)
            print(result.stderr, file=sys.stderr)

    except FileNotFoundError:
        print("Error: ffmpeg is not installed or not in your PATH.", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print("Error executing ffmpeg command:", file=sys.stderr)
        print(f"Command: {' '.join(e.cmd)}", file=sys.stderr)
        print(f"Return code: {e.returncode}", file=sys.stderr)
        print("--- ffmpeg stdout ---", file=sys.stderr)
        print(e.stdout, file=sys.stderr)
        print("--- ffmpeg stderr ---", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <audio_file>", file=sys.stderr)
        sys.exit(1)

    audio_file = sys.argv[1]
    output_file = "output.opus"
    
    trim_audio_based_on_speech(audio_file, output_file)
