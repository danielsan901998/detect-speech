#!/home/daniel/.venvs/detect-speech/bin/python3
import sys
from pathlib import Path
import argparse
from silero_vad import VADIterator, load_silero_vad, read_audio
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


def trim_audio_based_on_speech(
    audio_path: str,
    output_path: str,
    trim_start: bool = True,
    trim_end: bool = True,
):
    """
    Detects speech in an audio file, trims silence from the beginning and end,
    and saves the result to a new file using ffmpeg.
    """
    # torch.hub.load can be slow, so let the user know what's happening
    # print("Loading Silero VAD model, this may take a moment...", file=sys.stderr)

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

    vad_iterator = VADIterator(model)

    window_size_samples = 512
    final_start_sample = 0
    final_end_sample = len(wav)

    if trim_start:  # trim start
        found_start_speech = False
        for i in range(0, len(wav), window_size_samples):
            chunk = wav[i : i + window_size_samples]
            if len(chunk) < window_size_samples:
                break
            speech_dict = vad_iterator(chunk)
            if speech_dict and "start" in speech_dict:
                final_start_sample = max(0,speech_dict["start"]-window_size_samples)
                found_start_speech = True
                break
        if not found_start_speech:
            print(
                "No speech detected at the start. Not creating an output file.",
                file=sys.stderr,
            )
            return
    if trim_end:  # trim end
        # Iterate backwards from the end of the file in 1s chunks
        # usage: range(start, stop, step)
        seek_step_samples = int(1 * SAMPLING_RATE)
        audio_len = len(wav)
        found_end_speech = False
        for chunk_start in range(
            max(0, audio_len - seek_step_samples),
            -seek_step_samples,
            -seek_step_samples,
        ):
            # Handle the final (first in time) chunk which might be shorter
            actual_start = max(0, chunk_start)
            actual_end = min(audio_len, chunk_start + seek_step_samples)

            chunk = wav[actual_start:actual_end]
            vad_iterator.reset_states()  # Crucial: Reset for each new chunk

            last_speech_in_chunk = None
            is_speaking = False

            # Scan this chunk FORWARD (standard VAD usage)
            for i in range(0, len(chunk), window_size_samples):
                window = chunk[i : i + window_size_samples]
                if len(window) < window_size_samples:
                    break

                speech_dict = vad_iterator(window)

                # Track the latest speech timestamp found
                if speech_dict:
                    if "start" in speech_dict:
                        is_speaking = True
                    if "end" in speech_dict:
                        is_speaking = False
                        last_speech_in_chunk = i + speech_dict["end"]

            # If the chunk ends while someone is still speaking, the "end" is the chunk boundary
            if is_speaking:
                last_speech_in_chunk = len(chunk)

            # If we found speech in this chunk, we are done!
            if last_speech_in_chunk is not None:
                final_end_sample = actual_start + last_speech_in_chunk
                found_end_speech = True
                break

        if not found_end_speech:
            print(
                "No speech detected at the end. Not creating an output file.",
                file=sys.stderr,
            )
            return

    start_time = final_start_sample / SAMPLING_RATE

    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-i",
        str(audio_path),
        "-ss",
        str(start_time),
    ]

    if final_end_sample < len(wav):  # Only add -to if end is actually trimmed
        end_time = final_end_sample / SAMPLING_RATE
        command.extend(["-to", str(end_time)])
        print(
            f"Detected speech from {format_timestamp(start_time)} to {format_timestamp(end_time)}.",
            file=sys.stderr,
        )
    else:  # Only start is trimmed or no trimming at all
        print(f"Detected speech from {format_timestamp(start_time)}.", file=sys.stderr)

    command.extend(["-c", "copy", "-y", str(output_path)])

    print(f"Trimming audio and saving to {output_path}...", file=sys.stderr)

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
    parser = argparse.ArgumentParser(
        description="Detects speech in an audio file and trims silence."
    )
    parser.add_argument("audio_file", type=Path, help="Path to the input audio file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/trim-output.opus"),
        help="Path to the output audio file (default: /tmp/trim-output.opus).",
    )
    parser.add_argument(
        "--trim-start",
        action="store_true",
        help="Enable trimming silence from the beginning of the audio. By default, both start and end are trimmed. If only --trim-start is specified, trimming from the end will be disabled.",
    )
    parser.add_argument(
        "--trim-end",
        action="store_true",
        help="Enable trimming silence from the end of the audio. By default, both start and end are trimmed. If only --trim-end is specified, trimming from the start will be disabled.",
    )
    args = parser.parse_args()

    # Logic to handle default True and specific overrides
    call_trim_start = args.trim_start or (not args.trim_start and not args.trim_end)
    call_trim_end = args.trim_end or (not args.trim_start and not args.trim_end)

    trim_audio_based_on_speech(
        args.audio_file,
        args.output,
        trim_start=call_trim_start,
        trim_end=call_trim_end,
    )
