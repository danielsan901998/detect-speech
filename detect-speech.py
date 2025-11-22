#!/home/daniel/.venvs/detect-speech/bin/python3
import sys
from pathlib import Path
import argparse
from silero_vad import VADIterator, load_silero_vad
import subprocess
import torchaudio
from torchcodec.decoders import AudioDecoder
import torch # Added torch import for tensor operations


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
    try:
        model = load_silero_vad()
    except Exception as e:
        print(f"Failed to load Silero VAD model: {e}", file=sys.stderr)
        sys.exit(1)

    VAD_SAMPLING_RATE = 16000  # Silero VAD expects 16kHz mono audio
    VAD_WINDOW_SIZE_SAMPLES = 512 # VAD window size for processing chunks

    try:
        decoder = AudioDecoder(audio_path)
        total_duration_seconds = decoder.metadata.duration_seconds_from_header
        original_sample_rate = decoder.metadata.sample_rate
    except Exception as e:
        print(f"Failed to read audio file {audio_path}: {e}", file=sys.stderr)
        sys.exit(1)

    resampler = None
    if original_sample_rate != VAD_SAMPLING_RATE:
        resampler = torchaudio.transforms.Resample(original_sample_rate, VAD_SAMPLING_RATE)

    # Initialize final trim points in seconds
    final_start_seconds = 0.0
    final_end_seconds = total_duration_seconds

    # --- Trim start logic ---
    if trim_start:
        found_start_speech = False
        vad_iterator_start = VADIterator(model)
        chunk_processing_size_seconds = 5.0 # Process audio in 5-second chunks

        current_offset_seconds = 0.0
        while current_offset_seconds < total_duration_seconds:
            chunk_end_for_request = min(current_offset_seconds + chunk_processing_size_seconds, total_duration_seconds)
            
            # Get samples from original audio at its sample rate
            samples_in_chunk_obj = decoder.get_samples_played_in_range(current_offset_seconds, chunk_end_for_request)
            wav_chunk_original_sr = samples_in_chunk_obj.data
            
            # Ensure mono (mean across channels)
            if wav_chunk_original_sr.ndim > 1 and wav_chunk_original_sr.size(0) > 1:
                wav_chunk_original_sr = wav_chunk_original_sr.mean(dim=0, keepdim=True)
            
            # Resample to VAD_SAMPLING_RATE
            if resampler:
                wav_chunk_vad_sr = resampler(wav_chunk_original_sr)
            else:
                wav_chunk_vad_sr = wav_chunk_original_sr
            wav_chunk_vad_sr = wav_chunk_vad_sr.squeeze(0) # Remove channel dimension

            # Iterate through VAD windows in this chunk
            for i in range(0, len(wav_chunk_vad_sr), VAD_WINDOW_SIZE_SAMPLES):
                vad_window = wav_chunk_vad_sr[i : i + VAD_WINDOW_SIZE_SAMPLES]
                if len(vad_window) < VAD_WINDOW_SIZE_SAMPLES:
                    continue # Skip partial window at the very end of the file/chunk

                speech_dict = vad_iterator_start(vad_window)
                if speech_dict and "start" in speech_dict:
                    # Calculate the start time in the global timeline based on absolute sample index from VADIterator
                    final_start_seconds = speech_dict["start"] / VAD_SAMPLING_RATE
                    found_start_speech = True
                    break # Found speech, no need to process more for start
            if found_start_speech:
                break
            current_offset_seconds = chunk_end_for_request
        
        if not found_start_speech:
            print(
                "No speech detected at the start. Not creating an output file.",
                file=sys.stderr,
            )
            return

    # --- Trim end logic ---
    if trim_end:
        found_end_speech = False
        vad_iterator_end = VADIterator(model)
        chunk_processing_size_seconds = 5.0 # Process audio in 5-second chunks

        current_offset_seconds = total_duration_seconds
        while current_offset_seconds > 0:
            chunk_start_for_request = max(0.0, current_offset_seconds - chunk_processing_size_seconds)
            
            samples_in_chunk_obj = decoder.get_samples_played_in_range(chunk_start_for_request, current_offset_seconds)
            wav_chunk_original_sr = samples_in_chunk_obj.data

            if wav_chunk_original_sr.ndim > 1 and wav_chunk_original_sr.size(0) > 1:
                wav_chunk_original_sr = wav_chunk_original_sr.mean(dim=0, keepdim=True)
            
            if resampler:
                wav_chunk_vad_sr = resampler(wav_chunk_original_sr)
            else:
                wav_chunk_vad_sr = wav_chunk_original_sr
            wav_chunk_vad_sr = wav_chunk_vad_sr.squeeze(0)

            last_speech_offset_in_chunk_vad_sr = None
            is_speaking_in_chunk = False
            
            vad_iterator_end.reset_states() # Reset for each backward chunk processing
            
            for i in range(0, len(wav_chunk_vad_sr), VAD_WINDOW_SIZE_SAMPLES):
                vad_window = wav_chunk_vad_sr[i : i + VAD_WINDOW_SIZE_SAMPLES]
                if len(vad_window) < VAD_WINDOW_SIZE_SAMPLES:
                    continue

                speech_dict = vad_iterator_end(vad_window)
                if speech_dict:
                    if "start" in speech_dict:
                        is_speaking_in_chunk = True
                    if "end" in speech_dict:
                        is_speaking_in_chunk = False
                        last_speech_offset_in_chunk_vad_sr = speech_dict["end"]
            
            if is_speaking_in_chunk: # If VAD ends with speech in this chunk (no "end" event)
                last_speech_offset_in_chunk_vad_sr = len(wav_chunk_vad_sr)
            
            if last_speech_offset_in_chunk_vad_sr is not None:
                # Calculate the global end time
                final_end_seconds = chunk_start_for_request + last_speech_offset_in_chunk_vad_sr / VAD_SAMPLING_RATE
                found_end_speech = True
                break
            
            current_offset_seconds = chunk_start_for_request
        
        if not found_end_speech:
            print(
                "No speech detected at the end. Not creating an output file.",
                file=sys.stderr,
            )
            return

    # FFmpeg command construction
    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-i",
        str(audio_path),
        "-ss",
        str(final_start_seconds),
    ]

    if final_end_seconds < total_duration_seconds:  # Only add -to if end is actually trimmed
        command.extend(["-to", str(final_end_seconds)])
        print(
            f"Detected speech from {format_timestamp(final_start_seconds)} to {format_timestamp(final_end_seconds)}.",
            file=sys.stderr,
        )
    else:  # Only start is trimmed or no trimming at all
        print(f"Detected speech from {format_timestamp(final_start_seconds)}.", file=sys.stderr)

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
