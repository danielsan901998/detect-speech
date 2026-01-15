#include "whisper.h"
#include "common-whisper.h"

extern "C" {
#include <libavutil/log.h>
}

#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <algorithm>

void whisper_log_callback(ggml_log_level level, const char * text, void * user_data) {
    (void)user_data;
    static ggml_log_level last_level = GGML_LOG_LEVEL_NONE;
    if (level != GGML_LOG_LEVEL_CONT) {
        last_level = level;
    }
    if (last_level == GGML_LOG_LEVEL_ERROR || last_level == GGML_LOG_LEVEL_WARN) {
        fprintf(stderr, "%s", text);
    }
}

int main(int argc, char ** argv) {
    whisper_log_set(whisper_log_callback, nullptr);
    av_log_set_level(AV_LOG_ERROR);

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <audio_file> [options]\n", argv[0]);
        fprintf(stderr, "\nOptions:\n");
        fprintf(stderr, "  --output <file>    Output file path (default: overwrites input file)\n");
        fprintf(stderr, "  --trim-start, -s   Trim only the silence at the beginning\n");
        fprintf(stderr, "  --trim-end, -e     Trim only the silence at the end\n");
        return 1;
    }

    std::string audio_file = argv[1];
    std::string output_file = "";
    bool trim_start_requested = false;
    bool trim_end_requested = false;
    bool output_specified = false;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--output" && i + 1 < argc) {
            output_file = argv[++i];
            output_specified = true;
        } else if (arg == "--trim-start" || arg == "-s") {
            trim_start_requested = true;
        } else if (arg == "--trim-end" || arg == "-e") {
            trim_end_requested = true;
        } else if (arg == "--replace" || arg == "-i") {
            output_specified = false;
        }
    }

    bool replace_input = !output_specified;

    if (replace_input) {
        char tmp_template[] = "/tmp/detect-speech-XXXXXX.opus";
        int fd = mkstemps(tmp_template, 5);
        if (fd == -1) {
            fprintf(stderr, "Error: Failed to create temporary file.\n");
            return 1;
        }
        close(fd);
        output_file = tmp_template;
    }

    bool call_trim_start = trim_start_requested || (!trim_start_requested && !trim_end_requested);
    bool call_trim_end = trim_end_requested || (!trim_start_requested && !trim_end_requested);

    // Load audio data
    std::vector<float> pcmf32;
    std::vector<std::vector<float>> pcmf32s;
    if (!read_audio_data(audio_file, pcmf32, pcmf32s, false)) {
        fprintf(stderr, "Error: Failed to read audio data from %s\n", audio_file.c_str());
        return 1;
    }

    // Initialize VAD context
    struct whisper_vad_context_params vparams = whisper_vad_default_context_params();
    // Path to VAD model - using the one from detect-word.cpp reference
    struct whisper_vad_context * vctx = whisper_vad_init_from_file_with_params("/home/daniel/archivos/ggml-silero-v6.2.0.bin", vparams);
    if (vctx == nullptr) {
        fprintf(stderr, "Error: Failed to initialize VAD context.\n");
        return 1;
    }

    // Detect speech segments
    struct whisper_vad_params vad_params = whisper_vad_default_params();
    struct whisper_vad_segments * segments = whisper_vad_segments_from_samples(vctx, vad_params, pcmf32.data(), pcmf32.size());
    if (segments == nullptr) {
        fprintf(stderr, "Error: Failed to detect speech segments.\n");
        whisper_vad_free(vctx);
        return 1;
    }

    int n_vad_segments = whisper_vad_segments_n_segments(segments);

    if (n_vad_segments == 0) {
        fprintf(stderr, "No speech detected. Not creating an output file.\n");
        whisper_vad_free_segments(segments);
        whisper_vad_free(vctx);
        return 0;
    }

    float final_start_seconds = 0.0f;
    float total_duration_seconds = (float)pcmf32.size() / 16000.0f;
    float final_end_seconds = total_duration_seconds;

    if (call_trim_start) {
        // whisper_vad_segments_get_segment_t0 returns time in centiseconds (10ms units)
        final_start_seconds = whisper_vad_segments_get_segment_t0(segments, 0) * 0.01f;
        // Subtract 1 second to avoid skipping the first word
        final_start_seconds = std::max(0.0f, final_start_seconds - 0.5f);
    }

    if (call_trim_end) {
        final_end_seconds = whisper_vad_segments_get_segment_t1(segments, n_vad_segments - 1) * 0.01f;
        // Add 1 second to avoid skipping the last word
        final_end_seconds = std::min(total_duration_seconds, final_end_seconds + 0.5f);
    }

    whisper_vad_free_segments(segments);
    whisper_vad_free(vctx);

    // FFmpeg command construction
    // We use -ss before -i for faster seeking if it's a long file, 
    // but -ss after -i is more accurate for some formats when using -c copy.
    
    std::string trim_cmd = "ffmpeg -hide_banner -loglevel error -nostdin -y -i \"" + audio_file + "\" -ss " + std::to_string(final_start_seconds);
    
    if (final_end_seconds < total_duration_seconds) {
        trim_cmd += " -to " + std::to_string(final_end_seconds);
        fprintf(stderr, "Detected speech from %.3f to %.3f.\n", final_start_seconds, final_end_seconds);
    } else {
        fprintf(stderr, "Detected speech from %.3f.\n", final_start_seconds);
    }

    trim_cmd += " -c copy \"" + output_file + "\"";

    fprintf(stderr, "Trimming audio and saving to %s...\n", output_file.c_str());
    if (system(trim_cmd.c_str()) != 0) {
        fprintf(stderr, "Error: Failed to trim audio using ffmpeg.\n");
        return 1;
    }

    fprintf(stderr, "Successfully created %s.\n", output_file.c_str());

    if (replace_input) {
        if (rename(output_file.c_str(), audio_file.c_str()) != 0) {
            // Rename might fail across different filesystems, try manual copy/remove if needed
            // but for /tmp to home it might fail. Let's use a simple rename first.
            // Actually, many systems have /tmp as a separate partition.
            // If rename fails, we can try using a system command as a fallback.
            std::string mv_cmd = "mv \"" + output_file + "\" \"" + audio_file + "\"";
            if (system(mv_cmd.c_str()) != 0) {
                fprintf(stderr, "Error: Failed to replace original file %s with %s.\n", audio_file.c_str(), output_file.c_str());
                return 1;
            }
        }
        fprintf(stderr, "Original file %s has been overwritten.\n", audio_file.c_str());
    }

    return 0;
}
