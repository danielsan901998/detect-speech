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

    std::string audio_file = "";
    std::string output_file = "";
    std::string model_path = "/home/daniel/archivos/ggml-silero-v6.2.0.bin";
    bool trim_start_requested = false;
    bool trim_end_requested = false;
    bool output_specified = false;

    for (int i = 1; i < argc; ++i) {
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
        } else if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg[0] != '-') {
            if (audio_file.empty()) {
                audio_file = arg;
            } else {
                fprintf(stderr, "Error: Multiple audio files specified: %s and %s\n", audio_file.c_str(), arg.c_str());
                return 1;
            }
        } else {
            fprintf(stderr, "Error: Unknown option %s\n", arg.c_str());
            return 1;
        }
    }

    if (audio_file.empty()) {
        fprintf(stderr, "Usage: %s <audio_file> [options]\n", argv[0]);
        fprintf(stderr, "\nOptions:\n");
        fprintf(stderr, "  --output <file>    Output file path (default: overwrites input file)\n");
        fprintf(stderr, "  --trim-start, -s   Trim only the silence at the beginning\n");
        fprintf(stderr, "  --trim-end, -e     Trim only the silence at the end\n");
        fprintf(stderr, "  --model <file>     Path to Silero VAD model\n");
        return 1;
    }

    if (const char* env_model = getenv("WHISPER_VAD_MODEL")) {
        model_path = env_model;
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
    
    struct whisper_vad_context * vctx = whisper_vad_init_from_file_with_params(model_path.c_str(), vparams);
    if (vctx == nullptr) {
        fprintf(stderr, "Error: Failed to initialize VAD context using model from %s\n", model_path.c_str());
        return 1;
    }

    // Detect speech segments
    struct whisper_vad_params vad_params = whisper_vad_default_params();

    float total_duration_seconds = (float)pcmf32.size() / (float)WHISPER_SAMPLE_RATE;
    float final_start_seconds = 0.0f;
    float final_end_seconds = total_duration_seconds;
    bool speech_detected = false;

    // Process in 30s chunks to find start and end
    const int chunk_size_samples = 30 * WHISPER_SAMPLE_RATE;

    if (call_trim_start) {
        for (int i = 0; i < (int)pcmf32.size(); i += chunk_size_samples) {
            int n_samples = std::min(chunk_size_samples, (int)pcmf32.size() - i);
            struct whisper_vad_segments * segments = whisper_vad_segments_from_samples(vctx, vad_params, pcmf32.data() + i, n_samples);
            if (segments) {
                if (whisper_vad_segments_n_segments(segments) > 0) {
                    final_start_seconds = (float)i / (float)WHISPER_SAMPLE_RATE + whisper_vad_segments_get_segment_t0(segments, 0) * 0.01f;
                    final_start_seconds = std::max(0.0f, final_start_seconds - 0.5f);
                    speech_detected = true;
                    whisper_vad_free_segments(segments);
                    break;
                }
                whisper_vad_free_segments(segments);
            }
        }
    }

    if (call_trim_end && (speech_detected || !call_trim_start)) {
        bool end_found = false;
        for (int i = (int)pcmf32.size(); i > 0; i -= chunk_size_samples) {
            int start_sample = std::max(0, i - chunk_size_samples);
            int n_samples = i - start_sample;
            struct whisper_vad_segments * segments = whisper_vad_segments_from_samples(vctx, vad_params, pcmf32.data() + start_sample, n_samples);
            if (segments) {
                int n_seg = whisper_vad_segments_n_segments(segments);
                if (n_seg > 0) {
                    final_end_seconds = (float)start_sample / (float)WHISPER_SAMPLE_RATE + whisper_vad_segments_get_segment_t1(segments, n_seg - 1) * 0.01f;
                    final_end_seconds = std::min(total_duration_seconds, final_end_seconds + 0.5f);
                    speech_detected = true;
                    end_found = true;
                    whisper_vad_free_segments(segments);
                    break;
                }
                whisper_vad_free_segments(segments);
            }
            if (call_trim_start && start_sample <= (int)(final_start_seconds * WHISPER_SAMPLE_RATE)) {
                break;
            }
        }
    }

    whisper_vad_free(vctx);

    if (!speech_detected) {
        fprintf(stderr, "No speech detected. Not creating an output file.\n");
        return 0;
    }

    if (final_start_seconds <= 0.01f && final_end_seconds >= total_duration_seconds - 0.01f) {
        fprintf(stderr, "No significant silence detected. Not creating an output file.\n");
        if (replace_input) {
            remove(output_file.c_str());
        }
        return 0;
    }

    // FFmpeg command construction
    // We use -ss before -i for faster seeking if it's a long file
    std::string trim_cmd = "ffmpeg -hide_banner -loglevel error -nostdin -y -ss " + std::to_string(final_start_seconds) + " -i \"" + audio_file + "\"";
    
    if (final_end_seconds < total_duration_seconds) {
        trim_cmd += " -to " + std::to_string(final_end_seconds - final_start_seconds);
        fprintf(stderr, "Detected speech from %.3f to %.3f (duration: %.3f).\n", 
                final_start_seconds, final_end_seconds, final_end_seconds - final_start_seconds);
    } else {
        fprintf(stderr, "Detected speech from %.3f.\n", final_start_seconds);
    }

    trim_cmd += " -c copy \"" + output_file + "\"";

    fprintf(stderr, "Trimming audio and saving to %s...\n", output_file.c_str());
    if (system(trim_cmd.c_str()) != 0) {
        fprintf(stderr, "Error: Failed to trim audio using ffmpeg.\n");
        if (replace_input) remove(output_file.c_str());
        return 1;
    }

    fprintf(stderr, "Successfully created %s.\n", output_file.c_str());

    if (replace_input) {
        std::string mv_cmd = "mv \"" + output_file + "\" \"" + audio_file + "\"";
        if (system(mv_cmd.c_str()) != 0) {
            fprintf(stderr, "Error: Failed to replace original file %s with %s.\n", audio_file.c_str(), output_file.c_str());
            return 1;
        }
        fprintf(stderr, "Original file %s has been overwritten.\n", audio_file.c_str());
    }

    return 0;
}

