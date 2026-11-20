#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

struct options {
    std::string model;
    std::string tokens;
    std::string logits_out;
    std::string metrics_out;
    int seq_len = 2048;
    int windows = 1;
    int decode_steps = 0;
    int threads = 16;
    int threads_batch = 16;
};

static options parse_args(int argc, char ** argv) {
    options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (i + 1 == argc) {
            throw std::runtime_error("missing value for " + arg);
        }
        const std::string value = argv[++i];
        if (arg == "--model") opt.model = value;
        else if (arg == "--tokens") opt.tokens = value;
        else if (arg == "--logits-out") opt.logits_out = value;
        else if (arg == "--metrics-out") opt.metrics_out = value;
        else if (arg == "--seq-len") opt.seq_len = std::stoi(value);
        else if (arg == "--windows") opt.windows = std::stoi(value);
        else if (arg == "--decode-steps") opt.decode_steps = std::stoi(value);
        else if (arg == "--threads") opt.threads = std::stoi(value);
        else if (arg == "--threads-batch") opt.threads_batch = std::stoi(value);
        else throw std::runtime_error("unknown argument: " + arg);
    }
    if (opt.model.empty() || opt.tokens.empty() || opt.seq_len < 1 || opt.windows < 1 ||
        opt.decode_steps < 0 || opt.threads < 1 || opt.threads_batch < 1) {
        throw std::runtime_error("required: --model M --tokens T [--seq-len N --windows N --decode-steps N]");
    }
    if ((opt.decode_steps > 0 || !opt.logits_out.empty()) && opt.windows != 1) {
        throw std::runtime_error("--decode-steps and --logits-out require --windows 1");
    }
    return opt;
}

static std::vector<uint32_t> read_tokens(const options & opt) {
    std::ifstream in(opt.tokens, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("cannot open tokens: " + opt.tokens);
    const size_t required = size_t(opt.windows) * (opt.seq_len + 1) + opt.decode_steps;
    const size_t bytes = size_t(in.tellg());
    if (bytes < required * sizeof(uint32_t)) {
        throw std::runtime_error("token file too short: need " + std::to_string(required) + " uint32 tokens");
    }
    std::vector<uint32_t> tokens(required);
    in.seekg(0);
    in.read(reinterpret_cast<char *>(tokens.data()), tokens.size() * sizeof(uint32_t));
    if (!in) throw std::runtime_error("failed reading token stream");
    return tokens;
}

struct token_metrics {
    double ce = 0.0;
    int64_t correct = 0;
};

static token_metrics measure_logits(const float * logits, const uint32_t * targets, int count, int vocab) {
    token_metrics result;
    double ce = 0.0;
    int64_t correct = 0;
#pragma omp parallel for reduction(+ : ce, correct) schedule(static)
    for (int i = 0; i < count; ++i) {
        const float * row = logits + size_t(i) * vocab;
        float max_value = -std::numeric_limits<float>::infinity();
        int argmax = 0;
        for (int j = 0; j < vocab; ++j) {
            if (row[j] > max_value) {
                max_value = row[j];
                argmax = j;
            }
        }
        double denom = 0.0;
        for (int j = 0; j < vocab; ++j) denom += std::exp(row[j] - max_value);
        const int target = int(targets[i]);
        ce += std::log(denom) + max_value - row[target];
        correct += argmax == target;
    }
    result.ce = ce;
    result.correct = correct;
    return result;
}

static void write_logits(std::ofstream & out, llama_context * ctx, int count, int vocab) {
    for (int i = 0; i < count; ++i) {
        const float * logits = llama_get_logits_ith(ctx, i);
        if (!logits) throw std::runtime_error("missing logits at index " + std::to_string(i));
        out.write(reinterpret_cast<const char *>(logits), size_t(vocab) * sizeof(float));
    }
    if (!out) throw std::runtime_error("failed writing logits");
}

int main(int argc, char ** argv) {
    try {
        const options opt = parse_args(argc, argv);
        const auto tokens = read_tokens(opt);
        llama_backend_init();
        ggml_backend_load_all();
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = 0;
        llama_model * model = llama_model_load_from_file(opt.model.c_str(), model_params);
        if (!model) throw std::runtime_error("model load failed");
        const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        for (uint32_t token : tokens) {
            if (token >= uint32_t(vocab)) throw std::runtime_error("token outside model vocabulary");
        }
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = opt.seq_len + opt.decode_steps + 1;
        ctx_params.n_batch = opt.seq_len;
        ctx_params.n_ubatch = opt.seq_len;
        ctx_params.n_seq_max = 1;
        ctx_params.n_threads = opt.threads;
        ctx_params.n_threads_batch = opt.threads_batch;
        ctx_params.no_perf = false;
        llama_context * ctx = llama_init_from_model(model, ctx_params);
        if (!ctx) throw std::runtime_error("context initialization failed");

        std::ofstream logits_stream;
        if (!opt.logits_out.empty()) {
            logits_stream.open(opt.logits_out, std::ios::binary);
            if (!logits_stream) throw std::runtime_error("cannot create logits output");
            char magic[8] = {'B','N','E','L','G','1',0,0};
            const uint32_t header[3] = {uint32_t(opt.seq_len), uint32_t(opt.decode_steps), uint32_t(vocab)};
            logits_stream.write(magic, sizeof(magic));
            logits_stream.write(reinterpret_cast<const char *>(header), sizeof(header));
        }

        double ce_sum = 0.0;
        int64_t correct = 0;
        int64_t predictions = 0;
        double prefill_seconds = 0.0;
        double decode_seconds = 0.0;
        const auto eval_start = std::chrono::steady_clock::now();
        for (int window = 0; window < opt.windows; ++window) {
            llama_memory_clear(llama_get_memory(ctx), false);
            const size_t offset = size_t(window) * (opt.seq_len + 1);
            llama_batch batch = llama_batch_init(opt.seq_len, 0, 1);
            for (int i = 0; i < opt.seq_len; ++i) {
                batch.token[i] = llama_token(tokens[offset + i]);
                batch.pos[i] = i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = 1;
            }
            batch.n_tokens = opt.seq_len;
            const auto prefill_start = std::chrono::steady_clock::now();
            const int status = llama_decode(ctx, batch);
            prefill_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - prefill_start).count();
            if (status != 0) throw std::runtime_error("prefill failed with status " + std::to_string(status));
            const float * logits = llama_get_logits(ctx);
            const token_metrics prefill = measure_logits(logits, tokens.data() + offset + 1, opt.seq_len, vocab);
            ce_sum += prefill.ce;
            correct += prefill.correct;
            predictions += opt.seq_len;
            if (!opt.logits_out.empty()) write_logits(logits_stream, ctx, opt.seq_len, vocab);
            llama_batch_free(batch);

            if (window == 0) {
                for (int step = 0; step < opt.decode_steps; ++step) {
                    llama_batch one = llama_batch_init(1, 0, 1);
                    one.token[0] = llama_token(tokens[offset + opt.seq_len + step]);
                    one.pos[0] = opt.seq_len + step;
                    one.n_seq_id[0] = 1;
                    one.seq_id[0][0] = 0;
                    one.logits[0] = 1;
                    one.n_tokens = 1;
                    const auto decode_start = std::chrono::steady_clock::now();
                    const int decode_status = llama_decode(ctx, one);
                    decode_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - decode_start).count();
                    if (decode_status != 0) throw std::runtime_error("cached decode failed");
                    const token_metrics decoded = measure_logits(llama_get_logits(ctx), tokens.data() + offset + opt.seq_len + step + 1, 1, vocab);
                    ce_sum += decoded.ce;
                    correct += decoded.correct;
                    ++predictions;
                    if (!opt.logits_out.empty()) write_logits(logits_stream, ctx, 1, vocab);
                    llama_batch_free(one);
                }
            }
            if (opt.windows > 1) {
                std::fprintf(stderr, "window %d/%d cumulative_ce %.9f\n", window + 1, opt.windows, ce_sum / predictions);
            }
        }
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - eval_start).count();
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        const double ce = ce_sum / predictions;
        const double accuracy = double(correct) / predictions;
        std::printf("tokens %lld CE %.9f top1 %.9f elapsed %.3f s\n",
                    (long long)predictions, ce, accuracy, elapsed);
        if (!opt.metrics_out.empty()) {
            std::ofstream out(opt.metrics_out);
            if (!out) throw std::runtime_error("cannot create metrics output");
            out.precision(12);
            out << "{\n"
                << "  \"model\": \"" << opt.model << "\",\n"
                << "  \"tokens\": \"" << opt.tokens << "\",\n"
                << "  \"sequence_length\": " << opt.seq_len << ",\n"
                << "  \"windows\": " << opt.windows << ",\n"
                << "  \"decode_steps\": " << opt.decode_steps << ",\n"
                << "  \"token_predictions\": " << predictions << ",\n"
                << "  \"cross_entropy\": " << ce << ",\n"
                << "  \"top1_accuracy\": " << accuracy << ",\n"
                << "  \"elapsed_seconds\": " << elapsed << ",\n"
                << "  \"prefill_seconds\": " << prefill_seconds << ",\n"
                << "  \"decode_seconds\": " << decode_seconds << ",\n"
                << "  \"threads\": " << opt.threads << ",\n"
                << "  \"threads_batch\": " << opt.threads_batch << "\n}\n";
        }
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "bitnet-eval: %s\n", error.what());
        return 1;
    }
}
