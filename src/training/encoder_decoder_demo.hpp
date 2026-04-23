#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include <architecture/encoder-decoder.hpp>
#include <training/evaluation.hpp>
#include <training/sgd.hpp>

namespace llm::training::demo {

struct OverfitRunResult {
    float initial_loss;
    float final_loss;
    size_t epochs;
    size_t samples;
};

struct TinyDataset {
    std::vector<arch::Matrix> src;
    std::vector<arch::Matrix> tgt;
    std::vector<arch::Matrix> target;
};

inline TinyDataset make_tiny_random_dataset(
    size_t samples = 8,
    size_t input_dim = 3,
    uint32_t seed = 7) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    TinyDataset ds;
    ds.src.reserve(samples);
    ds.tgt.reserve(samples);
    ds.target.reserve(samples);

    for (size_t i = 0; i < samples; ++i) {
        arch::Vector x(input_dim);
        for (float& v : x) {
            v = dist(rng);
        }

        // Auto-encoding style task: predict source values.
        ds.src.push_back(arch::Matrix{ x });
        ds.tgt.push_back(arch::Matrix{ arch::Vector(input_dim, 0.0f) });
        ds.target.push_back(arch::Matrix{ x });
    }

    return ds;
}

inline float eval_mse_dataset(
    const arch::EncoderDecoderModel& model,
    const TinyDataset& ds) {
    if (ds.src.size() != ds.tgt.size() || ds.src.size() != ds.target.size()) {
        throw std::runtime_error("Dataset shape mismatch");
    }
    if (ds.src.empty()) {
        return 0.0f;
    }

    float total = 0.0f;
    for (size_t i = 0; i < ds.src.size(); ++i) {
        const arch::Matrix pred = model.forward(ds.src[i], ds.tgt[i]);
        total += evaluation::mean_squared_error(pred[0], ds.target[i][0]);
    }
    return total / static_cast<float>(ds.src.size());
}

inline OverfitRunResult run_minimal_encoder_decoder_sgd_overfit(
    size_t epochs = 1500,
    float learning_rate = 5e-2f,
    uint32_t seed = 7) {
    constexpr size_t input_dim = 3;
    constexpr size_t hidden_dim = 6;
    constexpr size_t output_dim = 3;

    TinyDataset ds = make_tiny_random_dataset(8, input_dim, seed);

    std::mt19937 rng(seed + 17U);
    std::normal_distribution<float> init_dist(0.0f, 0.15f);

    ParameterTensor enc_w{std::vector<float>(hidden_dim * input_dim), std::vector<float>(hidden_dim * input_dim, 0.0f)};
    ParameterTensor enc_b{std::vector<float>(hidden_dim, 0.0f), std::vector<float>(hidden_dim, 0.0f)};
    ParameterTensor dec_w{std::vector<float>(output_dim * hidden_dim), std::vector<float>(output_dim * hidden_dim, 0.0f)};
    ParameterTensor dec_b{std::vector<float>(output_dim, 0.0f), std::vector<float>(output_dim, 0.0f)};

    for (float& v : enc_w.values) {
        v = init_dist(rng);
    }
    for (float& v : dec_w.values) {
        v = init_dist(rng);
    }

    auto linear = [](const std::vector<float>& w, size_t out_dim, size_t in_dim,
                     const arch::Vector& x, const std::vector<float>& b) {
        arch::Vector y(out_dim, 0.0f);
        for (size_t o = 0; o < out_dim; ++o) {
            float s = b[o];
            for (size_t i = 0; i < in_dim; ++i) {
                s += w[o * in_dim + i] * x[i];
            }
            y[o] = s;
        }
        return y;
    };

    arch::EncoderDecoderModel model;
    model.set_encoder([&](const arch::Matrix& src) {
        const arch::Vector h_raw = linear(enc_w.values, hidden_dim, input_dim, src[0], enc_b.values);
        arch::Vector h(hidden_dim, 0.0f);
        for (size_t i = 0; i < hidden_dim; ++i) {
            h[i] = std::tanh(h_raw[i]);
        }
        return arch::Matrix{ h };
    });

    model.set_decoder([&](const arch::Matrix& tgt, const arch::Matrix& memory) {
        (void)tgt;
        const arch::Vector y = linear(dec_w.values, output_dim, hidden_dim, memory[0], dec_b.values);
        return arch::Matrix{ y };
    });

    std::vector<ParameterTensor> params;
    params.reserve(4);
    params.push_back(ParameterTensor{enc_w.values, enc_w.grads});
    params.push_back(ParameterTensor{enc_b.values, enc_b.grads});
    params.push_back(ParameterTensor{dec_w.values, dec_w.grads});
    params.push_back(ParameterTensor{dec_b.values, dec_b.grads});

    SGD sgd(learning_rate, 0.0f);

    // Keep references in sync with optimizer-owned parameter buffers.
    auto sync_from_params = [&]() {
        enc_w.values = params[0].values;
        enc_b.values = params[1].values;
        dec_w.values = params[2].values;
        dec_b.values = params[3].values;
    };

    const float initial_loss = eval_mse_dataset(model, ds);

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        for (size_t s = 0; s < ds.src.size(); ++s) {
            std::fill(params[0].grads.begin(), params[0].grads.end(), 0.0f);
            std::fill(params[1].grads.begin(), params[1].grads.end(), 0.0f);
            std::fill(params[2].grads.begin(), params[2].grads.end(), 0.0f);
            std::fill(params[3].grads.begin(), params[3].grads.end(), 0.0f);

            const arch::Vector& x = ds.src[s][0];
            const arch::Vector& target = ds.target[s][0];

            arch::Vector h_raw = linear(params[0].values, hidden_dim, input_dim, x, params[1].values);
            arch::Vector h(hidden_dim, 0.0f);
            for (size_t i = 0; i < hidden_dim; ++i) {
                h[i] = std::tanh(h_raw[i]);
            }
            arch::Vector y = linear(params[2].values, output_dim, hidden_dim, h, params[3].values);

            arch::Vector dldy(output_dim, 0.0f);
            for (size_t o = 0; o < output_dim; ++o) {
                dldy[o] = (2.0f / static_cast<float>(output_dim)) * (y[o] - target[o]);
            }

            for (size_t o = 0; o < output_dim; ++o) {
                params[3].grads[o] += dldy[o];
                for (size_t hidx = 0; hidx < hidden_dim; ++hidx) {
                    params[2].grads[o * hidden_dim + hidx] += dldy[o] * h[hidx];
                }
            }

            arch::Vector dldh(hidden_dim, 0.0f);
            for (size_t hidx = 0; hidx < hidden_dim; ++hidx) {
                float sacc = 0.0f;
                for (size_t o = 0; o < output_dim; ++o) {
                    sacc += dldy[o] * params[2].values[o * hidden_dim + hidx];
                }
                dldh[hidx] = sacc;
            }

            arch::Vector dldh_raw(hidden_dim, 0.0f);
            for (size_t hidx = 0; hidx < hidden_dim; ++hidx) {
                dldh_raw[hidx] = dldh[hidx] * (1.0f - h[hidx] * h[hidx]);
            }

            for (size_t hidx = 0; hidx < hidden_dim; ++hidx) {
                params[1].grads[hidx] += dldh_raw[hidx];
                for (size_t i = 0; i < input_dim; ++i) {
                    params[0].grads[hidx * input_dim + i] += dldh_raw[hidx] * x[i];
                }
            }

            sgd.step(params);
        }

        sync_from_params();
    }

    const float final_loss = eval_mse_dataset(model, ds);
    return OverfitRunResult{initial_loss, final_loss, epochs, ds.src.size()};
}

} // namespace llm::training::demo
