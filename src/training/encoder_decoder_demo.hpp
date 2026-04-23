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
        for (arch::Scalar& v : x) {
            v = static_cast<arch::Scalar>(dist(rng));
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

    ParameterTensor enc_w{
        std::vector<arch::Scalar>(hidden_dim * input_dim),
        std::vector<arch::Scalar>(hidden_dim * input_dim, static_cast<arch::Scalar>(0.0f))};
    ParameterTensor enc_b{
        std::vector<arch::Scalar>(hidden_dim, static_cast<arch::Scalar>(0.0f)),
        std::vector<arch::Scalar>(hidden_dim, static_cast<arch::Scalar>(0.0f))};
    ParameterTensor dec_w{
        std::vector<arch::Scalar>(output_dim * hidden_dim),
        std::vector<arch::Scalar>(output_dim * hidden_dim, static_cast<arch::Scalar>(0.0f))};
    ParameterTensor dec_b{
        std::vector<arch::Scalar>(output_dim, static_cast<arch::Scalar>(0.0f)),
        std::vector<arch::Scalar>(output_dim, static_cast<arch::Scalar>(0.0f))};

    for (arch::Scalar& v : enc_w.values) {
        v = static_cast<arch::Scalar>(init_dist(rng));
    }
    for (arch::Scalar& v : dec_w.values) {
        v = static_cast<arch::Scalar>(init_dist(rng));
    }

    arch::EncoderDecoderModel model;
    model.set_encoder([&](const arch::Matrix& src) {
        const arch::Vector h_raw = arch::sycl_ops::linear(enc_w.values, hidden_dim, input_dim, src[0], enc_b.values);
        arch::Vector h(hidden_dim, static_cast<arch::Scalar>(0.0f));
        for (size_t i = 0; i < hidden_dim; ++i) {
            h[i] = static_cast<arch::Scalar>(std::tanh(static_cast<float>(h_raw[i])));
        }
        return arch::Matrix{ h };
    });

    model.set_decoder([&](const arch::Matrix& tgt, const arch::Matrix& memory) {
        (void)tgt;
        const arch::Vector y = arch::sycl_ops::linear(dec_w.values, output_dim, hidden_dim, memory[0], dec_b.values);
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
            std::fill(params[0].grads.begin(), params[0].grads.end(), static_cast<arch::Scalar>(0.0f));
            std::fill(params[1].grads.begin(), params[1].grads.end(), static_cast<arch::Scalar>(0.0f));
            std::fill(params[2].grads.begin(), params[2].grads.end(), static_cast<arch::Scalar>(0.0f));
            std::fill(params[3].grads.begin(), params[3].grads.end(), static_cast<arch::Scalar>(0.0f));

            const arch::Vector& x = ds.src[s][0];
            const arch::Vector& target = ds.target[s][0];

            arch::Vector h_raw = arch::sycl_ops::linear(params[0].values, hidden_dim, input_dim, x, params[1].values);
            arch::Vector h(hidden_dim, static_cast<arch::Scalar>(0.0f));
            for (size_t i = 0; i < hidden_dim; ++i) {
                h[i] = static_cast<arch::Scalar>(std::tanh(static_cast<float>(h_raw[i])));
            }
            arch::Vector y = arch::sycl_ops::linear(params[2].values, output_dim, hidden_dim, h, params[3].values);

            arch::Vector dldy(output_dim, static_cast<arch::Scalar>(0.0f));
            for (size_t o = 0; o < output_dim; ++o) {
                const float dy = (2.0f / static_cast<float>(output_dim))
                    * (static_cast<float>(y[o]) - static_cast<float>(target[o]));
                dldy[o] = static_cast<arch::Scalar>(dy);
            }

            for (size_t o = 0; o < output_dim; ++o) {
                params[3].grads[o] = static_cast<arch::Scalar>(
                    static_cast<float>(params[3].grads[o]) + static_cast<float>(dldy[o]));
                for (size_t hidx = 0; hidx < hidden_dim; ++hidx) {
                    const float g = static_cast<float>(params[2].grads[o * hidden_dim + hidx])
                        + static_cast<float>(dldy[o]) * static_cast<float>(h[hidx]);
                    params[2].grads[o * hidden_dim + hidx] = static_cast<arch::Scalar>(g);
                }
            }

            arch::Vector dldh(hidden_dim, static_cast<arch::Scalar>(0.0f));
            for (size_t hidx = 0; hidx < hidden_dim; ++hidx) {
                float sacc = 0.0f;
                for (size_t o = 0; o < output_dim; ++o) {
                    sacc += static_cast<float>(dldy[o])
                        * static_cast<float>(params[2].values[o * hidden_dim + hidx]);
                }
                dldh[hidx] = static_cast<arch::Scalar>(sacc);
            }

            arch::Vector dldh_raw(hidden_dim, static_cast<arch::Scalar>(0.0f));
            for (size_t hidx = 0; hidx < hidden_dim; ++hidx) {
                const float hv = static_cast<float>(h[hidx]);
                const float grad = static_cast<float>(dldh[hidx]) * (1.0f - hv * hv);
                dldh_raw[hidx] = static_cast<arch::Scalar>(grad);
            }

            for (size_t hidx = 0; hidx < hidden_dim; ++hidx) {
                params[1].grads[hidx] = static_cast<arch::Scalar>(
                    static_cast<float>(params[1].grads[hidx]) + static_cast<float>(dldh_raw[hidx]));
                for (size_t i = 0; i < input_dim; ++i) {
                    const float g = static_cast<float>(params[0].grads[hidx * input_dim + i])
                        + static_cast<float>(dldh_raw[hidx]) * static_cast<float>(x[i]);
                    params[0].grads[hidx * input_dim + i] = static_cast<arch::Scalar>(g);
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
