#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <mutex>
#include <exception>
#include <utility>
#include <vector>

#include <architecture/attention.hpp>
#include <architecture/looping_retnet.hpp>
#include <architecture/retentive.hpp>
#include <long_term/memory_module.hpp>
#include <training/sgd.hpp>

namespace llm::training::looping {

struct SequenceExample {
    std::string input;
    std::string target;
};

enum class TrainMode : uint8_t {
    FiniteDifference = 0,
    BackpropHeads = 1,
    BackpropFull = 2,
};

enum class OutputLossType : uint8_t {
    CrossEntropy = 0,         // standard -log(p_t)
    WeightedCrossEntropy = 1, // w_t * -log(p_t)
    FocalLoss = 2,            // (1-p_t)^gamma * -log(p_t), optionally with class weights
};

struct TrainConfig {
    TrainMode mode = TrainMode::BackpropFull;
    size_t epochs = 256;
    float learning_rate = 3.0e-4f;      // peak LR after warmup (Adam-safe default)
    float min_learning_rate = 1e-6f;  // absolute LR floor
    float min_learning_rate_ratio = 0.1f; // relative floor vs peak LR
    size_t warmup_epochs = 32;         // linear warmup epochs
    float warmup_start_ratio = 0.1f;  // warmup starts at peak_lr * ratio
    float loss_ema_beta = 0.9f;       // moving-average smoothing for logging
    float weight_decay = 1e-4f;
    float sgd_momentum = 0.9f;
    float max_gradient_abs = 5.0f;    // per-coordinate grad clip
    float max_parameter_abs = 100.0f; // post-step parameter clamp
    bool enable_instability_backoff = true;
    float instability_skip_ratio_threshold = 0.25f;
    float instability_repair_ratio_threshold = 0.01f;
    float instability_lr_backoff = 0.5f;
    size_t instability_cooldown_epochs = 5;
    float min_effective_learning_rate = 1e-5f;

    // Finite-difference gradient estimate over randomly sampled coordinates.
    float fd_eps = 5e-3f;
    size_t grad_coordinate_samples = 512;
    size_t min_grad_coordinate_samples = 128;

    // Minibatch support used by both modes where applicable.
    size_t batch_size = 256;
    // Host-side parallelism for per-example work in minibatches.
    // 0 = auto (use hardware_concurrency), 1 = force single-thread.
    size_t host_threads = 0;
    // Avoid thread overhead for tiny batches.
    size_t min_parallel_batch_examples = 16;
    bool disable_memory_writes_when_query_disabled = true;

    // Backprop-over-heads mode knobs.
    bool backprop_force_single_step = false;
    bool backprop_include_loop_supervision = true;
    size_t backprop_fd_check_samples = 0;
    float backprop_fd_check_eps = 1e-3f;

    // Memory-module training knobs.
    float memory_query_penalty = 0.05f; // penalize no-query behavior
    float memory_miss_penalty = 0.05f;  // penalize empty query results
    float memory_alignment_weight = 0.1f;
    float memory_edge_budget_penalty = 0.02f;
    float loop_supervision_weight = 0.2f;
    float force_query_prob = 0.15f;
    float load_gate_supervision_weight = 0.1f;
    bool force_output = false;          // keep false for full looping behavior
    bool enable_query = true;
    bool use_parallel_retention = true;

    size_t forced_loop_min = 1;
    size_t forced_loop_max = 4;

    // memory_warmup_epochs: let the base recurrent weights stabilize before
    // enabling memory writes/queries (important at 1M params scale).
    size_t memory_warmup_epochs = 16;
    // memory_curriculum_epochs: ramp forced_loop_max from forced_loop_min up to
    // forced_loop_max over this many epochs.
    size_t memory_curriculum_epochs = 48;

    // Rolling window size and variance/mean² threshold for adaptive instability.
    size_t adaptive_instability_window = 5;
    float adaptive_instability_var_ratio = 0.05f;

    // Penalises large changes in raw output logits between consecutive steps.
    // weight 0 disables the term entirely.
    float multistep_consistency_weight = 0.0f;

    uint32_t seed = 7;
    // 0 = use all examples each epoch; >0 = randomly sample this many (with replacement)
    size_t samples_per_epoch = 0;

    // Output character-prediction loss type.
    OutputLossType output_loss_type = OutputLossType::FocalLoss;
    // Focusing parameter for FocalLoss (>= 0; 0 == standard CE, 2 is a common default).
    float focal_gamma = 3.0f;
    // Per-class weights for WeightedCrossEntropy / FocalLoss.  Empty == uniform (all 1.0).
    // Size must equal the vocabulary size (256 for byte-level models).
    std::vector<float> class_weights;
    // Focal-loss class probability cap: p_t = min(sampled_probability, min_p_t).
    // Keep in [0, 1].  Default 1.0 preserves sampled probabilities unchanged.
    float min_p_t = 0.001f;
    // Number of examples randomly sampled (with replacement) to estimate per-class p_t.
    // 0 means use the full dataset.
    size_t p_t_sampling_examples = 4096;

    memory::MemoryConfig memory_cfg{};
};

struct EpochResult {
    size_t epoch = 0;
    float avg_loss = 0.0f;
};

class LoopingRetNetSGDTrainer {
    TrainConfig cfg_;
    std::vector<float> sampled_target_pt_;

    static constexpr float kNonFinitePenalty = 100.0f;

    float learning_rate_for_epoch(size_t epoch_1_based) const {
        if (cfg_.epochs <= 1) {
            return cfg_.learning_rate;
        }

        const float peak_lr = cfg_.learning_rate;
        const float ratio_floor = peak_lr * std::max(0.0f, cfg_.min_learning_rate_ratio);
        const float min_lr = std::max(cfg_.min_learning_rate, ratio_floor);

        const size_t max_warmup = (cfg_.epochs > 1) ? (cfg_.epochs - 1) : 0;
        const size_t warmup_epochs = std::min(cfg_.warmup_epochs, max_warmup);

        if (warmup_epochs > 0 && epoch_1_based <= warmup_epochs) {
            const float warmup_start = std::max(min_lr, peak_lr * std::max(0.0f, cfg_.warmup_start_ratio));
            if (warmup_epochs == 1) {
                return peak_lr;
            }
            const float tw = static_cast<float>(epoch_1_based - 1)
                / static_cast<float>(warmup_epochs - 1);
            return warmup_start + (peak_lr - warmup_start) * tw;
        }

        const size_t decay_start_epoch = warmup_epochs + 1;
        const size_t decay_steps = cfg_.epochs - decay_start_epoch;
        if (decay_steps == 0) {
            return peak_lr;
        }

        const float t = static_cast<float>(epoch_1_based - decay_start_epoch)
            / static_cast<float>(decay_steps);
        constexpr float kPi = 3.14159265358979323846f;
        const float cos_decay = 0.5f * (1.0f + std::cos(kPi * t));
        return min_lr + (peak_lr - min_lr) * cos_decay;
    }

    size_t grad_samples_for_epoch(size_t epoch_1_based) const {
        const size_t max_samples = std::max<size_t>(1, cfg_.grad_coordinate_samples);
        const size_t min_samples = std::max<size_t>(1, std::min(cfg_.min_grad_coordinate_samples, max_samples));
        if (cfg_.epochs <= 1 || max_samples == min_samples) {
            return max_samples;
        }

        const float t = static_cast<float>(epoch_1_based - 1)
            / static_cast<float>(cfg_.epochs - 1);
        const float smooth = t * t;
        const float cur = static_cast<float>(max_samples)
            + (static_cast<float>(min_samples) - static_cast<float>(max_samples)) * smooth;
        return std::max<size_t>(1, static_cast<size_t>(std::lround(cur)));
    }

    size_t effective_host_threads() const {
        if (cfg_.host_threads > 0) {
            return std::max<size_t>(1, cfg_.host_threads);
        }
        const unsigned int hc = std::thread::hardware_concurrency();
        return std::max<size_t>(1, static_cast<size_t>(hc == 0 ? 1 : hc));
    }

    static uint32_t per_example_seed(
        uint32_t base_seed,
        size_t epoch_1_based,
        size_t batch_start,
        size_t example_index)
    {
        uint64_t x = static_cast<uint64_t>(base_seed) + 0x9e3779b97f4a7c15ULL;
        x ^= static_cast<uint64_t>(epoch_1_based) + 0xBF58476D1CE4E5B9ULL + (x << 6) + (x >> 2);
        x ^= static_cast<uint64_t>(batch_start) + 0x94D049BB133111EBULL + (x << 6) + (x >> 2);
        x ^= static_cast<uint64_t>(example_index) + 0xD6E8FEB86659FD93ULL + (x << 6) + (x >> 2);
        x ^= (x >> 30);
        x *= 0xBF58476D1CE4E5B9ULL;
        x ^= (x >> 27);
        x *= 0x94D049BB133111EBULL;
        x ^= (x >> 31);
        return static_cast<uint32_t>(x & 0xFFFFFFFFu);
    }

    template <typename Fn>
    static void parallel_for_ranges(
        size_t begin,
        size_t end,
        size_t thread_count,
        Fn&& fn)
    {
        if (end <= begin) {
            return;
        }
        const size_t n = end - begin;
        const size_t workers = std::max<size_t>(1, std::min(thread_count, n));
        if (workers == 1) {
            fn(begin, end, 0);
            return;
        }

        std::vector<std::thread> threads;
        threads.reserve(workers);
        std::exception_ptr first_error;
        std::mutex error_mtx;

        for (size_t w = 0; w < workers; ++w) {
            const size_t s = begin + (w * n) / workers;
            const size_t e = begin + ((w + 1) * n) / workers;
            threads.emplace_back([&, s, e, w]() {
                try {
                    if (s < e) {
                        fn(s, e, w);
                    }
                } catch (...) {
                    std::lock_guard<std::mutex> lock(error_mtx);
                    if (!first_error) {
                        first_error = std::current_exception();
                    }
                }
            });
        }

        for (auto& t : threads) {
            t.join();
        }
        if (first_error) {
            std::rethrow_exception(first_error);
        }
    }

    bool memory_active_for_epoch(size_t epoch_1_based) const {
        if (cfg_.memory_warmup_epochs == 0) {
            return true;
        }
        return epoch_1_based > cfg_.memory_warmup_epochs;
    }

    size_t effective_forced_loop_max_for_epoch(size_t epoch_1_based) const {
        const size_t base_max = std::max(cfg_.forced_loop_min, cfg_.forced_loop_max);
        const size_t base_min = cfg_.forced_loop_min;
        if (cfg_.memory_curriculum_epochs == 0
            || epoch_1_based > cfg_.memory_curriculum_epochs
            || base_max <= base_min) {
            return base_max;
        }
        const float t = static_cast<float>(epoch_1_based)
            / static_cast<float>(cfg_.memory_curriculum_epochs);
        const float interp = static_cast<float>(base_min)
            + t * static_cast<float>(base_max - base_min);
        return std::max(base_min,
            std::min(base_max, static_cast<size_t>(std::lround(interp))));
    }

    static std::vector<size_t> stratified_coordinate_indices(
        size_t param_count,
        size_t sample_count,
        std::mt19937& rng)
    {
        std::vector<size_t> out;
        if (param_count == 0 || sample_count == 0) {
            return out;
        }

        out.reserve(sample_count);
        std::uniform_int_distribution<size_t> tail_dist(0, param_count - 1);
        const size_t block_count = std::min(sample_count, param_count);
        for (size_t b = 0; b < block_count; ++b) {
            const size_t start = (b * param_count) / block_count;
            size_t end = ((b + 1) * param_count) / block_count;
            if (end <= start) {
                end = std::min(param_count, start + 1);
            }
            std::uniform_int_distribution<size_t> block_dist(start, end - 1);
            out.push_back(block_dist(rng));
        }
        while (out.size() < sample_count) {
            out.push_back(tail_dist(rng));
        }
        return out;
    }

    static float cosine_similarity(const arch::Vector& a, const arch::Vector& b) {
        if (a.empty() || b.empty() || a.size() != b.size()) {
            return 0.0f;
        }
        float dot = 0.0f;
        float na = 0.0f;
        float nb = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) {
            const float av = static_cast<float>(a[i]);
            const float bv = static_cast<float>(b[i]);
            dot += av * bv;
            na += av * av;
            nb += bv * bv;
        }
        if (na <= 1e-12f || nb <= 1e-12f) {
            return 0.0f;
        }
        return dot / (std::sqrt(na) * std::sqrt(nb));
    }

    static float action_ce_from_logits(const arch::Vector& logits, size_t target_class) {
        if (logits.empty() || target_class >= logits.size()) {
            return 0.0f;
        }

        float max_logit = -std::numeric_limits<float>::infinity();
        bool found_finite = false;
        for (arch::Scalar v : logits) {
            const float fv = static_cast<float>(v);
            if (!std::isfinite(fv)) {
                continue;
            }
            max_logit = std::max(max_logit, fv);
            found_finite = true;
        }
        if (!found_finite) {
            return kNonFinitePenalty;
        }

        float sum_exp = 0.0f;
        for (arch::Scalar v : logits) {
            const float fv = static_cast<float>(v);
            if (!std::isfinite(fv)) {
                continue;
            }
            const float shifted = std::clamp(fv - max_logit, -80.0f, 0.0f);
            sum_exp += std::exp(shifted);
        }
        const float target_logit = static_cast<float>(logits[target_class]);
        if (!std::isfinite(target_logit) || !std::isfinite(sum_exp) || sum_exp <= 0.0f) {
            return kNonFinitePenalty;
        }

        const float lp = target_logit - max_logit - std::log(sum_exp);
        if (!std::isfinite(lp)) {
            return kNonFinitePenalty;
        }
        return std::max(0.0f, -lp);
    }

    static std::pair<float, float> binary_ce_and_grad(float logit, float target) {
        const float p = 1.0f / (1.0f + std::exp(-logit));
        const float eps = 1e-6f;
        const float ce = -(target * std::log(std::max(eps, p))
            + (1.0f - target) * std::log(std::max(eps, 1.0f - p)));
        const float grad = p - target;
        return {ce, grad};
    }

    static float cross_entropy_from_logits(const arch::Vector& logits, uint8_t target) {
        if (logits.empty()) {
            throw std::runtime_error("LoopingRetNetSGDTrainer: empty logits");
        }
        const size_t cls = static_cast<size_t>(target);
        if (cls >= logits.size()) {
            throw std::runtime_error("LoopingRetNetSGDTrainer: target out of range");
        }

        float max_logit = -std::numeric_limits<float>::infinity();
        bool found_finite = false;
        for (arch::Scalar v : logits) {
            const float fv = static_cast<float>(v);
            if (!std::isfinite(fv)) {
                continue;
            }
            max_logit = std::max(max_logit, fv);
            found_finite = true;
        }
        if (!found_finite) {
            return kNonFinitePenalty;
        }

        float sum_exp = 0.0f;
        for (arch::Scalar v : logits) {
            const float fv = static_cast<float>(v);
            if (!std::isfinite(fv)) {
                continue;
            }
            const float shifted = std::clamp(fv - max_logit, -80.0f, 0.0f);
            sum_exp += std::exp(shifted);
        }
        const float target_logit = static_cast<float>(logits[cls]);
        if (!std::isfinite(target_logit) || !std::isfinite(sum_exp) || sum_exp <= 0.0f) {
            return kNonFinitePenalty;
        }

        const float log_prob = target_logit - max_logit - std::log(sum_exp);
        if (!std::isfinite(log_prob)) {
            return kNonFinitePenalty;
        }
        return std::max(0.0f, -log_prob);
    }

    static arch::Vector softmax_ce_grad_from_logits(
        const arch::Vector& logits,
        size_t target_class,
        float& out_ce)
    {
        arch::Vector grad(logits.size(), static_cast<Scalar>(0.0f));
        out_ce = kNonFinitePenalty;
        if (logits.empty() || target_class >= logits.size()) {
            return grad;
        }

        float max_logit = -std::numeric_limits<float>::infinity();
        for (const Scalar v : logits) {
            const float fv = static_cast<float>(v);
            if (!std::isfinite(fv)) {
                continue;
            }
            max_logit = std::max(max_logit, fv);
        }
        if (!std::isfinite(max_logit)) {
            return grad;
        }

        std::vector<float> exps(logits.size(), 0.0f);
        float sum_exp = 0.0f;
        for (size_t i = 0; i < logits.size(); ++i) {
            const float lv = static_cast<float>(logits[i]);
            if (!std::isfinite(lv)) {
                continue;
            }
            const float shifted = std::clamp(lv - max_logit, -80.0f, 0.0f);
            const float e = std::exp(shifted);
            exps[i] = e;
            sum_exp += e;
        }
        if (!std::isfinite(sum_exp) || sum_exp <= 0.0f) {
            return grad;
        }

        for (size_t i = 0; i < logits.size(); ++i) {
            float p = exps[i] / sum_exp;
            if (i == target_class) {
                p -= 1.0f;
            }
            grad[i] = static_cast<Scalar>(p);
        }

        const float target_logit = static_cast<float>(logits[target_class]);
        if (!std::isfinite(target_logit)) {
            return grad;
        }
        const float log_prob = target_logit - max_logit - std::log(sum_exp);
        if (!std::isfinite(log_prob)) {
            return grad;
        }
        out_ce = std::max(0.0f, -log_prob);
        return grad;
    }

    static size_t action_supervision_target_from_logits(
        const arch::Vector& current_output_logits,
        uint8_t target,
        bool enable_query,
        size_t decision_index)
    {
        const float cur_ce = cross_entropy_from_logits(current_output_logits, target);
        if (!std::isfinite(cur_ce)) {
            return static_cast<size_t>(llm::arch::ModelAction::OUTPUT);
        }

        // Use only current-step confidence to avoid leaking future-step information
        // into action supervision targets.
        constexpr float kOutputConfidenceCeThreshold = 0.7f;
        if (cur_ce <= kOutputConfidenceCeThreshold) {
            return static_cast<size_t>(llm::arch::ModelAction::OUTPUT);
        }

        // Prefer QUERY on the first deferred decision when querying is enabled,
        // then LOOP on later deferred decisions.
        if (enable_query && decision_index == 0) {
            return static_cast<size_t>(llm::arch::ModelAction::QUERY_MEMORY);
        }
        return static_cast<size_t>(llm::arch::ModelAction::LOOP);
    }

    static void clip_gradients_abs(std::vector<Scalar>& grads, float max_abs) {
        if (max_abs <= 0.0f) {
            return;
        }
        for (auto& g : grads) {
            const float gf = static_cast<float>(g);
            g = static_cast<Scalar>(std::clamp(gf, -max_abs, max_abs));
        }
    }

    static arch::Vector dgroup_norm(
        const arch::Vector& y,
        const arch::Vector& dout,
        float eps = 1e-5f)
    {
        arch::Vector dx(y.size(), static_cast<Scalar>(0.0f));
        if (y.empty() || y.size() != dout.size()) {
            return dx;
        }

        const size_t n = y.size();
        float mean = 0.0f;
        for (const auto v : y) {
            mean += static_cast<float>(v);
        }
        mean /= static_cast<float>(n);

        float var = 0.0f;
        for (const auto v : y) {
            const float d = static_cast<float>(v) - mean;
            var += d * d;
        }
        var /= static_cast<float>(n);

        const float inv = 1.0f / std::sqrt(var + eps);
        std::vector<float> xhat(n, 0.0f);
        float sum_dout = 0.0f;
        float sum_dout_xhat = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            xhat[i] = (static_cast<float>(y[i]) - mean) * inv;
            const float g = static_cast<float>(dout[i]);
            sum_dout += g;
            sum_dout_xhat += g * xhat[i];
        }

        const float nf = static_cast<float>(n);
        for (size_t i = 0; i < n; ++i) {
            const float g = static_cast<float>(dout[i]);
            const float v = (inv / nf) * (nf * g - sum_dout - xhat[i] * sum_dout_xhat);
            dx[i] = static_cast<Scalar>(v);
        }
        return dx;
    }

    float class_weight_for(size_t target) const {
        if (cfg_.class_weights.empty() || target >= cfg_.class_weights.size()) {
            return 1.0f;
        }
        return std::max(0.0f, cfg_.class_weights[target]);
    }

    float sampled_pt_for(size_t target) const {
        if (sampled_target_pt_.empty() || target >= sampled_target_pt_.size()) {
            return 0.0f;
        }
        return sampled_target_pt_[target];
    }

    void rebuild_sampled_target_pt(
        const std::vector<SequenceExample>& dataset,
        size_t output_dim,
        std::mt19937& rng)
    {
        sampled_target_pt_.assign(output_dim, 0.0f);
        if (dataset.empty() || output_dim == 0) {
            return;
        }

        const float pt_cap = std::clamp(cfg_.min_p_t, 0.0f, 1.0f);
        const size_t sample_count = (cfg_.p_t_sampling_examples == 0)
            ? dataset.size()
            : cfg_.p_t_sampling_examples;

        std::vector<uint64_t> counts(output_dim, 0);
        uint64_t total = 0;

        if (sample_count >= dataset.size()) {
            for (const auto& ex : dataset) {
                const size_t steps = std::min(ex.input.size(), ex.target.size());
                for (size_t i = 0; i < steps; ++i) {
                    const size_t cls = static_cast<size_t>(static_cast<uint8_t>(ex.target[i]));
                    if (cls < output_dim) {
                        ++counts[cls];
                        ++total;
                    }
                }
            }
        } else {
            std::uniform_int_distribution<size_t> idx_dist(0, dataset.size() - 1);
            for (size_t s = 0; s < sample_count; ++s) {
                const auto& ex = dataset[idx_dist(rng)];
                const size_t steps = std::min(ex.input.size(), ex.target.size());
                for (size_t i = 0; i < steps; ++i) {
                    const size_t cls = static_cast<size_t>(static_cast<uint8_t>(ex.target[i]));
                    if (cls < output_dim) {
                        ++counts[cls];
                        ++total;
                    }
                }
            }
        }

        if (total == 0) {
            return;
        }

        const float inv_total = 1.0f / static_cast<float>(total);
        for (size_t c = 0; c < output_dim; ++c) {
            const float prob = static_cast<float>(counts[c]) * inv_total;
            sampled_target_pt_[c] = std::min(prob, pt_cap);
        }
    }

    // Dispatch to the configured output loss type (no gradient).
    float output_loss_from_logits(const arch::Vector& logits, size_t target) const {
        const float ce = cross_entropy_from_logits(logits, static_cast<uint8_t>(target));
        switch (cfg_.output_loss_type) {
            case OutputLossType::WeightedCrossEntropy:
                return class_weight_for(target) * ce;
            case OutputLossType::FocalLoss: {
                if (!std::isfinite(ce) || ce >= kNonFinitePenalty) {
                    return kNonFinitePenalty;
                }
                const float p_t = sampled_pt_for(target);
                const float one_minus_pt = std::max(0.0f, 1.0f - p_t);
                const float gamma = cfg_.focal_gamma;
                const float mod = (gamma > 0.0f) ? std::pow(one_minus_pt, gamma) : 1.0f;
                return class_weight_for(target) * mod * ce;
            }
            default:
                return ce;
        }
    }

    // Compute output loss and its gradient w.r.t. the (param_tanh-scaled) logits.
    // Dispatches to the configured output loss type.
    arch::Vector output_loss_and_grad(
        const arch::Vector& logits, size_t target, float& out_loss) const
    {
        switch (cfg_.output_loss_type) {
            case OutputLossType::WeightedCrossEntropy: {
                float ce = 0.0f;
                arch::Vector grad = softmax_ce_grad_from_logits(logits, target, ce);
                const float w = class_weight_for(target);
                out_loss = w * ce;
                for (auto& g : grad)
                    g = static_cast<Scalar>(static_cast<float>(g) * w);
                return grad;
            }
            case OutputLossType::FocalLoss: {
                float ce = 0.0f;
                arch::Vector grad = softmax_ce_grad_from_logits(logits, target, ce);
                if (!std::isfinite(ce) || ce >= kNonFinitePenalty) {
                    out_loss = kNonFinitePenalty;
                    return grad;
                }
                if (ce <= 0.0f) {
                    out_loss = 0.0f;
                    std::fill(grad.begin(), grad.end(), static_cast<Scalar>(0.0f));
                    return grad;
                }
                const float p_t = sampled_pt_for(target);
                const float gamma = cfg_.focal_gamma;
                const float one_minus_pt = std::max(0.0f, 1.0f - p_t);
                const float focal_mod = (gamma > 0.0f) ? std::pow(one_minus_pt, gamma) : 1.0f;
                out_loss = class_weight_for(target) * focal_mod * ce;
                // With sampled dataset p_t treated as constant per class,
                // focal contributes a constant factor on CE gradients.
                const float scale = class_weight_for(target) * focal_mod;
                for (auto& g : grad)
                    g = static_cast<Scalar>(static_cast<float>(g) * scale);
                return grad;
            }
            default: {
                float ce = 0.0f;
                arch::Vector grad = softmax_ce_grad_from_logits(logits, target, ce);
                out_loss = ce;
                return grad;
            }
        }
    }

    float sequence_loss_with_memory(
        arch::LoopingRetNet& model,
        const SequenceExample& ex,
        std::mt19937& rng,
        memory::GraphMemoryBridge& bridge,
        memory::MultiHopQuery& query,
        size_t epoch_1_based = 0) const {
        if (ex.input.empty() || ex.target.empty()) {
            return 0.0f;
        }

        const size_t steps = std::min(ex.input.size(), ex.target.size());
        if (steps == 0) {
            return 0.0f;
        }
        std::uniform_int_distribution<size_t> start_dist(0, steps - 1);
        const size_t start = start_dist(rng);
        const size_t used_steps = steps;

        const auto& mcfg = cfg_.memory_cfg;
        const arch::LoopConfig& lcfg = model.config();

        if (mcfg.semvec_dim != lcfg.v_dim) {
            throw std::runtime_error("LoopingRetNetSGDTrainer: memory semvec_dim must match model v_dim");
        }

        arch::KVState recurrent_state;
        arch::AttentionMemory kv_cache;

        const bool memory_epoch_active = memory_active_for_epoch(epoch_1_based);
        const bool eff_enable_query = cfg_.enable_query && memory_epoch_active;
        const size_t eff_loop_max_cfg = effective_forced_loop_max_for_epoch(epoch_1_based);

        const size_t loop_min = std::max<size_t>(1, cfg_.forced_loop_min);
        const size_t loop_max = std::max(loop_min,
            std::min<size_t>(eff_loop_max_cfg, model.config().max_steps));
        const float eff_query_prob = eff_enable_query
            ? std::clamp(cfg_.force_query_prob, 0.0f, 1.0f)
            : 0.0f;
        std::uniform_int_distribution<size_t> loop_dist(loop_min, loop_max);
        std::bernoulli_distribution force_query_dist(eff_query_prob);

        float loss = 0.0f;
        arch::Vector prev_step_raw_logits; // Phase 4: multi-step consistency
        for (size_t ti = 0; ti < used_steps; ++ti) {
            const size_t t = (start + ti) % steps;
            const size_t forced_loops = loop_dist(rng);
            const bool force_query_first = eff_enable_query && force_query_dist(rng);
            const bool write_memory = memory_epoch_active
                && (!cfg_.disable_memory_writes_when_query_disabled || eff_enable_query);
            const auto trace = model.step_with_trace(
                ex.input[t],
                recurrent_state,
                kv_cache,
                bridge,
                query,
                cfg_.force_output,
                eff_enable_query,
                forced_loops,
                cfg_.use_parallel_retention,
                write_memory,
                force_query_first);

            loss += output_loss_from_logits(trace.logits, static_cast<size_t>(static_cast<uint8_t>(ex.target[t])));
            if (!std::isfinite(loss)) {
                return kNonFinitePenalty;
            }

            // Train the action decision output: if delaying output helps,
            // supervise QUERY first (when enabled), then LOOP; otherwise OUTPUT.
            if (trace.per_step_output_logits.size() > 1 && trace.per_step_action_logits.size() == trace.per_step_output_logits.size()) {
                for (size_t i = 0; i + 1 < trace.per_step_output_logits.size(); ++i) {
                    const size_t target = action_supervision_target_from_logits(
                        trace.per_step_output_logits[i],
                        static_cast<uint8_t>(ex.target[t]),
                        eff_enable_query,
                        i);
                    loss += cfg_.loop_supervision_weight
                        * action_ce_from_logits(trace.per_step_action_logits[i], target);
                    if (!std::isfinite(loss)) {
                        return kNonFinitePenalty;
                    }
                }
            }

            if (eff_enable_query && trace.query_count == 0) {
                loss += cfg_.memory_query_penalty;
            }

            if (!trace.per_step_load_logits.empty() && trace.per_step_action_logits.size() == trace.per_step_output_logits.size()) {
                const size_t load_steps = std::min(trace.per_step_load_logits.size(), trace.per_step_output_logits.size());
                for (size_t li = 0; li < load_steps; ++li) {
                    const size_t target_action = action_supervision_target_from_logits(
                        trace.per_step_output_logits[li],
                        static_cast<uint8_t>(ex.target[t]),
                        eff_enable_query,
                        li);
                    const float target_load = (target_action == static_cast<size_t>(llm::arch::ModelAction::QUERY_MEMORY))
                        ? 1.0f
                        : 0.0f;
                    const auto [load_ce, _] = binary_ce_and_grad(
                        static_cast<float>(trace.per_step_load_logits[li]),
                        target_load);
                    loss += cfg_.load_gate_supervision_weight * load_ce;
                    if (!std::isfinite(loss)) {
                        return kNonFinitePenalty;
                    }
                }
            }

            if (eff_enable_query && !kv_cache.keys.empty()) {
                const arch::AttentionMemory queried = query.query(kv_cache.keys.back());
                if (queried.empty()) {
                    const float miss_scale = 1.0f + 0.1f * std::min<float>(10.0f, static_cast<float>(kv_cache.keys.size()));
                    loss += cfg_.memory_miss_penalty * miss_scale;
                } else {
                    // Robust memory signal: queried values should align with
                    // the latest value in cache.
                    const arch::Vector& target_value = kv_cache.values.back();
                    float best_sim = -1.0f;
                    for (const auto& qv : queried.values) {
                        best_sim = std::max(best_sim, cosine_similarity(qv, target_value));
                    }
                    loss += cfg_.memory_alignment_weight * (1.0f - best_sim);
                }
            }

            // Prevent unchecked KV-cache growth from dominating compute.
            const float allowed_edges = static_cast<float>(cfg_.memory_cfg.max_write_entries * (ti + 1));
            const float actual_edges = static_cast<float>(kv_cache.keys.size());
            if (actual_edges > allowed_edges && allowed_edges > 0.0f) {
                const float over_ratio = (actual_edges - allowed_edges) / allowed_edges;
                loss += cfg_.memory_edge_budget_penalty * over_ratio;
                if (!std::isfinite(loss)) {
                    return kNonFinitePenalty;
                }
            }

            // Phase 4: multi-step consistency — penalise drift in raw output logits
            // between consecutive sequence positions.
            if (cfg_.multistep_consistency_weight > 0.0f
                && !prev_step_raw_logits.empty()
                && !trace.raw_logits.empty()
                && trace.raw_logits.size() == prev_step_raw_logits.size()) {
                const size_t D = trace.raw_logits.size();
                float cons = 0.0f;
                for (size_t o = 0; o < D; ++o) {
                    const float diff = static_cast<float>(trace.raw_logits[o])
                        - static_cast<float>(prev_step_raw_logits[o]);
                    cons += diff * diff;
                }
                loss += cfg_.multistep_consistency_weight * cons / static_cast<float>(D);
                if (!std::isfinite(loss)) {
                    return kNonFinitePenalty;
                }
            }
            prev_step_raw_logits = trace.raw_logits;
        }

        const float avg = loss / static_cast<float>(used_steps);
        if (!std::isfinite(avg)) {
            return kNonFinitePenalty;
        }
        return avg;
    }

    float sequence_loss(
        arch::LoopingRetNet& model,
        const SequenceExample& ex,
        std::mt19937& rng) const {
        const auto& mcfg = cfg_.memory_cfg;
        const arch::LoopConfig& lcfg = model.config();
        if (mcfg.semvec_dim != lcfg.v_dim) {
            throw std::runtime_error("LoopingRetNetSGDTrainer: memory semvec_dim must match model v_dim");
        }

        Graph graph;
        SpatialMap spatial_map;
        memory::NodeCompressor compressor(lcfg.v_dim, mcfg.semvec_dim, cfg_.seed);
        memory::GraphMemoryBridge bridge(graph, spatial_map, std::move(compressor), mcfg);
        memory::MultiHopQuery query(graph, spatial_map, mcfg);
        return sequence_loss_with_memory(model, ex, rng, bridge, query);
    }

    float dataset_loss(
        arch::LoopingRetNet& model,
        const std::vector<SequenceExample>& dataset,
        std::mt19937& rng,
        size_t epoch_1_based = 0) const {
        if (dataset.empty()) {
            return 0.0f;
        }

        const auto& mcfg = cfg_.memory_cfg;
        const arch::LoopConfig& lcfg = model.config();
        if (mcfg.semvec_dim != lcfg.v_dim) {
            throw std::runtime_error("LoopingRetNetSGDTrainer: memory semvec_dim must match model v_dim");
        }

        // Keep a persistent graph for the whole dataset pass so long-term memory
        // supervision reflects cross-example accumulation.
        Graph graph;
        SpatialMap spatial_map;
        memory::NodeCompressor compressor(lcfg.v_dim, mcfg.semvec_dim, cfg_.seed);
        memory::GraphMemoryBridge bridge(graph, spatial_map, std::move(compressor), mcfg);
        memory::MultiHopQuery query(graph, spatial_map, mcfg);

        float total_loss = 0.0f;
        for (const auto& ex : dataset) {
            const float loss = sequence_loss_with_memory(model, ex, rng, bridge, query, epoch_1_based);
            if (!std::isfinite(loss)) {
                return kNonFinitePenalty;
            }
            total_loss += loss;
        }
        return total_loss / static_cast<float>(dataset.size());
    }

public:
    explicit LoopingRetNetSGDTrainer(TrainConfig cfg) : cfg_(std::move(cfg)) {}

    std::vector<EpochResult> train(
        arch::LoopingRetNet& model,
        const std::vector<SequenceExample>& dataset)
    {
        if (dataset.empty()) {
            throw std::runtime_error("LoopingRetNetSGDTrainer: dataset is empty");
        }
        if (cfg_.fd_eps <= 0.0f) {
            throw std::runtime_error("LoopingRetNetSGDTrainer: fd_eps must be > 0");
        }
        if (cfg_.grad_coordinate_samples == 0) {
            throw std::runtime_error("LoopingRetNetSGDTrainer: grad_coordinate_samples must be > 0");
        }
        if (cfg_.batch_size == 0) {
            throw std::runtime_error("LoopingRetNetSGDTrainer: batch_size must be > 0");
        }

        std::mt19937 rng(cfg_.seed);
        rebuild_sampled_target_pt(dataset, model.output_dim(), rng);
        Adam optimizer(learning_rate_for_epoch(1), 0.9f, 0.999f, 1e-8f, cfg_.weight_decay);
        float lr_multiplier = 1.0f;
        size_t lr_cooldown_left = 0;
        float prev_epoch_loss = 0.0f;
        bool has_prev_epoch_loss = false;

        std::vector<EpochResult> history;
        history.reserve(cfg_.epochs);
        float loss_ema = 0.0f;
        bool has_ema = false;
        const float ema_beta = std::clamp(cfg_.loss_ema_beta, 0.0f, 0.9999f);

        // Phase 3: rolling window of recent losses for adaptive instability detection.
        std::vector<float> recent_losses;
        if (cfg_.adaptive_instability_window > 0) {
            recent_losses.reserve(cfg_.adaptive_instability_window + 1);
        }

        auto last_print_time = std::chrono::steady_clock::now();
        float inter_loss_delta = 0.0f;
        bool inter_unstable = false;

        for (size_t epoch = 0; epoch < cfg_.epochs; ++epoch) {
            const auto epoch_t0 = std::chrono::steady_clock::now();
            llm::arch::sycl_ops::reset_kernel_launch_counter();

            const float base_lr_epoch = learning_rate_for_epoch(epoch + 1);
            const float lr_epoch = std::max(
                cfg_.min_effective_learning_rate,
                base_lr_epoch * std::max(0.0f, lr_multiplier));
            optimizer.set_learning_rate(lr_epoch);
            double fd_ms = 0.0;
            size_t grad_samples_epoch = 0;
            float active_grad_fraction = 0.0f;
            size_t nonfinite_skips = 0;
            size_t repaired = 0;
            float epoch_loss = 0.0f;
            size_t epoch_token_count = 0;
            size_t epoch_example_count = 0;
            size_t objective_evals = 0;
            size_t query_events = 0;
            double seq_forward_ms = 0.0;
            float gradcheck_rel_err_sum = 0.0f;
            size_t gradcheck_count = 0;

            // Build and shuffle a per-epoch dataset view. With samples_per_epoch==0,
            // use all examples; otherwise sample with replacement.
            const size_t effective_samples_per_epoch =
                (cfg_.samples_per_epoch == 0) ? dataset.size() : cfg_.samples_per_epoch;

            std::vector<SequenceExample> epoch_dataset_storage;
            epoch_dataset_storage.reserve(effective_samples_per_epoch);
            if (effective_samples_per_epoch >= dataset.size()) {
                epoch_dataset_storage = dataset;
            } else {
                std::uniform_int_distribution<size_t> idx_dist(0, dataset.size() - 1);
                for (size_t s = 0; s < effective_samples_per_epoch; ++s) {
                    epoch_dataset_storage.push_back(dataset[idx_dist(rng)]);
                }
            }
            std::shuffle(epoch_dataset_storage.begin(), epoch_dataset_storage.end(), rng);
            const std::vector<SequenceExample>& epoch_dataset = epoch_dataset_storage;

            const char* mode_name_ep = (cfg_.mode == TrainMode::FiniteDifference)
                ? "FD"
                : (cfg_.mode == TrainMode::BackpropHeads ? "BP_HEADS" : "BP_FULL");

            auto maybe_print = [&](float cur_loss, size_t ep_done, size_t ep_total) {
                const auto now = std::chrono::steady_clock::now();
                const double since_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(now - last_print_time).count();
                if (since_ms < 100.0) return;
                last_print_time = now;
                const double elapsed_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(now - epoch_t0).count();
                const double elapsed_s = std::max(1e-6, elapsed_ms / 1000.0);
                const double cur_tok_s = static_cast<double>(epoch_token_count) / elapsed_s;
                const double cur_ex_s = static_cast<double>(epoch_example_count) / elapsed_s;
                const uint64_t cur_kernels = llm::arch::sycl_ops::kernel_launch_count();
                const float cur_gradcheck_err = (gradcheck_count == 0)
                    ? 0.0f
                    : (gradcheck_rel_err_sum / static_cast<float>(gradcheck_count));
                const float cur_skip_ratio = (grad_samples_epoch > 0)
                    ? static_cast<float>(nonfinite_skips) / static_cast<float>(grad_samples_epoch)
                    : 0.0f;
                std::cout
                    << "Epoch " << (epoch + 1) << "/" << cfg_.epochs
                    << " - Prog: " << ep_done << "/" << ep_total
                    << " (" << (ep_total > 0 ? static_cast<int>(100.0 * ep_done / ep_total) : 0) << "%)"
                    << " - Mode: " << mode_name_ep
                    << " - LR: " << lr_epoch
                    << " - BaseLR: " << base_lr_epoch
                    << " - LRScale: " << lr_multiplier
                    << " - FDsamples: " << grad_samples_epoch
                    << " - Loss: " << cur_loss
                    << " - LossDelta: " << inter_loss_delta
                    << " - LossEMA: " << loss_ema
                    << " - FDms: " << fd_ms
                    << " - SeqMs: " << seq_forward_ms
                    << " - ObjEvals: " << objective_evals
                    << " - QueryEvents: " << query_events
                    << " - EpochMs: " << elapsed_ms
                    << " - Tok/s: " << cur_tok_s
                    << " - Ex/s: " << cur_ex_s
                    << " - Kernels: " << cur_kernels
                    << " - GradCheckRelErr: " << cur_gradcheck_err
                    << " - ActiveGrad%: " << (100.0f * active_grad_fraction)
                    << " - NonFiniteSkips: " << nonfinite_skips
                    << " - Skip%: " << (100.0f * cur_skip_ratio);
                if (repaired > 0) {
                    std::cout << " - Repaired: " << repaired;
                }
                if (inter_unstable) {
                    std::cout << " - LRBackoff";
                }
                std::cout << "\n";
            };

            if (cfg_.mode == TrainMode::FiniteDifference) {
                grad_samples_epoch = grad_samples_for_epoch(epoch + 1);
                std::vector<Scalar*> refs = model.parameter_references();
                std::vector<Scalar> base(refs.size(), static_cast<Scalar>(0.0f));
                for (size_t i = 0; i < refs.size(); ++i) {
                    base[i] = *refs[i];
                }

                ParameterTensor tensor;
                tensor.values = base;
                tensor.grads.assign(base.size(), static_cast<Scalar>(0.0f));
                std::vector<size_t> grad_hits(base.size(), 0);

                const std::vector<size_t> coord_indices = stratified_coordinate_indices(
                    base.size(), grad_samples_epoch, rng);
                const auto fd_t0 = std::chrono::steady_clock::now();

                for (size_t s = 0; s < coord_indices.size(); ++s) {
                    const size_t idx = coord_indices[s];
                    const Scalar original = *refs[idx];

                    const uint32_t seed_base = rng();
                    std::mt19937 rng_plus(seed_base ^ 0x9e3779b9u);
                    std::mt19937 rng_minus(seed_base ^ 0x85ebca6bu);

                    *refs[idx] = static_cast<Scalar>(static_cast<float>(original) + cfg_.fd_eps);
                    const auto eval_t0 = std::chrono::steady_clock::now();
                    const float l_plus = dataset_loss(model, epoch_dataset, rng_plus, epoch + 1);
                    const auto eval_t1 = std::chrono::steady_clock::now();
                    seq_forward_ms += std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(eval_t1 - eval_t0).count();
                    ++objective_evals;

                    *refs[idx] = static_cast<Scalar>(static_cast<float>(original) - cfg_.fd_eps);
                    const auto eval_t2 = std::chrono::steady_clock::now();
                    const float l_minus = dataset_loss(model, epoch_dataset, rng_minus, epoch + 1);
                    const auto eval_t3 = std::chrono::steady_clock::now();
                    seq_forward_ms += std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(eval_t3 - eval_t2).count();
                    ++objective_evals;

                    *refs[idx] = original;

                    if (!std::isfinite(l_plus) || !std::isfinite(l_minus)) {
                        ++nonfinite_skips;
                        continue;
                    }

                    const float g = (l_plus - l_minus) / (2.0f * cfg_.fd_eps);
                    if (!std::isfinite(g)) {
                        ++nonfinite_skips;
                        continue;
                    }
                    const float g_clipped = std::clamp(g, -cfg_.max_gradient_abs, cfg_.max_gradient_abs);
                    tensor.grads[idx] = static_cast<Scalar>(
                        static_cast<float>(tensor.grads[idx]) + g_clipped);
                    ++grad_hits[idx];
                    maybe_print(epoch_loss, s + 1, coord_indices.size());
                }

                for (size_t i = 0; i < tensor.grads.size(); ++i) {
                    if (grad_hits[i] == 0) {
                        continue;
                    }
                    const float g_sum = static_cast<float>(tensor.grads[i]);
                    const float g_avg = g_sum / static_cast<float>(grad_hits[i]);
                    tensor.grads[i] = static_cast<Scalar>(g_avg);
                }

                const auto fd_t1 = std::chrono::steady_clock::now();
                fd_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(fd_t1 - fd_t0).count();

                size_t active_grad_coords = 0;
                for (size_t i = 0; i < grad_hits.size(); ++i) {
                    if (grad_hits[i] > 0) {
                        ++active_grad_coords;
                    }
                }
                active_grad_fraction = grad_hits.empty()
                    ? 0.0f
                    : static_cast<float>(active_grad_coords) / static_cast<float>(grad_hits.size());

                std::vector<ParameterTensor> params{tensor};
                optimizer.step(params);

                for (size_t i = 0; i < refs.size(); ++i) {
                    const float updated = static_cast<float>(params[0].values[i]);
                    float sanitized = updated;
                    if (!std::isfinite(updated)) {
                        sanitized = static_cast<float>(base[i]);
                        ++repaired;
                    }
                    sanitized = std::clamp(sanitized, -cfg_.max_parameter_abs, cfg_.max_parameter_abs);
                    *refs[i] = static_cast<Scalar>(sanitized);
                }

                const auto eval_epoch_t0 = std::chrono::steady_clock::now();
                epoch_loss = dataset_loss(model, epoch_dataset, rng, epoch + 1);
                const auto eval_epoch_t1 = std::chrono::steady_clock::now();
                seq_forward_ms += std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(eval_epoch_t1 - eval_epoch_t0).count();
                ++objective_evals;
                for (const auto& ex : epoch_dataset) {
                    epoch_example_count += 1;
                    epoch_token_count += std::min(ex.input.size(), ex.target.size());
                }
            } else if (cfg_.mode == TrainMode::BackpropHeads) {
                // Backprop mode: analytic gradients for output/action heads only.
                // This is deterministic by default and avoids finite-difference cost.
                auto& out_w = model.output_head_weights();
                auto& out_b = model.output_head_bias();
                auto& act_w = model.action_head_weights();
                auto& act_b = model.action_head_bias();
                auto& load_w = model.load_gate_weights();
                auto& load_b = model.load_gate_bias();

                const size_t out_in = model.output_input_dim();
                const size_t out_dim = model.output_dim();
                const size_t act_in = model.action_input_dim();
                const size_t act_dim = model.action_dim();
                const size_t load_in = model.load_gate_input_dim();

                const auto bp_t0 = std::chrono::steady_clock::now();
                float epoch_loss_acc = 0.0f;

                for (size_t batch_start = 0; batch_start < epoch_dataset.size(); batch_start += cfg_.batch_size) {
                    const size_t batch_end = std::min(batch_start + cfg_.batch_size, epoch_dataset.size());

                    ParameterTensor p_out_w;
                    p_out_w.values = out_w;
                    p_out_w.grads.assign(out_w.size(), static_cast<Scalar>(0.0f));

                    ParameterTensor p_out_b;
                    p_out_b.values = out_b;
                    p_out_b.grads.assign(out_b.size(), static_cast<Scalar>(0.0f));

                    ParameterTensor p_theta;
                    p_theta.values = {model.output_theta()};
                    p_theta.grads = {static_cast<Scalar>(0.0f)};

                    ParameterTensor p_act_w;
                    ParameterTensor p_act_b;
                    if (cfg_.backprop_include_loop_supervision) {
                        p_act_w.values = act_w;
                        p_act_w.grads.assign(act_w.size(), static_cast<Scalar>(0.0f));
                        p_act_b.values = act_b;
                        p_act_b.grads.assign(act_b.size(), static_cast<Scalar>(0.0f));
                    }

                    // so the gate receives gradient even when querying is disabled.
                    ParameterTensor p_load_w;
                    p_load_w.values = load_w;
                    p_load_w.grads.assign(load_w.size(), static_cast<Scalar>(0.0f));
                    ParameterTensor p_load_b;
                    p_load_b.values = load_b;
                    p_load_b.grads.assign(load_b.size(), static_cast<Scalar>(0.0f));

                    const size_t batch_examples = batch_end - batch_start;
                    const bool use_parallel_examples = (effective_host_threads() > 1)
                        && (batch_examples >= cfg_.min_parallel_batch_examples);
                    const size_t example_threads = use_parallel_examples
                        ? std::min(effective_host_threads(), batch_examples)
                        : 1;

                    size_t batch_tokens = 0;
                    std::mutex accum_mutex;
                    auto add_grad_vec = [](std::vector<Scalar>& dst, const std::vector<Scalar>& src) {
                        if (dst.size() != src.size()) {
                            return;
                        }
                        for (size_t i = 0; i < dst.size(); ++i) {
                            dst[i] = static_cast<Scalar>(
                                static_cast<float>(dst[i]) + static_cast<float>(src[i]));
                        }
                    };

                    parallel_for_ranges(
                        batch_start,
                        batch_end,
                        example_threads,
                        [&](size_t ex_begin, size_t ex_end, size_t /*worker_id*/) {
                            std::vector<Scalar> local_out_w_grads(out_w.size(), static_cast<Scalar>(0.0f));
                            std::vector<Scalar> local_out_b_grads(out_b.size(), static_cast<Scalar>(0.0f));
                            Scalar local_theta_grad = static_cast<Scalar>(0.0f);
                            std::vector<Scalar> local_act_w_grads(
                                cfg_.backprop_include_loop_supervision ? act_w.size() : 0,
                                static_cast<Scalar>(0.0f));
                            std::vector<Scalar> local_act_b_grads(
                                cfg_.backprop_include_loop_supervision ? act_b.size() : 0,
                                static_cast<Scalar>(0.0f));
                            std::vector<Scalar> local_load_w_grads(load_w.size(), static_cast<Scalar>(0.0f));
                            std::vector<Scalar> local_load_b_grads(load_b.size(), static_cast<Scalar>(0.0f));

                            float local_epoch_loss_acc = 0.0f;
                            size_t local_batch_tokens = 0;
                            size_t local_epoch_token_count = 0;
                            size_t local_epoch_example_count = 0;
                            size_t local_nonfinite_skips = 0;
                            size_t local_query_events = 0;
                            size_t local_objective_evals = 0;
                            double local_seq_forward_ms = 0.0;

                            for (size_t exi = ex_begin; exi < ex_end; ++exi) {
                                const auto& ex = epoch_dataset[exi];
                                if (ex.input.empty() || ex.target.empty()) {
                                    continue;
                                }

                                const size_t steps = std::min(ex.input.size(), ex.target.size());
                                if (steps == 0) {
                                    continue;
                                }

                                std::mt19937 ex_rng(per_example_seed(cfg_.seed, epoch + 1, batch_start, exi));
                                std::uniform_int_distribution<size_t> start_dist(0, steps - 1);
                                const size_t start = start_dist(ex_rng);
                                std::bernoulli_distribution force_query_dist(
                                    std::clamp(cfg_.force_query_prob, 0.0f, 1.0f));

                                Graph graph;
                                SpatialMap spatial_map;
                                memory::NodeCompressor compressor(model.config().v_dim, cfg_.memory_cfg.semvec_dim, cfg_.seed);
                                memory::GraphMemoryBridge bridge(graph, spatial_map, std::move(compressor), cfg_.memory_cfg);
                                memory::MultiHopQuery query(graph, spatial_map, cfg_.memory_cfg);

                                arch::KVState recurrent_state;
                                arch::AttentionMemory kv_cache;

                                arch::Vector bp_prev_raw_logits; // Phase 4: multi-step consistency
                                for (size_t ti = 0; ti < steps; ++ti) {
                                    const size_t t = (start + ti) % steps;

                                    const bool bp_memory_active = memory_active_for_epoch(epoch + 1);
                                    const bool use_query = cfg_.backprop_force_single_step
                                        ? false
                                        : (cfg_.enable_query && bp_memory_active);
                                    const bool force_output = cfg_.backprop_force_single_step ? true : cfg_.force_output;
                                    const size_t forced_loops = cfg_.backprop_force_single_step
                                        ? 1
                                        : std::max<size_t>(1, cfg_.forced_loop_min);
                                    const bool write_memory = !cfg_.disable_memory_writes_when_query_disabled || use_query;
                                    const bool force_query_first = use_query && force_query_dist(ex_rng);

                                    const auto trace_t0 = std::chrono::steady_clock::now();
                                    const auto trace = model.step_with_trace(
                                        ex.input[t],
                                        recurrent_state,
                                        kv_cache,
                                        bridge,
                                        query,
                                        force_output,
                                        use_query,
                                        forced_loops,
                                        false,
                                        write_memory,
                                        force_query_first);
                                    const auto trace_t1 = std::chrono::steady_clock::now();
                                    local_seq_forward_ms += std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(trace_t1 - trace_t0).count();
                                    local_query_events += trace.query_count;
                                    ++local_objective_evals;

                                    float out_ce = 0.0f;
                                    arch::Vector d_logits = output_loss_and_grad(
                                        trace.logits,
                                        static_cast<size_t>(static_cast<uint8_t>(ex.target[t])),
                                        out_ce);
                                    if (!std::isfinite(out_ce)) {
                                        ++local_nonfinite_skips;
                                        continue;
                                    }
                                    local_epoch_loss_acc += out_ce;

                                    const auto tanh_grad = arch::activation::dparam_tanh(
                                        trace.raw_logits,
                                        model.output_theta(),
                                        d_logits);

                                    arch::Vector d_raw_combined = tanh_grad.dx;
                                    if (cfg_.multistep_consistency_weight > 0.0f
                                        && !bp_prev_raw_logits.empty()
                                        && !trace.raw_logits.empty()
                                        && trace.raw_logits.size() == bp_prev_raw_logits.size()) {
                                        const size_t D = trace.raw_logits.size();
                                        const float w_norm = cfg_.multistep_consistency_weight * 2.0f
                                            / static_cast<float>(D);
                                        float cons = 0.0f;
                                        for (size_t o = 0; o < D; ++o) {
                                            const float diff = static_cast<float>(trace.raw_logits[o])
                                                - static_cast<float>(bp_prev_raw_logits[o]);
                                            cons += diff * diff;
                                            d_raw_combined[o] = static_cast<Scalar>(
                                                static_cast<float>(d_raw_combined[o]) + w_norm * diff);
                                        }
                                        local_epoch_loss_acc += cfg_.multistep_consistency_weight
                                            * cons / static_cast<float>(D);
                                    }
                                    bp_prev_raw_logits = trace.raw_logits;

                                    if (cfg_.backprop_fd_check_samples > 0
                                        && !use_parallel_examples
                                        && gradcheck_count < cfg_.backprop_fd_check_samples
                                        && !trace.final_state.empty()
                                        && !trace.raw_logits.empty()) {
                                        const size_t o = gradcheck_count % out_dim;
                                        const size_t i = (gradcheck_count * 17) % out_in;
                                        const size_t wi = o * out_in + i;

                                        const float analytic = static_cast<float>(tanh_grad.dx[o])
                                            * static_cast<float>(trace.final_state[i]);

                                        auto ce_with_weight = [&](float wv) {
                                            arch::Vector raw(out_dim, static_cast<Scalar>(0.0f));
                                            for (size_t oo = 0; oo < out_dim; ++oo) {
                                                float s = static_cast<float>(out_b[oo]);
                                                for (size_t ii = 0; ii < out_in; ++ii) {
                                                    const size_t wii = oo * out_in + ii;
                                                    const float ww = (wii == wi) ? wv : static_cast<float>(out_w[wii]);
                                                    s += ww * static_cast<float>(trace.final_state[ii]);
                                                }
                                                raw[oo] = static_cast<Scalar>(s);
                                            }
                                            const arch::Vector logits = arch::activation::param_tanh(raw, model.output_theta());
                                            return output_loss_from_logits(
                                                logits,
                                                static_cast<size_t>(static_cast<uint8_t>(ex.target[t])));
                                        };

                                        const float eps = std::max(1e-5f, cfg_.backprop_fd_check_eps);
                                        const float base_w = static_cast<float>(out_w[wi]);
                                        const float l_plus = ce_with_weight(base_w + eps);
                                        const float l_minus = ce_with_weight(base_w - eps);
                                        if (std::isfinite(l_plus) && std::isfinite(l_minus)) {
                                            const float fd = (l_plus - l_minus) / (2.0f * eps);
                                            const float denom = std::max(1e-5f, std::fabs(fd));
                                            gradcheck_rel_err_sum += std::fabs(analytic - fd) / denom;
                                            ++gradcheck_count;
                                        }
                                    }

                                    for (size_t o = 0; o < out_dim; ++o) {
                                        const float d_raw = static_cast<float>(d_raw_combined[o]);
                                        if (!std::isfinite(d_raw)) {
                                            ++local_nonfinite_skips;
                                            continue;
                                        }
                                        local_out_b_grads[o] = static_cast<Scalar>(
                                            static_cast<float>(local_out_b_grads[o]) + d_raw);
                                        for (size_t i = 0; i < out_in; ++i) {
                                            const float s = static_cast<float>(trace.final_state[i]);
                                            const size_t wi = o * out_in + i;
                                            local_out_w_grads[wi] = static_cast<Scalar>(
                                                static_cast<float>(local_out_w_grads[wi]) + d_raw * s);
                                        }
                                    }
                                    local_theta_grad = static_cast<Scalar>(
                                        static_cast<float>(local_theta_grad) + static_cast<float>(tanh_grad.dtheta));

                                    if (cfg_.backprop_include_loop_supervision
                                        && trace.per_step_output_logits.size() > 1
                                        && trace.per_step_action_logits.size() == trace.per_step_states.size()) {
                                        for (size_t si = 0; si + 1 < trace.per_step_output_logits.size(); ++si) {
                                            const size_t target_action = action_supervision_target_from_logits(
                                                trace.per_step_output_logits[si],
                                                static_cast<uint8_t>(ex.target[t]),
                                                use_query,
                                                si);
                                            float action_ce = 0.0f;
                                            arch::Vector d_action = softmax_ce_grad_from_logits(
                                                trace.per_step_action_logits[si],
                                                target_action,
                                                action_ce);
                                            if (!std::isfinite(action_ce)) {
                                                ++local_nonfinite_skips;
                                                continue;
                                            }
                                            local_epoch_loss_acc += cfg_.loop_supervision_weight * action_ce;

                                            const auto& state = trace.per_step_states[si];
                                            for (size_t a = 0; a < act_dim; ++a) {
                                                const float dag = cfg_.loop_supervision_weight * static_cast<float>(d_action[a]);
                                                local_act_b_grads[a] = static_cast<Scalar>(
                                                    static_cast<float>(local_act_b_grads[a]) + dag);
                                                for (size_t i = 0; i < act_in; ++i) {
                                                    const size_t wi = a * act_in + i;
                                                    local_act_w_grads[wi] = static_cast<Scalar>(
                                                        static_cast<float>(local_act_w_grads[wi]) + dag * static_cast<float>(state[i]));
                                                }
                                            }
                                        }
                                    }

                                    if (trace.per_step_load_logits.size() == trace.per_step_states.size()
                                        && trace.per_step_output_logits.size() == trace.per_step_states.size()) {
                                        for (size_t si = 0; si < trace.per_step_states.size(); ++si) {
                                            const size_t target_action = action_supervision_target_from_logits(
                                                trace.per_step_output_logits[si],
                                                static_cast<uint8_t>(ex.target[t]),
                                                use_query,
                                                si);
                                            const float target_load =
                                                (target_action == static_cast<size_t>(llm::arch::ModelAction::QUERY_MEMORY))
                                                ? 1.0f
                                                : 0.0f;
                                            const auto [load_ce, d_logit] = binary_ce_and_grad(
                                                static_cast<float>(trace.per_step_load_logits[si]),
                                                target_load);
                                            if (!std::isfinite(load_ce) || !std::isfinite(d_logit)) {
                                                ++local_nonfinite_skips;
                                                continue;
                                            }
                                            local_epoch_loss_acc += cfg_.load_gate_supervision_weight * load_ce;
                                            const float d = cfg_.load_gate_supervision_weight * d_logit;
                                            local_load_b_grads[0] = static_cast<Scalar>(
                                                static_cast<float>(local_load_b_grads[0]) + d);
                                            const auto& state = trace.per_step_states[si];
                                            for (size_t i = 0; i < load_in; ++i) {
                                                local_load_w_grads[i] = static_cast<Scalar>(
                                                    static_cast<float>(local_load_w_grads[i]) + d * static_cast<float>(state[i]));
                                            }
                                        }
                                    }

                                    ++local_batch_tokens;
                                    ++local_epoch_token_count;
                                }
                                ++local_epoch_example_count;
                            }

                            std::lock_guard<std::mutex> lock(accum_mutex);
                            add_grad_vec(p_out_w.grads, local_out_w_grads);
                            add_grad_vec(p_out_b.grads, local_out_b_grads);
                            p_theta.grads[0] = static_cast<Scalar>(
                                static_cast<float>(p_theta.grads[0]) + static_cast<float>(local_theta_grad));
                            if (cfg_.backprop_include_loop_supervision) {
                                add_grad_vec(p_act_w.grads, local_act_w_grads);
                                add_grad_vec(p_act_b.grads, local_act_b_grads);
                            }
                            add_grad_vec(p_load_w.grads, local_load_w_grads);
                            add_grad_vec(p_load_b.grads, local_load_b_grads);

                            epoch_loss_acc += local_epoch_loss_acc;
                            batch_tokens += local_batch_tokens;
                            epoch_token_count += local_epoch_token_count;
                            epoch_example_count += local_epoch_example_count;
                            nonfinite_skips += local_nonfinite_skips;
                            query_events += local_query_events;
                            objective_evals += local_objective_evals;
                            seq_forward_ms += local_seq_forward_ms;
                        });

                    if (batch_tokens == 0) {
                        continue;
                    }

                    const float inv = 1.0f / static_cast<float>(batch_tokens);
                    for (auto& g : p_out_w.grads) {
                        g = static_cast<Scalar>(static_cast<float>(g) * inv);
                    }
                    for (auto& g : p_out_b.grads) {
                        g = static_cast<Scalar>(static_cast<float>(g) * inv);
                    }
                    p_theta.grads[0] = static_cast<Scalar>(static_cast<float>(p_theta.grads[0]) * inv);
                    if (cfg_.backprop_include_loop_supervision) {
                        for (auto& g : p_act_w.grads) {
                            g = static_cast<Scalar>(static_cast<float>(g) * inv);
                        }
                        for (auto& g : p_act_b.grads) {
                            g = static_cast<Scalar>(static_cast<float>(g) * inv);
                        }
                    }

                    for (auto& g : p_load_w.grads) {
                        g = static_cast<Scalar>(static_cast<float>(g) * inv);
                    }
                    for (auto& g : p_load_b.grads) {
                        g = static_cast<Scalar>(static_cast<float>(g) * inv);
                    }

                    clip_gradients_abs(p_out_w.grads, cfg_.max_gradient_abs);
                    clip_gradients_abs(p_out_b.grads, cfg_.max_gradient_abs);
                    clip_gradients_abs(p_act_w.grads, cfg_.max_gradient_abs);
                    clip_gradients_abs(p_act_b.grads, cfg_.max_gradient_abs);
                    clip_gradients_abs(p_load_w.grads, cfg_.max_gradient_abs);
                    clip_gradients_abs(p_load_b.grads, cfg_.max_gradient_abs);
                    clip_gradients_abs(p_theta.grads, cfg_.max_gradient_abs);

                    std::vector<ParameterTensor> params;
                    params.push_back(p_out_w);
                    params.push_back(p_out_b);
                    params.push_back(p_theta);
                    if (cfg_.backprop_include_loop_supervision) {
                        params.push_back(p_act_w);
                        params.push_back(p_act_b);
                    }

                    params.push_back(p_load_w);
                    params.push_back(p_load_b);
                    optimizer.step(params);

                    p_out_w = params[0];
                    p_out_b = params[1];
                    p_theta = params[2];
                    size_t p_idx = 3;
                    if (cfg_.backprop_include_loop_supervision) {
                        p_act_w = params[p_idx++];
                        p_act_b = params[p_idx++];
                    }

                    p_load_w = params[p_idx++];
                    p_load_b = params[p_idx++];

                    for (size_t i = 0; i < out_w.size(); ++i) {
                        const float v = static_cast<float>(p_out_w.values[i]);
                        if (!std::isfinite(v)) {
                            ++repaired;
                            continue;
                        }
                        out_w[i] = static_cast<Scalar>(std::clamp(v, -cfg_.max_parameter_abs, cfg_.max_parameter_abs));
                    }
                    for (size_t i = 0; i < out_b.size(); ++i) {
                        const float v = static_cast<float>(p_out_b.values[i]);
                        if (!std::isfinite(v)) {
                            ++repaired;
                            continue;
                        }
                        out_b[i] = static_cast<Scalar>(std::clamp(v, -cfg_.max_parameter_abs, cfg_.max_parameter_abs));
                    }
                    {
                        const float v = static_cast<float>(p_theta.values[0]);
                        if (std::isfinite(v)) {
                            model.set_output_theta(static_cast<Scalar>(std::clamp(v, -cfg_.max_parameter_abs, cfg_.max_parameter_abs)));
                        } else {
                            ++repaired;
                        }
                    }
                    if (cfg_.backprop_include_loop_supervision) {
                        for (size_t i = 0; i < act_w.size(); ++i) {
                            const float v = static_cast<float>(p_act_w.values[i]);
                            if (!std::isfinite(v)) {
                                ++repaired;
                                continue;
                            }
                            act_w[i] = static_cast<Scalar>(std::clamp(v, -cfg_.max_parameter_abs, cfg_.max_parameter_abs));
                        }
                        for (size_t i = 0; i < act_b.size(); ++i) {
                            const float v = static_cast<float>(p_act_b.values[i]);
                            if (!std::isfinite(v)) {
                                ++repaired;
                                continue;
                            }
                            act_b[i] = static_cast<Scalar>(std::clamp(v, -cfg_.max_parameter_abs, cfg_.max_parameter_abs));
                        }
                    }

                    for (size_t i = 0; i < load_w.size(); ++i) {
                        const float v = static_cast<float>(p_load_w.values[i]);
                        if (!std::isfinite(v)) {
                            ++repaired;
                            continue;
                        }
                        load_w[i] = static_cast<Scalar>(std::clamp(v, -cfg_.max_parameter_abs, cfg_.max_parameter_abs));
                    }
                    for (size_t i = 0; i < load_b.size(); ++i) {
                        const float v = static_cast<float>(p_load_b.values[i]);
                        if (!std::isfinite(v)) {
                            ++repaired;
                            continue;
                        }
                        load_b[i] = static_cast<Scalar>(std::clamp(v, -cfg_.max_parameter_abs, cfg_.max_parameter_abs));
                    }
                    maybe_print(epoch_token_count > 0
                        ? epoch_loss_acc / static_cast<float>(epoch_token_count)
                        : 0.0f,
                        std::min(batch_start + cfg_.batch_size, epoch_dataset.size()),
                        epoch_dataset.size());
                }

                const auto bp_t1 = std::chrono::steady_clock::now();
                fd_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(bp_t1 - bp_t0).count();
                grad_samples_epoch = epoch_token_count;
                active_grad_fraction = (epoch_token_count > 0) ? 1.0f : 0.0f;
                if (epoch_token_count == 0) {
                    epoch_loss = kNonFinitePenalty;
                } else {
                    epoch_loss = epoch_loss_acc / static_cast<float>(epoch_token_count);
                }
            } else {
                // Full backprop mode: deterministic single-step analytic gradients
                // through embeddings, projections, hidden stack, and heads.
                auto& embed = model.embedding_table();
                auto& proj_q = model.proj_q_layer();
                auto& gate_q = model.gate_q_layer();
                auto& proj_k = model.proj_k_layer();
                auto& gate_k = model.gate_k_layer();
                auto& proj_v = model.proj_v_layer();
                auto& gate_v = model.gate_v_layer();
                auto& hidden = model.hidden_stack_layers();
                auto& out_w = model.output_head_weights();
                auto& out_b = model.output_head_bias();
                auto& act_w = model.action_head_weights();
                auto& act_b = model.action_head_bias();
                auto& load_w = model.load_gate_weights();
                auto& load_b = model.load_gate_bias();

                const size_t model_dim = model.config().model_dim;
                const size_t qk_dim = model.config().qk_dim;
                const size_t v_dim = model.config().v_dim;
                const size_t out_dim = model.output_dim();
                const size_t act_dim = model.action_dim();
                const size_t load_in = model.load_gate_input_dim();

                auto linear_backward = [](
                    const std::vector<Scalar>& w,
                    size_t outd,
                    size_t ind,
                    const arch::Vector& x,
                    const arch::Vector& dout,
                    std::vector<Scalar>& grad_w,
                    std::vector<Scalar>& grad_b,
                    arch::Vector& dx) {
                    for (size_t o = 0; o < outd; ++o) {
                        const float go = static_cast<float>(dout[o]);
                        grad_b[o] = static_cast<Scalar>(static_cast<float>(grad_b[o]) + go);
                        for (size_t i = 0; i < ind; ++i) {
                            const size_t wi = o * ind + i;
                            grad_w[wi] = static_cast<Scalar>(
                                static_cast<float>(grad_w[wi]) + go * static_cast<float>(x[i]));
                            dx[i] = static_cast<Scalar>(
                                static_cast<float>(dx[i]) + go * static_cast<float>(w[wi]));
                        }
                    }
                };

                auto flat_embeddings = [&]() {
                    std::vector<Scalar> v;
                    v.reserve(embed.size() * model_dim);
                    for (const auto& row : embed) {
                        v.insert(v.end(), row.begin(), row.end());
                    }
                    return v;
                };

                auto write_embeddings = [&](const std::vector<Scalar>& v) {
                    size_t idx = 0;
                    for (auto& row : embed) {
                        for (size_t i = 0; i < row.size(); ++i) {
                            row[i] = v[idx++];
                        }
                    }
                };

                const auto bp_t0 = std::chrono::steady_clock::now();
                float epoch_loss_acc = 0.0f;

                for (size_t batch_start = 0; batch_start < epoch_dataset.size(); batch_start += cfg_.batch_size) {
                    const size_t batch_end = std::min(batch_start + cfg_.batch_size, epoch_dataset.size());

                    ParameterTensor p_embed{flat_embeddings(), {}};
                    p_embed.grads.assign(p_embed.values.size(), static_cast<Scalar>(0.0f));

                    ParameterTensor p_pq{proj_q.w, std::vector<Scalar>(proj_q.w.size(), static_cast<Scalar>(0.0f))};
                    ParameterTensor p_bq{proj_q.b, std::vector<Scalar>(proj_q.b.size(), static_cast<Scalar>(0.0f))};
                    ParameterTensor p_gq{gate_q.w, std::vector<Scalar>(gate_q.w.size(), static_cast<Scalar>(0.0f))};
                    ParameterTensor p_bgq{gate_q.b, std::vector<Scalar>(gate_q.b.size(), static_cast<Scalar>(0.0f))};

                    ParameterTensor p_pk{proj_k.w, std::vector<Scalar>(proj_k.w.size(), static_cast<Scalar>(0.0f))};
                    ParameterTensor p_bk{proj_k.b, std::vector<Scalar>(proj_k.b.size(), static_cast<Scalar>(0.0f))};
                    ParameterTensor p_gk{gate_k.w, std::vector<Scalar>(gate_k.w.size(), static_cast<Scalar>(0.0f))};
                    ParameterTensor p_bgk{gate_k.b, std::vector<Scalar>(gate_k.b.size(), static_cast<Scalar>(0.0f))};

                    ParameterTensor p_pv{proj_v.w, std::vector<Scalar>(proj_v.w.size(), static_cast<Scalar>(0.0f))};
                    ParameterTensor p_bv{proj_v.b, std::vector<Scalar>(proj_v.b.size(), static_cast<Scalar>(0.0f))};
                    ParameterTensor p_gv{gate_v.w, std::vector<Scalar>(gate_v.w.size(), static_cast<Scalar>(0.0f))};
                    ParameterTensor p_bgv{gate_v.b, std::vector<Scalar>(gate_v.b.size(), static_cast<Scalar>(0.0f))};

                    std::vector<ParameterTensor> p_hidden_w;
                    std::vector<ParameterTensor> p_hidden_b;
                    p_hidden_w.reserve(hidden.size());
                    p_hidden_b.reserve(hidden.size());
                    for (const auto& h : hidden) {
                        p_hidden_w.push_back(ParameterTensor{h.w, std::vector<Scalar>(h.w.size(), static_cast<Scalar>(0.0f))});
                        p_hidden_b.push_back(ParameterTensor{h.b, std::vector<Scalar>(h.b.size(), static_cast<Scalar>(0.0f))});
                    }

                    ParameterTensor p_out_w{out_w, std::vector<Scalar>(out_w.size(), static_cast<Scalar>(0.0f))};
                    ParameterTensor p_out_b{out_b, std::vector<Scalar>(out_b.size(), static_cast<Scalar>(0.0f))};
                    ParameterTensor p_theta{{model.output_theta()}, {static_cast<Scalar>(0.0f)}};
                    ParameterTensor p_act_w{act_w, std::vector<Scalar>(act_w.size(), static_cast<Scalar>(0.0f))};
                    ParameterTensor p_act_b{act_b, std::vector<Scalar>(act_b.size(), static_cast<Scalar>(0.0f))};
                    ParameterTensor p_load_w{load_w, std::vector<Scalar>(load_w.size(), static_cast<Scalar>(0.0f))};
                    ParameterTensor p_load_b{load_b, std::vector<Scalar>(load_b.size(), static_cast<Scalar>(0.0f))};

                    const size_t batch_examples = batch_end - batch_start;
                    const bool use_parallel_examples = (effective_host_threads() > 1)
                        && (batch_examples >= cfg_.min_parallel_batch_examples);
                    const size_t example_threads = use_parallel_examples
                        ? std::min(effective_host_threads(), batch_examples)
                        : 1;

                    size_t batch_tokens = 0;
                    std::mutex accum_mutex;
                    auto add_grad_vec = [](std::vector<Scalar>& dst, const std::vector<Scalar>& src) {
                        if (dst.size() != src.size()) {
                            return;
                        }
                        for (size_t i = 0; i < dst.size(); ++i) {
                            dst[i] = static_cast<Scalar>(
                                static_cast<float>(dst[i]) + static_cast<float>(src[i]));
                        }
                    };

                    parallel_for_ranges(
                        batch_start,
                        batch_end,
                        example_threads,
                        [&](size_t ex_begin, size_t ex_end, size_t /*worker_id*/) {
                        std::vector<Scalar> lg_embed(p_embed.grads.size(), static_cast<Scalar>(0.0f));
                        std::vector<Scalar> lg_pq(p_pq.grads.size(), static_cast<Scalar>(0.0f));
                        std::vector<Scalar> lg_bq(p_bq.grads.size(), static_cast<Scalar>(0.0f));
                        std::vector<Scalar> lg_gq(p_gq.grads.size(), static_cast<Scalar>(0.0f));
                        std::vector<Scalar> lg_bgq(p_bgq.grads.size(), static_cast<Scalar>(0.0f));
                        std::vector<Scalar> lg_pk(p_pk.grads.size(), static_cast<Scalar>(0.0f));
                        std::vector<Scalar> lg_bk(p_bk.grads.size(), static_cast<Scalar>(0.0f));
                        std::vector<Scalar> lg_gk(p_gk.grads.size(), static_cast<Scalar>(0.0f));
                        std::vector<Scalar> lg_bgk(p_bgk.grads.size(), static_cast<Scalar>(0.0f));
                        std::vector<Scalar> lg_pv(p_pv.grads.size(), static_cast<Scalar>(0.0f));
                        std::vector<Scalar> lg_bv(p_bv.grads.size(), static_cast<Scalar>(0.0f));
                        std::vector<Scalar> lg_gv(p_gv.grads.size(), static_cast<Scalar>(0.0f));
                        std::vector<Scalar> lg_bgv(p_bgv.grads.size(), static_cast<Scalar>(0.0f));
                        std::vector<std::vector<Scalar>> lg_hidden_w(hidden.size());
                        std::vector<std::vector<Scalar>> lg_hidden_b(hidden.size());
                        for (size_t hi = 0; hi < hidden.size(); ++hi) {
                            lg_hidden_w[hi].assign(p_hidden_w[hi].grads.size(), static_cast<Scalar>(0.0f));
                            lg_hidden_b[hi].assign(p_hidden_b[hi].grads.size(), static_cast<Scalar>(0.0f));
                        }
                        std::vector<Scalar> lg_out_w(p_out_w.grads.size(), static_cast<Scalar>(0.0f));
                        std::vector<Scalar> lg_out_b(p_out_b.grads.size(), static_cast<Scalar>(0.0f));
                        std::vector<Scalar> lg_act_w(p_act_w.grads.size(), static_cast<Scalar>(0.0f));
                        std::vector<Scalar> lg_act_b(p_act_b.grads.size(), static_cast<Scalar>(0.0f));
                        std::vector<Scalar> lg_load_w(p_load_w.grads.size(), static_cast<Scalar>(0.0f));
                        std::vector<Scalar> lg_load_b(p_load_b.grads.size(), static_cast<Scalar>(0.0f));
                        Scalar lg_theta = static_cast<Scalar>(0.0f);

                        float local_epoch_loss_acc = 0.0f;
                        size_t local_batch_tokens = 0;
                        size_t local_epoch_tokens = 0;
                        size_t local_epoch_examples = 0;
                        size_t local_nonfinite_skips = 0;
                        size_t local_objective_evals = 0;

                        for (size_t exi = ex_begin; exi < ex_end; ++exi) {
                        const auto& ex = epoch_dataset[exi];
                        const size_t steps = std::min(ex.input.size(), ex.target.size());
                        if (steps == 0) {
                            continue;
                        }
                        std::mt19937 ex_rng(per_example_seed(cfg_.seed, epoch + 1, batch_start, exi));
                        std::uniform_int_distribution<size_t> start_dist(0, steps - 1);
                        const size_t start = start_dist(ex_rng);

                        struct StepCache {
                            size_t emb_off = 0;
                            uint8_t target = 0;
                            arch::Vector x;
                            arch::Vector zq, gq, q;
                            arch::Vector zk, gk, k;
                            arch::Vector zv, gv, v;
                            arch::KVState retained_state;
                            arch::Vector ret_raw;
                            arch::Vector state;
                            std::vector<arch::Vector> hidden_inputs;
                            std::vector<arch::Vector> hidden_preacts;
                            arch::Vector hstate;
                            arch::Vector action_logits;
                            Scalar load_logit = static_cast<Scalar>(0.0f);
                            arch::Vector raw_logits;
                            arch::Vector logits;
                        };

                        std::vector<StepCache> caches;
                        caches.reserve(steps);

                        arch::KVState recurrent_state(qk_dim, arch::Vector(v_dim, static_cast<Scalar>(0.0f)));
                        bool sequence_valid = true;

                        for (size_t ti = 0; ti < steps; ++ti) {
                            const size_t t = (start + ti) % steps;
                            StepCache sc;
                            const uint8_t ch = static_cast<uint8_t>(ex.input[t]);
                            sc.emb_off = static_cast<size_t>(ch) * model_dim;
                            sc.target = static_cast<uint8_t>(ex.target[t]);
                            sc.x = embed[static_cast<size_t>(ch)];

                            sc.zq = arch::sycl_ops::linear(p_pq.values, qk_dim, model_dim, sc.x, p_bq.values);
                            sc.gq = arch::sycl_ops::linear(p_gq.values, qk_dim, model_dim, sc.x, p_bgq.values);
                            sc.q = arch::activation::swiglu(sc.zq, sc.gq);

                            sc.zk = arch::sycl_ops::linear(p_pk.values, qk_dim, model_dim, sc.x, p_bk.values);
                            sc.gk = arch::sycl_ops::linear(p_gk.values, qk_dim, model_dim, sc.x, p_bgk.values);
                            sc.k = arch::activation::swiglu(sc.zk, sc.gk);

                            sc.zv = arch::sycl_ops::linear(p_pv.values, v_dim, model_dim, sc.x, p_bv.values);
                            sc.gv = arch::sycl_ops::linear(p_gv.values, v_dim, model_dim, sc.x, p_bgv.values);
                            sc.v = arch::activation::swiglu(sc.zv, sc.gv);

                            for (size_t i = 0; i < qk_dim; ++i) {
                                for (size_t j = 0; j < v_dim; ++j) {
                                    const float updated = model.config().decay * static_cast<float>(recurrent_state[i][j])
                                        + static_cast<float>(sc.k[i]) * static_cast<float>(sc.v[j]);
                                    recurrent_state[i][j] = static_cast<Scalar>(updated);
                                }
                            }
                            sc.retained_state = recurrent_state;

                            sc.ret_raw.assign(v_dim, static_cast<Scalar>(0.0f));
                            for (size_t j = 0; j < v_dim; ++j) {
                                float acc = 0.0f;
                                for (size_t i = 0; i < qk_dim; ++i) {
                                    acc += static_cast<float>(sc.q[i]) * static_cast<float>(recurrent_state[i][j]);
                                }
                                sc.ret_raw[j] = static_cast<Scalar>(acc);
                            }
                            const arch::Vector ret_norm = arch::group_norm(sc.ret_raw);

                            sc.state.assign(v_dim, static_cast<Scalar>(0.0f));
                            for (size_t i = 0; i < v_dim; ++i) {
                                sc.state[i] = static_cast<Scalar>(static_cast<float>(ret_norm[i]) + static_cast<float>(sc.v[i]));
                            }

                            sc.hidden_inputs.reserve(hidden.size());
                            sc.hidden_preacts.reserve(hidden.size());
                            sc.hstate = sc.state;
                            for (size_t hi = 0; hi < hidden.size(); ++hi) {
                                sc.hidden_inputs.push_back(sc.hstate);
                                arch::Vector hz = arch::sycl_ops::linear(
                                    p_hidden_w[hi].values,
                                    hidden[hi].out_dim,
                                    hidden[hi].in_dim,
                                    sc.hstate,
                                    p_hidden_b[hi].values);
                                sc.hidden_preacts.push_back(hz);
                                sc.hstate = arch::activation::swiglu(hz, hz);
                            }

                            sc.action_logits = arch::sycl_ops::linear(
                                p_act_w.values, act_dim, v_dim, sc.hstate, p_act_b.values);
                            {
                                const arch::Vector load_logits = arch::sycl_ops::linear(
                                    p_load_w.values, 1, load_in, sc.hstate, p_load_b.values);
                                sc.load_logit = load_logits.empty() ? static_cast<Scalar>(0.0f) : load_logits[0];
                            }
                            sc.raw_logits = arch::sycl_ops::linear(
                                p_out_w.values, out_dim, v_dim, sc.hstate, p_out_b.values);
                            sc.logits = arch::activation::param_tanh(sc.raw_logits, p_theta.values[0]);

                            float out_ce = 0.0f;
                            (void)output_loss_and_grad(
                                sc.logits,
                                static_cast<size_t>(sc.target),
                                out_ce);
                            if (!std::isfinite(out_ce)) {
                                ++local_nonfinite_skips;
                                sequence_valid = false;
                                break;
                            }
                            local_epoch_loss_acc += out_ce;
                            ++local_objective_evals;

                            caches.push_back(std::move(sc));
                        }

                        if (!sequence_valid || caches.empty()) {
                            continue;
                        }

                        // Penalise ||raw_logits[t] - raw_logits[t-1]||^2 at each step.
                        const bool bf_memory_active = memory_active_for_epoch(epoch + 1);
                        std::vector<arch::Vector> d_raw_consistency(
                            caches.size(),
                            arch::Vector(out_dim, static_cast<Scalar>(0.0f)));
                        if (cfg_.multistep_consistency_weight > 0.0f && caches.size() > 1) {
                            const float w_norm = cfg_.multistep_consistency_weight * 2.0f
                                / static_cast<float>(out_dim);
                            for (size_t tr = 0; tr < caches.size(); ++tr) {
                                // Forward neighbour contribution
                                if (tr + 1 < caches.size()) {
                                    float cons = 0.0f;
                                    for (size_t o = 0; o < out_dim; ++o) {
                                        const float diff = static_cast<float>(caches[tr + 1].raw_logits[o])
                                            - static_cast<float>(caches[tr].raw_logits[o]);
                                        cons += diff * diff;
                                        // d/d_raw[tr] += 2w/D*(raw[tr] - raw[tr+1])
                                        d_raw_consistency[tr][o] = static_cast<Scalar>(
                                            static_cast<float>(d_raw_consistency[tr][o]) - w_norm * diff);
                                        // d/d_raw[tr+1] += 2w/D*(raw[tr+1] - raw[tr])
                                        d_raw_consistency[tr + 1][o] = static_cast<Scalar>(
                                            static_cast<float>(d_raw_consistency[tr + 1][o]) + w_norm * diff);
                                    }
                                    local_epoch_loss_acc += cfg_.multistep_consistency_weight
                                        * cons / static_cast<float>(out_dim);
                                }
                            }
                        }

                        arch::KVState d_retained_next(qk_dim, arch::Vector(v_dim, static_cast<Scalar>(0.0f)));
                        for (size_t tr = caches.size(); tr-- > 0;) {
                            const StepCache& sc = caches[tr];

                            float out_ce = 0.0f;
                            arch::Vector d_logits = output_loss_and_grad(
                                sc.logits,
                                static_cast<size_t>(sc.target),
                                out_ce);
                            const auto tanh_grad = arch::activation::dparam_tanh(sc.raw_logits, p_theta.values[0], d_logits);

                            // Merge tanh backward with consistency correction.
                            arch::Vector d_raw_combined = tanh_grad.dx;
                            for (size_t o = 0; o < out_dim && o < d_raw_consistency[tr].size(); ++o) {
                                d_raw_combined[o] = static_cast<Scalar>(
                                    static_cast<float>(d_raw_combined[o])
                                    + static_cast<float>(d_raw_consistency[tr][o]));
                            }

                            arch::Vector d_hstate(v_dim, static_cast<Scalar>(0.0f));
                            linear_backward(
                                p_out_w.values, out_dim, v_dim, sc.hstate, d_raw_combined,
                                lg_out_w, lg_out_b, d_hstate);
                            lg_theta = static_cast<Scalar>(
                                static_cast<float>(lg_theta) + static_cast<float>(tanh_grad.dtheta));

                            if (cfg_.backprop_include_loop_supervision) {
                                // BackpropFull trains with a forced single-step forward pass
                                // (equivalent to always choosing OUTPUT on the first inner
                                // iteration).  Using the sequence-step index 'tr' as
                                // decision_index was wrong: it caused the action head to be
                                // supervised toward LOOP/QUERY while the forward always forced
                                // OUTPUT, so at inference the model would loop into state-space
                                // regions never seen during training and produce all spaces.
                                // Always supervise OUTPUT to keep training consistent with the
                                // single-step forward computation.
                                const size_t target_action =
                                    static_cast<size_t>(llm::arch::ModelAction::OUTPUT);

                                float action_ce = 0.0f;
                                arch::Vector d_action = softmax_ce_grad_from_logits(
                                    sc.action_logits,
                                    target_action,
                                    action_ce);
                                if (std::isfinite(action_ce)) {
                                    local_epoch_loss_acc += cfg_.loop_supervision_weight * action_ce;
                                    for (size_t i = 0; i < d_action.size(); ++i) {
                                        d_action[i] = static_cast<Scalar>(
                                            static_cast<float>(d_action[i]) * cfg_.loop_supervision_weight);
                                    }
                                    arch::Vector d_hstate_action(v_dim, static_cast<Scalar>(0.0f));
                                    linear_backward(
                                        p_act_w.values, act_dim, v_dim, sc.hstate, d_action,
                                        lg_act_w, lg_act_b, d_hstate_action);
                                    for (size_t i = 0; i < v_dim; ++i) {
                                        d_hstate[i] = static_cast<Scalar>(
                                            static_cast<float>(d_hstate[i]) + static_cast<float>(d_hstate_action[i]));
                                    }
                                }

                                if (cfg_.enable_query) {
                                    const float target_load = (target_action == static_cast<size_t>(llm::arch::ModelAction::QUERY_MEMORY))
                                        ? 1.0f
                                        : 0.0f;
                                    const auto [load_ce, d_load] = binary_ce_and_grad(
                                        static_cast<float>(sc.load_logit),
                                        target_load);
                                    if (std::isfinite(load_ce) && std::isfinite(d_load)) {
                                        local_epoch_loss_acc += cfg_.load_gate_supervision_weight * load_ce;
                                        const float d = cfg_.load_gate_supervision_weight * d_load;
                                        lg_load_b[0] = static_cast<Scalar>(
                                            static_cast<float>(lg_load_b[0]) + d);
                                        for (size_t i = 0; i < load_in; ++i) {
                                            lg_load_w[i] = static_cast<Scalar>(
                                                static_cast<float>(lg_load_w[i]) + d * static_cast<float>(sc.hstate[i]));
                                        }
                                    }
                                }
                            }

                            for (size_t hri = hidden.size(); hri-- > 0;) {
                                const arch::Vector dself = d_hstate;
                                const auto g = arch::activation::dswiglu(sc.hidden_preacts[hri], sc.hidden_preacts[hri], dself);
                                arch::Vector dz(sc.hidden_preacts[hri].size(), static_cast<Scalar>(0.0f));
                                for (size_t i = 0; i < dz.size(); ++i) {
                                    dz[i] = static_cast<Scalar>(
                                        static_cast<float>(g.dx[i]) + static_cast<float>(g.dgate[i]));
                                }
                                arch::Vector dprev(sc.hidden_inputs[hri].size(), static_cast<Scalar>(0.0f));
                                linear_backward(
                                    p_hidden_w[hri].values,
                                    hidden[hri].out_dim,
                                    hidden[hri].in_dim,
                                    sc.hidden_inputs[hri],
                                    dz,
                                    lg_hidden_w[hri],
                                    lg_hidden_b[hri],
                                    dprev);
                                d_hstate = std::move(dprev);
                            }

                            arch::Vector d_ret = d_hstate;
                            arch::Vector d_v(v_dim, static_cast<Scalar>(0.0f));
                            for (size_t i = 0; i < v_dim; ++i) {
                                d_v[i] = static_cast<Scalar>(
                                    static_cast<float>(d_v[i]) + static_cast<float>(d_hstate[i]));
                            }

                            const arch::Vector d_ret_raw = dgroup_norm(sc.ret_raw, d_ret);

                            arch::Vector d_q(qk_dim, static_cast<Scalar>(0.0f));
                            arch::Vector d_k(qk_dim, static_cast<Scalar>(0.0f));
                            arch::KVState d_retained_cur = d_retained_next;

                            for (size_t i = 0; i < qk_dim; ++i) {
                                float dq_i = 0.0f;
                                for (size_t j = 0; j < v_dim; ++j) {
                                    dq_i += static_cast<float>(d_ret_raw[j])
                                        * static_cast<float>(sc.retained_state[i][j]);

                                    const float ds = static_cast<float>(d_retained_cur[i][j])
                                        + static_cast<float>(sc.q[i]) * static_cast<float>(d_ret_raw[j]);

                                    d_k[i] = static_cast<Scalar>(
                                        static_cast<float>(d_k[i]) + ds * static_cast<float>(sc.v[j]));
                                    d_v[j] = static_cast<Scalar>(
                                        static_cast<float>(d_v[j]) + ds * static_cast<float>(sc.k[i]));
                                    d_retained_cur[i][j] = static_cast<Scalar>(model.config().decay * ds);
                                }
                                d_q[i] = static_cast<Scalar>(dq_i);
                            }
                            d_retained_next = std::move(d_retained_cur);

                            const auto gq_bw = arch::activation::dswiglu(sc.zq, sc.gq, d_q);
                            const auto gk_bw = arch::activation::dswiglu(sc.zk, sc.gk, d_k);
                            const auto gv_bw = arch::activation::dswiglu(sc.zv, sc.gv, d_v);

                            arch::Vector d_x(model_dim, static_cast<Scalar>(0.0f));
                            {
                                arch::Vector tdx(model_dim, static_cast<Scalar>(0.0f));
                                linear_backward(p_pq.values, qk_dim, model_dim, sc.x, gq_bw.dx, lg_pq, lg_bq, tdx);
                                for (size_t i = 0; i < model_dim; ++i) {
                                    d_x[i] = static_cast<Scalar>(static_cast<float>(d_x[i]) + static_cast<float>(tdx[i]));
                                }
                            }
                            {
                                arch::Vector tdx(model_dim, static_cast<Scalar>(0.0f));
                                linear_backward(p_gq.values, qk_dim, model_dim, sc.x, gq_bw.dgate, lg_gq, lg_bgq, tdx);
                                for (size_t i = 0; i < model_dim; ++i) {
                                    d_x[i] = static_cast<Scalar>(static_cast<float>(d_x[i]) + static_cast<float>(tdx[i]));
                                }
                            }
                            {
                                arch::Vector tdx(model_dim, static_cast<Scalar>(0.0f));
                                linear_backward(p_pk.values, qk_dim, model_dim, sc.x, gk_bw.dx, lg_pk, lg_bk, tdx);
                                for (size_t i = 0; i < model_dim; ++i) {
                                    d_x[i] = static_cast<Scalar>(static_cast<float>(d_x[i]) + static_cast<float>(tdx[i]));
                                }
                            }
                            {
                                arch::Vector tdx(model_dim, static_cast<Scalar>(0.0f));
                                linear_backward(p_gk.values, qk_dim, model_dim, sc.x, gk_bw.dgate, lg_gk, lg_bgk, tdx);
                                for (size_t i = 0; i < model_dim; ++i) {
                                    d_x[i] = static_cast<Scalar>(static_cast<float>(d_x[i]) + static_cast<float>(tdx[i]));
                                }
                            }
                            {
                                arch::Vector tdx(model_dim, static_cast<Scalar>(0.0f));
                                linear_backward(p_pv.values, v_dim, model_dim, sc.x, gv_bw.dx, lg_pv, lg_bv, tdx);
                                for (size_t i = 0; i < model_dim; ++i) {
                                    d_x[i] = static_cast<Scalar>(static_cast<float>(d_x[i]) + static_cast<float>(tdx[i]));
                                }
                            }
                            {
                                arch::Vector tdx(model_dim, static_cast<Scalar>(0.0f));
                                linear_backward(p_gv.values, v_dim, model_dim, sc.x, gv_bw.dgate, lg_gv, lg_bgv, tdx);
                                for (size_t i = 0; i < model_dim; ++i) {
                                    d_x[i] = static_cast<Scalar>(static_cast<float>(d_x[i]) + static_cast<float>(tdx[i]));
                                }
                            }

                            for (size_t i = 0; i < model_dim; ++i) {
                                lg_embed[sc.emb_off + i] = static_cast<Scalar>(
                                    static_cast<float>(lg_embed[sc.emb_off + i]) + static_cast<float>(d_x[i]));
                            }
                        }

                        local_batch_tokens += caches.size();
                        local_epoch_tokens += caches.size();
                        ++local_epoch_examples;
                        }

                        std::lock_guard<std::mutex> lock(accum_mutex);
                        add_grad_vec(p_embed.grads, lg_embed);
                        add_grad_vec(p_pq.grads, lg_pq);
                        add_grad_vec(p_bq.grads, lg_bq);
                        add_grad_vec(p_gq.grads, lg_gq);
                        add_grad_vec(p_bgq.grads, lg_bgq);
                        add_grad_vec(p_pk.grads, lg_pk);
                        add_grad_vec(p_bk.grads, lg_bk);
                        add_grad_vec(p_gk.grads, lg_gk);
                        add_grad_vec(p_bgk.grads, lg_bgk);
                        add_grad_vec(p_pv.grads, lg_pv);
                        add_grad_vec(p_bv.grads, lg_bv);
                        add_grad_vec(p_gv.grads, lg_gv);
                        add_grad_vec(p_bgv.grads, lg_bgv);
                        for (size_t hi = 0; hi < hidden.size(); ++hi) {
                            add_grad_vec(p_hidden_w[hi].grads, lg_hidden_w[hi]);
                            add_grad_vec(p_hidden_b[hi].grads, lg_hidden_b[hi]);
                        }
                        add_grad_vec(p_out_w.grads, lg_out_w);
                        add_grad_vec(p_out_b.grads, lg_out_b);
                        add_grad_vec(p_act_w.grads, lg_act_w);
                        add_grad_vec(p_act_b.grads, lg_act_b);
                        add_grad_vec(p_load_w.grads, lg_load_w);
                        add_grad_vec(p_load_b.grads, lg_load_b);
                        p_theta.grads[0] = static_cast<Scalar>(
                            static_cast<float>(p_theta.grads[0]) + static_cast<float>(lg_theta));

                        epoch_loss_acc += local_epoch_loss_acc;
                        batch_tokens += local_batch_tokens;
                        epoch_token_count += local_epoch_tokens;
                        epoch_example_count += local_epoch_examples;
                        nonfinite_skips += local_nonfinite_skips;
                        objective_evals += local_objective_evals;
                        });

                    if (batch_tokens == 0) {
                        continue;
                    }

                    const float inv = 1.0f / static_cast<float>(batch_tokens);
                    auto scale_grads = [inv](std::vector<Scalar>& g) {
                        for (auto& x : g) {
                            x = static_cast<Scalar>(static_cast<float>(x) * inv);
                        }
                    };

                    scale_grads(p_embed.grads);
                    scale_grads(p_pq.grads); scale_grads(p_bq.grads); scale_grads(p_gq.grads); scale_grads(p_bgq.grads);
                    scale_grads(p_pk.grads); scale_grads(p_bk.grads); scale_grads(p_gk.grads); scale_grads(p_bgk.grads);
                    scale_grads(p_pv.grads); scale_grads(p_bv.grads); scale_grads(p_gv.grads); scale_grads(p_bgv.grads);
                    for (size_t hi = 0; hi < hidden.size(); ++hi) {
                        scale_grads(p_hidden_w[hi].grads);
                        scale_grads(p_hidden_b[hi].grads);
                    }
                    scale_grads(p_out_w.grads); scale_grads(p_out_b.grads); scale_grads(p_act_w.grads); scale_grads(p_act_b.grads);
                    scale_grads(p_load_w.grads); scale_grads(p_load_b.grads);
                    p_theta.grads[0] = static_cast<Scalar>(static_cast<float>(p_theta.grads[0]) * inv);

                    clip_gradients_abs(p_embed.grads, cfg_.max_gradient_abs);
                    clip_gradients_abs(p_pq.grads, cfg_.max_gradient_abs);
                    clip_gradients_abs(p_bq.grads, cfg_.max_gradient_abs);
                    clip_gradients_abs(p_gq.grads, cfg_.max_gradient_abs);
                    clip_gradients_abs(p_bgq.grads, cfg_.max_gradient_abs);
                    clip_gradients_abs(p_pk.grads, cfg_.max_gradient_abs);
                    clip_gradients_abs(p_bk.grads, cfg_.max_gradient_abs);
                    clip_gradients_abs(p_gk.grads, cfg_.max_gradient_abs);
                    clip_gradients_abs(p_bgk.grads, cfg_.max_gradient_abs);
                    clip_gradients_abs(p_pv.grads, cfg_.max_gradient_abs);
                    clip_gradients_abs(p_bv.grads, cfg_.max_gradient_abs);
                    clip_gradients_abs(p_gv.grads, cfg_.max_gradient_abs);
                    clip_gradients_abs(p_bgv.grads, cfg_.max_gradient_abs);
                    for (size_t hi = 0; hi < hidden.size(); ++hi) {
                        clip_gradients_abs(p_hidden_w[hi].grads, cfg_.max_gradient_abs);
                        clip_gradients_abs(p_hidden_b[hi].grads, cfg_.max_gradient_abs);
                    }
                    clip_gradients_abs(p_out_w.grads, cfg_.max_gradient_abs);
                    clip_gradients_abs(p_out_b.grads, cfg_.max_gradient_abs);
                    clip_gradients_abs(p_act_w.grads, cfg_.max_gradient_abs);
                    clip_gradients_abs(p_act_b.grads, cfg_.max_gradient_abs);
                    clip_gradients_abs(p_load_w.grads, cfg_.max_gradient_abs);
                    clip_gradients_abs(p_load_b.grads, cfg_.max_gradient_abs);
                    clip_gradients_abs(p_theta.grads, cfg_.max_gradient_abs);

                    std::vector<ParameterTensor> params;
                    params.push_back(p_embed);
                    params.push_back(p_pq); params.push_back(p_bq); params.push_back(p_gq); params.push_back(p_bgq);
                    params.push_back(p_pk); params.push_back(p_bk); params.push_back(p_gk); params.push_back(p_bgk);
                    params.push_back(p_pv); params.push_back(p_bv); params.push_back(p_gv); params.push_back(p_bgv);
                    for (size_t hi = 0; hi < hidden.size(); ++hi) {
                        params.push_back(p_hidden_w[hi]);
                        params.push_back(p_hidden_b[hi]);
                    }
                    params.push_back(p_out_w); params.push_back(p_out_b);
                    params.push_back(p_act_w); params.push_back(p_act_b);
                    params.push_back(p_load_w); params.push_back(p_load_b);
                    params.push_back(p_theta);

                    optimizer.step(params);

                    size_t pi = 0;
                    p_embed = params[pi++];
                    p_pq = params[pi++]; p_bq = params[pi++]; p_gq = params[pi++]; p_bgq = params[pi++];
                    p_pk = params[pi++]; p_bk = params[pi++]; p_gk = params[pi++]; p_bgk = params[pi++];
                    p_pv = params[pi++]; p_bv = params[pi++]; p_gv = params[pi++]; p_bgv = params[pi++];
                    for (size_t hi = 0; hi < hidden.size(); ++hi) {
                        p_hidden_w[hi] = params[pi++];
                        p_hidden_b[hi] = params[pi++];
                    }
                    p_out_w = params[pi++]; p_out_b = params[pi++];
                    p_act_w = params[pi++]; p_act_b = params[pi++];
                    p_load_w = params[pi++]; p_load_b = params[pi++];
                    p_theta = params[pi++];

                    // write back with finite guards
                    write_embeddings(p_embed.values);
                    auto clamp_vec = [&](std::vector<Scalar>& vec) {
                        for (auto& vv : vec) {
                            const float f = static_cast<float>(vv);
                            if (!std::isfinite(f)) {
                                ++repaired;
                                continue;
                            }
                            vv = static_cast<Scalar>(std::clamp(f, -cfg_.max_parameter_abs, cfg_.max_parameter_abs));
                        }
                    };

                    proj_q.w = p_pq.values; clamp_vec(proj_q.w); proj_q.b = p_bq.values; clamp_vec(proj_q.b);
                    gate_q.w = p_gq.values; clamp_vec(gate_q.w); gate_q.b = p_bgq.values; clamp_vec(gate_q.b);
                    proj_k.w = p_pk.values; clamp_vec(proj_k.w); proj_k.b = p_bk.values; clamp_vec(proj_k.b);
                    gate_k.w = p_gk.values; clamp_vec(gate_k.w); gate_k.b = p_bgk.values; clamp_vec(gate_k.b);
                    proj_v.w = p_pv.values; clamp_vec(proj_v.w); proj_v.b = p_bv.values; clamp_vec(proj_v.b);
                    gate_v.w = p_gv.values; clamp_vec(gate_v.w); gate_v.b = p_bgv.values; clamp_vec(gate_v.b);
                    for (size_t hi = 0; hi < hidden.size(); ++hi) {
                        hidden[hi].w = p_hidden_w[hi].values; clamp_vec(hidden[hi].w);
                        hidden[hi].b = p_hidden_b[hi].values; clamp_vec(hidden[hi].b);
                    }
                    out_w = p_out_w.values; clamp_vec(out_w);
                    out_b = p_out_b.values; clamp_vec(out_b);
                    act_w = p_act_w.values; clamp_vec(act_w);
                    act_b = p_act_b.values; clamp_vec(act_b);
                    load_w = p_load_w.values; clamp_vec(load_w);
                    load_b = p_load_b.values; clamp_vec(load_b);
                    {
                        const float th = static_cast<float>(p_theta.values[0]);
                        if (std::isfinite(th)) {
                            model.set_output_theta(static_cast<Scalar>(std::clamp(th, -cfg_.max_parameter_abs, cfg_.max_parameter_abs)));
                        } else {
                            ++repaired;
                        }
                    }
                    maybe_print(epoch_token_count > 0
                        ? epoch_loss_acc / static_cast<float>(epoch_token_count)
                        : 0.0f,
                        std::min(batch_start + cfg_.batch_size, epoch_dataset.size()),
                        epoch_dataset.size());
                }

                const auto bp_t1 = std::chrono::steady_clock::now();
                fd_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(bp_t1 - bp_t0).count();
                grad_samples_epoch = epoch_token_count;
                active_grad_fraction = (epoch_token_count > 0) ? 1.0f : 0.0f;
                if (epoch_token_count == 0) {
                    epoch_loss = kNonFinitePenalty;
                } else {
                    epoch_loss = epoch_loss_acc / static_cast<float>(epoch_token_count);
                }
            }

            if (!std::isfinite(epoch_loss)) {
                epoch_loss = has_ema ? loss_ema : kNonFinitePenalty;
            }
            history.push_back(EpochResult{epoch + 1, epoch_loss});

            if (!has_ema) {
                loss_ema = epoch_loss;
                has_ema = true;
            } else {
                loss_ema = ema_beta * loss_ema + (1.0f - ema_beta) * epoch_loss;
            }

            const float loss_delta = has_prev_epoch_loss
                ? (epoch_loss - prev_epoch_loss)
                : 0.0f;
            prev_epoch_loss = epoch_loss;
            has_prev_epoch_loss = true;

            const float nonfinite_skip_ratio = (grad_samples_epoch == 0)
                ? 0.0f
                : static_cast<float>(nonfinite_skips) / static_cast<float>(grad_samples_epoch);
            const size_t repair_den = (cfg_.mode == TrainMode::FiniteDifference)
                ? model.parameter_count()
                : (model.output_head_weights().size()
                    + model.output_head_bias().size()
                    + (cfg_.backprop_include_loop_supervision
                        ? model.action_head_weights().size() + model.action_head_bias().size()
                        : 0)
                    + 1);
            const float repaired_ratio = (repair_den == 0)
                ? 0.0f
                : static_cast<float>(repaired) / static_cast<float>(repair_den);

            if (cfg_.adaptive_instability_window > 0 && std::isfinite(epoch_loss)) {
                recent_losses.push_back(epoch_loss);
                if (recent_losses.size() > cfg_.adaptive_instability_window) {
                    recent_losses.erase(recent_losses.begin());
                }
            }

            bool unstable_epoch = false;
            if (cfg_.enable_instability_backoff) {
                const bool skip_unstable =
                    (nonfinite_skip_ratio >= cfg_.instability_skip_ratio_threshold)
                    || (repaired_ratio >= cfg_.instability_repair_ratio_threshold);

                bool variance_unstable = false;
                if (cfg_.adaptive_instability_var_ratio > 0.0f
                    && recent_losses.size() >= 3) {
                    float mean_l = 0.0f;
                    for (float l : recent_losses) {
                        mean_l += l;
                    }
                    mean_l /= static_cast<float>(recent_losses.size());
                    float var_l = 0.0f;
                    for (float l : recent_losses) {
                        const float d = l - mean_l;
                        var_l += d * d;
                    }
                    var_l /= static_cast<float>(recent_losses.size());
                    const float mean_sq = mean_l * mean_l;
                    if (mean_sq > 1e-8f) {
                        variance_unstable = (var_l / mean_sq) > cfg_.adaptive_instability_var_ratio;
                    }
                }

                unstable_epoch = skip_unstable || variance_unstable;

                if (unstable_epoch) {
                    const float backoff = std::clamp(cfg_.instability_lr_backoff, 0.05f, 0.99f);
                    lr_multiplier = std::max(backoff * lr_multiplier, 0.01f);
                    lr_cooldown_left = cfg_.instability_cooldown_epochs;
                } else {
                    if (lr_cooldown_left > 0) {
                        --lr_cooldown_left;
                    } else {
                        // Recover cautiously after a stable period.
                        lr_multiplier = std::min(1.0f, lr_multiplier * 1.05f);
                    }
                }
            }
            inter_loss_delta = loss_delta;
            inter_unstable = unstable_epoch;
            maybe_print(epoch_loss, epoch_dataset.size(), epoch_dataset.size());
        }

        return history;
    }
};

inline std::vector<SequenceExample> make_shift_dataset(const std::vector<std::string>& texts) {
    std::vector<SequenceExample> dataset;
    dataset.reserve(texts.size());

    for (const std::string& s : texts) {
        if (s.size() < 2) {
            continue;
        }
        SequenceExample ex;
        ex.input = s.substr(0, s.size() - 1);
        ex.target = s.substr(1);
        dataset.push_back(std::move(ex));
    }

    return dataset;
}

} // namespace llm::training::looping
