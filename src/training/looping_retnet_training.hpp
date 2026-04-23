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

struct TrainConfig {
    TrainMode mode = TrainMode::FiniteDifference;
    size_t epochs = 5;
    float learning_rate = 1e-2f;      // peak LR after warmup
    float min_learning_rate = 1e-4f;  // absolute LR floor
    float min_learning_rate_ratio = 0.1f; // relative floor vs peak LR
    size_t warmup_epochs = 2;         // linear warmup epochs
    float warmup_start_ratio = 0.2f;  // warmup starts at peak_lr * ratio
    float loss_ema_beta = 0.9f;       // moving-average smoothing for logging
    float weight_decay = 1e-4f;
    float sgd_momentum = 0.9f;
    float max_gradient_abs = 10.0f;   // per-coordinate finite-diff grad clip
    float max_parameter_abs = 100.0f; // post-step parameter clamp
    bool enable_instability_backoff = true;
    float instability_skip_ratio_threshold = 0.25f;
    float instability_repair_ratio_threshold = 0.01f;
    float instability_lr_backoff = 0.5f;
    size_t instability_cooldown_epochs = 3;
    float min_effective_learning_rate = 1e-5f;

    // Finite-difference gradient estimate over randomly sampled coordinates.
    float fd_eps = 1e-2f;
    size_t grad_coordinate_samples = 256;
    size_t min_grad_coordinate_samples = 32;

    // Minibatch support used by both modes where applicable.
    size_t batch_size = 1;
    bool disable_memory_writes_when_query_disabled = true;

    // Backprop-over-heads mode knobs.
    bool backprop_force_single_step = true;
    bool backprop_include_loop_supervision = true;
    size_t backprop_fd_check_samples = 0;
    float backprop_fd_check_eps = 1e-3f;

    // Memory-module training knobs.
    float memory_query_penalty = 0.05f; // penalize no-query behavior
    float memory_miss_penalty = 0.05f;  // penalize empty query results
    float memory_alignment_weight = 0.1f;
    float memory_edge_budget_penalty = 0.02f;
    float loop_supervision_weight = 0.2f;
    bool force_output = false;          // keep false for full looping behavior
    bool enable_query = true;
    bool use_parallel_retention = true;

    size_t forced_loop_min = 1;
    size_t forced_loop_max = 5;

    uint32_t seed = 7;

    memory::MemoryConfig memory_cfg{};
};

struct EpochResult {
    size_t epoch = 0;
    float avg_loss = 0.0f;
};

class LoopingRetNetSGDTrainer {
    TrainConfig cfg_;

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

    static size_t loop_supervision_target_from_logits(
        const arch::Vector& current_output_logits,
        const arch::Vector& next_output_logits,
        uint8_t target)
    {
        const float cur_ce = cross_entropy_from_logits(current_output_logits, target);
        const float nxt_ce = cross_entropy_from_logits(next_output_logits, target);
        const bool should_loop = (nxt_ce + 1e-4f) < cur_ce;
        return should_loop
            ? static_cast<size_t>(llm::arch::ModelAction::LOOP)
            : static_cast<size_t>(llm::arch::ModelAction::OUTPUT);
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

    float sequence_loss(
        arch::LoopingRetNet& model,
        const SequenceExample& ex,
        std::mt19937& rng) const {
        if (ex.input.empty() || ex.target.empty()) {
            return 0.0f;
        }

        const size_t steps = std::min(ex.input.size(), ex.target.size());
        if (steps == 0) {
            return 0.0f;
        }

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

        arch::KVState recurrent_state;
        arch::AttentionMemory kv_cache;

        const size_t loop_min = std::max<size_t>(1, cfg_.forced_loop_min);
        const size_t loop_max = std::max(loop_min, std::min<size_t>(cfg_.forced_loop_max, model.config().max_steps));
        std::uniform_int_distribution<size_t> loop_dist(loop_min, loop_max);

        float loss = 0.0f;
        for (size_t t = 0; t < steps; ++t) {
            const size_t forced_loops = loop_dist(rng);
            const auto trace = model.step_with_trace(
                ex.input[t],
                recurrent_state,
                kv_cache,
                bridge,
                query,
                cfg_.force_output,
                cfg_.enable_query,
                forced_loops,
                cfg_.use_parallel_retention);

            loss += cross_entropy_from_logits(trace.logits, static_cast<uint8_t>(ex.target[t]));
            if (!std::isfinite(loss)) {
                return kNonFinitePenalty;
            }

            // Train the loop decision output: if doing one more loop lowered
            // output CE, supervise LOOP; otherwise supervise OUTPUT.
            if (trace.per_step_output_logits.size() > 1 && trace.per_step_action_logits.size() == trace.per_step_output_logits.size()) {
                for (size_t i = 0; i + 1 < trace.per_step_output_logits.size(); ++i) {
                    const float cur_ce = cross_entropy_from_logits(
                        trace.per_step_output_logits[i], static_cast<uint8_t>(ex.target[t]));
                    const float nxt_ce = cross_entropy_from_logits(
                        trace.per_step_output_logits[i + 1], static_cast<uint8_t>(ex.target[t]));
                    const bool should_loop = (nxt_ce + 1e-4f) < cur_ce;
                    const size_t target = should_loop
                        ? static_cast<size_t>(llm::arch::ModelAction::LOOP)
                        : static_cast<size_t>(llm::arch::ModelAction::OUTPUT);
                    loss += cfg_.loop_supervision_weight
                        * action_ce_from_logits(trace.per_step_action_logits[i], target);
                    if (!std::isfinite(loss)) {
                        return kNonFinitePenalty;
                    }
                }
            }

            if (cfg_.enable_query && trace.query_count == 0) {
                loss += cfg_.memory_query_penalty;
            }

            if (cfg_.enable_query && !kv_cache.keys.empty()) {
                const arch::AttentionMemory queried = query.query(kv_cache.keys.back());
                if (queried.empty()) {
                    loss += cfg_.memory_miss_penalty;
                } else {
                    // Robust memory signal: queried values should align with
                    // the latest value in cache.
                    const arch::Vector& target_value = kv_cache.values.back();
                    float best_sim = -1.0f;
                    for (const auto& qv : queried.values) {
                        best_sim = std::max(best_sim, cosine_similarity(qv, target_value));
                    }
                    loss += cfg_.memory_alignment_weight * (1.0f - 0.5f * (best_sim + 1.0f));
                }
            }

            // Prevent unchecked KV-cache growth from dominating compute.
            const float allowed_edges = static_cast<float>(cfg_.memory_cfg.max_write_entries * (t + 1));
            const float actual_edges = static_cast<float>(kv_cache.keys.size());
            if (actual_edges > allowed_edges) {
                loss += cfg_.memory_edge_budget_penalty * (actual_edges - allowed_edges);
                if (!std::isfinite(loss)) {
                    return kNonFinitePenalty;
                }
            }
        }

        const float avg = loss / static_cast<float>(steps);
        if (!std::isfinite(avg)) {
            return kNonFinitePenalty;
        }
        return avg;
    }

    float dataset_loss(
        arch::LoopingRetNet& model,
        const std::vector<SequenceExample>& dataset,
        std::mt19937& rng) const {
        if (dataset.empty()) {
            return 0.0f;
        }
        float total = 0.0f;
        for (const auto& ex : dataset) {
            const float s = sequence_loss(model, ex, rng);
            if (!std::isfinite(s)) {
                return kNonFinitePenalty;
            }
            total += s;
            if (!std::isfinite(total)) {
                return kNonFinitePenalty;
            }
        }
        const float avg = total / static_cast<float>(dataset.size());
        if (!std::isfinite(avg)) {
            return kNonFinitePenalty;
        }
        return avg;
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
        SGD optimizer(learning_rate_for_epoch(1), cfg_.weight_decay, cfg_.sgd_momentum);
        float lr_multiplier = 1.0f;
        size_t lr_cooldown_left = 0;
        float prev_epoch_loss = 0.0f;
        bool has_prev_epoch_loss = false;

        std::vector<EpochResult> history;
        history.reserve(cfg_.epochs);
        float loss_ema = 0.0f;
        bool has_ema = false;
        const float ema_beta = std::clamp(cfg_.loss_ema_beta, 0.0f, 0.9999f);

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

                    const std::mt19937 eval_rng_state = rng;

                    *refs[idx] = static_cast<Scalar>(static_cast<float>(original) + cfg_.fd_eps);
                    std::mt19937 rng_plus = eval_rng_state;
                    const auto eval_t0 = std::chrono::steady_clock::now();
                    const float l_plus = dataset_loss(model, dataset, rng_plus);
                    const auto eval_t1 = std::chrono::steady_clock::now();
                    seq_forward_ms += std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(eval_t1 - eval_t0).count();
                    ++objective_evals;

                    *refs[idx] = static_cast<Scalar>(static_cast<float>(original) - cfg_.fd_eps);
                    std::mt19937 rng_minus = eval_rng_state;
                    const auto eval_t2 = std::chrono::steady_clock::now();
                    const float l_minus = dataset_loss(model, dataset, rng_minus);
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
                epoch_loss = dataset_loss(model, dataset, rng);
                const auto eval_epoch_t1 = std::chrono::steady_clock::now();
                seq_forward_ms += std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(eval_epoch_t1 - eval_epoch_t0).count();
                ++objective_evals;
                for (const auto& ex : dataset) {
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

                const size_t out_in = model.output_input_dim();
                const size_t out_dim = model.output_dim();
                const size_t act_in = model.action_input_dim();
                const size_t act_dim = model.action_dim();

                const auto bp_t0 = std::chrono::steady_clock::now();
                float epoch_loss_acc = 0.0f;

                for (size_t batch_start = 0; batch_start < dataset.size(); batch_start += cfg_.batch_size) {
                    const size_t batch_end = std::min(batch_start + cfg_.batch_size, dataset.size());

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

                    size_t batch_tokens = 0;

                    for (size_t exi = batch_start; exi < batch_end; ++exi) {
                        const auto& ex = dataset[exi];
                        if (ex.input.empty() || ex.target.empty()) {
                            continue;
                        }

                        const size_t steps = std::min(ex.input.size(), ex.target.size());
                        if (steps == 0) {
                            continue;
                        }

                        Graph graph;
                        SpatialMap spatial_map;
                        memory::NodeCompressor compressor(model.config().v_dim, cfg_.memory_cfg.semvec_dim, cfg_.seed);
                        memory::GraphMemoryBridge bridge(graph, spatial_map, std::move(compressor), cfg_.memory_cfg);
                        memory::MultiHopQuery query(graph, spatial_map, cfg_.memory_cfg);

                        arch::KVState recurrent_state;
                        arch::AttentionMemory kv_cache;

                        for (size_t t = 0; t < steps; ++t) {
                            const bool use_query = cfg_.backprop_force_single_step ? false : cfg_.enable_query;
                            const bool force_output = cfg_.backprop_force_single_step ? true : cfg_.force_output;
                            const size_t forced_loops = cfg_.backprop_force_single_step
                                ? 1
                                : std::max<size_t>(1, cfg_.forced_loop_min);
                            const bool write_memory = !cfg_.disable_memory_writes_when_query_disabled || use_query;

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
                                write_memory);
                            const auto trace_t1 = std::chrono::steady_clock::now();
                            seq_forward_ms += std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(trace_t1 - trace_t0).count();
                            query_events += trace.query_count;
                            ++objective_evals;

                            float out_ce = 0.0f;
                            arch::Vector d_logits = softmax_ce_grad_from_logits(
                                trace.logits,
                                static_cast<size_t>(static_cast<uint8_t>(ex.target[t])),
                                out_ce);
                            if (!std::isfinite(out_ce)) {
                                ++nonfinite_skips;
                                continue;
                            }
                            epoch_loss_acc += out_ce;

                            const auto tanh_grad = arch::activation::dparam_tanh(
                                trace.raw_logits,
                                model.output_theta(),
                                d_logits);

                            if (cfg_.backprop_fd_check_samples > 0
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
                                    return cross_entropy_from_logits(
                                        logits,
                                        static_cast<uint8_t>(ex.target[t]));
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
                                const float d_raw = static_cast<float>(tanh_grad.dx[o]);
                                if (!std::isfinite(d_raw)) {
                                    ++nonfinite_skips;
                                    continue;
                                }
                                p_out_b.grads[o] = static_cast<Scalar>(
                                    static_cast<float>(p_out_b.grads[o]) + d_raw);
                                for (size_t i = 0; i < out_in; ++i) {
                                    const float s = static_cast<float>(trace.final_state[i]);
                                    const size_t wi = o * out_in + i;
                                    p_out_w.grads[wi] = static_cast<Scalar>(
                                        static_cast<float>(p_out_w.grads[wi]) + d_raw * s);
                                }
                            }
                            p_theta.grads[0] = static_cast<Scalar>(
                                static_cast<float>(p_theta.grads[0]) + static_cast<float>(tanh_grad.dtheta));

                            if (cfg_.backprop_include_loop_supervision
                                && trace.per_step_output_logits.size() > 1
                                && trace.per_step_action_logits.size() == trace.per_step_states.size()) {
                                for (size_t si = 0; si + 1 < trace.per_step_output_logits.size(); ++si) {
                                    const size_t target_action = loop_supervision_target_from_logits(
                                        trace.per_step_output_logits[si],
                                        trace.per_step_output_logits[si + 1],
                                        static_cast<uint8_t>(ex.target[t]));
                                    float action_ce = 0.0f;
                                    arch::Vector d_action = softmax_ce_grad_from_logits(
                                        trace.per_step_action_logits[si],
                                        target_action,
                                        action_ce);
                                    if (!std::isfinite(action_ce)) {
                                        ++nonfinite_skips;
                                        continue;
                                    }
                                    epoch_loss_acc += cfg_.loop_supervision_weight * action_ce;

                                    const auto& state = trace.per_step_states[si];
                                    for (size_t a = 0; a < act_dim; ++a) {
                                        const float dag = cfg_.loop_supervision_weight * static_cast<float>(d_action[a]);
                                        p_act_b.grads[a] = static_cast<Scalar>(
                                            static_cast<float>(p_act_b.grads[a]) + dag);
                                        for (size_t i = 0; i < act_in; ++i) {
                                            const size_t wi = a * act_in + i;
                                            p_act_w.grads[wi] = static_cast<Scalar>(
                                                static_cast<float>(p_act_w.grads[wi]) + dag * static_cast<float>(state[i]));
                                        }
                                    }
                                }
                            }

                            ++batch_tokens;
                            ++epoch_token_count;
                        }
                        ++epoch_example_count;
                    }

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

                    std::vector<ParameterTensor> params;
                    params.push_back(p_out_w);
                    params.push_back(p_out_b);
                    params.push_back(p_theta);
                    if (cfg_.backprop_include_loop_supervision) {
                        params.push_back(p_act_w);
                        params.push_back(p_act_b);
                    }
                    optimizer.step(params);

                    p_out_w = params[0];
                    p_out_b = params[1];
                    p_theta = params[2];
                    if (cfg_.backprop_include_loop_supervision) {
                        p_act_w = params[3];
                        p_act_b = params[4];
                    }

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

                const size_t model_dim = model.config().model_dim;
                const size_t qk_dim = model.config().qk_dim;
                const size_t v_dim = model.config().v_dim;
                const size_t out_dim = model.output_dim();
                const size_t act_dim = model.action_dim();

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

                for (size_t batch_start = 0; batch_start < dataset.size(); batch_start += cfg_.batch_size) {
                    const size_t batch_end = std::min(batch_start + cfg_.batch_size, dataset.size());

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

                    size_t batch_tokens = 0;

                    for (size_t exi = batch_start; exi < batch_end; ++exi) {
                        const auto& ex = dataset[exi];
                        const size_t steps = std::min(ex.input.size(), ex.target.size());
                        if (steps == 0) {
                            continue;
                        }

                        for (size_t t = 0; t < steps; ++t) {
                            const uint8_t ch = static_cast<uint8_t>(ex.input[t]);
                            const size_t emb_off = static_cast<size_t>(ch) * model_dim;

                            arch::Vector x = embed[static_cast<size_t>(ch)];

                            arch::Vector zq = arch::sycl_ops::linear(p_pq.values, qk_dim, model_dim, x, p_bq.values);
                            arch::Vector gq = arch::sycl_ops::linear(p_gq.values, qk_dim, model_dim, x, p_bgq.values);
                            arch::Vector q = arch::activation::swiglu(zq, gq);

                            arch::Vector zk = arch::sycl_ops::linear(p_pk.values, qk_dim, model_dim, x, p_bk.values);
                            arch::Vector gk = arch::sycl_ops::linear(p_gk.values, qk_dim, model_dim, x, p_bgk.values);
                            arch::Vector k = arch::activation::swiglu(zk, gk);

                            arch::Vector zv = arch::sycl_ops::linear(p_pv.values, v_dim, model_dim, x, p_bv.values);
                            arch::Vector gv = arch::sycl_ops::linear(p_gv.values, v_dim, model_dim, x, p_bgv.values);
                            arch::Vector v = arch::activation::swiglu(zv, gv);

                            float alpha = 0.0f;
                            for (size_t i = 0; i < q.size(); ++i) {
                                alpha += static_cast<float>(q[i]) * static_cast<float>(k[i]);
                            }

                            arch::Vector y(v_dim, static_cast<Scalar>(0.0f));
                            for (size_t i = 0; i < v_dim; ++i) {
                                y[i] = static_cast<Scalar>(alpha * static_cast<float>(v[i]));
                            }
                            arch::Vector ret = arch::group_norm(y);

                            arch::Vector state(v_dim, static_cast<Scalar>(0.0f));
                            for (size_t i = 0; i < v_dim; ++i) {
                                state[i] = static_cast<Scalar>(static_cast<float>(ret[i]) + static_cast<float>(v[i]));
                            }

                            std::vector<arch::Vector> hidden_inputs;
                            std::vector<arch::Vector> hidden_preacts;
                            hidden_inputs.reserve(hidden.size());
                            hidden_preacts.reserve(hidden.size());
                            arch::Vector hstate = state;
                            for (size_t hi = 0; hi < hidden.size(); ++hi) {
                                hidden_inputs.push_back(hstate);
                                arch::Vector hz = arch::sycl_ops::linear(
                                    p_hidden_w[hi].values,
                                    hidden[hi].out_dim,
                                    hidden[hi].in_dim,
                                    hstate,
                                    p_hidden_b[hi].values);
                                hidden_preacts.push_back(hz);
                                hstate = arch::activation::swiglu(hz, hz);
                            }

                            arch::Vector action_logits = arch::sycl_ops::linear(
                                p_act_w.values, act_dim, v_dim, hstate, p_act_b.values);
                            arch::Vector raw_logits = arch::sycl_ops::linear(
                                p_out_w.values, out_dim, v_dim, hstate, p_out_b.values);
                            arch::Vector logits = arch::activation::param_tanh(raw_logits, p_theta.values[0]);

                            float out_ce = 0.0f;
                            arch::Vector d_logits = softmax_ce_grad_from_logits(
                                logits,
                                static_cast<size_t>(static_cast<uint8_t>(ex.target[t])),
                                out_ce);
                            if (!std::isfinite(out_ce)) {
                                ++nonfinite_skips;
                                continue;
                            }
                            epoch_loss_acc += out_ce;

                            const auto tanh_grad = arch::activation::dparam_tanh(raw_logits, p_theta.values[0], d_logits);

                            arch::Vector d_hstate(v_dim, static_cast<Scalar>(0.0f));
                            linear_backward(
                                p_out_w.values, out_dim, v_dim, hstate, tanh_grad.dx,
                                p_out_w.grads, p_out_b.grads, d_hstate);
                            p_theta.grads[0] = static_cast<Scalar>(
                                static_cast<float>(p_theta.grads[0]) + static_cast<float>(tanh_grad.dtheta));

                            float action_ce = 0.0f;
                            arch::Vector d_action = softmax_ce_grad_from_logits(
                                action_logits,
                                static_cast<size_t>(llm::arch::ModelAction::OUTPUT),
                                action_ce);
                            if (std::isfinite(action_ce)) {
                                epoch_loss_acc += cfg_.loop_supervision_weight * action_ce;
                                for (size_t i = 0; i < d_action.size(); ++i) {
                                    d_action[i] = static_cast<Scalar>(
                                        static_cast<float>(d_action[i]) * cfg_.loop_supervision_weight);
                                }
                                arch::Vector d_hstate_action(v_dim, static_cast<Scalar>(0.0f));
                                linear_backward(
                                    p_act_w.values, act_dim, v_dim, hstate, d_action,
                                    p_act_w.grads, p_act_b.grads, d_hstate_action);
                                for (size_t i = 0; i < v_dim; ++i) {
                                    d_hstate[i] = static_cast<Scalar>(
                                        static_cast<float>(d_hstate[i]) + static_cast<float>(d_hstate_action[i]));
                                }
                            }

                            // Hidden stack backward.
                            for (size_t hri = hidden.size(); hri-- > 0;) {
                                const arch::Vector dself = d_hstate;
                                const auto g = arch::activation::dswiglu(hidden_preacts[hri], hidden_preacts[hri], dself);
                                arch::Vector dz(hidden_preacts[hri].size(), static_cast<Scalar>(0.0f));
                                for (size_t i = 0; i < dz.size(); ++i) {
                                    dz[i] = static_cast<Scalar>(
                                        static_cast<float>(g.dx[i]) + static_cast<float>(g.dgate[i]));
                                }
                                arch::Vector dprev(hidden_inputs[hri].size(), static_cast<Scalar>(0.0f));
                                linear_backward(
                                    p_hidden_w[hri].values,
                                    hidden[hri].out_dim,
                                    hidden[hri].in_dim,
                                    hidden_inputs[hri],
                                    dz,
                                    p_hidden_w[hri].grads,
                                    p_hidden_b[hri].grads,
                                    dprev);
                                d_hstate = std::move(dprev);
                            }

                            // Backward through state = ret + v
                            arch::Vector d_ret = d_hstate;
                            arch::Vector d_v(v_dim, static_cast<Scalar>(0.0f));
                            for (size_t i = 0; i < v_dim; ++i) {
                                d_v[i] = static_cast<Scalar>(
                                    static_cast<float>(d_v[i]) + static_cast<float>(d_hstate[i]));
                            }

                            // ret = group_norm(y), y = alpha * v
                            const arch::Vector d_y = dgroup_norm(y, d_ret);
                            float d_alpha = 0.0f;
                            for (size_t i = 0; i < v_dim; ++i) {
                                const float gy = static_cast<float>(d_y[i]);
                                d_alpha += gy * static_cast<float>(v[i]);
                                d_v[i] = static_cast<Scalar>(
                                    static_cast<float>(d_v[i]) + gy * alpha);
                            }

                            arch::Vector d_q(qk_dim, static_cast<Scalar>(0.0f));
                            arch::Vector d_k(qk_dim, static_cast<Scalar>(0.0f));
                            for (size_t i = 0; i < qk_dim; ++i) {
                                d_q[i] = static_cast<Scalar>(d_alpha * static_cast<float>(k[i]));
                                d_k[i] = static_cast<Scalar>(d_alpha * static_cast<float>(q[i]));
                            }

                            const auto gq_bw = arch::activation::dswiglu(zq, gq, d_q);
                            const auto gk_bw = arch::activation::dswiglu(zk, gk, d_k);
                            const auto gv_bw = arch::activation::dswiglu(zv, gv, d_v);

                            arch::Vector d_x(model_dim, static_cast<Scalar>(0.0f));
                            {
                                arch::Vector tdx(model_dim, static_cast<Scalar>(0.0f));
                                linear_backward(p_pq.values, qk_dim, model_dim, x, gq_bw.dx, p_pq.grads, p_bq.grads, tdx);
                                for (size_t i = 0; i < model_dim; ++i) {
                                    d_x[i] = static_cast<Scalar>(static_cast<float>(d_x[i]) + static_cast<float>(tdx[i]));
                                }
                            }
                            {
                                arch::Vector tdx(model_dim, static_cast<Scalar>(0.0f));
                                linear_backward(p_gq.values, qk_dim, model_dim, x, gq_bw.dgate, p_gq.grads, p_bgq.grads, tdx);
                                for (size_t i = 0; i < model_dim; ++i) {
                                    d_x[i] = static_cast<Scalar>(static_cast<float>(d_x[i]) + static_cast<float>(tdx[i]));
                                }
                            }
                            {
                                arch::Vector tdx(model_dim, static_cast<Scalar>(0.0f));
                                linear_backward(p_pk.values, qk_dim, model_dim, x, gk_bw.dx, p_pk.grads, p_bk.grads, tdx);
                                for (size_t i = 0; i < model_dim; ++i) {
                                    d_x[i] = static_cast<Scalar>(static_cast<float>(d_x[i]) + static_cast<float>(tdx[i]));
                                }
                            }
                            {
                                arch::Vector tdx(model_dim, static_cast<Scalar>(0.0f));
                                linear_backward(p_gk.values, qk_dim, model_dim, x, gk_bw.dgate, p_gk.grads, p_bgk.grads, tdx);
                                for (size_t i = 0; i < model_dim; ++i) {
                                    d_x[i] = static_cast<Scalar>(static_cast<float>(d_x[i]) + static_cast<float>(tdx[i]));
                                }
                            }
                            {
                                arch::Vector tdx(model_dim, static_cast<Scalar>(0.0f));
                                linear_backward(p_pv.values, v_dim, model_dim, x, gv_bw.dx, p_pv.grads, p_bv.grads, tdx);
                                for (size_t i = 0; i < model_dim; ++i) {
                                    d_x[i] = static_cast<Scalar>(static_cast<float>(d_x[i]) + static_cast<float>(tdx[i]));
                                }
                            }
                            {
                                arch::Vector tdx(model_dim, static_cast<Scalar>(0.0f));
                                linear_backward(p_gv.values, v_dim, model_dim, x, gv_bw.dgate, p_gv.grads, p_bgv.grads, tdx);
                                for (size_t i = 0; i < model_dim; ++i) {
                                    d_x[i] = static_cast<Scalar>(static_cast<float>(d_x[i]) + static_cast<float>(tdx[i]));
                                }
                            }

                            for (size_t i = 0; i < model_dim; ++i) {
                                p_embed.grads[emb_off + i] = static_cast<Scalar>(
                                    static_cast<float>(p_embed.grads[emb_off + i]) + static_cast<float>(d_x[i]));
                            }

                            ++batch_tokens;
                            ++epoch_token_count;
                            ++objective_evals;
                        }
                        ++epoch_example_count;
                    }

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
                    p_theta.grads[0] = static_cast<Scalar>(static_cast<float>(p_theta.grads[0]) * inv);

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
                    {
                        const float th = static_cast<float>(p_theta.values[0]);
                        if (std::isfinite(th)) {
                            model.set_output_theta(static_cast<Scalar>(std::clamp(th, -cfg_.max_parameter_abs, cfg_.max_parameter_abs)));
                        } else {
                            ++repaired;
                        }
                    }
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

            bool unstable_epoch = false;
            if (cfg_.enable_instability_backoff) {
                unstable_epoch = (nonfinite_skip_ratio >= cfg_.instability_skip_ratio_threshold)
                    || (repaired_ratio >= cfg_.instability_repair_ratio_threshold);

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

            const auto epoch_t1 = std::chrono::steady_clock::now();
            const double epoch_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(epoch_t1 - epoch_t0).count();
            const double epoch_s = std::max(1e-6, epoch_ms / 1000.0);
            const double tokens_per_sec = static_cast<double>(epoch_token_count) / epoch_s;
            const double examples_per_sec = static_cast<double>(epoch_example_count) / epoch_s;
            const uint64_t kernel_launches = llm::arch::sycl_ops::kernel_launch_count();
            const float gradcheck_rel_err = (gradcheck_count == 0)
                ? 0.0f
                : (gradcheck_rel_err_sum / static_cast<float>(gradcheck_count));

            const char* mode_name = (cfg_.mode == TrainMode::FiniteDifference)
                ? "FD"
                : (cfg_.mode == TrainMode::BackpropHeads ? "BP_HEADS" : "BP_FULL");

            std::cout << "Epoch " << (epoch + 1) << "/" << cfg_.epochs
                      << " - Mode: " << mode_name
                      << " - LR: " << lr_epoch
                      << " - BaseLR: " << base_lr_epoch
                      << " - LRScale: " << lr_multiplier
                      << " - FDsamples: " << grad_samples_epoch
                      << " - Loss: " << epoch_loss
                      << " - LossDelta: " << loss_delta
                      << " - LossEMA: " << loss_ema
                      << " - FDms: " << fd_ms
                      << " - SeqMs: " << seq_forward_ms
                      << " - ObjEvals: " << objective_evals
                      << " - QueryEvents: " << query_events
                      << " - EpochMs: " << epoch_ms
                      << " - Tok/s: " << tokens_per_sec
                      << " - Ex/s: " << examples_per_sec
                      << " - Kernels: " << kernel_launches
                      << " - GradCheckRelErr: " << gradcheck_rel_err
                      << " - ActiveGrad%: " << (100.0f * active_grad_fraction)
                      << " - NonFiniteSkips: " << nonfinite_skips
                      << " - Skip%: " << (100.0f * nonfinite_skip_ratio);
            if (repaired > 0) {
                std::cout << " - Repaired: " << repaired;
            }
            if (unstable_epoch) {
                std::cout << " - LRBackoff";
            }
            std::cout << "\n";
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
