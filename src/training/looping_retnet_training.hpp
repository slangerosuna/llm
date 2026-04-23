#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
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

struct TrainConfig {
    size_t epochs = 5;
    float learning_rate = 1e-2f;      // base LR at epoch 1
    float min_learning_rate = 1e-4f;  // final LR floor
    float weight_decay = 1e-4f;

    // Finite-difference gradient estimate over randomly sampled coordinates.
    float fd_eps = 1e-2f;
    size_t grad_coordinate_samples = 256;

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

    float learning_rate_for_epoch(size_t epoch_1_based) const {
        if (cfg_.epochs <= 1) {
            return cfg_.learning_rate;
        }
        const float t = static_cast<float>(epoch_1_based - 1)
            / static_cast<float>(cfg_.epochs - 1);
        constexpr float kPi = 3.14159265358979323846f;
        const float cos_decay = 0.5f * (1.0f + std::cos(kPi * t));
        return cfg_.min_learning_rate + (cfg_.learning_rate - cfg_.min_learning_rate) * cos_decay;
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
        float max_logit = static_cast<float>(logits[0]);
        for (arch::Scalar v : logits) {
            max_logit = std::max(max_logit, static_cast<float>(v));
        }
        float sum_exp = 0.0f;
        for (arch::Scalar v : logits) {
            sum_exp += std::exp(static_cast<float>(v) - max_logit);
        }
        const float lp = static_cast<float>(logits[target_class]) - max_logit - std::log(sum_exp);
        return -lp;
    }

    static float cross_entropy_from_logits(const arch::Vector& logits, uint8_t target) {
        if (logits.empty()) {
            throw std::runtime_error("LoopingRetNetSGDTrainer: empty logits");
        }
        const size_t cls = static_cast<size_t>(target);
        if (cls >= logits.size()) {
            throw std::runtime_error("LoopingRetNetSGDTrainer: target out of range");
        }

        float max_logit = static_cast<float>(logits[0]);
        for (arch::Scalar v : logits) {
            max_logit = std::max(max_logit, static_cast<float>(v));
        }

        float sum_exp = 0.0f;
        for (arch::Scalar v : logits) {
            sum_exp += std::exp(static_cast<float>(v) - max_logit);
        }

        const float log_prob = static_cast<float>(logits[cls]) - max_logit - std::log(sum_exp);
        return -log_prob;
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
            }
        }

        return loss / static_cast<float>(steps);
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
            total += sequence_loss(model, ex, rng);
        }
        return total / static_cast<float>(dataset.size());
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

        std::mt19937 rng(cfg_.seed);

        std::vector<EpochResult> history;
        history.reserve(cfg_.epochs);

        for (size_t epoch = 0; epoch < cfg_.epochs; ++epoch) {
            const float lr_epoch = learning_rate_for_epoch(epoch + 1);
            SGD optimizer(lr_epoch, cfg_.weight_decay);

            std::vector<Scalar*> refs = model.parameter_references();
            std::vector<Scalar> base(refs.size(), static_cast<Scalar>(0.0f));
            for (size_t i = 0; i < refs.size(); ++i) {
                base[i] = *refs[i];
            }

            ParameterTensor tensor;
            tensor.values = base;
            tensor.grads.assign(base.size(), static_cast<Scalar>(0.0f));

            std::uniform_int_distribution<size_t> idx_dist(0, base.size() - 1);

            // Coordinate-sampled finite differences over full objective,
            // including memory insertion/query behavior.
            for (size_t s = 0; s < cfg_.grad_coordinate_samples; ++s) {
                const size_t idx = idx_dist(rng);
                const Scalar original = *refs[idx];

                *refs[idx] = static_cast<Scalar>(static_cast<float>(original) + cfg_.fd_eps);
                const float l_plus = dataset_loss(model, dataset, rng);

                *refs[idx] = static_cast<Scalar>(static_cast<float>(original) - cfg_.fd_eps);
                const float l_minus = dataset_loss(model, dataset, rng);

                *refs[idx] = original;

                const float g = (l_plus - l_minus) / (2.0f * cfg_.fd_eps);
                tensor.grads[idx] = static_cast<Scalar>(
                    static_cast<float>(tensor.grads[idx]) + g);
            }

            std::vector<ParameterTensor> params{tensor};
            optimizer.step(params);

            for (size_t i = 0; i < refs.size(); ++i) {
                *refs[i] = params[0].values[i];
            }

            const float epoch_loss = dataset_loss(model, dataset, rng);
            history.push_back(EpochResult{epoch + 1, epoch_loss});

            std::cout << "Epoch " << (epoch + 1) << "/" << cfg_.epochs
                      << " - LR: " << lr_epoch
                      << " - Loss: " << epoch_loss << "\n";
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
