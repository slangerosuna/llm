#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include <architecture/sequential.hpp>

namespace llm::arch {

// AttentionMemory stores keys, relations (smaller semantic edge labels),
// values, and per-entry prune_scores.
// Entries loaded from long-term graph memory receive a freshness penalty
// (higher prune_score) so they are evicted first, limiting compute waste.
//
// AttentionMemory is used *exclusively* as the communication channel
// between the RetNet recurrent state and the long-term graph database.
// It is never used to extend context within the same sequence.
struct AttentionMemory {
    Matrix keys;
    Matrix relations; // smaller edge-label per key/value pair
    Matrix values;
    std::vector<float> prune_scores; // higher = pruned sooner

    void push_back(Vector key, Vector relation, Vector value, float prune_score = 0.0f) {
        keys.push_back(std::move(key));
        relations.push_back(std::move(relation));
        values.push_back(std::move(value));
        prune_scores.push_back(prune_score);
    }

    size_t size() const { return keys.size(); }
    bool empty() const { return keys.empty(); }

    void prune_to_max(size_t max_entries) {
        if (max_entries == 0) {
            keys.clear();
            relations.clear();
            values.clear();
            prune_scores.clear();
            return;
        }

        const size_t n = size();
        if (n <= max_entries) {
            return;
        }

        std::vector<size_t> keep(n);
        std::iota(keep.begin(), keep.end(), 0);
        std::partial_sort(keep.begin(), keep.begin() + static_cast<std::ptrdiff_t>(max_entries), keep.end(),
            [&](size_t a, size_t b) {
                const float score_a = a < prune_scores.size() ? prune_scores[a] : std::numeric_limits<float>::infinity();
                const float score_b = b < prune_scores.size() ? prune_scores[b] : std::numeric_limits<float>::infinity();
                if (score_a != score_b) {
                    return score_a < score_b;
                }
                return a > b;
            });
        keep.resize(max_entries);
        std::sort(keep.begin(), keep.end());

        Matrix new_keys;
        Matrix new_relations;
        Matrix new_values;
        std::vector<float> new_prune_scores;
        new_keys.reserve(max_entries);
        new_relations.reserve(max_entries);
        new_values.reserve(max_entries);
        new_prune_scores.reserve(max_entries);

        for (size_t index : keep) {
            new_keys.push_back(std::move(keys[index]));
            new_relations.push_back(std::move(relations[index]));
            new_values.push_back(std::move(values[index]));
            new_prune_scores.push_back(prune_scores[index]);
        }

        keys = std::move(new_keys);
        relations = std::move(new_relations);
        values = std::move(new_values);
        prune_scores = std::move(new_prune_scores);
    }
};

class Attention {
    float prune_ratio_;

    static float dot(const Vector& a, const Vector& b) {
        return static_cast<float>(sycl_ops::dot(a, b));
    }

    static std::vector<float> softmax(const std::vector<float>& logits) {
        if (logits.empty()) { return {}; }
        const float m = *std::max_element(logits.begin(), logits.end());
        std::vector<float> exps(logits.size());
        float sum = 0.0f;
        for (size_t i = 0; i < logits.size(); ++i) {
            exps[i] = std::exp(logits[i] - m);
            sum += exps[i];
        }
        for (float& v : exps) { v /= sum; }
        return exps;
    }

    // Rank by (score - penalty); entries with high penalties (freshly loaded
    // graph results) are deprioritised and pruned first.
    std::vector<size_t> prune_indices(
        const std::vector<float>& scores,
        const std::vector<float>& penalties) const
    {
        const size_t n = scores.size();
        std::vector<size_t> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        const size_t keep = std::max<size_t>(1, static_cast<size_t>(n * (1.0f - prune_ratio_)));
        std::partial_sort(idx.begin(), idx.begin() + keep, idx.end(), [&](size_t a, size_t b) {
            const float pa = a < penalties.size() ? penalties[a] : 0.0f;
            const float pb = b < penalties.size() ? penalties[b] : 0.0f;
            return (scores[a] - pa) > (scores[b] - pb);
        });
        idx.resize(keep);
        return idx;
    }

public:
    explicit Attention(float prune_ratio = 0.8f)
        : prune_ratio_(std::clamp(prune_ratio, 0.0f, 0.99f)) {}

    // current: projected from the current token step (prune_score = 0).
    // long_term: loaded from graph via MultiHopQuery (may carry fresh penalties).
    Matrix compute(
        const Matrix& query,
        const AttentionMemory& current,
        const AttentionMemory& long_term) const
    {
        Matrix all_k = current.keys;
        Matrix all_v = current.values;
        std::vector<float> all_penalties = current.prune_scores;

        all_k.insert(all_k.end(), long_term.keys.begin(), long_term.keys.end());
        all_v.insert(all_v.end(), long_term.values.begin(), long_term.values.end());
        all_penalties.insert(all_penalties.end(),
            long_term.prune_scores.begin(), long_term.prune_scores.end());

        if (all_k.size() != all_v.size()) {
            throw std::runtime_error("Attention K/V size mismatch");
        }
        if (all_k.empty()) {
            const size_t dim = query.empty() ? 0 : query.front().size();
            return Matrix(query.size(), Vector(dim, static_cast<Scalar>(0.0f)));
        }

        const float scale = 1.0f / std::sqrt(static_cast<float>(query.front().size()));
        Matrix out(query.size(), Vector(all_v.front().size(), static_cast<Scalar>(0.0f)));

        for (size_t qi = 0; qi < query.size(); ++qi) {
            std::vector<float> scores(all_k.size());
            for (size_t ki = 0; ki < all_k.size(); ++ki) {
                scores[ki] = dot(query[qi], all_k[ki]) * scale;
            }

            const std::vector<size_t> kept = prune_indices(scores, all_penalties);
            std::vector<float> kept_scores;
            kept_scores.reserve(kept.size());
            for (size_t i : kept) { kept_scores.push_back(scores[i]); }

            const std::vector<float> probs = softmax(kept_scores);
            for (size_t i = 0; i < kept.size(); ++i) {
                const Vector& v = all_v[kept[i]];
                for (size_t d = 0; d < v.size(); ++d) {
                    const float acc = static_cast<float>(out[qi][d])
                        + probs[i] * static_cast<float>(v[d]);
                    out[qi][d] = static_cast<Scalar>(acc);
                }
            }
        }

        return out;
    }
};

} // namespace llm::arch
