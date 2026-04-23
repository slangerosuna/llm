#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include <architecture/sequential.hpp>

namespace llm::arch {

struct AttentionMemory {
	Matrix keys;
	Matrix values;
};

class Attention {
	float prune_ratio_;

	static float dot(const Vector& a, const Vector& b) {
		return static_cast<float>(sycl_ops::dot(a, b));
	}

	static std::vector<float> softmax(const std::vector<float>& logits) {
		if (logits.empty()) {
			return {};
		}
		const float m = *std::max_element(logits.begin(), logits.end());
		std::vector<float> exps(logits.size());
		float sum = 0.0f;
		for (size_t i = 0; i < logits.size(); ++i) {
			exps[i] = std::exp(logits[i] - m);
			sum += exps[i];
		}
		for (float& v : exps) {
			v /= sum;
		}
		return exps;
	}

	std::vector<size_t> prune_indices(const std::vector<float>& scores) const {
		std::vector<size_t> idx(scores.size());
		std::iota(idx.begin(), idx.end(), 0);
		const size_t keep = std::max<size_t>(1, static_cast<size_t>(scores.size() * (1.0f - prune_ratio_)));
		std::partial_sort(idx.begin(), idx.begin() + keep, idx.end(), [&](size_t a, size_t b) {
			return scores[a] > scores[b];
		});
		idx.resize(keep);
		return idx;
	}

public:
	explicit Attention(float prune_ratio = 0.8f) : prune_ratio_(std::clamp(prune_ratio, 0.0f, 0.99f)) {}

	Matrix compute(
		const Matrix& query,
		const AttentionMemory& current,
		const AttentionMemory& long_term,
		const AttentionMemory& recurrent_window) const {
		Matrix all_k = current.keys;
		Matrix all_v = current.values;

		all_k.insert(all_k.end(), long_term.keys.begin(), long_term.keys.end());
		all_v.insert(all_v.end(), long_term.values.begin(), long_term.values.end());
		all_k.insert(all_k.end(), recurrent_window.keys.begin(), recurrent_window.keys.end());
		all_v.insert(all_v.end(), recurrent_window.values.begin(), recurrent_window.values.end());

		if (all_k.size() != all_v.size()) {
			throw std::runtime_error("Attention K/V size mismatch");
		}
		if (all_k.empty()) {
			return Matrix(query.size(), Vector(query.empty() ? 0 : query.front().size(), static_cast<Scalar>(0.0f)));
		}

		const float scale = 1.0f / std::sqrt(static_cast<float>(query.front().size()));
		Matrix out(query.size(), Vector(all_v.front().size(), static_cast<Scalar>(0.0f)));

		for (size_t qi = 0; qi < query.size(); ++qi) {
			std::vector<float> scores(all_k.size());
			for (size_t k = 0; k < all_k.size(); ++k) {
				scores[k] = dot(query[qi], all_k[k]) * scale;
			}

			const std::vector<size_t> kept = prune_indices(scores);
			std::vector<float> kept_scores;
			kept_scores.reserve(kept.size());
			for (size_t i : kept) {
				kept_scores.push_back(scores[i]);
			}

			const std::vector<float> probs = softmax(kept_scores);
			for (size_t i = 0; i < kept.size(); ++i) {
				const Vector& v = all_v[kept[i]];
				for (size_t d = 0; d < v.size(); ++d) {
					const float acc = static_cast<float>(out[qi][d]) + probs[i] * static_cast<float>(v[d]);
					out[qi][d] = static_cast<Scalar>(acc);
				}
			}
		}

		return out;
	}
};

} // namespace llm::arch