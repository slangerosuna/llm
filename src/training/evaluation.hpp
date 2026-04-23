#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace llm::training::evaluation {

inline float mean_squared_error(const std::vector<float>& pred, const std::vector<float>& target) {
	if (pred.size() != target.size()) {
		throw std::runtime_error("MSE shape mismatch");
	}
	if (pred.empty()) {
		return 0.0f;
	}

	float sum = 0.0f;
	for (size_t i = 0; i < pred.size(); ++i) {
		const float d = pred[i] - target[i];
		sum += d * d;
	}
	return sum / static_cast<float>(pred.size());
}

inline float cross_entropy_from_probs(const std::vector<float>& probs, size_t target_index, float eps = 1e-8f) {
	if (target_index >= probs.size()) {
		throw std::runtime_error("Cross entropy target out of range");
	}
	const float p = std::max(probs[target_index], eps);
	return -std::log(p);
}

inline float batch_mean_cross_entropy(
	const std::vector<std::vector<float>>& probs_batch,
	const std::vector<size_t>& target_indices,
	float eps = 1e-8f) {
	if (probs_batch.size() != target_indices.size()) {
		throw std::runtime_error("Batch CE shape mismatch");
	}
	if (probs_batch.empty()) {
		return 0.0f;
	}

	float total = 0.0f;
	for (size_t i = 0; i < probs_batch.size(); ++i) {
		total += cross_entropy_from_probs(probs_batch[i], target_indices[i], eps);
	}
	return total / static_cast<float>(probs_batch.size());
}

inline float top1_accuracy(
	const std::vector<std::vector<float>>& logits_batch,
	const std::vector<size_t>& target_indices) {
	if (logits_batch.size() != target_indices.size()) {
		throw std::runtime_error("Accuracy shape mismatch");
	}
	if (logits_batch.empty()) {
		return 0.0f;
	}

	size_t correct = 0;
	for (size_t i = 0; i < logits_batch.size(); ++i) {
		const auto& row = logits_batch[i];
		const size_t pred = static_cast<size_t>(std::distance(row.begin(), std::max_element(row.begin(), row.end())));
		if (pred == target_indices[i]) {
			++correct;
		}
	}

	return static_cast<float>(correct) / static_cast<float>(logits_batch.size());
}

} // namespace llm::training::evaluation