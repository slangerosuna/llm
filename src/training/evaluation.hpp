#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <sycl/sycl.hpp>

namespace llm::training::evaluation {

#if defined(__STDCPP_FLOAT16_T__)
using Scalar = std::float16_t;
#elif defined(__FLT16_MANT_DIG__)
using Scalar = _Float16;
#else
#error "float16_t is not supported by this compiler"
#endif

inline float mean_squared_error(const std::vector<Scalar>& pred, const std::vector<Scalar>& target) {
	if (pred.size() != target.size()) {
		throw std::runtime_error("MSE shape mismatch");
	}
	if (pred.empty()) {
		return 0.0f;
	}

	std::vector<float> sq(pred.size(), 0.0f);
	sycl::queue q{sycl::default_selector_v};
	sycl::buffer<Scalar> p_buf(pred.data(), sycl::range<1>(pred.size()));
	sycl::buffer<Scalar> t_buf(target.data(), sycl::range<1>(target.size()));
	sycl::buffer<float> s_buf(sq.data(), sycl::range<1>(sq.size()));

	q.submit([&](sycl::handler& h) {
		auto p_acc = p_buf.get_access<sycl::access::mode::read>(h);
		auto t_acc = t_buf.get_access<sycl::access::mode::read>(h);
		auto s_acc = s_buf.get_access<sycl::access::mode::discard_write>(h);
		h.parallel_for(sycl::range<1>(pred.size()), [=](sycl::id<1> i) {
			const float d = static_cast<float>(p_acc[i]) - static_cast<float>(t_acc[i]);
			s_acc[i] = d * d;
		});
	});
	q.wait();

	float sum = 0.0f;
	for (float v : sq) {
		sum += v;
	}
	return sum / static_cast<float>(pred.size());
}

inline float cross_entropy_from_probs(const std::vector<Scalar>& probs, size_t target_index, float eps = 1e-8f) {
	if (target_index >= probs.size()) {
		throw std::runtime_error("Cross entropy target out of range");
	}
	const float p = std::max(static_cast<float>(probs[target_index]), eps);
	return -std::log(p);
}

inline float batch_mean_cross_entropy(
	const std::vector<std::vector<Scalar>>& probs_batch,
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
	const std::vector<std::vector<Scalar>>& logits_batch,
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
		const size_t pred = static_cast<size_t>(std::distance(
			row.begin(),
			std::max_element(row.begin(), row.end(), [](Scalar a, Scalar b) {
				return static_cast<float>(a) < static_cast<float>(b);
			})));
		if (pred == target_indices[i]) {
			++correct;
		}
	}

	return static_cast<float>(correct) / static_cast<float>(logits_batch.size());
}

} // namespace llm::training::evaluation