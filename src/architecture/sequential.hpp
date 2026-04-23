#pragma once

#include <algorithm>
#include <future>
#include <stdexcept>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

namespace llm::arch {

#if defined(__STDCPP_FLOAT16_T__)
using Scalar = std::float16_t;
#elif defined(__FLT16_MANT_DIG__)
using Scalar = _Float16;
#else
#error "float16_t is not supported by this compiler"
#endif
using Vector = std::vector<Scalar>;
using Matrix = std::vector<Vector>;

namespace sycl_ops {

inline sycl::queue& default_queue() {
	static sycl::queue q{sycl::default_selector_v};
	return q;
}

inline Scalar dot(const Vector& a, const Vector& b) {
	if (a.size() != b.size()) {
		throw std::runtime_error("SYCL dot shape mismatch");
	}
	if (a.empty()) {
		return static_cast<Scalar>(0.0f);
	}

	std::vector<float> partial(a.size(), 0.0f);
	sycl::buffer<Scalar> a_buf(a.data(), sycl::range<1>(a.size()));
	sycl::buffer<Scalar> b_buf(b.data(), sycl::range<1>(b.size()));
	sycl::buffer<float> p_buf(partial.data(), sycl::range<1>(partial.size()));

	default_queue().submit([&](sycl::handler& h) {
		auto a_acc = a_buf.get_access<sycl::access::mode::read>(h);
		auto b_acc = b_buf.get_access<sycl::access::mode::read>(h);
		auto p_acc = p_buf.get_access<sycl::access::mode::discard_write>(h);
		h.parallel_for(sycl::range<1>(a.size()), [=](sycl::id<1> i) {
			const size_t idx = i[0];
			p_acc[i] = static_cast<float>(a_acc[i]) * static_cast<float>(b_acc[i]);
		});
	});
	default_queue().wait();

	float sum = 0.0f;
	for (float v : partial) {
		sum += v;
	}
	return static_cast<Scalar>(sum);
}

inline Vector linear(
	const std::vector<Scalar>& w,
	size_t out_dim,
	size_t in_dim,
	const Vector& x,
	const std::vector<Scalar>& b) {
	if (x.size() != in_dim || w.size() != out_dim * in_dim || b.size() != out_dim) {
		throw std::runtime_error("SYCL linear shape mismatch");
	}

	Vector y(out_dim, static_cast<Scalar>(0.0f));
	sycl::buffer<Scalar> w_buf(w.data(), sycl::range<1>(w.size()));
	sycl::buffer<Scalar> x_buf(x.data(), sycl::range<1>(x.size()));
	sycl::buffer<Scalar> b_buf(b.data(), sycl::range<1>(b.size()));
	sycl::buffer<Scalar> y_buf(y.data(), sycl::range<1>(y.size()));

	default_queue().submit([&](sycl::handler& h) {
		auto w_acc = w_buf.get_access<sycl::access::mode::read>(h);
		auto x_acc = x_buf.get_access<sycl::access::mode::read>(h);
		auto b_acc = b_buf.get_access<sycl::access::mode::read>(h);
		auto y_acc = y_buf.get_access<sycl::access::mode::discard_write>(h);
		h.parallel_for(sycl::range<1>(out_dim), [=](sycl::id<1> o) {
			const size_t out_i = o[0];
			float s = static_cast<float>(b_acc[o]);
			for (size_t i = 0; i < in_dim; ++i) {
				s += static_cast<float>(w_acc[out_i * in_dim + i]) * static_cast<float>(x_acc[i]);
			}
			y_acc[o] = static_cast<Scalar>(s);
		});
	});
	default_queue().wait();

	return y;
}

} // namespace sycl_ops

class Layer {
public:
	virtual ~Layer() = default;
	virtual Matrix forward(const Matrix& input) = 0;
};

class SequentialModel {
protected:
	std::vector<Layer*> layers_;

public:
	virtual ~SequentialModel() = default;

	void add_layer(Layer* layer) {
		layers_.push_back(layer);
	}

	virtual Matrix forward(const Matrix& input) const {
		Matrix out = input;
		for (Layer* layer : layers_) {
			if (layer == nullptr) {
				throw std::runtime_error("Null layer in SequentialModel");
			}
			out = layer->forward(out);
		}
		return out;
	}

	// Queue-style parallelism baseline: fan out independent batch forwards.
	virtual std::vector<Matrix> forward_parallel_batches(
		const std::vector<Matrix>& batches,
		size_t queue_count = 1) const {
		(void)queue_count;
		std::vector<std::future<Matrix>> futures;
		futures.reserve(batches.size());

		for (const Matrix& batch : batches) {
			futures.emplace_back(std::async(std::launch::async, [this, batch]() {
				return this->forward(batch);
			}));
		}

		std::vector<Matrix> outputs;
		outputs.reserve(batches.size());
		for (auto& f : futures) {
			outputs.push_back(f.get());
		}
		return outputs;
	}

	// Pipeline baseline: preserve order while streaming forward one-by-one.
	virtual std::vector<Matrix> forward_pipeline_batches(const std::vector<Matrix>& batches) const {
		std::vector<Matrix> outputs;
		outputs.reserve(batches.size());
		for (const Matrix& batch : batches) {
			outputs.push_back(forward(batch));
		}
		return outputs;
	}
};

} // namespace llm::arch