#pragma once

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <future>
#include <string>
#include <stdexcept>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

namespace llm::arch {

#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define LLM_ASAN_BUILD 1
#endif
#endif

#if defined(__SANITIZE_ADDRESS__)
#define LLM_ASAN_BUILD 1
#endif

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

inline std::atomic<uint64_t>& kernel_launch_counter() {
	static std::atomic<uint64_t> counter{0};
	return counter;
}

inline void reset_kernel_launch_counter() {
	kernel_launch_counter().store(0, std::memory_order_relaxed);
}

inline uint64_t kernel_launch_count() {
	return kernel_launch_counter().load(std::memory_order_relaxed);
}

inline bool env_flag_set(const char* name) {
	const char* v = std::getenv(name);
	if (!v) {
		return false;
	}
	return std::string(v) != "0" && std::string(v) != "false" && std::string(v) != "FALSE";
}

inline sycl::queue& default_queue() {
	static sycl::queue q = [] {
		const bool require_gpu = env_flag_set("LLM_REQUIRE_GPU");

		if (require_gpu) {
			return sycl::queue{sycl::gpu_selector_v};
		}

		try {
			return sycl::queue{sycl::gpu_selector_v};
		} catch (...) {
			try {
				return sycl::queue{sycl::default_selector_v};
			} catch (...) {
				throw std::runtime_error(
					"SYCL queue creation failed for both GPU and default selectors. "
					"Set ONEAPI_DEVICE_SELECTOR or SYCL_DEVICE_FILTER to a valid GPU.");
			}
		}
	}();
	return q;
}

inline bool use_sycl_kernels() {
	static bool use = [] {
		if (env_flag_set("LLM_FORCE_CPU_MATH")) {
			return false;
		}
		if (env_flag_set("LLM_FORCE_SYCL_MATH")) {
			return true;
		}

		#if defined(LLM_ASAN_BUILD)
		return false;
		#else
		try {
			return !default_queue().get_device().is_cpu();
		} catch (...) {
			return false;
		}
		#endif
	}();
	return use;
}

inline Scalar dot(const Vector& a, const Vector& b) {
	if (a.size() != b.size()) {
		throw std::runtime_error("SYCL dot shape mismatch");
	}
	if (a.empty()) {
		return static_cast<Scalar>(0.0f);
	}

	if (!use_sycl_kernels()) {
	float result = 0.0f;
	for (size_t i = 0; i < a.size(); ++i) {
		result += static_cast<float>(a[i]) * static_cast<float>(b[i]);
	}
	return static_cast<Scalar>(result);
}

	auto& q = default_queue();
	sycl::buffer<Scalar> a_buf(a.data(), sycl::range<1>(a.size()));
	sycl::buffer<Scalar> b_buf(b.data(), sycl::range<1>(b.size()));
	float result = 0.0f;
	sycl::buffer<float> out_buf(&result, sycl::range<1>(1));

	q.submit([&](sycl::handler& h) {
		auto a_acc = a_buf.get_access<sycl::access::mode::read>(h);
		auto b_acc = b_buf.get_access<sycl::access::mode::read>(h);
		auto sum = sycl::reduction(out_buf, h, sycl::plus<float>());
		h.parallel_for(sycl::range<1>(a.size()), sum, [=](sycl::id<1> i, auto& acc) {
			acc += static_cast<float>(a_acc[i]) * static_cast<float>(b_acc[i]);
		});
	});
	kernel_launch_counter().fetch_add(1, std::memory_order_relaxed);
	q.wait();
	return static_cast<Scalar>(result);
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

	if (!use_sycl_kernels()) {
	for (size_t o = 0; o < out_dim; ++o) {
		float s = static_cast<float>(b[o]);
		for (size_t i = 0; i < in_dim; ++i) {
			s += static_cast<float>(w[o * in_dim + i]) * static_cast<float>(x[i]);
		}
		y[o] = static_cast<Scalar>(s);
	}
	return y;
}

	auto& q = default_queue();
	sycl::buffer<Scalar> w_buf(w.data(), sycl::range<1>(w.size()));
	sycl::buffer<Scalar> x_buf(x.data(), sycl::range<1>(x.size()));
	sycl::buffer<Scalar> b_buf(b.data(), sycl::range<1>(b.size()));
	sycl::buffer<Scalar> y_buf(y.data(), sycl::range<1>(y.size()));

	q.submit([&](sycl::handler& h) {
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
	kernel_launch_counter().fetch_add(1, std::memory_order_relaxed);
	q.wait();

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