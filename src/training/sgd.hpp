#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <sycl/sycl.hpp>

namespace llm::training {

#if defined(__STDCPP_FLOAT16_T__)
using Scalar = std::float16_t;
#elif defined(__FLT16_MANT_DIG__)
using Scalar = _Float16;
#else
#error "float16_t is not supported by this compiler"
#endif

struct ParameterTensor {
	std::vector<Scalar> values;
	std::vector<Scalar> grads;
};

class SGD {
	float lr_;
	float weight_decay_;

public:
	explicit SGD(float learning_rate = 1e-3f, float weight_decay = 0.0f)
	  : lr_(learning_rate), weight_decay_(weight_decay) {}

	void step(std::vector<ParameterTensor>& params) const {
		sycl::queue q{sycl::default_selector_v};
		const float lr = lr_;
		const float wd = weight_decay_;
		for (auto& p : params) {
			if (p.values.size() != p.grads.size()) {
				throw std::runtime_error("SGD parameter/gradient shape mismatch");
			}

			sycl::buffer<Scalar> v_buf(p.values.data(), sycl::range<1>(p.values.size()));
			sycl::buffer<Scalar> g_buf(p.grads.data(), sycl::range<1>(p.grads.size()));
			q.submit([&](sycl::handler& h) {
				auto v_acc = v_buf.get_access<sycl::access::mode::read_write>(h);
				auto g_acc = g_buf.get_access<sycl::access::mode::read>(h);
				h.parallel_for(sycl::range<1>(p.values.size()), [=](sycl::id<1> i) {
					const float v = static_cast<float>(v_acc[i]);
					const float g = static_cast<float>(g_acc[i]);
					const float updated = v - lr * (g + wd * v);
					v_acc[i] = static_cast<Scalar>(updated);
				});
			});
		}
		q.wait();
	}
};

class AdamW {
	float lr_;
	float beta1_;
	float beta2_;
	float eps_;
	float weight_decay_;
	size_t t_;
	std::vector<std::vector<Scalar>> m_;
	std::vector<std::vector<Scalar>> v_;

public:
	AdamW(
		float learning_rate = 1e-3f,
		float beta1 = 0.9f,
		float beta2 = 0.999f,
		float eps = 1e-8f,
		float weight_decay = 1e-2f)
	  : lr_(learning_rate),
		beta1_(beta1),
		beta2_(beta2),
		eps_(eps),
		weight_decay_(weight_decay),
		t_(0) {}

	void step(std::vector<ParameterTensor>& params) {
		if (m_.size() != params.size()) {
			m_.assign(params.size(), {});
			v_.assign(params.size(), {});
		}

		++t_;
		const float b1_corr = 1.0f - std::pow(beta1_, static_cast<float>(t_));
		const float b2_corr = 1.0f - std::pow(beta2_, static_cast<float>(t_));

		for (size_t pidx = 0; pidx < params.size(); ++pidx) {
			auto& p = params[pidx];
			if (p.values.size() != p.grads.size()) {
				throw std::runtime_error("AdamW parameter/gradient shape mismatch");
			}

			if (m_[pidx].size() != p.values.size()) {
				m_[pidx].assign(p.values.size(), static_cast<Scalar>(0.0f));
				v_[pidx].assign(p.values.size(), static_cast<Scalar>(0.0f));
			}

			for (size_t i = 0; i < p.values.size(); ++i) {
				const float g = static_cast<float>(p.grads[i]);
				const float m = beta1_ * static_cast<float>(m_[pidx][i]) + (1.0f - beta1_) * g;
				const float v = beta2_ * static_cast<float>(v_[pidx][i]) + (1.0f - beta2_) * g * g;

				const float m_hat = m / b1_corr;
				const float v_hat = v / b2_corr;

				float pv = static_cast<float>(p.values[i]);
				pv *= (1.0f - lr_ * weight_decay_);
				pv -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);

				m_[pidx][i] = static_cast<Scalar>(m);
				v_[pidx][i] = static_cast<Scalar>(v);
				p.values[i] = static_cast<Scalar>(pv);
			}
		}
	}
};

} // namespace llm::training