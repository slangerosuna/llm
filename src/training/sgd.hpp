#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <sycl/sycl.hpp>

#include <architecture/sequential.hpp>

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
		auto& q = llm::arch::sycl_ops::default_queue();
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
		auto& q = llm::arch::sycl_ops::default_queue();
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

			sycl::buffer<Scalar> val_buf(p.values.data(), sycl::range<1>(p.values.size()));
			sycl::buffer<Scalar> grad_buf(p.grads.data(), sycl::range<1>(p.grads.size()));
			sycl::buffer<Scalar> m_buf(m_[pidx].data(), sycl::range<1>(m_[pidx].size()));
			sycl::buffer<Scalar> v_buf(v_[pidx].data(), sycl::range<1>(v_[pidx].size()));

			const float lr = lr_;
			const float b1 = beta1_;
			const float b2 = beta2_;
			const float eps = eps_;
			const float wd = weight_decay_;

			q.submit([&](sycl::handler& h) {
				auto val_acc = val_buf.get_access<sycl::access::mode::read_write>(h);
				auto grad_acc = grad_buf.get_access<sycl::access::mode::read>(h);
				auto m_acc = m_buf.get_access<sycl::access::mode::read_write>(h);
				auto v_acc = v_buf.get_access<sycl::access::mode::read_write>(h);

				h.parallel_for(sycl::range<1>(p.values.size()), [=](sycl::id<1> i) {
					const float g = static_cast<float>(grad_acc[i]);
					const float m = b1 * static_cast<float>(m_acc[i]) + (1.0f - b1) * g;
					const float v = b2 * static_cast<float>(v_acc[i]) + (1.0f - b2) * g * g;

					const float m_hat = m / b1_corr;
					const float v_hat = v / b2_corr;

					float pv = static_cast<float>(val_acc[i]);
					pv *= (1.0f - lr * wd);
					pv -= lr * m_hat / (sycl::sqrt(v_hat) + eps);

					m_acc[i] = static_cast<Scalar>(m);
					v_acc[i] = static_cast<Scalar>(v);
					val_acc[i] = static_cast<Scalar>(pv);
				});
			});
		}

		q.wait();
	}
};

} // namespace llm::training