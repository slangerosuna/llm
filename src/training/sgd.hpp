#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <sycl/sycl.hpp>

#include <architecture/sequential.hpp>

namespace llm::training {

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

struct ParameterTensor {
	std::vector<Scalar> values;
	std::vector<Scalar> grads;
};

class SGD {
	float lr_;
	float weight_decay_;
	float momentum_;
	std::vector<std::vector<Scalar>> velocity_;

public:
	explicit SGD(
		float learning_rate = 1e-3f,
		float weight_decay = 0.0f,
		float momentum = 0.0f)
	  : lr_(learning_rate), weight_decay_(weight_decay), momentum_(momentum) {}

	void set_learning_rate(float learning_rate) { lr_ = learning_rate; }

	float learning_rate() const { return lr_; }

	void step(std::vector<ParameterTensor>& params) {
		if (momentum_ > 0.0f) {
			if (velocity_.size() != params.size()) {
				velocity_.assign(params.size(), {});
			}
			for (size_t pidx = 0; pidx < params.size(); ++pidx) {
				if (velocity_[pidx].size() != params[pidx].values.size()) {
					velocity_[pidx].assign(params[pidx].values.size(), static_cast<Scalar>(0.0f));
				}
			}
		}

	if (!llm::arch::sycl_ops::use_sycl_kernels()) {
		for (size_t pidx = 0; pidx < params.size(); ++pidx) {
			auto& p = params[pidx];
			if (p.values.size() != p.grads.size()) {
				throw std::runtime_error("SGD parameter/gradient shape mismatch");
			}
			for (size_t i = 0; i < p.values.size(); ++i) {
				const float v = static_cast<float>(p.values[i]);
				const float g = static_cast<float>(p.grads[i]);
				float update = g;
				if (momentum_ > 0.0f) {
					float vel = static_cast<float>(velocity_[pidx][i]);
					vel = momentum_ * vel + g;
					velocity_[pidx][i] = static_cast<Scalar>(vel);
					update = vel;
				}
				const float updated = v - lr_ * (update + weight_decay_ * v);
				p.values[i] = static_cast<Scalar>(updated);
			}
		}
		return;
	}
		auto& q = llm::arch::sycl_ops::default_queue();
		const float lr = lr_;
		const float wd = weight_decay_;
		const float momentum = momentum_;
		for (size_t pidx = 0; pidx < params.size(); ++pidx) {
			auto& p = params[pidx];
			if (p.values.size() != p.grads.size()) {
				throw std::runtime_error("SGD parameter/gradient shape mismatch");
			}

			sycl::buffer<Scalar> v_buf(p.values.data(), sycl::range<1>(p.values.size()));
			sycl::buffer<Scalar> g_buf(p.grads.data(), sycl::range<1>(p.grads.size()));
			sycl::buffer<Scalar> vel_buf(
				(momentum > 0.0f) ? velocity_[pidx].data() : p.grads.data(),
				sycl::range<1>(p.values.size()));
			q.submit([&](sycl::handler& h) {
				auto v_acc = v_buf.get_access<sycl::access::mode::read_write>(h);
				auto g_acc = g_buf.get_access<sycl::access::mode::read>(h);
				auto vel_acc = vel_buf.get_access<sycl::access::mode::read_write>(h);
				h.parallel_for(sycl::range<1>(p.values.size()), [=](sycl::id<1> i) {
					const float v = static_cast<float>(v_acc[i]);
					const float g = static_cast<float>(g_acc[i]);
					float update = g;
					if (momentum > 0.0f) {
						const float prev_vel = static_cast<float>(vel_acc[i]);
						const float new_vel = momentum * prev_vel + g;
						vel_acc[i] = static_cast<Scalar>(new_vel);
						update = new_vel;
					}
					const float updated = v - lr * (update + wd * v);
					v_acc[i] = static_cast<Scalar>(updated);
				});
			});
		}
		q.wait();
	}
};

class Adam {
	float lr_;
	float beta1_;
	float beta2_;
	float eps_;
	float weight_decay_;
	size_t t_;
	std::vector<std::vector<float>> m_;
	std::vector<std::vector<float>> v_;

public:
	explicit Adam(
		float learning_rate = 1e-3f,
		float beta1 = 0.9f,
		float beta2 = 0.999f,
		float eps = 1e-8f,
		float weight_decay = 0.0f)
	  : lr_(learning_rate),
		beta1_(beta1),
		beta2_(beta2),
		eps_(eps),
		weight_decay_(weight_decay),
		t_(0) {}

	void set_learning_rate(float learning_rate) { lr_ = learning_rate; }

	float learning_rate() const { return lr_; }

	void step(std::vector<ParameterTensor>& params) {
		if (m_.size() != params.size()) {
			m_.assign(params.size(), {});
			v_.assign(params.size(), {});
		}

		++t_;
		const float b1_corr = 1.0f - std::pow(beta1_, static_cast<float>(t_));
		const float b2_corr = 1.0f - std::pow(beta2_, static_cast<float>(t_));

		if (!llm::arch::sycl_ops::use_sycl_kernels()) {
			for (size_t pidx = 0; pidx < params.size(); ++pidx) {
				auto& p = params[pidx];
				if (p.values.size() != p.grads.size()) {
					throw std::runtime_error("Adam parameter/gradient shape mismatch");
				}

				if (m_[pidx].size() != p.values.size()) {
					m_[pidx].assign(p.values.size(), 0.0f);
					v_[pidx].assign(p.values.size(), 0.0f);
				}

				for (size_t i = 0; i < p.values.size(); ++i) {
					const float pv = static_cast<float>(p.values[i]);
					const float grad = static_cast<float>(p.grads[i]) + weight_decay_ * pv;
					const float m = beta1_ * m_[pidx][i] + (1.0f - beta1_) * grad;
					const float v = beta2_ * v_[pidx][i] + (1.0f - beta2_) * grad * grad;

					const float m_hat = m / b1_corr;
					const float v_hat = v / b2_corr;
					const float updated = pv - lr_ * m_hat / (std::sqrt(v_hat) + eps_);

					m_[pidx][i] = m;
					v_[pidx][i] = v;
					p.values[i] = static_cast<Scalar>(updated);
				}
			}
			return;
		}

		auto& q = llm::arch::sycl_ops::default_queue();
		const float lr = lr_;
		const float b1 = beta1_;
		const float b2 = beta2_;
		const float eps = eps_;
		const float wd = weight_decay_;

		for (size_t pidx = 0; pidx < params.size(); ++pidx) {
			auto& p = params[pidx];
			if (p.values.size() != p.grads.size()) {
				throw std::runtime_error("Adam parameter/gradient shape mismatch");
			}

			if (m_[pidx].size() != p.values.size()) {
				m_[pidx].assign(p.values.size(), 0.0f);
				v_[pidx].assign(p.values.size(), 0.0f);
			}

			sycl::buffer<Scalar> val_buf(p.values.data(), sycl::range<1>(p.values.size()));
			sycl::buffer<Scalar> grad_buf(p.grads.data(), sycl::range<1>(p.grads.size()));
			sycl::buffer<float> m_buf(m_[pidx].data(), sycl::range<1>(m_[pidx].size()));
			sycl::buffer<float> v_buf(v_[pidx].data(), sycl::range<1>(v_[pidx].size()));

			q.submit([&](sycl::handler& h) {
				auto val_acc = val_buf.get_access<sycl::access::mode::read_write>(h);
				auto grad_acc = grad_buf.get_access<sycl::access::mode::read>(h);
				auto m_acc = m_buf.get_access<sycl::access::mode::read_write>(h);
				auto v_acc = v_buf.get_access<sycl::access::mode::read_write>(h);

				h.parallel_for(sycl::range<1>(p.values.size()), [=](sycl::id<1> i) {
					const float pv = static_cast<float>(val_acc[i]);
					const float grad = static_cast<float>(grad_acc[i]) + wd * pv;
					const float m = b1 * m_acc[i] + (1.0f - b1) * grad;
					const float v = b2 * v_acc[i] + (1.0f - b2) * grad * grad;

					const float m_hat = m / b1_corr;
					const float v_hat = v / b2_corr;
					const float updated = pv - lr * m_hat / (sycl::sqrt(v_hat) + eps);

					m_acc[i] = m;
					v_acc[i] = v;
					val_acc[i] = static_cast<Scalar>(updated);
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
		if (!llm::arch::sycl_ops::use_sycl_kernels()) {
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
		return;
	}
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