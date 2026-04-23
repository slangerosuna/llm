#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace llm::training {

struct ParameterTensor {
	std::vector<float> values;
	std::vector<float> grads;
};

class SGD {
	float lr_;
	float weight_decay_;

public:
	explicit SGD(float learning_rate = 1e-3f, float weight_decay = 0.0f)
	  : lr_(learning_rate), weight_decay_(weight_decay) {}

	void step(std::vector<ParameterTensor>& params) const {
		for (auto& p : params) {
			if (p.values.size() != p.grads.size()) {
				throw std::runtime_error("SGD parameter/gradient shape mismatch");
			}
			for (size_t i = 0; i < p.values.size(); ++i) {
				const float wd_term = weight_decay_ * p.values[i];
				p.values[i] -= lr_ * (p.grads[i] + wd_term);
			}
		}
	}
};

class AdamW {
	float lr_;
	float beta1_;
	float beta2_;
	float eps_;
	float weight_decay_;
	size_t t_;
	std::vector<std::vector<float>> m_;
	std::vector<std::vector<float>> v_;

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
				m_[pidx].assign(p.values.size(), 0.0f);
				v_[pidx].assign(p.values.size(), 0.0f);
			}

			for (size_t i = 0; i < p.values.size(); ++i) {
				const float g = p.grads[i];
				m_[pidx][i] = beta1_ * m_[pidx][i] + (1.0f - beta1_) * g;
				v_[pidx][i] = beta2_ * v_[pidx][i] + (1.0f - beta2_) * g * g;

				const float m_hat = m_[pidx][i] / b1_corr;
				const float v_hat = v_[pidx][i] / b2_corr;

				p.values[i] *= (1.0f - lr_ * weight_decay_);
				p.values[i] -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
			}
		}
	}
};

} // namespace llm::training