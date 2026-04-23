#pragma once

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace llm::arch::activation {

inline float sigmoid(float x) {
	if (x >= 0.0f) {
		const float z = std::exp(-x);
		return 1.0f / (1.0f + z);
	}
	const float z = std::exp(x);
	return z / (1.0f + z);
}

inline float softplus(float x) {
	if (x > 20.0f) {
		return x;
	}
	return std::log1p(std::exp(x));
}

// SwiGLU(x, g) = (x * sigmoid(x)) * g
inline std::vector<float> swiglu(const std::vector<float>& x, const std::vector<float>& gate) {
	if (x.size() != gate.size()) {
		throw std::runtime_error("SwiGLU shape mismatch");
	}

	std::vector<float> out(x.size());
	for (size_t i = 0; i < x.size(); ++i) {
		out[i] = (x[i] * sigmoid(x[i])) * gate[i];
	}
	return out;
}

struct SwiGLUGrad {
	std::vector<float> dx;
	std::vector<float> dgate;
};

inline SwiGLUGrad dswiglu(
	const std::vector<float>& x,
	const std::vector<float>& gate,
	const std::vector<float>& dout) {
	if (x.size() != gate.size() || x.size() != dout.size()) {
		throw std::runtime_error("dSwiGLU shape mismatch");
	}

	SwiGLUGrad grad{std::vector<float>(x.size()), std::vector<float>(x.size())};
	for (size_t i = 0; i < x.size(); ++i) {
		const float s = sigmoid(x[i]);
		const float swish = x[i] * s;
		const float dswish_dx = s + x[i] * s * (1.0f - s);
		grad.dx[i] = dout[i] * gate[i] * dswish_dx;
		grad.dgate[i] = dout[i] * swish;
	}
	return grad;
}

// ParamTanh(x; theta, eps) = tanh((softplus(theta) + eps) * x)
inline std::vector<float> param_tanh(const std::vector<float>& x, float theta, float epsilon = 1e-6f) {
	const float a = softplus(theta) + epsilon;
	std::vector<float> out(x.size());
	for (size_t i = 0; i < x.size(); ++i) {
		out[i] = std::tanh(a * x[i]);
	}
	return out;
}

struct ParamTanhGrad {
	std::vector<float> dx;
	float dtheta;
};

inline ParamTanhGrad dparam_tanh(
	const std::vector<float>& x,
	float theta,
	const std::vector<float>& dout,
	float epsilon = 1e-6f) {
	if (x.size() != dout.size()) {
		throw std::runtime_error("dParamTanh shape mismatch");
	}

	const float sp = softplus(theta);
	const float a = sp + epsilon;
	const float da_dtheta = sigmoid(theta);

	ParamTanhGrad grad{std::vector<float>(x.size()), 0.0f};
	for (size_t i = 0; i < x.size(); ++i) {
		const float t = std::tanh(a * x[i]);
		const float sech2 = 1.0f - t * t;
		grad.dx[i] = dout[i] * sech2 * a;
		grad.dtheta += dout[i] * sech2 * x[i] * da_dtheta;
	}

	return grad;
}

} // namespace llm::arch::activation