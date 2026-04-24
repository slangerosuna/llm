#pragma once

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#include <architecture/sequential.hpp>

namespace llm::arch::activation {

#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define LLM_ASAN_BUILD 1
#endif
#endif

#if defined(__SANITIZE_ADDRESS__)
#define LLM_ASAN_BUILD 1
#endif

inline Scalar sigmoid(Scalar x) {
	const float xf = static_cast<float>(x);
	if (x >= 0.0f) {
		const float z = std::exp(-xf);
		return static_cast<Scalar>(1.0f / (1.0f + z));
	}
	const float z = std::exp(xf);
	return static_cast<Scalar>(z / (1.0f + z));
}

inline Scalar softplus(Scalar x) {
	const float xf = static_cast<float>(x);
	if (xf > 20.0f) {
		return x;
	}
	return static_cast<Scalar>(std::log1p(std::exp(xf)));
}

// SwiGLU(x, g) = (x * sigmoid(x)) * g
inline Vector swiglu(const Vector& x, const Vector& gate) {
	if (x.size() != gate.size()) {
		throw std::runtime_error("SwiGLU shape mismatch");
	}

	Vector out(x.size(), static_cast<Scalar>(0.0f));

	if (!sycl_ops::use_sycl_kernels()) {
	for (size_t i = 0; i < x.size(); ++i) {
		const float xv = static_cast<float>(x[i]);
		const float gv = static_cast<float>(gate[i]);
		const float s = 1.0f / (1.0f + std::exp(-xv));
		out[i] = static_cast<Scalar>((xv * s) * gv);
	}
	return out;
}

	sycl::buffer<Scalar> x_buf(x.data(), sycl::range<1>(x.size()));
	sycl::buffer<Scalar> g_buf(gate.data(), sycl::range<1>(gate.size()));
	sycl::buffer<Scalar> o_buf(out.data(), sycl::range<1>(out.size()));

	sycl_ops::default_queue().submit([&](sycl::handler& h) {
		auto x_acc = x_buf.get_access<sycl::access::mode::read>(h);
		auto g_acc = g_buf.get_access<sycl::access::mode::read>(h);
		auto o_acc = o_buf.get_access<sycl::access::mode::discard_write>(h);
		h.parallel_for(sycl::range<1>(x.size()), [=](sycl::id<1> i) {
			const float xv = static_cast<float>(x_acc[i]);
			const float gv = static_cast<float>(g_acc[i]);
			const float s = 1.0f / (1.0f + sycl::exp(-xv));
			o_acc[i] = static_cast<Scalar>((xv * s) * gv);
		});
	});
	sycl_ops::default_queue().wait();

	return out;
}

struct SwiGLUGrad {
	Vector dx;
	Vector dgate;
};

inline SwiGLUGrad dswiglu(
	const Vector& x,
	const Vector& gate,
	const Vector& dout) {
	if (x.size() != gate.size() || x.size() != dout.size()) {
		throw std::runtime_error("dSwiGLU shape mismatch");
	}

	SwiGLUGrad grad{Vector(x.size(), static_cast<Scalar>(0.0f)), Vector(x.size(), static_cast<Scalar>(0.0f))};

	if (!sycl_ops::use_sycl_kernels()) {
	for (size_t i = 0; i < x.size(); ++i) {
		const float xv = static_cast<float>(x[i]);
		const float gv = static_cast<float>(gate[i]);
		const float dv = static_cast<float>(dout[i]);
		const float s = 1.0f / (1.0f + std::exp(-xv));
		const float swish = xv * s;
		const float dswish_dx = s + xv * s * (1.0f - s);
		grad.dx[i] = static_cast<Scalar>(dv * gv * dswish_dx);
		grad.dgate[i] = static_cast<Scalar>(dv * swish);
	}
	return grad;
}

	sycl::buffer<Scalar> x_buf(x.data(), sycl::range<1>(x.size()));
	sycl::buffer<Scalar> g_buf(gate.data(), sycl::range<1>(gate.size()));
	sycl::buffer<Scalar> d_buf(dout.data(), sycl::range<1>(dout.size()));
	sycl::buffer<Scalar> dx_buf(grad.dx.data(), sycl::range<1>(grad.dx.size()));
	sycl::buffer<Scalar> dg_buf(grad.dgate.data(), sycl::range<1>(grad.dgate.size()));

	sycl_ops::default_queue().submit([&](sycl::handler& h) {
		auto x_acc = x_buf.get_access<sycl::access::mode::read>(h);
		auto g_acc = g_buf.get_access<sycl::access::mode::read>(h);
		auto d_acc = d_buf.get_access<sycl::access::mode::read>(h);
		auto dx_acc = dx_buf.get_access<sycl::access::mode::discard_write>(h);
		auto dg_acc = dg_buf.get_access<sycl::access::mode::discard_write>(h);
		h.parallel_for(sycl::range<1>(x.size()), [=](sycl::id<1> i) {
			const float xv = static_cast<float>(x_acc[i]);
			const float gv = static_cast<float>(g_acc[i]);
			const float dv = static_cast<float>(d_acc[i]);
			const float s = 1.0f / (1.0f + sycl::exp(-xv));
			const float swish = xv * s;
			const float dswish_dx = s + xv * s * (1.0f - s);
			dx_acc[i] = static_cast<Scalar>(dv * gv * dswish_dx);
			dg_acc[i] = static_cast<Scalar>(dv * swish);
		});
	});
	sycl_ops::default_queue().wait();

	return grad;
}

// Learned output-logit scaling. Using a saturating tanh on logits makes
// high-frequency characters like space an easy collapse target, so keep the
// trainable gain but leave logits unsquashed for CE/argmax.
inline Vector param_tanh(const Vector& x, Scalar theta, Scalar epsilon = static_cast<Scalar>(1e-6f)) {
	const float a = static_cast<float>(softplus(theta)) + static_cast<float>(epsilon);
	Vector out(x.size(), static_cast<Scalar>(0.0f));

	if (!sycl_ops::use_sycl_kernels()) {
	for (size_t i = 0; i < x.size(); ++i) {
		const float xv = static_cast<float>(x[i]);
		out[i] = static_cast<Scalar>(a * xv);
	}
	return out;
}

	sycl::buffer<Scalar> x_buf(x.data(), sycl::range<1>(x.size()));
	sycl::buffer<Scalar> o_buf(out.data(), sycl::range<1>(out.size()));
	sycl_ops::default_queue().submit([&](sycl::handler& h) {
		auto x_acc = x_buf.get_access<sycl::access::mode::read>(h);
		auto o_acc = o_buf.get_access<sycl::access::mode::discard_write>(h);
		h.parallel_for(sycl::range<1>(x.size()), [=](sycl::id<1> i) {
			const float xv = static_cast<float>(x_acc[i]);
			o_acc[i] = static_cast<Scalar>(a * xv);
		});
	});
	sycl_ops::default_queue().wait();

	return out;
}

struct ParamTanhGrad {
	Vector dx;
	Scalar dtheta;
};

inline ParamTanhGrad dparam_tanh(
	const Vector& x,
	Scalar theta,
	const Vector& dout,
	Scalar epsilon = static_cast<Scalar>(1e-6f)) {
	if (x.size() != dout.size()) {
		throw std::runtime_error("dParamTanh shape mismatch");
	}

	const float sp = static_cast<float>(softplus(theta));
	const float a = sp + static_cast<float>(epsilon);
	const float da_dtheta = static_cast<float>(sigmoid(theta));

	Vector dx(x.size(), static_cast<Scalar>(0.0f));
	float dtheta = 0.0f;

	if (!sycl_ops::use_sycl_kernels()) {
	for (size_t i = 0; i < x.size(); ++i) {
		const float xv = static_cast<float>(x[i]);
		const float dv = static_cast<float>(dout[i]);
		dx[i] = static_cast<Scalar>(dv * a);
		dtheta += dv * xv * da_dtheta;
	}
	return ParamTanhGrad{std::move(dx), static_cast<Scalar>(dtheta)};
}

	auto& q = sycl_ops::default_queue();

	sycl::buffer<Scalar> x_buf(x.data(), sycl::range<1>(x.size()));
	sycl::buffer<Scalar> d_buf(dout.data(), sycl::range<1>(dout.size()));
	sycl::buffer<Scalar> dx_buf(dx.data(), sycl::range<1>(dx.size()));
	sycl::buffer<float> dt_buf(&dtheta, sycl::range<1>(1));

	q.submit([&](sycl::handler& h) {
		auto x_acc = x_buf.get_access<sycl::access::mode::read>(h);
		auto d_acc = d_buf.get_access<sycl::access::mode::read>(h);
		auto dx_acc = dx_buf.get_access<sycl::access::mode::discard_write>(h);
		auto dtheta_sum = sycl::reduction(dt_buf, h, sycl::plus<float>());
		h.parallel_for(sycl::range<1>(x.size()), dtheta_sum, [=](sycl::id<1> i, auto& sum) {
			const float xv = static_cast<float>(x_acc[i]);
			const float dv = static_cast<float>(d_acc[i]);
			dx_acc[i] = static_cast<Scalar>(dv * a);
			sum += dv * xv * da_dtheta;
		});
	});
	q.wait();

	return ParamTanhGrad{std::move(dx), static_cast<Scalar>(dtheta)};
}

} // namespace llm::arch::activation