#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include <architecture/sequential.hpp>

namespace llm::arch {

using KVState = std::vector<std::vector<Scalar>>; // qk_dim x v_dim

inline Vector group_norm(const Vector& x, float eps = 1e-5f) {
	if (x.empty()) {
		return x;
	}
	float mean = 0.0f;
	for (Scalar v : x) {
		mean += static_cast<float>(v);
	}
	mean /= static_cast<float>(x.size());

	float var = 0.0f;
	for (Scalar v : x) {
		const float d = static_cast<float>(v) - mean;
		var += d * d;
	}
	var /= static_cast<float>(x.size());

	const float inv = 1.0f / std::sqrt(var + eps);
	Vector y(x.size(), static_cast<Scalar>(0.0f));
	for (size_t i = 0; i < x.size(); ++i) {
		y[i] = static_cast<Scalar>((static_cast<float>(x[i]) - mean) * inv);
	}
	return y;
}

class Retention {
	static float dot(const Vector& a, const Vector& b) {
		return static_cast<float>(sycl_ops::dot(a, b));
	}

public:
	// q,k,v are len x dim. decay_mask is len x len.
	static Matrix parallel(
		const Matrix& q,
		const Matrix& k,
		const Matrix& v,
		const Matrix& decay_mask) {
		if (q.size() != k.size() || q.size() != v.size() || q.size() != decay_mask.size()) {
			throw std::runtime_error("ParallelRetention shape mismatch");
		}

		const size_t len = q.size();
		const size_t vdim = v.empty() ? 0 : v.front().size();
		Matrix out(len, Vector(vdim, static_cast<Scalar>(0.0f)));

		for (size_t i = 0; i < len; ++i) {
			for (size_t j = 0; j < len; ++j) {
				const float r = dot(q[i], k[j]) * decay_mask[i][j];
				for (size_t d = 0; d < vdim; ++d) {
					const float acc = static_cast<float>(out[i][d]) + r * static_cast<float>(v[j][d]);
					out[i][d] = static_cast<Scalar>(acc);
				}
			}
			out[i] = group_norm(out[i]);
		}

		return out;
	}

	// q,k,v are 1 x dim for current step.
	static std::pair<Vector, KVState> recurrent(
		const Vector& q,
		const Vector& k,
		const Vector& v,
		const KVState& past_kv,
		float decay) {
		if (k.size() != q.size()) {
			throw std::runtime_error("RecurrentRetention q/k mismatch");
		}

		KVState cur = past_kv;
		if (cur.empty()) {
			cur.assign(k.size(), Vector(v.size(), static_cast<Scalar>(0.0f)));
		}

		for (size_t i = 0; i < k.size(); ++i) {
			for (size_t j = 0; j < v.size(); ++j) {
				const float updated = decay * static_cast<float>(cur[i][j])
					+ static_cast<float>(k[i]) * static_cast<float>(v[j]);
				cur[i][j] = static_cast<Scalar>(updated);
			}
		}

		Vector out(v.size(), static_cast<Scalar>(0.0f));
		for (size_t i = 0; i < q.size(); ++i) {
			for (size_t j = 0; j < v.size(); ++j) {
				const float acc = static_cast<float>(out[j])
					+ static_cast<float>(q[i]) * static_cast<float>(cur[i][j]);
				out[j] = static_cast<Scalar>(acc);
			}
		}

		out = group_norm(out);
		return {out, cur};
	}

	static std::pair<Matrix, KVState> chunkwise(
		const Matrix& q,
		const Matrix& k,
		const Matrix& v,
		const KVState& past_kv,
		const Matrix& decay_mask,
		float chunk_decay,
		const Vector& inner_decay) {
		Matrix inner = parallel(q, k, v, decay_mask);
		Matrix out = inner;

		for (size_t i = 0; i < q.size(); ++i) {
			Vector cross(v.front().size(), static_cast<Scalar>(0.0f));
			for (size_t a = 0; a < q[i].size() && a < past_kv.size(); ++a) {
				for (size_t b = 0; b < cross.size() && b < past_kv[a].size(); ++b) {
					const float acc = static_cast<float>(cross[b])
						+ static_cast<float>(q[i][a]) * static_cast<float>(past_kv[a][b]);
					cross[b] = static_cast<Scalar>(acc);
				}
			}
			const float id = (i < inner_decay.size()) ? inner_decay[i] : 1.0f;
			for (size_t d = 0; d < cross.size(); ++d) {
				const float acc = static_cast<float>(out[i][d]) + static_cast<float>(cross[d]) * id;
				out[i][d] = static_cast<Scalar>(acc);
			}
			out[i] = group_norm(out[i]);
		}

		KVState cur = past_kv;
		if (cur.empty()) {
			cur.assign(k.front().size(), Vector(v.front().size(), static_cast<Scalar>(0.0f)));
		}
		for (size_t i = 0; i < cur.size(); ++i) {
			for (size_t j = 0; j < cur[i].size(); ++j) {
				float kv_sum = 0.0f;
				for (size_t t = 0; t < k.size(); ++t) {
					kv_sum += static_cast<float>(k[t][i]) * static_cast<float>(v[t][j]);
				}
				const float updated = chunk_decay * static_cast<float>(cur[i][j]) + kv_sum;
				cur[i][j] = static_cast<Scalar>(updated);
			}
		}

		return {out, cur};
	}
};

} // namespace llm::arch