#pragma once

#include <algorithm>
#include <future>
#include <stdexcept>
#include <utility>
#include <vector>

namespace llm::arch {

using Vector = std::vector<float>;
using Matrix = std::vector<Vector>;

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