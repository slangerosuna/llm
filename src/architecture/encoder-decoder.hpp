#pragma once

#include <functional>
#include <utility>

#include <architecture/sequential.hpp>

namespace llm::arch {

class EncoderDecoderModel : public SequentialModel {
	std::function<Matrix(const Matrix&)> encoder_;
	std::function<Matrix(const Matrix&, const Matrix&)> decoder_;

public:
	EncoderDecoderModel() = default;

	EncoderDecoderModel(
		std::function<Matrix(const Matrix&)> encoder,
		std::function<Matrix(const Matrix&, const Matrix&)> decoder)
	  : encoder_(std::move(encoder)), decoder_(std::move(decoder)) {}

	void set_encoder(std::function<Matrix(const Matrix&)> encoder) {
		encoder_ = std::move(encoder);
	}

	void set_decoder(std::function<Matrix(const Matrix&, const Matrix&)> decoder) {
		decoder_ = std::move(decoder);
	}

	Matrix encode(const Matrix& src) const {
		if (encoder_) {
			return encoder_(src);
		}
		return SequentialModel::forward(src);
	}

	Matrix decode(const Matrix& tgt, const Matrix& memory) const {
		if (decoder_) {
			return decoder_(tgt, memory);
		}
		(void)memory;
		return SequentialModel::forward(tgt);
	}

	Matrix forward(const Matrix& src, const Matrix& tgt) const {
		const Matrix memory = encode(src);
		return decode(tgt, memory);
	}
};

} // namespace llm::arch