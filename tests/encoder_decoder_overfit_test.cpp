#include <iostream>
#include <stdexcept>

#include <training/encoder_decoder_demo.hpp>

int main() {
    const auto result = llm::training::demo::run_minimal_encoder_decoder_sgd_overfit(1200, 0.05f, 42);

    std::cout << "initial=" << result.initial_loss << " final=" << result.final_loss << "\n";

    if (!(result.final_loss < result.initial_loss)) {
        throw std::runtime_error("Training did not reduce loss");
    }

    if (result.final_loss > 0.02f) {
        throw std::runtime_error("Model did not overfit tiny dataset strongly enough");
    }

    return 0;
}
