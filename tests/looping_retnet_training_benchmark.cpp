#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <architecture/looping_retnet.hpp>
#include <training/looping_retnet_training.hpp>
#include <training/training_presets.hpp>

namespace {

struct BenchCase {
    std::string name;
    llm::training::looping::TrainConfig cfg;
};

struct BenchResult {
    std::string name;
    float first_loss = 0.0f;
    float final_loss = 0.0f;
    float loss_drop = 0.0f;
    double total_ms = 0.0;
    double ms_per_epoch = 0.0;
    double loss_drop_per_min = 0.0;
};

BenchResult run_case(const BenchCase& c) {
    using namespace llm::training::looping;

    auto model_cfg = presets::medium_loop_config();
    llm::arch::LoopingRetNet model(model_cfg, c.cfg.seed);

    auto cfg = c.cfg;
    cfg.memory_cfg.semvec_dim = model_cfg.v_dim;
    cfg.memory_cfg.max_hop_depth = 2;
    cfg.memory_cfg.max_hop_breadth = 4;
    cfg.memory_cfg.max_write_entries = 2;

    const auto dataset = make_shift_dataset(std::vector<std::string>{
        "hello world",
        "looping retnet",
        "memory graph",
        "benchmark sequence",
    });

    LoopingRetNetSGDTrainer trainer(cfg);

    const auto t0 = std::chrono::steady_clock::now();
    const auto history = trainer.train(model, dataset);
    const auto t1 = std::chrono::steady_clock::now();

    BenchResult r;
    r.name = c.name;
    r.total_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t1 - t0).count();
    r.ms_per_epoch = history.empty() ? 0.0 : r.total_ms / static_cast<double>(history.size());
    if (!history.empty()) {
        r.first_loss = history.front().avg_loss;
        r.final_loss = history.back().avg_loss;
        r.loss_drop = r.first_loss - r.final_loss;
    }
    const double total_min = std::max(1e-9, r.total_ms / 60000.0);
    r.loss_drop_per_min = static_cast<double>(r.loss_drop) / total_min;
    return r;
}

} // namespace

int main() {
    using namespace llm::training::looping;

    auto fd = presets::medium_fd_config();
    fd.epochs = 24;

    auto fd_fast = fd;
    fd_fast.grad_coordinate_samples = 32;
    fd_fast.min_grad_coordinate_samples = 16;

    auto bp = presets::medium_backprop_heads_config();
    bp.epochs = 24;

    const std::vector<BenchCase> cases{
        {"fd_baseline", fd},
        {"fd_fast", fd_fast},
        {"bp_heads", bp},
    };

    std::vector<BenchResult> results;
    results.reserve(cases.size());
    for (const auto& c : cases) {
        results.push_back(run_case(c));
    }

    std::cout << "| case | first_loss | final_loss | loss_drop | total_ms | ms_per_epoch | loss_drop_per_min |\n";
    std::cout << "|---|---:|---:|---:|---:|---:|---:|\n";
    std::cout << std::fixed << std::setprecision(6);
    for (const auto& r : results) {
        std::cout << "| " << r.name
                  << " | " << r.first_loss
                  << " | " << r.final_loss
                  << " | " << r.loss_drop
                  << " | " << r.total_ms
                  << " | " << r.ms_per_epoch
                  << " | " << r.loss_drop_per_min
                  << " |\n";
    }

    return 0;
}
