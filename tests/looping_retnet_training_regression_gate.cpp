#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <architecture/looping_retnet.hpp>
#include <training/looping_retnet_training.hpp>
#include <training/training_presets.hpp>

namespace {

struct RunStat {
    std::vector<llm::training::looping::EpochResult> history;
    double total_ms = 0.0;
};

RunStat run_training(
    llm::arch::LoopingRetNet& model,
    const llm::training::looping::TrainConfig& cfg,
    const std::vector<llm::training::looping::SequenceExample>& dataset) {
    llm::training::looping::LoopingRetNetSGDTrainer trainer(cfg);
    const auto t0 = std::chrono::steady_clock::now();
    auto history = trainer.train(model, dataset);
    const auto t1 = std::chrono::steady_clock::now();

    RunStat s;
    s.history = std::move(history);
    s.total_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t1 - t0).count();
    return s;
}

void assert_history_finite(const std::vector<llm::training::looping::EpochResult>& h) {
    if (h.empty()) {
        throw std::runtime_error("regression_gate: empty history");
    }
    for (const auto& e : h) {
        if (!std::isfinite(e.avg_loss)) {
            throw std::runtime_error("regression_gate: non-finite loss in history");
        }
    }
}

} // namespace

int main() {
    using namespace llm::training::looping;

    const auto model_cfg = presets::medium_loop_config();
    const auto dataset = make_shift_dataset(std::vector<std::string>{
        "hello world",
        "looping retnet",
        "memory graph",
        "regression gate",
    });
    if (dataset.empty()) {
        throw std::runtime_error("regression_gate: empty dataset");
    }

    // Gate 1: convergence and finite losses in BP mode.
    auto bp_cfg = presets::medium_backprop_heads_config();
    bp_cfg.epochs = 16;
    bp_cfg.learning_rate = 5e-1f;
    bp_cfg.min_learning_rate = bp_cfg.learning_rate;
    bp_cfg.min_learning_rate_ratio = 1.0f;
    bp_cfg.warmup_epochs = 1;
    bp_cfg.sgd_momentum = 0.0f;

    llm::arch::LoopingRetNet bp_model(model_cfg, bp_cfg.seed);
    const auto bp_stat = run_training(bp_model, bp_cfg, dataset);
    assert_history_finite(bp_stat.history);

    const float bp_first = bp_stat.history.front().avg_loss;
    const float bp_last = bp_stat.history.back().avg_loss;
    if (bp_last > bp_first + 1e-3f) {
        throw std::runtime_error("regression_gate: BP loss did not improve");
    }

    // Gate 2: checkpoint/resume sanity with deterministic replay.
    llm::arch::LoopingRetNet full_model(model_cfg, bp_cfg.seed);
    auto full_cfg = bp_cfg;
    full_cfg.epochs = 12;
    const auto full_stat = run_training(full_model, full_cfg, dataset);
    assert_history_finite(full_stat.history);

    llm::arch::LoopingRetNet split_model(model_cfg, bp_cfg.seed);
    auto part_cfg = bp_cfg;
    part_cfg.epochs = 6;
    const auto part1 = run_training(split_model, part_cfg, dataset);
    assert_history_finite(part1.history);

    const std::string ckpt_path = "./build/retnet_regression_ckpt.bin";
    split_model.save_to_file(ckpt_path);
    split_model = llm::arch::LoopingRetNet::load_from_file(ckpt_path);

    const auto part2 = run_training(split_model, part_cfg, dataset);
    assert_history_finite(part2.history);

    const uint64_t full_checksum = full_model.parameter_checksum();
    const uint64_t split_checksum = split_model.parameter_checksum();
    if (full_checksum != split_checksum) {
        throw std::runtime_error("regression_gate: checkpoint/replay checksum mismatch");
    }

    // Gate 3: performance guard (BP should not regress >20% vs FD on same setup).
    auto fd_cfg = presets::medium_fd_config();
    fd_cfg.epochs = 8;
    fd_cfg.learning_rate = 1.0f;
    fd_cfg.grad_coordinate_samples = 32;
    fd_cfg.min_grad_coordinate_samples = 16;

    llm::arch::LoopingRetNet fd_model(model_cfg, fd_cfg.seed);
    const auto fd_stat = run_training(fd_model, fd_cfg, dataset);
    assert_history_finite(fd_stat.history);

    auto bp_perf_cfg = bp_cfg;
    bp_perf_cfg.epochs = 8;
    llm::arch::LoopingRetNet bp_perf_model(model_cfg, bp_perf_cfg.seed);
    const auto bp_perf_stat = run_training(bp_perf_model, bp_perf_cfg, dataset);
    assert_history_finite(bp_perf_stat.history);

    const double fd_ms_per_epoch = fd_stat.total_ms / static_cast<double>(fd_stat.history.size());
    const double bp_ms_per_epoch = bp_perf_stat.total_ms / static_cast<double>(bp_perf_stat.history.size());
    if (bp_ms_per_epoch > fd_ms_per_epoch * 1.2) {
        throw std::runtime_error("regression_gate: BP epoch time regressed by >20% vs FD baseline");
    }

    std::cout << "regression_gate_pass"
              << " bp_first=" << bp_first
              << " bp_last=" << bp_last
              << " fd_ms_per_epoch=" << fd_ms_per_epoch
              << " bp_ms_per_epoch=" << bp_ms_per_epoch
              << "\n";
    return 0;
}
