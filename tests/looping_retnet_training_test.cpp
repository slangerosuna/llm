#include <iostream>
#include <stdexcept>
#include <vector>

#include <architecture/looping_retnet.hpp>
#include <training/looping_retnet_training.hpp>

int main() {
    using llm::arch::LoopConfig;
    using llm::arch::LoopingRetNet;
    using llm::training::looping::LoopingRetNetSGDTrainer;
    using llm::training::looping::TrainConfig;
    using llm::training::looping::make_shift_dataset;

    LoopConfig model_cfg;
    model_cfg.model_dim = 16;
    model_cfg.qk_dim = 32;
    model_cfg.v_dim = 32;
    model_cfg.rel_dim = 16;
    model_cfg.hidden_layers = 4;
    model_cfg.max_steps = 16;

    LoopingRetNet model(model_cfg, 42);

    TrainConfig train_cfg;
    train_cfg.mode = llm::training::looping::TrainMode::BackpropFull;
    train_cfg.warmup_epochs = 4;
    train_cfg.epochs = 256;
    train_cfg.learning_rate = 2.0e-2f;
    train_cfg.min_learning_rate_ratio = 0.1f;
    train_cfg.weight_decay = 0.0f;
    train_cfg.sgd_momentum = 0.9f;
    train_cfg.batch_size = 8;
    train_cfg.enable_instability_backoff = true;
    train_cfg.instability_skip_ratio_threshold = 0.20f;
    train_cfg.instability_repair_ratio_threshold = 0.005f;
    train_cfg.instability_lr_backoff = 0.5f;
    train_cfg.instability_cooldown_epochs = 3;
    train_cfg.backprop_force_single_step = true;
    train_cfg.backprop_include_loop_supervision = true;
    train_cfg.backprop_fd_check_samples = 4;
    train_cfg.backprop_fd_check_eps = 1e-3f;
    train_cfg.memory_query_penalty = 0.01f;
    train_cfg.memory_miss_penalty = 0.01f;
    train_cfg.memory_alignment_weight = 0.05f;
    train_cfg.loop_supervision_weight = 0.1f;
    train_cfg.enable_query = false;
    train_cfg.force_output = true;
    train_cfg.use_parallel_retention = false;
    train_cfg.forced_loop_min = 1;
    train_cfg.forced_loop_max = 1;
    train_cfg.seed = 13;
    train_cfg.memory_cfg.semvec_dim = model_cfg.v_dim;
    train_cfg.memory_cfg.max_hop_depth = 2;
    train_cfg.memory_cfg.max_hop_breadth = 4;
    train_cfg.memory_cfg.max_write_entries = 2;

    const auto dataset = make_shift_dataset(std::vector<std::string>{
        "hello world",
        "looping retnet",
        "memory graph"
    });
    if (dataset.empty()) {
        throw std::runtime_error("looping_retnet_training_test: empty dataset");
    }

    const uint64_t before = model.parameter_checksum();

    LoopingRetNetSGDTrainer trainer(train_cfg);
    const auto history = trainer.train(model, dataset);

    if (history.empty()) {
        throw std::runtime_error("looping_retnet_training_test: empty training history");
    }

    const uint64_t after = model.parameter_checksum();
    std::cout << "before=" << before << " after=" << after
              << " final_loss=" << history.back().avg_loss << "\n";

    if (before == after) {
        throw std::runtime_error("looping_retnet_training_test: parameters did not update");
    }

    return 0;
}
