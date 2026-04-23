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
    model_cfg.model_dim = 4;
    model_cfg.qk_dim = 2;
    model_cfg.v_dim = 2;
    model_cfg.rel_dim = 1;
    model_cfg.max_steps = 1;

    LoopingRetNet model(model_cfg, 7);

    TrainConfig train_cfg;
    train_cfg.epochs = 1;
    train_cfg.learning_rate = 1e-1f;
    train_cfg.weight_decay = 0.0f;
    train_cfg.fd_eps = 1e-2f;
    train_cfg.grad_coordinate_samples = 1;
    train_cfg.memory_query_penalty = 0.01f;
    train_cfg.memory_miss_penalty = 0.01f;
    train_cfg.memory_alignment_weight = 0.02f;
    train_cfg.loop_supervision_weight = 0.05f;
    train_cfg.enable_query = true;
    train_cfg.force_output = false;
    train_cfg.use_parallel_retention = false;
    train_cfg.forced_loop_min = 1;
    train_cfg.forced_loop_max = 1;
    train_cfg.seed = 7;
    train_cfg.memory_cfg.semvec_dim = model_cfg.v_dim;
    train_cfg.memory_cfg.max_hop_depth = 1;
    train_cfg.memory_cfg.max_hop_breadth = 1;
    train_cfg.memory_cfg.max_write_entries = 1;

    const auto dataset = make_shift_dataset(std::vector<std::string>{"ab"});
    if (dataset.empty()) {
        throw std::runtime_error("looping_retnet_training_leak_smoke_test: empty dataset");
    }

    LoopingRetNetSGDTrainer trainer(train_cfg);
    const auto history = trainer.train(model, dataset);

    if (history.empty()) {
        throw std::runtime_error("looping_retnet_training_leak_smoke_test: empty history");
    }

    std::cout << "final_loss=" << history.back().avg_loss << "\n";
    return 0;
}
