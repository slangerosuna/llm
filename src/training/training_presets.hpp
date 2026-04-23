#pragma once

#include <architecture/looping_retnet.hpp>
#include <training/looping_retnet_training.hpp>

namespace llm::training::looping::presets {

inline arch::LoopConfig medium_loop_config() {
    arch::LoopConfig cfg;
    cfg.model_dim = 32;
    cfg.qk_dim = 16;
    cfg.v_dim = 16;
    cfg.rel_dim = 8;
    cfg.hidden_layers = 2;
    cfg.max_steps = 4;
    return cfg;
}

inline TrainConfig medium_fd_config() {
    TrainConfig cfg;
    cfg.mode = TrainMode::FiniteDifference;
    cfg.epochs = 48;
    cfg.warmup_epochs = 6;
    cfg.learning_rate = 1.0f;
    cfg.min_learning_rate_ratio = 0.1f;
    cfg.fd_eps = 1e-2f;
    cfg.grad_coordinate_samples = 64;
    cfg.min_grad_coordinate_samples = 24;
    cfg.batch_size = 2;
    cfg.sgd_momentum = 0.9f;
    cfg.enable_instability_backoff = true;
    cfg.instability_skip_ratio_threshold = 0.20f;
    cfg.instability_repair_ratio_threshold = 0.005f;
    cfg.instability_lr_backoff = 0.5f;
    cfg.instability_cooldown_epochs = 3;
    cfg.enable_query = false;
    cfg.force_output = true;
    cfg.forced_loop_min = 1;
    cfg.forced_loop_max = 1;
    cfg.use_parallel_retention = false;
    cfg.seed = 13;
    return cfg;
}

inline TrainConfig medium_backprop_heads_config() {
    TrainConfig cfg = medium_fd_config();
    cfg.mode = TrainMode::BackpropHeads;
    cfg.epochs = 48;
    cfg.learning_rate = 5e-1f;
    cfg.batch_size = 4;
    cfg.backprop_force_single_step = true;
    cfg.backprop_include_loop_supervision = true;
    cfg.backprop_fd_check_samples = 4;
    cfg.backprop_fd_check_eps = 1e-3f;
    return cfg;
}

} // namespace llm::training::looping::presets
