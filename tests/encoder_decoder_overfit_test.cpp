#include <iostream>
#include <stdexcept>

#include <training/encoder_decoder_demo.hpp>

static void run_looping_retnet_smoke();

int main() {
    const auto result = llm::training::demo::run_minimal_encoder_decoder_sgd_overfit(200, 0.05f, 42);

    std::cout << "initial=" << result.initial_loss << " final=" << result.final_loss << "\n";

    if (!(result.final_loss < result.initial_loss)) {
        throw std::runtime_error("Training did not reduce loss");
    }

    if (result.final_loss > 0.02f) {
        throw std::runtime_error("Model did not overfit tiny dataset strongly enough");
    }

    run_looping_retnet_smoke();
    return 0;
}

// ── LoopingRetNet smoke test ──────────────────────────────────────────────────
#include <architecture/looping_retnet.hpp>
#include <long_term/memory_module.hpp>

static void run_looping_retnet_smoke() {
    using namespace llm::arch;
    using namespace llm::memory;

    LoopConfig cfg;
    cfg.char_vocab = 256;
    cfg.model_dim  = 32; // tiny for speed
    cfg.qk_dim     = 16;
    cfg.v_dim      = 16;
    cfg.rel_dim    = 8;
    cfg.max_steps  = 4;

    LoopingRetNet model(cfg, 99);

    MemoryConfig mcfg;
    mcfg.semvec_dim          = cfg.v_dim; // must match
    mcfg.max_hop_depth       = 2;
    mcfg.max_hop_breadth     = 4;
    mcfg.max_write_entries   = 2;
    mcfg.merge_threshold_sq  = 0.1f;
    mcfg.fresh_prune_penalty = 1.0f;

    Graph      graph;
    SpatialMap spatial_map;

    // Compressor: key/value dim is qk_dim / v_dim; graph stores semvec_dim vecs.
    // Use v_dim as the input dimension for the compressor.
    NodeCompressor compressor(cfg.v_dim, mcfg.semvec_dim, /*seed=*/7);

    GraphMemoryBridge bridge(graph, spatial_map, std::move(compressor), mcfg);
    MultiHopQuery     query_engine(graph, spatial_map, mcfg);

    KVState        state;
    AttentionMemory kv_cache;

    const std::string prompt = "hello";
    for (char c : prompt) {
        auto out = model.step(c, state, kv_cache, bridge, query_engine);
        (void)out;
    }

    // If we reach here without throwing, the smoke test passes.
}

// Append call to main via a static initialiser trick avoids editing the
// original main; instead we call it from a separate test entry point that
// the CTest adds alongside the overfit test.
