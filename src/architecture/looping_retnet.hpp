#pragma once

// looping_retnet.hpp
//
// LoopingRetNet: token-based language model that combines:
//   • A RetNet recurrent core (recurrent mode, one step per token).
//   • An Attention head used to read and write to and from long-term graph
//     memory via AttentionMemory (not for in-sequence context).
//   • A second attention head with a separate cache for "chrono" memory,
//     which is purely recency-based and not written to by the model.
//   • A 4-class action head that chooses on every inner iteration:
//       0 – OUTPUT:        emit a token and advance to the next input token.
//       1 – QUERY_MEMORY:  run a multihop graph query, inject results into the
//                          retentive KRV cache, and loop again (no output yet).
//       2 – LOOP:          run the RetNet core again without querying or outputting.
//       3 – CREATE_MEMORY: generate a new memory tuple from a dedicated LM head,
//                          inject into KRV + graph, then loop again.
//
// The model loops until it selects OUTPUT or exhausts max_steps, at which point
// it is forced to output.  This lets it perform up to max_steps-1 "thinking"
// steps, including bounded multihop memory lookups.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <architecture/activation.hpp>
#include <architecture/sequential.hpp>
#include <architecture/attention.hpp>
#include <architecture/retentive.hpp>
#include <long_term/memory_module.hpp>

namespace llm::arch {

// ── Model action enum ─────────────────────────────────────────────────────────

enum class ModelAction : uint8_t {
    OUTPUT = 0,
    QUERY_MEMORY = 1,
    LOOP = 2,
    CREATE_MEMORY = 3,
};

struct ActionControl {
    bool enabled = false;
    float output_bias = 0.0f;
    float query_bias = 0.0f;
    float loop_bias = 0.0f;
    float create_bias = 0.0f;
    bool force_create_memory_first = false;
};

// ── Configuration ─────────────────────────────────────────────────────────────

// this config is for a 1M model
struct LoopConfig {
    size_t char_vocab  = 4096;  // token vocabulary size
    size_t model_dim   = 512;  // embedding / hidden dimension
    size_t qk_dim      = 256;   // query/key dimension for RetNet and Attention
    size_t v_dim       = 256;   // value dimension (== semvec_dim for graph compat)
    size_t rel_dim     = 128;   // relation vector dimension (smaller edge label)
    size_t hidden_layers = 8;  // extra v_dim -> v_dim hidden layers before heads
    float  decay       = 0.9f; // RetNet recurrent decay
    size_t max_steps   = 16;    // max inner iterations before forced output
};

// ── Simple learned linear projection ─────────────────────────────────────────

struct LinearProj {
    std::vector<Scalar> w; // out_dim × in_dim row-major
    std::vector<Scalar> b; // out_dim
    size_t in_dim  = 0;
    size_t out_dim = 0;

    LinearProj() = default;

    LinearProj(size_t in, size_t out, float scale, std::mt19937& rng)
        : w(out * in), b(out, static_cast<Scalar>(0.0f)), in_dim(in), out_dim(out)
    {
        std::normal_distribution<float> dist(0.0f, scale);
        for (Scalar& s : w) { s = static_cast<Scalar>(dist(rng)); }
    }

    Vector forward(const Vector& x) const {
        return sycl_ops::linear(w, out_dim, in_dim, x, b);
    }
};

// ── LoopingRetNet ─────────────────────────────────────────────────────────────

class LoopingRetNet {
    LoopConfig cfg_;

    // Token embedding table: char_vocab × model_dim
    std::vector<Vector> embed_table_;

    // Projection into RetNet Q/K/V/R (relation).
    LinearProj proj_q_; // model_dim → qk_dim
    LinearProj gate_q_;
    LinearProj proj_k_; // model_dim → qk_dim
    LinearProj gate_k_;
    LinearProj proj_v_; // model_dim → v_dim
    LinearProj gate_v_;
    LinearProj proj_r_; // model_dim → rel_dim  (smaller relation vector)
    LinearProj gate_r_;

    // Projects fused state back to model_dim for the next loop iteration.
    LinearProj loop_proj_; // v_dim → model_dim

    // Optional hidden stack over fused state before heads.
    std::vector<LinearProj> hidden_layers_;

    // 4-class action head.
    LinearProj action_head_; // v_dim → 4

    // Query-load gate head.
    LinearProj load_head_; // v_dim → 1

    // Dual-attention mixing heads and memory generation/ordering heads.
    LinearProj chrono_mix_head_;   // v_dim -> 1
    LinearProj krv_mix_head_;      // v_dim -> 1
    LinearProj memory_entry_head_; // v_dim -> (qk_dim + rel_dim + v_dim)
    LinearProj krv_order_head_;    // v_dim -> 1

    // Token output head.
    LinearProj output_head_; // v_dim → char_vocab
    Scalar output_theta_ = static_cast<Scalar>(1.0f);

    Attention chrono_attention_;
    Attention krv_attention_;

    static constexpr uint32_t kModelMagic = 0x4C524E54; // "LRNT"
    static constexpr uint32_t kModelVersion = 4;

    // ── Helpers ───────────────────────────────────────────────────────────────

    const Vector& embed(size_t token_id) const {
        if (token_id >= embed_table_.size()) {
            throw std::runtime_error("LoopingRetNet: input token id out of range");
        }
        return embed_table_[token_id];
    }

    ModelAction pick_action(const Vector& state) const {
        const Vector logits = action_head_.forward(state);
        if (logits.size() < 4) { return ModelAction::OUTPUT; }

        // Stable softmax → argmax.
        float m = static_cast<float>(logits[0]);
        for (const Scalar v : logits) { m = std::max(m, static_cast<float>(v)); }
        size_t best = 0;
        float  best_exp = std::exp(static_cast<float>(logits[0]) - m);
        for (size_t i = 1; i < logits.size(); ++i) {
            const float e = std::exp(static_cast<float>(logits[i]) - m);
            if (e > best_exp) { best_exp = e; best = i; }
        }
        return static_cast<ModelAction>(best);
    }

    size_t pick_token(const Vector& state) const {
        const Vector raw = output_head_.forward(state);
        const Vector logits = activation::param_tanh(raw, output_theta_);
        size_t best = 0;
        float  best_val = static_cast<float>(logits[0]);
        for (size_t i = 1; i < logits.size(); ++i) {
            const float v = static_cast<float>(logits[i]);
            if (v > best_val) { best_val = v; best = i; }
        }
        return best;
    }

    Vector output_logits(const Vector& state) const {
        const Vector raw = output_head_.forward(state);
        return activation::param_tanh(raw, output_theta_);
    }

    // Fuse RetNet output and attention output into a single state vector.
    // Both attention vectors are v_dim; output is v_dim.
    static Vector fuse_dual(
        const Vector& ret,
        const Vector& chrono_attn,
        const Vector& krv_attn,
        float chrono_mix,
        float krv_mix)
    {
        const size_t dim = std::min(ret.size(), std::min(chrono_attn.size(), krv_attn.size()));
        Vector out(dim, static_cast<Scalar>(0.0f));
        for (size_t i = 0; i < dim; ++i) {
            out[i] = static_cast<Scalar>(
                static_cast<float>(ret[i])
                + chrono_mix * static_cast<float>(chrono_attn[i])
                + krv_mix * static_cast<float>(krv_attn[i]));
        }
        return out;
    }

    static Vector swiglu_self(const Vector& x) {
        // Self-gated SwiGLU block for unified non-final activations.
        return activation::swiglu(x, x);
    }

    static float sigmoid(float x) {
        return 1.0f / (1.0f + std::exp(-x));
    }

    static Vector relation_zeros(size_t rel_dim) {
        return Vector(rel_dim, static_cast<Scalar>(0.0f));
    }

    static AttentionMemory evict_to_size(AttentionMemory& mem, size_t max_entries) {
        AttentionMemory evicted;
        if (max_entries == 0 || mem.size() <= max_entries) {
            if (max_entries == 0 && !mem.empty()) {
                evicted = std::move(mem);
                mem = AttentionMemory{};
            }
            return evicted;
        }

        const size_t n = mem.size();
        std::vector<size_t> keep(n);
        for (size_t i = 0; i < n; ++i) {
            keep[i] = i;
        }
        std::partial_sort(
            keep.begin(),
            keep.begin() + static_cast<std::ptrdiff_t>(max_entries),
            keep.end(),
            [&](size_t a, size_t b) {
                const float score_a = (a < mem.prune_scores.size())
                    ? mem.prune_scores[a]
                    : std::numeric_limits<float>::infinity();
                const float score_b = (b < mem.prune_scores.size())
                    ? mem.prune_scores[b]
                    : std::numeric_limits<float>::infinity();
                if (score_a != score_b) {
                    return score_a < score_b;
                }
                return a > b;
            });

        std::vector<uint8_t> keep_mask(n, 0);
        for (size_t i = 0; i < max_entries; ++i) {
            keep_mask[keep[i]] = 1;
        }

        AttentionMemory kept;
        kept.keys.reserve(max_entries);
        kept.relations.reserve(max_entries);
        kept.values.reserve(max_entries);
        kept.prune_scores.reserve(max_entries);

        for (size_t i = 0; i < n; ++i) {
            if (keep_mask[i]) {
                kept.keys.push_back(std::move(mem.keys[i]));
                kept.relations.push_back(std::move(mem.relations[i]));
                kept.values.push_back(std::move(mem.values[i]));
                kept.prune_scores.push_back(mem.prune_scores[i]);
            } else {
                evicted.keys.push_back(std::move(mem.keys[i]));
                evicted.relations.push_back(std::move(mem.relations[i]));
                evicted.values.push_back(std::move(mem.values[i]));
                evicted.prune_scores.push_back(mem.prune_scores[i]);
            }
        }

        mem = std::move(kept);
        return evicted;
    }

    static void append_memory(AttentionMemory& dst, AttentionMemory&& src) {
        for (size_t i = 0; i < src.size(); ++i) {
            dst.push_back(
                std::move(src.keys[i]),
                std::move(src.relations[i]),
                std::move(src.values[i]),
                src.prune_scores[i]);
        }
    }

    void prune_krv_cache_with_graph_write(
        AttentionMemory& krv_cache,
        memory::GraphMemoryBridge& bridge,
        bool enable_memory_write) const
    {
        const AttentionMemory evicted = evict_to_size(krv_cache, kv_cache_limit());
        if (enable_memory_write && !evicted.empty()) {
            bridge.write(evicted);
        }
    }

    static void prune_chrono_cache(AttentionMemory& chrono_cache, size_t max_entries) {
        if (max_entries == 0 || chrono_cache.size() <= max_entries) {
            if (max_entries == 0) {
                chrono_cache = AttentionMemory{};
            }
            return;
        }
        const size_t remove_n = chrono_cache.size() - max_entries;
        chrono_cache.keys.erase(
            chrono_cache.keys.begin(),
            chrono_cache.keys.begin() + static_cast<std::ptrdiff_t>(remove_n));
        chrono_cache.relations.erase(
            chrono_cache.relations.begin(),
            chrono_cache.relations.begin() + static_cast<std::ptrdiff_t>(remove_n));
        chrono_cache.values.erase(
            chrono_cache.values.begin(),
            chrono_cache.values.begin() + static_cast<std::ptrdiff_t>(remove_n));
        chrono_cache.prune_scores.erase(
            chrono_cache.prune_scores.begin(),
            chrono_cache.prune_scores.begin() + static_cast<std::ptrdiff_t>(remove_n));
    }

    size_t kv_cache_limit() const {
        return std::max<size_t>(32, cfg_.max_steps * 8);
    }

    static void write_u32(std::ostream& os, uint32_t v) {
        os.write(reinterpret_cast<const char*>(&v), sizeof(v));
    }

    static void write_u64(std::ostream& os, uint64_t v) {
        os.write(reinterpret_cast<const char*>(&v), sizeof(v));
    }

    static uint32_t read_u32(std::istream& is) {
        uint32_t v = 0;
        is.read(reinterpret_cast<char*>(&v), sizeof(v));
        return v;
    }

    static uint64_t read_u64(std::istream& is) {
        uint64_t v = 0;
        is.read(reinterpret_cast<char*>(&v), sizeof(v));
        return v;
    }

    static void write_scalar_vec(std::ostream& os, const std::vector<Scalar>& vec) {
        write_u64(os, static_cast<uint64_t>(vec.size()));
        for (Scalar s : vec) {
            const float f = static_cast<float>(s);
            os.write(reinterpret_cast<const char*>(&f), sizeof(f));
        }
    }

    static std::vector<Scalar> read_scalar_vec(std::istream& is) {
        const uint64_t n = read_u64(is);
        std::vector<Scalar> vec(static_cast<size_t>(n), static_cast<Scalar>(0.0f));
        for (size_t i = 0; i < vec.size(); ++i) {
            float f = 0.0f;
            is.read(reinterpret_cast<char*>(&f), sizeof(f));
            vec[i] = static_cast<Scalar>(f);
        }
        return vec;
    }

    static void write_linear(std::ostream& os, const LinearProj& l) {
        write_u64(os, static_cast<uint64_t>(l.in_dim));
        write_u64(os, static_cast<uint64_t>(l.out_dim));
        write_scalar_vec(os, l.w);
        write_scalar_vec(os, l.b);
    }

    static LinearProj read_linear(std::istream& is) {
        LinearProj l;
        l.in_dim = static_cast<size_t>(read_u64(is));
        l.out_dim = static_cast<size_t>(read_u64(is));
        l.w = read_scalar_vec(is);
        l.b = read_scalar_vec(is);
        if (l.w.size() != l.in_dim * l.out_dim || l.b.size() != l.out_dim) {
            throw std::runtime_error("LoopingRetNet load: invalid linear projection shape");
        }
        return l;
    }

    static uint64_t fnv1a_u64(uint64_t h, uint64_t x) {
        constexpr uint64_t kPrime = 1099511628211ULL;
        h ^= x;
        h *= kPrime;
        return h;
    }

    static uint64_t hash_scalar_vec(uint64_t h, const std::vector<Scalar>& vec) {
        h = fnv1a_u64(h, static_cast<uint64_t>(vec.size()));
        for (Scalar s : vec) {
            const float f = static_cast<float>(s);
            const uint32_t* bits = reinterpret_cast<const uint32_t*>(&f);
            h = fnv1a_u64(h, static_cast<uint64_t>(*bits));
        }
        return h;
    }

    static uint64_t hash_linear(uint64_t h, const LinearProj& l) {
        h = fnv1a_u64(h, static_cast<uint64_t>(l.in_dim));
        h = fnv1a_u64(h, static_cast<uint64_t>(l.out_dim));
        h = hash_scalar_vec(h, l.w);
        h = hash_scalar_vec(h, l.b);
        return h;
    }

    Vector apply_hidden_stack(const Vector& input) const {
        Vector out = input;
        for (const auto& layer : hidden_layers_) {
            out = swiglu_self(layer.forward(out));
        }
        return out;
    }

public:
    explicit LoopingRetNet(const LoopConfig& cfg, uint32_t seed = 42)
        : cfg_(cfg), chrono_attention_(0.8f), krv_attention_(0.8f)
    {
        std::mt19937 rng(seed);
        const float s = 0.02f;

        embed_table_.resize(cfg.char_vocab, Vector(cfg.model_dim, static_cast<Scalar>(0.0f)));
        std::normal_distribution<float> dist(0.0f, s);
        for (auto& vec : embed_table_) {
            for (Scalar& e : vec) { e = static_cast<Scalar>(dist(rng)); }
        }

        proj_q_     = LinearProj(cfg.model_dim, cfg.qk_dim,    s, rng);
        gate_q_     = LinearProj(cfg.model_dim, cfg.qk_dim,    s, rng);
        proj_k_     = LinearProj(cfg.model_dim, cfg.qk_dim,    s, rng);
        gate_k_     = LinearProj(cfg.model_dim, cfg.qk_dim,    s, rng);
        proj_v_     = LinearProj(cfg.model_dim, cfg.v_dim,     s, rng);
        gate_v_     = LinearProj(cfg.model_dim, cfg.v_dim,     s, rng);
        proj_r_     = LinearProj(cfg.model_dim, cfg.rel_dim,   s, rng);
        gate_r_     = LinearProj(cfg.model_dim, cfg.rel_dim,   s, rng);
        loop_proj_  = LinearProj(cfg.v_dim,     cfg.model_dim, s, rng);
        hidden_layers_.clear();
        hidden_layers_.reserve(cfg.hidden_layers);
        for (size_t i = 0; i < cfg.hidden_layers; ++i) {
            hidden_layers_.emplace_back(cfg.v_dim, cfg.v_dim, s, rng);
        }
        action_head_= LinearProj(cfg.v_dim,     4,             s, rng);
        load_head_  = LinearProj(cfg.v_dim,     1,             s, rng);
        chrono_mix_head_ = LinearProj(cfg.v_dim, 1,            s, rng);
        krv_mix_head_ = LinearProj(cfg.v_dim,    1,            s, rng);
        memory_entry_head_ = LinearProj(cfg.v_dim, cfg.qk_dim + cfg.rel_dim + cfg.v_dim, s, rng);
        krv_order_head_ = LinearProj(cfg.v_dim,  1,            s, rng);
        output_head_= LinearProj(cfg.v_dim,     cfg.char_vocab, s, rng);
        output_theta_ = static_cast<Scalar>(1.0f);
    }

    const LoopConfig& config() const {
        return cfg_;
    }

    void save_to_file(const std::string& path) const {
        std::ofstream os(path, std::ios::binary | std::ios::trunc);
        if (!os) {
            throw std::runtime_error("LoopingRetNet save: failed to open output file");
        }

        write_u32(os, kModelMagic);
        write_u32(os, kModelVersion);

        write_u64(os, static_cast<uint64_t>(cfg_.char_vocab));
        write_u64(os, static_cast<uint64_t>(cfg_.model_dim));
        write_u64(os, static_cast<uint64_t>(cfg_.qk_dim));
        write_u64(os, static_cast<uint64_t>(cfg_.v_dim));
        write_u64(os, static_cast<uint64_t>(cfg_.rel_dim));
        write_u64(os, static_cast<uint64_t>(cfg_.hidden_layers));
        os.write(reinterpret_cast<const char*>(&cfg_.decay), sizeof(cfg_.decay));
        write_u64(os, static_cast<uint64_t>(cfg_.max_steps));

        write_u64(os, static_cast<uint64_t>(embed_table_.size()));
        for (const Vector& v : embed_table_) {
            write_scalar_vec(os, v);
        }

        write_linear(os, proj_q_);
        write_linear(os, gate_q_);
        write_linear(os, proj_k_);
        write_linear(os, gate_k_);
        write_linear(os, proj_v_);
        write_linear(os, gate_v_);
        write_linear(os, proj_r_);
        write_linear(os, gate_r_);
        write_linear(os, loop_proj_);
        write_u64(os, static_cast<uint64_t>(hidden_layers_.size()));
        for (const auto& layer : hidden_layers_) {
            write_linear(os, layer);
        }
        write_linear(os, action_head_);
        write_linear(os, load_head_);
        write_linear(os, chrono_mix_head_);
        write_linear(os, krv_mix_head_);
        write_linear(os, memory_entry_head_);
        write_linear(os, krv_order_head_);
        write_linear(os, output_head_);
        os.write(reinterpret_cast<const char*>(&output_theta_), sizeof(output_theta_));

        if (!os) {
            throw std::runtime_error("LoopingRetNet save: failed while writing model");
        }
    }

    static LoopingRetNet load_from_file(const std::string& path) {
        std::ifstream is(path, std::ios::binary);
        if (!is) {
            throw std::runtime_error("LoopingRetNet load: failed to open input file");
        }

        const uint32_t magic = read_u32(is);
        const uint32_t version = read_u32(is);
        if (magic != kModelMagic) {
            throw std::runtime_error("LoopingRetNet load: invalid file magic");
        }
        if (version != 1 && version != 2 && version != 3 && version != kModelVersion) {
            throw std::runtime_error("LoopingRetNet load: unsupported version");
        }

        LoopConfig cfg;
        cfg.char_vocab = static_cast<size_t>(read_u64(is));
        cfg.model_dim = static_cast<size_t>(read_u64(is));
        cfg.qk_dim = static_cast<size_t>(read_u64(is));
        cfg.v_dim = static_cast<size_t>(read_u64(is));
        cfg.rel_dim = static_cast<size_t>(read_u64(is));
        cfg.hidden_layers = (version >= 2)
            ? static_cast<size_t>(read_u64(is))
            : 0;
        is.read(reinterpret_cast<char*>(&cfg.decay), sizeof(cfg.decay));
        cfg.max_steps = static_cast<size_t>(read_u64(is));

        LoopingRetNet model(cfg, 0);

        const uint64_t embed_rows = read_u64(is);
        model.embed_table_.assign(static_cast<size_t>(embed_rows), Vector{});
        for (Vector& v : model.embed_table_) {
            v = read_scalar_vec(is);
        }

        model.proj_q_ = read_linear(is);
        model.gate_q_ = read_linear(is);
        model.proj_k_ = read_linear(is);
        model.gate_k_ = read_linear(is);
        model.proj_v_ = read_linear(is);
        model.gate_v_ = read_linear(is);
        model.proj_r_ = read_linear(is);
        model.gate_r_ = read_linear(is);
        model.loop_proj_ = read_linear(is);
        if (version >= 2) {
            const size_t hidden_n = static_cast<size_t>(read_u64(is));
            model.hidden_layers_.clear();
            model.hidden_layers_.reserve(hidden_n);
            for (size_t i = 0; i < hidden_n; ++i) {
                model.hidden_layers_.push_back(read_linear(is));
            }
            model.cfg_.hidden_layers = hidden_n;
        } else {
            model.hidden_layers_.clear();
            model.cfg_.hidden_layers = 0;
        }
        model.action_head_ = read_linear(is);
        if (model.action_head_.out_dim == 3) {
            LinearProj expanded;
            expanded.in_dim = cfg.v_dim;
            expanded.out_dim = 4;
            expanded.w.assign(expanded.in_dim * expanded.out_dim, static_cast<Scalar>(0.0f));
            expanded.b.assign(expanded.out_dim, static_cast<Scalar>(0.0f));
            for (size_t o = 0; o < 3; ++o) {
                expanded.b[o] = model.action_head_.b[o];
                for (size_t i = 0; i < expanded.in_dim; ++i) {
                    expanded.w[o * expanded.in_dim + i] =
                        model.action_head_.w[o * expanded.in_dim + i];
                }
            }
            model.action_head_ = std::move(expanded);
        }
        if (version >= 3) {
            model.load_head_ = read_linear(is);
        } else {
            model.load_head_.in_dim = cfg.v_dim;
            model.load_head_.out_dim = 1;
            model.load_head_.w.assign(cfg.v_dim, static_cast<Scalar>(0.0f));
            model.load_head_.b.assign(1, static_cast<Scalar>(4.0f));
        }
        if (version >= 4) {
            model.chrono_mix_head_ = read_linear(is);
            model.krv_mix_head_ = read_linear(is);
            model.memory_entry_head_ = read_linear(is);
            model.krv_order_head_ = read_linear(is);
        } else {
            model.chrono_mix_head_.in_dim = cfg.v_dim;
            model.chrono_mix_head_.out_dim = 1;
            model.chrono_mix_head_.w.assign(cfg.v_dim, static_cast<Scalar>(0.0f));
            model.chrono_mix_head_.b.assign(1, static_cast<Scalar>(0.0f));

            model.krv_mix_head_.in_dim = cfg.v_dim;
            model.krv_mix_head_.out_dim = 1;
            model.krv_mix_head_.w.assign(cfg.v_dim, static_cast<Scalar>(0.0f));
            model.krv_mix_head_.b.assign(1, static_cast<Scalar>(0.0f));

            model.memory_entry_head_.in_dim = cfg.v_dim;
            model.memory_entry_head_.out_dim = cfg.qk_dim + cfg.rel_dim + cfg.v_dim;
            model.memory_entry_head_.w.assign(
                model.memory_entry_head_.in_dim * model.memory_entry_head_.out_dim,
                static_cast<Scalar>(0.0f));
            model.memory_entry_head_.b.assign(
                model.memory_entry_head_.out_dim,
                static_cast<Scalar>(0.0f));

            model.krv_order_head_.in_dim = cfg.v_dim;
            model.krv_order_head_.out_dim = 1;
            model.krv_order_head_.w.assign(cfg.v_dim, static_cast<Scalar>(0.0f));
            model.krv_order_head_.b.assign(1, static_cast<Scalar>(0.0f));
        }
        model.output_head_ = read_linear(is);
        is.read(reinterpret_cast<char*>(&model.output_theta_), sizeof(model.output_theta_));

        if (!is) {
            throw std::runtime_error("LoopingRetNet load: failed while reading model");
        }

        return model;
    }

    uint64_t parameter_checksum() const {
        uint64_t h = 1469598103934665603ULL;
        h = fnv1a_u64(h, static_cast<uint64_t>(cfg_.char_vocab));
        h = fnv1a_u64(h, static_cast<uint64_t>(cfg_.model_dim));
        h = fnv1a_u64(h, static_cast<uint64_t>(cfg_.qk_dim));
        h = fnv1a_u64(h, static_cast<uint64_t>(cfg_.v_dim));
        h = fnv1a_u64(h, static_cast<uint64_t>(cfg_.rel_dim));
        h = fnv1a_u64(h, static_cast<uint64_t>(cfg_.hidden_layers));
        h = fnv1a_u64(h, static_cast<uint64_t>(cfg_.max_steps));

        for (const Vector& v : embed_table_) {
            h = hash_scalar_vec(h, v);
        }

        h = hash_linear(h, proj_q_);
        h = hash_linear(h, gate_q_);
        h = hash_linear(h, proj_k_);
        h = hash_linear(h, gate_k_);
        h = hash_linear(h, proj_v_);
        h = hash_linear(h, gate_v_);
        h = hash_linear(h, proj_r_);
        h = hash_linear(h, gate_r_);
        h = hash_linear(h, loop_proj_);
        for (const auto& layer : hidden_layers_) {
            h = hash_linear(h, layer);
        }
        h = hash_linear(h, action_head_);
        h = hash_linear(h, load_head_);
        h = hash_linear(h, chrono_mix_head_);
        h = hash_linear(h, krv_mix_head_);
        h = hash_linear(h, memory_entry_head_);
        h = hash_linear(h, krv_order_head_);
        h = hash_linear(h, output_head_);
        h = fnv1a_u64(h, static_cast<uint64_t>(static_cast<float>(output_theta_) * 1000000.0f));
        return h;
    }

    std::vector<Scalar> export_parameters() const {
        std::vector<Scalar> flat;

        size_t total = 0;
        for (const Vector& v : embed_table_) {
            total += v.size();
        }
        total += proj_q_.w.size() + proj_q_.b.size();
        total += gate_q_.w.size() + gate_q_.b.size();
        total += proj_k_.w.size() + proj_k_.b.size();
        total += gate_k_.w.size() + gate_k_.b.size();
        total += proj_v_.w.size() + proj_v_.b.size();
        total += gate_v_.w.size() + gate_v_.b.size();
        total += proj_r_.w.size() + proj_r_.b.size();
        total += gate_r_.w.size() + gate_r_.b.size();
        total += loop_proj_.w.size() + loop_proj_.b.size();
        for (const auto& l : hidden_layers_) {
            total += l.w.size() + l.b.size();
        }
        total += action_head_.w.size() + action_head_.b.size();
        total += load_head_.w.size() + load_head_.b.size();
        total += chrono_mix_head_.w.size() + chrono_mix_head_.b.size();
        total += krv_mix_head_.w.size() + krv_mix_head_.b.size();
        total += memory_entry_head_.w.size() + memory_entry_head_.b.size();
        total += krv_order_head_.w.size() + krv_order_head_.b.size();
        total += output_head_.w.size() + output_head_.b.size();
        total += 1;

        flat.reserve(total);
        for (const Vector& v : embed_table_) {
            flat.insert(flat.end(), v.begin(), v.end());
        }

        auto append_linear = [&flat](const LinearProj& l) {
            flat.insert(flat.end(), l.w.begin(), l.w.end());
            flat.insert(flat.end(), l.b.begin(), l.b.end());
        };

        append_linear(proj_q_);
        append_linear(gate_q_);
        append_linear(proj_k_);
        append_linear(gate_k_);
        append_linear(proj_v_);
        append_linear(gate_v_);
        append_linear(proj_r_);
        append_linear(gate_r_);
        append_linear(loop_proj_);
        for (const auto& l : hidden_layers_) {
            append_linear(l);
        }
        append_linear(action_head_);
        append_linear(load_head_);
        append_linear(chrono_mix_head_);
        append_linear(krv_mix_head_);
        append_linear(memory_entry_head_);
        append_linear(krv_order_head_);
        append_linear(output_head_);
        flat.push_back(output_theta_);

        return flat;
    }

    std::vector<Scalar*> parameter_references() {
        std::vector<Scalar*> refs;
        refs.reserve(parameter_count());

        for (Vector& v : embed_table_) {
            for (Scalar& s : v) {
                refs.push_back(&s);
            }
        }

        auto append_linear = [&refs](LinearProj& l) {
            for (Scalar& s : l.w) {
                refs.push_back(&s);
            }
            for (Scalar& s : l.b) {
                refs.push_back(&s);
            }
        };

        append_linear(proj_q_);
        append_linear(gate_q_);
        append_linear(proj_k_);
        append_linear(gate_k_);
        append_linear(proj_v_);
        append_linear(gate_v_);
        append_linear(proj_r_);
        append_linear(gate_r_);
        append_linear(loop_proj_);
        for (auto& l : hidden_layers_) {
            append_linear(l);
        }
        append_linear(action_head_);
        append_linear(load_head_);
        append_linear(chrono_mix_head_);
        append_linear(krv_mix_head_);
        append_linear(memory_entry_head_);
        append_linear(krv_order_head_);
        append_linear(output_head_);

        refs.push_back(&output_theta_);
        return refs;
    }

    size_t parameter_count() const {
        size_t total = 1;
        for (const Vector& v : embed_table_) {
            total += v.size();
        }

        auto add_linear = [&total](const LinearProj& l) {
            total += l.w.size() + l.b.size();
        };

        add_linear(proj_q_);
        add_linear(gate_q_);
        add_linear(proj_k_);
        add_linear(gate_k_);
        add_linear(proj_v_);
        add_linear(gate_v_);
        add_linear(proj_r_);
        add_linear(gate_r_);
        add_linear(loop_proj_);
        for (const auto& l : hidden_layers_) {
            add_linear(l);
        }
        add_linear(action_head_);
        add_linear(load_head_);
        add_linear(chrono_mix_head_);
        add_linear(krv_mix_head_);
        add_linear(memory_entry_head_);
        add_linear(krv_order_head_);
        add_linear(output_head_);
        return total;
    }

    void import_parameters(const std::vector<Scalar>& flat) {
        size_t idx = 0;

        auto take_into = [&flat, &idx](std::vector<Scalar>& dst) {
            if (idx + dst.size() > flat.size()) {
                throw std::runtime_error("LoopingRetNet import: parameter vector too small");
            }
            std::copy(flat.begin() + static_cast<std::ptrdiff_t>(idx),
                      flat.begin() + static_cast<std::ptrdiff_t>(idx + dst.size()),
                      dst.begin());
            idx += dst.size();
        };

        for (Vector& v : embed_table_) {
            take_into(v);
        }

        auto load_linear = [&take_into](LinearProj& l) {
            take_into(l.w);
            take_into(l.b);
        };

        load_linear(proj_q_);
        load_linear(gate_q_);
        load_linear(proj_k_);
        load_linear(gate_k_);
        load_linear(proj_v_);
        load_linear(gate_v_);
        load_linear(proj_r_);
        load_linear(gate_r_);
        load_linear(loop_proj_);
        for (auto& l : hidden_layers_) {
            load_linear(l);
        }
        load_linear(action_head_);
        load_linear(load_head_);
        load_linear(chrono_mix_head_);
        load_linear(krv_mix_head_);
        load_linear(memory_entry_head_);
        load_linear(krv_order_head_);
        load_linear(output_head_);

        if (idx + 1 > flat.size()) {
            throw std::runtime_error("LoopingRetNet import: missing output_theta");
        }
        output_theta_ = flat[idx++];

        if (idx != flat.size()) {
            throw std::runtime_error("LoopingRetNet import: parameter vector too large");
        }
    }

    // ── Per-token step ────────────────────────────────────────────────────────

    struct StepOutput {
        size_t output_token;
        size_t inner_steps_taken;
    };

    struct StepTrace {
        size_t output_token;
        size_t inner_steps_taken;
        size_t query_count;
        size_t generated_memory_count;
        bool has_generated_memory;
        Vector logits;
        Vector raw_logits;
        Vector final_state;
        Vector generated_memory_state;
        Vector generated_key;
        Vector generated_relation;
        Vector generated_value;
        std::vector<Vector> per_step_output_logits;
        std::vector<Vector> per_step_action_logits;
        std::vector<Vector> per_step_states;
        std::vector<Scalar> per_step_load_logits;
        std::vector<uint8_t> per_step_query_hits;
    };

    // Processes one input token ID.
    //
    // recurrent_state: KVState carrying the RetNet hidden state (read+write).
    // chrono_kv_cache: Chronologically ordered in-sequence KV attention cache;
    //                  aggressively pruned and never written to long-term graph.
    // krv_cache:       Retentive KRV cache; interacts with graph and evicted
    //                  entries are written back to long-term memory.
    // bridge:          Writes active KV entries into the long-term graph.
    // query_engine:    Performs multihop graph lookups on demand.
    //
    // Returns the emitted character and how many inner steps were consumed.
    StepTrace step_with_trace(
        size_t                    input_token,
        KVState&                  recurrent_state,
        AttentionMemory&          chrono_kv_cache,
        AttentionMemory&          krv_cache,
        memory::GraphMemoryBridge& bridge,
        memory::MultiHopQuery&    query_engine,
        bool                      force_output = false,
        bool                      enable_query = true,
        size_t                    forced_loop_count = 0,
        bool                      use_parallel_retention = false,
        bool                      enable_memory_write = true,
        bool                      force_query_first = false,
        const ActionControl*      action_control = nullptr)
    {
        // x starts as the token embedding and may be overridden to the
        // fused-state loop projection on subsequent inner steps.
        Vector x = embed(input_token);

        size_t query_count = 0;
        size_t generated_memory_count = 0;
        bool has_generated_memory = false;
        Vector generated_memory_state;
        Vector generated_key;
        Vector generated_relation;
        Vector generated_value;
        Matrix q_hist;
        Matrix k_hist;
        Matrix v_hist;
        std::vector<Vector> all_output_logits;
        std::vector<Vector> all_action_logits;
        std::vector<Vector> all_states;
        std::vector<Scalar> all_load_logits;
        std::vector<uint8_t> all_query_hits;

        const size_t forced_loops = (forced_loop_count == 0)
            ? 0
            : std::max<size_t>(1, std::min<size_t>(forced_loop_count, std::min<size_t>(5, cfg_.max_steps)));

        for (size_t step_i = 0; step_i < cfg_.max_steps; ++step_i) {
            // ── Project into Q/K/V/R ─────────────────────────────────────
            const Vector q = activation::swiglu(proj_q_.forward(x), gate_q_.forward(x));
            const Vector k = activation::swiglu(proj_k_.forward(x), gate_k_.forward(x));
            const Vector v = activation::swiglu(proj_v_.forward(x), gate_v_.forward(x));
            const Vector r = activation::swiglu(proj_r_.forward(x), gate_r_.forward(x));

            // ── RetNet recurrent step ────────────────────────────────────
            Vector ret_out;
            if (use_parallel_retention) {
                q_hist.push_back(q);
                k_hist.push_back(k);
                v_hist.push_back(v);

                Matrix decay_mask(q_hist.size(), Vector(q_hist.size(), static_cast<Scalar>(0.0f)));
                for (size_t i = 0; i < q_hist.size(); ++i) {
                    for (size_t j = 0; j <= i; ++j) {
                        decay_mask[i][j] = static_cast<Scalar>(
                            std::pow(cfg_.decay, static_cast<float>(i - j)));
                    }
                }

                const Matrix par = Retention::parallel(q_hist, k_hist, v_hist, decay_mask);
                ret_out = par.back();
            } else {
                auto [recur_out, new_state] =
                    Retention::recurrent(q, k, v, recurrent_state, cfg_.decay);
                ret_out = std::move(recur_out);
                recurrent_state = std::move(new_state);
            }

            // ── Dual attention heads ──────────────────────────────────────
            AttentionMemory current_chrono;
            current_chrono.push_back(k, relation_zeros(cfg_.rel_dim), v, /*prune_score=*/0.0f);

            AttentionMemory current_krv;
            current_krv.push_back(k, r, v, /*prune_score=*/0.0f);

            const Matrix chrono_attn_out = chrono_attention_.compute(Matrix{q}, current_chrono, chrono_kv_cache);
            const Matrix krv_attn_out = krv_attention_.compute(Matrix{q}, current_krv, krv_cache);
            const Vector chrono_attn_vec = chrono_attn_out.empty() ? ret_out : chrono_attn_out.front();
            const Vector krv_attn_vec = krv_attn_out.empty() ? ret_out : krv_attn_out.front();

            const float chrono_mix = sigmoid(static_cast<float>(chrono_mix_head_.forward(ret_out)[0]));
            const float krv_mix = sigmoid(static_cast<float>(krv_mix_head_.forward(ret_out)[0]));

            // ── Fuse ─────────────────────────────────────────────────────
            Vector state = fuse_dual(ret_out, chrono_attn_vec, krv_attn_vec, chrono_mix, krv_mix);
            state = apply_hidden_stack(state);

            // ── Update dual caches and write to graph where applicable ───
            chrono_kv_cache.push_back(k, relation_zeros(cfg_.rel_dim), v, /*prune_score=*/0.0f);
            prune_chrono_cache(chrono_kv_cache, kv_cache_limit());

            if (enable_memory_write) {
                const float order_importance = sigmoid(static_cast<float>(krv_order_head_.forward(state)[0]));
                const float prune_score = 1.0f - order_importance;
                AttentionMemory current_write;
                current_write.push_back(k, r, v, prune_score);
                bridge.write(current_write);
                krv_cache.push_back(k, r, v, prune_score);
                prune_krv_cache_with_graph_write(krv_cache, bridge, enable_memory_write);
            }

            // ── Action decision ───────────────────────────────────────────
            const bool last_step = (step_i == cfg_.max_steps - 1);
            Vector action_logits = action_head_.forward(state);
            if (action_control && action_control->enabled && action_logits.size() >= 4) {
                action_logits[0] = static_cast<Scalar>(
                    static_cast<float>(action_logits[0]) + action_control->output_bias);
                action_logits[1] = static_cast<Scalar>(
                    static_cast<float>(action_logits[1]) + action_control->query_bias);
                action_logits[2] = static_cast<Scalar>(
                    static_cast<float>(action_logits[2]) + action_control->loop_bias);
                action_logits[3] = static_cast<Scalar>(
                    static_cast<float>(action_logits[3]) + action_control->create_bias);
            }
            const Vector load_logits = load_head_.forward(state);
            const float load_logit = load_logits.empty() ? 0.0f : static_cast<float>(load_logits[0]);
            ModelAction action = ModelAction::OUTPUT;
            if (!force_output) {
                if (action_logits.size() >= 4) {
                    float best = static_cast<float>(action_logits[0]);
                    size_t best_i = 0;
                    for (size_t ai = 1; ai < 4; ++ai) {
                        const float av = static_cast<float>(action_logits[ai]);
                        if (av > best) {
                            best = av;
                            best_i = ai;
                        }
                    }
                    action = static_cast<ModelAction>(best_i);
                } else {
                    action = pick_action(state);
                }
            }

            if (forced_loops > 0) {
                action = (step_i + 1 < forced_loops) ? ModelAction::LOOP : ModelAction::OUTPUT;
            }
            if (!force_output && force_query_first && enable_query && step_i == 0 && !last_step) {
                action = ModelAction::QUERY_MEMORY;
            }
            if (!force_output
                && action_control
                && action_control->force_create_memory_first
                && step_i == 0
                && !last_step) {
                action = ModelAction::CREATE_MEMORY;
            }

            const Vector raw_logits = output_head_.forward(state);
            const Vector logits = activation::param_tanh(raw_logits, output_theta_);
            all_output_logits.push_back(logits);
            all_action_logits.push_back(action_logits);
            all_states.push_back(state);
            all_load_logits.push_back(static_cast<Scalar>(load_logit));
            all_query_hits.push_back(0);

            // Force output on the last step to guarantee termination.
            if (last_step) { action = ModelAction::OUTPUT; }

            if (action == ModelAction::QUERY_MEMORY && enable_query) {
                // Multihop graph query; add results to KRV cache with penalty.
                AttentionMemory queried = query_engine.query(q);
                ++query_count;
                if (!queried.empty()) {
                    all_query_hits.back() = 1;
                }
                const float load_prob = 1.0f / (1.0f + std::exp(-load_logit));
                if (load_prob >= 0.5f) {
                    append_memory(krv_cache, std::move(queried));
                    prune_krv_cache_with_graph_write(krv_cache, bridge, enable_memory_write);
                }
                // Loop again using the fused state projected back to model_dim.
                x = activation::swiglu(loop_proj_.forward(state), loop_proj_.forward(state));
                continue;
            }

            if (action == ModelAction::CREATE_MEMORY) {
                const Vector mem_head = memory_entry_head_.forward(state);
                const size_t k_dim = cfg_.qk_dim;
                const size_t r_dim = cfg_.rel_dim;
                const size_t v_dim = cfg_.v_dim;
                if (mem_head.size() >= k_dim + r_dim + v_dim) {
                    Vector mk(k_dim, static_cast<Scalar>(0.0f));
                    Vector mr(r_dim, static_cast<Scalar>(0.0f));
                    Vector mv(v_dim, static_cast<Scalar>(0.0f));
                    for (size_t i = 0; i < k_dim; ++i) {
                        mk[i] = mem_head[i];
                    }
                    for (size_t i = 0; i < r_dim; ++i) {
                        mr[i] = mem_head[k_dim + i];
                    }
                    for (size_t i = 0; i < v_dim; ++i) {
                        mv[i] = mem_head[k_dim + r_dim + i];
                    }
                    mk = swiglu_self(mk);
                    mr = swiglu_self(mr);
                    mv = swiglu_self(mv);
                    has_generated_memory = true;
                    generated_memory_state = state;
                    generated_key = mk;
                    generated_relation = mr;
                    generated_value = mv;
                    const float order_importance = sigmoid(static_cast<float>(krv_order_head_.forward(state)[0]));
                    const float prune_score = 1.0f - order_importance;
                    AttentionMemory generated;
                    generated.push_back(std::move(mk), std::move(mr), std::move(mv), prune_score);
                    if (enable_memory_write) {
                        bridge.write(generated);
                    }
                    append_memory(krv_cache, std::move(generated));
                    prune_krv_cache_with_graph_write(krv_cache, bridge, enable_memory_write);
                    ++generated_memory_count;
                }
                x = activation::swiglu(loop_proj_.forward(state), loop_proj_.forward(state));
                continue;
            }

            if (action == ModelAction::LOOP) {
                // Think again without querying memory.
                x = activation::swiglu(loop_proj_.forward(state), loop_proj_.forward(state));
                continue;
            }

            // action == OUTPUT
            StepTrace trace;
            trace.output_token = pick_token(state);
            trace.inner_steps_taken = step_i + 1;
            trace.query_count = query_count;
            trace.generated_memory_count = generated_memory_count;
            trace.has_generated_memory = has_generated_memory;
            trace.logits = logits;
            trace.raw_logits = raw_logits;
            trace.final_state = state;
            trace.generated_memory_state = std::move(generated_memory_state);
            trace.generated_key = std::move(generated_key);
            trace.generated_relation = std::move(generated_relation);
            trace.generated_value = std::move(generated_value);
            trace.per_step_output_logits = std::move(all_output_logits);
            trace.per_step_action_logits = std::move(all_action_logits);
            trace.per_step_states = std::move(all_states);
            trace.per_step_load_logits = std::move(all_load_logits);
            trace.per_step_query_hits = std::move(all_query_hits);
            return trace;
        }

        // Unreachable: last_step forces OUTPUT above.
        throw std::logic_error("LoopingRetNet::step: fell through step loop");
    }

    StepOutput step(
        size_t                    input_token,
        KVState&                  recurrent_state,
        AttentionMemory&          chrono_kv_cache,
        AttentionMemory&          krv_cache,
        memory::GraphMemoryBridge& bridge,
        memory::MultiHopQuery&    query_engine)
    {
        const StepTrace t = step_with_trace(
            input_token,
            recurrent_state,
            chrono_kv_cache,
            krv_cache,
            bridge,
            query_engine,
            false,
            true,
            0,
            false,
            true,
            false,
            nullptr);
        return {t.output_token, t.inner_steps_taken};
    }

    size_t output_dim() const {
        return output_head_.out_dim;
    }

    size_t output_input_dim() const {
        return output_head_.in_dim;
    }

    Scalar output_theta() const {
        return output_theta_;
    }

    void set_output_theta(Scalar theta) {
        output_theta_ = theta;
    }

    std::vector<Scalar>& output_head_weights() {
        return output_head_.w;
    }

    std::vector<Scalar>& output_head_bias() {
        return output_head_.b;
    }

    const std::vector<Scalar>& output_head_weights() const {
        return output_head_.w;
    }

    const std::vector<Scalar>& output_head_bias() const {
        return output_head_.b;
    }

    size_t action_dim() const {
        return action_head_.out_dim;
    }

    size_t action_input_dim() const {
        return action_head_.in_dim;
    }

    std::vector<Scalar>& action_head_weights() {
        return action_head_.w;
    }

    std::vector<Scalar>& action_head_bias() {
        return action_head_.b;
    }

    const std::vector<Scalar>& action_head_weights() const {
        return action_head_.w;
    }

    const std::vector<Scalar>& action_head_bias() const {
        return action_head_.b;
    }

    size_t load_gate_input_dim() const {
        return load_head_.in_dim;
    }

    std::vector<Scalar>& load_gate_weights() {
        return load_head_.w;
    }

    std::vector<Scalar>& load_gate_bias() {
        return load_head_.b;
    }

    const std::vector<Scalar>& load_gate_weights() const {
        return load_head_.w;
    }

    const std::vector<Scalar>& load_gate_bias() const {
        return load_head_.b;
    }

    size_t memory_entry_input_dim() const {
        return memory_entry_head_.in_dim;
    }

    size_t memory_entry_output_dim() const {
        return memory_entry_head_.out_dim;
    }

    std::vector<Scalar>& memory_entry_weights() {
        return memory_entry_head_.w;
    }

    std::vector<Scalar>& memory_entry_bias() {
        return memory_entry_head_.b;
    }

    const std::vector<Scalar>& memory_entry_weights() const {
        return memory_entry_head_.w;
    }

    const std::vector<Scalar>& memory_entry_bias() const {
        return memory_entry_head_.b;
    }

    size_t hidden_layer_count() const {
        return hidden_layers_.size();
    }

    std::vector<Vector>& embedding_table() {
        return embed_table_;
    }

    const std::vector<Vector>& embedding_table() const {
        return embed_table_;
    }

    LinearProj& proj_q_layer() { return proj_q_; }
    LinearProj& gate_q_layer() { return gate_q_; }
    LinearProj& proj_k_layer() { return proj_k_; }
    LinearProj& gate_k_layer() { return gate_k_; }
    LinearProj& proj_v_layer() { return proj_v_; }
    LinearProj& gate_v_layer() { return gate_v_; }
    LinearProj& proj_r_layer() { return proj_r_; }
    LinearProj& gate_r_layer() { return gate_r_; }
    LinearProj& loop_proj_layer() { return loop_proj_; }

    std::vector<LinearProj>& hidden_stack_layers() {
        return hidden_layers_;
    }

    const std::vector<LinearProj>& hidden_stack_layers() const {
        return hidden_layers_;
    }
};

} // namespace llm::arch
