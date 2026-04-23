#pragma once

// looping_retnet.hpp
//
// LoopingRetNet: character-based language model that combines:
//   • A RetNet recurrent core (recurrent mode, one step per character).
//   • An Attention module used *only* to read from long-term graph memory via
//     AttentionMemory (not for in-sequence context).
//   • A 3-class action head that chooses on every inner iteration:
//       0 – OUTPUT:        emit a character and advance to the next input token.
//       1 – QUERY_MEMORY:  run a multihop graph query, inject results into the
//                          KV cache, and loop again (no output yet).
//       2 – LOOP:          run the RetNet core again without querying or outputting.
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

enum class ModelAction : uint8_t { OUTPUT = 0, QUERY_MEMORY = 1, LOOP = 2 };

// ── Configuration ─────────────────────────────────────────────────────────────

struct LoopConfig {
    size_t char_vocab  = 256;  // byte-level vocabulary size
    size_t model_dim   = 128;  // embedding / hidden dimension
    size_t qk_dim      = 64;   // query/key dimension for RetNet and Attention
    size_t v_dim       = 64;   // value dimension (== semvec_dim for graph compat)
    size_t rel_dim     = 32;   // relation vector dimension (smaller edge label)
    float  decay       = 0.9f; // RetNet recurrent decay
    size_t max_steps   = 8;    // max inner iterations before forced output
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

    // Character embedding table: char_vocab × model_dim
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

    // 3-class action head.
    LinearProj action_head_; // v_dim → 3

    // Character output head.
    LinearProj output_head_; // v_dim → char_vocab
    Scalar output_theta_ = static_cast<Scalar>(1.0f);

    Attention attention_;

    static constexpr uint32_t kModelMagic = 0x4C524E54; // "LRNT"
    static constexpr uint32_t kModelVersion = 1;

    // ── Helpers ───────────────────────────────────────────────────────────────

    const Vector& embed(char c) const {
        return embed_table_[static_cast<uint8_t>(c)];
    }

    ModelAction pick_action(const Vector& state) const {
        const Vector logits = action_head_.forward(state);
        if (logits.size() < 3) { return ModelAction::OUTPUT; }

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

    char pick_char(const Vector& state) const {
        const Vector raw = output_head_.forward(state);
        const Vector logits = activation::param_tanh(raw, output_theta_);
        size_t best = 0;
        float  best_val = static_cast<float>(logits[0]);
        for (size_t i = 1; i < logits.size(); ++i) {
            const float v = static_cast<float>(logits[i]);
            if (v > best_val) { best_val = v; best = i; }
        }
        return static_cast<char>(best);
    }

    Vector output_logits(const Vector& state) const {
        const Vector raw = output_head_.forward(state);
        return activation::param_tanh(raw, output_theta_);
    }

    // Fuse RetNet output and attention output into a single state vector.
    // Both are v_dim; output is v_dim.
    static Vector fuse(const Vector& ret, const Vector& attn) {
        const size_t dim = std::min(ret.size(), attn.size());
        Vector out(dim, static_cast<Scalar>(0.0f));
        for (size_t i = 0; i < dim; ++i) {
            out[i] = static_cast<Scalar>(
                static_cast<float>(ret[i]) + static_cast<float>(attn[i]));
        }
        return out;
    }

    static Vector swiglu_self(const Vector& x) {
        // Self-gated SwiGLU block for unified non-final activations.
        return activation::swiglu(x, x);
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

public:
    explicit LoopingRetNet(const LoopConfig& cfg, uint32_t seed = 42)
        : cfg_(cfg), attention_(0.8f)
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
        action_head_= LinearProj(cfg.v_dim,     3,             s, rng);
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
        write_linear(os, action_head_);
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
        if (version != kModelVersion) {
            throw std::runtime_error("LoopingRetNet load: unsupported version");
        }

        LoopConfig cfg;
        cfg.char_vocab = static_cast<size_t>(read_u64(is));
        cfg.model_dim = static_cast<size_t>(read_u64(is));
        cfg.qk_dim = static_cast<size_t>(read_u64(is));
        cfg.v_dim = static_cast<size_t>(read_u64(is));
        cfg.rel_dim = static_cast<size_t>(read_u64(is));
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
        model.action_head_ = read_linear(is);
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
        h = hash_linear(h, action_head_);
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
        total += action_head_.w.size() + action_head_.b.size();
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
        append_linear(action_head_);
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
        append_linear(action_head_);
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
        add_linear(action_head_);
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
        load_linear(action_head_);
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
        char   output_char;
        size_t inner_steps_taken;
    };

    struct StepTrace {
        char   output_char;
        size_t inner_steps_taken;
        size_t query_count;
        Vector logits;
        std::vector<Vector> per_step_output_logits;
        std::vector<Vector> per_step_action_logits;
    };

    // Processes one input character token.
    //
    // recurrent_state: KVState carrying the RetNet hidden state (read+write).
    // kv_cache:        AttentionMemory used solely to bridge graph ↔ RetNet.
    //                  Accumulated across tokens; caller can prune it.
    // bridge:          Writes active KV entries into the long-term graph.
    // query_engine:    Performs multihop graph lookups on demand.
    //
    // Returns the emitted character and how many inner steps were consumed.
    StepTrace step_with_trace(
        char                      input_char,
        KVState&                  recurrent_state,
        AttentionMemory&          kv_cache,
        memory::GraphMemoryBridge& bridge,
        memory::MultiHopQuery&    query_engine,
        bool                      force_output = false,
        bool                      enable_query = true,
        size_t                    forced_loop_count = 0,
        bool                      use_parallel_retention = false)
    {
        // x starts as the character embedding and may be overridden to the
        // fused-state loop projection on subsequent inner steps.
        Vector x = embed(input_char);

        size_t query_count = 0;
        Matrix q_hist;
        Matrix k_hist;
        Matrix v_hist;
        std::vector<Vector> all_output_logits;
        std::vector<Vector> all_action_logits;

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

            // ── Attention over KV cache (graph bridge only) ───────────────
            // Build a one-entry current AttentionMemory for this step.
            AttentionMemory current;
            current.push_back(k, r, v, /*prune_score=*/0.0f);

            const Matrix attn_out = attention_.compute(Matrix{q}, current, kv_cache);
            const Vector attn_vec = attn_out.empty() ? ret_out : attn_out.front();

            // ── Fuse ─────────────────────────────────────────────────────
            Vector state = fuse(ret_out, attn_vec);

            // ── Write current entry to graph and accumulate in kv_cache ──
            bridge.write(current);
            kv_cache.push_back(k, r, v, /*prune_score=*/0.0f);
            kv_cache.prune_to_max(kv_cache_limit());

            // ── Action decision ───────────────────────────────────────────
            const bool last_step = (step_i == cfg_.max_steps - 1);
            Vector action_logits = action_head_.forward(state);
            ModelAction action = force_output ? ModelAction::OUTPUT : pick_action(state);

            if (forced_loops > 0) {
                action = (step_i + 1 < forced_loops) ? ModelAction::LOOP : ModelAction::OUTPUT;
            }

            const Vector logits = output_logits(state);
            all_output_logits.push_back(logits);
            all_action_logits.push_back(action_logits);

            // Force output on the last step to guarantee termination.
            if (last_step) { action = ModelAction::OUTPUT; }

            if (action == ModelAction::QUERY_MEMORY && enable_query) {
                // Multihop graph query; add results to kv_cache with penalty.
                AttentionMemory queried = query_engine.query(q);
                ++query_count;
                for (size_t qi = 0; qi < queried.size(); ++qi) {
                    kv_cache.push_back(
                        std::move(queried.keys[qi]),
                        std::move(queried.relations[qi]),
                        std::move(queried.values[qi]),
                        queried.prune_scores[qi]);
                }
                kv_cache.prune_to_max(kv_cache_limit());
                // Loop again using the fused state projected back to model_dim.
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
            trace.output_char = pick_char(state);
            trace.inner_steps_taken = step_i + 1;
            trace.query_count = query_count;
            trace.logits = logits;
            trace.per_step_output_logits = std::move(all_output_logits);
            trace.per_step_action_logits = std::move(all_action_logits);
            return trace;
        }

        // Unreachable: last_step forces OUTPUT above.
        throw std::logic_error("LoopingRetNet::step: fell through step loop");
    }

    StepOutput step(
        char                      input_char,
        KVState&                  recurrent_state,
        AttentionMemory&          kv_cache,
        memory::GraphMemoryBridge& bridge,
        memory::MultiHopQuery&    query_engine)
    {
        const StepTrace t = step_with_trace(
            input_char,
            recurrent_state,
            kv_cache,
            bridge,
            query_engine,
            false,
            true);
        return {t.output_char, t.inner_steps_taken};
    }
};

} // namespace llm::arch
