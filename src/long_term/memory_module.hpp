#pragma once

// memory_module.hpp
//
// Bridges the RetNet recurrent state with the long-term graph database via
// AttentionMemory.  Three components:
//
//  1. NodeCompressor  – encodes a full-dim Vector → SemVec via a linear
//                       projection (stand-in for a trained encoder).
//  2. GraphMemoryBridge – examines an AttentionMemory and writes edges into
//                         Graph.  Key → from-node, relation → edge semvec,
//                         value → to-node (both compressed).  Nodes merge
//                         when their L2² distance is below a threshold.
//  3. MultiHopQuery   – BFS from the nearest graph node, bounded by
//                       max_hop_depth and max_hop_breadth, returns all
//                       traversed edges as AttentionMemory with a freshness
//                       prune-score penalty.

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <architecture/sequential.hpp>
#include <architecture/attention.hpp>
#include <long_term/graph.hpp>
#include <long_term/spatial_map.hpp>

namespace llm::memory {

// ── Utility helpers ──────────────────────────────────────────────────────────

// Convert a float16 Vector to a heap-allocated SemVec of the same dimension.
inline SemVec vec_to_semvec(const arch::Vector& v) {
    SemVec sv(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        sv.data[i] = static_cast<float16_t>(v[i]);
    }
    return sv;
}

// Convert a SemVec back to a float16 Vector.
inline arch::Vector semvec_to_vec(const SemVec& sv) {
    arch::Vector v(sv.dimensions);
    for (size_t i = 0; i < sv.dimensions; ++i) {
        v[i] = static_cast<arch::Scalar>(sv.data[i]);
    }
    return v;
}

// Squared L2 distance between two same-dimension SemVecs.
inline float semvec_l2sq(const SemVec& a, const SemVec& b) {
    if (a.dimensions != b.dimensions) {
        throw std::runtime_error("semvec_l2sq: dimension mismatch");
    }
    float d = 0.0f;
    for (size_t i = 0; i < a.dimensions; ++i) {
        const float diff = static_cast<float>(a.data[i]) - static_cast<float>(b.data[i]);
        d += diff * diff;
    }
    return d;
}

// ── Configuration ─────────────────────────────────────────────────────────────

struct MemoryConfig {
    size_t semvec_dim        = 64;   // SemVec dimension (must match qk_dim in model)
    float  merge_threshold_sq = 0.01f; // L2² below which two nodes are merged
    float  fresh_prune_penalty = 2.0f; // added to prune_score for all queried edges
    size_t max_hop_depth     = 3;    // BFS depth limit
    size_t max_hop_breadth   = 8;    // max edges followed per frontier node per hop
    size_t max_write_entries = 4;    // max KV entries written to graph per call
};

// ── NodeCompressor ────────────────────────────────────────────────────────────
//
// Projects an arbitrary-length Vector down to semvec_dim using a learned
// linear layer (simple stand-in; swap with a proper encoder at training time).

class NodeCompressor {
    std::vector<float> w_; // semvec_dim × in_dim row-major
    size_t in_dim_;
    size_t out_dim_;

public:
    // Construct with a random initialised projection.
    NodeCompressor(size_t in_dim, size_t out_dim, uint32_t seed = 42)
        : w_(out_dim * in_dim), in_dim_(in_dim), out_dim_(out_dim)
    {
        // Kaiming-like init: scale = sqrt(2/in_dim)
        const float scale = std::sqrt(2.0f / static_cast<float>(in_dim));
        std::mt19937 rng(seed);
        std::normal_distribution<float> dist(0.0f, scale);
        for (float& f : w_) { f = dist(rng); }
    }

    // Construct from an existing weight matrix (row-major, out×in).
    NodeCompressor(std::vector<float> weights, size_t in_dim, size_t out_dim)
        : w_(std::move(weights)), in_dim_(in_dim), out_dim_(out_dim)
    {
        if (w_.size() != out_dim_ * in_dim_) {
            throw std::runtime_error("NodeCompressor: weight size mismatch");
        }
    }

    SemVec compress(const arch::Vector& v) const {
        SemVec sv(out_dim_);
        const size_t usable = std::min(v.size(), in_dim_);
        for (size_t o = 0; o < out_dim_; ++o) {
            float acc = 0.0f;
            for (size_t i = 0; i < usable; ++i) {
                acc += w_[o * in_dim_ + i] * static_cast<float>(v[i]);
            }
            sv.data[o] = static_cast<float16_t>(acc);
        }
        return sv;
    }
};

// ── GraphMemoryBridge ─────────────────────────────────────────────────────────
//
// Writes AttentionMemory entries into the Graph.
//   key      → from-node  SemVec
//   relation → edge       SemVec
//   value    → to-node    SemVec  (both key and value compressed by NodeCompressor)
//
// Nodes whose compressed representation is within merge_threshold_sq are
// merged (the existing node ID is reused; no duplicate inserted).

class GraphMemoryBridge {
    Graph&          graph_;
    SpatialMap&     spatial_map_;
    NodeCompressor  compressor_;
    MemoryConfig    cfg_;
    size_t          next_node_id_ = 0;

    // Returns the ID of an existing node close enough to sv, or inserts a new
    // one and returns its ID.  Modifies graph_ and spatial_map_.
    size_t find_or_create_node(SemVec& sv) {
        std::vector<std::pair<size_t, SemVec*>> neighbors;
        spatial_map_.get_nearest_neighbors(&sv, 1, neighbors);

        if (!neighbors.empty()) {
            const float dist_sq = semvec_l2sq(sv, *neighbors[0].second);
            if (dist_sq < cfg_.merge_threshold_sq) {
                return neighbors[0].first; // merge: reuse existing node
            }
        }

        // Create a new node.  Graph::add_node takes SemVec by value (move).
        const size_t id = next_node_id_++;
        SemVec node_sv(sv.dimensions);
        std::memcpy(node_sv.data, sv.data, sv.dimensions * sizeof(float16_t));
        graph_.add_node(id, std::move(node_sv));

        SemVec* stored = graph_.get_node_sem_vec(id);
        if (stored) {
            spatial_map_.insert(id, stored);
        }
        return id;
    }

public:
    GraphMemoryBridge(Graph& g, SpatialMap& sm, NodeCompressor comp,
                      MemoryConfig cfg = {})
        : graph_(g), spatial_map_(sm), compressor_(std::move(comp)), cfg_(cfg) {}

    // Writes up to cfg_.max_write_entries edges from mem into the graph.
    void write(const arch::AttentionMemory& mem) {
        const size_t n = std::min({
            mem.keys.size(), mem.relations.size(), mem.values.size(),
            cfg_.max_write_entries
        });

        for (size_t i = 0; i < n; ++i) {
            SemVec from_sv = compressor_.compress(mem.keys[i]);
            SemVec rel_sv  = compressor_.compress(mem.relations[i]);
            SemVec to_sv   = compressor_.compress(mem.values[i]);

            const size_t from_id = find_or_create_node(from_sv);
            const size_t to_id   = find_or_create_node(to_sv);

            // Copy rel_sv as edge label (Graph takes SemVec by move).
            SemVec edge_rel(rel_sv.dimensions);
            std::memcpy(edge_rel.data, rel_sv.data, rel_sv.dimensions * sizeof(float16_t));

            graph_.add_edge(from_id, to_id, std::move(edge_rel), 1.0f);
        }
    }
};

// ── MultiHopQuery ─────────────────────────────────────────────────────────────
//
// BFS from the nearest node to query_vec.  At each hop, up to max_hop_breadth
// outgoing edges are followed.  All visited edges are returned as an
// AttentionMemory whose prune_scores are set to fresh_prune_penalty so they
// are preferentially evicted relative to entries derived from the active
// token step.

class MultiHopQuery {
    Graph&      graph_;
    SpatialMap& spatial_map_;
    MemoryConfig cfg_;

public:
    MultiHopQuery(Graph& g, SpatialMap& sm, MemoryConfig cfg = {})
        : graph_(g), spatial_map_(sm), cfg_(cfg) {}

    // query_vec should be the projected query vector from the current step
    // (same dimension as cfg_.semvec_dim).
    arch::AttentionMemory query(const arch::Vector& query_vec) {
        arch::AttentionMemory result;

        // Find the nearest starting node.
        SemVec qsv(query_vec.size());
        for (size_t i = 0; i < query_vec.size(); ++i) {
            qsv.data[i] = static_cast<float16_t>(query_vec[i]);
        }

        std::vector<std::pair<size_t, SemVec*>> starts;
        spatial_map_.get_nearest_neighbors(&qsv, 1, starts);
        if (starts.empty()) { return result; }

        // BFS.
        std::vector<size_t> frontier = { starts[0].first };
        std::unordered_set<size_t> visited;
        visited.insert(frontier[0]);

        for (size_t depth = 0; depth < cfg_.max_hop_depth && !frontier.empty(); ++depth) {
            std::vector<size_t> next_frontier;

            for (const size_t node_id : frontier) {
                SemVec* from_sv = graph_.get_node_sem_vec(node_id);
                if (!from_sv) { continue; }

                std::vector<std::pair<size_t, std::pair<SemVec*, float>>> neighbors;
                graph_.get_neighbors_indices(node_id, neighbors);

                size_t edge_count = 0;
                for (auto& [to_id, ep] : neighbors) {
                    if (edge_count >= cfg_.max_hop_breadth) { break; }

                    SemVec* to_sv = graph_.get_node_sem_vec(to_id);
                    if (!to_sv) { continue; }

                    result.push_back(
                        semvec_to_vec(*from_sv),
                        semvec_to_vec(*ep.first),
                        semvec_to_vec(*to_sv),
                        cfg_.fresh_prune_penalty
                    );

                    if (visited.find(to_id) == visited.end()) {
                        visited.insert(to_id);
                        next_frontier.push_back(to_id);
                    }
                    ++edge_count;
                }
            }

            frontier = std::move(next_frontier);
        }

        return result;
    }
};

} // namespace llm::memory
