#include <long_term/graph.hpp>
#include <long_term/spatial_map.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

static constexpr uint64_t GRAPH_FILE_MAGIC = 0x4C4C4D4752415048ULL; // "LLMGRAPH"

// ---------------------------------------------------------------------------
// Helpers: typed binary read/write that throw on I/O failure
// ---------------------------------------------------------------------------
static inline void checked_write(std::FILE* f, const void* buf, size_t bytes) {
    if (std::fwrite(buf, 1, bytes, f) != bytes) {
        throw std::runtime_error("Graph file write error");
    }
}

static inline void checked_read(std::FILE* f, void* buf, size_t bytes) {
    if (std::fread(buf, 1, bytes, f) != bytes) {
        throw std::runtime_error("Graph file read error or unexpected EOF");
    }
}

template<typename T>
static inline void write_pod(std::FILE* f, T value) {
    checked_write(f, &value, sizeof(T));
}

template<typename T>
static inline T read_pod(std::FILE* f) {
    T value;
    checked_read(f, &value, sizeof(T));
    return value;
}

// ---------------------------------------------------------------------------
// File specification (little-endian, packed):
//
// Header (32 bytes):
//   [0-7]   uint64_t magic
//   [8-11]  uint32_t node semvec dimensions
//  [12-15]  uint32_t edge semvec dimensions
//  [16-23]  uint64_t number of nodes
//  [24-31]  uint64_t number of edges
//
// Node entry (8 + node_dims*2 bytes each):
//   [0-7]   uint64_t node index
//   [8-..]  float16_t[node_dims]
//
// Edge entry (20 + edge_dims*2 bytes each):
//   [0-7]   uint64_t from node index
//   [8-15]  uint64_t to node index
//  [16-19]  float weight
//  [20-..]  float16_t[edge_dims]
// ---------------------------------------------------------------------------

void Graph::save_to_file(const std::string& filename) const {
    // Derive consistent dims from the first node / first edge.
    uint32_t node_dims = 0;
    uint32_t edge_dims = 0;
    if (!nodes.empty()) {
        node_dims = static_cast<uint32_t>(nodes.begin()->second.dimensions);
    }
    for (const auto& [from, vec] : edges) {
        if (!vec.empty()) {
            edge_dims = static_cast<uint32_t>(vec.front().sem_vec.dimensions);
            break;
        }
    }

    // Count total edges across all adjacency lists
    uint64_t total_edges = 0;
    for (const auto& [from, vec] : edges) {
        total_edges += vec.size();
    }

    std::FILE* f = std::fopen(filename.c_str(), "wb");
    if (f == nullptr) {
        throw std::runtime_error("Cannot open '" + filename + "' for writing");
    }

    try {
        // Header
        write_pod(f, GRAPH_FILE_MAGIC);
        write_pod(f, node_dims);
        write_pod(f, edge_dims);
        write_pod(f, static_cast<uint64_t>(nodes.size()));
        write_pod(f, total_edges);

        // Node entries
        for (const auto& [index, sem_vec] : nodes) {
            write_pod(f, static_cast<uint64_t>(index));
            checked_write(f, sem_vec.data,
                          sem_vec.dimensions * sizeof(std::float16_t));
        }

        // Edge entries
        for (const auto& [from, vec] : edges) {
            for (const auto& entry : vec) {
                write_pod(f, static_cast<uint64_t>(from));
                write_pod(f, static_cast<uint64_t>(entry.to_index));
                write_pod(f, entry.weight);
                checked_write(f, entry.sem_vec.data,
                              entry.sem_vec.dimensions * sizeof(std::float16_t));
            }
        }
    } catch (...) {
        std::fclose(f);
        throw;
    }

    std::fclose(f);
}

void Graph::load_impl(const std::string& filename, SpatialMap* spatial_map) {
    std::FILE* f = std::fopen(filename.c_str(), "rb");
    if (f == nullptr) {
        throw std::runtime_error("Cannot open '" + filename + "' for reading");
    }

    try {
        const uint64_t magic = read_pod<uint64_t>(f);
        if (magic != GRAPH_FILE_MAGIC) {
            throw std::runtime_error("Invalid graph file magic");
        }
        const uint32_t node_dims = read_pod<uint32_t>(f);
        const uint32_t edge_dims = read_pod<uint32_t>(f);
        const uint64_t num_nodes = read_pod<uint64_t>(f);
        const uint64_t num_edges = read_pod<uint64_t>(f);

        nodes.clear();
        edges.clear();
        nodes.reserve(static_cast<size_t>(num_nodes));

        for (uint64_t ni = 0; ni < num_nodes; ++ni) {
            const uint64_t index = read_pod<uint64_t>(f);
            SemVec sv(node_dims);
            checked_read(f, sv.data, node_dims * sizeof(std::float16_t));
            auto [it, inserted] = nodes.emplace(static_cast<size_t>(index), std::move(sv));
            if (spatial_map != nullptr) {
                spatial_map->insert(it->first, &it->second);
            }
        }

        for (uint64_t ei = 0; ei < num_edges; ++ei) {
            const uint64_t from   = read_pod<uint64_t>(f);
            const uint64_t to     = read_pod<uint64_t>(f);
            const float    weight = read_pod<float>(f);
            SemVec sv(edge_dims);
            checked_read(f, sv.data, edge_dims * sizeof(std::float16_t));
            edges[static_cast<size_t>(from)].emplace_back(
                static_cast<size_t>(to), std::move(sv), weight);
        }
    } catch (...) {
        std::fclose(f);
        throw;
    }

    std::fclose(f);
}

void Graph::load_from_file(const std::string& filename) {
    load_impl(filename, nullptr);
}

void Graph::load_from_file(const std::string& filename, SpatialMap& spatial_map) {
    load_impl(filename, &spatial_map);
}
