#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <utility>
#include <unordered_map>

#include <long_term/sem_vec.hpp>
#include <long_term/spatial_map.hpp>

struct EdgeEntry {
    size_t to_index;
    SemVec sem_vec;
    float weight;

    EdgeEntry(size_t to, SemVec sv, float w)
      : to_index(to), sem_vec(std::move(sv)), weight(w) {}

    EdgeEntry(EdgeEntry&&) = default;
    EdgeEntry& operator=(EdgeEntry&&) = default;
    EdgeEntry(const EdgeEntry&) = delete;
    EdgeEntry& operator=(const EdgeEntry&) = delete;
};

class Graph {
    std::unordered_map<size_t, SemVec> nodes;

    // edges are directional, e.g. this is a digraph
    // node_index -> [(to_index, edge_sem_vec, edge_weight)]
    std::unordered_map<size_t, std::vector<EdgeEntry>> edges;

    void load_impl(const std::string& filename, SpatialMap* spatial_map);

public:
    Graph() = default;
    ~Graph() = default;

    void load_from_file(const std::string& filename);
    void load_from_file(const std::string& filename, SpatialMap& spatial_map);
    void save_to_file(const std::string& filename) const;

    void add_node(size_t node_index, SemVec sem_vec) {
        nodes.emplace(node_index, std::move(sem_vec));
    }
    void add_edge(size_t from_index, size_t to_index, SemVec edge_sem_vec, float weight) {
        edges[from_index].emplace_back(to_index, std::move(edge_sem_vec), weight);
    }
    void get_neighbors_indices(size_t node_index, std::vector<std::pair<size_t, std::pair<SemVec*, float>>>& neighbors) {
        neighbors.clear();
        auto it = edges.find(node_index);
        if (it == edges.end()) {
            return;
        }

        for (auto& entry : it->second) {
            neighbors.emplace_back(entry.to_index, std::make_pair(&entry.sem_vec, entry.weight));
        }
    }
    void get_neighbors(size_t node_index, std::vector<std::pair<SemVec*, std::pair<SemVec*, float>>>& neighbors) {
        std::vector<std::pair<size_t, std::pair<SemVec*, float>>> vec;
        get_neighbors_indices(node_index, vec);
        neighbors.clear();
        for (const auto& [neighbor_index, pair] : vec) {
            SemVec* neighbor_sem_vec = get_node_sem_vec(neighbor_index);
            if (neighbor_sem_vec != nullptr) {
                neighbors.emplace_back(neighbor_sem_vec, pair);
            }
        }
    }
    SemVec* get_node_sem_vec(size_t node_index) {
        auto it = nodes.find(node_index);
        if (it == nodes.end()) {
            return nullptr;
        }
        return &it->second;
    }
};