#pragma once

#include "long_term/graph.hpp"
#include "long_term/spatial_map.hpp"
#include "long_term/sem_vec.hpp"

class MemoryModule {
    Graph graph;
    SpatialMap spatial_map;

public:
    MemoryModule() = default;

    void load_from_file(const std::string& filename) { graph.load_from_file(filename, spatial_map); }

};