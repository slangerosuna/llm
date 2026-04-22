#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <utility>

#include <long_term/sem_vec.hpp>

class SpatialMap {
    // node_index, sem_vec
    std::vector<std::vector<std::pair<size_t, SemVec*>>> buckets;

    uint8_t splits_per_dimension;
        size_t bucket_size;
        size_t entry_count;

        void rehash(size_t new_bucket_count);
        bool should_grow(size_t target_bucket_size) const;

public:
    SpatialMap(size_t num_buckets, uint8_t splits_per_dimension)
            : buckets(std::max<size_t>(num_buckets, 1)),
                bucket_size(8),
                entry_count(0),
        splits_per_dimension(splits_per_dimension) {}

    SpatialMap() : SpatialMap(1024, 4) {}

    void insert(size_t node_index, SemVec *sem_vec);
    void get_nearest_neighbors(SemVec *query_vec, size_t k, std::vector<std::pair<size_t, SemVec*>>& neighbors);
};
