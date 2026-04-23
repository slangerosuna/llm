#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>

#include <architecture/looping_retnet.hpp>

int main() {
    using llm::arch::LoopConfig;
    using llm::arch::LoopingRetNet;

    LoopConfig cfg;
    cfg.model_dim = 32;
    cfg.qk_dim = 16;
    cfg.v_dim = 16;
    cfg.rel_dim = 8;
    cfg.max_steps = 4;

    const std::string path = "/tmp/looping_retnet_roundtrip.bin";

    LoopingRetNet model(cfg, 123);
    const uint64_t before = model.parameter_checksum();
    model.save_to_file(path);

    LoopingRetNet loaded = LoopingRetNet::load_from_file(path);
    const uint64_t after = loaded.parameter_checksum();

    std::remove(path.c_str());

    std::cout << "before_checksum=" << before << " after_checksum=" << after << "\n";
    if (before != after) {
        throw std::runtime_error("LoopingRetNet serialization round-trip mismatch");
    }

    return 0;
}
