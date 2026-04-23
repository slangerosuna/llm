# Local developer workflow for strict SYCL + Boost CMake project.

SHELL := /bin/bash

SYCL_CXX ?= icpx
BUILD_DIR ?= build
BUILD_TYPE ?= Debug
JOBS ?= $(shell nproc)

# Load local runtime env if present (e.g., SERVER_HOST / SERVER_PORT)
-include .env
export

.PHONY: build test run configure clean benchmark

configure:
	@command -v $(SYCL_CXX) >/dev/null || (echo "Compiler '$(SYCL_CXX)' not found on PATH" && exit 1)
	cmake --fresh -S . -B $(BUILD_DIR) \
		-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
		-DCMAKE_CXX_COMPILER=$(SYCL_CXX)

build: configure
	cmake --build $(BUILD_DIR) -j$(JOBS)

test: build
	ctest --test-dir $(BUILD_DIR) --output-on-failure

run: build
	./$(BUILD_DIR)/llm_sycl

benchmark: build
	./scripts/run_training_benchmarks.sh

clean:
	rm -rf $(BUILD_DIR)
