#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

make build

mkdir -p build/benchmarks

TS="$(date +%Y%m%d_%H%M%S)"
OUT="build/benchmarks/training_benchmark_${TS}.md"

{
  echo "# Training Benchmark"
  echo
  echo "date: $(date -Iseconds)"
  echo "commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo
  ./build/tests/looping_retnet_training_benchmark
} | tee "$OUT"

echo "saved benchmark table to $OUT"
