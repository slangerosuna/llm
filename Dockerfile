FROM intel/oneapi-basekit:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
  libboost-all-dev \
  cmake \
  make && \
  rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

EXPOSE 8080

CMD ["bash", "-lc", "source /opt/intel/oneapi/setvars.sh --force >/dev/null && cmake -S . -B build-container -DCMAKE_CXX_COMPILER=icpx && cmake --build build-container -j\"$(nproc)\" && exec ./build-container/llm_sycl"]
