FROM intel/oneapi-basekit:latest

WORKDIR /app

# Copy source tree into the image.
COPY . /app

# Configure and build with oneAPI's SYCL toolchain.
RUN /bin/bash -lc "source /opt/intel/oneapi/setvars.sh && \
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=icpx && \
    cmake --build build -j$(nproc)"

# Keep stdin open for interactive chat.
ENTRYPOINT ["/bin/bash", "-lc", "source /opt/intel/oneapi/setvars.sh && exec ./build/llm_sycl chat -i final_model.bin -tok final_tokenizer.csv"]
