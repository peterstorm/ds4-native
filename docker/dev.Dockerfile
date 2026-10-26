# Build/test image: the CUDA 13.4 + gcc 13 + cmake 3.31 image already on the
# desktop (ghcr.io/local-inference-lab/vllm:karmic-kraken-beta, id 7b5c335cc647) plus the
# few dev packages it lacks. Reusing it avoids pulling another CUDA base.
FROM ghcr.io/local-inference-lab/vllm:karmic-kraken-beta
RUN apt-get update \
 && apt-get install -y --no-install-recommends doctest-dev liburing-dev \
 && rm -rf /var/lib/apt/lists/*
ENTRYPOINT []
