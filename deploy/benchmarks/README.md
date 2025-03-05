# Benchmarks Utils

## Serving the Model 

To launch the serving model, use the following command:

```shell
MODEL=cognitivecomputations/DeepSeek-V3-AWQ \
    TENSOR_PARALLEL_SIZE=8 \
    PIPELINE_PARALLEL_SIZE=1 \
    MAX_MODEL_LEN=8000 \
    MAX_NUM_SEQS=128 \
    docker compose -f docker-compose.yaml up vllm-server
```

## Run benchmarks with Docker

**Build the Bechmark Image**

First, build the Docker image for benchmarking:

```shell
docker build -t vllm-benchmark -f Dockerfile .
```

**Run the Benchmark Container**

Next, run the container with the specified parameters:

```shell
docker run --network host \
    -v ./benchmark-results:/app/home/benchmark-results \
    -v ./run-all-benchmarks.sh:/app/home/run-all-benchmarks.sh \
    vllm-benchmark -m cognitivecomputations/DeepSeek-V3-AWQ \
    -d sonnet --request-rate 1,1,20 -p 100 -i 1024 -o 256
```

## Running Benchmarks on a Local Machine

Follow these steps to run the benchmarks locally:

1. Install Python using asdf (recommended):

    ```shell
    # Install python, asdf recommend
    asdf plugin add python
    asdf install python 
    ```

2. Create a virtual environment and install the required packages:

    ```shell
    python -m venv .venv 
    source .venv/bin/activate 
    pip install -r requirements.txt 
    ```

3. Run the benchmark script:

    ```shell
    ./run-all-benchmarks.sh -m cognitivecomputations/DeepSeek-V3-AWQ \
        -d sonnet --request-rate 1,1,20 -p 100 -i 1024 -o 256
    ```