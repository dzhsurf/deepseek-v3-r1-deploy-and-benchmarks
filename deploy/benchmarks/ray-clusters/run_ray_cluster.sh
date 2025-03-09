#!/bin/bash

EXPECTED_ACTIVE_NODES=2

usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --head                  Run as Ray head node."
    echo "  --worker                Run as Ray worker node."
    echo "  --port PORT             Port for Ray head (required for head mode)."
    echo "  --address ADDRESS       Ray head address (required for worker mode)."
    echo "  --expected-active N     Expected number of active nodes (only for head mode, default is 2)."
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --head)
            RAY_MODE="head"
            shift 1
            ;;
        --worker)
            RAY_MODE="worker"
            shift 1
            ;;
        --port)
            if [[ -n "$2" ]]; then
                RAYHEAD_PORT="$2"
                shift 2
            else
                usage
            fi
            ;;
        --port=*)
            RAYHEAD_PORT="${1#*=}"
            shift 1
            ;;
        --address)
            if [[ -n "$2" ]]; then
                RAYHEAD_ADDRESS="$2"
                shift 2
            else
                usage
            fi
            ;;
        --address=*)
            RAYHEAD_ADDRESS="${1#*=}"
            shift 1
            ;;
        --expected-active)
            if [[ -n "$2" ]]; then
                EXPECTED_ACTIVE_NODES="$2"
                shift 2
            else
                usage
            fi
            ;;
        --expected-active=*)
            EXPECTED_ACTIVE_NODES="${1#*=}"
            shift 1
            ;;
        *)
            usage
            ;;
    esac
done

# Check input parameters
if [[ "$RAY_MODE" != "head" && "$RAY_MODE" != "worker" ]]; then
    usage
fi

if [[ "$RAY_MODE" == "head" && -z "$RAYHEAD_PORT" ]]; then
    echo "Error: head mode must specify --port"
    usage
fi

if [[ "$RAY_MODE" == "worker" && -z "$RAYHEAD_ADDRESS" ]]; then
    echo "Error: worker mode must specify --address"
    usage
fi

# launch vllm
run_vllm() {
    echo "Starting vllm server..."
    # vllm serve $MODEL # ...
    vllm serve ${MODEL:-Infermatic/Llama-3.3-70B-Instruct-FP8-Dynamic} \
        --host 0.0.0.0 \
        --port 8000 \
        --distributed-executor-backend ray \
        --tensor-parallel-size ${TENSOR_PARALLEL_SIZE:-2} \
        --pipeline-parallel-size ${PIPELINE_PARALLEL_SIZE:-1} \
        --max-model-len ${MAX_MODEL_LEN:-8000} \
        --max-num-seqs ${NUM_SEQS:-128} \
        --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS:-32768}  \
        --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION:-0.95} \
        --disable-log-requests \
        --trust-remote-code \
        --enable-prefix-caching \
        --enable-chunked-prefill
}

# loop check ray cluster statue until worker ready
wait_until_worker_ready() {
    expected_count=$1
    echo "Waiting until active node count >= ${expected_count}..."
    while true; do
        active_count=$(ray status 2>&1 | awk '/Active:/,/^Pending:/{if ($1 ~ /^[0-9]+$/) sum += $1} END {print (sum == "" ? 0 : sum)}')
        echo "Current active nodes: ${active_count}"
        if (( active_count >= expected_count )); then
            echo "Expected active node count ${expected_count} reached."
            break
        fi
        sleep 5
    done
}

run_rayhead() {
    port="$1"
    echo "Starting Ray head node on port ${port}..."
    ray start --head --port="${port}"
    echo "Ray head started."

    echo "Waiting for worker nodes to join..."
    wait_until_worker_ready "${EXPECTED_ACTIVE_NODES}"

    echo "All worker nodes are ready. Starting vllm server..."
    run_vllm
}

run_rayworker() {
    head_address="$1"
    echo "Starting Ray worker node connecting to head at ${head_address}..."
    ray start --block --address="${head_address}"
}

if [[ "$RAY_MODE" == "head" ]]; then
    run_rayhead "$RAYHEAD_PORT"
elif [[ "$RAY_MODE" == "worker" ]]; then
    run_rayworker "$RAYHEAD_ADDRESS"
fi