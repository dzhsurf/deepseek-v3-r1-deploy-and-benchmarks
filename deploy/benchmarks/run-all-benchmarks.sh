#!/bin/bash

usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -m, --model MODEL_NAME    Model name (required)"
    echo "  -d, --dataset DATASET     Dataset type (sonnet|all) (default: all)"
    echo "  -p, --prompts PROMPTS     Comma-separated list of prompt counts (default: 100,200)"
    echo "  -i, --input-lens LENS     Comma-separated list of input lengths (default: 512,1024)"
    echo "  -o, --output-lens LENS    Comma-separated list of output lengths (default: 512,1024)"
    echo "  --prefix-len LENGTH    Prefix length for sonnet dataset (default: 200)"
    echo "  --result-dir DIR        Directory to save benchmark results (default: ./benchmark-results)"
    echo "  -r, --request-rate RATES    Initial QPS,Step,Peak QPS (default: 1,1,10)"
    echo "                              Or single Peak QPS value (e.g., 20 means 1,1,20)"
    
    echo
    echo "Example:"
    echo "  run sonnet benchmarks with control prompts and input/output lengths, with qps range (1,1,10):"
    echo "  $0 -m Infermatic/Llama-3.3-70B-Instruct-FP8-Dynamic -d sonnet --request-rate 1,1,6 -p 50 -i 512 -o 128"
    echo " run sonnet benchmarks with more configurations:"
    echo "  $0 -m Infermatic/Llama-3.3-70B-Instruct-FP8-Dynamic -d sonnet --request-rate 1,1,6 -p 50 -i 512,1024,2048 -o 128,256,512"
    echo ""
    exit 1
}

# Default values
MODEL=""
DATASET="all"
PROMPTS_STR="100"
INPUT_LENS_STR="1024"
OUTPUT_LENS_STR="512"
SLEEP_TIME=5
BACKEND=vllm
BASE_URL=http://127.0.0.1:8000
SCRIPT_DIR=./
REQUEST_RATE_STR="1,1,10"
PREFIX_LEN_STR="200"
RESULT_DIR="./benchmark-results"


# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        -p|--prompts)
            PROMPTS_STR="$2"
            shift 2
            ;;
        -i|--input-lens)
            INPUT_LENS_STR="$2"
            shift 2
            ;;
        -o|--output-lens)
            OUTPUT_LENS_STR="$2"
            shift 2
            ;;
        -s|--sleep)
            SLEEP_TIME="$2"
            shift 2
            ;;
        -r|--request-rate)
            REQUEST_RATE_STR="$2"
            shift 2
            ;;
        --prefix-len)
            PREFIX_LEN_STR="$2"
            shift 2
            ;;
        --result-dir)
            RESULT_DIR="$2"
            shift 2
            ;;
        *)
            usage
            ;;
    esac
done

# Validate required parameters
if [ -z "$MODEL" ]; then
    echo "Error: Model name is required"
    usage
fi

# Convert comma-separated strings to arrays
IFS=',' read -ra NUM_PROMPTS <<< "$PROMPTS_STR"
IFS=',' read -ra INPUT_LENS <<< "$INPUT_LENS_STR"
IFS=',' read -ra OUTPUT_LENS <<< "$OUTPUT_LENS_STR"

# Parse request rate parameters
IFS=',' read -ra REQUEST_RATE <<< "$REQUEST_RATE_STR"
if [ ${#REQUEST_RATE[@]} -eq 1 ]; then
    # If only one value provided, use it as peak QPS with step=1
    INIT_QPS=1
    STEP_QPS=1
    PEAK_QPS=${REQUEST_RATE[0]}
else
    # Use provided values for init, step, and peak QPS
    INIT_QPS=${REQUEST_RATE[0]}
    STEP_QPS=${REQUEST_RATE[1]}
    PEAK_QPS=${REQUEST_RATE[2]}
fi
INIT_QPS=${REQUEST_RATE[0]}
STEP_QPS=${REQUEST_RATE[1]}
PEAK_QPS=${REQUEST_RATE[2]}

# Print configuration
echo "Running benchmarks with:"
echo "Model: ${MODEL}"
echo "Dataset: ${DATASET}"
echo "Prompts: ${NUM_PROMPTS[*]}"
echo "Input lengths: ${INPUT_LENS[*]}"
echo "Output lengths: ${OUTPUT_LENS[*]}"
echo "Prefix length: ${PREFIX_LEN_STR}"
echo "Sleep time: ${SLEEP_TIME}s"
echo "Request rate (QPS): ${INIT_QPS} to ${PEAK_QPS} (step: ${STEP_QPS})"


mkdir -p "${RESULT_DIR}"
echo "Result directory: ${RESULT_DIR}"

# Convert DATASET to lowercase for case-insensitive comparison
DATASET_LOWER=$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')

function run_benchmark_sonnet() {
    local prompts=$1
    local input_len=$2
    local output_len=$3
    local qps=$4
    local model_name=${MODEL//\//-}

    echo "Running Sonnet benchmark with $prompts prompts, input_len=$input_len, output_len=$output_len, QPS=$qps"

    python ${SCRIPT_DIR}/benchmark_serving.py \
        --backend $BACKEND \
        --base-url $BASE_URL \
        --model $MODEL \
        --dataset-name sonnet \
        --dataset-path ${SCRIPT_DIR}/sample/sonnet.txt \
        --num-prompts ${prompts} \
        --sonnet-input-len $input_len \
        --sonnet-output-len $output_len \
        --sonnet-prefix-len $PREFIX_LEN_STR \
        --request-rate $qps \
        --save-result \
        --burstiness 0.99 \
        --result-dir "${RESULT_DIR}" \
        --result-filename ${model_name}-sonnet-c${prompts}-i${input_len}-o${output_len}-q${qps}.json

    echo "Sleeping for ${SLEEP_TIME} seconds..."
    sleep ${SLEEP_TIME}
}

echo "Running ${DATASET_LOWER} benchmarks..."


if [[ "$DATASET_LOWER" == "all" || "$DATASET_LOWER" == "sonnet" ]]; then
    for input in "${INPUT_LENS[@]}"; do
        for output in "${OUTPUT_LENS[@]}"; do
            for prompts in "${NUM_PROMPTS[@]}"; do
                for ((qps = INIT_QPS; qps <= PEAK_QPS; qps += STEP_QPS)); do
                    run_benchmark_sonnet $prompts $input $output $qps
                done
            done
        done
    done
fi
