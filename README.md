# Deploy DeepSeek-V3/R1 671B on 8xH100 and Throughput Benchmarks

>  Note: This repository is currently in an editing phase.

This document provides a comprehensive guide for deploying the DeepSeek V3/R1 671B model with serving on a single machine with 8xH100 GPUs. It also contains detailed performance throughput benchmarks under various parameter configurations. This document is intended to help users understand the deployment process as well as the service capabilities provided within a hardware resource environment.



## Table of Contents

1. [Deployment Environment and System Configuration](#deployment-environment-and-system-configuration)
2. [Environment Setup and Benchmarking](#environment-setup-and-benchmarking)
3. [Benchmarking LLM Throughput and Concurrency](#benchmarking-llm-throughput-and-concurrency)
4. [Comparative Analysis](#comparative-analysis)
5. [Multi-Node Deployment and Testing under Kubernetes](#multi-node-deployment-and-testing-under-kubernetes)
   - [Deployment Configuration](#deployment-configuration)
   - [Parameter Configuration](#parameter-configuration)
   - [Performance Test Results](#performance-test-results)
   - [Comparative Analysis](#comparative-analysis)

---



## Deployment Environment and System Configuration

The deployment has been tested on a high-performance machine configured with the following hardware components:

- 8× NVIDIA H100 SXM5 GPUs
- 104 Intel Xeon Platinum 8470 CPUs
- 1024GB DDR5 RAM
- 100 Gbps Ethernet connectivity

The software stack includes:

- vLLM version 0.7.3
- Docker with NVIDIA-Container-Runtime

Additionally, the following models are utilized:

- cognitivecomputations/DeepSeek-V3-AWQ
- cognitivecomputations/DeepSeek-R1-AWQ

This environment ensures that the DeepSeek V3/R1 671B model is deployed efficiently, enabling extensive throughput benchmarking and serving capabilities on a single machine setup.



**Why Use the 4-Bit AWQ Quantized Model?**

Deploying the full-scale 671B DeepSeek V3 model in FP8 would require an enormous amount of GPU memory (FP16 can reach around 1.4TB in VRAM). For example, the model’s 671B main parameters plus an additional 14B MTP parameters already demand around 685GB of VRAM when stored in FP8. With 8xH100 GPUs offering a total of 640GB VRAM, this setup falls short of the requirements.

Under AWQ, the model's 671B parameters are quantized from 8-bit to 4-bit, reducing weight storage to approximately 335GB. Additionally, there are about 37B activated parameters stored in FP16, taking up roughly 37GB. Combined, this consumes around 400GB of VRAM, allowing the model to run on an 8×H100 setup with 640GB VRAM and leaving ample space for cache during serving deployment.



## Environment Setup and Benchmarking

### Prerequisites  
Make sure you have the following before proceeding:  
- Ubuntu 22.04  
- NVIDIA drivers installed

### Base Environment Installation  
Install Docker and NVIDIA Container Runtime. You can follow the official documentation.

```shell
# Docker 
https://docs.docker.com/engine/install/ubuntu/
# Nvidia Container Toolkit
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
```

Or use the provided setup script:

```shell
./deploy/scripts/env_setup.sh
```

### Setting Up the vLLM Environment  

Navigate to the `deploy/benchmarks` directory. Create a `.env` file and set your Hugging Face token along with any other necessary environment variables:

```shell
HF_TOKEN=...
```

### Running the vLLM Server

Execute the following command to start the vllm server. Customize the parameters (e.g., `MODEL`, `TENSOR_PARALLEL_SIZE`, etc.) as needed:

```shell
MODEL=cognitivecomputations/DeepSeek-V3-AWQ \
    TENSOR_PARALLEL_SIZE=8 \
    PIPELINE_PARALLEL_SIZE=1 \
    MAX_MODEL_LEN=8000 \
    MAX_NUM_SEQS=128 \
    docker compose -f docker-compose.yaml up vllm-server
```

### Running the Benchmark

To execute the benchmark, run the appropriate commands as outlined in our instructions. For more details about how to run benchmarks and configure the environment, please refer to [Benchmarks Utils](./deploy/benchmarks/README.md).



## Benchmarking LLM Throughput and Concurrency

This section evaluates the service's performance by measuring throughput under different concurrency limitations on the server.

vLLM config

```shell
MODEL=cognitivecomputations/DeepSeek-V3-AWQ
TENSOR_PARALLEL_SIZE=8
PIPELINE_PARALLEL_SIZE=1
MAX_MODEL_LEN=8000
MAX_NUM_SEQS=128
```

Benchmarks config

```shell
# input token is 1024, output token is 256
-i 1024 -o 256 
```

**Concurrency 1-5 Throughput**

![](./docs/images/throughput-max-concurrency-1-5.jpg)

When Max Concurrency is set to 1, it can be observed that under a single-user request, the token output throughput is around 33 tokens per second. For a model of the 671B scale, this performance is not too bad. Naturally, as the level of concurrency increases, the total throughput rises and the device utilization improves.



**Concurrency 1-200 Throughput**

![](./docs/images/throughput-max-concurrency.jpg)

In tests covering scenarios with concurrency levels from 1 to 200, the data shows that when concurrency reaches about 100, the throughput has peaked—yielding a total token throughput of roughly 3000 and an output token throughput of about 620. Some might question whether this ceiling is due to the limitations imposed by vLLM’s `MAX_NUM_SEQS` setting, which might prevent running more tasks concurrently. In the tests for concurrency throughput from 1 to 200, the tests for 1–120 concurrency were conducted with `MAX_NUM_SEQS` set to 128, while those for 100–200 concurrency were run with `MAX_NUM_SEQS` set to 200. Comparing the collected data, no improvement was observed. Furthermore, monitoring in vLLM shows that although a larger `MAX_NUM_SEQS` was set, not all tasks marked as running are actually executed—instead, once the number of running tasks exceeds around 100, they begin to enter a pending state.



**Does a higher level of concurrency, implying greater throughput, necessarily mean better performance?** 

Not necessarily. The growth in token output throughput is in fact much lower than the increase in input throughput; the boost in input throughput is largely due to techniques such as `prefix caching` and `chunked prefill`. However, there is little improvement in the output efficiency of the model's decoder stage. Moreover, more concurrency only serves to distribute the response latency evenly across each request. Below is a comparison of ITL (Inter-token Latency) under different concurrency levels.

**Concurrency 1-100 ITL**

![](./docs/images/itl-max-concurrency.jpg)

For example, if business requirements stipulate that the inter-token response latency must not exceed 50ms, the chart shows that when concurrency reaches 15, the latency is already close to this limit. With any higher level of concurrency, the latency will exceed the requirement.

However, one observation is that when max concurrency is set to 15, the token output throughput is about 250. Dividing 250 by 15 gives roughly 17, meaning that on average, a user receives about 17 tokens per second. Dividing 1000 by 17 yields approximately 58, which in practice slightly exceeds the 50ms requirement, because these statistics only reflect the median ITL. One can also compare the Mean ITL and P99 ITL values. It becomes evident that imposing a P99 requirement of 50ms per request is extremely stringent for such ultra-large parameter models.

![](./docs/images/median-p99-itl-max-concurrency.jpg)

Of course, if the response time requirement is relaxed to 200ms, even a single node can still provide reasonable levels of concurrency and throughput.



Regarding TTFT (Total Time to First Token) and TTLT (Total Time to Last Token), as the token output throughput is limited, the time until the first token appears is also notably affected by the level of concurrency. When the concurrency reaches 8, the first token does not appear until after 1 second.

**Concurrency 1-24 TTFT/TTLT**

![](./docs/images/ttft-ttlt-max-concurrency-1-24.jpg)

**Concurrency 1-100 TTFT/TTLT**

![](./docs/images/ttft-ttlt-max-concurrency.jpg)



**How does the model's processing performance behave under different QPS?** 

**QPS 1-20: Server vs Client Request Rate** 

![](./docs/images/server-client-request-rate.jpg)

It can be seen that when the server's processing speed reaches 2.75 QPS.