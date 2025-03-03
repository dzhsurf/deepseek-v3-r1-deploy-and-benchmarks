# Deploy DeepSeek-V3/R1 671B on 8xH100 and Throughput Benchmarks

>  Note: This repository is currently in an editing phase.

This document provides a comprehensive guide for deploying the DeepSeek V3/R1 671B model with serving on a single machine with 8xH100 GPUs. It also contains detailed performance throughput benchmarks under various parameter configurations. This document is intended to help users understand the deployment process as well as the service capabilities provided within a hardware resource environment.



## Table of Contents

2. [Deployment Environment and System Configuration](#deployment-environment-and-system-configuration)
3. [Installation and Deployment](#installation-and-deployment)
   - [Prerequisites](#prerequisites)
   - [Installation Steps](#installation-steps)
   - [Deployment Configuration](#deployment-configuration)
4. [Performance Throughput Benchmarking](#performance-throughput-benchmarking)
   - [Benchmarking Methodology](#benchmarking-methodology)
   - [Parameter Configurations and Evaluation](#parameter-configurations-and-evaluation)
   - [Performance Results](#performance-results)
5. [Comparative Analysis](#comparative-analysis)
   - [Different Parameter Settings](#different-parameter-settings)
   - [Observations and Recommendations](#observations-and-recommendations)
6. [Multi-Node Deployment and Testing under Kubernetes](#multi-node-deployment-and-testing-under-kubernetes)
   - [Deployment Configuration](#deployment-configuration)
   - [Parameter Configuration](#parameter-configuration)
   - [Performance Test Results](#performance-test-results)
   - [Comparative Analysis](#comparative-analysis)
6. [License](#license)

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



## Installation and Deployment



## Performance Throughput Benchmarking



## Comparative Analysis



## License