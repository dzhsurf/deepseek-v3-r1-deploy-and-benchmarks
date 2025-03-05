FROM python:3.11

WORKDIR /app/home 

COPY requirements.txt .
RUN pip install -r requirements.txt 

ADD vllm-benchmark-code/backend_request_func.py .
ADD vllm-benchmark-code/benchmark_serving.py .
ADD vllm-benchmark-code/benchmark_utils.py .
ADD run-all-benchmarks.sh .
ADD sample/ ./sample/

SHELL ["/bin/bash", "-c"]
ENTRYPOINT ["./run-all-benchmarks.sh"]