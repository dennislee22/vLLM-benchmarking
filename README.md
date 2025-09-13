# vLLM Benchmarking

This article details the benchmarking results for the vLLM inference engine using the `Qwen2-7B-Instruct`, `Qwen2.5-32B`, `Qwen2-72B` and `gpt-oss-120B` models, tested on a system equipped with NVIDIA A100 80GB GPU. These results highlight vLLM’s ability to efficiently utilize GPU resources using PagedAttention and continuous batching techniques. A key focus of the benchmarking is to evaluate how generation throughput performance correlates with the configured KV cache memory capacity. To explore this, the `gpu-memory-utilization` setting was varied to determine whether the relationship between memory allocation and throughput is linear. The tests reveal how performance dynamically shifts between being compute-bound (limited by GPU processing power) and memory-bound (constrained by VRAM/GRAM capacity).

<img width="333" height="260" alt="image" src="https://github.com/user-attachments/assets/71cd73fc-e574-4d5d-9468-19b4a9b6c754" />

My GPU Calculator:

<img width="827" height="366" alt="image" src="https://github.com/user-attachments/assets/cbc3422a-6eb7-4623-9cb2-5f877c35442d" />

## <a name="toc_0"></a>Table of Contents
[//]: # (TOC)
[Performance Criteria](#toc_0)<br>
[Platform Requirement](#toc_1)<br>
[Procedure](#toc_2)<br>
[✍️ Test 1: 2 GPU pods host 7B model in 2 different nodes (Available KV cache memory in each GPU: 11.03 GiB)](#toc_3)<br>
[✍️ Test 2: 2 GPU pods host 7B model in the same node (Available KV cache memory in each GPU: 12.02 GiB)](#toc_4)<br>
[✍️ Test 3: 2 GPU pods host 7B model in the same node (Available KV cache memory in each GPU: 31.83 GiB)](#toc_5)<br>
[✍️ Test 4: 1 GPU pod hosts 7B model (Available KV cache memory: 5.01 GiB)](#toc_6)<br>
[✍️ Test 5: 1 GPU pod hosts 7B model (Available KV cache memory: 24.83 GiB)](#toc_7)<br>
[✍️ Test 6: 2 GPU pods host 32B model in the same node (Available KV cache memory in each GPU: 8.22 GiB)](#toc_8)<br>
[✍️ Test 7: 1 GPU pod hosts 32B model (Available KV cache memory: 9.70 GiB)](#toc_9)<br>
[✍️ Test 8: 2 GPU pods host 72B model in the same node (Available KV cache memory in each GPU: 6.77 GiB)](#toc_10)<br>
[✍️ Test 9: 4 GPU pods host 120B model across different nodes (Available KV cache memory in each GPU: 45.32 GiB)](#toc_11)<br>

## <a name="toc_0"></a>Performance Criteria

📈 Concurrent Prompts:
Increasing concurrent prompts initially leads to a rapid increase in total throughput. The system hits a peak performance at a tipping point. Beyond this point, throughput gradually declines or flat due to the overhead of managing too many requests (compute gridlock). At low concurrency, the GPU is underutilized. Increasing the concurrent prompts/user allows the GPU to process more data in parallel, hiding memory latency and maximizing compute saturation. Once saturated, adding more requests creates scheduling overhead and contention for resources, leading to diminishing returns and eventually a performance drop/flat.

📈 KV Cache Reservation:
KV Cache acts as an "enabler" for concurrency. Increasing the KV Cache allows more prompts to be held in GRAM, which in turn allows the system to reach the optimal point on the concurrency curve. However, allocating more KV Cache than is needed to serve the optimal number of concurrent prompts yields no additional throughput. Throughput is capped by the lesser of the compute limit (how many requests the GPU can process efficiently) and the memory limit (how many requests can fit in the KV Cache). For example, if your cache only fits 50 prompts, your throughput is capped at the 50-user performance level, even if 200 users are sending requests. Once you have enough cache to hold 200+ prompts, the bottleneck shifts entirely to compute, and extra GRAM for the cache provides no benefit.

<img width="400" height="313" alt="image" src="https://github.com/user-attachments/assets/18a290e1-b968-45d9-ad41-6d59e2deb1b9" />

Failing to allocate the minimum required KV cache memory can cause the engine to crash at startup. For example, log below shows a negative available KV cache memory (-0.53 GiB) and an immediate failure to initialize.

```
(EngineCore_0 pid=1321) INFO 09-06 03:10:08 [gpu_worker.py:276] Available KV cache memory: -0.53 GiB
(EngineCore_0 pid=1321) ERROR 09-06 03:10:08 [core.py:700] EngineCore failed to start.
```

📈 Context Length:
A longer context requires proportionally more GRAM for its KV Cache. This means for a fixed total KV Cache size, fewer requests can be processed concurrently, which can lower throughput if it pushes you below the optimal concurrency level. The initial processing of the prompt (prefill) takes longer for a longer context. 

📈 Input vs. Output (Context Length) Ratio:
For a fixed context length, shifting the ratio from prompt-heavy (e.g., 90% input, 10% output) to generation-heavy (e.g., 10% input, 90% output) massively increases the output token throughput by nearly 3x (from ~810 tok/sec to ~2400 tok/sec in the benchmarks). This is due to the difference between the prefill and decoding stages. Prefill (Input) is a highly parallel computation to process the entire input prompt at once. Decoding (Output) is an iterative process that generates one token at a time. It's less computationally intense but is memory-bandwidth bound. The "Output Token Throughput" metric measures the speed of the decoding stage. When a request has a long output with less input, the high fixed cost of the prefill is amortized over many generated tokens. This makes the average time-per-output-token very low, resulting in a very high tok/sec metric. Conversely, a short output with long input gives the prefill cost little time to be amortized, resulting in a lower tok/sec.

📈 Superior Performance with GPUs on the Same Node:
The configuration with 2 GPUs in the same node consistently and significantly outperforms the setup with 2 GPUs across 2 different nodes. This is likely due to faster inter-GPU communication within a single node (e.g., via NVLink or PCIe) compared to the network latency between different nodes.

<img width="500" height="800" alt="image" src="https://github.com/user-attachments/assets/fc7413d7-88fd-49ef-8945-6faf6155f12b" />

## <a name="toc_1"></a>Platform Requirement
✅ Python 3.11/10

✅ Cloudera AI (CAI) / Cloudera Machine Learning (CML) 1.5.x

## <a name="toc_2"></a>Procedure

1. Create a new CAI project with 1G shared memory.

<img width="700" height="188" alt="image" src="https://github.com/user-attachments/assets/41c6fe14-5c1c-48fe-b8eb-cfda20d28e06" />
   
3. Install python libraries.
  ```
  pip install vllm torch transformers ipywidgets gradio ray[default] flashinfer-python
  ```

3. Download the pre-trained LLM into the project of the CAI/CML platform using either `git clone` or `wget`.
Example:
  ```
  git lfs clone https://huggingface.co/Qwen/Qwen2-7B-Instruct
  ```

4. Download the vLLM benchmarking tool.
  ```
  git clone https://github.com/vllm-project/vllm.git
  ```

#### <a name="toc_3"></a>✍️ Test 1: 2 GPU pods host 7B model in 2 different nodes (Available KV cache memory in each GPU: 11.03 GiB)

- This setup uses Ray alongside `cml.workers_v1 module`. Please see this script [run-vllm.py](run-vllm.py).

- Create Application to expose vLLM API endpoint. As vLLM utilizes Ray, this application will also host the Ray HEAD and Ray dashboard with 1 GPU device.
<img width="460" height="730" alt="image" src="https://github.com/user-attachments/assets/d128c611-969d-4e01-9fe4-54a73f9db055" />

- Start the `vllm-api` application and ensure that the model is fully loaded into the GPU before starting the `gradio-app` application. The code will spawn one Ray HEAD pod along with its associated worker pod within seconds.

```
NAME               READY   STATUS    RESTARTS   AGE   IP             NODE        NOMINATED NODE   READINESS GATES
5c4bmyq953zzetaj   5/5     Running   0          28m   10.254.6.219   worker-19   <none>           <none>
ip46y52fijaguyzo   5/5     Running   0          29m   10.254.5.36    worker-20   <none>           <none>
```
- Startup log: [vllm-7B-2gpuA10040GB-0.5GRAM-diff-nodes.log](log/vllm-7B-2gpuA10040GB-0.5GRAM-diff-nodes.log)

<img width="700" height="192" alt="image" src="https://github.com/user-attachments/assets/eda7e695-1ae3-4e2e-a704-c9f74e0bda5b" />

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2-7B-Instruct --dataset-name random \
--host 10.254.5.35 \
--num-prompts 10 \
--random-input-len 1024 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     10        
Benchmark duration (s):                  46.96     
Total input tokens:                      10239     
Total generated tokens:                  10240     
Request throughput (req/s):              0.21      
Output token throughput (tok/s):         218.04    
Total Token throughput (tok/s):          436.07    
---------------Time to First Token----------------
Mean TTFT (ms):                          2111.51   
Median TTFT (ms):                        2244.91   
P99 TTFT (ms):                           3346.14   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          43.77     
Median TPOT (ms):                        43.65     
P99 TPOT (ms):                           45.56     
---------------Inter-token Latency----------------
Mean ITL (ms):                           43.77     
Median ITL (ms):                         42.39     
P99 ITL (ms):                            51.04     
==================================================

, Running: 10 reqs, Waiting: 0 reqs, GPU KV cache usage: 3.5%, Prefix cache hit rate: 53.7%
(APIServer pid=158) INFO 09-02 13:08:10 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 238.7 tokens/s, Running: 10 reqs, Waiting: 0 reqs, GPU KV cache usage: 4.1%, Prefix cache hit rate: 53.7%
(APIServer pid=158) INFO 09-02 13:08:20 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 240.2 tokens/s, Running: 10 reqs, Waiting: 0 reqs, GPU KV cache usage: 4.7%, Prefix cache hit rate: 53.7%
```

<img width="700" height="195" alt="image" src="https://github.com/user-attachments/assets/0e550702-69e6-45d9-affd-a8fe434816f3" />

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2-7B-Instruct --dataset-name random \
--host 10.254.5.35 \
--num-prompts 10 \
--random-input-len 4096 \
--random-output-len 4096

============ Serving Benchmark Result ============
Successful requests:                     10        
Benchmark duration (s):                  197.10    
Total input tokens:                      40952     
Total generated tokens:                  40960     
Request throughput (req/s):              0.05      
Output token throughput (tok/s):         207.81    
Total Token throughput (tok/s):          415.58    
---------------Time to First Token----------------
Mean TTFT (ms):                          5713.39   
Median TTFT (ms):                        5747.03   
P99 TTFT (ms):                           10212.65  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          46.68     
Median TPOT (ms):                        46.68     
P99 TPOT (ms):                           47.79     
---------------Inter-token Latency----------------
Mean ITL (ms):                           46.70     
Median ITL (ms):                         45.44     
P99 ITL (ms):                            53.34     
==================================================

(APIServer pid=158) INFO 09-02 13:14:10 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 218.5 tokens/s, Running: 10 reqs, Waiting: 0 reqs, GPU KV cache usage: 18.5%, Prefix cache hit rate: 39.1%
(APIServer pid=158) INFO 09-02 13:14:20 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 220.2 tokens/s, Running: 10 reqs, Waiting: 0 reqs, GPU KV cache usage: 19.0%, Prefix cache hit rate: 39.1%
(APIServer pid=158) INFO 09-02 13:14:30 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 216.3 tokens/s, Running: 10 reqs, Waiting: 0 reqs, GPU KV cache usage: 19.6%, Prefix cache hit rate: 39.1%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2-7B-Instruct --dataset-name random \
--host 10.254.5.35 \
--num-prompts 10 \
--random-input-len 1024 \
--random-output-len 7168

============ Serving Benchmark Result ============
Successful requests:                     10        
Benchmark duration (s):                  308.35    
Total input tokens:                      10239     
Total generated tokens:                  69058     
Request throughput (req/s):              0.03      
Output token throughput (tok/s):         223.96    
Total Token throughput (tok/s):          257.17    
---------------Time to First Token----------------
Mean TTFT (ms):                          185.12    
Median TTFT (ms):                        192.35    
P99 TTFT (ms):                           196.59    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          42.97     
Median TPOT (ms):                        42.99     
P99 TPOT (ms):                           43.00     
---------------Inter-token Latency----------------
Mean ITL (ms):                           42.98     
Median ITL (ms):                         42.66     
P99 ITL (ms):                            51.08     
==================================================

, Running: 9 reqs, Waiting: 0 reqs, GPU KV cache usage: 16.0%, Prefix cache hit rate: 47.6%
(APIServer pid=158) INFO 09-02 13:24:12 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 206.3 tokens/s, Running: 9 reqs, Waiting: 0 reqs, GPU KV cache usage: 16.5%, Prefix cache hit rate: 47.6%
(APIServer pid=158) INFO 09-02 13:24:22 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 203.4 tokens/s, Running: 9 reqs, Waiting: 0 reqs, GPU KV cache usage: 17.0%, Prefix cache hit rate: 47.6%
(APIServer pid=158) INFO 09-02 13:24:32 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 207.4 tokens/s, Running: 9 reqs, Waiting: 0 reqs, GPU KV cache usage: 17.5%, Prefix cache hit rate: 47.6%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2-7B-Instruct --dataset-name random \
--host 10.254.5.35 \
--num-prompts 50 \
--random-input-len 1024 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     50        
Benchmark duration (s):                  86.87     
Total input tokens:                      50977     
Total generated tokens:                  51119     
Request throughput (req/s):              0.58      
Output token throughput (tok/s):         588.46    
Total Token throughput (tok/s):          1175.28   
---------------Time to First Token----------------
Mean TTFT (ms):                          6954.60   
Median TTFT (ms):                        6680.77   
P99 TTFT (ms):                           14827.09  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          77.62     
Median TPOT (ms):                        77.88     
P99 TPOT (ms):                           83.55     
---------------Inter-token Latency----------------
Mean ITL (ms):                           77.61     
Median ITL (ms):                         70.98     
P99 ITL (ms):                            718.62    
==================================================

(APIServer pid=158) INFO 09-02 13:26:52 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 708.5 tokens/s, Running: 50 reqs, Waiting: 0 reqs, GPU KV cache usage: 20.4%, Prefix cache hit rate: 37.1%
(APIServer pid=158) INFO 09-02 13:27:02 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 732.6 tokens/s, Running: 50 reqs, Waiting: 0 reqs, GPU KV cache usage: 22.1%, Prefix cache hit rate: 37.1%
(APIServer pid=158) INFO 09-02 13:27:12 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 698.0 tokens/s, Running: 49 reqs, Waiting: 0 reqs, GPU KV cache usage: 23.4%, Prefix cache hit rate: 37.1%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2-7B-Instruct --dataset-name random \
--host 10.254.5.35 \
--num-prompts 50 \
--random-input-len 4096 \
--random-output-len 4096

============ Serving Benchmark Result ============
Successful requests:                     50        
Benchmark duration (s):                  355.84    
Total input tokens:                      204464    
Total generated tokens:                  200735    
Request throughput (req/s):              0.14      
Output token throughput (tok/s):         564.12    
Total Token throughput (tok/s):          1138.72   
---------------Time to First Token----------------
Mean TTFT (ms):                          20051.92  
Median TTFT (ms):                        19176.00  
P99 TTFT (ms):                           45678.69  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          84.67     
Median TPOT (ms):                        82.08     
P99 TPOT (ms):                           159.94    
---------------Inter-token Latency----------------
Mean ITL (ms):                           81.71     
Median ITL (ms):                         76.17     
P99 ITL (ms):                            89.35     
==================================================

(APIServer pid=158) INFO 09-02 13:35:23 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 640.5 tokens/s, Running: 49 reqs, Waiting: 0 reqs, GPU KV cache usage: 93.3%, Prefix cache hit rate: 39.5%
(APIServer pid=158) INFO 09-02 13:35:33 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 625.5 tokens/s, Running: 49 reqs, Waiting: 0 reqs, GPU KV cache usage: 94.9%, Prefix cache hit rate: 39.5%
(APIServer pid=158) INFO 09-02 13:35:43 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 631.0 tokens/s, Running: 49 reqs, Waiting: 0 reqs, GPU KV cache usage: 96.4%, Prefix cache hit rate: 39.5%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2-7B-Instruct --dataset-name random \
--host 10.254.5.35 \
--num-prompts 50 \
--random-input-len 1024 \
--random-output-len 7168

============ Serving Benchmark Result ============
Successful requests:                     50        
Benchmark duration (s):                  535.71    
Total input tokens:                      50977     
Total generated tokens:                  350353    
Request throughput (req/s):              0.09      
Output token throughput (tok/s):         654.00    
Total Token throughput (tok/s):          749.15    
---------------Time to First Token----------------
Mean TTFT (ms):                          432.37    
Median TTFT (ms):                        515.77    
P99 TTFT (ms):                           549.18    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          74.62     
Median TPOT (ms):                        74.67     
P99 TPOT (ms):                           74.69     
---------------Inter-token Latency----------------
Mean ITL (ms):                           74.67     
Median ITL (ms):                         74.61     
P99 ITL (ms):                            82.40     
==================================================

(APIServer pid=158) INFO 09-02 13:48:25 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 627.7 tokens/s, Running: 48 reqs, Waiting: 0 reqs, GPU KV cache usage: 91.3%, Prefix cache hit rate: 47.4%
(APIServer pid=158) INFO 09-02 13:48:35 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 613.5 tokens/s, Running: 48 reqs, Waiting: 0 reqs, GPU KV cache usage: 92.8%, Prefix cache hit rate: 47.4%
(APIServer pid=158) INFO 09-02 13:48:45 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 629.7 tokens/s, Running: 48 reqs, Waiting: 0 reqs, GPU KV cache usage: 94.3%, Prefix cache hit rate: 47.4%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2-7B-Instruct --dataset-name random \
--host 10.254.5.35 \
--num-prompts 100 \
--random-input-len 1024 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     100       
Benchmark duration (s):                  114.01    
Total input tokens:                      102140    
Total generated tokens:                  100633    
Request throughput (req/s):              0.88      
Output token throughput (tok/s):         882.67    
Total Token throughput (tok/s):          1778.56   
---------------Time to First Token----------------
Mean TTFT (ms):                          6007.90   
Median TTFT (ms):                        1542.24   
P99 TTFT (ms):                           20100.59  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          104.61    
Median TPOT (ms):                        108.91    
P99 TPOT (ms):                           109.69    
---------------Inter-token Latency----------------
Mean ITL (ms):                           104.61    
Median ITL (ms):                         92.30     
P99 ITL (ms):                            746.54    
==================================================

(APIServer pid=158) INFO 09-02 13:53:16 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 1046.4 tokens/s, Running: 98 reqs, Waiting: 0 reqs, GPU KV cache usage: 41.2%, Prefix cache hit rate: 47.8%
(APIServer pid=158) INFO 09-02 13:53:26 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 1036.1 tokens/s, Running: 98 reqs, Waiting: 0 reqs, GPU KV cache usage: 43.7%, Prefix cache hit rate: 47.8%
(APIServer pid=158) INFO 09-02 13:53:36 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 1036.0 tokens/s, Running: 97 reqs, Waiting: 0 reqs, GPU KV cache usage: 45.7%, Prefix cache hit rate: 47.8%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2-7B-Instruct --dataset-name random \
--host 10.254.5.35 \
--num-prompts 100 \
--random-input-len 4096 \
--random-output-len 4096

============ Serving Benchmark Result ============
Successful requests:                     100       
Benchmark duration (s):                  818.58    
Total input tokens:                      409048    
Total generated tokens:                  401433    
Request throughput (req/s):              0.12      
Output token throughput (tok/s):         490.40    
Total Token throughput (tok/s):          990.10    
---------------Time to First Token----------------
Mean TTFT (ms):                          74806.04  
Median TTFT (ms):                        74078.45  
P99 TTFT (ms):                           150976.72 
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          144.83    
Median TPOT (ms):                        124.58    
P99 TPOT (ms):                           308.77    
---------------Inter-token Latency----------------
Mean ITL (ms):                           137.21    
Median ITL (ms):                         88.67     
P99 ITL (ms):                            781.95    
==================================================

(APIServer pid=114) INFO 09-03 06:08:58 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 679.1 tokens/s, Running: 56 reqs, Waiting: 42 reqs, GPU KV cache usage: 99.4%, Prefix cache hit rate: 49.1%
(APIServer pid=114) INFO 09-03 06:09:08 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 673.9 tokens/s, Running: 55 reqs, Waiting: 43 reqs, GPU KV cache usage: 99.1%, Prefix cache hit rate: 49.2%
(APIServer pid=114) INFO 09-03 06:09:18 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 578.6 tokens/s, Running: 55 reqs, Waiting: 42 reqs, GPU KV cache usage: 98.6%, Prefix cache hit rate: 49.0%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2-7B-Instruct --dataset-name random \
--host 10.254.5.39 \
--num-prompts 100 \
--random-input-len 1024 \
--random-output-len 7168

============ Serving Benchmark Result ============
Successful requests:                     100       
Benchmark duration (s):                  955.23    
Total input tokens:                      102140    
Total generated tokens:                  698617    
Request throughput (req/s):              0.10      
Output token throughput (tok/s):         731.36    
Total Token throughput (tok/s):          838.29    
---------------Time to First Token----------------
Mean TTFT (ms):                          681.41    
Median TTFT (ms):                        691.92    
P99 TTFT (ms):                           1048.86   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          105.61    
Median TPOT (ms):                        92.46     
P99 TPOT (ms):                           132.96    
---------------Inter-token Latency----------------
Mean ITL (ms):                           105.96    
Median ITL (ms):                         92.57     
P99 ITL (ms):                            108.09    
==================================================

(APIServer pid=114) INFO 09-03 06:32:01 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 804.6 tokens/s, Running: 75 reqs, Waiting: 22 reqs, GPU KV cache usage: 99.6%, Prefix cache hit rate: 50.5%
(APIServer pid=114) INFO 09-03 06:32:11 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 783.5 tokens/s, Running: 73 reqs, Waiting: 24 reqs, GPU KV cache usage: 98.8%, Prefix cache hit rate: 50.3%
(APIServer pid=114) INFO 09-03 06:32:21 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 790.2 tokens/s, Running: 72 reqs, Waiting: 25 reqs, GPU KV cache usage: 99.2%, Prefix cache hit rate: 50.9%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2-7B-Instruct --dataset-name random \
--host 10.254.5.39 \
--num-prompts 200 \
--random-input-len 1024 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     200       
Benchmark duration (s):                  210.25    
Total input tokens:                      204007    
Total generated tokens:                  200344    
Request throughput (req/s):              0.95      
Output token throughput (tok/s):         952.87    
Total Token throughput (tok/s):          1923.16   
---------------Time to First Token----------------
Mean TTFT (ms):                          38495.54  
Median TTFT (ms):                        37967.20  
P99 TTFT (ms):                           77008.96  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          165.86    
Median TPOT (ms):                        165.83    
P99 TPOT (ms):                           222.26    
---------------Inter-token Latency----------------
Mean ITL (ms):                           164.49    
Median ITL (ms):                         133.72    
P99 ITL (ms):                            764.40    
==================================================

(APIServer pid=119) INFO 09-04 05:30:59 [loggers.py:123] Engine 000: Avg prompt throughput: 2547.3 tokens/s, Avg generation throughput: 151.0 tokens/s, Running: 129 reqs, Waiting: 70 reqs, GPU KV cache usage: 30.6%, Prefix cache hit rate: 0.0%
(APIServer pid=119) INFO 09-04 05:31:09 [loggers.py:123] Engine 000: Avg prompt throughput: 2438.4 tokens/s, Avg generation throughput: 182.8 tokens/s, Running: 153 reqs, Waiting: 46 reqs, GPU KV cache usage: 36.6%, Prefix cache hit rate: 0.0%
(APIServer pid=119) INFO 09-04 05:31:19 [loggers.py:123] Engine 000: Avg prompt throughput: 2629.8 tokens/s, Avg generation throughput: 231.7 tokens/s, Running: 179 reqs, Waiting: 20 reqs, GPU KV cache usage: 43.1%, Prefix cache hit rate: 0.0%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2-7B-Instruct --dataset-name random \
--host 10.254.5.39 \
--num-prompts 500 \
--random-input-len 1024 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     500       
Benchmark duration (s):                  487.86    
Total input tokens:                      510414    
Total generated tokens:                  502996    
Request throughput (req/s):              1.02      
Output token throughput (tok/s):         1031.03   
Total Token throughput (tok/s):          2077.26   
---------------Time to First Token----------------
Mean TTFT (ms):                          135930.37 
Median TTFT (ms):                        40236.68  
P99 TTFT (ms):                           322714.38 
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          207.50    
Median TPOT (ms):                        193.62    
P99 TPOT (ms):                           346.71    
---------------Inter-token Latency----------------
Mean ITL (ms):                           203.14    
Median ITL (ms):                         157.63    
P99 ITL (ms):                            801.79    
==================================================

(APIServer pid=119) INFO 09-04 05:36:31 [loggers.py:123] Engine 000: Avg prompt throughput: 1708.0 tokens/s, Avg generation throughput: 646.0 tokens/s, Running: 256 reqs, Waiting: 242 reqs, GPU KV cache usage: 62.6%, Prefix cache hit rate: 35.3%
(APIServer pid=119) INFO 09-04 05:36:41 [loggers.py:123] Engine 000: Avg prompt throughput: 102.2 tokens/s, Avg generation throughput: 1558.5 tokens/s, Running: 255 reqs, Waiting: 241 reqs, GPU KV cache usage: 65.9%, Prefix cache hit rate: 35.2%
(APIServer pid=119) INFO 09-04 05:36:51 [loggers.py:123] Engine 000: Avg prompt throughput: 202.9 tokens/s, Avg generation throughput: 1521.5 tokens/s, Running: 256 reqs, Waiting: 239 reqs, GPU KV cache usage: 69.1%, Prefix cache hit rate: 35.1%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2-7B-Instruct --dataset-name random \
--host 10.254.5.39 \
--num-prompts 1000 \
--random-input-len 1024 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     1000      
Benchmark duration (s):                  1069.51   
Total input tokens:                      1021646   
Total generated tokens:                  1004370   
Request throughput (req/s):              0.94      
Output token throughput (tok/s):         939.09    
Total Token throughput (tok/s):          1894.34   
---------------Time to First Token----------------
Mean TTFT (ms):                          450011.50 
Median TTFT (ms):                        389526.45 
P99 TTFT (ms):                           919066.09 
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          240.08    
Median TPOT (ms):                        249.10    
P99 TPOT (ms):                           308.26    
---------------Inter-token Latency----------------
Mean ITL (ms):                           237.85    
Median ITL (ms):                         158.92    
P99 ITL (ms):                            808.02    
==================================================

(APIServer pid=119) INFO 09-04 06:02:08 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 1584.7 tokens/s, Running: 256 reqs, Waiting: 215 reqs, GPU KV cache usage: 99.9%, Prefix cache hit rate: 21.4%
(APIServer pid=119) INFO 09-04 06:02:18 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 1543.8 tokens/s, Running: 244 reqs, Waiting: 227 reqs, GPU KV cache usage: 99.9%, Prefix cache hit rate: 21.1%
(APIServer pid=119) INFO 09-04 06:02:28 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 1525.7 tokens/s, Running: 234 reqs, Waiting: 237 reqs, GPU KV cache usage: 99.8%, Prefix cache hit rate: 21.2%
```

#### <a name="toc_4"></a>✍️ Test 2: 2 GPU pods host 7B model in the same node (Available KV cache memory in each GPU: 12.02 GiB)

- Startup log: [vllm-7B-2gpuA10080GB-0.25GRAM-same-node.log](log/vllm-7B-2gpuA10080GB-0.25GRAM-same-node.log)

<img width="700" height="409" alt="image" src="https://github.com/user-attachments/assets/e6e42a91-f5a6-406d-8b23-b3dde2c76f21" />

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2-7B-Instruct --dataset-name random \
--host localhost \
--num-prompts 1 \
--random-input-len 1024 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     1         
Benchmark duration (s):                  9.06      
Total input tokens:                      1024      
Total generated tokens:                  1024      
Request throughput (req/s):              0.11      
Output token throughput (tok/s):         113.07    
Total Token throughput (tok/s):          226.14    
---------------Time to First Token----------------
Mean TTFT (ms):                          34.69     
Median TTFT (ms):                        34.69     
P99 TTFT (ms):                           34.69     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          8.82      
Median TPOT (ms):                        8.82      
P99 TPOT (ms):                           8.82      
---------------Inter-token Latency----------------
Mean ITL (ms):                           8.82      
Median ITL (ms):                         8.76      
P99 ITL (ms):                            10.24     
==================================================

(APIServer pid=1423) INFO 09-06 21:18:42 [loggers.py:123] Engine 000: Avg prompt throughput: 102.4 tokens/s, Avg generation throughput: 112.7 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.4%, Prefix cache hit rate: 49.2%
(APIServer pid=1423) INFO 09-06 21:18:52 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 29.3 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 49.2%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2-7B-Instruct --dataset-name random \
--host localhost \
--num-prompts 20 \
--random-input-len 1024 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     20        
Benchmark duration (s):                  15.10     
Total input tokens:                      20460     
Total generated tokens:                  20377     
Request throughput (req/s):              1.32      
Output token throughput (tok/s):         1349.66   
Total Token throughput (tok/s):          2704.81   
---------------Time to First Token----------------
Mean TTFT (ms):                          550.45    
Median TTFT (ms):                        405.39    
P99 TTFT (ms):                           1038.90   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          14.17     
Median TPOT (ms):                        14.30     
P99 TPOT (ms):                           14.52     
---------------Inter-token Latency----------------
Mean ITL (ms):                           14.16     
Median ITL (ms):                         13.56     
P99 ITL (ms):                            19.49     
==================================================

(APIServer pid=1423) INFO 09-06 21:23:02 [loggers.py:123] Engine 000: Avg prompt throughput: 2045.5 tokens/s, Avg generation throughput: 427.0 tokens/s, Running: 20 reqs, Waiting: 0 reqs, GPU KV cache usage: 5.4%, Prefix cache hit rate: 40.6%
(APIServer pid=1423) INFO 09-06 21:23:12 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 1455.7 tokens/s, Running: 20 reqs, Waiting: 0 reqs, GPU KV cache usage: 8.6%, Prefix cache hit rate: 40.6%
(APIServer pid=1423) INFO 09-06 21:23:22 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 228.5 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 40.6%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2-7B-Instruct --dataset-name random \
--host localhost \
--num-prompts 30 \
--random-input-len 1024 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     30        
Benchmark duration (s):                  16.58     
Total input tokens:                      30699     
Total generated tokens:                  30617     
Request throughput (req/s):              1.81      
Output token throughput (tok/s):         1847.17   
Total Token throughput (tok/s):          3699.30   
---------------Time to First Token----------------
Mean TTFT (ms):                          633.56    
Median TTFT (ms):                        729.59    
P99 TTFT (ms):                           1101.59   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          15.52     
Median TPOT (ms):                        15.43     
P99 TPOT (ms):                           15.87     
---------------Inter-token Latency----------------
Mean ITL (ms):                           15.52     
Median ITL (ms):                         14.92     
P99 ITL (ms):                            23.51     
==================================================

(APIServer pid=1423) INFO 09-06 21:24:22 [loggers.py:123] Engine 000: Avg prompt throughput: 3068.8 tokens/s, Avg generation throughput: 641.8 tokens/s, Running: 30 reqs, Waiting: 0 reqs, GPU KV cache usage: 8.1%, Prefix cache hit rate: 53.0%
(APIServer pid=1423) INFO 09-06 21:24:32 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 1994.4 tokens/s, Running: 30 reqs, Waiting: 0 reqs, GPU KV cache usage: 12.6%, Prefix cache hit rate: 53.0%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2-7B-Instruct --dataset-name random \
--host localhost \
--num-prompts 75 \
--random-input-len 1024 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     75        
Benchmark duration (s):                  27.52     
Total input tokens:                      76558     
Total generated tokens:                  76710     
Request throughput (req/s):              2.73      
Output token throughput (tok/s):         2787.42   
Total Token throughput (tok/s):          5569.32   
---------------Time to First Token----------------
Mean TTFT (ms):                          1740.61   
Median TTFT (ms):                        1217.12   
P99 TTFT (ms):                           4508.35   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          24.93     
Median TPOT (ms):                        25.39     
P99 TPOT (ms):                           26.24     
---------------Inter-token Latency----------------
Mean ITL (ms):                           24.93     
Median ITL (ms):                         22.37     
P99 ITL (ms):                            182.60    
==================================================

(APIServer pid=1423) INFO 09-06 21:26:42 [loggers.py:123] Engine 000: Avg prompt throughput: 7650.6 tokens/s, Avg generation throughput: 1428.8 tokens/s, Running: 75 reqs, Waiting: 0 reqs, GPU KV cache usage: 20.3%, Prefix cache hit rate: 46.1%
(APIServer pid=1423) INFO 09-06 21:26:52 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 3359.4 tokens/s, Running: 75 reqs, Waiting: 0 reqs, GPU KV cache usage: 27.7%, Prefix cache hit rate: 46.1%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2-7B-Instruct --dataset-name random \
--host localhost \
--num-prompts 10 \
--random-input-len 1024 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     10        
Benchmark duration (s):                  13.92     
Total input tokens:                      10239     
Total generated tokens:                  10240     
Request throughput (req/s):              0.72      
Output token throughput (tok/s):         735.47    
Total Token throughput (tok/s):          1470.87   
---------------Time to First Token----------------
Mean TTFT (ms):                          601.91    
Median TTFT (ms):                        632.50    
P99 TTFT (ms):                           916.26    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          12.99     
Median TPOT (ms):                        12.96     
P99 TPOT (ms):                           13.43     
---------------Inter-token Latency----------------
Mean ITL (ms):                           12.99     
Median ITL (ms):                         12.56     
P99 ITL (ms):                            20.16     
==================================================

(APIServer pid=2202) INFO 09-03 01:50:09 [loggers.py:123] Engine 000: Avg prompt throughput: 1023.7 tokens/s, Avg generation throughput: 540.5 tokens/s, Running: 10 reqs, Waiting: 0 reqs, GPU KV cache usage: 3.4%, Prefix cache hit rate: 48.0%
(APIServer pid=2202) INFO 09-03 01:50:19 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 511.8 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 48.0%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2-7B-Instruct --dataset-name random \
--host localhost \
--num-prompts 10 \
--random-input-len 4096 \
--random-output-len 4096

============ Serving Benchmark Result ============
Successful requests:                     10        
Benchmark duration (s):                  58.58     
Total input tokens:                      40952     
Total generated tokens:                  40960     
Request throughput (req/s):              0.17      
Output token throughput (tok/s):         699.24    
Total Token throughput (tok/s):          1398.34   
---------------Time to First Token----------------
Mean TTFT (ms):                          1676.30   
Median TTFT (ms):                        1790.69   
P99 TTFT (ms):                           2613.44   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          13.88     
Median TPOT (ms):                        13.85     
P99 TPOT (ms):                           14.13     
---------------Inter-token Latency----------------
Mean ITL (ms):                           13.88     
Median ITL (ms):                         13.52     
P99 ITL (ms):                            17.73     
==================================================

(APIServer pid=2202) INFO 09-03 01:52:49 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 737.8 tokens/s, Running: 10 reqs, Waiting: 0 reqs, GPU KV cache usage: 15.0%, Prefix cache hit rate: 42.2%
(APIServer pid=2202) INFO 09-03 01:52:59 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 720.9 tokens/s, Running: 10 reqs, Waiting: 0 reqs, GPU KV cache usage: 16.6%, Prefix cache hit rate: 42.2%
(APIServer pid=2202) INFO 09-03 01:53:09 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 715.9 tokens/s, Running: 10 reqs, Waiting: 0 reqs, GPU KV cache usage: 18.2%, Prefix cache hit rate: 42.2%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2-7B-Instruct --dataset-name random \
--host localhost \
--num-prompts 10 \
--random-input-len 1024 \
--random-output-len 7168

============ Serving Benchmark Result ============
Successful requests:                     10        
Benchmark duration (s):                  95.04     
Total input tokens:                      10239     
Total generated tokens:                  69058     
Request throughput (req/s):              0.11      
Output token throughput (tok/s):         726.59    
Total Token throughput (tok/s):          834.32    
---------------Time to First Token----------------
Mean TTFT (ms):                          111.47    
Median TTFT (ms):                        112.36    
P99 TTFT (ms):                           122.76    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          13.23     
Median TPOT (ms):                        13.24     
P99 TPOT (ms):                           13.25     
---------------Inter-token Latency----------------
Mean ITL (ms):                           13.23     
Median ITL (ms):                         13.11     
P99 ITL (ms):                            18.22     
==================================================

(APIServer pid=2202) INFO 09-03 01:56:49 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 689.3 tokens/s, Running: 9 reqs, Waiting: 0 reqs, GPU KV cache usage: 12.1%, Prefix cache hit rate: 42.6%
(APIServer pid=2202) INFO 09-03 01:56:59 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 663.2 tokens/s, Running: 9 reqs, Waiting: 0 reqs, GPU KV cache usage: 13.6%, Prefix cache hit rate: 42.6%
(APIServer pid=2202) INFO 09-03 01:57:09 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 662.3 tokens/s, Running: 9 reqs, Waiting: 0 reqs, GPU KV cache usage: 15.1%, Prefix cache hit rate: 42.6%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2-7B-Instruct --dataset-name random \
--host localhost \
--num-prompts 50 \
--random-input-len 1024 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     50        
Benchmark duration (s):                  22.30     
Total input tokens:                      50977     
Total generated tokens:                  51097     
Request throughput (req/s):              2.24      
Output token throughput (tok/s):         2291.62   
Total Token throughput (tok/s):          4577.87   
---------------Time to First Token----------------
Mean TTFT (ms):                          2154.87   
Median TTFT (ms):                        2371.40   
P99 TTFT (ms):                           3732.60   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          19.53     
Median TPOT (ms):                        19.33     
P99 TPOT (ms):                           21.29     
---------------Inter-token Latency----------------
Mean ITL (ms):                           19.53     
Median ITL (ms):                         17.94     
P99 ITL (ms):                            30.98     
==================================================

(APIServer pid=2202) INFO 09-03 02:00:09 [loggers.py:123] Engine 000: Avg prompt throughput: 5095.0 tokens/s, Avg generation throughput: 150.2 tokens/s, Running: 50 reqs, Waiting: 0 reqs, GPU KV cache usage: 11.6%, Prefix cache hit rate: 43.2%
(APIServer pid=2202) INFO 09-03 02:00:19 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 2798.5 tokens/s, Running: 50 reqs, Waiting: 0 reqs, GPU KV cache usage: 17.8%, Prefix cache hit rate: 44.4%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2-7B-Instruct --dataset-name random \
--host localhost \
--num-prompts 50 \
--random-input-len 4096 \
--random-output-len 4096

============ Serving Benchmark Result ============
Successful requests:                     50        
Benchmark duration (s):                  102.53    
Total input tokens:                      204464    
Total generated tokens:                  200709    
Request throughput (req/s):              0.49      
Output token throughput (tok/s):         1957.56   
Total Token throughput (tok/s):          3951.74   
---------------Time to First Token----------------
Mean TTFT (ms):                          5291.07   
Median TTFT (ms):                        5070.00   
P99 TTFT (ms):                           11893.81  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          27.01     
Median TPOT (ms):                        23.72     
P99 TPOT (ms):                           110.85    
---------------Inter-token Latency----------------
Mean ITL (ms):                           23.61     
Median ITL (ms):                         22.08     
P99 ITL (ms):                            31.11     
==================================================

(APIServer pid=2202) INFO 09-03 02:03:39 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 2174.2 tokens/s, Running: 49 reqs, Waiting: 0 reqs, GPU KV cache usage: 75.8%, Prefix cache hit rate: 44.5%
(APIServer pid=2202) INFO 09-03 02:03:49 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 2125.4 tokens/s, Running: 49 reqs, Waiting: 0 reqs, GPU KV cache usage: 80.5%, Prefix cache hit rate: 44.5%
(APIServer pid=2202) INFO 09-03 02:03:59 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 2066.6 tokens/s, Running: 49 reqs, Waiting: 0 reqs, GPU KV cache usage: 85.1%, Prefix cache hit rate: 44.5%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2-7B-Instruct --dataset-name random \
--host localhost \
--num-prompts 50 \
--random-input-len 1024 \
--random-output-len 7168

============ Serving Benchmark Result ============
Successful requests:                     50        
Benchmark duration (s):                  146.97    
Total input tokens:                      50977     
Total generated tokens:                  349917    
Request throughput (req/s):              0.34      
Output token throughput (tok/s):         2380.86   
Total Token throughput (tok/s):          2727.71   
---------------Time to First Token----------------
Mean TTFT (ms):                          279.73    
Median TTFT (ms):                        278.82    
P99 TTFT (ms):                           348.43    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          20.40     
Median TPOT (ms):                        20.46     
P99 TPOT (ms):                           20.46     
---------------Inter-token Latency----------------
Mean ITL (ms):                           20.44     
Median ITL (ms):                         20.46     
P99 ITL (ms):                            26.64     
==================================================

(APIServer pid=2202) INFO 09-03 02:08:19 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 2182.6 tokens/s, Running: 48 reqs, Waiting: 0 reqs, GPU KV cache usage: 77.9%, Prefix cache hit rate: 47.4%
(APIServer pid=2202) INFO 09-03 02:08:29 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 2135.3 tokens/s, Running: 48 reqs, Waiting: 0 reqs, GPU KV cache usage: 82.7%, Prefix cache hit rate: 47.4%
(APIServer pid=2202) INFO 09-03 02:08:39 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 2082.3 tokens/s, Running: 48 reqs, Waiting: 0 reqs, GPU KV cache usage: 87.3%, Prefix cache hit rate: 47.4%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2-7B-Instruct --dataset-name random \
--host localhost \
--num-prompts 100 \
--random-input-len 1024 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     100       
Benchmark duration (s):                  35.12     
Total input tokens:                      102140    
Total generated tokens:                  101694    
Request throughput (req/s):              2.85      
Output token throughput (tok/s):         2895.58   
Total Token throughput (tok/s):          5803.86   
---------------Time to First Token----------------
Mean TTFT (ms):                          5023.97   
Median TTFT (ms):                        4966.24   
P99 TTFT (ms):                           9505.05   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          28.98     
Median TPOT (ms):                        29.04     
P99 TPOT (ms):                           32.77     
---------------Inter-token Latency----------------
Mean ITL (ms):                           28.98     
Median ITL (ms):                         25.04     
P99 ITL (ms):                            186.15    
==================================================

(APIServer pid=1217) INFO 09-03 01:13:01 [loggers.py:123] Engine 000: Avg prompt throughput: 2556.2 tokens/s, Avg generation throughput: 3233.3 tokens/s, Running: 100 reqs, Waiting: 0 reqs, GPU KV cache usage: 30.3%, Prefix cache hit rate: 1.0%
(APIServer pid=1217) INFO 09-03 01:13:11 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 3967.2 tokens/s, Running: 99 reqs, Waiting: 0 reqs, GPU KV cache usage: 38.8%, Prefix cache hit rate: 1.0%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2-7B-Instruct --dataset-name random \
--host localhost \
--num-prompts 100 \
--random-input-len 4096 \
--random-output-len 4096

============ Serving Benchmark Result ============
Successful requests:                     100       
Benchmark duration (s):                  243.10    
Total input tokens:                      409048    
Total generated tokens:                  397651    
Request throughput (req/s):              0.41      
Output token throughput (tok/s):         1635.76   
Total Token throughput (tok/s):          3318.40   
---------------Time to First Token----------------
Mean TTFT (ms):                          20237.07  
Median TTFT (ms):                        19708.69  
P99 TTFT (ms):                           41456.89  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          44.04     
Median TPOT (ms):                        38.29     
P99 TPOT (ms):                           111.73    
---------------Inter-token Latency----------------
Mean ITL (ms):                           41.59     
Median ITL (ms):                         28.28     
P99 ITL (ms):                            224.41    
==================================================

(APIServer pid=2202) INFO 09-03 01:33:49 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 2312.7 tokens/s, Running: 61 reqs, Waiting: 36 reqs, GPU KV cache usage: 99.6%, Prefix cache hit rate: 48.5%
(APIServer pid=2202) INFO 09-03 01:33:59 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 2185.0 tokens/s, Running: 58 reqs, Waiting: 39 reqs, GPU KV cache usage: 99.5%, Prefix cache hit rate: 48.8%
(APIServer pid=2202) INFO 09-03 01:34:09 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 2108.4 tokens/s, Running: 55 reqs, Waiting: 42 reqs, GPU KV cache usage: 99.0%, Prefix cache hit rate: 49.1%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2-7B-Instruct --dataset-name random \
--host localhost \
--num-prompts 100 \
--random-input-len 1024 \
--random-output-len 7168

============ Serving Benchmark Result ============
Successful requests:                     100       
Benchmark duration (s):                  295.52    
Total input tokens:                      102140    
Total generated tokens:                  710553    
Request throughput (req/s):              0.34      
Output token throughput (tok/s):         2404.42   
Total Token throughput (tok/s):          2750.05   
---------------Time to First Token----------------
Mean TTFT (ms):                          3659.36   
Median TTFT (ms):                        4546.82   
P99 TTFT (ms):                           4865.53   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          32.63     
Median TPOT (ms):                        28.91     
P99 TPOT (ms):                           40.48     
---------------Inter-token Latency----------------
Mean ITL (ms):                           32.66     
Median ITL (ms):                         27.66     
P99 ITL (ms):                            38.93     
==================================================

(APIServer pid=2202) INFO 09-03 01:40:49 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 2283.7 tokens/s, Running: 60 reqs, Waiting: 39 reqs, GPU KV cache usage: 98.9%, Prefix cache hit rate: 50.0%
(APIServer pid=2202) INFO 09-03 01:40:59 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 2183.6 tokens/s, Running: 57 reqs, Waiting: 42 reqs, GPU KV cache usage: 98.7%, Prefix cache hit rate: 50.0%
(APIServer pid=2202) INFO 09-03 01:41:09 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 2124.6 tokens/s, Running: 55 reqs, Waiting: 44 reqs, GPU KV cache usage: 99.8%, Prefix cache hit rate: 49.7%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2-7B-Instruct --dataset-name random \
--host localhost \
--num-prompts 100 \
--random-input-len 7168 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     100       
Benchmark duration (s):                  125.62    
Total input tokens:                      716529    
Total generated tokens:                  101734    
Request throughput (req/s):              0.80      
Output token throughput (tok/s):         809.82    
Total Token throughput (tok/s):          6513.54   
---------------Time to First Token----------------
Mean TTFT (ms):                          48097.87  
Median TTFT (ms):                        36414.96  
P99 TTFT (ms):                           103681.10 
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          55.10     
Median TPOT (ms):                        65.65     
P99 TPOT (ms):                           72.70     
---------------Inter-token Latency----------------
Mean ITL (ms):                           55.13     
Median ITL (ms):                         26.01     
P99 ITL (ms):                            234.15    
==================================================

(APIServer pid=2202) INFO 09-03 01:46:19 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 2132.9 tokens/s, Running: 58 reqs, Waiting: 42 reqs, GPU KV cache usage: 99.9%, Prefix cache hit rate: 48.0%
(APIServer pid=2202) INFO 09-03 01:46:29 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 1928.2 tokens/s, Running: 55 reqs, Waiting: 43 reqs, GPU KV cache usage: 98.2%, Prefix cache hit rate: 48.5%
(APIServer pid=2202) INFO 09-03 01:46:39 [loggers.py:123] Engine 000: Avg prompt throughput: 5017.5 tokens/s, Avg generation throughput: 233.4 tokens/s, Running: 55 reqs, Waiting: 31 reqs, GPU KV cache usage: 95.9%, Prefix cache hit rate: 48.4%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2-7B-Instruct --dataset-name random \
--host localhost \
--num-prompts 200 \
--random-input-len 1024 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     200       
Benchmark duration (s):                  54.44     
Total input tokens:                      204007    
Total generated tokens:                  201481    
Request throughput (req/s):              3.67      
Output token throughput (tok/s):         3700.99   
Total Token throughput (tok/s):          7448.37   
---------------Time to First Token----------------
Mean TTFT (ms):                          3490.37   
Median TTFT (ms):                        1227.51   
P99 TTFT (ms):                           11471.29  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          49.23     
Median TPOT (ms):                        51.00     
P99 TPOT (ms):                           62.41     
---------------Inter-token Latency----------------
Mean ITL (ms):                           48.97     
Median ITL (ms):                         43.04     
P99 ITL (ms):                            208.40    
==================================================

(APIServer pid=1217) INFO 09-03 01:15:51 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 4645.8 tokens/s, Running: 196 reqs, Waiting: 0 reqs, GPU KV cache usage: 68.8%, Prefix cache hit rate: 33.3%
(APIServer pid=1217) INFO 09-03 01:16:01 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 4539.7 tokens/s, Running: 196 reqs, Waiting: 0 reqs, GPU KV cache usage: 79.0%, Prefix cache hit rate: 33.3%
(APIServer pid=1217) INFO 09-03 01:16:11 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 4402.6 tokens/s, Running: 82 reqs, Waiting: 0 reqs, GPU KV cache usage: 36.9%, Prefix cache hit rate: 33.3%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2-7B-Instruct --dataset-name random \
--host localhost \
--num-prompts 500 \
--random-input-len 1024 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     500       
Benchmark duration (s):                  143.45    
Total input tokens:                      510414    
Total generated tokens:                  504295    
Request throughput (req/s):              3.49      
Output token throughput (tok/s):         3515.49   
Total Token throughput (tok/s):          7073.64   
---------------Time to First Token----------------
Mean TTFT (ms):                          38204.56  
Median TTFT (ms):                        7941.96   
P99 TTFT (ms):                           90563.28  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          62.28     
Median TPOT (ms):                        59.75     
P99 TPOT (ms):                           89.49     
---------------Inter-token Latency----------------
Mean ITL (ms):                           61.67     
Median ITL (ms):                         50.56     
P99 ITL (ms):                            225.64    
==================================================

(APIServer pid=1217) INFO 09-03 01:18:41 [loggers.py:123] Engine 000: Avg prompt throughput: 102.3 tokens/s, Avg generation throughput: 4962.0 tokens/s, Running: 256 reqs, Waiting: 238 reqs, GPU KV cache usage: 95.1%, Prefix cache hit rate: 52.8%
(APIServer pid=1217) INFO 09-03 01:18:51 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 4897.4 tokens/s, Running: 240 reqs, Waiting: 254 reqs, GPU KV cache usage: 99.8%, Prefix cache hit rate: 51.4%
(APIServer pid=1217) INFO 09-03 01:19:01 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 3883.4 tokens/s, Running: 128 reqs, Waiting: 266 reqs, GPU KV cache usage: 57.6%, Prefix cache hit rate: 47.8%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2-7B-Instruct --dataset-name random \
--host localhost \
--num-prompts 1000 \
--random-input-len 1024 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     1000      
Benchmark duration (s):                  320.48    
Total input tokens:                      1021646   
Total generated tokens:                  1002399   
Request throughput (req/s):              3.12      
Output token throughput (tok/s):         3127.81   
Total Token throughput (tok/s):          6315.67   
---------------Time to First Token----------------
Mean TTFT (ms):                          132939.26 
Median TTFT (ms):                        111853.63 
P99 TTFT (ms):                           273499.13 
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          73.12     
Median TPOT (ms):                        75.48     
P99 TPOT (ms):                           128.47    
---------------Inter-token Latency----------------
Mean ITL (ms):                           71.94     
Median ITL (ms):                         50.20     
P99 ITL (ms):                            240.71    
==================================================

(APIServer pid=144) INFO 09-03 23:10:51 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 4910.1 tokens/s, Running: 256 reqs, Waiting: 738 reqs, GPU KV cache usage: 96.1%, Prefix cache hit rate: 19.1%
(APIServer pid=144) INFO 09-03 23:11:01 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 4766.0 tokens/s, Running: 238 reqs, Waiting: 756 reqs, GPU KV cache usage: 99.8%, Prefix cache hit rate: 21.1%
(APIServer pid=144) INFO 09-03 23:11:11 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 2792.3 tokens/s, Running: 205 reqs, Waiting: 748 reqs, GPU KV cache usage: 90.4%, Prefix cache hit rate: 22.1%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2-7B-Instruct --dataset-name random \
--host localhost \
--num-prompts 500 \
--random-input-len 128 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     500       
Benchmark duration (s):                  92.00     
Total input tokens:                      63699     
Total generated tokens:                  458216    
Request throughput (req/s):              5.43      
Output token throughput (tok/s):         4980.67   
Total Token throughput (tok/s):          5673.05   
---------------Time to First Token----------------
Mean TTFT (ms):                          23181.68  
Median TTFT (ms):                        3114.76   
P99 TTFT (ms):                           53236.58  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          44.52     
Median TPOT (ms):                        46.98     
P99 TPOT (ms):                           52.90     
---------------Inter-token Latency----------------
Mean ITL (ms):                           44.56     
Median ITL (ms):                         43.54     
P99 ITL (ms):                            91.09     
==================================================

(APIServer pid=144) INFO 09-03 23:40:02 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 5049.7 tokens/s, Running: 185 reqs, Waiting: 0 reqs, GPU KV cache usage: 28.1%, Prefix cache hit rate: 10.0%
(APIServer pid=144) INFO 09-03 23:40:12 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 4804.8 tokens/s, Running: 175 reqs, Waiting: 0 reqs, GPU KV cache usage: 36.6%, Prefix cache hit rate: 10.1%
(APIServer pid=144) INFO 09-03 23:40:22 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 3810.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 10.1%
(APIServer pid=144) INFO 09-03 23:40:32 [loggers.py:123] Engine 000: Avg prompt throughput:
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2-7B-Instruct --dataset-name random \
--host localhost \
--num-prompts 500 \
--random-input-len 1024 \
--random-output-len 128

============ Serving Benchmark Result ============
Successful requests:                     500       
Benchmark duration (s):                  61.34     
Total input tokens:                      510414    
Total generated tokens:                  63591     
Request throughput (req/s):              8.15      
Output token throughput (tok/s):         1036.74   
Total Token throughput (tok/s):          9358.19   
---------------Time to First Token----------------
Mean TTFT (ms):                          27812.72  
Median TTFT (ms):                        27000.39  
P99 TTFT (ms):                           57065.13  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          174.10    
Median TPOT (ms):                        205.83    
P99 TPOT (ms):                           219.62    
---------------Inter-token Latency----------------
Mean ITL (ms):                           173.98    
Median ITL (ms):                         216.84    
P99 ITL (ms):                            230.72    
==================================================

(APIServer pid=144) INFO 09-03 23:39:12 [loggers.py:123] Engine 000: Avg prompt throughput: 140.7 tokens/s, Avg generation throughput: 5395.9 tokens/s, Running: 256 reqs, Waiting: 210 reqs, GPU KV cache usage: 32.2%, Prefix cache hit rate: 12.5%
(APIServer pid=144) INFO 09-03 23:39:22 [loggers.py:123] Engine 000: Avg prompt throughput: 89.5 tokens/s, Avg generation throughput: 5371.0 tokens/s, Running: 256 reqs, Waiting: 203 reqs, GPU KV cache usage: 43.2%, Prefix cache hit rate: 12.7%
(APIServer pid=144) INFO 09-03 23:39:32 [loggers.py:123] Engine 000: Avg prompt throughput: 76.7 tokens/s, Avg generation throughput: 5267.1 tokens/s, Running: 256 reqs, Waiting: 197 reqs, GPU KV cache usage: 54.0%, Prefix cache hit rate: 12.7%
```

#### <a name="toc_5"></a>✍️ Test 3: 2 GPU pods host 7B model in the same node (Available KV cache memory in each GPU: 31.83 GiB)

- Startup log: [vllm-7B-2gpuA10080GB-0.5GRAM-same-node.log](log/vllm-7B-2gpuA10080GB-0.5GRAM-same-node.log)

<img width="700" height="404" alt="image" src="https://github.com/user-attachments/assets/30db9775-e231-4f9b-bf7c-f7c18f7cfecb" />

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2-7B-Instruct --dataset-name random \
--host localhost \
--num-prompts 500 \
--random-input-len 1024 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     500       
Benchmark duration (s):                  142.29    
Total input tokens:                      510414    
Total generated tokens:                  503887    
Request throughput (req/s):              3.51      
Output token throughput (tok/s):         3541.34   
Total Token throughput (tok/s):          7128.55   
---------------Time to First Token----------------
Mean TTFT (ms):                          43636.24  
Median TTFT (ms):                        18233.81  
P99 TTFT (ms):                           93427.34  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          63.40     
Median TPOT (ms):                        64.65     
P99 TPOT (ms):                           125.97    
---------------Inter-token Latency----------------
Mean ITL (ms):                           62.23     
Median ITL (ms):                         49.63     
P99 ITL (ms):                            243.66    
==================================================

(APIServer pid=3761) INFO 09-03 02:23:12 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 4964.4 tokens/s, Running: 256 reqs, Waiting: 240 reqs, GPU KV cache usage: 41.7%, Prefix cache hit rate: 24.8%
(APIServer pid=3761) INFO 09-03 02:23:22 [loggers.py:123] Engine 000: Avg prompt throughput: 5628.8 tokens/s, Avg generation throughput: 2219.8 tokens/s, Running: 250 reqs, Waiting: 184 reqs, GPU KV cache usage: 37.8%, Prefix cache hit rate: 24.0%
(APIServer pid=3761) INFO 09-03 02:23:32 [loggers.py:123] Engine 000: Avg prompt throughput: 8351.5 tokens/s, Avg generation throughput: 966.0 tokens/s, Running: 170 reqs, Waiting: 102 reqs, GPU KV cache usage: 17.8%, Prefix cache hit rate: 20.3%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2-7B-Instruct --dataset-name random \
--host localhost \
--num-prompts 500 \
--random-input-len 1024 \
--random-output-len 128

============ Serving Benchmark Result ============
Successful requests:                     500       
Benchmark duration (s):                  60.17     
Total input tokens:                      510414    
Total generated tokens:                  63590     
Request throughput (req/s):              8.31      
Output token throughput (tok/s):         1056.89   
Total Token throughput (tok/s):          9540.17   
---------------Time to First Token----------------
Mean TTFT (ms):                          27092.85  
Median TTFT (ms):                        25713.76  
P99 TTFT (ms):                           55862.23  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          174.26    
Median TPOT (ms):                        205.38    
P99 TPOT (ms):                           220.34    
---------------Inter-token Latency----------------
Mean ITL (ms):                           174.08    
Median ITL (ms):                         216.69    
P99 ITL (ms):                            233.63    
==================================================

(APIServer pid=144) INFO 09-03 23:29:42 [loggers.py:123] Engine 000: Avg prompt throughput: 9592.6 tokens/s, Avg generation throughput: 1026.7 tokens/s, Running: 250 reqs, Waiting: 230 reqs, GPU KV cache usage: 60.5%, Prefix cache hit rate: 18.1%
(APIServer pid=144) INFO 09-03 23:29:52 [loggers.py:123] Engine 000: Avg prompt throughput: 8088.1 tokens/s, Avg generation throughput: 1105.2 tokens/s, Running: 240 reqs, Waiting: 151 reqs, GPU KV cache usage: 58.2%, Prefix cache hit rate: 17.5%
(APIServer pid=144) INFO 09-03 23:30:02 [loggers.py:123] Engine 000: Avg prompt throughput: 8445.0 tokens/s, Avg generation throughput: 1098.0 tokens/s, Running: 236 reqs, Waiting: 68 reqs, GPU KV cache usage: 57.2%, Prefix cache hit rate: 17.0%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2-7B-Instruct --dataset-name random \
--host localhost \
--num-prompts 1000 \
--random-input-len 1024 \
--random-output-len 128

============ Serving Benchmark Result ============
Successful requests:                     1000      
Benchmark duration (s):                  115.70    
Total input tokens:                      1021646   
Total generated tokens:                  126973    
Request throughput (req/s):              8.64      
Output token throughput (tok/s):         1097.47   
Total Token throughput (tok/s):          9927.94   
---------------Time to First Token----------------
Mean TTFT (ms):                          55811.58  
Median TTFT (ms):                        57269.96  
P99 TTFT (ms):                           110722.62 
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          189.35    
Median TPOT (ms):                        203.71    
P99 TPOT (ms):                           217.69    
---------------Inter-token Latency----------------
Mean ITL (ms):                           189.52    
Median ITL (ms):                         215.65    
P99 ITL (ms):                            234.64    
==================================================

(APIServer pid=3761) INFO 09-03 02:32:22 [loggers.py:123] Engine 000: Avg prompt throughput: 8681.8 tokens/s, Avg generation throughput: 1067.8 tokens/s, Running: 230 reqs, Waiting: 603 reqs, GPU KV cache usage: 21.0%, Prefix cache hit rate: 32.2%
(APIServer pid=3761) INFO 09-03 02:32:32 [loggers.py:123] Engine 000: Avg prompt throughput: 8456.7 tokens/s, Avg generation throughput: 1053.5 tokens/s, Running: 229 reqs, Waiting: 520 reqs, GPU KV cache usage: 20.9%, Prefix cache hit rate: 31.3%
(APIServer pid=3761) INFO 09-03 02:32:42 [loggers.py:123] Engine 000: Avg prompt throughput: 8990.4 tokens/s, Avg generation throughput: 1082.1 tokens/s, Running: 231 reqs, Waiting: 432 reqs, GPU KV cache usage: 21.1%, Prefix cache hit rate: 30.6%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2-7B-Instruct --dataset-name random \
--host localhost \
--num-prompts 2000 \
--random-input-len 1024 \
--random-output-len 128

============ Serving Benchmark Result ============
Successful requests:                     1633      
Benchmark duration (s):                  169.20    
Total input tokens:                      1667985   
Total generated tokens:                  206950    
Request throughput (req/s):              9.65      
Output token throughput (tok/s):         1223.08   
Total Token throughput (tok/s):          11080.95  
---------------Time to First Token----------------
Mean TTFT (ms):                          74563.36  
Median TTFT (ms):                        67010.85  
P99 TTFT (ms):                           163100.51 
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          173.06    
Median TPOT (ms):                        214.70    
P99 TPOT (ms):                           217.59    
---------------Inter-token Latency----------------
Mean ITL (ms):                           172.97    
Median ITL (ms):                         213.99    
P99 ITL (ms):                            233.31    
==================================================

(APIServer pid=3761) INFO 09-03 02:28:42 [loggers.py:123] Engine 000: Avg prompt throughput: 8690.2 tokens/s, Avg generation throughput: 1048.6 tokens/s, Running: 226 reqs, Waiting: 754 reqs, GPU KV cache usage: 20.5%, Prefix cache hit rate: 48.8%
(APIServer pid=3761) INFO 09-03 02:28:52 [loggers.py:123] Engine 000: Avg prompt throughput: 8585.7 tokens/s, Avg generation throughput: 1071.0 tokens/s, Running: 228 reqs, Waiting: 670 reqs, GPU KV cache usage: 20.8%, Prefix cache hit rate: 46.8%
(APIServer pid=3761) INFO 09-03 02:29:02 [loggers.py:123] Engine 000: Avg prompt throughput: 8372.9 tokens/s, Avg generation throughput: 1051.1 tokens/s, Running: 229 reqs, Waiting: 588 reqs, GPU KV cache usage: 21.0%, Prefix cache hit rate: 45.1%
```


### <a name="toc_6"></a>✍️ Test 4: 1 GPU pod hosts 7B model (Available KV cache memory: 5.01 GiB)

- Startup.log: [vllm-7B-1gpuA10080GB-0.25GRAM.log](log/vllm-7B-1gpuA10080GB-0.25GRAM.log)

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2-7B-Instruct --dataset-name random \
--host localhost \
--num-prompts 1 \
--random-input-len 1024 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     1         
Benchmark duration (s):                  12.06     
Total input tokens:                      1024      
Total generated tokens:                  1024      
Request throughput (req/s):              0.08      
Output token throughput (tok/s):         84.88     
Total Token throughput (tok/s):          169.76    
---------------Time to First Token----------------
Mean TTFT (ms):                          36.46     
Median TTFT (ms):                        36.46     
P99 TTFT (ms):                           36.46     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          11.76     
Median TPOT (ms):                        11.76     
P99 TPOT (ms):                           11.76     
---------------Inter-token Latency----------------
Mean ITL (ms):                           11.76     
Median ITL (ms):                         11.65     
P99 ITL (ms):                            14.87     
==================================================

(APIServer pid=113) INFO 09-06 20:58:38 [loggers.py:123] Engine 000: Avg prompt throughput: 102.4 tokens/s, Avg generation throughput: 85.1 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 1.9%, Prefix cache hit rate: 22.7%
(APIServer pid=113) INFO 09-06 20:58:48 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 24.2 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 22.7%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2-7B-Instruct --dataset-name random \
--host localhost \
--num-prompts 10 \
--random-input-len 1024 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     10        
Benchmark duration (s):                  14.31     
Total input tokens:                      10239     
Total generated tokens:                  10240     
Request throughput (req/s):              0.70      
Output token throughput (tok/s):         715.43    
Total Token throughput (tok/s):          1430.80   
---------------Time to First Token----------------
Mean TTFT (ms):                          503.72    
Median TTFT (ms):                        553.62    
P99 TTFT (ms):                           699.66    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          13.46     
Median TPOT (ms):                        13.41     
P99 TPOT (ms):                           13.76     
---------------Inter-token Latency----------------
Mean ITL (ms):                           13.46     
Median ITL (ms):                         13.09     
P99 ITL (ms):                            19.49     
==================================================

(APIServer pid=113) INFO 09-06 20:57:18 [loggers.py:123] Engine 000: Avg prompt throughput: 1023.8 tokens/s, Avg generation throughput: 424.1 tokens/s, Running: 10 reqs, Waiting: 0 reqs, GPU KV cache usage: 15.2%, Prefix cache hit rate: 8.9%
(APIServer pid=113) INFO 09-06 20:57:28 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 635.7 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 8.9%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2-7B-Instruct --dataset-name random \
--host localhost \
--num-prompts 20 \
--random-input-len 1024 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     20        
Benchmark duration (s):                  15.22     
Total input tokens:                      20460     
Total generated tokens:                  20377     
Request throughput (req/s):              1.31      
Output token throughput (tok/s):         1338.72   
Total Token throughput (tok/s):          2682.90   
---------------Time to First Token----------------
Mean TTFT (ms):                          463.10    
Median TTFT (ms):                        364.37    
P99 TTFT (ms):                           843.99    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          14.36     
Median TPOT (ms):                        14.46     
P99 TPOT (ms):                           14.61     
---------------Inter-token Latency----------------
Mean ITL (ms):                           14.36     
Median ITL (ms):                         13.92     
P99 ITL (ms):                            19.99     
==================================================

(APIServer pid=113) INFO 09-06 21:00:58 [loggers.py:123] Engine 000: Avg prompt throughput: 2045.4 tokens/s, Avg generation throughput: 256.2 tokens/s, Running: 20 reqs, Waiting: 0 reqs, GPU KV cache usage: 23.9%, Prefix cache hit rate: 40.6%
(APIServer pid=113) INFO 09-06 21:01:08 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 1435.6 tokens/s, Running: 20 reqs, Waiting: 0 reqs, GPU KV cache usage: 39.2%, Prefix cache hit rate: 40.6%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2-7B-Instruct --dataset-name random \
--host localhost \
--num-prompts 30 \
--random-input-len 1024 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     30        
Benchmark duration (s):                  16.74     
Total input tokens:                      30699     
Total generated tokens:                  30639     
Request throughput (req/s):              1.79      
Output token throughput (tok/s):         1830.69   
Total Token throughput (tok/s):          3664.96   
---------------Time to First Token----------------
Mean TTFT (ms):                          1033.68   
Median TTFT (ms):                        935.87    
P99 TTFT (ms):                           1444.56   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          15.19     
Median TPOT (ms):                        15.28     
P99 TPOT (ms):                           15.37     
---------------Inter-token Latency----------------
Mean ITL (ms):                           15.27     
Median ITL (ms):                         14.83     
P99 ITL (ms):                            19.28     
==================================================

(APIServer pid=113) INFO 09-06 21:02:58 [loggers.py:123] Engine 000: Avg prompt throughput: 3068.6 tokens/s, Avg generation throughput: 1707.3 tokens/s, Running: 30 reqs, Waiting: 0 reqs, GPU KV cache usage: 51.1%, Prefix cache hit rate: 53.0%
(APIServer pid=113) INFO 09-06 21:03:08 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 1358.5 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 53.0%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2-7B-Instruct --dataset-name random \
--host localhost \
--num-prompts 50 \
--random-input-len 1024 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     50        
Benchmark duration (s):                  21.86     
Total input tokens:                      50977     
Total generated tokens:                  51119     
Request throughput (req/s):              2.29      
Output token throughput (tok/s):         2338.38   
Total Token throughput (tok/s):          4670.27   
---------------Time to First Token----------------
Mean TTFT (ms):                          693.75    
Median TTFT (ms):                        424.74    
P99 TTFT (ms):                           1690.56   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          18.46     
Median TPOT (ms):                        18.55     
P99 TPOT (ms):                           19.38     
---------------Inter-token Latency----------------
Mean ITL (ms):                           18.46     
Median ITL (ms):                         17.08     
P99 ITL (ms):                            50.19     
==================================================

(APIServer pid=113) INFO 09-06 21:04:18 [loggers.py:123] Engine 000: Avg prompt throughput: 4790.1 tokens/s, Avg generation throughput: 104.3 tokens/s, Running: 48 reqs, Waiting: 2 reqs, GPU KV cache usage: 52.3%, Prefix cache hit rate: 57.1%
(APIServer pid=113) INFO 09-06 21:04:28 [loggers.py:123] Engine 000: Avg prompt throughput: 307.0 tokens/s, Avg generation throughput: 2938.3 tokens/s, Running: 50 reqs, Waiting: 0 reqs, GPU KV cache usage: 86.5%, Prefix cache hit rate: 56.1%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2-7B-Instruct --dataset-name random \
--host localhost \
--num-prompts 75 \
--random-input-len 1024 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     75        
Benchmark duration (s):                  35.99     
Total input tokens:                      76558     
Total generated tokens:                  76719     
Request throughput (req/s):              2.08      
Output token throughput (tok/s):         2131.91   
Total Token throughput (tok/s):          4259.34   
---------------Time to First Token----------------
Mean TTFT (ms):                          1140.94   
Median TTFT (ms):                        791.30    
P99 TTFT (ms):                           2675.82   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          24.02     
Median TPOT (ms):                        20.90     
P99 TPOT (ms):                           32.32     
---------------Inter-token Latency----------------
Mean ITL (ms):                           24.03     
Median ITL (ms):                         18.67     
P99 ITL (ms):                            147.82    
==================================================

(APIServer pid=113) INFO 09-06 21:07:28 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 3388.2 tokens/s, Running: 57 reqs, Waiting: 18 reqs, GPU KV cache usage: 99.4%, Prefix cache hit rate: 45.9%
(APIServer pid=113) INFO 09-06 21:07:38 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 2093.1 tokens/s, Running: 34 reqs, Waiting: 6 reqs, GPU KV cache usage: 62.9%, Prefix cache hit rate: 45.7%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2-7B-Instruct --dataset-name random \
--host localhost \
--num-prompts 100 \
--random-input-len 1024 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     100       
Benchmark duration (s):                  47.81     
Total input tokens:                      102140    
Total generated tokens:                  101578    
Request throughput (req/s):              2.09      
Output token throughput (tok/s):         2124.52   
Total Token throughput (tok/s):          4260.80   
---------------Time to First Token----------------
Mean TTFT (ms):                          6772.41   
Median TTFT (ms):                        4105.02   
P99 TTFT (ms):                           30414.46  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          28.69     
Median TPOT (ms):                        24.73     
P99 TPOT (ms):                           40.23     
---------------Inter-token Latency----------------
Mean ITL (ms):                           28.31     
Median ITL (ms):                         18.11     
P99 ITL (ms):                            153.69    
==================================================

(APIServer pid=1671) INFO 09-04 01:23:14 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 3455.7 tokens/s, Running: 55 reqs, Waiting: 45 reqs, GPU KV cache usage: 99.1%, Prefix cache hit rate: 42.7%
(APIServer pid=1671) INFO 09-04 01:23:24 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 1816.8 tokens/s, Running: 37 reqs, Waiting: 18 reqs, GPU KV cache usage: 58.7%, Prefix cache hit rate: 43.0%
(APIServer pid=1671) INFO 09-04 01:23:34 [loggers.py:123] Engine 000: Avg prompt throughput: 1125.9 tokens/s, Avg generation throughput: 2226.3 tokens/s, Running: 40 reqs, Waiting: 0 reqs, GPU KV cache usage: 71.3%, Prefix cache hit rate: 42.5%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2-7B-Instruct --dataset-name random \
--host localhost \
--num-prompts 200 \
--random-input-len 1024 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     200       
Benchmark duration (s):                  97.25     
Total input tokens:                      204007    
Total generated tokens:                  200083    
Request throughput (req/s):              2.06      
Output token throughput (tok/s):         2057.51   
Total Token throughput (tok/s):          4155.36   
---------------Time to First Token----------------
Mean TTFT (ms):                          28904.67  
Median TTFT (ms):                        30624.81  
P99 TTFT (ms):                           60770.93  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          29.87     
Median TPOT (ms):                        26.43     
P99 TPOT (ms):                           48.82     
---------------Inter-token Latency----------------
Mean ITL (ms):                           29.54     
Median ITL (ms):                         18.90     
P99 ITL (ms):                            154.97    
==================================================

(APIServer pid=1671) INFO 09-04 01:19:44 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 2641.1 tokens/s, Running: 55 reqs, Waiting: 82 reqs, GPU KV cache usage: 96.7%, Prefix cache hit rate: 42.2%
(APIServer pid=1671) INFO 09-04 01:19:54 [loggers.py:123] Engine 000: Avg prompt throughput: 2028.4 tokens/s, Avg generation throughput: 2172.1 tokens/s, Running: 56 reqs, Waiting: 59 reqs, GPU KV cache usage: 100.0%, Prefix cache hit rate: 41.7%
(APIServer pid=1671) INFO 09-04 01:20:04 [loggers.py:123] Engine 000: Avg prompt throughput: 6006.1 tokens/s, Avg generation throughput: 1924.3 tokens/s, Running: 72 reqs, Waiting: 8 reqs, GPU KV cache usage: 99.9%, Prefix cache hit rate: 40.8%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2-7B-Instruct --dataset-name random \
--host localhost \
--num-prompts 500 \
--random-input-len 1024 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     500       
Benchmark duration (s):                  231.04    
Total input tokens:                      510414    
Total generated tokens:                  501465    
Request throughput (req/s):              2.16      
Output token throughput (tok/s):         2170.51   
Total Token throughput (tok/s):          4379.75   
---------------Time to First Token----------------
Mean TTFT (ms):                          95836.65  
Median TTFT (ms):                        88718.78  
P99 TTFT (ms):                           199001.09 
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          31.56     
Median TPOT (ms):                        26.23     
P99 TPOT (ms):                           49.04     
---------------Inter-token Latency----------------
Mean ITL (ms):                           30.58     
Median ITL (ms):                         19.00     
P99 ITL (ms):                            156.25    
==================================================

(APIServer pid=1671) INFO 09-04 01:13:04 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 2684.9 tokens/s, Running: 49 reqs, Waiting: 74 reqs, GPU KV cache usage: 96.1%, Prefix cache hit rate: 43.3%
(APIServer pid=1671) INFO 09-04 01:13:14 [loggers.py:123] Engine 000: Avg prompt throughput: 4464.8 tokens/s, Avg generation throughput: 1210.7 tokens/s, Running: 65 reqs, Waiting: 7 reqs, GPU KV cache usage: 78.5%, Prefix cache hit rate: 43.8%
(APIServer pid=1671) INFO 09-04 01:13:24 [loggers.py:123] Engine 000: Avg prompt throughput: 818.7 tokens/s, Avg generation throughput: 3266.4 tokens/s, Running: 58 reqs, Waiting: 10 reqs, GPU KV cache usage: 98.8%, Prefix cache hit rate: 43.9%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2-7B-Instruct --dataset-name random \
--host localhost \
--num-prompts 1000 \
--random-input-len 1024 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     894       
Benchmark duration (s):                  409.78    
Total input tokens:                      913314    
Total generated tokens:                  894823    
Request throughput (req/s):              2.18      
Output token throughput (tok/s):         2183.69   
Total Token throughput (tok/s):          4412.51   
---------------Time to First Token----------------
Mean TTFT (ms):                          186280.69 
Median TTFT (ms):                        185516.70 
P99 TTFT (ms):                           393555.87 
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          32.41     
Median TPOT (ms):                        26.45     
P99 TPOT (ms):                           52.92     
---------------Inter-token Latency----------------
Mean ITL (ms):                           31.46     
Median ITL (ms):                         19.09     
P99 ITL (ms):                            159.96    
==================================================

(APIServer pid=2583) INFO 09-06 21:41:33 [loggers.py:123] Engine 000: Avg prompt throughput: 3680.2 tokens/s, Avg generation throughput: 2920.9 tokens/s, Running: 65 reqs, Waiting: 829 reqs, GPU KV cache usage: 98.6%, Prefix cache hit rate: 37.9%
(APIServer pid=2583) INFO 09-06 21:41:43 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 2981.7 tokens/s, Running: 47 reqs, Waiting: 846 reqs, GPU KV cache usage: 98.4%, Prefix cache hit rate: 44.3%
(APIServer pid=2583) INFO 09-06 21:41:53 [loggers.py:123] Engine 000: Avg prompt throughput: 3268.3 tokens/s, Avg generation throughput: 1152.7 tokens/s, Running: 69 reqs, Waiting: 777 reqs, GPU KV cache usage: 99.1%, Prefix cache hit rate: 40.8%
```

### <a name="toc_7"></a>✍️ Test 5: 1 GPU pod hosts 7B model (Available KV cache memory: 24.83 GiB)

- Startup.log: [vllm-7B-1gpuA10080GB-0.5GRAM.log](log/vllm-7B-1gpuA10080GB-0.5GRAM.log)

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2-7B-Instruct --dataset-name random \
--host localhost \
--num-prompts 1 \
--random-input-len 1024 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     1         
Benchmark duration (s):                  12.05     
Total input tokens:                      1024      
Total generated tokens:                  1024      
Request throughput (req/s):              0.08      
Output token throughput (tok/s):         84.95     
Total Token throughput (tok/s):          169.91    
---------------Time to First Token----------------
Mean TTFT (ms):                          36.23     
Median TTFT (ms):                        36.23     
P99 TTFT (ms):                           36.23     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          11.75     
Median TPOT (ms):                        11.75     
P99 TPOT (ms):                           11.75     
---------------Inter-token Latency----------------
Mean ITL (ms):                           11.75     
Median ITL (ms):                         11.64     
P99 ITL (ms):                            15.43     
==================================================

(APIServer pid=136) INFO 09-09 01:43:06 [loggers.py:123] Engine 000: Avg prompt throughput: 58.7 tokens/s, Avg generation throughput: 85.2 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.4%, Prefix cache hit rate: 56.1%
(APIServer pid=136) INFO 09-09 01:43:16 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 38.8 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 56.1%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2-7B-Instruct --dataset-name random \
--host localhost \
--num-prompts 10 \
--random-input-len 1024 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     10        
Benchmark duration (s):                  14.30     
Total input tokens:                      10239     
Total generated tokens:                  10240     
Request throughput (req/s):              0.70      
Output token throughput (tok/s):         715.92    
Total Token throughput (tok/s):          1431.76   
---------------Time to First Token----------------
Mean TTFT (ms):                          488.52    
Median TTFT (ms):                        511.95    
P99 TTFT (ms):                           723.97    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          13.47     
Median TPOT (ms):                        13.45     
P99 TPOT (ms):                           13.79     
---------------Inter-token Latency----------------
Mean ITL (ms):                           13.48     
Median ITL (ms):                         13.09     
P99 ITL (ms):                            19.63     
==================================================

(APIServer pid=118) INFO 09-09 02:24:57 [loggers.py:123] Engine 000: Avg prompt throughput: 1023.7 tokens/s, Avg generation throughput: 93.8 tokens/s, Running: 10 reqs, Waiting: 0 reqs, GPU KV cache usage: 2.3%, Prefix cache hit rate: 8.9%
(APIServer pid=118) INFO 09-09 02:25:07 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 757.9 tokens/s, Running: 10 reqs, Waiting: 0 reqs, GPU KV cache usage: 3.9%, Prefix cache hit rate: 8.9%
(APIServer pid=118) INFO 09-09 02:25:17 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 248.9 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 8.9%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2-7B-Instruct --dataset-name random \
--host localhost \
--num-prompts 20 \
--random-input-len 1024 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     20        
Benchmark duration (s):                  14.70     
Total input tokens:                      20460     
Total generated tokens:                  20399     
Request throughput (req/s):              1.36      
Output token throughput (tok/s):         1387.31   
Total Token throughput (tok/s):          2778.76   
---------------Time to First Token----------------
Mean TTFT (ms):                          195.24    
Median TTFT (ms):                        199.99    
P99 TTFT (ms):                           239.46    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          14.12     
Median TPOT (ms):                        14.12     
P99 TPOT (ms):                           14.15     
---------------Inter-token Latency----------------
Mean ITL (ms):                           14.12     
Median ITL (ms):                         13.97     
P99 ITL (ms):                            20.04     
==================================================

(APIServer pid=136) INFO 09-09 01:38:28 [loggers.py:123] Engine 000: Avg prompt throughput: 2045.5 tokens/s, Avg generation throughput: 1046.3 tokens/s, Running: 20 reqs, Waiting: 0 reqs, GPU KV cache usage: 6.6%, Prefix cache hit rate: 50.7%
(APIServer pid=136) INFO 09-09 01:38:38 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 1016.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 50.7%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2-7B-Instruct --dataset-name random \
--host localhost \
--num-prompts 30 \
--random-input-len 1024 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     30        
Benchmark duration (s):                  15.56     
Total input tokens:                      30699     
Total generated tokens:                  30615     
Request throughput (req/s):              1.93      
Output token throughput (tok/s):         1967.90   
Total Token throughput (tok/s):          3941.19   
---------------Time to First Token----------------
Mean TTFT (ms):                          241.74    
Median TTFT (ms):                        240.32    
P99 TTFT (ms):                           274.05    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          14.93     
Median TPOT (ms):                        14.93     
P99 TPOT (ms):                           14.95     
---------------Inter-token Latency----------------
Mean ITL (ms):                           14.93     
Median ITL (ms):                         14.80     
P99 ITL (ms):                            21.42     
==================================================

(APIServer pid=136) INFO 09-09 01:36:48 [loggers.py:123] Engine 000: Avg prompt throughput: 3069.1 tokens/s, Avg generation throughput: 1175.4 tokens/s, Running: 30 reqs, Waiting: 0 reqs, GPU KV cache usage: 9.1%, Prefix cache hit rate: 38.5%
(APIServer pid=136) INFO 09-09 01:36:58 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 1921.4 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 38.5%
(APIServer pid=136) INFO 09-09 01:37:08 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 38.5%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2-7B-Instruct --dataset-name random \
--host localhost \
--num-prompts 50 \
--random-input-len 1024 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     50        
Benchmark duration (s):                  21.70     
Total input tokens:                      50977     
Total generated tokens:                  51119     
Request throughput (req/s):              2.30      
Output token throughput (tok/s):         2356.12   
Total Token throughput (tok/s):          4705.70   
---------------Time to First Token----------------
Mean TTFT (ms):                          2580.51   
Median TTFT (ms):                        2490.45   
P99 TTFT (ms):                           4084.26   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          18.47     
Median TPOT (ms):                        18.57     
P99 TPOT (ms):                           19.47     
---------------Inter-token Latency----------------
Mean ITL (ms):                           18.49     
Median ITL (ms):                         17.04     
P99 ITL (ms):                            136.93    
==================================================

(APIServer pid=136) INFO 09-09 01:31:28 [loggers.py:123] Engine 000: Avg prompt throughput: 1772.9 tokens/s, Avg generation throughput: 657.5 tokens/s, Running: 50 reqs, Waiting: 0 reqs, GPU KV cache usage: 14.9%, Prefix cache hit rate: 1.9%
(APIServer pid=136) INFO 09-09 01:31:38 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 2863.4 tokens/s, Running: 50 reqs, Waiting: 0 reqs, GPU KV cache usage: 21.1%, Prefix cache hit rate: 1.9%
(APIServer pid=136) INFO 09-09 01:31:48 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 457.8 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 1.9%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2-7B-Instruct --dataset-name random \
--host localhost \
--num-prompts 75 \
--random-input-len 1024 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     75        
Benchmark duration (s):                  25.64     
Total input tokens:                      76558     
Total generated tokens:                  76697     
Request throughput (req/s):              2.92      
Output token throughput (tok/s):         2991.02   
Total Token throughput (tok/s):          5976.62   
---------------Time to First Token----------------
Mean TTFT (ms):                          2598.85   
Median TTFT (ms):                        2543.58   
P99 TTFT (ms):                           5063.67   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          22.21     
Median TPOT (ms):                        22.27     
P99 TPOT (ms):                           24.15     
---------------Inter-token Latency----------------
Mean ITL (ms):                           22.21     
Median ITL (ms):                         19.98     
P99 ITL (ms):                            145.71    
==================================================

(APIServer pid=118) INFO 09-09 02:26:47 [loggers.py:123] Engine 000: Avg prompt throughput: 7650.1 tokens/s, Avg generation throughput: 974.6 tokens/s, Running: 75 reqs, Waiting: 0 reqs, GPU KV cache usage: 18.6%, Prefix cache hit rate: 13.6%
(APIServer pid=118) INFO 09-09 02:26:57 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 3792.5 tokens/s, Running: 75 reqs, Waiting: 0 reqs, GPU KV cache usage: 26.8%, Prefix cache hit rate: 13.6%
(APIServer pid=118) INFO 09-09 02:27:07 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 2923.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 13.6%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2-7B-Instruct --dataset-name random \
--host localhost \
--num-prompts 100 \
--random-input-len 1024 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     100       
Benchmark duration (s):                  30.31     
Total input tokens:                      102140    
Total generated tokens:                  101590    
Request throughput (req/s):              3.30      
Output token throughput (tok/s):         3352.15   
Total Token throughput (tok/s):          6722.45   
---------------Time to First Token----------------
Mean TTFT (ms):                          3983.10   
Median TTFT (ms):                        3896.62   
P99 TTFT (ms):                           7664.70   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          25.26     
Median TPOT (ms):                        25.36     
P99 TPOT (ms):                           28.23     
---------------Inter-token Latency----------------
Mean ITL (ms):                           25.27     
Median ITL (ms):                         21.76     
P99 ITL (ms):                            154.14    
==================================================

(APIServer pid=2739) INFO 09-04 01:33:00 [loggers.py:123] Engine 000: Avg prompt throughput: 10204.0 tokens/s, Avg generation throughput: 1209.6 tokens/s, Running: 100 reqs, Waiting: 0 reqs, GPU KV cache usage: 24.7%, Prefix cache hit rate: 1.0%
(APIServer pid=2739) INFO 09-04 01:33:10 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 4691.9 tokens/s, Running: 99 reqs, Waiting: 0 reqs, GPU KV cache usage: 34.5%, Prefix cache hit rate: 1.0%
(APIServer pid=2739) INFO 09-04 01:33:20 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 4142.2 tokens/s, Running: 66 reqs, Waiting: 0 reqs, GPU KV cache usage: 28.9%, Prefix cache hit rate: 1.0%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2-7B-Instruct --dataset-name random \
--host localhost \
--num-prompts 200 \
--random-input-len 1024 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     200       
Benchmark duration (s):                  45.11     
Total input tokens:                      204007    
Total generated tokens:                  200419    
Request throughput (req/s):              4.43      
Output token throughput (tok/s):         4442.73   
Total Token throughput (tok/s):          8964.99   
---------------Time to First Token----------------
Mean TTFT (ms):                          3158.70   
Median TTFT (ms):                        1179.16   
P99 TTFT (ms):                           10167.58  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          40.34     
Median TPOT (ms):                        41.97     
P99 TPOT (ms):                           47.17     
---------------Inter-token Latency----------------
Mean ITL (ms):                           40.20     
Median ITL (ms):                         34.84     
P99 ITL (ms):                            184.58    
==================================================

(APIServer pid=2739) INFO 09-04 01:34:50 [loggers.py:123] Engine 000: Avg prompt throughput: 2836.3 tokens/s, Avg generation throughput: 4789.1 tokens/s, Running: 196 reqs, Waiting: 0 reqs, GPU KV cache usage: 54.8%, Prefix cache hit rate: 33.3%
(APIServer pid=2739) INFO 09-04 01:35:00 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 5737.5 tokens/s, Running: 195 reqs, Waiting: 0 reqs, GPU KV cache usage: 66.8%, Prefix cache hit rate: 33.3%
(APIServer pid=2739) INFO 09-04 01:35:10 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 5380.9 tokens/s, Running: 195 reqs, Waiting: 0 reqs, GPU KV cache usage: 78.4%, Prefix cache hit rate: 33.3%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2-7B-Instruct --dataset-name random \
--host localhost \
--num-prompts 500 \
--random-input-len 1024 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     500       
Benchmark duration (s):                  119.23    
Total input tokens:                      510414    
Total generated tokens:                  502621    
Request throughput (req/s):              4.19      
Output token throughput (tok/s):         4215.38   
Total Token throughput (tok/s):          8496.12   
---------------Time to First Token----------------
Mean TTFT (ms):                          34943.37  
Median TTFT (ms):                        10041.30  
P99 TTFT (ms):                           79671.05  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          52.41     
Median TPOT (ms):                        52.99     
P99 TPOT (ms):                           73.34     
---------------Inter-token Latency----------------
Mean ITL (ms):                           51.75     
Median ITL (ms):                         39.53     
P99 ITL (ms):                            231.33    
==================================================

(APIServer pid=2739) INFO 09-04 01:39:31 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 6241.3 tokens/s, Running: 256 reqs, Waiting: 238 reqs, GPU KV cache usage: 96.4%, Prefix cache hit rate: 47.2%
(APIServer pid=2739) INFO 09-04 01:39:41 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 5749.3 tokens/s, Running: 233 reqs, Waiting: 261 reqs, GPU KV cache usage: 99.8%, Prefix cache hit rate: 46.1%
(APIServer pid=2739) INFO 09-04 01:39:51 [loggers.py:123] Engine 000: Avg prompt throughput: 2966.2 tokens/s, Avg generation throughput: 1134.0 tokens/s, Running: 178 reqs, Waiting: 208 reqs, GPU KV cache usage: 70.0%, Prefix cache hit rate: 42.3%
(APIServer pid=2739) INFO 09-04 01:40:01 [loggers.py:123] Engine 000: Avg prompt throughput: 9187.6 tokens/s, Avg generation throughput: 829.2 tokens/s, Running: 142 reqs, Waiting: 118 reqs, GPU KV cache usage: 36.4%, Prefix cache hit rate: 38.5%
(APIServer pid=2739) INFO 09-04 01:40:11 [loggers.py:123] Engine 000: Avg prompt throughput: 9687.7 tokens/s, Avg generation throughput: 968.4 tokens/s, Running: 230 reqs, Waiting: 23 reqs, GPU KV cache usage: 56.8%, Prefix cache hit rate: 35.2%
(APIServer pid=2739) INFO 09-04 01:40:21 [loggers.py:123] Engine 000: Avg prompt throughput: 2441.7 tokens/s, Avg generation throughput: 4863.9 tokens/s, Running: 238 reqs, Waiting: 0 reqs, GPU KV cache usage: 66.1%, Prefix cache hit rate: 34.5%
(APIServer pid=2739) INFO 09-04 01:40:31 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 6297.4 tokens/s, Running: 236 reqs, Waiting: 0 reqs, GPU KV cache usage: 78.8%, Prefix cache hit rate: 34.5%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2-7B-Instruct --dataset-name random \
--host localhost \
--num-prompts 1000 \
--random-input-len 1024 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     924       
Benchmark duration (s):                  260.86    
Total input tokens:                      944227    
Total generated tokens:                  926930    
Request throughput (req/s):              3.54      
Output token throughput (tok/s):         3553.40   
Total Token throughput (tok/s):          7173.11   
---------------Time to First Token----------------
Mean TTFT (ms):                          104453.69 
Median TTFT (ms):                        90413.12  
P99 TTFT (ms):                           229208.82 
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          63.32     
Median TPOT (ms):                        66.32     
P99 TPOT (ms):                           126.77    
---------------Inter-token Latency----------------
Mean ITL (ms):                           61.78     
Median ITL (ms):                         39.66     
P99 ITL (ms):                            249.71    
==================================================

(APIServer pid=2739) INFO 09-04 01:43:01 [loggers.py:123] Engine 000: Avg prompt throughput: 102.3 tokens/s, Avg generation throughput: 6190.1 tokens/s, Running: 256 reqs, Waiting: 663 reqs, GPU KV cache usage: 90.2%, Prefix cache hit rate: 28.5%
(APIServer pid=2739) INFO 09-04 01:43:11 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 6051.8 tokens/s, Running: 246 reqs, Waiting: 673 reqs, GPU KV cache usage: 99.7%, Prefix cache hit rate: 29.3%
(APIServer pid=2739) INFO 09-04 01:43:21 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 2958.8 tokens/s, Running: 214 reqs, Waiting: 662 reqs, GPU KV cache usage: 90.7%, Prefix cache hit rate: 30.7%
```

### <a name="toc_8"></a>✍️ Test 6: 2 GPU pods host 32B model in the same node (Available KV cache memory in each GPU: 8.22 GiB)

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2.5-32B --dataset-name random \
--host localhost \
--num-prompts 1 \
--random-input-len 1024 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     1         
Benchmark duration (s):                  29.94     
Total input tokens:                      1024      
Total generated tokens:                  1024      
Request throughput (req/s):              0.03      
Output token throughput (tok/s):         34.20     
Total Token throughput (tok/s):          68.40     
---------------Time to First Token----------------
Mean TTFT (ms):                          61.26     
Median TTFT (ms):                        61.26     
P99 TTFT (ms):                           61.26     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          29.21     
Median TPOT (ms):                        29.21     
P99 TPOT (ms):                           29.21     
---------------Inter-token Latency----------------
Mean ITL (ms):                           29.21     
Median ITL (ms):                         29.10     
P99 ITL (ms):                            32.15     
==================================================

(APIServer pid=126) INFO 09-07 00:41:06 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 34.3 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 2.3%, Prefix cache hit rate: 49.2%
(APIServer pid=126) INFO 09-07 00:41:16 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 34.2 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 2.8%, Prefix cache hit rate: 49.2%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2.5-32B --dataset-name random \
--host localhost \
--num-prompts 10 \
--random-input-len 1024 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     10        
Benchmark duration (s):                  42.22     
Total input tokens:                      10239     
Total generated tokens:                  10240     
Request throughput (req/s):              0.24      
Output token throughput (tok/s):         242.57    
Total Token throughput (tok/s):          485.11    
---------------Time to First Token----------------
Mean TTFT (ms):                          1853.66   
Median TTFT (ms):                        1965.27   
P99 TTFT (ms):                           2940.46   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          39.38     
Median TPOT (ms):                        39.27     
P99 TPOT (ms):                           40.92     
---------------Inter-token Latency----------------
Mean ITL (ms):                           39.38     
Median ITL (ms):                         38.12     
P99 ITL (ms):                            44.38     
==================================================

(APIServer pid=126) INFO 09-07 00:42:36 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 262.0 tokens/s, Running: 10 reqs, Waiting: 0 reqs, GPU KV cache usage: 20.2%, Prefix cache hit rate: 22.7%
(APIServer pid=126) INFO 09-07 00:42:46 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 262.0 tokens/s, Running: 10 reqs, Waiting: 0 reqs, GPU KV cache usage: 24.0%, Prefix cache hit rate: 22.7%
(APIServer pid=126) INFO 09-07 00:42:56 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 259.0 tokens/s, Running: 10 reqs, Waiting: 0 reqs, GPU KV cache usage: 27.8%, Prefix cache hit rate: 22.7%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2.5-32B --dataset-name random \
--host localhost \
--num-prompts 15 \
--random-input-len 1024 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     15        
Benchmark duration (s):                  40.58     
Total input tokens:                      15340     
Total generated tokens:                  15360     
Request throughput (req/s):              0.37      
Output token throughput (tok/s):         378.49    
Total Token throughput (tok/s):          756.49    
---------------Time to First Token----------------
Mean TTFT (ms):                          213.45    
Median TTFT (ms):                        218.21    
P99 TTFT (ms):                           236.19    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          39.44     
Median TPOT (ms):                        39.43     
P99 TPOT (ms):                           39.49     
---------------Inter-token Latency----------------
Mean ITL (ms):                           39.44     
Median ITL (ms):                         39.15     
P99 ITL (ms):                            47.49     
==================================================

(APIServer pid=126) INFO 09-07 00:56:57 [loggers.py:123] Engine 000: Avg prompt throughput: 1533.6 tokens/s, Avg generation throughput: 258.8 tokens/s, Running: 15 reqs, Waiting: 0 reqs, GPU KV cache usage: 26.7%, Prefix cache hit rate: 46.6%
(APIServer pid=126) INFO 09-07 00:57:07 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 386.9 tokens/s, Running: 15 reqs, Waiting: 0 reqs, GPU KV cache usage: 32.5%, Prefix cache hit rate: 46.6%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2.5-32B --dataset-name random \
--host localhost \
--num-prompts 20 \
--random-input-len 1024 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     20        
Benchmark duration (s):                  44.11     
Total input tokens:                      20460     
Total generated tokens:                  20480     
Request throughput (req/s):              0.45      
Output token throughput (tok/s):         464.29    
Total Token throughput (tok/s):          928.13    
---------------Time to First Token----------------
Mean TTFT (ms):                          284.31    
Median TTFT (ms):                        288.20    
P99 TTFT (ms):                           310.65    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          42.81     
Median TPOT (ms):                        42.81     
P99 TPOT (ms):                           42.87     
---------------Inter-token Latency----------------
Mean ITL (ms):                           42.81     
Median ITL (ms):                         42.62     
P99 ITL (ms):                            50.33     
==================================================

(APIServer pid=126) INFO 09-07 00:55:07 [loggers.py:123] Engine 000: Avg prompt throughput: 2045.5 tokens/s, Avg generation throughput: 428.5 tokens/s, Running: 20 reqs, Waiting: 0 reqs, GPU KV cache usage: 37.1%, Prefix cache hit rate: 45.4%
(APIServer pid=126) INFO 09-07 00:55:17 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 471.8 tokens/s, Running: 20 reqs, Waiting: 0 reqs, GPU KV cache usage: 43.8%, Prefix cache hit rate: 45.9%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2.5-32B --dataset-name random \
--host localhost \
--num-prompts 30 \
--random-input-len 1024 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     30        
Benchmark duration (s):                  56.86     
Total input tokens:                      30699     
Total generated tokens:                  30720     
Request throughput (req/s):              0.53      
Output token throughput (tok/s):         540.32    
Total Token throughput (tok/s):          1080.27   
---------------Time to First Token----------------
Mean TTFT (ms):                          5368.61   
Median TTFT (ms):                        5441.43   
P99 TTFT (ms):                           9262.94   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          50.04     
Median TPOT (ms):                        49.98     
P99 TPOT (ms):                           54.35     
---------------Inter-token Latency----------------
Mean ITL (ms):                           50.04     
Median ITL (ms):                         46.29     
P99 ITL (ms):                            57.31     
==================================================

(APIServer pid=126) INFO 09-07 00:52:17 [loggers.py:123] Engine 000: Avg prompt throughput: 1023.8 tokens/s, Avg generation throughput: 23.0 tokens/s, Running: 11 reqs, Waiting: 19 reqs, GPU KV cache usage: 16.9%, Prefix cache hit rate: 46.0%
(APIServer pid=126) INFO 09-07 00:52:27 [loggers.py:123] Engine 000: Avg prompt throughput: 2045.2 tokens/s, Avg generation throughput: 336.1 tokens/s, Running: 30 reqs, Waiting: 0 reqs, GPU KV cache usage: 51.0%, Prefix cache hit rate: 45.8%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2.5-32B --dataset-name random \
--host localhost \
--num-prompts 50 \
--random-input-len 1024 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     50        
Benchmark duration (s):                  100.75    
Total input tokens:                      50977     
Total generated tokens:                  51200     
Request throughput (req/s):              0.50      
Output token throughput (tok/s):         508.18    
Total Token throughput (tok/s):          1014.14   
---------------Time to First Token----------------
Mean TTFT (ms):                          6119.18   
Median TTFT (ms):                        5903.04   
P99 TTFT (ms):                           12946.41  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          68.64     
Median TPOT (ms):                        63.81     
P99 TPOT (ms):                           85.33     
---------------Inter-token Latency----------------
Mean ITL (ms):                           68.64     
Median ITL (ms):                         51.58     
P99 ITL (ms):                            632.22    
==================================================

(APIServer pid=126) INFO 09-07 00:45:07 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 899.6 tokens/s, Running: 45 reqs, Waiting: 5 reqs, GPU KV cache usage: 98.5%, Prefix cache hit rate: 40.8%
(APIServer pid=126) INFO 09-07 00:45:17 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 813.3 tokens/s, Running: 40 reqs, Waiting: 10 reqs, GPU KV cache usage: 99.2%, Prefix cache hit rate: 44.6%
(APIServer pid=126) INFO 09-07 00:45:27 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 740.3 tokens/s, Running: 36 reqs, Waiting: 14 reqs, GPU KV cache usage: 99.9%, Prefix cache hit rate: 45.6%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2.5-32B --dataset-name random \
--host localhost \
--num-prompts 100 \
--random-input-len 1024 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     100       
Benchmark duration (s):                  193.49    
Total input tokens:                      102140    
Total generated tokens:                  102400    
Request throughput (req/s):              0.52      
Output token throughput (tok/s):         529.23    
Total Token throughput (tok/s):          1057.13   
---------------Time to First Token----------------
Mean TTFT (ms):                          44143.73  
Median TTFT (ms):                        11137.92  
P99 TTFT (ms):                           151305.50 
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          80.12     
Median TPOT (ms):                        68.97     
P99 TPOT (ms):                           137.35    
---------------Inter-token Latency----------------
Mean ITL (ms):                           80.12     
Median ITL (ms):                         51.73     
P99 ITL (ms):                            640.28    
==================================================

(APIServer pid=126) INFO 09-07 00:47:47 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 815.0 tokens/s, Running: 40 reqs, Waiting: 60 reqs, GPU KV cache usage: 99.1%, Prefix cache hit rate: 46.3%
(APIServer pid=126) INFO 09-07 00:47:57 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 741.4 tokens/s, Running: 36 reqs, Waiting: 64 reqs, GPU KV cache usage: 99.8%, Prefix cache hit rate: 46.5%
(APIServer pid=126) INFO 09-07 00:48:07 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 610.6 tokens/s, Running: 32 reqs, Waiting: 66 reqs, GPU KV cache usage: 96.9%, Prefix cache hit rate: 46.9%
```

### <a name="toc_9"></a>✍️ Test 7: 1 GPU pod hosts 32B model (Available KV cache memory: 9.70 GiB)

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2.5-32B --dataset-name random \
--host localhost \
--num-prompts 10 \
--random-input-len 1024 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     10        
Benchmark duration (s):                  54.96     
Total input tokens:                      10239     
Total generated tokens:                  10240     
Request throughput (req/s):              0.18      
Output token throughput (tok/s):         186.31    
Total Token throughput (tok/s):          372.60    
---------------Time to First Token----------------
Mean TTFT (ms):                          2017.23   
Median TTFT (ms):                        2187.82   
P99 TTFT (ms):                           2830.05   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          51.64     
Median TPOT (ms):                        51.48     
P99 TPOT (ms):                           53.04     
---------------Inter-token Latency----------------
Mean ITL (ms):                           51.64     
Median ITL (ms):                         50.97     
P99 ITL (ms):                            54.66     
==================================================

(APIServer pid=133) INFO 09-09 20:52:06 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 196.0 tokens/s, Running: 10 reqs, Waiting: 0 reqs, GPU KV cache usage: 39.9%, Prefix cache hit rate: 8.9%
(APIServer pid=133) INFO 09-09 20:52:16 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 195.0 tokens/s, Running: 10 reqs, Waiting: 0 reqs, GPU KV cache usage: 44.7%, Prefix cache hit rate: 8.9%
(APIServer pid=133) INFO 09-09 20:52:26 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 194.0 tokens/s, Running: 10 reqs, Waiting: 0 reqs, GPU KV cache usage: 49.6%, Prefix cache hit rate: 8.9%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2.5-32B --dataset-name random \
--host localhost \
--num-prompts 50 \
--random-input-len 1024 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     50        
Benchmark duration (s):                  160.30    
Total input tokens:                      50977     
Total generated tokens:                  51200     
Request throughput (req/s):              0.31      
Output token throughput (tok/s):         319.41    
Total Token throughput (tok/s):          637.42    
---------------Time to First Token----------------
Mean TTFT (ms):                          21486.84  
Median TTFT (ms):                        5747.70   
P99 TTFT (ms):                           77583.49  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          79.95     
Median TPOT (ms):                        64.25     
P99 TPOT (ms):                           124.67    
---------------Inter-token Latency----------------
Mean ITL (ms):                           79.95     
Median ITL (ms):                         55.95     
P99 ITL (ms):                            612.25    
==================================================

(APIServer pid=133) INFO 09-09 20:57:36 [loggers.py:123] Engine 000: Avg prompt throughput: 1841.1 tokens/s, Avg generation throughput: 21.2 tokens/s, Running: 19 reqs, Waiting: 31 reqs, GPU KV cache usage: 49.2%, Prefix cache hit rate: 35.0%
(APIServer pid=133) INFO 09-09 20:57:46 [loggers.py:123] Engine 000: Avg prompt throughput: 2042.5 tokens/s, Avg generation throughput: 305.5 tokens/s, Running: 35 reqs, Waiting: 15 reqs, GPU KV cache usage: 98.3%, Prefix cache hit rate: 38.0%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2.5-32B --dataset-name random \
--host localhost \
--num-prompts 75 \
--random-input-len 1024 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     75        
Benchmark duration (s):                  226.03    
Total input tokens:                      76558     
Total generated tokens:                  76800     
Request throughput (req/s):              0.33      
Output token throughput (tok/s):         339.77    
Total Token throughput (tok/s):          678.47    
---------------Time to First Token----------------
Mean TTFT (ms):                          61850.53  
Median TTFT (ms):                        10445.43  
P99 TTFT (ms):                           150225.82 
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          76.73     
Median TPOT (ms):                        66.38     
P99 TPOT (ms):                           130.54    
---------------Inter-token Latency----------------
Mean ITL (ms):                           76.73     
Median ITL (ms):                         55.91     
P99 ITL (ms):                            619.24    
==================================================

(APIServer pid=133) INFO 09-09 21:03:56 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 411.0 tokens/s, Running: 21 reqs, Waiting: 54 reqs, GPU KV cache usage: 95.5%, Prefix cache hit rate: 48.3%
(APIServer pid=133) INFO 09-09 21:04:06 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 364.4 tokens/s, Running: 20 reqs, Waiting: 55 reqs, GPU KV cache usage: 99.8%, Prefix cache hit rate: 48.2%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2.5-32B --dataset-name random \
--host localhost \
--num-prompts 100 \
--random-input-len 1024 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     100       
Benchmark duration (s):                  282.30    
Total input tokens:                      102140    
Total generated tokens:                  102400    
Request throughput (req/s):              0.35      
Output token throughput (tok/s):         362.73    
Total Token throughput (tok/s):          724.55    
---------------Time to First Token----------------
Mean TTFT (ms):                          97867.77  
Median TTFT (ms):                        98670.42  
P99 TTFT (ms):                           223160.11 
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          77.73     
Median TPOT (ms):                        69.39     
P99 TPOT (ms):                           129.68    
---------------Inter-token Latency----------------
Mean ITL (ms):                           77.74     
Median ITL (ms):                         55.94     
P99 ITL (ms):                            623.64    
==================================================

(APIServer pid=1358) INFO 09-09 21:23:33 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 469.3 tokens/s, Running: 25 reqs, Waiting: 75 reqs, GPU KV cache usage: 99.4%, Prefix cache hit rate: 46.6%
(APIServer pid=1358) INFO 09-09 21:23:43 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 421.6 tokens/s, Running: 22 reqs, Waiting: 78 reqs, GPU KV cache usage: 97.4%, Prefix cache hit rate: 48.0%
```

### <a name="toc_10"></a>✍️ Test 8: 2 GPU pods host 72B model in the same node (Available KV cache memory in each GPU: 6.77 GiB)

<img width="700" height="404" alt="image" src="https://github.com/user-attachments/assets/87c5c3a3-92fa-473c-ae1e-f68c49c8c723" />

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2-72B --dataset-name random \
--host localhost \
--num-prompts 1 \
--random-input-len 1024 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     1         
Benchmark duration (s):                  59.43     
Total input tokens:                      1024      
Total generated tokens:                  1024      
Request throughput (req/s):              0.02      
Output token throughput (tok/s):         17.23     
Total Token throughput (tok/s):          34.46     
---------------Time to First Token----------------
Mean TTFT (ms):                          93.18     
Median TTFT (ms):                        93.18     
P99 TTFT (ms):                           93.18     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          58.00     
Median TPOT (ms):                        58.00     
P99 TPOT (ms):                           58.00     
---------------Inter-token Latency----------------
Mean ITL (ms):                           58.00     
Median ITL (ms):                         57.57     
P99 ITL (ms):                            68.27     
==================================================

(APIServer pid=3151) INFO 09-06 22:08:03 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 17.4 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 3.4%, Prefix cache hit rate: 49.2%
(APIServer pid=3151) INFO 09-06 22:08:13 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 17.2 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 3.8%, Prefix cache hit rate: 49.2%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2-72B --dataset-name random \
--host localhost \
--num-prompts 10 \
--random-input-len 1024 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     10        
Benchmark duration (s):                  78.74     
Total input tokens:                      10239     
Total generated tokens:                  10240     
Request throughput (req/s):              0.13      
Output token throughput (tok/s):         130.06    
Total Token throughput (tok/s):          260.10    
---------------Time to First Token----------------
Mean TTFT (ms):                          3842.51   
Median TTFT (ms):                        3959.71   
P99 TTFT (ms):                           5962.30   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          73.04     
Median TPOT (ms):                        72.93     
P99 TPOT (ms):                           75.79     
---------------Inter-token Latency----------------
Mean ITL (ms):                           73.04     
Median ITL (ms):                         70.93     
P99 ITL (ms):                            76.74     
==================================================

(APIServer pid=3151) INFO 09-06 22:08:03 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 17.4 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 3.4%, Prefix cache hit rate: 49.2%
(APIServer pid=3151) INFO 09-06 22:08:13 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 17.2 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 3.8%, Prefix cache hit rate: 49.2%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2-72B --dataset-name random \
--host localhost \
--num-prompts 30 \
--random-input-len 1024 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     30        
Benchmark duration (s):                  148.43    
Total input tokens:                      30699     
Total generated tokens:                  30156     
Request throughput (req/s):              0.20      
Output token throughput (tok/s):         203.16    
Total Token throughput (tok/s):          409.98    
---------------Time to First Token----------------
Mean TTFT (ms):                          10949.72  
Median TTFT (ms):                        11190.15  
P99 TTFT (ms):                           19102.59  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          104.71    
Median TPOT (ms):                        102.82    
P99 TPOT (ms):                           125.32    
---------------Inter-token Latency----------------
Mean ITL (ms):                           104.74    
Median ITL (ms):                         86.42     
P99 ITL (ms):                            1284.35   
==================================================

(APIServer pid=2014) INFO 09-07 01:25:11 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 292.9 tokens/s, Running: 25 reqs, Waiting: 4 reqs, GPU KV cache usage: 99.6%, Prefix cache hit rate: 48.4%
(APIServer pid=2014) INFO 09-07 01:25:21 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 286.9 tokens/s, Running: 23 reqs, Waiting: 6 reqs, GPU KV cache usage: 97.9%, Prefix cache hit rate: 50.9%
(APIServer pid=2014) INFO 09-07 01:25:31 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 270.2 tokens/s, Running: 22 reqs, Waiting: 7 reqs, GPU KV cache usage: 99.7%, Prefix cache hit rate: 49.1%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2-72B --dataset-name random \
--host localhost \
--num-prompts 50 \
--random-input-len 1024 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     50        
Benchmark duration (s):                  209.24    
Total input tokens:                      50977     
Total generated tokens:                  50373     
Request throughput (req/s):              0.24      
Output token throughput (tok/s):         240.74    
Total Token throughput (tok/s):          484.37    
---------------Time to First Token----------------
Mean TTFT (ms):                          26588.12  
Median TTFT (ms):                        8052.65   
P99 TTFT (ms):                           126927.08 
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          123.66    
Median TPOT (ms):                        101.99    
P99 TPOT (ms):                           186.88    
---------------Inter-token Latency----------------
Mean ITL (ms):                           123.49    
Median ITL (ms):                         86.10     
P99 ITL (ms):                            1292.34   
==================================================

(APIServer pid=2014) INFO 09-07 01:29:42 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 381.4 tokens/s, Running: 33 reqs, Waiting: 17 reqs, GPU KV cache usage: 97.4%, Prefix cache hit rate: 47.8%
(APIServer pid=2014) INFO 09-07 01:29:52 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 358.9 tokens/s, Running: 31 reqs, Waiting: 19 reqs, GPU KV cache usage: 99.4%, Prefix cache hit rate: 48.3%
(APIServer pid=2014) INFO 09-07 01:30:02 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 326.1 tokens/s, Running: 28 reqs, Waiting: 21 reqs, GPU KV cache usage: 96.9%, Prefix cache hit rate: 47.0%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2-72B --dataset-name random \
--host localhost \
--num-prompts 100 \
--random-input-len 1024 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     100       
Benchmark duration (s):                  452.91    
Total input tokens:                      102140    
Total generated tokens:                  99612     
Request throughput (req/s):              0.22      
Output token throughput (tok/s):         219.94    
Total Token throughput (tok/s):          445.46    
---------------Time to First Token----------------
Mean TTFT (ms):                          142448.58 
Median TTFT (ms):                        138530.00 
P99 TTFT (ms):                           381085.60 
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          139.24    
Median TPOT (ms):                        119.61    
P99 TPOT (ms):                           225.70    
---------------Inter-token Latency----------------
Mean ITL (ms):                           133.63    
Median ITL (ms):                         87.47     
P99 ITL (ms):                            1310.43   
==================================================

(APIServer pid=2014) INFO 09-07 01:36:42 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 405.6 tokens/s, Running: 36 reqs, Waiting: 64 reqs, GPU KV cache usage: 99.0%, Prefix cache hit rate: 45.6%
(APIServer pid=2014) INFO 09-07 01:36:52 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 370.6 tokens/s, Running: 33 reqs, Waiting: 67 reqs, GPU KV cache usage: 98.8%, Prefix cache hit rate: 45.8%
(APIServer pid=2014) INFO 09-07 01:37:02 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 356.4 tokens/s, Running: 30 reqs, Waiting: 70 reqs, GPU KV cache usage: 97.5%, Prefix cache hit rate: 46.1%
```

### <a name="toc_11"></a>✍️ Test 9: 4 GPU pods host 120B model across different nodes (Available KV cache memory in each GPU: 45.32 GiB)

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model gpt-oss-120b --dataset-name random \
--host 10.42.11.242 \
--num-prompts 1 \
--random-input-len 1024 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     1         
Benchmark duration (s):                  5.68      
Total input tokens:                      857       
Total generated tokens:                  64        
Request throughput (req/s):              0.18      
Output token throughput (tok/s):         11.26     
Total Token throughput (tok/s):          162.02    
---------------Time to First Token----------------
Mean TTFT (ms):                          229.47    
Median TTFT (ms):                        229.47    
P99 TTFT (ms):                           229.47    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          86.57     
Median TPOT (ms):                        86.57     
P99 TPOT (ms):                           86.57     
---------------Inter-token Latency----------------
Mean ITL (ms):                           86.57     
Median ITL (ms):                         85.33     
P99 ITL (ms):                            116.79    
==================================================

(APIServer pid=1009) INFO 09-10 23:59:06 [loggers.py:123] Engine 000: Avg prompt throughput: 85.7 tokens/s, Avg generation throughput: 11.2 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 20.1%
(APIServer pid=1009) INFO 09-10 23:59:16 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 20.1%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model gpt-oss-120b --dataset-name random \
--host 10.42.11.242 \
--num-prompts 10 \
--random-input-len 1024 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     10        
Benchmark duration (s):                  167.67    
Total input tokens:                      10073     
Total generated tokens:                  3204      
Request throughput (req/s):              0.06      
Output token throughput (tok/s):         19.11     
Total Token throughput (tok/s):          79.18     
---------------Time to First Token----------------
Mean TTFT (ms):                          19042.98  
Median TTFT (ms):                        20764.59  
P99 TTFT (ms):                           26200.93  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          967.54    
Median TPOT (ms):                        950.94    
P99 TPOT (ms):                           2633.91   
---------------Inter-token Latency----------------
Mean ITL (ms):                           173.01    
Median ITL (ms):                         134.77    
P99 ITL (ms):                            230.16    
==================================================

(APIServer pid=1009) INFO 09-10 23:55:46 [loggers.py:123] Engine 000: Avg prompt throughput: 102.4 tokens/s, Avg generation throughput: 0.2 tokens/s, Running: 3 reqs, Waiting: 7 reqs, GPU KV cache usage: 0.1%, Prefix cache hit rate: 0.0%
(APIServer pid=1009) INFO 09-10 23:55:56 [loggers.py:123] Engine 000: Avg prompt throughput: 409.6 tokens/s, Avg generation throughput: 1.0 tokens/s, Running: 7 reqs, Waiting: 3 reqs, GPU KV cache usage: 0.2%, Prefix cache hit rate: 0.0%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model gpt-oss-120b --dataset-name random \
--host 10.42.11.242 \
--num-prompts 30 \
--random-input-len 1024 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     30        
Benchmark duration (s):                  243.66    
Total input tokens:                      30553     
Total generated tokens:                  9032      
Request throughput (req/s):              0.12      
Output token throughput (tok/s):         37.07     
Total Token throughput (tok/s):          162.46    
---------------Time to First Token----------------
Mean TTFT (ms):                          23613.37  
Median TTFT (ms):                        20646.46  
P99 TTFT (ms):                           54579.41  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          1854.10   
Median TPOT (ms):                        1384.31   
P99 TPOT (ms):                           4805.25   
---------------Inter-token Latency----------------
Mean ITL (ms):                           290.73    
Median ITL (ms):                         180.45    
P99 ITL (ms):                            5123.33   
==================================================

(APIServer pid=1009) INFO 09-11 00:03:46 [loggers.py:123] Engine 000: Avg prompt throughput: 1228.8 tokens/s, Avg generation throughput: 2.4 tokens/s, Running: 14 reqs, Waiting: 16 reqs, GPU KV cache usage: 0.3%, Prefix cache hit rate: 48.1%
(APIServer pid=1009) INFO 09-11 00:03:56 [loggers.py:123] Engine 000: Avg prompt throughput: 409.6 tokens/s, Avg generation throughput: 3.2 tokens/s, Running: 18 reqs, Waiting: 12 reqs, GPU KV cache usage: 0.4%, Prefix cache hit rate: 41.9%
(APIServer pid=1009) INFO 09-11 00:04:06 [loggers.py:123] Engine 000: Avg prompt throughput: 409.6 tokens/s, Avg generation throughput: 4.0 tokens/s, Running: 22 reqs, Waiting: 8 reqs, GPU KV cache usage: 0.5%, Prefix cache hit rate: 37.1%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model gpt-oss-120b --dataset-name random \
--host 10.42.11.242 \
--num-prompts 50 \
--random-input-len 1024 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     50        
Benchmark duration (s):                  256.95    
Total input tokens:                      50846     
Total generated tokens:                  13007     
Request throughput (req/s):              0.19      
Output token throughput (tok/s):         50.62     
Total Token throughput (tok/s):          248.50    
---------------Time to First Token----------------
Mean TTFT (ms):                          19438.81  
Median TTFT (ms):                        5573.53   
P99 TTFT (ms):                           52919.71  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          2046.05   
Median TPOT (ms):                        1453.35   
P99 TPOT (ms):                           4637.37   
---------------Inter-token Latency----------------
Mean ITL (ms):                           327.55    
Median ITL (ms):                         198.37    
P99 ITL (ms):                            5302.38   
==================================================


(APIServer pid=1009) INFO 09-11 00:14:06 [loggers.py:123] Engine 000: Avg prompt throughput: 409.6 tokens/s, Avg generation throughput: 6.4 tokens/s, Running: 34 reqs, Waiting: 16 reqs, GPU KV cache usage: 0.8%, Prefix cache hit rate: 64.1%
(APIServer pid=1009) INFO 09-11 00:14:16 [loggers.py:123] Engine 000: Avg prompt throughput: 390.9 tokens/s, Avg generation throughput: 7.2 tokens/s, Running: 38 reqs, Waiting: 12 reqs, GPU KV cache usage: 0.9%, Prefix cache hit rate: 61.9%
(APIServer pid=1009) INFO 09-11 00:14:26 [loggers.py:123] Engine 000: Avg prompt throughput: 409.6 tokens/s, Avg generation throughput: 8.0 tokens/s, Running: 42 reqs, Waiting: 8 reqs, GPU KV cache usage: 1.0%, Prefix cache hit rate: 59.8%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model gpt-oss-120b --dataset-name random \
--host 10.42.11.242 \
--num-prompts 75 \
--random-input-len 1024 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     75        
Benchmark duration (s):                  310.58    
Total input tokens:                      76446     
Total generated tokens:                  19128     
Request throughput (req/s):              0.24      
Output token throughput (tok/s):         61.59     
Total Token throughput (tok/s):          307.73    
---------------Time to First Token----------------
Mean TTFT (ms):                          21988.70  
Median TTFT (ms):                        16562.36  
P99 TTFT (ms):                           71032.51  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          2903.98   
Median TPOT (ms):                        2982.35   
P99 TPOT (ms):                           5384.79   
---------------Inter-token Latency----------------
Mean ITL (ms):                           417.86    
Median ITL (ms):                         233.56    
P99 ITL (ms):                            5489.25   
==================================================

(APIServer pid=1009) INFO 09-11 00:21:56 [loggers.py:123] Engine 000: Avg prompt throughput: 409.6 tokens/s, Avg generation throughput: 11.6 tokens/s, Running: 60 reqs, Waiting: 15 reqs, GPU KV cache usage: 1.4%, Prefix cache hit rate: 66.7%
(APIServer pid=1009) INFO 09-11 00:22:06 [loggers.py:123] Engine 000: Avg prompt throughput: 307.2 tokens/s, Avg generation throughput: 12.3 tokens/s, Running: 63 reqs, Waiting: 12 reqs, GPU KV cache usage: 1.4%, Prefix cache hit rate: 65.6%
(APIServer pid=1009) INFO 09-11 00:22:16 [loggers.py:123] Engine 000: Avg prompt throughput: 409.5 tokens/s, Avg generation throughput: 13.0 tokens/s, Running: 67 reqs, Waiting: 8 reqs, GPU KV cache usage: 1.5%, Prefix cache hit rate: 64.2%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model gpt-oss-120b --dataset-name random \
--host 10.42.11.242 \
--num-prompts 100 \
--random-input-len 1024 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     100       
Benchmark duration (s):                  250.30    
Total input tokens:                      102046    
Total generated tokens:                  26295     
Request throughput (req/s):              0.40      
Output token throughput (tok/s):         105.06    
Total Token throughput (tok/s):          512.76    
---------------Time to First Token----------------
Mean TTFT (ms):                          4869.18   
Median TTFT (ms):                        4911.63   
P99 TTFT (ms):                           4963.33   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          427.93    
Median TPOT (ms):                        524.41    
P99 TPOT (ms):                           525.10    
---------------Inter-token Latency----------------
Mean ITL (ms):                           249.88    
Median ITL (ms):                         233.10    
P99 ITL (ms):                            540.85    
==================================================

(APIServer pid=1009) INFO 09-11 01:04:26 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 88.2 tokens/s, Running: 21 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.7%, Prefix cache hit rate: 74.0%
(APIServer pid=1009) INFO 09-11 01:04:36 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 94.5 tokens/s, Running: 21 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.7%, Prefix cache hit rate: 74.0%
(APIServer pid=1009) INFO 09-11 01:04:46 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 98.7 tokens/s, Running: 21 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.8%, Prefix cache hit rate: 74.0%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model gpt-oss-120b --dataset-name random \
--host 10.42.11.242 \
--num-prompts 200 \
--random-input-len 1024 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     200       
Benchmark duration (s):                  321.65    
Total input tokens:                      204259    
Total generated tokens:                  41499     
Request throughput (req/s):              0.62      
Output token throughput (tok/s):         129.02    
Total Token throughput (tok/s):          764.05    
---------------Time to First Token----------------
Mean TTFT (ms):                          7246.46   
Median TTFT (ms):                        6622.51   
P99 TTFT (ms):                           9735.39   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          800.55    
Median TPOT (ms):                        835.04    
P99 TPOT (ms):                           1512.28   
---------------Inter-token Latency----------------
Mean ITL (ms):                           341.76    
Median ITL (ms):                         297.02    
P99 ITL (ms):                            931.08    
==================================================

(APIServer pid=1009) INFO 09-11 01:14:56 [loggers.py:123] Engine 000: Avg prompt throughput: 409.6 tokens/s, Avg generation throughput: 19.6 tokens/s, Running: 100 reqs, Waiting: 100 reqs, GPU KV cache usage: 2.3%, Prefix cache hit rate: 80.4%
(APIServer pid=1009) INFO 09-11 01:15:06 [loggers.py:123] Engine 000: Avg prompt throughput: 409.6 tokens/s, Avg generation throughput: 20.4 tokens/s, Running: 103 reqs, Waiting: 97 reqs, GPU KV cache usage: 2.3%, Prefix cache hit rate: 80.0%
(APIServer pid=1009) INFO 09-11 01:15:16 [loggers.py:123] Engine 000: Avg prompt throughput: 102.4 tokens/s, Avg generation throughput: 10.4 tokens/s, Running: 105 reqs, Waiting: 95 reqs, GPU KV cache usage: 2.4%, Prefix cache hit rate: 79.7%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model gpt-oss-120b --dataset-name random \
--host 10.42.11.242 \
--num-prompts 300 \
--random-input-len 1024 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     300       
Benchmark duration (s):                  701.99    
Total input tokens:                      306659    
Total generated tokens:                  69485     
Request throughput (req/s):              0.43      
Output token throughput (tok/s):         98.98     
Total Token throughput (tok/s):          535.83    
---------------Time to First Token----------------
Mean TTFT (ms):                          83858.26  
Median TTFT (ms):                        10239.00  
P99 TTFT (ms):                           298431.09 
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          4030.65   
Median TPOT (ms):                        5519.06   
P99 TPOT (ms):                           5604.54   
---------------Inter-token Latency----------------
Mean ITL (ms):                           836.51    
Median ITL (ms):                         400.14    
P99 ITL (ms):                            5746.57   
==================================================

(APIServer pid=1009) INFO 09-11 01:31:16 [loggers.py:123] Engine 000: Avg prompt throughput: 409.6 tokens/s, Avg generation throughput: 34.2 tokens/s, Running: 173 reqs, Waiting: 127 reqs, GPU KV cache usage: 3.9%, Prefix cache hit rate: 79.7%
(APIServer pid=1009) INFO 09-11 01:31:26 [loggers.py:123] Engine 000: Avg prompt throughput: 409.6 tokens/s, Avg generation throughput: 35.0 tokens/s, Running: 177 reqs, Waiting: 123 reqs, GPU KV cache usage: 4.0%, Prefix cache hit rate: 79.4%
(APIServer pid=1009) INFO 09-11 01:31:36 [loggers.py:123] Engine 000: Avg prompt throughput: 102.4 tokens/s, Avg generation throughput: 17.7 tokens/s, Running: 178 reqs, Waiting: 122 reqs, GPU KV cache usage: 4.0%, Prefix cache hit rate: 79.4%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model gpt-oss-120b --dataset-name random \
--host 10.42.11.242 \
--num-prompts 400 \
--random-input-len 1024 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     400       
Benchmark duration (s):                  873.95    
Total input tokens:                      409059    
Total generated tokens:                  106578    
Request throughput (req/s):              0.46      
Output token throughput (tok/s):         121.95    
Total Token throughput (tok/s):          590.01    
---------------Time to First Token----------------
Mean TTFT (ms):                          61278.55  
Median TTFT (ms):                        11316.18  
P99 TTFT (ms):                           321387.81 
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          2411.28   
Median TPOT (ms):                        1781.71   
P99 TPOT (ms):                           5621.62   
---------------Inter-token Latency----------------
Mean ITL (ms):                           925.99    
Median ITL (ms):                         565.22    
P99 ITL (ms):                            5653.89   
==================================================

(APIServer pid=1009) INFO 09-11 01:43:37 [loggers.py:123] Engine 000: Avg prompt throughput: 26073.1 tokens/s, Avg generation throughput: 57.0 tokens/s, Running: 256 reqs, Waiting: 144 reqs, GPU KV cache usage: 5.7%, Prefix cache hit rate: 78.7%
(APIServer pid=1009) INFO 09-11 01:43:47 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 204.8 tokens/s, Running: 238 reqs, Waiting: 144 reqs, GPU KV cache usage: 5.3%, Prefix cache hit rate: 78.7%
(APIServer pid=1009) INFO 09-11 01:43:57 [loggers.py:123] Engine 000: Avg prompt throughput: 3481.5 tokens/s, Avg generation throughput: 47.0 tokens/s, Running: 161 reqs, Waiting: 109 reqs, GPU KV cache usage: 3.6%, Prefix cache hit rate: 79.1%
```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model gpt-oss-120b --dataset-name random \
--host 10.42.11.242 \
--num-prompts 500 \
--random-input-len 1024 \
--random-output-len 

============ Serving Benchmark Result ============
Successful requests:                     500       
Benchmark duration (s):                  879.63    
Total input tokens:                      511459    
Total generated tokens:                  113510    
Request throughput (req/s):              0.57      
Output token throughput (tok/s):         129.04    
Total Token throughput (tok/s):          710.49    
---------------Time to First Token----------------
Mean TTFT (ms):                          91005.74  
Median TTFT (ms):                        13093.68  
P99 TTFT (ms):                           324563.23 
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          2391.57   
Median TPOT (ms):                        1619.49   
P99 TPOT (ms):                           5521.92   
---------------Inter-token Latency----------------
Mean ITL (ms):                           924.32    
Median ITL (ms):                         560.37    
P99 ITL (ms):                            5664.75   
==================================================

(APIServer pid=1009) INFO 09-11 02:03:07 [loggers.py:123] Engine 000: Avg prompt throughput: 409.6 tokens/s, Avg generation throughput: 27.3 tokens/s, Running: 136 reqs, Waiting: 100 reqs, GPU KV cache usage: 3.1%, Prefix cache hit rate: 76.3%
(APIServer pid=1009) INFO 09-11 02:03:17 [loggers.py:123] Engine 000: Avg prompt throughput: 409.6 tokens/s, Avg generation throughput: 27.4 tokens/s, Running: 134 reqs, Waiting: 96 reqs, GPU KV cache usage: 3.1%, Prefix cache hit rate: 76.1%
(APIServer pid=1009) INFO 09-11 02:03:27 [loggers.py:123] Engine 000: Avg prompt throughput: 204.8 tokens/s, Avg generation throughput: 13.5 tokens/s, Running: 135 reqs, Waiting: 94 reqs, GPU KV cache usage: 3.1%, Prefix cache hit rate: 76.0%
```
