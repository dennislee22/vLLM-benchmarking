# vLLM Benchmarking


## Platform Requirement
✅ Python 3.11/10

✅ Cloudera AI (CAI) / Cloudera Machine Learning (CML) 1.5.x

## Procedure

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

4. Create Application to expose vLLM API endpoint. As vLLM utilizes Ray, this application will also host the Ray HEAD and Ray dashboard with 1 GPU device.
<img width="460" height="730" alt="image" src="https://github.com/user-attachments/assets/d128c611-969d-4e01-9fe4-54a73f9db055" />

5. Start the `vllm-api` application and ensure that the model is fully loaded into the GPU before starting the `gradio-app` application. The code will spawn one Ray HEAD pod along with its associated worker pod within seconds.

```
NAME               READY   STATUS    RESTARTS   AGE   IP             NODE        NOMINATED NODE   READINESS GATES
5c4bmyq953zzetaj   5/5     Running   0          28m   10.254.6.219   worker-19   <none>           <none>
ip46y52fijaguyzo   5/5     Running   0          29m   10.254.5.36    worker-20   <none>           <none>
```

### ✍️ Test 1: 2 GPU pods hosted on 2 different nodes (Available KV cache memory: 11.03 GiB)


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


### ✍️ Test 2: 2 GPU pods hosted in the same node (Available KV cache memory: 12.02 GiB)


<img width="700" height="409" alt="image" src="https://github.com/user-attachments/assets/e6e42a91-f5a6-406d-8b23-b3dde2c76f21" />


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


### ✍️ Test 3: 2 GPU pods hosted in the same node (Available KV cache memory: 31.83 GiB)

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
Benchmark duration (s):                  15.92     
Total input tokens:                      510414    
Total generated tokens:                  63475     
Request throughput (req/s):              31.40     
Output token throughput (tok/s):         3986.76   
Total Token throughput (tok/s):          36044.99  
---------------Time to First Token----------------
Mean TTFT (ms):                          5005.35   
Median TTFT (ms):                        3319.26   
P99 TTFT (ms):                           10440.97  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          54.57     
Median TPOT (ms):                        55.65     
P99 TPOT (ms):                           60.34     
---------------Inter-token Latency----------------
Mean ITL (ms):                           54.43     
Median ITL (ms):                         47.70     
P99 ITL (ms):                            130.33    
==================================================

(APIServer pid=3761) INFO 09-03 02:25:52 [loggers.py:123] Engine 000: Avg prompt throughput: 26419.6 tokens/s, Avg generation throughput: 2886.1 tokens/s, Running: 256 reqs, Waiting: 242 reqs, GPU KV cache usage: 24.5%, Prefix cache hit rate: 41.4%
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
