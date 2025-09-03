# vLLM Benchmarking


## Platform Requirement
✅ Python 3.11/10

✅ Cloudera AI (CAI) / Cloudera Machine Learning (CML) 1.5.x

## Procedure

1. Create a new CAI project.
   
2. Install python libraries.
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



```

```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2-7B-Instruct --dataset-name random \
--host 10.254.5.35 \
--num-prompts 100 \
--random-input-len 1024 \
--random-output-len 7168


```


### ✍️ Test: 2 GPU pods hosted on 2 different nodes (Available KV cache memory: 12.02 GiB)

<img width="700" height="188" alt="image" src="https://github.com/user-attachments/assets/41c6fe14-5c1c-48fe-b8eb-cfda20d28e06" />


<img width="700" height="409" alt="image" src="https://github.com/user-attachments/assets/e6e42a91-f5a6-406d-8b23-b3dde2c76f21" />

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
