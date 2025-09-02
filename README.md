# vLLM Benchmarking

- 2 pods hosted on different nodes

```
import subprocess, os, time
import cml.workers_v1 as workers

DASHBOARD_IP = os.environ['CDSW_IP_ADDRESS']

command = "ray start --head --block --include-dashboard=true --dashboard-port=$CDSW_READONLY_PORT --num-cpus=4 --num-gpus=1 &" 
subprocess.run(command, shell = True, executable="/bin/bash")

with open("RAY_HEAD_IP", 'w') as output_file:
    output_file.write(DASHBOARD_IP)

ray_head_addr = DASHBOARD_IP + ':6379'
ray_url = f"ray://{DASHBOARD_IP}:10001" 
worker_start_cmd = f"!ray start --block --address={ray_head_addr}"

time.sleep(7)
ray_workers = workers.launch_workers(
    n=1,
    cpu=2, 
    memory=48,
    nvidia_gpu=1,
    code=worker_start_cmd,
)

os.system("vllm serve Qwen2.5-32B-Instruct --port 8081 --tensor-parallel-size 2 --max-model-len 8192 > vllm.log 2>&1 &")
```

- Results:
```
$ python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm --host 10.254.5.30 --port 8081 \
--endpoint='/v1/completions' --model Qwen2.5-32B-Instruct --dataset-name random  --num-prompts 10 --random-input-len 1024
```


<img width="700" height="201" alt="image" src="https://github.com/user-attachments/assets/604131c9-8972-492c-a490-2c5dedbf5de1" />

Test 1: 32B model

```
============ Serving Benchmark Result ============
Successful requests:                     10        
Benchmark duration (s):                  15.32     
Total input tokens:                      10239     
Total generated tokens:                  1280      
Request throughput (req/s):              0.65      
Output token throughput (tok/s):         83.54     
Total Token throughput (tok/s):          751.76    
---------------Time to First Token----------------
Mean TTFT (ms):                          406.76    
Median TTFT (ms):                        426.54    
P99 TTFT (ms):                           434.09    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          117.26    
Median TPOT (ms):                        117.19    
P99 TPOT (ms):                           117.92    
---------------Inter-token Latency----------------
Mean ITL (ms):                           117.26    
Median ITL (ms):                         117.27    
P99 ITL (ms):                            130.44    
==================================================
```

```
============ Serving Benchmark Result ============
Successful requests:                     10        
Benchmark duration (s):                  15.24     
Total input tokens:                      20431     
Total generated tokens:                  1280      
Request throughput (req/s):              0.66      
Output token throughput (tok/s):         83.97     
Total Token throughput (tok/s):          1424.23   
---------------Time to First Token----------------
Mean TTFT (ms):                          402.07    
Median TTFT (ms):                        423.50    
P99 TTFT (ms):                           429.40    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          116.70    
Median TPOT (ms):                        116.61    
P99 TPOT (ms):                           117.41    
---------------Inter-token Latency----------------
Mean ITL (ms):                           116.70    
Median ITL (ms):                         116.04    
P99 ITL (ms):                            129.78    
==================================================
```

```
============ Serving Benchmark Result ============
Successful requests:                     10        
Benchmark duration (s):                  63.72     
Total input tokens:                      40952     
Total generated tokens:                  1280      
Request throughput (req/s):              0.16      
Output token throughput (tok/s):         20.09     
Total Token throughput (tok/s):          662.77    
---------------Time to First Token----------------
Mean TTFT (ms):                          26058.47  
Median TTFT (ms):                        20612.53  
P99 TTFT (ms):                           52809.36  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          197.37    
Median TPOT (ms):                        244.78    
P99 TPOT (ms):                           245.36    
---------------Inter-token Latency----------------
Mean ITL (ms):                           197.37    
Median ITL (ms):                         94.89     
P99 ITL (ms):                            2420.75   
==================================================

(APIServer pid=118) INFO 09-02 05:20:47 [loggers.py:123] Engine 000: Avg prompt throughput: 1634.2 tokens/s, Avg generation throughput: 5.1 tokens/s, Running: 16 reqs, Waiting: 3 reqs, GPU KV cache usage: 63.1%, Prefix cache hit rate: 37.4%
(APIServer pid=118) INFO 09-02 05:20:57 [loggers.py:123] Engine 000: Avg prompt throughput: 409.0 tokens/s, Avg generation throughput: 109.9 tokens/s, Running: 19 reqs, Waiting: 0 reqs, GPU KV cache usage: 79.3%, Prefix cache hit rate: 36.9%
(APIServer pid=118) INFO 09-02 05:21:07 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 127.6 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.1%, Prefix cache hit rate: 36.9%
```


```
============ Serving Benchmark Result ============
Successful requests:                     10        
Benchmark duration (s):                  122.88    
Total input tokens:                      80586     
Total generated tokens:                  1280      
Request throughput (req/s):              0.08      
Output token throughput (tok/s):         10.42     
Total Token throughput (tok/s):          666.23    
---------------Time to First Token----------------
Mean TTFT (ms):                          55980.42  
Median TTFT (ms):                        52079.35  
P99 TTFT (ms):                           112779.23 
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          179.10    
Median TPOT (ms):                        182.48    
P99 TPOT (ms):                           234.76    
---------------Inter-token Latency----------------
Mean ITL (ms):                           179.10    
Median ITL (ms):                         85.82     
P99 ITL (ms):                            2428.84   
==================================================

(APIServer pid=118) INFO 09-02 05:53:29 [loggers.py:123] Engine 000: Avg prompt throughput: 1609.2 tokens/s, Avg generation throughput: 25.8 tokens/s, Running: 3 reqs, Waiting: 7 reqs, GPU KV cache usage: 94.0%, Prefix cache hit rate: 21.5%
(APIServer pid=118) INFO 09-02 05:53:39 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 12.0 tokens/s, Running: 3 reqs, Waiting: 6 reqs, GPU KV cache usage: 81.6%, Prefix cache hit rate: 20.1%
(APIServer pid=118) INFO 09-02 05:53:49 [loggers.py:123] Engine 000: Avg prompt throughput: 803.1 tokens/s, Avg generation throughput: 0.5 tokens/s, Running: 2 reqs, Waiting: 5 reqs, GPU KV cache usage: 50.2%, Prefix cache hit rate: 20.0%
(APIServer pid=118) INFO 09-02 05:53:59 [loggers.py:123] Engine 000: Avg prompt throughput: 799.7 tokens/s, Avg generation throughput: 0.7 tokens/s, Running: 3 reqs, Waiting: 4 reqs, GPU KV cache usage: 81.6%, Prefix cache hit rate: 20.0%
(APIServer pid=118) INFO 09-02 05:54:09 [loggers.py:123] Engine 000: Avg prompt throughput: 805.6 tokens/s, Avg generation throughput: 26.6 tokens/s, Running: 3 reqs, Waiting: 4 reqs, GPU KV cache usage: 93.8%, Prefix cache hit rate: 18.8%
(APIServer pid=118) INFO 09-02 05:54:19 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 10.0 tokens/s, Running: 3 reqs, Waiting: 3 reqs, GPU KV cache usage: 83.0%, Prefix cache hit rate: 18.5%
(APIServer pid=118) INFO 09-02 05:54:30 [loggers.py:123] Engine 000: Avg prompt throughput: 803.1 tokens/s, Avg generation throughput: 1.1 tokens/s, Running: 3 reqs, Waiting: 2 reqs, GPU KV cache usage: 91.1%, Prefix cache hit rate: 18.5%
(APIServer pid=118) INFO 09-02 05:54:40 [loggers.py:123] Engine 000: Avg prompt throughput: 805.4 tokens/s, Avg generation throughput: 0.9 tokens/s, Running: 3 reqs, Waiting: 1 reqs, GPU KV cache usage: 91.1%, Prefix cache hit rate: 18.4%
(APIServer pid=118) INFO 09-02 05:54:50 [loggers.py:123] Engine 000: Avg prompt throughput: 805.3 tokens/s, Avg generation throughput: 28.8 tokens/s, Running: 3 reqs, Waiting: 1 reqs, GPU KV cache usage: 94.1%, Prefix cache hit rate: 18.5%
```


```
$ python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm --host 10.254.5.31 --port 8081 --endpoint='/v1/completions' --model Qwen2.5-32B-Instruct --dataset-name random  --num-prompts 10 --random-input-len 4096 --random-output-len 4096

============ Serving Benchmark Result ============
Successful requests:                     10        
Benchmark duration (s):                  1076.61   
Total input tokens:                      40952     
Total generated tokens:                  40960     
Request throughput (req/s):              0.01      
Output token throughput (tok/s):         38.05     
Total Token throughput (tok/s):          76.08     
---------------Time to First Token----------------
Mean TTFT (ms):                          233987.48 
Median TTFT (ms):                        23589.87  
P99 TTFT (ms):                           752280.98 
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          113.57    
Median TPOT (ms):                        94.28     
P99 TPOT (ms):                           176.24    
---------------Inter-token Latency----------------
Mean ITL (ms):                           113.60    
Median ITL (ms):                         83.72     
P99 ITL (ms):                            112.26    
==================================================

(APIServer pid=117) INFO 09-02 06:26:02 [loggers.py:123] Engine 000: Avg prompt throughput: 408.8 tokens/s, Avg generation throughput: 6.3 tokens/s, Running: 2 reqs, Waiting: 8 reqs, GPU KV cache usage: 31.6%, Prefix cache hit rate: 33.2%
(APIServer pid=117) INFO 09-02 06:26:12 [loggers.py:123] Engine 000: Avg prompt throughput: 818.0 tokens/s, Avg generation throughput: 1.0 tokens/s, Running: 4 reqs, Waiting: 6 reqs, GPU KV cache usage: 63.1%, Prefix cache hit rate: 19.9%
(APIServer pid=117) INFO 09-02 06:26:22 [loggers.py:123] Engine 000: Avg prompt throughput: 1225.9 tokens/s, Avg generation throughput: 9.6 tokens/s, Running: 6 reqs, Waiting: 4 reqs, GPU KV cache usage: 94.9%, Prefix cache hit rate: 5.0%
(APIServer pid=117) INFO 09-02 06:26:32 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 59.9 tokens/s, Running: 6 reqs, Waiting: 4 reqs, GPU KV cache usage: 97.2%, Prefix cache hit rate: 0.8%
(APIServer pid=117) INFO 09-02 06:26:42 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 59.8 tokens/s, Running: 6 reqs, Waiting: 4 reqs, GPU KV cache usage: 99.4%, Prefix cache hit rate: 0.5%
(APIServer pid=117) INFO 09-02 06:26:52 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 53.1 tokens/s, Running: 5 reqs, Waiting: 5 reqs, GPU KV cache usage: 84.9%, Prefix cache hit rate: 24.8%
(APIServer pid=117) INFO 09-02 06:27:02 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 49.4 tokens/s, Running: 5 reqs, Waiting: 5 reqs, GPU KV cache usage: 86.8%, Prefix cache hit rate: 39.5%
(APIServer pid=117) INFO 09-02 06:27:12 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 49.8 tokens/s, Running: 5 reqs, Waiting: 5 reqs, GPU KV cache usage: 88.7%, Prefix cache hit rate: 46.2%
(APIServer pid=117) INFO 09-02 06:27:22 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 49.9 tokens/s, Running: 5 reqs, Waiting: 5 reqs, GPU KV cache usage: 90.6%, Prefix cache hit rate: 48.8%


```


Test 2: 7B model

Before loading KV cache,
<img width="700" height="200" alt="image" src="https://github.com/user-attachments/assets/77dbd949-88e9-46f0-b833-78888a31d083" />

```
Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]
Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:57<02:52, 57.61s/it]
Loading safetensors checkpoint shards:  50% Completed | 2/4 [01:52<01:51, 55.96s/it]
```

After KV cache is loaded,
<img width="700" height="196" alt="image" src="https://github.com/user-attachments/assets/7b127f14-2ef5-40b0-b78b-90f9fecf6f75" />


```
============ Serving Benchmark Result ============
Successful requests:                     250       
Benchmark duration (s):                  41.02     
Total input tokens:                      255124    
Total generated tokens:                  31742     
Request throughput (req/s):              6.09      
Output token throughput (tok/s):         773.85    
Total Token throughput (tok/s):          6993.59   
---------------Time to First Token----------------
Mean TTFT (ms):                          4030.55   
Median TTFT (ms):                        1628.12   
P99 TTFT (ms):                           24148.84  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          279.17    
Median TPOT (ms):                        295.72    
P99 TPOT (ms):                           298.42    
---------------Inter-token Latency----------------
Mean ITL (ms):                           277.47    
Median ITL (ms):                         153.01    
P99 ITL (ms):                            818.49    
==================================================

(APIServer pid=119) INFO 09-02 08:42:37 [loggers.py:123] Engine 000: Avg prompt throughput: 2541.7 tokens/s, Avg generation throughput: 117.6 tokens/s, Running: 103 reqs, Waiting: 40 reqs, GPU KV cache usage: 26.2%, Prefix cache hit rate: 58.4%
(APIServer pid=119) INFO 09-02 08:42:47 [loggers.py:123] Engine 000: Avg prompt throughput: 2858.4 tokens/s, Avg generation throughput: 163.7 tokens/s, Running: 130 reqs, Waiting: 12 reqs, GPU KV cache usage: 33.2%, Prefix cache hit rate: 56.9%
```

```
============ Serving Benchmark Result ============
Successful requests:                     400       
Benchmark duration (s):                  177.07    
Total input tokens:                      408335    
Total generated tokens:                  50923     
Request throughput (req/s):              2.26      
Output token throughput (tok/s):         287.58    
Total Token throughput (tok/s):          2593.63   
---------------Time to First Token----------------
Mean TTFT (ms):                          78042.97  
Median TTFT (ms):                        75852.90  
P99 TTFT (ms):                           163364.23 
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          595.26    
Median TPOT (ms):                        717.87    
P99 TPOT (ms):                           795.95    
---------------Inter-token Latency----------------
Mean ITL (ms):                           594.95    
Median ITL (ms):                         785.18    
P99 ITL (ms):                            815.75    
==================================================

(APIServer pid=119) INFO 09-02 08:54:47 [loggers.py:123] Engine 000: Avg prompt throughput: 2351.4 tokens/s, Avg generation throughput: 302.4 tokens/s, Running: 241 reqs, Waiting: 154 reqs, GPU KV cache usage: 63.5%, Prefix cache hit rate: 35.5%
(APIServer pid=119) INFO 09-02 08:54:57 [loggers.py:123] Engine 000: Avg prompt throughput: 2144.6 tokens/s, Avg generation throughput: 286.7 tokens/s, Running: 238 reqs, Waiting: 133 reqs, GPU KV cache usage: 62.8%, Prefix cache hit rate: 35.0%
(APIServer pid=119) INFO 09-02 08:55:07 [loggers.py:123] Engine 000: Avg prompt throughput: 2147.6 tokens/s, Avg generation throughput: 285.6 tokens/s, Running: 236 reqs, Waiting: 112 reqs, GPU KV cache usage: 62.4%, Prefix cache hit rate: 34.6%
(APIServer pid=119) INFO 09-02 08:55:17 [loggers.py:123] Engine 000: Avg prompt throughput: 2351.4 tokens/s, Avg generation throughput: 305.2 tokens/s, Running: 233 reqs, Waiting: 89 reqs, GPU KV cache usage: 61.6%, Prefix cache hit rate: 34.1%
(APIServer pid=119) INFO 09-02 08:55:27 [loggers.py:123] Engine 000: Avg prompt throughput: 2225.6 tokens/s, Avg generation throughput: 277.8 tokens/s, Running: 231 reqs, Waiting: 67 reqs, GPU KV cache usage: 60.9%, Prefix cache hit rate: 33.7%
(APIServer pid=119) INFO 09-02 08:55:37 [loggers.py:123] Engine 000: Avg prompt throughput: 2441.5 tokens/s, Avg generation throughput: 301.4 tokens/s, Running: 231 reqs, Waiting: 43 reqs, GPU KV cache usage: 60.7%, Prefix cache hit rate: 33.3%
```
####################

- Test: 2 GPU pods hosted on 2 different nodes
```
(EngineCore_0 pid=1250) (RayWorkerWrapper pid=1376) INFO 09-02 13:02:34 [gpu_worker.py:276] Available KV cache memory: 11.03 GiB
(EngineCore_0 pid=1250) INFO 09-02 13:02:47 [core.py:214] init engine (profile, create kv cache, warmup model) took 62.12 seconds
(APIServer pid=158) INFO 09-02 13:02:48 [loggers.py:142] Engine 000: vllm cache_config_info with initialization after num_gpu_blocks is:
 25809
```

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


<img width="700" height="188" alt="image" src="https://github.com/user-attachments/assets/41c6fe14-5c1c-48fe-b8eb-cfda20d28e06" />

```
(VllmWorker TP1 pid=1546) INFO 09-02 10:29:38 [gpu_worker.py:276] Available KV cache memory: 11.15 GiB
(VllmWorker TP0 pid=1545) INFO 09-02 10:29:38 [gpu_worker.py:276] Available KV cache memory: 11.15 GiB
(EngineCore_0 pid=1473) INFO 09-02 10:29:39 [kv_cache_utils.py:849] GPU KV cache size: 417,520 tokens
(EngineCore_0 pid=1473) INFO 09-02 10:29:39 [kv_cache_utils.py:853] Maximum concurrency for 16,384 tokens per request: 25.48x
(EngineCore_0 pid=1473) INFO 09-02 10:29:39 [kv_cache_utils.py:849] GPU KV cache size: 417,520 tokens
(EngineCore_0 pid=1473) INFO 09-02 10:29:39 [kv_cache_utils.py:853] Maximum concurrency for 16,384 tokens per request: 25.48x
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
Benchmark duration (s):                  13.74     
Total input tokens:                      10239     
Total generated tokens:                  10073     
Request throughput (req/s):              0.73      
Output token throughput (tok/s):         732.88    
Total Token throughput (tok/s):          1477.84   
---------------Time to First Token----------------
Mean TTFT (ms):                          633.89    
Median TTFT (ms):                        699.61    
P99 TTFT (ms):                           881.58    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          12.79     
Median TPOT (ms):                        12.72     
P99 TPOT (ms):                           13.25     
---------------Inter-token Latency----------------
Mean ITL (ms):                           12.78     
Median ITL (ms):                         12.41     
P99 ITL (ms):                            18.16     
==================================================

(APIServer pid=1332) INFO 09-02 10:31:14 [loggers.py:123] Engine 000: Avg prompt throughput: 614.3 tokens/s, Avg generation throughput: 774.5 tokens/s, Running: 10 reqs, Waiting: 0 reqs, GPU KV cache usage: 4.3%, Prefix cache hit rate: 8.9%

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
Benchmark duration (s):                  57.91     
Total input tokens:                      40952     
Total generated tokens:                  40960     
Request throughput (req/s):              0.17      
Output token throughput (tok/s):         707.25    
Total Token throughput (tok/s):          1414.37   
---------------Time to First Token----------------
Mean TTFT (ms):                          1635.45   
Median TTFT (ms):                        1748.43   
P99 TTFT (ms):                           2578.75   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          13.73     
Median TPOT (ms):                        13.70     
P99 TPOT (ms):                           13.98     
---------------Inter-token Latency----------------
Mean ITL (ms):                           13.73     
Median ITL (ms):                         13.42     
P99 ITL (ms):                            16.37     
==================================================
(APIServer pid=1332) INFO 09-02 10:38:14 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 728.9 tokens/s, Running: 10 reqs, Waiting: 0 reqs, GPU KV cache usage: 16.4%, Prefix cache hit rate: 27.2%
(APIServer pid=1332) INFO 09-02 10:38:24 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 727.9 tokens/s, Running: 10 reqs, Waiting: 0 reqs, GPU KV cache usage: 18.1%, Prefix cache hit rate: 27.2%
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
Benchmark duration (s):                  94.44     
Total input tokens:                      10239     
Total generated tokens:                  69058     
Request throughput (req/s):              0.11      
Output token throughput (tok/s):         731.25    
Total Token throughput (tok/s):          839.67    
---------------Time to First Token----------------
Mean TTFT (ms):                          115.10    
Median TTFT (ms):                        115.16    
P99 TTFT (ms):                           130.27    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          13.14     
Median TPOT (ms):                        13.16     
P99 TPOT (ms):                           13.16     
---------------Inter-token Latency----------------
Mean ITL (ms):                           13.15     
Median ITL (ms):                         13.07     
P99 ITL (ms):                            18.10     
==================================================

(APIServer pid=1332) INFO 09-02 10:42:14 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 664.1 tokens/s, Running: 9 reqs, Waiting: 0 reqs, GPU KV cache usage: 15.1%, Prefix cache hit rate: 39.1%
(APIServer pid=1332) INFO 09-02 10:42:24 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 667.7 tokens/s, Running: 9 reqs, Waiting: 0 reqs, GPU KV cache usage: 16.7%, Prefix cache hit rate: 39.1%
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
Benchmark duration (s):                  34.47     
Total input tokens:                      102140    
Total generated tokens:                  101674    
Request throughput (req/s):              2.90      
Output token throughput (tok/s):         2949.72   
Total Token throughput (tok/s):          5912.95   
---------------Time to First Token----------------
Mean TTFT (ms):                          4341.42   
Median TTFT (ms):                        4136.42   
P99 TTFT (ms):                           8598.90   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          29.05     
Median TPOT (ms):                        29.25     
P99 TPOT (ms):                           32.47     
---------------Inter-token Latency----------------
Mean ITL (ms):                           29.06     
Median ITL (ms):                         25.36     
P99 ITL (ms):                            187.08    
==================================================

(APIServer pid=1332) INFO 09-02 10:44:34 [loggers.py:123] Engine 000: Avg prompt throughput: 409.4 tokens/s, Avg generation throughput: 3907.8 tokens/s, Running: 100 reqs, Waiting: 0 reqs, GPU KV cache usage: 34.5%, Prefix cache hit rate: 22.0%
(APIServer pid=1332) INFO 09-02 10:44:44 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 3869.9 tokens/s, Running: 99 reqs, Waiting: 0 reqs, GPU KV cache usage: 43.4%, Prefix cache hit rate: 22.0%
(APIServer pid=1332) INFO 09-02 10:44:54 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 
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
Benchmark duration (s):                  236.76    
Total input tokens:                      409048    
Total generated tokens:                  401433    
Request throughput (req/s):              0.42      
Output token throughput (tok/s):         1695.54   
Total Token throughput (tok/s):          3423.25   
---------------Time to First Token----------------
Mean TTFT (ms):                          13565.17  
Median TTFT (ms):                        13042.74  
P99 TTFT (ms):                           29220.30  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          41.60     
Median TPOT (ms):                        33.86     
P99 TPOT (ms):                           55.66     
---------------Inter-token Latency----------------
Mean ITL (ms):                           39.92     
Median ITL (ms):                         26.38     
P99 ITL (ms):                            217.12    
==================================================


(APIServer pid=1332) INFO 09-02 10:48:04 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 2615.4 tokens/s, Running: 72 reqs, Waiting: 26 reqs, GPU KV cache usage: 98.8%, Prefix cache hit rate: 48.1%
(APIServer pid=1332) INFO 09-02 10:48:14 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 2537.6 tokens/s, Running: 68 reqs, Waiting: 30 reqs, GPU KV cache usage: 99.2%, Prefix cache hit rate: 49.0%
(APIServer pid=1332) INFO 09-02 10:48:24 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 2435.1 tokens/s, Running: 64 reqs, Waiting: 34 reqs, GPU KV cache usage: 99.1%, Prefix cache hit rate: 49.1%
(APIServer pid=1332) INFO 09-02 10:48:34 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 2379.0 tokens/s, Running: 60 reqs, Waiting: 38 reqs, GPU KV cache usage: 98.4%, Prefix cache hit rate: 48.7%
(APIServer pid=1332) INFO 09-02 10:48:44 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 2231.6 tokens/s, Running: 57 reqs, Waiting: 41 reqs, GPU KV cache usage: 98.7%, Prefix cache hit rate: 49.2%
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
Benchmark duration (s):                  296.11    
Total input tokens:                      102140    
Total generated tokens:                  703511    
Request throughput (req/s):              0.34      
Output token throughput (tok/s):         2375.81   
Total Token throughput (tok/s):          2720.74   
---------------Time to First Token----------------
Mean TTFT (ms):                          3950.52   
Median TTFT (ms):                        4845.33   
P99 TTFT (ms):                           5380.54   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          32.27     
Median TPOT (ms):                        28.08     
P99 TPOT (ms):                           40.50     
---------------Inter-token Latency----------------
Mean ITL (ms):                           32.35     
Median ITL (ms):                         26.77     
P99 ITL (ms):                            36.85     
==================================================

(APIServer pid=1332) INFO 09-02 10:55:04 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 2767.5 tokens/s, Running: 78 reqs, Waiting: 20 reqs, GPU KV cache usage: 99.9%, Prefix cache hit rate: 48.8%
(APIServer pid=1332) INFO 09-02 10:55:14 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 2591.3 tokens/s, Running: 73 reqs, Waiting: 25 reqs, GPU KV cache usage: 99.5%, Prefix cache hit rate: 48.8%
(APIServer pid=1332) INFO 09-02 10:55:24 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 2568.6 tokens/s, Running: 68 reqs, Waiting: 30 reqs, GPU KV cache usage: 98.6%, Prefix cache hit rate: 48.7%
```

<img width="700" height="402" alt="image" src="https://github.com/user-attachments/assets/838c34dd-1545-4424-aef4-e862d816bdda" />


```
python /home/cdsw/vllm/benchmarks/benchmark_serving.py --backend vllm \
--port 8081 --endpoint='/v1/completions' --model Qwen2-7B-Instruct --dataset-name random \
--host localhost \
--num-prompts 200 \
--random-input-len 1024 \
--random-output-len 1024

============ Serving Benchmark Result ============
Successful requests:                     200       
Benchmark duration (s):                  59.20     
Total input tokens:                      204007    
Total generated tokens:                  202260    
Request throughput (req/s):              3.38      
Output token throughput (tok/s):         3416.82   
Total Token throughput (tok/s):          6863.15   
---------------Time to First Token----------------
Mean TTFT (ms):                          7152.59   
Median TTFT (ms):                        5473.72   
P99 TTFT (ms):                           15965.26  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          50.20     
Median TPOT (ms):                        51.55     
P99 TPOT (ms):                           56.52     
---------------Inter-token Latency----------------
Mean ITL (ms):                           50.03     
Median ITL (ms):                         43.20     
P99 ITL (ms):                            208.80    
==================================================

(APIServer pid=1332) INFO 09-02 11:00:34 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 4506.7 tokens/s, Running: 197 reqs, Waiting: 0 reqs, GPU KV cache usage: 67.4%, Prefix cache hit rate: 48.6%
(APIServer pid=1332) INFO 09-02 11:00:44 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 4566.5 tokens/s, Running: 197 reqs, Waiting: 0 reqs, GPU KV cache usage: 78.4%, Prefix cache hit rate: 48.6%
(APIServer pid=1332) INFO 09-02 11:00:55 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 4467.9 tokens/s, Running: 197 reqs, Waiting: 0 reqs, GPU KV cache usage: 89.1%, Prefix cache hit rate: 48.6%
```

