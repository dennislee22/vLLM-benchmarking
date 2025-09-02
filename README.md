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

Test 1: 10 requests with 1024 tokens each.

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
```

```
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
```

```
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

Test 2: 20

```
============ Serving Benchmark Result ============
Successful requests:                     20        
Benchmark duration (s):                  29.03     
Total input tokens:                      20460     
Total generated tokens:                  2433      
Request throughput (req/s):              0.69      
Output token throughput (tok/s):         83.81     
Total Token throughput (tok/s):          788.65    
---------------Time to First Token----------------
Mean TTFT (ms):                          5938.20   
Median TTFT (ms):                        3850.27   
P99 TTFT (ms):                           12594.95  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          178.91    
Median TPOT (ms):                        204.18    
P99 TPOT (ms):                           219.19    
---------------Inter-token Latency----------------
Mean ITL (ms):                           178.91    
Median ITL (ms):                         130.38    
P99 ITL (ms):                            2446.74   
==================================================
```

```
============ Serving Benchmark Result ============
Successful requests:                     30        
Benchmark duration (s):                  45.36     
Total input tokens:                      30699     
Total generated tokens:                  3713      
Request throughput (req/s):              0.66      
Output token throughput (tok/s):         81.85     
Total Token throughput (tok/s):          758.60    
---------------Time to First Token----------------
Mean TTFT (ms):                          7537.00   
Median TTFT (ms):                        2710.46   
P99 TTFT (ms):                           33352.17  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          179.13    
Median TPOT (ms):                        183.42    
P99 TPOT (ms):                           267.47    
---------------Inter-token Latency----------------
Mean ITL (ms):                           179.13    
Median ITL (ms):                         130.00    
P99 ITL (ms):                            2415.50   
==================================================
```

```
(APIServer pid=118) INFO 09-02 05:22:07 [loggers.py:123] Engine 000: Avg prompt throughput: 2043.0 tokens/s, Avg generation throughput: 10.3 tokens/s, Running: 20 reqs, Waiting: 9 reqs, GPU KV cache usage: 77.6%, Prefix cache hit rate: 58.3%
(APIServer pid=118) INFO 09-02 05:22:17 [loggers.py:123] Engine 000: Avg prompt throughput: 613.9 tokens/s, Avg generation throughput: 111.3 tokens/s, Running: 24 reqs, Waiting: 5 reqs, GPU KV cache usage: 98.7%, Prefix cache hit rate: 45.8%
(APIServer pid=118) INFO 09-02 05:22:27 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 174.8 tokens/s, Running: 22 reqs, Waiting: 7 reqs, GPU KV cache usage: 97.2%, Prefix cache hit rate: 54.4%
(APIServer pid=118) INFO 09-02 05:22:37 [loggers.py:123] Engine 000: Avg prompt throughput: 408.8 tokens/s, Avg generation throughput: 23.0 tokens/s, Running: 7 reqs, Waiting: 0 reqs, GPU KV cache usage: 29.0%, Prefix cache hit rate: 52.6%
(APIServer pid=118) INFO 09-02 05:22:47 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 59.0 tokens/s, Running: 4 reqs, Waiting: 0 reqs, GPU KV cache usage: 17.7%, Prefix cache hit rate: 52.6%
```

```
============ Serving Benchmark Result ============
Successful requests:                     40        
Benchmark duration (s):                  69.31     
Total input tokens:                      40894     
Total generated tokens:                  4866      
Request throughput (req/s):              0.58      
Output token throughput (tok/s):         70.20     
Total Token throughput (tok/s):          660.20    
---------------Time to First Token----------------
Mean TTFT (ms):                          21833.91  
Median TTFT (ms):                        13510.68  
P99 TTFT (ms):                           54392.07  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          232.45    
Median TPOT (ms):                        254.74    
P99 TPOT (ms):                           369.96    
---------------Inter-token Latency----------------
Mean ITL (ms):                           232.45    
Median ITL (ms):                         128.87    
P99 ITL (ms):                            2418.73   
==================================================
```

```
(APIServer pid=118) INFO 09-02 05:24:57 [loggers.py:123] Engine 000: Avg prompt throughput: 1228.4 tokens/s, Avg generation throughput: 12.9 tokens/s, Running: 13 reqs, Waiting: 10 reqs, GPU KV cache usage: 50.6%, Prefix cache hit rate: 54.7%
(APIServer pid=118) INFO 09-02 05:25:07 [loggers.py:123] Engine 000: Avg prompt throughput: 817.0 tokens/s, Avg generation throughput: 6.5 tokens/s, Running: 20 reqs, Waiting: 19 reqs, GPU KV cache usage: 78.8%, Prefix cache hit rate: 52.7%
(APIServer pid=118) INFO 09-02 05:25:17 [loggers.py:123] Engine 000: Avg prompt throughput: 613.4 tokens/s, Avg generation throughput: 79.7 tokens/s, Running: 24 reqs, Waiting: 15 reqs, GPU KV cache usage: 98.3%, Prefix cache hit rate: 50.7%
(APIServer pid=118) INFO 09-02 05:25:27 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 175.1 tokens/s, Running: 22 reqs, Waiting: 17 reqs, GPU KV cache usage: 96.6%, Prefix cache hit rate: 51.4%
(APIServer pid=118) INFO 09-02 05:25:37 [loggers.py:123] Engine 000: Avg prompt throughput: 305.9 tokens/s, Avg generation throughput: 34.5 tokens/s, Running: 12 reqs, Waiting: 10 reqs, GPU KV cache usage: 50.2%, Prefix cache hit rate: 51.1%
(APIServer pid=118) INFO 09-02 05:25:47 [loggers.py:123] Engine 000: Avg prompt throughput: 813.2 tokens/s, Avg generation throughput: 5.3 tokens/s, Running: 14 reqs, Waiting: 2 reqs, GPU KV cache usage: 55.9%, Prefix cache hit rate: 49.9%
(APIServer pid=118) INFO 09-02 05:25:57 [loggers.py:123] Engine 000: Avg prompt throughput: 306.7 tokens/s, Avg generation throughput: 113.0 tokens/s, Running: 14 reqs, Waiting: 0 reqs, GPU KV cache usage: 59.2%, Prefix cache hit rate: 49.7%
(APIServer pid=118) INFO 09-02 05:26:07 [loggers.py:123] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 70.4 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.1%, Prefix cache hit rate: 49.7%
```
