# Kimi-K3 w4a8 32p gpqa 运行时长分析报告

> 生成日期：2026-09-05
> workflow run：https://github.com/sgl-project/sglang/actions/runs/33777992960
> job：https://github.com/sgl-project/sglang/actions/runs/33777992960/job/101045025846 —— 结论：**success（2h20m33s）**

## 0. 用例定位（代码仓真实位置）

- 用例全名：`kimi_k3_w4a8_32p_gpqa`
- 测试类 / 方法：`TestNPUKimiK3_W4A8_32P_GPQA.test_npu_kimi_k3_w4a8_32p_gpqa`
- 文件路径：`test/registered/npu/accuracy/kimi_k3/test_npu_kimi_k3_w4a8_32p_gpqa.py`
- 基线精度：`accuracy = 0.935`；评测集 `gpqa_diamond`（198 题）
- 关键启动参数：
  - `--tp-size 64 --nnodes 4 --dp-size 4`
  - `--chunked-prefill-size 8192 --mem-fraction-static 0.72 --max-running-requests 64`
  - `--speculative-algorithm DSPARK --speculative-dspark-block-size 7`
  - `--speculative-draft-model-quantization unquant`（要求，不可改）
  - `--linear-attn-verify-backend triton`（要求，不可改）
  - `--disable-radix-cache`（要求，不可改）
  - `--reasoning-parser kimi_k3 --moe-a2a-backend deepep`
- 生成配置：`max_tokens=131072, temperature=1.0, top_p=0.95, extra_body={"reasoning_effort":"max"}, timeout=10000`

## 1. 结论摘要

- 结果：**success**，一轮通过，`mean_acc = 0.9444 ≥ 0.935`。
- 但总耗时 **2h20m33s**（13:44:31Z → 16:05:04Z），主要卡在两处：**评测生成 1h42m40s**（占 73%）与 **主模型权重加载 13.1min**（占 9%）。
- 慢的核心：**DSPARK 投机解码 accept rate 仅 0.20，投机近乎失效**；叠加线性注意力 triton verify、prefill 8.6 tok/s、每题输出 7128 token。

## 2. 时间线拆解

| 时间(Z) | 事件 | 耗时 |
|---|---|---|
| 13:44:31 | job 开始 | — |
| 13:46:26 → 13:52:59 | pod 调度 Pending→Running（4 节点 64 NPU 等资源） | ~6.5 min |
| 13:53:16 | Running test case | — |
| 13:54:28 → 14:07:38 | 主模型 `KimiK3ForConditionalGeneration` 权重加载（351 shards） | 788~790 s |
| 14:07:39 → 14:07:47 | 草稿模型 `DSparkDraftModel` 加载（0.90 GB） | 7~8 s |
| ~14:08 → 14:10 | CUDA graph 捕获 | target_verify ≈ 113.4 s |
| 14:19 → 16:01:46 | **唯一一轮 gpqa 评测** | **1h42m40s（6160 s）** |

引擎启动计时：`load_weight=798.03s`、`scheduler_e2e=935.75s`。

## 3. 根因分析（定量）

### 3.1 DSPARK 投机 accept rate 仅 0.20，投机基本失效（最大根因）
- 对全日志 618 条 `Decode batch`（DP0 视角）统计：**accept rate min 0.08 / max 0.37 / avg 0.203**（对应 `accept len` 仅 2.6~3.1）。
- 对比同 run 里 GLM 用 `NEXTN` 的 accept rate 0.796。
- 20% 命中率意味着 80% 草稿 token 被丢弃，投机加速被严重抵消。根因落在**采样分布太熵高**：`temperature=1.0 + top_p=0.95 + reasoning_effort=max`，目标分布扁平，草稿模型难命中。
- 属 cannot-change 的要求项（草稿 unquant、linear-attn triton、disable-radix-cache）已排除在改进外。

### 3.2 线性注意力 verify + 低并发导致 decode/prefill 双慢
- Kimi-K3 为 hybrid linear-attention（日志 `mamba num: 8, mamba usage: 0.50`，一半层为 Mamba）。
- `--linear-attn-verify-backend triton`（要求项）在 NPU 上非最快路径。
- 瞬时生成吞吐波动极大：`gen throughput` min 1.2 / max 215.2 / avg 88.1 tok/s（单 DP 组），requests 只有 8 条时能掉到 1.22 tok/s。
- **prefill 吞吐 avg 仅 8.61 tok/s**（min 0.73），最终 `Avg TTFT = 20.9s`。

### 3.3 输出长 + 每请求端到端 513s
- `max_tokens=131072 + reasoning_effort=max` → 平均每题输出 `7128.3 token`。
- 198 题 × 7128 ≈ 141 万 token 输出，在 6160s 内跑完。
- 单请求：`Avg Lat = 513.28s`、`Avg Thpt = 13.89 tok/s`。

### 3.4 权重加载 788s（次根因）
- 主模型 351 个 shard（GLM 仅 96），`mem usage 32.84 GB/rank`。
- 4 节点 × 16 NPU = 64 个 rank 同时从同一份模型存储读 351 shard，I/O 带宽被占满。
- 草稿模型仅 0.90GB/8s，说明瓶颈不在计算，而在**大模型 shard 的读取带宽**。

## 4. 关键日志证据

- 14:07:37 `Load weight end. elapsed=788.15s, type=KimiK3ForConditionalGeneration`
- 14:07:46 `Load weight end. elapsed=6.71s, type=DSparkDraftModel`
- decode 样例：`accept len: 3.09, accept rate: 0.30 ... accept len: 2.70, accept rate: 0.24`
- prefill 样例：`input throughput (token/s): 2.32 / 6.38 / 32.85`

## 5. 改进建议（按优先级）

> 约束说明：草稿模型 `unquant`（不量化）、`--linear-attn-verify-backend triton`、`--disable-radix-cache` 为 testcase 要求，不在改动范围内；以下建议均不触碰这三项。

1. 【最高优先级 · 提 accept rate】下调采样温度：把 `temperature` 从 1.0 降到 0.6 或改为 greedy，收紧 `top_p`，使目标分布峰值化，显著提升 DSPARK 草稿命中率（不改草稿模型本身）。
2. 【调度/吞吐】优化 prefill 与并发：`chunked-prefill-size`、`max-running-requests`、`STREAMS_PER_DEVICE` 等可调参数做 A/B，缓解 prefill 8.6 tok/s 与 decode 掉到 1.2 tok/s 的低效段。
3. 【输出长度】评估类场景可适度下调 `max_tokens`（131072）或去掉 `reasoning_effort=max`，缩短每题 7128 token 的输出。
4. 【权重加载提速】权重打进镜像或节点本地 NVMe 缓存，避免 64 rank 从共享盘抢带宽；提高 `enable_multithread_load` 并行度/预取，或合并 shard。
5. 【元数据】`register_npu_ci(est_time=3600)` 与实际 2h20m 严重不符，建议上调到 ≥7200，给调度与超时留余量。
6. 【基础设施】pod 6.5min 调度等待（4 节点 64 NPU）可通过资源预占/复用缓解。