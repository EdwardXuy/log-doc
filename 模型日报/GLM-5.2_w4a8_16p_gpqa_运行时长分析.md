# GLM-5.2 w4a8 16p gpqa 运行时长分析报告

> 生成日期：2026-09-05
> workflow run：https://github.com/sgl-project/sglang/actions/runs/33777992960
> job：https://github.com/sgl-project/sglang/actions/runs/33777992960/job/100993774837 —— 结论：**failure（超时）**

## 0. 用例定位（代码仓真实位置）

- 用例全名：`glm_5_2_w4a8_16p_gpqa`
- 测试类 / 方法：`TestNPUGLM_5_2_W4A8_16P_GPQA.test_npu_glm_5_2_w4a8_16p_gpqa`
- 文件路径：`test/registered/npu/accuracy/glm5_2/test_npu_glm_5_2_w4a8_16p_gpqa.py`
- 基线精度：`accuracy = 0.912`；评测集 `gpqa_diamond`（198 题）
- 关键启动参数（`GLM_5_2_W4A8_16P_TWO_NODE_OTHER_ARGS`）：
  - `--tp-size 32 --nnodes 2 --dp-size 8`
  - `--chunked-prefill-size 65536 --max-prefill-tokens 280000 --context-length 135000`
  - `--speculative-algorithm NEXTN --speculative-num-steps 3`
  - `--mem-fraction-static 0.76 --max-running-requests 32`
  - `--quantization modelslim --reasoning-parser glm45`
- 生成配置：`max_tokens=65536, temperature=1.0, eval_batch_size=32, stream=True`

## 1. 结论摘要

- 总耗时 **3h10m31s**（10:33:22Z → 13:43:53Z）。
- 第一轮评测实测 `mean_acc = 0.5`，远低于阈值 `0.8867`（0.912×修正系数），触发框架自动重试 `retrying (1/2)`。
- 第二轮评测跑到 83%（165/198）时，两个 pod 已各运行 3h5m，被 K8s/runner 兜底超时强杀（`Terminating 0 3h5m`），job 判失败。
- 本质：不是“性能慢到跑不完”，而是 **“首轮精度崩盘导致整轮重试”**，把单轮 1h38m 翻倍到 3h+，撞上 pod 3h 上限。

## 2. 时间线拆解

| 时间(Z) | 事件 | 耗时 |
|---|---|---|
| 10:33:22 | job 开始 | — |
| 10:35 → 10:40:36 | pod 调度 Pending→Running | ~5.5 min |
| 10:40:56 | Running test case | — |
| 10:42:11 → 10:45:50 | 主模型 `GlmMoeDsaForCausalLM` 权重加载（96 shards @1.99s/it） | 217~218 s |
| 10:45:56 → 10:46:20 | NEXTN 草稿模型 `GlmMoeDsaForCausalLMNextN` 加载（0.92 GB） | 21~23 s |
| ~10:46 → 10:47:16 | CUDA graph 捕获 | target_verify ≈ 44.8 s |
| 10:49 → 10:51:15 | evalscope 源码安装 + 启动 | ~1.5 min |
| 10:51:19 → 12:29:22 | **第一轮 gpqa 评测** | **1h38m（5883 s）** |
| 12:29:22 | `Accuracy 0.5 below threshold 0.8867..., retrying (1/2)` | — |
| 12:29:22 → 13:40:41 | 第二轮评测（跑到 165/198，83%） | ~1h11m，未完成 |
| 13:40:41 | 两 pod `Terminating 0 3h5m`，被强杀 | — |

## 3. 根因分析（定量）

### 3.1 精度 0.5 才是“超时”的直接原因
- 阈值 0.8867，实测 0.5（接近随机）。
- 非本次偶发：历史同一用例 8/27=0.4949、8/28=0.4646、9/2=0.5303，长期在 0.46~0.53 徘徊。
- 0.5 表示答案几乎全部判错，更可能是 **`reasoning-parser=glm45` 答案提取 / 输出截断 / 评测模板** 的问题，而非模型未收敛。
- 框架对“不达标”的执行是**全量重试一遍 198 题**，这是耗时放大器。

### 3.2 单轮就慢：平均每题输出 35123 token
- perf：`Avg Out = 35123.2 token`，`max_tokens=65536` + `temperature=1.0`（推理链全量输出）。
- 198 题累计输出 ≈ 695 万 token。
- decode 侧其实不慢：NEXTN 投机解码 `accept rate avg 0.796`（0.25~1.0），单 DP 组 `gen throughput avg 161.4 tok/s`，单请求折算 `43.08 tok/s`。
- 35123 / 43 ≈ 817s，与 `Avg Lat = 815.28 s` 完全吻合——**慢的根子是“每题要写 3.5 万 token”，不是生成速度**。

### 3.3 TTFT 122s 异常（两节点 deepep 部署下的首 token 慢）
- 输入仅 `Avg In = 270 token`，但 `TTFT = 122.23 s`（~122s），在两节点（`--tp-size 32`、每节点 16 NPU）+ `deepep` MoE 后端部署下首 token 明显偏慢。
- 但 prefill 原始吞吐并不低：`input throughput avg 392 tok/s`（峰值 3.4 万 tok/s）。
- 因此这 122s 的根因更偏向**排队/调度饿死**而非单纯 prefill 计算慢：`context-length 135000 / chunked-prefill 65536 / max-prefill 280000` 配合长 decode，长输出序列长期占满 decode 槽，新请求 prefill 在队列里被饿死，把 TTFT 抬到 2 分钟。

## 4. 关键日志证据

- 12:29:22 `The Final Accuracy from report: 0.5` / `Accuracy 0.5 below threshold 0.8867474747474747, retrying (1/2)...`
- 13:40:41 `ascend-...-sglang-node-0 ... Terminating 0 3h5m`（两节点均如此）
- 10:45:49 `Load weight end. elapsed=217.33s, type=GlmMoeDsaForCausalLM`
- 10:46:20 `Load weight end. elapsed=21.99s, type=GlmMoeDsaForCausalLMNextN`
- decode 样例：`accept len: 4.00, accept rate: 1.00 ... gen throughput (token/s): 118~225`

## 5. 改进建议（按优先级）

> 约束说明：草稿模型不量化（unquant）、`--linear-attn-verify-backend triton`、`--disable-radix-cache` 及本用例的 NEXTN 草稿未量化等「testcase 要求项」不在改动范围内，以下建议均不触碰上述配置。

1. 【最高优先级 · 正确性】先修精度：定位 `reasoning-parser=glm45` 的答案提取与 65536-token 长输出下的截断/格式问题，先跑 20 题小样本验证 score 回升，再全量，不要靠重试硬扛。
2. 【止损】在评测基类 `test_npu_accuracy_utils.py` 增加“首轮严重不达标（如 <0.5）直接 fail、不重试”，或限制重试轮次，避免单轮 1.5h × 2 的必然超时。
3. 【调度】压 TTFT：`chunked-prefill-size` 从 65536 调小、`max-prefill-tokens` 下调、`max-running-requests` 适当上调，减少 prefill 排队饿死；也可增加并行度/节点数（如拆 TP 或加节点）摊薄长上下文 prefill 压力。
4. 【输出长度】若评测目标是正确率而非长文本生成，可下调 `max_tokens`/`temperature`，或改用更短的评测模板/prompt（不改投机与量化等要求项），直接砍掉 3.5 万 token/题的主因。
5. 【元数据】`register_npu_ci(est_time=3600)` 与实际 1.5h/轮、可能 2 轮严重不符，建议上调到 ≥7200，给调度与超时留余量。
6. 【基础设施】权重加载 217s（96 shards）相对可控，可继续通过镜像预热/节点本地缓存压缩；pod 5.5min 调度等待可通过资源预占缓解。