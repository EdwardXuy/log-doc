# 失败用例开发对接说明（PR #711）

**目的**：本文档配合 PR 链接 + CI 日志链接一起发给对应开发。开发者可以快速定位：
- 这是哪个 GPU 源用例的 NPU 适配版
- 我（对接人）做了哪些参数适配
- 失败时 CI 日志里的原始报错片段
- 我对失败原因的初步分析（仅供方向参考，最终原因以开发诊断为准）

**主社区 GPU 测试源**：`https://github.com/sgl-project/sglang`
**本次 PR 仓库**：`https://github.com/Ascend/sglang/pull/711`（fork: `EdwardXuy/sglang-Ascend` → `Ascend/sglang:testcases`）
**本次 PR 分支**：`piecewise-ep-npu-testcases`
**CI 镜像**：`swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:main-cann8.5.0-a3`
**CI runner**：16-NPU A3（`linux-aarch64-a3-16`）
**本地 CI 日志**：`D:\debug-pr2\.trae\logs\final_27061329395.txt`（round 2 0.82 基线）/ `D:\debug-pr2\.trae\logs\final_27064148678.txt`（round 3 0.75 验证失败）

---

## 失败用例总览（6 个 hybrid 子用例 setUpClass 失败）

按字母序（unittest 实际执行顺序）：

| # | NPU 测试类 | GPU 源类 | 模型 | exit code | 失败阶段 |
|---|-----------|---------|------|----------|---------|
| 1 | `TestHybridDPEPCombined` | `Test21` | DeepSeek-V3.2-W8A8 | -9 (SIGKILL/OOM) | setUpClass 启动 |
| 2 | `TestHybridDeepEPAuto` | `Test10` | Qwen3-30B-A3B-W8A8 | -9 | setUpClass 启动 |
| 3 | `TestHybridDeepEPDPAttention` | `Test11` | Qwen3-30B-A3B-W8A8 | -9 | setUpClass 启动 |
| 4 | `TestHybridDeepEPFullStack` | `Test18` | Qwen3-30B-A3B-W8A8 | -9 | setUpClass 启动 |
| 5 | `TestHybridDPLMHead` | `Test06` | DeepSeek-V3.2-W8A8 | 1 (crash) | setUpClass 启动 |
| 6 | `TestHybridDPAttention` | `Test01` | DeepSeek-V3.2-W8A8 | 1 (crash) | setUpClass 启动 |
| 7 | `TestHybridFullDPAttention` | `Test02` | DeepSeek-V3.2-W8A8 | 1 (crash) | setUpClass 启动 |
| 8 | `TestHybridEPOnly` | `Test20` | DeepSeek-V3.2-W8A8 | -9 | setUpClass 启动 |

通过用例（4 个）：`TestHybridTPOnly` (mmlu=0.938)、`TestHybridDenseTPOne` (mmlu=0.938)、`TestHybridDPLMHead` (0.906, round 2 0.82 通过但 round 3 0.75 失败)、`TestHybridDPAttention` (0.906, 同上)。

---

## 通用 NPU 适配（适用 3 个失败用例及全部 10 个 hybrid 子用例）

### 启动参数基线
```bash
sglang serve --model-path <MODEL> \
  --trust-remote-code \
  --tp-size 16 \
  --quantization modelslim \
  --attention-backend ascend \
  --disable-cuda-graph \
  --mem-fraction-static 0.82 \
  --device npu --host 127.0.0.1 --port 11000
```
（+ 各子用例的特征参数，详见下文章节）

### 环境变量
```bash
PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
HCCL_BUFFSIZE=2048
HCCL_OP_EXPANSION_MODE=AIV
TASK_QUEUE_ENABLE=0
TRANSFORMERS_VERBOSITY=error
TRANSFORMERS_NO_ADVISORY_WARNINGS=1
PYTHONWARNINGS=ignore::FutureWarning,ignore::UserWarning,ignore::DeprecationWarning
```

### GPU 端 → NPU 端的关键差异

| 维度 | GPU 端 | NPU 端 |
|------|--------|--------|
| 注意力后端 | fa3 / triton / flashinfer | ascend（`--attention-backend ascend`） |
| 集合通信 | NCCL | HCCL |
| EP 后端 | mooncake elastic-ep | deepep（low_latency / auto / normal）+ EPLB + 冗余专家 |
| 拓扑 | 4× H100 / 8× H200 | 16× Ascend A3 |
| 模型 | `DeepSeek-V2-Lite`（MLA）、`DeepEP model` | `DeepSeek-V3.2-W8A8`、`Qwen3-30B-A3B-W8A8` |
| W8A8 量化 | 偶尔用 | 全部用（`--quantization modelslim`） |
| `--mem-fraction-static` | 0.5（GPU） | 0.82（hybrid）/ 0.9（elastic_ep） |

---

## 1. `TestHybridDPEPCombined`（GPU 源 `Test21`，DeepSeek-V3.2-W8A8）

### 1.1 GPU 源 (`Test21` in `test/registered/moe/test_hybrid_dp_ep_tp_mtp.py`)
- **测什么**：DP attention + EP（mixed sparse FFN 专家并行）
- **GPU 启动参数**（已剔除 GPU-only 项）：
  ```
  --tp 8
  --enable-dp-attention --dp 4
  --enable-eplb
  --ep 8
  ```
- **GPU 主社区链接**：https://github.com/sgl-project/sglang/blob/main/test/registered/moe/test_hybrid_dp_ep_tp_mtp.py

### 1.2 NPU 适配
- **模型替换**：`DeepSeek-V2-Lite` → `DeepSeek-V3.2-W8A8`
- **拓扑放大**：TP=8→16，DP=4 不变，EP=8→16
- **NPU 启动参数**（`test_npu_hybrid_dp_ep_tp_mtp.py` line 303）：
  ```
  --tp-size 16
  --quantization modelslim
  --enable-dp-attention
  --dp 4
  --ep-size 16
  --attention-backend ascend
  --disable-cuda-graph
  --mem-fraction-static 0.82
  ```

### 1.3 失败现象
- **退出码**：`code -9`（SIGKILL → OOM killer）
- **CI 日志关键片段**（`final_27061329395.txt` line 88815–88839）：
  ```
  2026-06-06T13:41:13.2778706Z .
  2026-06-06T13:41:13.2779989Z ERROR: setUpClass (__main__.TestHybridDPEPCombined)
  2026-06-06T13:41:13.2781032Z Traceback (most recent call last):
  2026-06-06T13:41:13.2782309Z   File "/sgl-workspace/sglang/python/sglang/test/test_utils.py", line 2188, in safe_setUpClass
  2026-06-06T13:41:13.2783932Z   File "/__w/sglang/sglang/test/registered/ascend/basic_function/parallel_strategy/expert_parallelism/GENERATED_20260606/test_npu_hybrid_dp_ep_tp_mtp.py", line 303, in setUpClass
  2026-06-06T13:41:13.2785217Z     cls.process = popen_launch_server(
  2026-06-06T13:41:13.2787041Z Exception: Server process exited with code -9. Check server logs for errors.
  ```
- **完整的 server 启动命令**（`final_27061329395.txt` line 27614）：
  ```
  command=sglang serve --model-path /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V3.2-W8A8 --trust-remote-code --tp-size 16 --quantization modelslim --enable-dp-attention --dp 4 --ep-size 16 --attention-backend ascend --disable-cuda-graph --mem-fraction-static 0.82 --device npu --host 127.0.0.1 --port 11000
  ```

### 1.4 涉及参数 / 特性
- `--ep-size 16`（专家并行 16 路）
- `--enable-dp-attention --dp 4`（数据并行 attention）
- `--mem-fraction-static 0.82`（静态内存比例）
- HCCL 通信（`HCCL_BUFFSIZE=2048`）

### 1.5 我的初步原因分析（待开发确认）
- 启动阶段 OOM：`mem-fraction-static=0.82` 在 16×A3 上可能不够 EP=16 + DP=4 同时启用时的通信缓冲
- 弹性 EP 的实现（EPLB + 冗余专家）在 16-NPU 上占用的 HCCL 缓冲可能超出 0.82 静态内存
- 建议方向：检查 ascend 端 EP=16 时 `--mem-fraction-static` 最低需求，或减小 `HCCL_BUFFSIZE`（已设为 2048）

---

## 2. `TestHybridDeepEPAuto`（GPU 源 `Test10`，Qwen3-30B-A3B-W8A8）

### 2.1 GPU 源 (`Test10`)
- **测什么**：DeepEP 后端（auto 模式）+ TP 基础
- **GPU 启动参数**：
  ```
  --tp 8
  --moe-a2a-backend deepep --deepep-mode auto
  ```
- **GPU 主社区链接**：https://github.com/sgl-project/sglang/blob/main/test/registered/moe/test_hybrid_dp_ep_tp_mtp.py

### 2.2 NPU 适配
- **模型替换**：GPU `DEFAULT_DEEPEP_MODEL_NAME_FOR_TEST` → NPU `QWEN3_30B_A3B_W8A8_WEIGHTS_PATH`
- **拓扑放大**：TP=8→16
- **NPU 启动参数**（`test_npu_hybrid_dp_ep_tp_mtp.py` line 349）：
  ```
  --tp-size 16
  --quantization modelslim
  --moe-a2a-backend deepep
  --deepep-mode auto
  --attention-backend ascend
  --disable-cuda-graph
  --mem-fraction-static 0.82
  ```

### 2.3 失败现象
- **退出码**：`code -9`
- **CI 日志关键片段**（`final_27061329395.txt` line 88841 附近）：
  ```
  2026-06-06T13:41:13.2788021Z ERROR: setUpClass (__main__.TestHybridDeepEPAuto)
  2026-06-06T13:41:13.2795192Z ...
  2026-06-06T13:41:13.2796037Z Exception: Server process exited with code -9. Check server logs for errors.
  ```
- **完整 server 启动命令**（`final_27061329395.txt` line 52929）：
  ```
  command=sglang serve --model-path /root/.cache/modelscope/hub/models/Qwen/Qwen3-30B-A3B-w8a8 --trust-remote-code --tp-size 16 --quantization modelslim --moe-a2a-backend deepep --deepep-mode auto --attention-backend ascend --disable-cuda-graph --mem-fraction-static 0.82 --device npu --host 127.0.0.1 --port 11000
  ```

### 2.4 涉及参数 / 特性
- `--moe-a2a-backend deepep`
- `--deepep-mode auto`（注意：低时延测试 `TestElasticEPTP` 用 `low_latency` 是通过的；此例用 `auto` 失败）
- Qwen3-30B-A3B（30B 总参 / 3B 激活）W8A8 模型

### 2.5 我的初步原因分析（待开发确认）
- DeepEP `auto` 模式在 16-NPU 上首次启动时的初始化内存开销可能比 `low_latency` 模式大
- 启动阶段 OOM（与 setUpClass 启动同时刻被打死）
- 建议方向：检查 ascend 后端 `--deepep-mode auto` 的内存峰值，或考虑切换到 `low_latency` 模式

---

## 3. `TestHybridDeepEPDPAttention`（GPU 源 `Test11`，Qwen3-30B-A3B-W8A8）

### 3.1 GPU 源 (`Test11`)
- **测什么**：DeepEP + DP attention 组合
- **GPU 启动参数**：
  ```
  --tp 8
  --enable-dp-attention --dp 4
  --moe-a2a-backend deepep --deepep-mode normal
  ```

### 3.2 NPU 适配
- **模型替换**：同上
- **拓扑放大**：TP=8→16，DP=4 不变
- **NPU 启动参数**（`test_npu_hybrid_dp_ep_tp_mtp.py` line 393）：
  ```
  --tp-size 16
  --quantization modelslim
  --enable-dp-attention
  --dp 4
  --moe-a2a-backend deepep
  --deepep-mode auto
  --attention-backend ascend
  --disable-cuda-graph
  --mem-fraction-static 0.82
  ```
- **完整 server 启动命令**（`final_27061329395.txt` line 55560）：
  ```
  command=sglang serve --model-path /root/.cache/modelscope/hub/models/Qwen/Qwen3-30B-A3B-w8a8 --trust-remote-code --tp-size 16 --quantization modelslim --enable-dp-attention --dp 4 --moe-a2a-backend deepep --deepep-mode auto --attention-backend ascend --disable-cuda-graph --mem-fraction-static 0.82 --device npu --host 127.0.0.1 --port 11000
  ```

### 3.3 失败现象
- **退出码**：`code -9`
- **CI 日志关键片段**（`final_27061329395.txt` line 88854 附近）：
  ```
  2026-06-06T13:41:13.2795192Z ERROR: setUpClass (__main__.TestHybridDeepEPDPAttention)
  2026-06-06T13:41:13.2802080Z ...
  2026-06-06T13:41:13.2802741Z Exception: Server process exited with code -9. Check server logs for errors.
  ```

### 3.4 涉及参数 / 特性
- DeepEP `auto` + DP attention + Qwen3-30B-A3B
- 与用例 2 (`TestHybridDeepEPAuto`) 的差异仅在加了 `--enable-dp-attention --dp 4`

### 3.5 我的初步原因分析（待开发确认）
- 启动阶段 OOM，与用例 2 同模式
- DeepEP `auto` + DP=4 联合时的 HCCL 通信缓冲需求更大

---

## 4. `TestHybridDeepEPFullStack`（GPU 源 `Test18`，Qwen3-30B-A3B-W8A8）

### 4.1 GPU 源 (`Test18`)
- **测什么**：DeepEP + DP attention + dense TP=1 + DP LM head（全栈组合）
- **GPU 启动参数**：
  ```
  --tp 8
  --enable-dp-attention --dp 4
  --moe-dense-tp-size 1
  --enable-dp-lm-head
  --moe-a2a-backend deepep --deepep-mode normal
  ```

### 4.2 NPU 适配
- **NPU 启动参数**（`test_npu_hybrid_dp_ep_tp_mtp.py` line 440）：
  ```
  --tp-size 16
  --quantization modelslim
  --enable-dp-attention
  --dp 4
  --moe-dense-tp-size 1
  --enable-dp-lm-head
  --moe-a2a-backend deepep
  --deepep-mode auto
  --attention-backend ascend
  --disable-cuda-graph
  --mem-fraction-static 0.82
  ```
- **完整 server 启动命令**（`final_27061329395.txt` line 58124）：
  ```
  command=sglang serve --model-path /root/.cache/modelscope/hub/models/Qwen/Qwen3-30B-A3B-w8a8 --trust-remote-code --tp-size 16 --quantization modelslim --enable-dp-attention --dp 4 --moe-dense-tp-size 1 --enable-dp-lm-head --moe-a2a-backend deepep --deepep-mode auto --attention-backend ascend --disable-cuda-graph --mem-fraction-static 0.82 --device npu --host 127.0.0.1 --port 11000
  ```

### 4.3 失败现象
- **退出码**：`code -9`
- **CI 日志关键片段**（`final_27061329395.txt` line 88867 附近）：
  ```
  2026-06-06T13:41:13.2802080Z ERROR: setUpClass (__main__.TestHybridDeepEPFullStack)
  2026-06-06T13:41:13.2815207Z ...
  2026-06-06T13:41:13.2815807Z Exception: Server process exited with code -9. Check server logs for errors.
  ```

### 4.4 涉及参数 / 特性
- DeepEP + DP attention + dense TP=1 + DP LM head 全栈
- 同时启用 3 个并行维度，集合通信拓扑最复杂

### 4.5 我的初步原因分析（待开发确认）
- 全栈组合下 HCCL 通信资源竞争加剧
- 启动阶段 OOM

---

## 5. `TestHybridDPLMHead`（GPU 源 `Test06`，DeepSeek-V3.2-W8A8）

### 5.1 GPU 源 (`Test06`)
- **测什么**：DP attention + DP LM head
- **GPU 启动参数**：
  ```
  --tp 8
  --enable-dp-attention --dp 4
  --enable-dp-lm-head
  ```

### 5.2 NPU 适配
- **NPU 启动参数**（`test_npu_hybrid_dp_ep_tp_mtp.py` line 219）：
  ```
  --tp-size 16
  --quantization modelslim
  --enable-dp-attention
  --dp 4
  --enable-dp-lm-head
  --attention-backend ascend
  --disable-cuda-graph
  --mem-fraction-static 0.82
  ```

### 5.3 失败现象
- **退出码**：`code 1`（crash，非 OOM）
- **CI 日志关键片段**（`final_27061329395.txt` line 88841 之后，line 88893 前的 141057 行附近）：
  ```
  2026-06-06T13:41:13.5942314Z ERROR: setUpClass (__main__.TestHybridDPLMHead)
  2026-06-06T13:41:13.5943085Z Traceback (most recent call last):
  2026-06-06T13:41:13.5943721Z   File "/sgl-workspace/sglang/python/sglang/test/test_utils.py", line 2188, in safe_setUpClass
  2026-06-06T13:41:13.5944832Z   File "/__w/sglang/sglang/test/registered/ascend/basic_function/parallel_strategy/expert_parallelism/GENERATED_20260606/test_npu_hybrid_dp_ep_tp_mtp.py", line 219, in setUpClass
  2026-06-06T13:41:13.5945583Z     cls.process = popen_launch_server(
  2026-06-06T13:41:13.5945967Z                   ^^^^^^^^^^^^^^^^^^^^
  2026-06-06T13:41:13.5946413Z   File "/sgl-workspace/sglang/python/sglang/test/test_utils.py", line 1027, in popen_launch_server
  2026-06-06T13:41:13.5947093Z     raise Exception(error_msg + ". Check server logs for errors.")
  2026-06-06T13:41:13.5947835Z Exception: Server process exited with code 1. Check server logs for errors.
  ```

### 5.4 涉及参数 / 特性
- `--enable-dp-attention --dp 4`（**和用例 6/7 一样的 DP attention 路径**）
- `--enable-dp-lm-head`
- 不涉及 deepep / EP

### 5.5 我的初步原因分析（待开发确认）
- exit code 1（不是 OOM）→ 不是内存问题，是 ascend 后端 assertion / RuntimeError
- **关键点：所有启用了 `--enable-dp-attention` 的 3 个用例（TestDPAttention / TestDPLMHead / TestFullDPAttention）全部 exit code 1**
- 怀疑方向：DP attention 在 ascend 后端与 `--enable-dp-lm-head` / `--dp 4/16` 的组合下有未处理的 assertion

---

## 6. `TestHybridDPAttention`（GPU 源 `Test01`，DeepSeek-V3.2-W8A8）

### 6.1 GPU 源 (`Test01`)
- **测什么**：DP attention（DP=4）基础
- **GPU 启动参数**：
  ```
  --tp 8
  --enable-dp-attention --dp 4
  ```

### 6.2 NPU 适配
- **NPU 启动参数**（`test_npu_hybrid_dp_ep_tp_mtp.py` line 94）：
  ```
  --tp-size 16
  --quantization modelslim
  --enable-dp-attention
  --dp 4
  --attention-backend ascend
  --disable-cuda-graph
  --mem-fraction-static 0.82
  ```

### 6.3 失败现象
- **退出码**：`code 1`
- **CI 日志关键片段**（`final_27061329395.txt` line 88815 附近）：
  ```
  2026-06-06T13:41:13.5897395Z ERROR: setUpClass (__main__.TestHybridDPAttention)
  2026-06-06T13:41:13.5898395Z   File "/sgl-workspace/sglang/python/sglang/test/test_utils.py", line 2188, in safe_setUpClass
  2026-06-06T13:41:13.5899546Z   File "/__w/sglang/sglang/test/registered/ascend/basic_function/parallel_strategy/expert_parallelism/GENERATED_20260606/test_npu_hybrid_dp_ep_tp_mtp.py", line 94, in setUpClass
  2026-06-06T13:41:13.5902548Z     cls.process = popen_launch_server(
  2026-06-06T13:41:13.5904364Z Exception: Server process exited with code 1. Check server logs for errors.
  ```

### 6.4 涉及参数 / 特性
- `--enable-dp-attention --dp 4`
- 不涉及 deepep / EP

### 6.5 我的初步原因分析（待开发确认）
- 与用例 5/7 模式相同，全部 DP attention 用例失败
- 怀疑方向：DP attention 路径在 ascend 后端的稳定性问题

---

## 7. `TestHybridFullDPAttention`（GPU 源 `Test02`，DeepSeek-V3.2-W8A8）

### 7.1 GPU 源 (`Test02`)
- **测什么**：全 DP attention（DP=8）
- **GPU 启动参数**：
  ```
  --tp 8
  --enable-dp-attention --dp 8
  ```

### 7.2 NPU 适配
- **拓扑放大**：DP=8→DP=16（TP=16 拓扑下 DP=16 才是全 DP）
- **NPU 启动参数**（`test_npu_hybrid_dp_ep_tp_mtp.py` line 132）：
  ```
  --tp-size 16
  --quantization modelslim
  --enable-dp-attention
  --dp 16
  --attention-backend ascend
  --disable-cuda-graph
  --mem-fraction-static 0.82
  ```

### 7.3 失败现象
- **退出码**：`code 1`
- **CI 日志关键片段**（`final_27061329395.txt` line 88893 附近）：
  ```
  2026-06-06T13:41:13.2815717Z ERROR: setUpClass (__main__.TestHybridFullDPAttention)
  2026-06-06T13:41:13.2816143Z ...
  2026-06-06T13:41:13.2816517Z Exception: Server process exited with code 1. Check server logs for errors.
  ```

### 7.4 涉及参数 / 特性
- `--enable-dp-attention --dp 16`（极端 DP，TP=16 时所有 rank 都属不同 DP 组）

### 7.5 我的初步原因分析（待开发确认）
- 与用例 5/6 模式相同，DP attention 失败
- DP=16 可能是更极端的 case

---

## 8. `TestHybridEPOnly`（GPU 源 `Test20`，DeepSeek-V3.2-W8A8）

### 8.1 GPU 源 (`Test20`)
- **测什么**：EP=8 专家并行
- **GPU 启动参数**：
  ```
  --tp 8
  --ep 8
  --enable-eplb
  ```

### 8.2 NPU 适配
- **拓扑放大**：EP=8→EP=16
- **NPU 启动参数**（`test_npu_hybrid_dp_ep_tp_mtp.py` line 262）：
  ```
  --tp-size 16
  --quantization modelslim
  --ep-size 16
  --attention-backend ascend
  --disable-cuda-graph
  --mem-fraction-static 0.82
  ```
- **完整 server 启动命令**（`final_27061329395.txt` line 61727）：
  ```
  command=sglang serve --model-path /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V3.2-W8A8 --trust-remote-code --tp-size 16 --quantization modelslim --ep-size 16 --attention-backend ascend --disable-cuda-graph --mem-fraction-static 0.82 --device npu --host 127.0.0.1 --port 11000
  ```

### 8.3 失败现象
- **退出码**：`code -9`
- **CI 日志关键片段**（`final_27061329395.txt` line 88880 附近）：
  ```
  2026-06-06T13:41:13.2808946Z ERROR: setUpClass (__main__.TestHybridEPOnly)
  2026-06-06T13:41:13.2815717Z ...
  2026-06-06T13:41:13.2816807Z Exception: Server process exited with code -9. Check server logs for errors.
  ```

### 8.4 涉及参数 / 特性
- `--ep-size 16`（专家并行 16 路）
- 不涉及 deepep（用 ascend 默认 EPLB）

### 8.5 我的初步原因分析（待开发确认）
- 启动阶段 OOM
- 与 `TestElasticEPTP`（用 `--ep-num-redundant-experts 32` + `deepep low_latency` + `mem-fraction-static 0.9`）对比，0.82 在 EP=16 上可能不够

---

## 9. 错误片段参考：HCCL 内存分配失败（出自 elastic_ep 启动阶段 round 2 0.82 试跑）

虽然 elastic_ep 最终通过了（gsm8k=0.95），但启动早期也短暂出现过 HCCL 内存分配失败，可作为参考：

```
.trae\logs\final_27061329395.txt:5967: RuntimeError: InnerRunOpApi:../torch_npu/csrc/framework/OpParamMaker.cpp:287 OPS function error: HcclAllreduce, error code is 24
.trae\logs\final_27061329395.txt:5969: [PID: 4906] 2026-06-06-12:17:01.267.772 Memory_Allocation_Failure(EL0004): Failed to allocate memory requested by HCCL module.
.trae\logs\final_27061329395.txt:6021: RuntimeError: InnerRunOpApi:../torch_npu/csrc/framework/OpParamMaker.cpp:287 OPS function error: HcclAllreduce, error code is 24
```

（elastic_ep 后续重试成功）

---

## 10. 联系开发时的关键问题清单

请按下列方向对号入座找开发：

| 问题 | 用例 | exit code | 涉及特性 | 找谁 |
|------|------|----------|---------|------|
| 1. DP attention 在 ascend 后端启动崩溃（3 个用例） | TestDPAttention / TestDPLMHead / TestFullDPAttention | 1 | `--enable-dp-attention` | DP attention 负责开发 |
| 2. DeepEP `auto` 模式在 16-NPU 上 OOM（3 个用例） | TestDeepEPAuto / TestDeepEPDPAttention / TestDeepEPFullStack | -9 | `--deepep-mode auto` | DeepEP 负责开发 |
| 3. EP=16 + DeepSeek 大模型 OOM（1 个用例） | TestEPOnly | -9 | `--ep-size 16` + 静态内存预算 | EPLB / 内存优化 / 集合通信开发 |
| 4. DP attention + EP 联合 OOM（1 个用例） | TestDPEPCombined | -9 | `--dp 4 --ep-size 16` | DP attention × EP 联合开发 |

---

## 11. 我已尝试的修复（供参考）

1. **日志抑制**（PR commit `788465b`）— 有效：消除"transformers 打印过多导致 CI 失败"的问题
2. **`mem-fraction-static 0.82 → 0.9`**（PR commit `53f7545`，仅 elastic_ep）— 有效：elastic_ep 通过
3. **`mem-fraction-static 0.82 → 0.75`**（PR commit `ba8219a`，全部 10 个 hybrid）— 失败：使通过数从 4/10 降到 2/10
4. **回退到 0.82**（PR commit `0995eb4`，当前最新）— 等待 CI 验证

**未尝试**（需要开发确认后再做）：
- 给失败用例单独降低 `--max-running-requests` / `--chunked-prefill-size`
- 关闭 `--enforce-piecewise-cuda-graph`（已默认关闭）
- 调整 `HCCL_BUFFSIZE=2048` 上下
- 切换 DeepEP `auto` → `low_latency`（已对比 elastic_ep 的 low_latency 模式是过的）

---

## 12. 本地 CI 日志位置（开发可直接下载）

- **PR round 2 完整日志**（0.82 基线，含本次 6 个失败用例的报错）：
  - `D:\debug-pr2\.trae\logs\final_27061329395.txt`（10.8 MB）
- **PR round 3 完整日志**（0.75 验证失败参考）：
  - `D:\debug-pr2\.trae\logs\final_27064148678.txt`（16.4 MB）
- **PR round 4 日志**（0.82 重新提交，待生成）：
  - `D:\debug-pr2\.trae\logs\final_27114881322.txt`
- **状态文件**：`D:\debug-pr2\.trae\logs\state_<run_id>.json`

**GitHub 原始链接**（开发自己再下也行）：
- 失败用例的 CI logs：https://github.com/Ascend/sglang/actions/runs/27061329395
- 也可走 `https://api.github.com/repos/Ascend/sglang/actions/runs/27061329395/jobs` 取 job id 后下 logs
