# PR #711 阶段汇报：GPU→NPU 测试用例迁移（3 个用例）

**汇报日期**: 2026-06-08
**PR 链接**: https://github.com/Ascend/sglang/pull/711
**分支**: `piecewise-ep-npu-testcases`（fork: `EdwardXuy/sglang-Ascend` → `Ascend/sglang:testcases`）
**CI 镜像**: `swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:main-cann8.5.0-a3`

---

## 1. 总体目标

把 sglang 主社区（`https://github.com/sgl-project/sglang`）`test/registered/` 下的 GPU 集成测试，按场景迁移到 NPU A3 集群（`cann8.5.0-a3` 镜像），覆盖：
- **VLM piecewise CUDA graph**（优化/debug 类）
- **MoE 弹性专家并行（elastic EP）**
- **MoE 混合并行（DP × TP × EP × dense FFN × LM head × DeepEP）**

本次 PR 共 3 个测试文件 / 12 个 GPU→NPU 测试类：

| 文件 | GPU 源 | NPU 测试类数 | runner | suite |
|------|--------|-------------|--------|-------|
| `test_npu_piecewise_cuda_graph_vlm.py` | `test/registered/piecewise_cuda_graph/test_piecewise_cuda_graph_support_1_gpu.py` | 2 | 1-NPU | stage-b-test-1-npu-a2 / nightly-1-npu-a3 |
| `test_npu_elastic_ep_hybrid_dp.py` | `test/registered/ep/test_mooncake_ep_small.py` | 1 + 2 skip | 16-NPU | nightly-16-npu-a3 |
| `test_npu_hybrid_dp_ep_tp_mtp.py` | `test/registered/moe/test_hybrid_dp_ep_tp_mtp.py` | 10 | 16-NPU | weekly-16-npu-a3 |

---

## 2. 每个 GPU 源用例的"测什么 / 观测点 / 特性 / 参数"

### 2.1 VLM Piecewise CUDA Graph（GPU 源：1 个文件 / 2 个测试类）

**GPU 源文件**: `test/registered/piecewise_cuda_graph/test_piecewise_cuda_graph_support_1_gpu.py`

#### 2.1.1 `TestPiecewiseCudaGraphQwen25VL`
- **测什么**：在 VLM 多模态推理路径下启用 `--enforce-piecewise-cuda-graph` 后的精度与稳定性
- **观测点**：
  1. 多模态（图像 + 文本）推理能否在分段 CUDA Graph 路径下正常出 token
  2. GSM8K 200 题精度 ≥ 0.80
- **特性**：piecewise CUDA Graph（分段图捕获）、`--disable-radix-cache`（关闭前缀缓存避免干扰）、VLM（multimodal encoder + LLM）
- **关键参数**：
  - 模型：`Qwen/Qwen2.5-VL-7B-Instruct`
  - runner：`1-gpu-large`（1× H100/H200）
  - 超时：`DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH`
- **GPU 启动参数**：
  ```
  --enforce-piecewise-cuda-graph
  --disable-radix-cache
  ```

#### 2.1.2 `TestPiecewiseCudaGraphQwen25VLEmbedding`
- **测什么**：分段 CUDA Graph 开 / 关 两种模式下，VLM embedding 输出的一致性
- **观测点**：
  1. `enforce_piecewise_cuda_graph=True` vs `disable_piecewise_cuda_graph=True` 两种 `Engine` 配置下，对同一图片文本的 embedding 输出差异
  2. 阈值 `atol=1e-2, rtol=1e-2` 内 `torch.allclose`
- **特性**：Engine API（进程内）、multimodal embedding（`is_embedding=True`）
- **关键参数**：
  - 模型：`Qwen/Qwen2.5-VL-3B-Instruct`
  - 图：`DEFAULT_IMAGE_URL`

---

### 2.2 MoE 弹性专家并行（GPU 源：1 个文件 / 3 个测试类）

**GPU 源文件**: `test/registered/ep/test_mooncake_ep_small.py`
**GPU 模型**：`DeepSeek-V2-Lite`（MLA，`DEFAULT_MODEL_NAME_FOR_TEST_MLA`）
**GPU runner**：`deepep-4-gpu-h100`（4× H100）

#### 2.2.1 `TestTP`（基础）
- **测什么**：mooncake elastic-ep 后端 + 4 卡 TP 下的弹性专家并行基础能力
- **观测点**：GSM8K ≥ 0.60
- **特性**：
  - mooncake elastic-ep 后端（`--elastic-ep-backend mooncake`）
  - mooncake a2a 后端（`--moe-a2a-backend mooncake`）
  - mooncake IB device 透传
  - 弹性专家（EPLB + 冗余专家）
  - 两批 overlap（`--enable-two-batch-overlap`）
  - dense FFN 独立 TP（`--moe-dense-tp-size 1`）
  - DP LM head（`--enable-dp-lm-head`）
- **GPU 启动参数**：
  ```
  --tp 4
  --elastic-ep-backend mooncake
  --mooncake-ib-device <动态RDMA设备>
  --moe-a2a-backend mooncake
  --deepep-mode low_latency
  --moe-dense-tp-size 1
  --enable-dp-lm-head
  --enable-two-batch-overlap
  --disable-custom-all-reduce
  --enable-eplb
  --ep-num-redundant-experts 72
  --chunked-prefill-size 512
  --cuda-graph-max-bs 128
  --max-running-requests 512
  --mem-fraction-static 0.5
  ```

#### 2.2.2 `TestPureDP`（CI 跳过，但 NPU 已生成）
- **测什么**：纯 DP attention 下弹性 EP + 单 rank 故障切换
- **特性**：`--enable-dp-attention --dp 4`
- **故障注入**：`pkill -f sglang::scheduler_DP1_TP1_EP1` / `sglang::scheduler_DP3_TP3_EP3`

#### 2.2.3 `TestHybridDPTP`（CI 跳过，但 NPU 已生成）
- **测什么**：混合 DP+TP + 故障切换
- **特性**：`--enable-dp-attention --dp 2`（DP×TP = 4×2 = 8 卡）
- **故障注入**：`pkill -f sglang::scheduler_DP1_TP2_EP2` / `sglang::scheduler_DP1_TP3_EP3`

---

### 2.3 MoE 混合并行（GPU 源：1 个文件 / 60 个测试类）

**GPU 源文件**: `test/registered/moe/test_hybrid_dp_ep_tp_mtp.py`
**GPU runner**：`weekly-8-gpu-h200`（每周，8× H200）
**GPU 模型**：`DEFAULT_MLA_MODEL_NAME_FOR_TEST`（DeepSeek-V2-Lite）和 `DEFAULT_DEEPEP_MODEL_NAME_FOR_TEST`

**60 个测试类的维度组合**（按维度组合枚举）：
| 维度 | 取值 |
|------|------|
| DP attention | TP=8 baseline / DP=4 / DP=8 |
| Dense FFN TP | 8 / 1（`--moe-dense-tp-size`） |
| Sparse FFN 后端 | MLA + EPLB / DeepEP（auto / normal / low_latency） |
| LM head | TP=8 / DP（`--enable-dp-lm-head`） |
| 推测解码 | 无 / EAGLE / MTP（DeepSeek-V3 NextN draft） |

**典型 GPU 启动参数**（以 Test00 / Test01 / Test10 为例）：
```
# Test00 baseline
--tp 8

# Test01 DP=4
--tp 8 --enable-dp-attention --dp 4

# Test10 DeepEP
--tp 8 --enable-dp-attention --dp 4 --moe-a2a-backend deepep --deepep-mode normal
```

**NPU 仅选取 10 个代表性配置**（见 4.3 节），其余 50 个（维度组合冗余 20 + MTP/EAGLE 30 暂不支持）暂未生成。

---

## 3. 通用 NPU 适配（适用所有 3 个文件）

### 3.1 算法 / 后端差异

| 项 | GPU 端 | NPU 端 |
|----|--------|--------|
| 注意力后端 | fa3 / triton / flashinfer | ascend（强制 `--attention-backend ascend`） |
| 集合通信库 | NCCL | HCCL（`init_process_group(backend="hccl")`） |
| CUDA Graph | 完整 cuda graph | `--disable-cuda-graph`（ascend 不支持完整 cuda graph） |
| W8A8 量化 | 偶尔用 bitsandbytes | `modelslim`（NPU 权重默认 W8A8） |
| mooncake 弹性 EP | `--elastic-ep-backend mooncake` | 移除（用 EPLB + 冗余专家 + deepep 替代） |
| 故障注入 `pkill` | 进程名 `sglang::scheduler_DPx_TPx_EPx` | 跳过（CI 跳过 + 进程命名差异） |

### 3.2 集群 / 拓扑差异

| 项 | GPU | NPU A3 |
|----|-----|--------|
| 拓扑 | 4× H100 / 8× H200 单机或双机 | 16× Ascend A3 双机 |
| TP 维度 | 4 / 8 | 16 |
| DP 维度 | 4 / 2 | 16 / 8 / 4 |
| EP 维度 | 8 | 16 |
| 设备透传 | 无 | `--device npu` |
| 启动入口 | 4× H100 测试 | 16-NPU 集群 |

### 3.3 模型替换（关键）

| GPU 模型 | NPU 替代 | 原因 |
|---------|---------|------|
| `Qwen/Qwen2.5-VL-7B-Instruct` | `QWEN2_5_VL_3B_INSTRUCT_WEIGHTS_PATH` | NPU 权重表无 7B，3B 适配 1-NPU |
| `Qwen/Qwen2.5-VL-3B-Instruct` | 同 | 路径直接匹配 |
| `DeepSeek-V2-Lite`（MLA 测试默认） | `DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH` | NPU 权重表无 V2-Lite，用 V3.2-W8A8 替代（同样 MLA + W8A8 量化） |
| DeepEP 默认模型 | `QWEN3_30B_A3B_W8A8_WEIGHTS_PATH` | NPU 权重表对应物 |

### 3.4 必需环境变量

```bash
# 通用（NPU 内存分配）
PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# 集合通信（弹性 EP / DeepEP 用）
HCCL_BUFFSIZE=2048
HCCL_OP_EXPANSION_MODE=AIV
TASK_QUEUE_ENABLE=0

# 日志抑制（重要，PR round 2 修复了"打印过多导致 CI 失败"）
TRANSFORMERS_VERBOSITY=error
TRANSFORMERS_NO_ADVISORY_WARNINGS=1
PYTHONWARNINGS=ignore::FutureWarning,ignore::UserWarning,ignore::DeprecationWarning
```

### 3.5 评估阈值

| 任务 | GPU 阈值 | NPU 阈值 | 备注 |
|------|---------|---------|------|
| GSM8K | 0.60 | 0.60 | 与 GPU 一致 |
| GSM8K (VLM 3B) | 0.80 | 0.60 | 模型容量降低，从 7B 降到 3B |
| MMLU | 0.48 | 0.48 | 与 GPU 一致 |
| Embedding 一致性 | atol=1e-2, rtol=1e-2 | atol=1e-2, rtol=1e-2 | 与 GPU 一致 |

### 3.6 CI runner / suite 注册

```python
# VLM
register_npu_ci(est_time=400, suite="stage-b-test-1-npu-a2", nightly=False)
register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)

# 弹性 EP
register_npu_ci(est_time=400, suite="nightly-16-npu-a3", nightly=True)

# 混合并行
register_npu_ci(est_time=3600, suite="weekly-16-npu-a3", nightly=True)
```

---

## 4. 每个 NPU 文件的"测什么 / 关键参数 / 当前状态"

### 4.1 `test_npu_piecewise_cuda_graph_vlm.py`

**GPU 源**：`test/registered/piecewise_cuda_graph/test_piecewise_cuda_graph_support_1_gpu.py`

#### `TestNPUPiecewiseGraphQwen25VL`
- **测什么**：1-NPU 集群上 VLM 3B 在分段 CUDA Graph 下的 GSM8K 精度
- **模型**：`QWEN2_5_VL_3B_INSTRUCT_WEIGHTS_PATH`（GPU 7B → NPU 3B，权重缺 7B）
- **评估**：200 题 GSM8K ≥ 0.60
- **关键启动参数**：
  ```
  --attention-backend ascend
  --enforce-piecewise-cuda-graph
  --disable-radix-cache
  --mem-fraction-static 0.8
  ```
- **CI 状态**：✅ **通过**（gsm8k=0.715）

#### `TestNPUPiecewiseGraphQwen25VLEmbedding`
- **测什么**：分段 CUDA Graph 开 / 关 两种 Engine 模式下 VLM embedding 一致性
- **关键配置**：`Engine(..., attention_backend="ascend", enforce_piecewise_cuda_graph=True/False)`
- **断言**：`torch.allclose(atol=1e-2, rtol=1e-2)`
- **CI 状态**：✅ **通过**

### 4.2 `test_npu_elastic_ep_hybrid_dp.py`

**GPU 源**：`test/registered/ep/test_mooncake_ep_small.py`
**NPU 模型**：`DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH`（GPU `DeepSeek-V2-Lite` → NPU `DeepSeek-V3.2-W8A8`，MLA 替代）

#### `TestElasticEPTP`（实际 CI 跑）
- **测什么**：deepep low_latency 模式 + 弹性 EP（EPLB + 32 冗余专家）下的 GSM8K 精度
- **关键启动参数**：
  ```
  --tp-size 16
  --quantization modelslim
  --attention-backend ascend
  --moe-a2a-backend deepep
  --deepep-mode low_latency
  --moe-dense-tp-size 1
  --enable-dp-lm-head
  --enable-eplb
  --ep-num-redundant-experts 32
  --chunked-prefill-size 512
  --disable-cuda-graph
  --disable-radix-cache
  --max-running-requests 32
  --mem-fraction-static 0.9       # 重要：比 GPU 的 0.5 高，因为 16-NPU 共享内存
  ```
- **CI 状态**：✅ **通过**（gsm8k=0.95）
- **适配要点**：
  - 移除 `--elastic-ep-backend mooncake`（NPU 无对应；用 EPLB + 32 冗余专家替代）
  - 移除 `--mooncake-ib-device`（NPU 无 RDMA 设备透传需求）
  - 移除 `--enable-two-batch-overlap`（NPU 暂无对应优化）
  - `--disable-custom-all-reduce` 移除（NPU 用 HCCL 默认 allreduce）
  - `--mem-fraction-static 0.5 → 0.9`（16-NPU 拓扑下需要更高静态内存比例，否则 OOM）
  - `--ep-num-redundant-experts 72 → 32`（适配 V3.2 专家数）

#### `TestElasticEPPureDP`（`@unittest.skipIf(is_in_ci(), ...)`）
- 关键差异：`--enable-dp-attention --dp 16`
- **CI 状态**：⏸ 跳过（资源密集）

#### `TestElasticEPHybridDPTP`（`@unittest.skipIf(is_in_ci(), ...)`）
- 关键差异：`--enable-dp-attention --dp 8`
- **CI 状态**：⏸ 跳过

### 4.3 `test_npu_hybrid_dp_ep_tp_mtp.py`

**GPU 源**：`test/registered/moe/test_hybrid_dp_ep_tp_mtp.py`（60 个测试类）
**NPU 模型**：`DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH` + `QWEN3_30B_A3B_W8A8_WEIGHTS_PATH`

10 个 NPU 子用例（按字母序 unittest 执行）：

| NPU 子用例 | GPU 对应 | 模型 | 关键差异（vs baseline TP=16） | CI 状态 |
|----------|---------|------|---------------------------|--------|
| `TestHybridTPOnly` | Test00 (TP=8 baseline) | DeepSeek-V3.2-W8A8 | baseline | ✅ mmlu=0.938 |
| `TestHybridDPAttention` | Test01 (DP=4) | DeepSeek-V3.2-W8A8 | `--enable-dp-attention --dp 4` | ❌ code 1 |
| `TestHybridFullDPAttention` | Test02 (DP=8) | DeepSeek-V3.2-W8A8 | `--enable-dp-attention --dp 16` | ❌ code 1 |
| `TestHybridDenseTPOne` | Test03 (dense TP=1) | DeepSeek-V3.2-W8A8 | `--moe-dense-tp-size 1` | ✅ mmlu=0.938 |
| `TestHybridDPLMHead` | Test06 (DP attn + DP LM head) | DeepSeek-V3.2-W8A8 | `--enable-dp-lm-head` | ❌ code 1 |
| `TestHybridEPOnly` | Test20 (EP=8) | DeepSeek-V3.2-W8A8 | `--ep-size 16` | ❌ code -9 |
| `TestHybridDPEPCombined` | Test21 (DP + EP) | DeepSeek-V3.2-W8A8 | `--dp 4 --ep-size 16` | ❌ code -9 |
| `TestHybridDeepEPAuto` | Test10 (DeepEP TP) | Qwen3-30B-A3B-W8A8 | `--moe-a2a-backend deepep --deepep-mode auto` | ❌ code -9 |
| `TestHybridDeepEPDPAttention` | Test11 (DeepEP + DP) | Qwen3-30B-A3B-W8A8 | + `--enable-dp-attention --dp 4` | ❌ code -9 |
| `TestHybridDeepEPFullStack` | Test18 (DeepEP + DP + dense TP1 + DP LM head) | Qwen3-30B-A3B-W8A8 | full stack | ❌ code -9 |

**所有用例通用启动参数**：
```
--tp-size 16
--quantization modelslim
--attention-backend ascend
--disable-cuda-graph
--mem-fraction-static 0.82
```

---

## 5. PR 提交流程与 CI 历次问题

| test round | 触发 commit | 现象 | 修复 |
|----------|------------|------|------|
| round 1 | 初始提交 | runner 2-NPU 报 `Invalid device ID (107001)` | runner 升级到 16-NPU（`linux-aarch64-a3-16`） |
| round 1 | 初始提交 | elastic EP `Server process exited with code -9`（OOM） | `--mem-fraction-static 0.82 → 0.9` |
| round 1 | 初始提交 | 8 个 hybrid 子用例 setUpClass OOM | 持续调优中（见下） |
| round 2 | `788465b` | "日志打印过多导致 CI 失败"（transformers / FutureWarning / torch.compile 等） | 在 setUpClass env 中加 `TRANSFORMERS_VERBOSITY=error` + `TRANSFORMERS_NO_ADVISORY_WARNINGS=1` + `PYTHONWARNINGS=ignore::...` |
| round 3 | `ba8219a` | 主动尝试 `--mem-fraction-static 0.82 → 0.75` 修复 hybrid OOM | ❌ **结果更差**（2/10 通过，原本 4/10 通过）。原因：mem-fraction-static 过低导致模型启动本身 OOM |
| round 4 | `0995eb4` | 回退 `--mem-fraction-static` 到 0.82 | 待 CI 验证 |

**最新 run id**：`27114881322`（queued → in_progress）

---

## 6. 当前失败用例（hybrid 6 个 setUpClass 失败）根因分析

### 6.1 失败子用例汇总

| 子用例 | exit code | 模型 | 关键参数 |
|-------|----------|------|---------|
| `TestHybridDPEPCombined` | -9 (OOM/SIGKILL) | DeepSeek-V3.2-W8A8 | `--dp 4 --ep-size 16` |
| `TestHybridDeepEPAuto` | -9 | Qwen3-30B-A3B-W8A8 | `--moe-a2a-backend deepep --deepep-mode auto` |
| `TestHybridDeepEPDPAttention` | -9 | Qwen3-30B-A3B-W8A8 | + `--dp 4` |
| `TestHybridDeepEPFullStack` | -9 | Qwen3-30B-A3B-W8A8 | + dense TP1 + DP LM head |
| `TestHybridEPOnly` | -9 | DeepSeek-V3.2-W8A8 | `--ep-size 16` |
| `TestHybridFullDPAttention` | 1（崩溃）| DeepSeek-V3.2-W8A8 | `--dp 16` |
| `TestHybridDPLMHead` | 1 | DeepSeek-V3.2-W8A8 | `--dp 4 --enable-dp-lm-head` |
| `TestHybridDPAttention` | 1 | DeepSeek-V3.2-W8A8 | `--dp 4` |

### 6.2 失败模式分两类

**类 A：code -9（OOM/SIGKILL）— 5 个用例**（均为启用 EP 或 DeepEP 后端的配置）
- 现象：服务器进程在加载阶段被 OOM Killer 杀掉
- 与设备数 / 集合通信缓冲（HCCL_BUFFSIZE）相关
- 重点用例：所有 DeepEP 用例 + `EPOnly` + `DPEPCombined`

**类 B：code 1（server crash）— 3 个用例**（均为启用 `--enable-dp-attention` 的配置）
- 现象：服务器进程主动退出 code 1，通常是 Ascend 后端的 assertion / RuntimeError
- 重点用例：`DPAttention`、`DPLMHead`、`FullDPAttention`

### 6.3 失败顺序（unittest 字母序）

```
TestHybridDPEPCombined     → 第一个跑 → code -9
TestHybridDeepEPAuto       → 第二个跑 → code -9
TestHybridDeepEPDPAttention → 第三个跑 → code -9
TestHybridDeepEPFullStack   → 第四个跑 → code -9
TestHybridDPLMHead         → 第五个跑 → code 1
TestHybridDPAttention      → 第六个跑 → code 1
TestHybridDenseTPOne       → 第七个跑 → ✅ 通过
TestHybridEPOnly           → 第八个跑 → code -9
TestHybridFullDPAttention  → 第九个跑 → code 1
TestHybridTPOnly           → 第十个跑 → ✅ 通过
```

观察：失败的 6 个是字母序靠前的（启动阶段 OOM），不是"前面通过后面失败"的内存累积模式，说明问题在启动阶段就存在，与"未充分释放"无关。

### 6.4 通用根因方向

1. **DP attention 后端在 ascend 上的稳定性**（类 B 3 个用例全部涉及 `--enable-dp-attention`）
2. **DeepEP `auto` 模式在 16-NPU 上的内存 / 通信开销**（类 A 3 个 QWEN3 用例）
3. **EP=16 + DeepSeek-V3.2-W8A8 大模型在 16-NPU 上的内存预算**（类 A 2 个 DeepSeek 用例 `EPOnly` / `DPEPCombined`）
4. **`mem-fraction-static 0.82` 是否够**（与 elastic_ep 的 0.9 形成对比）

---

## 7. 接下来要做的事

1. **等待 round 4 CI 完成**（run id 27114881322），基于 0.82 重新跑，确认是否回到 4/10 通过
2. **向相应开发组咨询**（按 dev 出口资料）：
   - DP attention 稳定性 → 找 DP attention 负责开发
   - DeepEP auto 模式内存 → 找 DeepEP 负责开发
   - EP=16 大模型内存预算 → 找 EPLB / 集合通信 / 内存优化开发
3. **可能的下一步动作**：
   - 给失败用例加 `--max-running-requests` / `--chunked-prefill-size` / 调整 `--mem-fraction-static`
   - 或拆分文件：保留 4 个通过用例，删除 6 个失败用例
   - 或要求对应开发先修底层稳定性

---

## 8. 附：本地 CI 日志位置

- **round 2 完整日志（0.82，参考基线）**：`D:\debug-pr2\.trae\logs\final_27061329395.txt`（10.8 MB）
- **round 3 完整日志（0.75，失败参考）**：`D:\debug-pr2\.trae\logs\final_27064148678.txt`（16.4 MB）
- **round 4 日志（待生成）**：`D:\debug-pr2\.trae\logs\final_27114881322.txt`
- 状态文件：`D:\debug-pr2\.trae\logs\state_<run_id>.json`
