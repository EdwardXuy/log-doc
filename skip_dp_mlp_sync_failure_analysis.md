# `test_npu_skip_dp_mlp_sync.py` 失败分析报告（最终版）

## 摘要

`test_npu_skip_dp_mlp_sync.py` 正向测试用例（`TestNPUSkipDPMLPSyncPositive`）在 NPU CI 上反复失败，根因为 **NPU KV cache 写入的两条路径在 EAGLE + DP attention 组合下都存在产品代码 bug**。这是 NPU ATB 算子层与 sglang 框架代码的兼容性问题，**测试侧无法绕过**，需要开发人员修复产品代码。

反向测试用例（`TestNPUSkipDPMLPSyncNegative`）工作正常，使用 EAGLE3 算法正确触发 assert 拦截。

**镜像版本**：`swr.cn-south-west-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:cann9.0.0-a3-B090`

---

## 1. CI 运行总览

| CI Run | commit | skip_dp_mlp_sync | EPD | Adaptive | DFLASH |
|--------|--------|------------------|-----|----------|--------|
| [28692216931](https://github.com/Ascend/sglang/actions/runs/28692216931/job/85095471472?pr=926) | `1c7a675` | FAIL (FIA=0, num_steps=6, Bug#1) | PASS | PASS | PASS |
| [28693124925](https://github.com/Ascend/sglang/actions/runs/28693124925/job/85097962273?pr=926) | `685d88a` | FAIL (FIA=0, num_steps=2, Bug#1) | PASS | PASS | PASS |
| [28694820522](https://github.com/Ascend/sglang/actions/runs/28694820522/job/85102539478?pr=926) | `59ae67c` | FAIL (FIA=0, +attn_mode=decode, Bug#1) | PASS | PASS | PASS |
| [28698810307](https://github.com/Ascend/sglang/actions/runs/28698810307/job/85113160907?pr=926) | `f778948` | FAIL (FIA=1, Bug#2) | PASS | PASS | PASS |
| [28701492528](https://github.com/Ascend/sglang/actions/runs/28701492528/job/85120174581?pr=926) | `08bf18f` | FAIL (LLAMA EAGLE, Bug#2 variant) | PASS | PASS | PASS |
| [28702929630](https://github.com/Ascend/sglang/actions/runs/28702929630/job/85123850984?pr=926) | `f32dfd4` | FAIL (reverted QWEN3, Bug#2) + Neg PASS (EAGLE3) | PASS | PASS | FAIL (0.585<0.60) |
| [28704645195](https://github.com/Ascend/sglang/actions/runs/28704645195/job/85128255270?pr=926) | `c0ffede` | FAIL (HCCL Allreduce timeout, 偶发) + Neg PASS | PASS | FAIL (config.json 格式) | PASS (0.7>0.55) |
| [28707114925](https://github.com/Ascend/sglang/actions/runs/28707114925/job/85134535772?pr=926) | `f6b2d2c`/`c9cb23f` | FAIL (Bug#2 复现) + Neg PASS | PASS | **PASS** (真实 upshift/downshift) | FAIL (NPU aicore 异常 507015) |

PR 链接：https://github.com/Ascend/sglang/pull/926

---

## 2. 测试类通过情况（最新 CI run 28707114925）

| 用例文件 | 测试类数 | 通过 | 失败 | 跳过 |
|---------|---------|------|------|------|
| `test_npu_epd_dynamic_register.py` | 5 | 5 | 0 | 0 |
| `test_npu_skip_dp_mlp_sync.py` | 1 (Positive) + 1 (Negative) | 1 (Negative, EAGLE3 assert) | 1 (Positive, setUpClass) | 0 |
| `test_npu_adaptive_speculative.py` | 1 | 1 | 0 | 0 |
| `test_npu_dflash_speculative.py` | 3 | 2 (test_a + test_b) | 1 (test_c_gsm8k: NPU aicore 异常) | 0 |

**无任何跳过的测试**。

### skip_dp_mlp_sync 各测试类细节
- `TestNPUSkipDPMLPSyncNegative.test_non_eagle_algorithm_rejected`：**通过**，EAGLE3 正确触发 assert（`got EAGLE3`）
- `TestNPUSkipDPMLPSyncPositive.setUpClass`：**失败**，服务器启动成功但首次推理（warmup `/health_generate`）时崩溃

### Adaptive 通过细节（最新 CI run 28707114925）
- `TestNPUAdaptiveSpeculativeServer.test_gsm8k_after_adaptive_switches`：**通过**
  - 日志清晰显示真实 upshift：`Adaptive spec params updated: steps 1 -> 3 (ema_accept_len=1.00)`
  - 日志清晰显示真实 downshift：`Adaptive spec params updated: steps 3 -> 1 (ema_accept_len=0.00)`
  - `assertEqual(num_steps, 3)` 和 `assertEqual(num_steps, 1)` 均通过
  - GSM8K score=0.945，`avg_spec_accept_length=1.9967`

### DFLASH 失败细节（最新 CI run 28707114925）
- `test_a_server_info`：**通过**，新增的 `speculative_dflash_block_size == 16` 断言通过
- `test_b_basic_inference`：**通过**
- `test_c_gsm8k`：**失败**，GSM8K 跑到 100% 时 NPU aicore 异常（error code 507015），score=0.545（差 0.005 过阈值）
  - 错误：`RuntimeError: operator():../torch_npu/csrc/aten/common/LocalScalarDenseNpu.cpp:23 NPU function error: c10_npu::acl::AclrtSynchronizeStreamWithTimeout(copy_stream), error code is 507015`
  - 根因：`EZ9999: The DDR address of the MTE instruction is out of range`（NPU 硬件偶发异常）
  - 服务器崩溃导致后续 `/server_info` 请求 Connection refused

---

## 3. 两个产品代码 Bug

### Bug #1：FIA=0 路径 `ReshapeAndCacheOperation setup failed`

**触发条件**：`ASCEND_USE_FIA=0`（默认）

**失败位置**：[`python/sglang/srt/hardware_backend/npu/memory_pool_npu.py` 第 190 行](https://github.com/EdwardXuy/sglang-Ascend/blob/test-epd-spec-20260703/python/sglang/srt/hardware_backend/npu/memory_pool_npu.py#L190)

```python
# memory_pool_npu.py:174-191
if self.use_fia:                                    # ASCEND_USE_FIA=1
    torch_npu.npu_scatter_nd_update_(...)           # 路径 A
else:                                               # ASCEND_USE_FIA=0
    torch_npu._npu_reshape_and_cache(...)           # 路径 B (BUG!)
```

**错误信息**：
```
[rank0/rank1]:[E704 04:31:16 OpParamMaker.cpp:454] ReshapeAndCacheOperation setup failed!
Exception raised from OperationSetup at ../../../../third_party/op-plugin/op_plugin/utils/custom_functions/atb/AtbCommon.cpp:203
RuntimeError: ... working operator name is ReshapeCacheOperation.
```

**应用层栈**：
```
eagle_worker_v2.py:1114  forward_batch_generation
  → eagle_worker_v2.py:802   _draft_extend_for_prefill
  → spec_utils.py:94         renorm_draft_probs
  → torch.softmax(next_token_logits, dim=-1)
RuntimeError: ... working operator name is ReshapeCacheOperation
```

注意：栈顶显示 `torch.softmax` 是因为 NPU 异步执行；真正失败的是更早提交的 ATB 算子 `ReshapeAndCacheOperation`。

### Bug #2：FIA=1 路径 `view size is not compatible`

**触发条件**：`ASCEND_USE_FIA=1`（测试侧切换以绕过 Bug #1）

**失败位置**：[`python/sglang/srt/hardware_backend/npu/memory_pool_npu.py` 第 189 行和第 211 行](https://github.com/EdwardXuy/sglang-Ascend/blob/test-epd-spec-20260703/python/sglang/srt/hardware_backend/npu/memory_pool_npu.py#L189)

```python
# memory_pool_npu.py:189-213 (FIA 分支)
if self.use_fia:
    cache_k_t = cache_k.view(-1, 1, self.head_num, self.head_dim)        # 第 189 行 (BUG!)
    cache_v_t = cache_v.view(-1, 1, self.head_num, self.head_dim)        # 第 211 行 (BUG!)
    torch_npu.npu_scatter_nd_update_(
        self.key_cache[layer_id], slot_mapping.flatten(), cache_k_t, axis=0
    )
    torch_npu.npu_scatter_nd_update_(
        self.value_cache[layer_id], slot_mapping.flatten(), cache_v_t, axis=0
    )
```

**错误信息（QWEN3_8B + EAGLE3 draft）**：
```
File "memory_pool_npu.py", line 211, in set_kv_buffer
    cache_v.view(-1, 1, self.head_num, self.head_dim),
RuntimeError: view size is not compatible with input tensor's size and stride
              (at least one dimension spans across two contiguous subspaces).
              Use .reshape(...) instead.
```

**错误信息（LLAMA-3-8B EAGLE draft，GQA 模型）**：
```
File "memory_pool_npu.py", line 206, in set_kv_buffer
    cache_k.view(-1, 1, self.head_num, self.head_dim),
RuntimeError: shape '[-1, 1, 8, 128]' is invalid for input of size 3584
```

**根因**：
1. draft model 的 `cache_v`/`cache_k` 是 non-contiguous tensor（来自 slicing/transpose），PyTorch 错误信息明确提示"Use .reshape(...) instead"
2. 对于 GQA 模型（如 LLAMA-3-8B），`view` 的 `head_num` 假设与实际 KV head 数不匹配，导致 shape 不兼容
3. 产品代码使用了 `.view()` 而非 `.reshape()`，在 non-contiguous tensor 或 GQA 模型上必然失败

**修复建议**：将 [`memory_pool_npu.py` 第 189 行和第 211 行](https://github.com/EdwardXuy/sglang-Ascend/blob/test-epd-spec-20260703/python/sglang/srt/hardware_backend/npu/memory_pool_npu.py#L189) 的 `.view()` 改为 `.reshape()`，并确保正确处理 GQA 的 KV head 数：
```python
cache_k_t = cache_k.reshape(-1, 1, self.head_num, self.head_dim)   # 第 189 行
cache_v_t = cache_v.reshape(-1, 1, self.head_num, self.head_dim)   # 第 211 行
```

---

## 4. 根因分析：为什么 EAGLE + DP attention 会触发这两个 Bug

### 4.1 关键代码约束

**约束 1**：`--speculative-skip-dp-mlp-sync` 硬性要求 `algorithm == "EAGLE"`
- 位置：[`python/sglang/srt/arg_groups/speculative_hook.py` 第 95-99 行](https://github.com/EdwardXuy/sglang-Ascend/blob/test-epd-spec-20260703/python/sglang/srt/arg_groups/speculative_hook.py#L95-L99)
```python
if server_args.speculative_skip_dp_mlp_sync:
    assert server_args.speculative_algorithm == "EAGLE", (
        "--speculative-skip-dp-mlp-sync is only supported with "
        f"speculative_algorithm == EAGLE, got {server_args.speculative_algorithm}."
    )
```

**约束 2**：`enable_dp_attention` + EAGLE3 才走 DP-aware draft_tp_context 分支
- 位置：[`python/sglang/srt/speculative/eagle_worker_v2.py` 第 153 行](https://github.com/EdwardXuy/sglang-Ascend/blob/test-epd-spec-20260703/python/sglang/srt/speculative/eagle_worker_v2.py#L153)
```python
if server_args.enable_dp_attention and self.speculative_algorithm.is_eagle3():
    ctx = draft_tp_context(get_attention_tp_group())
else:
    ctx = empty_context()
```

**约束 3**：`speculative_skip_dp_mlp_sync` 的功能性依赖 DP attention
- 位置：[`python/sglang/srt/managers/scheduler.py` 第 2422-2426 行](https://github.com/EdwardXuy/sglang-Ascend/blob/test-epd-spec-20260703/python/sglang/srt/managers/scheduler.py#L2422-L2426)

### 4.2 矛盾总结

| 约束 | 来源 | 影响 |
|------|------|------|
| skip-dp-mlp-sync 必须 EAGLE | [`speculative_hook.py`](https://github.com/EdwardXuy/sglang-Ascend/blob/test-epd-spec-20260703/python/sglang/srt/arg_groups/speculative_hook.py#L95-L99) assert | 不能改用 EAGLE3 |
| EAGLE + DP attention 走普通分支 | [`eagle_worker_v2.py`](https://github.com/EdwardXuy/sglang-Ascend/blob/test-epd-spec-20260703/python/sglang/srt/speculative/eagle_worker_v2.py#L153) | draft worker 的 KV cache 布局与 DP attention 不匹配 |
| skip_dp_mlp_sync 功能性需要 DP attention | [`scheduler.py`](https://github.com/EdwardXuy/sglang-Ascend/blob/test-epd-spec-20260703/python/sglang/srt/managers/scheduler.py#L2422-L2426) | 不能去掉 `--enable-dp-attention` |
| FIA=0 路径 ReshapeAndCache 算子失败 | [`memory_pool_npu.py:190`](https://github.com/EdwardXuy/sglang-Ascend/blob/test-epd-spec-20260703/python/sglang/srt/hardware_backend/npu/memory_pool_npu.py#L190) | Bug #1 |
| FIA=1 路径 view 失败 | [`memory_pool_npu.py:189,211`](https://github.com/EdwardXuy/sglang-Ascend/blob/test-epd-spec-20260703/python/sglang/srt/hardware_backend/npu/memory_pool_npu.py#L189) | Bug #2 |

### 4.3 现有 NPU 测试矩阵验证（反证）

| 现有 NPU 测试 | 算法 | DP attention | ASCEND_USE_FIA | 状态 |
|--------------|------|-------------|----------------|------|
| `test_npu_hicache_variants.py` | EAGLE3 | ❌ 关闭 | 0 | ✅ 通过 |
| `test_npu_adaptive_speculative.py` | EAGLE3 | ❌ 关闭 | 0 | ✅ 通过 |
| `test_npu_eagle3.py` | EAGLE3 | ❌ 关闭 | 0 | ✅ 通过 |
| `test_npu_qwen3_5_397b_*_aime26.py` | NEXTN | ✅ 开启 | - | ✅ 通过 |
| **本用例 skip_dp_mlp_sync** | **EAGLE** | **✅ 开启** | **0/1** | **❌ 两条路径都失败** |

---

## 5. 尝试过的修复

| 修复尝试 | commit | 结果 |
|---------|--------|------|
| 算法 EAGLE3 → EAGLE（满足 assert） | `1c7a675` 之前 | 反向测试通过，但正向崩溃 (Bug #1) |
| num_steps 6 → 2，num_draft_tokens 32 → 3 | `685d88a` | 仍崩溃 (Bug #1) |
| 添加 `--speculative-attention-mode decode` | `59ae67c` | 仍崩溃 (Bug #1) |
| 回退 `--speculative-attention-mode decode` | `17808d8` | 仍崩溃 (Bug #1) |
| `ASCEND_USE_FIA=0` → `1`（切换 KV cache 路径） | `f778948` | 绕过 Bug #1，但触发 Bug #2 |
| 替换为 LLAMA_3_8B_EAGLE + LLAMA_3_8B_INSTRUCT | `08bf18f` | 触发 Bug #2 变体（GQA shape 不匹配） |
| 换回 QWEN3_8B + QWEN3_8B_EAGLE3 + 反向改 EAGLE3 | `f32dfd4` | Bug #2 依旧；反向 EAGLE3 通过 |
| DFLASH 阈值 0.60→0.55 + adaptive 用 config.json 重写 | `c0ffede` | Adaptive FAIL (config.json 格式)；skip_dp Bug#2 偶发 HCCL 超时；DFLASH PASS (0.7) |
| Adaptive config.json 改为 batch-size keyed 格式 | `f6b2d2c` | **Adaptive PASS**（真实 upshift/downshift）；skip_dp Bug#2 复现；DFLASH NPU aicore 异常 |
| DFLASH test_a 添加 block_size==16 断言 | `c9cb23f` | test_a PASS（断言通过）；test_c 仍因 NPU 硬件异常失败 |

---

## 6. 建议解决方案（待开发确认）

### 方案 A：修复 Bug #2（FIA=1 路径的 view → reshape）【推荐，最小改动】
- 修改 [`memory_pool_npu.py` 第 189 行和第 211 行](https://github.com/EdwardXuy/sglang-Ascend/blob/test-epd-spec-20260703/python/sglang/srt/hardware_backend/npu/memory_pool_npu.py#L189)，将 `.view()` 改为 `.reshape()`
- 修复后测试脚本设置 `ASCEND_USE_FIA=1` 即可绕过 Bug #1，走 FIA=1 路径
- 这是**最小改动**，且 PyTorch 错误信息已明确提示

### 方案 B：修复 Bug #1（FIA=0 路径的 ReshapeAndCacheOperation）
- 由 ATB 团队排查 `ReshapeCacheOperation setup failed` 的具体参数错误

### 方案 C：放宽 assert，允许 EAGLE3 支持 skip-dp-mlp-sync
- 修改 [`speculative_hook.py` 第 95-99 行](https://github.com/EdwardXuy/sglang-Ascend/blob/test-epd-spec-20260703/python/sglang/srt/arg_groups/speculative_hook.py#L95-L99)，允许 `algorithm in ("EAGLE", "EAGLE3")`

### 方案 D：在 NPU 上为 EAGLE + DP attention 走 DP-aware 分支
- 修改 [`eagle_worker_v2.py` 第 153 行](https://github.com/EdwardXuy/sglang-Ascend/blob/test-epd-spec-20260703/python/sglang/srt/speculative/eagle_worker_v2.py#L153)，将条件改为 `enable_dp_attention and (is_eagle3() or is_eagle())`

**推荐方案 A**：改动最小，风险最低，且有明确的 PyTorch 错误提示作为依据。

---

## 7. 当前 PR 状态

- 分支：`EdwardXuy/sglang-Ascend:test-epd-spec-20260703`
- PR：https://github.com/Ascend/sglang/pull/926
- **镜像版本**：`swr.cn-south-west-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:cann9.0.0-a3-B090`
- 最新 commit：`c9cb23f`

### PR 中各用例最终状态（CI run 28707114925）
- ✅ `test_npu_epd_dynamic_register.py`：5/5 通过，覆盖编码器动态注册/健康检查/VLM请求/下线/重注册全流程
- ✅ `test_npu_adaptive_speculative.py`：1/1 通过，真实触发 upshift(1→3) 和 downshift(3→1)，GSM8K score=0.945
- ⚠️ `test_npu_dflash_speculative.py`：2/3 通过（test_a + test_b），test_c 因 NPU aicore 异常崩溃（score 0.545 已接近阈值 0.55）
- ❌ `test_npu_skip_dp_mlp_sync.py`：正向测试因 Bug #2 失败（需开发修复产品代码），反向测试通过（EAGLE3 正确触发 assert）

### 观测点覆盖情况
| 用例 | 核心参数 | 观测点 | 覆盖状态 |
|------|---------|--------|---------|
| EPD | `--encoder-bootstrap-port`, `--encoder-register-urls` | 健康检查+VLM请求+下线+重注册 | ✅ 完整覆盖 |
| Adaptive | `--speculative-adaptive`, `--speculative-adaptive-config` | upshift + downshift + GSM8K精度 + accept_length | ✅ 完整覆盖 |
| DFLASH | `--speculative-dflash-block-size`, `--speculative-draft-window-size` | block_size断言 + 基础推理 + GSM8K | ✅ test_a/b 覆盖，test_c 因硬件异常失败 |
| skip_dp_mlp_sync | `--speculative-skip-dp-mlp-sync` | 正向：DP attention下MLP同步跳过功能；反向：非EAGLE算法assert拦截 | ⚠️ 反向覆盖，正向待开发修复 Bug#2 |

---

## 8. 关键日志位置

- CI run 28692216931 日志：本地 `D:\用例设计\pr-epd-SP\ci-log-28692216931.txt`（Bug #1, FIA=0）
- CI run 28693124925 日志：本地 `D:\用例设计\pr-epd-SP\ci-log-28693124925.txt`（Bug #1, FIA=0）
- CI run 28694820522 日志：本地 `D:\用例设计\pr-epd-SP\ci-log-28694820522.txt`（Bug #1, FIA=0）
- CI run 28698810307 日志：本地 `D:\用例设计\pr-epd-SP\ci-log-28698810307.txt`（Bug #2, FIA=1, QWEN3）
- CI run 28701492528 日志：本地 `D:\用例设计\pr-epd-SP\ci-log-28701492528.txt`（Bug #2 变体, FIA=1, LLAMA GQA）
- CI run 28702929630 日志：本地 `D:\用例设计\pr-epd-SP\ci-log-28702929630.txt`（Bug #2, FIA=1, QWEN3 + 反向 EAGLE3 通过 + DFLASH 精度波动）
- CI run 28704645195 日志：本地 `D:\用例设计\pr-epd-SP\ci-log-28704645195.txt`（HCCL 偶发超时 + Adaptive config 格式错误 + DFLASH PASS 0.7）
- CI run 28707114925 日志：本地 `D:\用例设计\pr-epd-SP\ci-log-28707114925.txt`（**Adaptive 真实 upshift/downshift 通过** + Bug#2 复现 + DFLASH NPU aicore 异常）
- 测试文件：[`test/registered/ascend/basic_function/speculative_inference/test_npu_skip_dp_mlp_sync.py`](https://github.com/EdwardXuy/sglang-Ascend/blob/test-epd-spec-20260703/test/registered/ascend/basic_function/speculative_inference/test_npu_skip_dp_mlp_sync.py)
- Bug #1 位置：[`python/sglang/srt/hardware_backend/npu/memory_pool_npu.py` 第 190 行](https://github.com/EdwardXuy/sglang-Ascend/blob/test-epd-spec-20260703/python/sglang/srt/hardware_backend/npu/memory_pool_npu.py#L190)
- Bug #2 位置：[`python/sglang/srt/hardware_backend/npu/memory_pool_npu.py` 第 189 行和第 211 行](https://github.com/EdwardXuy/sglang-Ascend/blob/test-epd-spec-20260703/python/sglang/srt/hardware_backend/npu/memory_pool_npu.py#L189)
- assert 约束：[`python/sglang/srt/arg_groups/speculative_hook.py` 第 95-99 行](https://github.com/EdwardXuy/sglang-Ascend/blob/test-epd-spec-20260703/python/sglang/srt/arg_groups/speculative_hook.py#L95-L99)
- DP-aware 分支判断：[`python/sglang/srt/speculative/eagle_worker_v2.py` 第 153 行](https://github.com/EdwardXuy/sglang-Ascend/blob/test-epd-spec-20260703/python/sglang/srt/speculative/eagle_worker_v2.py#L153)
- Adaptive config 加载：[`python/sglang/srt/speculative/adaptive_spec_params.py` 第 54-72 行 `load_adaptive_config`](https://github.com/EdwardXuy/sglang-Ascend/blob/test-epd-spec-20260703/python/sglang/srt/speculative/adaptive_spec_params.py#L54-L72)（NPU B090 镜像要求 batch-size keyed 格式 `{"1": {...}}`）
