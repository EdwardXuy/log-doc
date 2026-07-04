# DSV4-Flash W8A8 单机 PD 混部 CI 测试用例 - 错误分析报告

## 1. PR 与 CI 链接

| 项目 | 链接 |
|------|------|
| PR #928 | https://github.com/Ascend/sglang/pull/928 |
| CI Run #1 (28695920566) - 量化参数不匹配 | https://github.com/Ascend/sglang/actions/runs/28695920566 |
| CI Run #2 (28697333156) - FP8 weight_scale_inv 缺失 | https://github.com/Ascend/sglang/actions/runs/28697333156/job/85109191156?pr=928 |
| CI Run #3 (28697929131) - aclnnHcPre 算子缺失 | https://github.com/Ascend/sglang/actions/runs/28697929131/job/85111556316?pr=928 |
| CI Run #4 (28698221637) - NPUCompressStatePool.ratio AttributeError | https://github.com/Ascend/sglang/actions/runs/28698221637/job/85111556316?pr=928 |
| CI Run #5 (28699361112) - 同上 ratio 错误（加 --pull always 后） | https://github.com/Ascend/sglang/actions/runs/28699361112/job/85114680217?pr=928 |
| CI Run #6 (28700020700) - 性能用例先跑，同样 ratio 错误 | https://github.com/Ascend/sglang/actions/runs/28700020700/job/85116367063?pr=928 |

**参考 PR**：
- PR #882（DSV4 用例参考）：https://github.com/Ascend/sglang/pull/882
- PR #872（yml 格式参考）：https://github.com/Ascend/sglang/pull/872
- PR #885（yml 格式参考）：https://github.com/Ascend/sglang/pull/885

**参考分支/用例**：
- testcases 分支基础：https://github.com/EdwardXuy/sglang-Ascend/tree/testcases
- DSV3.2 性能用例参考：https://github.com/chenyang08056032/sglang/blob/testcases/test/registered/ascend/performance/deepseek_v3_2/test_npu_deepseek_v3_2_w8a8_1p1d_32p_in128k_out1k_26ms.py
- DSV3.2 AIME25 精度用例参考：https://github.com/chenyang08056032/sglang/blob/testcases/test/registered/ascend/accuracy/deepseek_v3_2/test_npu_deepseek_v3_2_8p_aime25.py
- Qwen3.5-397B GPQA 精度用例参考：https://github.com/chenyang08056032/sglang/blob/testcases/test/registered/ascend/accuracy/qwen3_5_397b/test_npu_qwen3_5_397b_8p_gpqa.py
- Qwen3.5-397B AIME26 精度用例参考：https://github.com/chenyang08056032/sglang/blob/testcases/test/registered/ascend/accuracy/qwen3_5_397b/test_npu_qwen3_5_397b_8p_aime26.py

## 2. 测试用例概述

### 2.1 性能用例
- **文件**：`test/registered/ascend/performance/deepseek_v4_flash/test_npu_deepseek_v4_flash_w8a8_16p_in8k_out1k_50ms.py`
- **场景**：单机 16 卡 PD 混部（TP=16, DP=16, EP=16, MTP EAGLE 投机解码）
- **基准**：random 数据集，input_len=8000, output_len=1000, num_prompts=160, max_concurrency=160
- **指标**：TPOT ≤ 50ms，output token throughput ≥ 1708 tokens/s

### 2.2 精度用例
- **文件**：`test/registered/ascend/accuracy/deepseek_v4_flash/test_npu_deepseek_v4_flash_w8a8_16p_aime26_gpqa.py`
- **包含两个 class**：
  - `TestNPUDeepSeekV4FlashW8A816PAIME26`：AIME26 数据集，limit=30，精度阈值 0.933
  - `TestNPUDeepSeekV4FlashW8A816PGPQA`：GPQA-Diamond 数据集，eval-batch-size=128，精度阈值 0.712

## 3. 参数与环境变量对比（vs 开发脚本 run_dsv4_flash.sh）

### 3.1 环境变量（24 项，完全一致 ✓）

| 类别 | 环境变量 | 值 | 一致 |
|------|---------|-----|------|
| 基础 | PYTORCH_NPU_ALLOC_CONF | expandable_segments:True | ✓ |
| 基础 | STREAMS_PER_DEVICE | 32 | ✓ |
| 基础 | INF_NAN_MODE_FORCE_DISABLE | 1 | ✓ |
| 基础 | SGLANG_SET_CPU_AFFINITY | 1 | ✓ |
| 网络通信 | HCCL_SOCKET_IFNAME | lo | ✓ |
| 网络通信 | GLOO_SOCKET_IFNAME | lo | ✓ |
| 网络通信 | HCCL_OP_EXPANSION_MODE | AIV | ✓ |
| DeepEP | HCCL_BUFFSIZE | 1000 | ✓ |
| DeepEP | DEEP_NORMAL_MODE_USE_INT8_QUANT | 1 | ✓ |
| DeepEP | DEEPEP_NORMAL_LONG_SEQ_ROUND | 16 | ✓ |
| DeepEP | DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS | 2048 | ✓ |
| DeepEP | DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ | 1 | ✓ |
| 跳过 GPU 分支 | SGLANG_OPT_FP8_WO_A_GEMM | 0 | ✓ |
| 跳过 GPU 分支 | SGLANG_OPT_USE_OVERLAP_STORE_CACHE | False | ✓ |
| 跳过 GPU 分支 | FORCE_DRAFT_MODEL_NON_QUANT | 1 | ✓ |
| 跳过 GPU 分支 | SGLANG_DSV4_FP4_EXPERTS | False | ✓ |
| 跳过 GPU 分支 | SGLANG_OPT_FUSE_WQA_WKV | 0 | ✓ |
| 跳过 GPU 分支 | SGLANG_OPT_BF16_FP32_GEMM_ALGO | torch | ✓ |
| 跳过 GPU 分支 | SGLANG_OPT_USE_FUSED_HASH_TOPK | False | ✓ |
| 跳过 GPU 分支 | SGLANG_OPT_USE_TILELANG_MHC_PRE | False | ✓ |
| 跳过 GPU 分支 | SGLANG_OPT_DEEPGEMM_HC_PRENORM | False | ✓ |
| 跳过 GPU 分支 | SGLANG_OPT_USE_TILELANG_MHC_POST | False | ✓ |
| MTP(EAGLE) | SGLANG_ENABLE_SPEC_V2 | 1 | ✓ |
| MTP(EAGLE) | SGLANG_ENABLE_OVERLAP_PLAN_STREAM | 1 | ✓ |

### 3.2 服务启动参数（24 项，完全一致 ✓）

| 参数 | 值 | 一致 |
|------|-----|------|
| --page-size | 128 | ✓ |
| --tp-size | 16 | ✓ |
| --trust-remote-code | (flag) | ✓ |
| --device | npu | ✓ |
| --attention-backend | dsv4 | ✓ |
| --watchdog-timeout | 9000 | ✓ |
| --mem-fraction-static | 0.85 | ✓ |
| --prefill-max-requests | 2 | ✓ |
| --disable-radix-cache | (flag) | ✓ |
| --chunked-prefill-size | -1 | ✓ |
| --max-running-requests | 160 | ✓ |
| --dp-size | 16 | ✓ |
| --enable-dp-attention | (flag) | ✓ |
| --moe-a2a-backend | deepep | ✓ |
| --deepep-mode | auto | ✓ |
| --quantization | modelslim | ✓ |
| --enable-dp-lm-head | (flag) | ✓ |
| --kv-cache-dtype | bfloat16 | ✓ |
| --cuda-graph-bs | 1 2 4 8 10 | ✓ |
| --speculative-algorithm | EAGLE | ✓ |
| --speculative-num-steps | 2 | ✓ |
| --speculative-eagle-topk | 1 | ✓ |
| --speculative-num-draft-tokens | 3 | ✓ |
| --host / --port | (CI 框架自动设置) | - |

### 3.3 唯一关键差异

| 项 | 开发脚本 run_dsv4_flash.sh | CI 环境 |
|----|---------------------------|---------|
| **sglang 代码来源** | `PYTHONPATH=/home/zkk/sglang/python`（开发本地已修复 bug 的版本） | 镜像 `/sgl-workspace/sglang/python`（B090 镜像内置版本，**含 ratio bug**） |
| 模型路径 | `/home/weights/DeepSeek-V4-Flash-w8a8-mtp-ms` | ModelScope 缓存路径（由框架自动解析） |

> **结论**：测试用例参数与环境变量已 100% 对齐开发脚本，无任何遗漏或差异。`ratio` 错误的根因是 CI 镜像中 sglang 运行时代码与开发本地代码版本不一致。

## 4. CI 错误演进历程

### Run #1（28695920566）- 量化参数不匹配 ❌→✅
- **错误**：`Quantization method mismatch`
- **原因**：用例用 `compressed-tensors`，模型权重是 `modelslim` 量化格式
- **修复**：将两个用例的 `--quantization` 改为 `modelslim`

### Run #2（28697333156）- FP8 weight_scale_inv 缺失 ❌→✅
- **错误**：`AssertionError: FP8 quant_config must create weight_scale_inv`
- **原因**：用例参数未对齐开发最新脚本（attention-backend、mem-fraction-static 等多出不一致）
- **修复**：参照 `run_dsv4_flash.sh` 全面更新用例参数与环境变量
  - `--attention-backend` ascend → dsv4
  - `--mem-fraction-static` 0.65 → 0.85
  - 新增 `--kv-cache-dtype bfloat16`
  - 新增 `--prefill-max-requests 2`
  - `--cuda-graph-bs` 调整为 `1 2 4 8 10`
  - 新增 DEEPEP 相关环境变量
  - 新增 MTP（EAGLE）相关环境变量

### Run #3（28697929131）- aclnnHcPre 算子缺失 ❌→✅
- **错误**：`RuntimeError: aclnnHcPre or aclnnHcPreGetWorkspaceSize not in libopapi.so`
- **原因**：DSV4 的 HCA（Hybrid Compressed Attention）需要自定义算子 `aclnnHcPre`，该算子位于 `custom_transformer` vendor 目录下，CI 环境未加载该 vendor
- **修复**：在 yml 的 Run test 步骤前增加两条 source 命令
  ```bash
  source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
  source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash
  ```

### Run #4（28698221637）- NPUCompressStatePool.ratio AttributeError ❌
- **错误**：`AttributeError: 'NPUCompressStatePool' object has no attribute 'ratio'`
- **位置**：`/sgl-workspace/sglang/python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py:982`
- **触发条件**：服务成功启动后，decode 阶段首次触发 EAGLE MTP verify 时

### Run #5（28699361112）- 同上 ratio 错误（加 --pull always 后）❌
- **错误**：与 Run #4 完全一致
- **目的验证**：加 `options: --pull always` 强制每次拉取最新 B090 镜像，排除 runner 缓存旧版镜像的可能
- **结论**：错误一致，**确认 B090 镜像本身就有此 bug**，不是缓存问题

### Run #6（28700020700）- 性能用例先跑，同样 ratio 错误 ❌
- **错误**：与 Run #4/#5 完全一致（`AttributeError: 'NPUCompressStatePool' object has no attribute 'ratio'`）
- **目的验证**：调整 yml 中 test_cases 顺序，性能用例先跑，验证是否仅精度用例受影响
- **执行时序**：
  - `08:08:08` Running test case: performance/.../in8k_out1k_50ms.py
  - `08:13:38` Uvicorn running on http://127.0.0.1:20066（服务成功启动）
  - `08:14:25` 16 个 DP/TP scheduler 同时 hit exception（ratio AttributeError）
  - `08:14:26` SIGQUIT received，scheduler_0 crashed with exit code -3
  - `08:14:41` ERROR: setUpClass，Server process exited with code -9
  - `08:14:41` FAILED (errors=1)
- **结论**：性能用例同样在 decode 阶段首次触发 EAGLE MTP verify 时崩溃，**确认 ratio bug 与用例类型无关，是 sglang 运行时（产品代码）的通用缺陷**。只要启用 EAGLE MTP，任何 DSV4 用例都会触发。

## 5. ratio 错误根因深度分析

### 5.1 ratio 是什么功能

`ratio` 指的是 **KV cache 压缩比率**（compress_ratio），是 DSV4 的 **Hybrid Compressed Attention（HCA）** 机制核心字段。

DSV4 采用混合压缩注意力，有两种压缩比：
- `compress_ratio = 4`（C4）：每 4 层共享一份 KV cache 压缩状态
- `compress_ratio = 128`（C128）：每 128 层共享一份 KV cache 压缩状态

`NPUCompressStatePool` 是 NPU 后端的压缩状态池类，`ratio` 属性本应标明该池是 C4 还是 C128，供 `clear_unaccepted_c128_draft_states()` 在 EAGLE MTP verify 阶段清理被拒绝的 draft token 状态时判断"是否为 C128 池"。

### 5.2 错误调用链

```
decode 阶段第一个 batch
  └─> EAGLE MTP verify()
        └─> clear_unaccepted_c128_draft_states()
              └─> pool.ratio != 128  (L982)
                    └─> AttributeError: 'NPUCompressStatePool' object has no attribute 'ratio'
                          └─> 所有 16 个 DP/TP scheduler 同时崩溃
                                └─> 服务进程退出 (code -9)
```

### 5.3 影响范围

- **性能用例（Run #6 验证）**：服务成功启动后 48 秒，decode 阶段首次触发 EAGLE MTP verify 时崩溃
- **精度用例（Run #4/#5 验证）**：AIME26 和 GPQA 两个 class 都在 decode 阶段首次触发 EAGLE MTP verify 时崩溃
- **结论**：ratio bug **与用例类型无关**，是 sglang 运行时（产品代码）的通用缺陷。只要启用 EAGLE MTP（`--speculative-algorithm EAGLE`），任何 DSV4 用例都会触发
- **崩溃机制**：所有 16 个 DP/TP scheduler 同时 hit exception → `SIGQUIT received` → `scheduler_0 crashed with exit code -3` → 服务进程退出 code -9 → `setUpClass` 抛 `Exception: Server process exited with code -9`

### 5.4 根因总结

| 项目 | 详情 |
|------|------|
| **错误** | `AttributeError: 'NPUCompressStatePool' object has no attribute 'ratio'` |
| **位置** | `/sgl-workspace/sglang/python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py:982` |
| **触发条件** | EAGLE MTP verify 阶段调用 `clear_unaccepted_c128_draft_states()` |
| **根因** | B090 镜像的 sglang 代码里 `NPUCompressStatePool` 类没定义 `ratio` 属性，但 `clear_unaccepted_c128_draft_states` 访问了它 |
| **影响范围** | 任何启用 EAGLE MTP 的 DSV4 用例都会触发 |
| **为何开发能跑通** | 开发 `run_dsv4_flash.sh` 设了 `PYTHONPATH=/home/zkk/sglang/python`，用的是开发本地已修复的 sglang 代码（`NPUCompressStatePool.__init__` 中已赋值 `self.ratio = compress_ratio`），但该修复未合入 B090 镜像 |
| **修复方** | 必须由 sglang 开发修复并在 `NPUCompressStatePool` 类的 `__init__` 中加上 `self.ratio = compress_ratio`，然后推新镜像 |

### 5.5 镜像拉取策略验证

GitHub Actions 的 `container` 配置默认不带 `--pull always`，行为是：
- runner 本地没有该镜像 → 拉取
- runner 本地已有该镜像（相同 tag）→ **直接用本地缓存的旧版本**，不会重新拉取

`B090` 是固定 tag（不是 `latest`）。Run #5 已通过加 `options: --pull always` 验证：
- 加了 `--pull always` 后错误完全一致
- **确认 B090 镜像本身就有这个 bug**，不是 runner 缓存旧版镜像导致

## 6. CI 用例代码来源说明

CI yml 第 70-75 行的关键逻辑决定了代码来源：

```bash
echo "Use sglang from image"
sglang_pkg_path=/sgl-workspace/sglang/python                              # ← 镜像里的 sglang
ascend_test_util_path=${sglang_pkg_path}/sglang/test/ascend
mkdir -p ${ascend_test_util_path}
mv ${ascend_test_util_path} ${ascend_test_util_path}_bak                   # ← 备份镜像的 test/ascend
cp -r ${sglang_source_path}/python/sglang/test/ascend ${ascend_test_util_path}  # ← 用 PR 的 test/ascend 覆盖
```

| 代码 | 来源 |
|------|------|
| `sglang/srt/**`（运行时，含 `deepseek_v4_memory_pool.py`、`models/`、`scheduler.py` 等） | **镜像** `/sgl-workspace/sglang/python` |
| `sglang/test/ascend/**`（测试 utils，含 `test_npu_performance_utils.py`、基类等） | **PR 代码**（覆盖镜像的） |
| `test/registered/ascend/**`（测试用例本身） | **PR 代码**（直接 `python -u` 执行） |

**这解释了为什么 ratio bug 无法通过修改 PR 用例解决**：bug 在镜像的 `sglang/srt/` 运行时代码里，PR 只能改 `sglang/test/ascend/` 和 `test/registered/ascend/`，改不了 `sglang/srt/`。

## 7. 建议修复方案

### 方案 A（推荐）：等开发推新镜像
1. 联系开发修复 `NPUCompressStatePool` 类，在 `__init__` 中补上 `self.ratio = compress_ratio`
2. 开发推送新版镜像（新 tag 如 `B091`，或覆盖 `B090`）
3. 我们改 yml 里的 image tag 重新跑

### 方案 B（临时）：CI 中覆盖镜像代码
1. 开发把修复后的 `deepseek_v4_memory_pool.py` 放到 PR 仓库某处（如 `test/registered/ascend/patches/`）
2. 在 yml 的 Run test 步骤前加一步 `cp` 用 patched 文件覆盖镜像里的版本
3. 待新镜像发布后移除该 patch

### 方案 C：合入 testcases 分支后从源码安装
1. 将修复合入 testcases 分支
2. CI yml 改为 `install_sglang_from_source: true`，从源码安装 sglang
3. 但此方案会显著增加 CI 运行时间

## 8. Run #6 详细分析（性能用例先跑）

### 8.1 执行时序

```
08:08:08  Running test case: performance/.../in8k_out1k_50ms.py
08:13:38  Started server process [1073]
08:13:38  Application startup complete
08:13:38  Uvicorn running on http://127.0.0.1:20066
08:14:25  [DP0~DP15 TP0~TP15 EP0~EP15] Scheduler hit an exception (×16)
          File ".../speculative/eagle_worker_v2.py", line 1158, in forward_batch_generation
            if pool is None or pool.ratio != 128:
          AttributeError: 'NPUCompressStatePool' object has no attribute 'ratio'
08:14:26  Subprocess scheduler_0 (pid=1225) crashed with exit code -3
08:14:26  SIGQUIT received. signum=None, frame=None
08:14:41  ERROR: setUpClass (__main__.TestNPUDeepSeekV4FlashW8A816PIn8kOut1k50ms)
08:14:41  Exception: Server process exited with code -9
08:14:41  FAILED (errors=1)
```

### 8.2 关键发现

1. **服务启动成功**：08:13:38 Uvicorn running，证明前 5 轮修复（modelslim 量化、dsv4 attention backend、custom vendor env、--pull always）全部有效
2. **崩溃发生在服务启动后 48 秒**：即从启动到进入第一个 decode batch 的 EAGLE MTP verify 阶段耗时约 48 秒
3. **错误完全一致**：与 Run #4/#5 的 ratio AttributeError 一字不差，证明 bug 在 sglang 运行时代码（产品代码），与用例类型/数据集无关
4. **16 个 scheduler 同时崩溃**：DP=16 配置下所有 scheduler 并行执行相同 verify 路径，同时触发 AttributeError
5. **崩溃位置**：`/sgl-workspace/sglang/python/sglang/srt/speculative/eagle_worker_v2.py:1158`
   - 调用链：`forward_batch_generation` → `clear_unaccepted_c128_draft_states` → `pool.ratio != 128`
   - `pool` 是 `NPUCompressStatePool` 实例，但该类未定义 `ratio` 属性

### 8.3 结论

**这是 sglang 产品代码（运行时）的 bug，不是测试用例配置问题。**

- **bug 所在**：`/sgl-workspace/sglang/python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` 中的 `NPUCompressStatePool` 类
- **缺失内容**：`__init__` 方法未给 `self.ratio` 赋值
- **触发位置**：`/sgl-workspace/sglang/python/sglang/srt/speculative/eagle_worker_v2.py:1158`
- **触发条件**：启用 EAGLE MTP（`--speculative-algorithm EAGLE`）的 DSV4 用例在 decode 阶段首次 verify
- **影响范围**：任何启用 EAGLE MTP 的 DSV4 用例（不限于本 PR 的两条用例）
- **修复方**：sglang 开发团队需在 `NPUCompressStatePool.__init__` 中补上 `self.ratio = compress_ratio`，并推新镜像（如 B091 或覆盖 B090）
- **PR 阻塞点**：在开发推新镜像前，本 PR 无法通过 CI 验证

## 9. 当前 PR 状态

- **性能用例**：参数与环境变量已 100% 对齐开发脚本，但因 ratio bug 未验证通过
- **精度用例**：参数与环境变量已 100% 对齐开发脚本，但因 ratio bug 未验证通过
- **CI yml**：已完成配置（含 `--pull always`、custom vendor env、性能用例优先执行）
- **阻塞点**：等待开发修复 B090 镜像中的 `NPUCompressStatePool.ratio` bug
