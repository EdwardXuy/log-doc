# NPU MTP Verify Bug Report — `move_intermediate_cache` UB Overflow

> 供开发人员定位与修复使用。本报告基于 PR #885 多次 CI 运行日志（run 28291331613、28346574313、28349327054 等）及本地仓库源码静态分析整理。

## 0. 对接开发建议（先看这个）

| 候选开发方向 | 是否对接 | 理由 |
|---|---|---|
| **`sgl_kernel_npu` mamba kernel 维护者** | ✅ **主选对接** | 崩溃 kernel `move_cache_dynamic_last_kernel_h_block` 位于 `sgl_kernel_npu/mamba/mamba_state_update_triton.py:21`，是 NPU mamba 算子包提供的 fused 实现，tiling 配置在 kernel 内部，主修方向（重写 tiling、按 L 维切片）由该 owner 完成 |
| **Triton / BiShengIR 编译开发** | ⚠️ 次选咨询 | 错误抛在 BiShengIR pipeline 的 UB overflow（tiling 阶段），如选择"关闭 multi-buffer"修复方案，需编译方配合调编译选项 |
| **NPU backend 框架层（`ascend_hybrid_linear_attn_backend.py`）维护者** | ⚠️ 备选 | 如选择"框架层回退到非 fused PyTorch op"方案，需该 owner 改 `update_mamba_state_after_mtp_verify` 实现 |
| **MTP / NEXTN 框架开发** | ❌ 不需对接 | `commit_mamba_states_after_verify`（spec_utils.py:577）框架逻辑在 GPU 上跑通，NPU 上也正确走到了 kernel 调用点，框架层无 bug |

**最短路径**：直接找 `sgl_kernel_npu` mamba kernel 的 owner，提供本报告 + 第 9.1 节的 5 个 CI run 链接。修复方案 7.1（重写 tiling）由其完成；如选方案 7.2（关 multi-buffer）需拉 triton 编译方入会；如选方案 7.3（框架回退）需拉 NPU backend 方入会。

---



## 1. 现象

- **影响范围**：`test_npu_qwen3_next_80b.py` 中 2 个 MTP 投机解码类共 7 个测试方法被 `@unittest.skip` 暂时禁用
  - `TestQwen3NextMTPTopk`（topk=4，树形 MTP）— 4 个方法
  - `TestQwen3NextMTPV2`（topk=1，线性 MTP）— 3 个方法
- **服务表现**：服务**可以正常启动**（`The server is fired up and ready to roll!`），但在收到第一个触发 NEXTN verify 路径的请求时崩溃（约 2-3 秒内 exit code 1）
- **GPU 行为**：同样的用例在 GPU 上跑通（PR 中可参考原始 GPU 用例 `test_qwen3_next_models_mtp.py`），故 NPU 上跳过是 NPU 独有问题，并非 GPU 用例本身跳过

## 2. 调用链与崩溃点

请求处理路径如下（基于 commit `5ca4235` 对应代码）：

```
Scheduler.run_batch                                    (scheduler.py:3185)
└── EagleWorkerV2.forward_batch_generation              (eagle_worker_v2.py:1116)
    └── EagleWorkerV2.verify                            (eagle_worker_v2.py:1510)
        └── commit_mamba_states_after_verify            (spec_utils.py:647)
            └── AscendHybridLinearAttnBackend
                .update_mamba_state_after_mtp_verify    (ascend_hybrid_linear_attn_backend.py:259)
                └── move_intermediate_cache             (sgl_kernel_npu/mamba/mamba_state_update_triton.py:126)
                    └── move_cache_dynamic_last_kernel_h_block[grid](...)   ← 此处触发
                        └── Triton JIT _do_compile     (triton/runtime/jit.py:787)
                            └── compile()              (triton/compiler/compiler.py:303)
                                └── MLIRCompilationError  ← 抛出
```

**崩溃位置**：`sgl_kernel_npu/mamba/mamba_state_update_triton.py:21`（kernel 定义行）在 BiShengIR/MLIR 编译阶段失败。注意 `move_intermediate_cache` 函数本身在 `:126`，但 Triton 抛错时定位的是 `@triton.jit` 装饰的 kernel 定义行（`:21`），即 `move_cache_dynamic_last_kernel_h_block`。

## 3. 原始错误日志（节选自 run 28291331613，commit `5ca4235`）

```
[2026-06-27 07:22:17]   File ".../sgl_kernel_npu/mamba/mamba_state_update_triton.py",
    line 126, in move_intermediate_cache
    move_cache_dynamic_last_kernel_h_block[grid](...)
[2026-06-27 07:22:17]   File ".../triton/runtime/jit.py", line 787, in _do_compile
    kernel = self.compile(src, target=target, options=options.__dict__)
[2026-06-27 07:22:17] triton.compiler.errors.MLIRCompilationError:
///------------------[ERROR][Triton][BEG]------------------
[ConvertLinalgRToBinary] encounters error:
loc(".../sgl_kernel_npu/mamba/mamba_state_update_triton.py":21:0): error: Failed to run BiShengHIR pipeline

loc(".../sgl_kernel_npu/mamba/mamba_state_update_triton.py":21:0): error: ub overflow,
    requires 2097152 bits while 1572864 bits available!
    (possible reason: tiling basic block is too large or block number is more
     than what user expect due to multi-buffer feature is enabled
     and some ops need extra local buffer.)

loc(".../sgl_kernel_npu/mamba/mamba_state_update_triton.py":21:0): error: Failed to run BiShengHIR pipeline
... (重复 3 次)
```

## 4. 根因分析

### 4.1 直接原因：UB（Unified Buffer）溢出

Ascend A3 硬件每核 UB 容量上限为 **1,572,864 bits ≈ 192 KB**。BiShengIR 在为
`move_cache_dynamic_last_kernel_h_block` 这个 Triton kernel 做 tiling 时，计算出的
UB 占用为 **2,097,152 bits ≈ 256 KB**，超出硬件容量 **524,288 bits ≈ 64 KB**（约 33%）。

错误信息给出的两个候选原因：
1. **tiling basic block is too large**：kernel 单个 block 内的中间 tensor 过大
2. **multi-buffer feature enabled, ops need extra local buffer**：双缓冲/多缓冲机制启用时，编译器会为每个 op 预留额外的 local buffer，叠加后超限

### 4.2 为什么 GPU 上不崩

| 维度 | GPU (CUDA) | NPU (Ascend) |
|---|---|---|
| Kernel 编译 | Triton → PTX/CUBIN | Triton → BiShengIR → CCE-CPU 代码 → 二进制 |
| 编译时机 | JIT，首次调用即编译 | JIT，BiShengIR pipeline 一次性静态 tiling |
| 显存层次 | 寄存器文件大、L1 共享显存可灵活分配 | UB 是固定大小的片上 SRAM，tiling 必须静态确定 |
| Buffer 模型 | 可动态 spill 到 L2/global | UB 不能 spill，编译期必须装下 |
| Mamba state kernel | 走原始的 advanced indexing PyTorch op | 走 `sgl_kernel_npu` 自定义 fused Triton kernel |

`sgl_kernel_npu` 为 NPU 提供的 `move_intermediate_cache` kernel 是 GPU 路径上不存在的优化实现（GPU 走 `tensor[bool_mask]` + `index_select`，没有 fused kernel）。这个 kernel 的 tiling 配置在静态编译时无法塞进 A3 的 UB。

### 4.3 为什么校准实验排除了其他因素

PR 推进期间做过 5 组校准实验，全部复现同一编译错误：

| 实验 | 配置 | 结果 |
|---|---|---|
| 基线 | tp=4, mem=0.8, draft=8, topk=4 | 同样 UB overflow |
| 减压 1 | tp=4, mem=0.7, draft=4, topk=2 | 同样 UB overflow |
| 减压 2 | tp=8, mem=0.75, draft=2, topk=1 | 同样 UB overflow |
| 纯线性 | topk=1, num_steps=1 (V2 路径) | 同样 UB overflow |
| 关 track | `--mamba-track-interval` 调大 | 同样 UB overflow |

→ 结论：错误**与 tp_size、mem_fraction_static、draft_tokens、topk、track_interval 无关**，是 kernel 自身的静态 tiling 缺陷。

## 5. 触发条件（开发者复现最小用例）

最小化复现：

```python
# 在已启动的 Qwen3-Next NPU 服务上，发任意带 spec verify 的请求
# 即触发崩溃。等服务 1 个请求即可。
curl -X POST http://127.0.0.1:11000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello",
    "sampling_params": {"max_new_tokens": 32}
  }'
# 服务端启动命令（关键参数）：
#   --speculative-algorithm NEXTN \
#   --speculative-num-steps 1 \
#   --speculative-eagle-topk 1 \
#   --speculative-num-draft-tokens 2 \
#   --attention-backend ascend \
#   --tp-size 8 \
#   --disable-cuda-graph \
#   --mem-fraction-static 0.75 \
#   --mamba-scheduler-strategy extra_buffer
```

只要走 `EagleWorkerV2.verify` → `commit_mamba_states_after_verify` 路径（即 Qwen3-Next 模型 + MTP 投机解码 + Ascend backend）就会触发。非 MTP 的普通请求（如 GSM8K）走 `forward_decode` 路径，不触发该 kernel，因此 `TestQwen3Next80B` 和 `TestQwen3NextLazyExtraBuffer` 正常通过。

## 6. 涉及代码（仓库 commit `5ca4235`）

| 角色 | 文件 | 关键行 |
|---|---|---|
| 调用方 | `python/sglang/srt/speculative/eagle_worker_v2.py` | `verify` 1510 |
| 调用方 | `python/sglang/srt/speculative/spec_utils.py` | `commit_mamba_states_after_verify` 577-647 |
| NPU 适配层 | `python/sglang/srt/hardware_backend/npu/attention/ascend_hybrid_linear_attn_backend.py` | `update_mamba_state_after_mtp_verify` 220-269 |
| **崩溃 kernel** | `site-packages/sgl_kernel_npu/mamba/mamba_state_update_triton.py` | kernel 定义 :21，调用 :126 |
| 装饰 | kernel 函数 `move_cache_dynamic_last_kernel_h_block` 上的 `@triton.jit` | :21 |

`update_mamba_state_after_mtp_verify` 的输入张量（来自 `mamba_caches.intermediate_ssm`，shape `[L, N, draft_token_num, ...]`）通过 `move_intermediate_cache` 在 verify 后做 gather-scatter 把每个 request 最后被接受的那一步的 mamba state commit 回持久化 cache。这正是 NPU 路径独有的 fused kernel。

## 7. 修复建议（按优先级排序）

### 7.1 Kernel 侧修复（推荐，根本解决）

针对 `move_cache_dynamic_last_kernel_h_block` 重新设计 tiling：

1. **减小 block 中的中间 tensor 维度**：把 `[L, N, draft_token_num, state_dim]` 的 gather 输出按 `L` 维度切片，每次只搬一部分 layer 的 state，循环 kernel 调用。代价是多几次 kernel launch，但 UB 占用线性下降。
2. **关闭 multi-buffer feature**：BiShengIR 编译选项中关闭 multi-buffer/pipeline，去掉额外预留的 local buffer。代价是失去指令级并行，性能可能下降 10-20%，但可让 kernel 编译通过。
3. **手动指定 tiling hint**：在 `@triton.jit` 装饰器或 kernel 内部通过 `tl.constexpr` 显式约束 `BLOCK_N`、`BLOCK_L` 等让 BiShengIR 在编译期就能确认 UB 占用 < 1,572,864 bits。

### 7.2 框架侧规避（临时）

如果短期内 kernel 侧无法修复，可在 `ascend_hybrid_linear_attn_backend.py:update_mamba_state_after_mtp_verify` 中替换为非 fused 的 PyTorch 实现（参考 GPU 路径的 `index_select` + `torch.gather`），代价是性能损失但功能可用。当前 PR 已用 `@unittest.skip` 跳过，待修复后移除装饰器即可恢复 CI。

### 7.3 硬件 / CANN 升级

A3（Ascend910_93）的 UB 是 192 KB；如果新一代 A3/B5 硬件 UB 增大或支持 UB spill，问题可能自动消失。但当前镜像 `cann9.0.0-a3-20260622` 上仍然存在。

## 8. 对 PR 的影响

- **PR #885 通过 CI**：当前 commit `5ca4235` 上 Qwen3-Next 总计 12 个方法，其中 5 个跑通（`TestQwen3Next80B.test_gsm8k`、`TestQwen3NextLazyExtraBuffer` 的 4 个），7 个因本 bug 暂时跳过。
- **跳过策略**：使用 `@unittest.skip("NPU mamba kernel fails to compile mamba_state_update for NEXTN")`，CI 中显示为 `OK (skipped=7)`，不影响整体 PR 通过。
- **恢复方式**：开发者修复 kernel 后，移除两个 MTP 类上的 `@unittest.skip` 装饰器并重跑 PR CI 即可。

## 9. 关键日志索引

### 9.1 **未跳过 + 报错**（MTP 方法执行后崩溃，复现 bug 的 CI run）

下列 run 在 PR #885 提交 `5ca4235`（加 `@unittest.skip`）之前，MTP 方法没有被跳过，服务启动后第一个 MTP 请求即触发崩溃。所有 run 均来自 PR #885，本地完整日志保存于 `d:\debug-pr-model\ci_logs3\`。

| CI Run | Job 链接 | Head commit | 现象 |
|---|---|---|---|
| [28281552424](https://github.com/Ascend/sglang/actions/runs/28281552424) | [job 83798164961](https://github.com/Ascend/sglang/actions/runs/28281552424/job/83798164961?pr=885) | 早于 5ca4235（V1 Topk 默认配置） | UB overflow，V1 Topk 类崩溃 |
| [28284609940](https://github.com/Ascend/sglang/actions/runs/28284609940) | [job 83806313487](https://github.com/Ascend/sglang/actions/runs/28284609940/job/83806313487?pr=885) | 早于 5ca4235 | UB overflow，继续复现 |
| [28291331613](https://github.com/Ascend/sglang/actions/runs/28291331613) | [job 83823714764](https://github.com/Ascend/sglang/actions/runs/28291331613/job/83823714764?pr=885) | 早于 5ca4235（V2 实验路径，tp=8/draft=2/mem=0.75） | UB overflow，V2 + tp8 仍崩 |
| [28346574313](https://github.com/Ascend/sglang/actions/runs/28346574313) | [job 83971027263](https://github.com/Ascend/sglang/actions/runs/28346574313/job/83971027263?pr=885) | `80fe186`（merge `d835386` experiment V2） | UB overflow（Qwen 部分），本地 `ci_logs3/` 完整日志对应此 run |
| [28349327054](https://github.com/Ascend/sglang/actions/runs/28349327054) | [job 83978889364](https://github.com/Ascend/sglang/actions/runs/28349327054/job/83978889364?pr=885) | 早于 5ca4235 | UB overflow，与 run 28346574313 表现一致 |

→ 5 个 run 全部复现同一 UB overflow 错误，跨多个 commit 和参数组合（详见上文 4.3 校准实验表），证明 bug 与参数无关。

### 9.2 加 `@unittest.skip` 后（CI 通过，对照基线）

| CI Run | Job 链接 | Commit | 现象 |
|---|---|---|---|
| [28355564390](https://github.com/Ascend/sglang/actions/runs/28355564390) | [job 83997617974](https://github.com/Ascend/sglang/actions/runs/28355564390/job/83997617974?pr=885) | `5ca4235` | **CI 全绿**（Qwen3-Next 5 测试通过、7 skip；MiniMax 2 测试通过） |

### 9.3 本地完整崩溃日志

- 完整日志文件：`d:\debug-pr-model\ci_logs3\single-node-poc\5_Run test.txt`
- 关键崩溃段：行 3407-3455（`MLIRCompilationError` 完整堆栈）
- 调用链段：行 3408-3420（`Scheduler.run_batch` → `EagleWorkerV2.verify` → `commit_mamba_states_after_verify` → `update_mamba_state_after_mtp_verify` → `move_intermediate_cache`）

## 10. 一句话结论

`sgl_kernel_npu/mamba/mamba_state_update_triton.py` 中的 fused gather-scatter kernel 在 BiShengIR 静态 tiling 阶段需要 256 KB UB，超出 A3 硬件 192 KB 上限。该 kernel 只在 NEXTN MTP verify 路径上被调用，因此仅影响 Qwen3-Next 模型的投机解码测试；修复方向是重新设计 kernel tiling 或在 framework 层回退到非 fused 实现。
