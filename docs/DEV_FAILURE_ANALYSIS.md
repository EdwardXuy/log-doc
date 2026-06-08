# NPU 失败用例 - 开发者定位文档

**作者**: EdwardXuy
**日期**: 2026-06-06
**目标读者**: sglang-Ascend 后端开发 (EAGLE3 / DP-attention / DP-LM-Head / 投机解码 / HCCL 通信方向)
**目的**: 详细描述 NPU CI 中一个 EAGLE3 推测解码测试用例的失败现象 + 已尝试的修复 + 当前定位结论 + 原始日志片段

---

## 0. 关键链接 (请先看)

| 资源 | 链接 |
|---|---|
| **PR** | https://github.com/Ascend/sglang/pull/710 |
| **PR 最新 commit** | `729b00cc07` |
| **失败 CI run (Round 10)** | https://github.com/Ascend/sglang/actions/runs/27063936417 |
| **失败 CI run (Round 9, 同症状)** | https://api.github.com/repos/Ascend/sglang/actions/runs/27063936417 (Round 9 ID 早于此) |
| **本地完整日志 (Round 10)** | `D:\debug-pr\.trae\logs\npu_round10_729b00c.log` (207 KB) |
| **本地完整日志 (Round 9)** | `D:\debug-pr\.trae\logs\npu_round9_8608977.log` |
| **GPU 上游原始测试** | https://github.com/sgl-project/sglang/blob/main/test/registered/spec/eagle/test_eagle_dp_attention.py |
| **GPU 上游同 PR 中通过的伴随测试** | https://github.com/sgl-project/sglang/blob/main/test/registered/spec/eagle/test_eagle_infer_beta_dp_attention.py |
| **NPU 端测试代码** | `sglang/test/registered/ascend/basic_function/speculative_inference/GENERATED_20260606/test_npu_eagle_dp_attention.py` |
| **NPU 端通过的伴随测试代码** | `sglang/test/registered/ascend/basic_function/speculative_inference/GENERATED_20260606/test_npu_eagle_infer_beta_dp_attention.py` |

> **建议**: 开发人员先看第 4 节"原始日志报错片段"和第 5 节"失败 vs 通过对比"，再决定定位方向。

---

## 1. 失败用例概述

- **测试文件**: `sglang/test/registered/ascend/basic_function/speculative_inference/GENERATED_20260606/test_npu_eagle_dp_attention.py`
- **测试类**: `TestNpuEAGLE3EngineDPAttention`
- **测试方法**: `test_a_gsm8k` (在 `setUpClass` 阶段就失败，未进入测试方法本身)
- **GPU 上游来源**: `test_eagle_dp_attention.py` (H100 4卡, DP-attention + DP-LM-Head + MoE-dense-TP)
- **NPU 端角色**: 在 PR #710 中与其他 2 个 EAGLE3 推测解码测试一同提供

### 1.1 涉及的 NPU 端特性

| 特性 | 涉及 |
|---|---|
| EAGLE3 推测解码算法 | ✅ |
| DP-attention (data-parallel attention) | ✅ |
| **DP-LM-Head (data-parallel LM head)** | ✅ ⚠️ (核心触发点) |
| **MoE-dense-TP-size=1 (MoE dense 部分走 TP)** | ✅ ⚠️ (核心触发点) |
| ascend 注意力后端 | ✅ |
| TP=2, DP=2 (4 卡) | ✅ |
| NPU 端投机解码 V2 (EAGLE3) | ✅ |

---

## 2. 测试启动参数 (从 setUpClass 提取)

```python
other_args = [
    "--trust-remote-code",
    "--speculative-algorithm", "EAGLE3",
    "--speculative-num-steps", "6",            # 较大
    "--speculative-eagle-topk", "1",           # NPU 端 topk 强制为 1
    "--speculative-num-draft-tokens", "32",    # 较大
    "--speculative-draft-model-path", cls.draft_model,  # Qwen3-8B_eagle3
    "--tp-size", "2",
    "--dp-size", "2",
    "--enable-dp-attention",
    "--enable-dp-lm-head",                     # ⚠️ 关键标志
    "--moe-dense-tp-size", "1",                # ⚠️ 关键标志
    "--attention-backend", "ascend",
    "--disable-cuda-graph",
    "--mem-fraction-static", "0.7",
]
env = NPU_ENV  # PYTORCH_NPU_ALLOC_CONF=expandable_segments:True, HCCL_BUFFSIZE=200,
               # SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1, HCCL_EXEC_TIMEOUT=200, STREAMS_PER_DEVICE=32
```

---

## 3. 失败现象

- **位置**: `setUpClass` 中 `popen_launch_server` 启动后约 **4 分 42 秒**
- **症状**: scheduler 进程收到 SIGKILL (`exit code -9`)
- **可重现性**: 100% 复现 (Round 9 和 Round 10 完全相同的错误)
- **伴随日志**: 出现大量 `Health check failed. Server couldn't get a response from detokenizer for last 20 seconds` 警告 (×12 次) 后，scheduler 抛 `RuntimeError`

---

## 4. 原始日志报错片段

### 4.1 Scheduler 主堆栈 (`scheduler.py → eagle_worker_v2.py → forward_batch_info.py`)
```
[2026-06-06 13:56:40 DP0 TP0] Scheduler hit an exception: Traceback (most recent call last):
  File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 4017, in run_scheduler_process
    scheduler.run_event_loop()
  File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 1423, in run_event_loop
    dispatch_event_loop(self)
  File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 3882, in dispatch_event_loop
    scheduler.event_loop_overlap()
  File "/usr/local/python3.11.15/lib/python3.11/site-packages/torch/utils/_contextlib.py", line 124, in decorate_context
    return func(*args, **kwargs)
  File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 1487, in event_loop_overlap
    batch_result = self.run_batch(batch)
  File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 3023, in run_batch
    batch_result = self.model_worker.forward_batch_generation(
  File "/sgl-workspace/sglang/python/sglang/srt/speculative/eagle_worker_v2.py", line 937, in forward_batch_generation
    self.draft_worker._draft_extend_for_prefill(
  File "/sgl-workspace/sglang/python/sglang/srt/speculative/eagle_worker_v2.py", line 665, in _draft_extend_for_prefill
    forward_batch = ForwardBatch.init_new(batch, self.draft_runner)
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/forward_batch_info.py", line 718, in init_new
    ).to(device, non_blocking=True)
```

### 4.2 NPU 驱动/HCCL 错误
```
RuntimeError: The Inner error is reported as above. The process exits for this inner error,
              and the current copy params are srclen=16, dstlen=16, kind=1.

[ERROR] 2026-06-06-13:56:40 (PID:10443, Device:0, RankID:-1) ERR00100 PTA call acl api failed.
        Possible Cause: 1. An exception occurs during the execution on some NPUs in the cluster.
        As a result, collective communication operation failed.
        2. The execution speed on some NPU in the cluster is too slow to complete a
        communication operation within the timeout interval. (The default timeout interval
        is 1800s, You can set the interval by using HCCL_EXEC_TIMEOUT.)
        3. The number of training samples of each NPU is inconsistent.
        4. Packet loss or other connectivity problems occur on the communication link.
        TraceBack (most recent call last):
        rtMemcpyAsync execution failed, reason=tsfw unknown error[FUNC:FuncErrorReason]
        [FILE:error_message_manage.cc][LINE:65]

EE9999[PID: 10443] 2026-06-06-13:56:40.031.249 (EE9999):
        rtDeviceSynchronizeWithTimeout execution failed, reason=tsfw unknown error
        [FUNC:FuncErrorReason][FILE:error_message_manage.cc][LINE:65]
EE9999[PID: 10443] 2026-06-06-13:56:40.033.942 (EE9999):
        rtDeviceSynchronizeWithTimeout execution failed, reason=tsfw unknown error
EE9999[PID: 10443] 2026-06-06-13:56:40.035.721 (EE9999):
        rtDeviceSynchronizeWithTimeout execution failed, reason=tsfw unknown error
EE9999[PID: 10443] 2026-06-06-13:56:40.056.402 (EE9999):
        rtDeviceSynchronizeWithTimeout execution failed, reason=tsfw unknown error
```

### 4.3 启动到失败之间 (Health check 警告 12 次)
```
[13:53:27] Health check failed. Server couldn't get a response from detokenizer for last 20 seconds.
           tic start time: 13:53:07. last_heartbeat time: 13:53:05
[13:53:42] Health check failed. ...
[13:53:57] Health check failed. ...
[13:54:12] Health check failed. ...
[13:54:27] Health check failed. ...
[13:54:42] Health check failed. ...
[13:54:57] Health check failed. ...
[13:55:12] Health check failed. ...
[13:55:27] Health check failed. ...
[13:55:42] Health check failed. ...
[13:55:57] Health check failed. ...
[13:56:12] Health check failed. ...
[13:56:27] Health check failed. ...
[13:56:40 DP0 TP0] Scheduler hit an exception: ...  ← 错误抛出
```

---

## 5. 失败用例 vs 通过用例 对比

### 5.1 通过的伴随测试 (`test_npu_eagle_infer_beta_dp_attention.py`)
- 同样使用 TP=2, DP=2, enable-dp-attention, EAGLE3, Qwen3-8B 模型
- **关键差异** (下方加 ⚠️):
  - 没有 `--enable-dp-lm-head`
  - 没有 `--moe-dense-tp-size 1`
  - `--speculative-num-draft-tokens 4` (vs 失败用例的 32)
  - `--speculative-num-steps 3` (vs 失败用例的 6)
- **CI 结果**: Round 10 **PASS** (188s), `score=0.95, avg_spec_accept_length=1.98`

### 5.2 决定性差异表
| 关键参数 | 通过的伴随测试 | 失败的当前测试 | 影响 |
|---|---|---|---|
| TP / DP | 2 / 2 | 2 / 2 | 相同 |
| `--enable-dp-attention` | ✅ | ✅ | 相同 |
| `--enable-dp-lm-head` | ❌ | ✅ ⚠️ | **疑似触发** |
| `--moe-dense-tp-size 1` | ❌ | ✅ ⚠️ | **疑似触发** |
| `num_steps` | 3 | 6 | 放大因子 |
| `num_draft_tokens` | 4 | 32 | 放大因子 |
| `topk` | 1 | 1 | 相同 |
| 错误时机 | 不出错 | 启动后 4m42s | 决定性 |

### 5.3 关键堆栈位置对照 (sglang/srt/)
- **失败堆栈入口**: `speculative/eagle_worker_v2.py:937 forward_batch_generation` → `eagle_worker_v2.py:665 _draft_extend_for_prefill`
- **失败堆栈中 `.to(device, non_blocking=True)`**: `model_executor/forward_batch_info.py:718 init_new`
- **伴随的 H2D copy srclen=16, dstlen=16, kind=1**: 这是 16 个元素的张量 (`HCCL_BUFFSIZE=200` 与之无关)

---

## 6. 已尝试的修复 (按 CI 轮次)

| Round | 提交 | 改动 | 结果 |
|---|---|---|---|
| 8 | `5dc509f` | 改 `--speculative-eagle-topk 10→1` (NPU 不支持) | 启动时仍失败，触发新 NPU 内部错误 |
| 9 | `860897d` | (无改动) | 错误 100% 复现 → 排除概率/抖动 |
| 10 | `729b00c` | 删 `--cuda-graph-max-bs 64` (与 `--disable-cuda-graph` 冲突) | 错误 100% 复现 → 排除 cuda-graph 冲突 |

> **结论**: 简单参数/标志调整手段已用尽。问题不是配置错误，是 sglang-Ascend NPU 端在特定参数组合下的内部 bug。

---

## 7. 我对根因的初步判断 (供开发参考)

按可能性从高到低:

### 7.1 怀疑 1: `forward_batch_info.py:718` 的 H2D copy 在 DP-LM-Head 下的张量切分 bug (最高)
- 错误发生在 `.to(device, non_blocking=True)`
- 错误信息中 `srclen=16, dstlen=16, kind=1` 表示是 16 元素张量 (大概率是 device mesh 元数据或 rank 信息)
- `kind=1` 在 torch 里通常指 `torch.cuda.memcpy_kind.D2H` (device-to-host)，与 `non_blocking=True` 一起
- DP-LM-Head 改变了 LM head 输出的并行切分方式，可能导致 `_draft_extend_for_prefill` 中的 ForwardBatch init 错误地构造了 device mesh 索引
- **建议定位方向**: 查 `forward_batch_info.py:718` 附近的 `.to(device, non_blocking=True)` 调用方，看是哪个张量；对比 DP-LM-Head 关闭时的张量切分逻辑

### 7.2 怀疑 2: NPU 端 `_draft_extend_for_prefill` 在 MoE-dense-TP=1 + DP-LM-Head 组合下的 HCCL 通信死锁
- 错误伴随 `acl api failed`、`rtMemcpyAsync`、`rtDeviceSynchronizeWithTimeout` tsfw unknown error
- `HCCL_EXEC_TIMEOUT=200` (我们设的) 没生效 → 实际超时是 1800s
- 在 NPU 4 卡 + EAGLE3 场景下，draft worker 与 target worker 之间通过 HCCL 通信
- DP-LM-Head 引入了额外的 group通信，可能在 draft worker 端与 target worker 端不一致
- **建议定位方向**: 把 `HCCL_EXEC_TIMEOUT` 调小到 30s 重跑，看是否更快暴露死锁点；或加 verbose 日志看 HCCL 调用链

### 7.3 怀疑 3: NPU 端在 EAGLE3 `num_draft_tokens=32` + `num_steps=6` 大投机规模下的内存分配失败
- `num_draft_tokens=32, num_steps=6` 在 ascend 后端可能触发内存碎片化
- 错误信息 `srclen=16, dstlen=16, kind=1` 中的 16 元素张量可能恰好是某个小算子的输入
- **建议定位方向**: 把 `num_draft_tokens` 改为 4、`num_steps` 改为 3 重跑 (这就是伴随测试的配置) → 验证通过 → 收窄到这两个参数

### 7.4 排除的可能性
- ❌ **CUDA Graph 冲突**: Round 10 删 `--cuda-graph-max-bs 64` 后错误仍复现
- ❌ **`topk=10` 不支持**: Round 8 改 topk=1 后错误形态从 `ValueError` 变为 NPU 内部 RuntimeError
- ❌ **测试抖动**: 决定性复现 2 次
- ❌ **依赖缺失**: tabulate 在 Round 7 已补
- ❌ **CI 镜像问题**: 同镜像下伴随测试通过

---

## 8. 建议开发人员反馈表

| 反馈类型 | 建议联系方向 |
|---|---|
| `forward_batch_info.py:718` H2D copy 在 DP-LM-Head 下的切分问题 | sglang-Ascend 端 **DP-attention / DP-LM-Head** 方向 owner |
| `_draft_extend_for_prefill` 在 EAGLE3 + DP 下的 NPU 适配 | sglang-Ascend 端 **EAGLE3 / 投机解码** 方向 owner |
| HCCL 通信在 4 卡 NPU 集群上 tsfw unknown error | sglang-Ascend 端 **HCCL 通信 / NPU 驱动** 方向 owner |
| 内存分配失败 (`PYTORCH_NPU_ALLOC_CONF`) | sglang-Ascend 端 **NPU 内存管理** 方向 owner |

> **如果您能定位到具体 owner，请在本表"反馈类型"列加 ✅ 标记，便于后续问题跟踪。**

---

## 9. 重现步骤 (开发人员本机或 CI 上重现)

```bash
# 1. 准备 ascend 4 卡环境 (A3 集群)
# 2. 拉取 PR 分支
git clone -b spec-inference-npu-testcases https://github.com/EdwardXuy/sglang-Ascend.git
cd sglang-Ascend
# 3. 安装 sglang-Ascend (按仓库 README)
# 4. 安装 tabulate
pip install tabulate
# 5. 跑失败用例
python -m unittest sglang.test.registered.ascend.basic_function.speculative_inference.GENERATED_20260606.test_npu_eagle_dp_attention
```

或者在 CI 上触发重跑:
- 访问 https://github.com/Ascend/sglang/actions/runs/27063936417
- 点击 "Re-run jobs"

---

## 10. 附: PR 中其他 2 个通过测试 (供对照, 同样的 ascend 4 卡环境)

1. **`test_npu_constrained_decoding_spec_reasoning.py`** (Round 10 PASS, 147s)
   - 1 卡 TP=1 + EAGLE3 + qwen3 reasoning + JSON Schema 约束
   - 同样的 ascend 镜像、同样的 `NPU_ENV` → 跑通

2. **`test_npu_eagle_infer_beta_dp_attention.py`** (Round 10 PASS, 188s)
   - 4 卡 TP=2, DP=2 + EAGLE3 + `num_draft_tokens=4` + `num_steps=3` (轻量参数)
   - **与失败测试**用同样的 Qwen3-8B + EAGLE3 + DP-attention，**仅少 3 个标志**

> 这两个通过测试表明: sglang-Ascend NPU 端 EAGLE3 + DP-attention **基本通路是工作的**；失败测试涉及的"激进参数 + DP-LM-Head + MoE-dense-TP-size"组合是问题触发点。

---

## 11. 附: Round 9 vs Round 10 日志差异分析

| 维度 | Round 9 (860897d) | Round 10 (729b00c) |
|---|---|---|
| 失败位置 | `setUpClass` line 80 | `setUpClass` line 80 (line number 相同) |
| 失败时机 | 启动后 4m42s | 启动后 4m42s (完全相同) |
| Scheduler 堆栈 | `eagle_worker_v2.py:937 → 665 → forward_batch_info.py:718` | **完全相同** |
| NPU 驱动错误 | `acl api failed / rtMemcpyAsync tsfw / rtDeviceSynchronizeWithTimeout` | **完全相同** |
| `--cuda-graph-max-bs 64` | 存在 | 删除 |
| 结论 | 错误**100% 一致** | 错误**100% 一致** |

→ 排除 `--cuda-graph-max-bs 64` 是根因。

---

## 12. 联系信息

- **PR 作者**: EdwardXuy
- **PR URL**: https://github.com/Ascend/sglang/pull/710
- **PR 评论页**: https://github.com/Ascend/sglang/pull/710 (请在该 PR 上评论, 便于追踪)
- **本地日志**: `D:\debug-pr\.trae\logs\npu_round10_729b00c.log` (开发人员可下载此文件而无需从 GitHub 重新 fetch)
