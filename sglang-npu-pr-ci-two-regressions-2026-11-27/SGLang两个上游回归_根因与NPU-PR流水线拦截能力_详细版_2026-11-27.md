# SGLang 两个上游回归的根因与 NPU PR 流水线拦截能力分析（详细版）

- 报告日期：2026-11-27
- 分析对象：sgl-project/sglang nightly NPU 流水线两起共性问题
- 结论时效：截至 2026-11-27
- 数据来源：GitHub Actions run/job 日志、PR diff、sglang 与 sgl-kernel-npu 代码、NPU CI workflow 与用例注册信息

---

## 0. 结论摘要

### 0.1 两个问题一句话结论

| 编号 | 问题 | 根因上游 PR | 失败阶段 | 影响面 | 当前状态 |
|---|---|---|---|---|---|
| 问题 1 | `TypeError: markov_sample_step_triton() got an unexpected keyword argument 'softcap'` | **#51298**（从 **#50186 Inkling-2-Small** 拆出） | 服务启动完成后的 **startup warmup 首个 decode batch** | 所有使用 DSPARK 的模型（DeepSeek-V4、Kimi-K3）服务无法拉起 | sgl-kernel-npu **#1070** 已修复（2026-11-20 合入） |
| 问题 2 | `capture_end ... error code is 107025` / `EE9999 ... task_type=3 (EVENT_WAIT)` | **#51900** | Scheduler 初始化阶段的 **decode NPU graph 捕获** | Kimi-K3 32P 三个 nightly 用例；GLM/DeepSeek 不受影响 | sglang **#53214** 修复 PR 已提交，尚未合入 |

### 0.2 三句话结论（可直接向主管复述）

1. **两起都是“NVIDIA 路径验证通过、NPU 路径被破坏”的上游回归**：问题 1 是 sglang 新增 `softcap` 参数后，sgl-kernel-npu 的 Python wrapper 签名未同步；问题 2 是 Kimi-K3 在 NPU 专属的共享专家 side-stream 路径上新增了 `wait_stream`，违反 ACL 图捕获规则。
2. **三个上游 PR 的 NPU PR 流水线都没有执行任何 NPU 用例**：`#51298`、`#50186`、`#51900` 均因缺少 `run-ci-npu-extra` 标签被 pr-gate 拦截；`#53214` 虽带齐标签，仍被 precision 选择器判定 `skip_all`，只跑了 CPU 单测。
3. **“有没有拦截”和“能不能拦截”要分开看**：
   - 问题 1：PR 套件里**存在**能拦截的用例（`extra-c-acc-test-16-npu-a3` 中的 DeepSeek-V4-Flash W8A8 8p DSPARK 用例），但**没有执行**；
   - 问题 2：PR 套件里**完全没有** Kimi-K3 用例（Kimi-K3 32P 只在 nightly 多机跑），即使执行也**拦不住**，属于结构性覆盖缺口。

### 0.3 责任边界

| 事项 | 社区侧 | 我们侧（NPU CI） |
|---|---|---|
| 问题 1 引入 | #51298 改了 sglang 侧接口，未同步 sgl-kernel-npu wrapper；sgl-kernel-npu #1070 已补兼容 | PR 流水线未运行；但套件里有可拦截用例，属于“门禁未触发” |
| 问题 2 引入 | #51900 修改了 NPU-only 的 SBO 分支，PR 描述称“NPU fine-grained path unchanged”，但非 fine-grained 路径实际被改；修复 #53214 已提交 | PR 套件无 Kimi-K3 用例，属于“覆盖能力不足”，即使跑也拦不住 |

---

## 1. 分析范围与方法

### 1.1 范围

- 问题 1：2026-11-19 nightly 首次出现的 `markov_sample_step_triton() got an unexpected keyword argument 'softcap'`。
- 问题 2：2026-11-21 nightly 首次出现的 Kimi-K3 decode graph `capture_end` 107025。
- 对应上游 PR 的 `PR Test (NPU)` checks：是否触发、是否执行用例、若执行能否发现。

### 1.2 证据等级

| 等级 | 含义 | 本报告中的使用 |
|---|---|---|
| A | GitHub Actions 原始日志/PR diff 直接可见 | 两起错误的堆栈、失败阶段、job 结论、标签、precision 决策 |
| B | 代码路径可直接推导 | DSV4 与 KimiK3 共用 DSpark `_draft_step`；SBO 分支为 NPU-only |
| C | 需要补证 | DeepSeek-V4 侧未保留容器内 traceback（pod 直接退出），只能由共享代码路径推断 |

> 说明：问题 1 的完整 Python 堆栈来自 Kimi-K3 job；同夜 3 个 DeepSeek-V4 job 的 decode pod 同样在启动阶段退出（job 日志只记录 `Pod ... is not running, status: Succeeded`，harness 未保留容器内 traceback）。由于 DSV4 与 KimiK3 共用 `DSparkWorkerV2 -> _draft_step -> MarkovSampleStep.execute` 路径，判定为同一根因；如需 100% 实证，需要补抓 DSV4 pod 日志（见 5.3）。

---

## 2. 问题 1：markov 采样算子不接受 `softcap` 参数

### 2.1 现象

- 首个失败 nightly：run **44562390109**（2026-11-19T16:06:48Z 创建，head `6515ac29d3`）。
- 同夜 6 个模型 job 失败：3 个 Kimi-K3 32P + 3 个 DeepSeek-V4。
- Kimi-K3 gpqa job（job **135859074748**）日志中，16 个 rank 在 `2026-11-19T20:54:09` 同一秒抛出：

```text
File "/root/sglang/python/sglang/kernels/ops/speculative/dspark/markov_sample.py", line 268, in execute
  return markov_sample_step_triton(**kwargs)
TypeError: markov_sample_step_triton() got an unexpected keyword argument 'softcap'
```

- Scheduler 退出后，`/health` 一直 503，用例最终健康检查超时。

### 2.2 上游 PR 定位

#### 2.2.1 直接根因：#51298

- 标题：`[Feat] dspark: Let the draft fold its verify commit and cap step logits`
- 链接：https://github.com/sgl-project/sglang/pull/51298
- 合入时间：**2026-11-19T07:15:38Z**
- merge commit：`21be2189de65d7b70467dbd61d833b17106c6d46`
- PR 描述原文关键句：

> Split out of #50186 (Inkling-2-Small support). The Inkling2 MTP draft runs on DSpark. ...
> Markov sampling: `BaseDSparkMarkovHead.step_softcap` / `step_vocab_limit` apply a tanh softcap to the step logits and mask ids from `vocab_limit` on. Both default to off.

该 PR 在 `MarkovSampleStep` 上新增 `softcap` / `vocab_limit` 两个可选参数，并在 `_draft_step` 中无条件传给 `MarkovSampleStep.execute()`。

#### 2.2.2 与 #50186（Inkling-2-Small）的关系

- #50186 标题：`[Feat] inkling2: Support Inkling-2-Small`
- 链接：https://github.com/sgl-project/sglang/pull/50186
- 合入时间：**2026-11-19T15:35:21Z**
- merge commit：**`6515ac29d3c2206ec4a9b54c40e6b2b25c3a43a3`**
- 该 commit 正是 11-19 nightly 的 head，也是大家印象中“带 inkling small 2 字样”的 PR。
- #51298 是从 #50186 中拆出的 DSpark 基础改动；#50186 本体主要是 Inkling2 模型支持。**引入 `softcap` 关键字的是 #51298，不是 #50186**；两者在 11-19 nightly 中同时存在，因此首个失败 run 的 head 指向 #50186。

#### 2.2.3 时间线

| 时间（UTC） | 事件 |
|---|---|
| 2026-11-18 18:03 | #51298 创建 PR |
| 2026-11-19 07:15 | #51298 合入 main（`21be2189de`） |
| 2026-11-19 15:35 | #50186 合入 main（`6515ac29d3`） |
| 2026-11-19 16:06 | nightly run 44562390109 创建（head `6515ac29d3`） |
| 2026-11-19 20:54 | Kimi-K3 16 个 rank 抛 `TypeError: ... 'softcap'` |
| 2026-11-20 03:50 | sgl-kernel-npu #1070 合入，补 wrapper 兼容 |

### 2.3 社区 PR 改了什么

`#51298` 对 `MarkovSampleStep` 的改动（节选）：

```python
# markov_sample.py
class MarkovSampleStep:
    @classmethod
    def execute(cls, **kwargs) -> torch.Tensor:
        if inputs_on_cuda(**kwargs):
            if _is_npu:
                from sgl_kernel_npu.operators.sample import markov_sample_step_triton
                return markov_sample_step_triton(**kwargs)   # 所有 kwargs 原样转发
            return cls.triton(**kwargs)
        return cls.torch(**kwargs)
```

```python
# dspark_draft.py
tokens = MarkovSampleStep.execute(
    base_logits=base_logits,
    bias=bias,
    temperatures=temperatures,
    key=noise.key,
    offset=noise.offset,
    step=step,
    step_dtype=markov_head.step_logits_dtype(base_logits.dtype),
    logits_out=logits_out,
    has_sampled_rows=None if logits_out is not None else False,
    softcap=markov_head.step_softcap,        # 新增
    vocab_limit=markov_head.step_vocab_limit, # 新增
    **sharded,
)
```

代码位置：

- [markov_sample.py L262-L270](file:///d:/debug-sgl-sglang/sglang-Ascend/python/sglang/kernels/ops/speculative/dspark/markov_sample.py#L262-L270)
- [dspark_draft.py L239-L253](file:///d:/debug-sgl-sglang/sglang-Ascend/python/sglang/srt/speculative/dspark_components/dspark_draft.py#L239-L253)
- [markov_sample.py L315-L320](file:///d:/debug-sgl-sglang/sglang-Ascend/python/sglang/kernels/ops/speculative/dspark/markov_sample.py#L315-L320)（`softcap` 参数定义）

关键点：即使 `step_softcap` / `step_vocab_limit` 都是 `None`，关键字仍然会被传入。CUDA 的 `MarkovSampleStep.triton` 已经接受这两个参数；NPU 的 `sgl_kernel_npu.operators.sample.markov_sample_step_triton` 仍是旧签名，因此直接 `TypeError`。

### 2.4 错误发生逻辑（逐层调用链）

```text
scheduler.event_loop_overlap                 (scheduler.py:2271)
  -> run_batch
  -> dspark_worker_v2.forward_batch_generation (dspark_worker_v2.py:799)
  -> _forward_decode                          (dspark_worker_v2.py:1088)
  -> proposer.propose                         (dspark_draft.py:529)
  -> sample_draft_block                       (dspark_draft.py:333)
  -> draft_markov_block                       (dspark_draft.py:297)
  -> _draft_step                              (dspark_draft.py:239)
  -> MarkovSampleStep.execute                 (markov_sample.py:263)
  -> sgl_kernel_npu markov_sample_step_triton(**kwargs)
  -> TypeError: unexpected keyword argument 'softcap'
```

#### 为什么只有 NPU 报错

- CUDA 路径：`MarkovSampleStep.triton` 是 sglang 自己的 Triton 实现，#51298 已同步加入 `softcap` / `vocab_limit` 形状参数。
- NPU 路径：`MarkovSampleStep.execute` 在 `_is_npu` 时改走 `sgl_kernel_npu.operators.sample.markov_sample_step_triton`。该函数位于**另一个仓库**，签名仍是旧版，未接受新增关键字。
- 这是跨仓库接口契约问题，不是 Triton 编译失败，也不是 CANN 运行时错误。

#### 为什么 DeepSeek-V4 和 Kimi-K3 都中招

- 两者都通过 `--speculative-algorithm DSPARK` 使用 DSpark draft。
- `_draft_step` 是 DSpark 共享实现，模型无关；只要走到 markov 采样，就会把 `softcap=None` 传进 NPU wrapper。
- GLM 等其他模型不走 DSpark，因此不受影响。

### 2.5 发生在服务拉起的哪个阶段

以 Kimi-K3 gpqa job 日志为准：

| 时间 | 日志 | 说明 |
|---|---|---|
| 2026-11-20 04:53:38 | `Engine startup timings ... cuda_graph={... draft_decode=5.84 ...}`、`Uvicorn running`、`Application startup complete` | 权重加载、KV cache、图捕获均已完成，HTTP server 已起来 |
| 04:53:39 | `Using a 448x448 image for Kimi K3 VLM startup warmup` | 进入启动 warmup |
| 04:53:40 - 04:54:01 | `GET /health` 连续返回 503 | warmup 尚未完成，健康检查未通过 |
| 04:54:09 | `Scheduler hit an exception ... TypeError ... 'softcap'` | warmup 的首个 decode batch 触发 markov 采样 |
| 之后 | `Server http://.../health failed to start in 3600s` | scheduler 退出，健康检查超时 |

结论：

- **不是模型加载阶段，也不是 graph capture 阶段**；
- 是**服务进程启动完成、HTTP server 已监听、startup warmup 请求触发首个 decode batch** 时出错；
- 从 `draft_decode=5.84s` 可见 draft decode 图捕获已经完成，说明 markov 采样调用发生在图捕获之外的 eager 路径，因此错误没有在 capture 阶段暴露。

### 2.6 影响面

- 所有使用 DSPARK 的模型服务拉起都会失败：DeepSeek-V4、Kimi-K3。
- 11-19 nightly 同夜 6 个模型 job 失败（3 个 Kimi-K3 32P + 3 个 DeepSeek-V4）。
- 若该 commit 进入发布分支，将影响所有 DSPARK + NPU 部署。

### 2.7 社区修复与现状

- 修复仓库/PR：**sgl-project/sgl-kernel-npu #1070** `[NPU] Guard Markov operator keyword compatibility`
- 链接：https://github.com/sgl-project/sgl-kernel-npu/pull/1070
- 合入时间：**2026-11-20T03:50:59Z**
- merge commit：`f5575fdc835a5b7dcd67d2b81b5b4c7cb7365362`
- 修复方式：两个 Markov wrapper 增加 `**kwargs`，**忽略**额外的 `softcap` / `vocab_limit`，不转发给后端 kernel。

```python
def markov_sample_step_triton(
    ...,
    exp_noise=None,
    impl="auto",
    **kwargs,          # 新增：忽略额外关键字
):
    """Dispatch to the selected backend, ignoring extra keyword arguments."""
```

PR 自述验证：`git diff --check` passed. **No physical NPU run.**（该修复本身也没有在真实 NPU 上跑过。）

#### 残留风险

#1070 只是“接受并忽略”这两个参数，并没有在 NPU 上实现 `tanh softcap` 与 `vocab_limit` 语义。当前 DeepSeek-V4 / Kimi-K3 的 `step_softcap` 为 `None`，所以数值无影响；但 Inkling2 MTP 等未来使用 `step_softcap` 的模型在 NPU 上会**静默忽略**该选项，存在精度偏差风险。建议后续二选一：

1. 在 sgl-kernel-npu 中真正实现 `softcap` / `vocab_limit`；
2. 或在参数非 `None` 时显式报错，避免静默错误。

### 2.8 为什么 CUDA/其他 CI 没发现

- 问题只在 NPU 分支：`_is_npu` 为真时才 import `sgl_kernel_npu`；
- NVIDIA PR 测试不会安装/调用 sgl-kernel-npu；
- sgl-kernel-npu 的改动没有反向触发 sglang 的 NPU PR 测试；
- 因此唯一能在合入前发现的是 **NPU PR 流水线**，而它没有执行用例（见第 4 章）。

---

## 3. 问题 2：Kimi-K3 decode 图捕获 capture_end 107025

### 3.1 现象

- 首个失败 nightly：run **44922677219**（2026-11-21T16:05:55Z 创建，head `9b4b6b6d`）。
- 3 个 Kimi-K3 32P 多机用例失败：gpqa、prefill、decode-prefix。
- job 136978977457 日志：

```text
RuntimeError: capture_end:../torch_npu/csrc/core/npu/NPUGraph.cpp:240
NPU function error: c10_npu::acl::AclmdlRICaptureEnd(...), error code is 107025
EE9999: The type of the task is not event record nor capture record,
stream_id=87, task_id=735, task_type=3 (EVENT_WAIT)
Exception: Capture cuda graph failed: ... error code is 107025
```

- 11-24 复现 run 45212673268；PR 验证 run 45343894905（CANN 9.2 镜像）同样报 107025，只是 torch_npu 行号变为 `NPUGraph.cpp:371`，说明与镜像无关。

### 3.2 上游 PR 定位

- 标题：`[Perf] kimi-k3: Order shared-expert side-stream frees on the current stream`
- 链接：https://github.com/sgl-project/sglang/pull/51900
- 合入时间：**2026-11-21T04:13:40Z**
- merge commit：`8396d60a14ef03b23d2cdaa5c64adb18eba7c703`
- PR 描述关键句：

> Under `_sbo_shared_overlap`, `KimiK3MoE` runs the shared experts on `alt_stream` ...
> The NPU fine-grained overlap path is unchanged.
> Ordering only; no numerics change.

PR 作者只在 GB300（NVIDIA）上验证了显存收益（每 GPU 节省 5.8 GB），没有覆盖 NPU 的非 fine-grained SBO 路径。

### 3.3 社区 PR 改了什么

`#51900` 对 `kimi_k3.py` 的改动（节选）：

```python
# 1) 共享专家输入不再 record_stream 到 alt_stream
-                shared_input.record_stream(self.alt_stream)
                 self.alt_stream.wait_stream(current_stream)
                 with torch.cuda.stream(self.alt_stream):
                     shared_output = self.shared_experts(shared_input)

# 2) 新增 shared_alt_output 与 release_shared_output
+        shared_alt_output = None
...
         def wait_and_finalize_shared_experts():
-            nonlocal shared_output
+            nonlocal shared_output, shared_alt_output
             ...
-            shared_output.record_stream(current_stream)
+            if fine_grained_overlap:
+                shared_output.record_stream(current_stream)
+            else:
+                shared_alt_output = shared_output
...
+        def release_shared_output():
+            nonlocal shared_alt_output
+            if shared_alt_output is not None:
+                self.alt_stream.wait_stream(torch.cuda.current_stream())
+                shared_alt_output = None
```

调用点：

- 非 latent MoE 路径末尾：`release_shared_output()`（L1498）
- latent MoE 路径末尾：`release_shared_output()`（L1540）

代码位置：

- [kimi_k3.py L1462-L1466](file:///d:/debug-sgl-sglang/sglang-Ascend/python/sglang/srt/models/kimi_k3.py#L1462-L1466)
- [kimi_k3.py L1495-L1498](file:///d:/debug-sgl-sglang/sglang-Ascend/python/sglang/srt/models/kimi_k3.py#L1495-L1498)
- [kimi_k3.py L1537-L1540](file:///d:/debug-sgl-sglang/sglang-Ascend/python/sglang/srt/models/kimi_k3.py#L1537-L1540)

### 3.4 错误发生逻辑

#### 触发条件

`_sbo_shared_overlap` 定义（L802-L806）：

```python
self._sbo_shared_overlap = (
    self._ep_a2a
    and self.shared_experts is not None
    and self.alt_stream is not None
)
```

Kimi-K3 32P 用例同时满足：

- `--moe-a2a-backend deepep` -> `_ep_a2a=True`
- `--enable-shared-experts-attn-tp` -> `shared_experts` 非空
- `alt_stream` 非空
- 未开启 fine-grained overlap（走 `else` 分支）

#### ACL 图捕获规则

- ACL 的 capture 模型要求：**capture 期间子 stream 上的最后一个任务必须是 event record**，不能是 event wait。
- `release_shared_output()` 里的 `self.alt_stream.wait_stream(torch.cuda.current_stream())` 会在 alt stream 上排入一个 **EVENT_WAIT** 任务。
- 该调用位于 `KimiK3MoE.forward` 内，而 decode 图捕获会执行完整的 MoE forward，因此 EVENT_WAIT 落在 capture 区间内。
- `capture_end` 校验失败：`capture model contains a stream that was not joined to the origin stream` -> 107025。

#### 为什么 CUDA CI 没发现

- `if _is_npu and self._sbo_shared_overlap: issue_shared()`（L1470）——整个 SBO 分支是 **NPU-only**。
- NVIDIA 的 Kimi-K3 测试（如 `test_kimi_k3_bfa_overlap.py`，`register_cuda_ci(4-gpu-b200)`）不会执行这段代码。
- PR #51900 的 NVIDIA 验证只覆盖显存收益，覆盖不到 NPU ACL 的 capture 约束。

#### 为什么只有 Kimi-K3 受影响

- 逻辑只在 `KimiK3MoE`（`kimi_k3.py`）中；
- GLM 不经过该分支；DeepSeek-V4 走 `dspark.py`，不使用 `KimiK3MoE`；
- 因此同夜 GLM 用例通过，DSV4 不受 #51900 影响。

### 3.5 发生在服务拉起的哪个阶段

job 138280635172 的堆栈（11-24，CANN 9.2）：

```text
scheduler.py:7235  run_scheduler_process -> Scheduler.__init__
scheduler.py:636   Scheduler.__init__ -> init_model_worker
scheduler.py:1201  init_model_worker -> init_all_cuda_graphs
scheduler.py:1179  init_all_cuda_graphs -> tp_worker.init_cuda_graphs
tp_worker.py:438   -> model_runner.init_cuda_graphs
model_runner.py:1270 -> capture_cuda_graphs
cuda_graph_setup.py:453 -> capture_decode_graph
cuda_graph_setup.py:851 -> GraphRunnerCls(model_runner)
decode_cuda_graph_runner.py:688 -> __init__ -> capture()
decode_cuda_graph_runner.py:1327 -> _capture()
decode_cuda_graph_runner.py:1400 -> _capture_one_stream()
decode_cuda_graph_runner.py:1455 -> capture_one_shape()
npu_cudagraph_backend.py:415 -> capture_one()
npu_cudagraph_backend.py:449 -> _capture_one() with graph_ctx(...)
torch_npu/npu/graphs.py:928 -> __exit__ -> capture_end()
RuntimeError: ... 107025
```

结论：

- 发生在 **Scheduler 进程初始化阶段**，具体是 `init_all_cuda_graphs -> capture_decode_graph -> DecodeCudaGraphRunner.capture` 的 **decode 图捕获**；
- 此时 HTTP server **尚未启动**，所以外部看到的是 `/health` 长时间不可用、3600s 超时；
- 与问题 1 的阶段不同：问题 1 在 server 起来后的 warmup，问题 2 在 server 起来之前的 scheduler 初始化。

### 3.6 影响面

- 3 个 Kimi-K3 32P 多机用例连续失败（gpqa / prefill / decode-prefix）；
- 11-21、11-22、11-23 nightly 均复现；11-24 PR 验证 run 仍复现；
- 只要 Kimi-K3 + 共享专家 + DeepEP + decode graph 的部署都会中招；
- GLM/DeepSeek 不受 #51900 影响。

### 3.7 社区修复与现状

- 修复 PR：**sgl-project/sglang #53214** `[Fix] [NPU] kimi-k3: Fix graph capture with shared-expert overlap`
- 链接：https://github.com/sgl-project/sglang/pull/53214
- 状态：**open**（2026-11-26 创建）
- 改动（+2/-1）：

```python
-            if fine_grained_overlap:
+            # TODO: Remove this workaround once PTA fixes side-stream graph capture.
+            if _is_npu:
                 shared_output.record_stream(current_stream)
             else:
                 shared_alt_output = shared_output
```

- 验证：单台 Ascend A3、12 层 Kimi-K3、TP16/EP16、DSPARK，target 与 draft 图捕获均成功，13 个 probe 请求通过。
- 注意：该修复 PR 虽然带了 `run-ci` + `run-ci-npu-extra`，但其 NPU PR 流水线仍然只跑了 CPU 单测，所有真实 NPU 套件被 precision 选择器跳过（见 4.3.2）。

---

## 4. NPU PR 流水线拦截能力分析

### 4.1 流水线结构

`pr-test-npu.yml` 的主要 job：

| Job | 内容 | 规模 |
|---|---|---|
| `check-changes` | 路径过滤，判断 `main_package` / `multimodal_gen` | ubuntu-latest |
| `pr-gate` | 标签门禁，`require-label: run-ci-npu-extra` | ubuntu-latest |
| `extra-a-test-npu` | CPU 单测（mock NPU） | ubuntu-latest |
| `extra-b-test-1/2/4/8/16-npu-a3` | 单机 1/2/4/8/16 卡 NPU 用例 | A3 单机 |
| `extra-c-acc-test-2/16-npu-a3` | 单机 accuracy 用例 | A3 单机 |
| `extra-c-perf-test-2/16-npu-a3` | 单机 perf 用例 | A3 单机 |
| `multimodal-gen-test-1/4-npu-a3` | 扩散多模态用例 | A3 单机 |
| `recommend-tests-from-coverage` | 基于覆盖率推荐用例并分桶 | ubuntu-latest |

关键限制：

- **PR 流水线最大只有单机 16 卡**；
- Kimi-K3 32P 是 `--nnodes 4`（4 节点 × 16 卡）的多机用例，只存在于 **nightly 多机 workflow**（`nightly-test-npu-e2e-multi-node.yml`），不在 PR 套件中。

### 4.2 门禁逻辑

`pr-gate.yml` L63-L73：

```yaml
- name: Skip tests
  id: skip
  if: >-
    env.IS_PR_EVENT == 'true' && (
      fromJson(steps.pr.outputs.draft)
      || !contains(fromJson(steps.pr.outputs.labels), 'run-ci')
      || inputs.require-label != '' && !contains(fromJson(steps.pr.outputs.labels), inputs.require-label)
    )
  run: echo "::notice::Tests skipped. ..."
```

输出 `opted-in=false`，所有测试 job 的 `if` 条件都包含 `needs.pr-gate.outputs.opted-in != 'false'`，因此全部 skip。

`pr-test-npu.yml` 中：

```yaml
pr-gate:
  needs: check-changes
  if: ${{ ... needs.check-changes.outputs.changes_exist == 'true' }}
  uses: ./.github/workflows/pr-gate.yml
  with:
    require-label: run-ci-npu-extra
```

也就是说，NPU PR 测试需要**同时**具备 `run-ci` 和 `run-ci-npu-extra` 两个标签。

### 4.3 三个上游 PR 的实际 checks

| PR | NPU run | 结论 | Job 统计 | 跳过原因 |
|---|---|---|---|---|
| #51298 | [44406803384](https://github.com/sgl-project/sglang/actions/runs/44406803384) | success | 22 个：4 success / 18 skipped | 标签 `[run-ci, jit-kernel]`，缺 `run-ci-npu-extra` |
| #50186 | [44504215994](https://github.com/sgl-project/sglang/actions/runs/44504215994) | success | 22 个：4 success / 18 skipped | 标签含 `run-ci`、`run-ci-nv-extra`，缺 `run-ci-npu-extra` |
| #51900 | [44845789645](https://github.com/sgl-project/sglang/actions/runs/44845789645) | success | 22 个：4 success / 18 skipped | 标签为空 `[]`，缺 `run-ci` 和 `run-ci-npu-extra` |
| #53214（修复） | [45933003123](https://github.com/sgl-project/sglang/actions/runs/45933003123) | success | 23 个：`extra-a-test-npu` success；`extra-b/c` 全部 skipped | 标签齐全，但 precision 选择器推荐 0 个用例 -> `skip_all` |

> 说明：前三个 run 中成功的 4 个 job 是 `check-changes`、`pr-gate`、`set-image-config`、`Analyze failure report`，没有任何真实 NPU 用例执行。`pr-test-npu-finish` 也因 `opted-in=false` 被 skip，所以 workflow 结论仍是 success。

#### 4.3.1 标签跳过（#51298 / #50186 / #51900）

pr-gate 日志原文：

- #51900：`PR Labels: []` -> `##[notice]Tests skipped. ... needs 'run-ci' and 'run-ci-npu-extra'.`
- #51298：`PR Labels: [run-ci,jit-kernel]` -> 同样 notice
- #50186：`PR Labels: [documentation,quant,...,run-ci,jit-kernel,...,run-ci-nv-extra]` -> 同样 notice

#### 4.3.2 precision 选择器跳过（#53214）

即使标签齐全，`recommend-tests-from-coverage` 的日志显示：

```text
=== No Test Cases Recommended ===
Recommended test entries: 0 (0 files)
Suite extra-b-test-1-npu-a3: 19 files, 0 recommended
Suite extra-b-test-2-npu-a3: 13 files, 0 recommended
...
Suite extra-c-acc-test-16-npu-a3: 2 files, 0 recommended
Decision: skip_all (no test recommended); 1 changed path(s), 0 full-run path(s)
##[notice]No test recommended — skipping all suites.
```

原因是 `PRECISION_MODE: 'true'`，且 `bucket_recommended_tests.py` 只有在变更命中 `_FULL_RUN_PATHS`（仅 `.github/workflows/pr-test-npu.yml`、`python/pyproject_npu.toml`、`scripts/ci/npu/npu_ci_install_dependency.sh`）时才全量跑；模型代码 `kimi_k3.py` 不在其中，推荐为 0 时直接 `skip_all`。

结果：#53214 的 NPU run 只有 `extra-a-test-npu`（CPU 单测）执行，所有真实 NPU 套件跳过，workflow 依然 success。

### 4.4 如果 NPU PR 用例全部运行，能否发现问题 1？

**结论：能，前提是 `extra-c-acc-test-16-npu-a3` 被完整执行（或覆盖率推荐命中该用例）。**

关键用例：

- 文件：`test/registered/npu/accuracy/deepseek_v4_flash/test_npu_deepseek_v4_flash_w8a8_8p_gpqa.py`
- 注册：`register_npu_ci(est_time=3600, suite="extra-c-acc-test-16-npu-a3")`
- 模型：DeepSeek-V4-Flash W8A8
- 启动参数：`--tp-size 16`、`--attn-dp-size 16`、`--moe-a2a-backend deepep`、`--speculative-algorithm DSPARK`、`--speculative-dspark-block-size 6`、`--cuda-graph-bs-decode 1 2 4 6 8 10`
- PR 模式：`_is_pr_pipeline` 为真时跑 gsm8k（nightly 模式跑 gpqa_diamond），但仍然会启动 DSPARK 服务并执行 decode。

因此，如果该 job 执行，服务会在 startup warmup 的首个 decode batch 触发 `TypeError: ... 'softcap'`，用例在 `wait_for_server`/accuracy 阶段失败。

其他相关用例的覆盖情况：

| 用例 | 套件 | 是否覆盖问题 1 | 原因 |
|---|---|---|---|
| `test_npu_deepseek_v4_flash_w8a8_8p_gpqa.py` | `extra-c-acc-test-16-npu-a3`（PR） | **是** | DSPARK + decode graph，真实 NPU 启动 |
| `test_dspark_npu_dispatch.py` | `extra-b-test-1-npu-a3`（PR） | 否 | 只测 BuildOutTokens / MarkovGreedyStep / argmax / accept / schedule / softmax，**没有调用 `MarkovSampleStep`** |
| `test_dspark_markov_sample.py` | CUDA `base-b-kernel-unit` | 否 | 只测 `MarkovSampleStep.triton` / `.torch`，`DEVICE=cuda`，NPU 不执行 |
| `test_inkling2.py` | CPU `base-a-test-cpu` + CUDA `base-c` | 否 | 不在 NPU 套件；CPU 路径走 `torch`，不会触发 NPU wrapper |
| `test_npu_deepseek_v4_flash_0731_w8a8_1p1d_16p_gpqa.py` | nightly（`suite=""`） | 否 | 只在 nightly 多机跑 |
| `test_npu_kimi_k3_w4a8_32p_gpqa.py` | nightly（`suite=""`） | 否 | 只在 nightly 4 节点跑 |

**风险点**：`extra-c-acc-test-16-npu-a3` 是否执行还取决于 precision 推荐。若 `markov_sample.py` 的变更没有被覆盖率基线推荐到该用例，bucket 会返回 `should_run=false`，job 仍会 skip。因此“能拦截”不等于“必然拦截”；只有全量运行或推荐命中时才成立。

### 4.5 如果 NPU PR 用例全部运行，能否发现问题 2？

**结论：不能。**

原因：

1. `test_npu_kimi_k3_w4a8_32p_gpqa.py` 及两个 performance 用例都是 `suite=""`、`nightly=True`，只在 nightly 多机 workflow 中执行；
2. PR 套件最大单机 16 卡，无法启动 `--nnodes 4`、`--tp-size 64` 的 Kimi-K3 32P；
3. Kimi-K3 相关单测：
   - `test_kimi_k3_bfa_overlap.py` 是 `register_cuda_ci(4-gpu-b200)`，CUDA only；
   - 其他 `test_kimi_k3_*` 是 CPU 单测，不执行 NPU SBO 分支；
4. NPU 的 `test_npu_graph_capture.py` 只验证最小 `torch.npu.NPUGraph` capture/replay，不涉及 `alt_stream.wait_stream` / KimiK3MoE；
5. 因此即使 `extra-a/b/c` 全部执行，也没有任何用例会走到 `KimiK3MoE` 的 SBO 分支。

### 4.6 拦截能力矩阵

| 问题 | PR 流水线是否执行了用例 | 套件里是否有能拦截的用例 | 最终能否拦截 | 缺口类型 |
|---|---|---|---|---|
| 问题 1（softcap） | 否（#51298 / #50186 均 0 用例） | 有（`extra-c-acc-test-16` 的 DSV4 8p DSPARK） | 全量运行：能；precision：不一定 | 门禁未触发 + 推荐可能漏检 |
| 问题 2（107025） | 否（#51900 0 用例） | 无（Kimi-K3 只在 nightly 多机） | 不能 | 结构性覆盖缺口 |

### 4.7 结论：是“没跑”还是“跑不到”

- **问题 1 是“没跑”**：PR 套件里有能拦截的 DSV4 8p DSPARK 用例，但三个 PR 都被标签门禁拦住；即使带标签，precision 也可能把它过滤掉。
- **问题 2 是“跑不到”**：PR 套件根本没有 Kimi-K3 多机用例，只有 nightly 才跑；因此不能把责任完全归因于“社区没加标签”，我们的 CI 覆盖也存在明确缺口。
- **共同问题**：workflow 在“0 个 NPU 用例执行”的情况下仍然给出 success，形成 false green，容易让合入方误以为 NPU 已验证。

---

## 5. 责任边界与改进建议

### 5.1 社区侧

1. **#51900**：修改了 NPU-only 的 SBO 分支，但 PR 描述只强调“NPU fine-grained path unchanged”，未覆盖非 fine-grained 路径；且没有 NPU 验证。修复 #53214 已提交，建议推动合入并补一条 NPU 图捕获回归用例。
2. **#51298**：sglang 新增 `softcap` / `vocab_limit` 关键字，但未同步 sgl-kernel-npu wrapper；属于跨仓库接口契约缺失。sgl-kernel-npu #1070 已补 `**kwargs` 兼容，但没有实现语义。
3. **流程**：模型/内核类 PR 默认不带 `run-ci-npu-extra`，NPU 测试实际是 opt-in；建议在 CONTRIBUTING 或 PR 模板中明确“修改 NPU 相关代码必须请求 NPU CI”。

### 5.2 我们侧 CI 改进建议

#### P0（建议本周落地）

1. **路径触发的自动 opt-in**
   - 对 `python/sglang/srt/models/kimi_k3.py`、`python/sglang/kernels/ops/speculative/dspark/**`、`python/sglang/srt/hardware_backend/npu/**` 等 NPU 敏感路径，自动添加 `run-ci-npu-extra` 或直接触发 NPU 套件，避免依赖作者手动加标签。
2. **补一个廉价的 NPU 单测：MarkovSampleStep 关键字兼容**
   - 放在 `extra-b-test-1-npu-a3`（约 90s 级）：
     - 构造最小 `base_logits/bias/temperatures/offset`，调用 `MarkovSampleStep.execute(..., softcap=None, vocab_limit=None)`；
     - 断言不抛 `TypeError`，并与 `MarkovSampleStep.torch` 结果一致；
     - 可再加 `softcap=1.0` / `vocab_limit=...` 的语义用例（若 NPU 尚未实现，先断言显式报错或记录 xfail）。
   - 该用例可在 PR 阶段直接拦截问题 1。
3. **补一个 NPU 图捕获 side-stream 单测**
   - 在 `extra-b-test-1-npu-a3` 增加“capture 区间内 `alt_stream.wait_stream(current_stream)` 会触发 107025”的最小复现/防回归用例；
   - 或者更好：用 12 层 Kimi-K3（#53214 已验证的单机 TP16/EP16 DSPARK 配置）做一个 capture smoke case，直接覆盖 `KimiK3MoE` SBO 分支。

#### P1（建议两周内）

4. **precision 选择器的兜底**
   - 对模型/内核代码变更，当推荐结果为 0 时，不要直接 `skip_all`，而是跑一个固定的 smoke 集（至少 extra-a + 一个单机 NPU smoke）；
   - 或把 `python/sglang/srt/models/**`、`python/sglang/kernels/ops/speculative/**` 加入“必须跑 smoke”的路径列表。
5. **Kimi-K3 单机 smoke 进 PR 套件**
   - 将 #53214 的 12 层、TP16/EP16、DSPARK 配置固化为 `extra-b-test-16-npu-a3` 或 `extra-c-acc-test-16-npu-a3` 的用例；
   - 这样问题 2 这类“只有多机才暴露”的回归可以在 PR 阶段以单机缩小模型拦截。
6. **多机用例的 PR 触发**
   - 对 Kimi-K3 32P 这类只能在 4 节点跑的场景，增加 `run-ci-npu-nightly` 标签，允许 PR 按需触发 nightly 多机套件；
   - 至少对修改 `kimi_k3.py` 的 PR 强制要求。

#### P2（流程与可观测性）

7. **false green 治理**
   - 当 NPU 敏感路径变更且 0 个 NPU 用例执行时，让 check 返回 neutral/失败，而不是 success；
   - 在 PR 页面明确展示“NPU tests skipped，原因：缺标签 / precision 推荐为空”。
8. **跨仓库契约测试**
   - 在 sglang 侧增加对 `sgl_kernel_npu` 公开算子签名的契约测试（可用 `inspect.signature` 在 NPU 环境校验），避免再次出现“sglang 加参数、NPU wrapper 不认”的问题。
9. **DSV4 pod 日志留存**
   - 当前 multi-node harness 在 pod 退出时只记录 phase（Succeeded），不保留容器内 traceback；建议在 pod 失败时抓取最近 N 行日志，便于后续直接定位。

### 5.3 验证计划

| 验证项 | 方法 | 预期 |
|---|---|---|
| 问题 1 能被 extra-c-16 拦截 | 在 #51298 合入后的代码上，手动触发 `extra-c-acc-test-16-npu-a3`（全量） | DSV4 8p DSPARK 用例在 startup warmup 失败，报 `TypeError ... 'softcap'` |
| 问题 1 已被 #1070 修复 | 用包含 #1070 的 sgl-kernel-npu 重跑同一用例 | 服务正常拉起，gsm8k 通过 |
| 问题 2 修复有效 | 用 #53214 的 12 层 Kimi-K3 单机 TP16/EP16 DSPARK 配置跑 capture | target/draft capture 成功，probe 请求通过 |
| PR 流水线可拦截问题 2 | 将上述 12 层 smoke 用例注册进 `extra-b-test-16-npu-a3`，在 #51900 代码上运行 | 用例失败，报 107025 |
| precision 兜底有效 | 修改 `kimi_k3.py` 但不加标签，观察推荐结果 | 不再 `skip_all`，至少跑 smoke 集 |

---

## 6. 附录

### 6.1 关键链接

| 对象 | 链接 |
|---|---|
| 问题 1 根因 PR #51298 | https://github.com/sgl-project/sglang/pull/51298 |
| 问题 1 关联 PR #50186（Inkling-2-Small） | https://github.com/sgl-project/sglang/pull/50186 |
| 问题 1 首个失败 nightly | https://github.com/sgl-project/sglang/actions/runs/44562390109 |
| 问题 1 Kimi-K3 失败 job | https://github.com/sgl-project/sglang/actions/runs/44562390109/job/135859074748 |
| 问题 1 修复 PR（sgl-kernel-npu #1070） | https://github.com/sgl-project/sgl-kernel-npu/pull/1070 |
| 问题 2 根因 PR #51900 | https://github.com/sgl-project/sglang/pull/51900 |
| 问题 2 首个失败 nightly | https://github.com/sgl-project/sglang/actions/runs/44922677219 |
| 问题 2 复现 nightly | https://github.com/sgl-project/sglang/actions/runs/45212673268 |
| 问题 2 PR 验证 run（CANN 9.2） | https://github.com/sgl-project/sglang/actions/runs/45343894905 |
| 问题 2 修复 PR #53214 | https://github.com/sgl-project/sglang/pull/53214 |
| #51900 NPU PR run | https://github.com/sgl-project/sglang/actions/runs/44845789645 |
| #51298 NPU PR run | https://github.com/sgl-project/sglang/actions/runs/44406803384 |
| #50186 NPU PR run | https://github.com/sgl-project/sglang/actions/runs/44504215994 |
| #53214 NPU PR run | https://github.com/sgl-project/sglang/actions/runs/45933003123 |

### 6.2 完整时间线

| 时间（UTC） | 事件 |
|---|---|
| 2026-11-18 18:03 | #51298 创建 |
| 2026-11-19 07:15 | #51298 合入（`21be2189de`） |
| 2026-11-19 15:35 | #50186 合入（`6515ac29d3`） |
| 2026-11-19 16:06 | nightly 44562390109 创建 |
| 2026-11-19 20:54 | Kimi-K3 抛 `TypeError: ... 'softcap'` |
| 2026-11-20 03:50 | sgl-kernel-npu #1070 合入 |
| 2026-11-21 04:13 | #51900 合入（`8396d60a14`） |
| 2026-11-21 16:05 | nightly 44922677219 创建 |
| 2026-11-21 21:11 | Kimi-K3 报 capture_end 107025 |
| 2026-11-23 | PR 验证 run 45343894905 仍复现 |
| 2026-11-26 02:29 | #53214 修复 PR 创建 |
| 2026-11-27 03:14 | #53214 的 NPU PR run：extra-a 通过，extra-b/c 全部 skip |

### 6.3 日志摘录

#### 问题 1（Kimi-K3，run 44562390109 / job 135859074748）

```text
[2026-11-20 04:53:38] Engine startup timings (s): load_weight=1018.46,
  kv_cache_allocation=2.09, scheduler_e2e=1338.04,
  cuda_graph={prefill=0.00, decode=0.00, target_verify=120.60,
  draft_prefill=0.00, draft_decode=5.84, draft_extend=0.00},
  tokenizer_e2e=1358.64
[2026-11-20 04:53:38] INFO: Uvicorn running on http://172.22.3.77:6677
[2026-11-20 04:53:38] INFO: Application startup complete.
[2026-11-20 04:53:39] Using a 448x448 image for Kimi K3 VLM startup warmup ...
[2026-11-20 04:53:40] GET /health HTTP/1.1 503 Service Unavailable
...
[2026-11-20 04:54:09] Scheduler hit an exception: Traceback (most recent call last):
  ...
  File ".../markov_sample.py", line 268, in execute
    return markov_sample_step_triton(**kwargs)
TypeError: markov_sample_step_triton() got an unexpected keyword argument 'softcap'
```

#### 问题 2（Kimi-K3，run 45343894905 / job 138280635172）

```text
[2026-11-24 15:35:27] Capture target verify NPU graph begin.
  backend=full, num_tokens_per_req=8, avail mem=16.93 GB
[2026-11-24 15:36:09] Capturing batches (bs=16 avail_mem=16.85 GB): 0%| | 0/2
[2026-11-24 15:36:10] Scheduler hit an exception:
  File ".../decode_cuda_graph_runner.py", line 688, in __init__
    self.capture()
  ...
  File ".../npu_cudagraph_backend.py", line 449, in _capture_one
    with graph_ctx(...)
  File ".../torch_npu/npu/graphs.py", line 928, in __exit__
    self.npu_graph.capture_end()
RuntimeError: capture_end:../torch_npu/csrc/core/npu/NPUGraph.cpp:371
  NPU function error: endCaptureErr, error code is 107025
[ERROR] Invalid_Argument(EE1017): Capture model validity check failed ...
  the last task (task_id=735) on the sub stream (stream_id=87)
  is not an event record task.
```

### 6.4 用例清单与归属

| 用例 | 注册套件 | 是否在 PR 流水线 | 备注 |
|---|---|---|---|
| `test_npu_deepseek_v4_flash_w8a8_8p_gpqa.py` | `extra-c-acc-test-16-npu-a3` + `full-acc-test-16-npu-a3`（nightly） | 是（extra-c-16） | 唯一覆盖 DSPARK 的 PR 用例；PR 模式跑 gsm8k |
| `test_npu_deepseek_v4_flash_0731_w8a8_1p1d_16p_gpqa.py` | `suite=""`，nightly | 否 | 问题 1 首个失败用例 |
| `test_npu_deepseek_v4_pro_0813_w4a8_1p1d_32p_gpqa.py` | `suite=""`，nightly | 否 | 2 节点 PD |
| `test_npu_kimi_k3_w4a8_32p_gpqa.py` | `suite=""`，nightly | 否 | 4 节点，问题 2 首个失败用例 |
| `test_npu_kimi_k3_w4a8_32p_mix_in128k_out1_prefill.py` | `suite=""`，nightly | 否 | 4 节点 perf |
| `test_npu_kimi_k3_w4a8_32p_mix_in128k_out1k_decode_prefix.py` | `suite=""`，nightly | 否 | 4 节点 perf |
| `test_dspark_npu_dispatch.py` | `extra-b-test-1-npu-a3` | 是 | 不覆盖 `MarkovSampleStep` |
| `test_dspark_markov_sample.py` | CUDA `base-b-kernel-unit` | 否（NPU） | `DEVICE=cuda`，只测 triton/torch |
| `test_kimi_k3_bfa_overlap.py` | CUDA `4-gpu-b200` | 否（NPU） | CUDA only，不覆盖 NPU SBO |
| `test_npu_graph_capture.py` | `extra-b-test-1-npu-a3` + nightly | 是 | 只做最小 NPUGraph capture/replay |

### 6.5 复现/验证命令（参考）

问题 2 的最小复现（来自 Kimi-K3 32P 用例）：

```bash
python3 -m sglang.launch_server \
  --model-path <Kimi-K3-W4A8> \
  --nnodes 4 --tp-size 64 --attn-dp-size 4 \
  --enable-dp-lm-head --enable-shared-experts-attn-tp \
  --moe-a2a-backend deepep --deepep-mode auto \
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path <Kimi-K3-DSpark> \
  --speculative-draft-model-quantization unquant \
  --speculative-dspark-block-size 7 \
  --cuda-graph-bs-decode 1 4 16 \
  --attention-backend ascend --device npu
```

观察 scheduler 日志中的 `Capture target verify NPU graph begin` / `Capturing batches` / `capture_end 107025`。

问题 1 的最小验证：

```bash
# 在 NPU 环境，构造 MarkovSampleStep 输入，调用 execute 并传入 softcap=None
python3 - <<'PY'
import torch
from sglang.kernels.ops.speculative.dspark.markov_sample import MarkovSampleStep
# base_logits/bias/temperatures/offset 按 _check 的约束构造
# MarkovSampleStep.execute(..., softcap=None, vocab_limit=None)
PY
```

修复前应复现 `TypeError: ... 'softcap'`；升级到含 sgl-kernel-npu #1070 的版本后应通过。

---

## 7. 附：给主管的汇报口径（详细版）

- 这两起不是我们用例写错，也不是镜像/CANN 问题，都是上游 PR 在 NVIDIA 验证通过后、NPU 路径被破坏的回归。
- 问题 1 已在 sgl-kernel-npu 侧修复（11-20 合入），当前 nightly 不再报 `softcap`；但修复只是“忽略参数”，NPU 尚未实现 softcap 语义，后续使用该选项的模型有精度风险。
- 问题 2 的修复 PR #53214 已提交，但还没合入；在合入前，Kimi-K3 32P 的 nightly 会持续失败。
- NPU PR 流水线对这三个上游 PR 都是 0 用例执行，且 workflow 显示 success；问题 1 本来有 DSV4 8p DSPARK 用例可以拦截，问题 2 则完全没有 Kimi-K3 用例，属于覆盖缺口。
- 我们已提出三类改进：路径自动触发 NPU CI、补 markov 关键字兼容单测、补 Kimi-K3 单机 capture smoke case；并把 precision 选择器的 `skip_all` 兜底列为 P1。
