# SGLang 两个上游回归：根因与 NPU CI 拦截情况（汇报版）

- 日期：2026-11-27
- 用途：主管向上级做进度汇报（可直接拆成 5 页 PPT）
- 一句话：两起都是上游 PR 的 NPU 路径回归；NPU PR 流水线均未执行用例，问题 1 本可拦截但没跑，问题 2 属于覆盖缺口、跑了也拦不住。

---

## 第 1 页：结论先行

| 问题 | 根因上游 PR | 失败阶段 | 影响 | 状态 |
|---|---|---|---|---|
| `TypeError: markov_sample_step_triton() got an unexpected keyword argument 'softcap'` | **#51298**（从 #50186 Inkling-2-Small 拆出） | 服务启动完成后的 **warmup 首个 decode** | DeepSeek-V4 / Kimi-K3（所有 DSPARK 模型）无法拉起 | sgl-kernel-npu **#1070 已修复**（11-20 合入） |
| Kimi-K3 decode 图捕获 `capture_end` **107025** / `EVENT_WAIT` | **#51900** | Scheduler 初始化阶段的 **decode graph 捕获** | Kimi-K3 32P 三个 nightly 用例连续失败 | sglang **#53214 修复 PR 已提交，未合入** |

**三句话：**

1. 不是用例问题、不是镜像/CANN 问题，是上游 PR 在 NVIDIA 验证通过、NPU 路径被破坏。
2. 三个上游 PR 的 NPU PR 流水线都是 **0 个 NPU 用例执行**，workflow 仍显示 success。
3. 问题 1 有可拦截用例但没跑；问题 2 PR 套件里没有 Kimi-K3 用例，是结构性覆盖缺口。

---

## 第 2 页：问题 1 发生了什么

**上游改了什么**

- #51298 给 DSpark 的 `MarkovSampleStep` 新增 `softcap` / `vocab_limit` 两个可选参数，并在 `_draft_step` 中无条件传入。
- CUDA 的 Triton 实现已同步支持；NPU 的 `sgl_kernel_npu.markov_sample_step_triton` 仍是旧签名。

**错误怎么发生**

```text
DSpark decode -> _draft_step -> MarkovSampleStep.execute
  -> markov_sample_step_triton(**kwargs)
  -> TypeError: unexpected keyword argument 'softcap'
```

- 只要模型用 DSPARK（DeepSeek-V4、Kimi-K3）就会触发。
- GLM 不用 DSPARK，不受影响。

**在哪个阶段**

- 不是模型加载，也不是图捕获；
- 是 HTTP server 已启动、startup warmup 的首个 decode batch；
- 之后 scheduler 退出，`/health` 503，用例健康检查超时。

**修复**

- sgl-kernel-npu #1070（11-20 合入）：wrapper 增加 `**kwargs`，接受并忽略 `softcap` / `vocab_limit`。
- 残留风险：NPU 尚未实现这两个参数的语义，未来使用 `softcap` 的模型可能静默偏差。

---

## 第 3 页：问题 2 发生了什么

**上游改了什么**

- #51900 为优化显存，在 Kimi-K3 共享专家 side-stream 路径新增 `release_shared_output()`：
  `alt_stream.wait_stream(current_stream)`。
- 调用点位于 `KimiK3MoE.forward` 内（L1498 / L1540）。

**错误怎么发生**

- Kimi-K3 32P 开启共享专家 + DeepEP + decode 图；
- 图捕获执行 MoE forward 时，`wait_stream` 在 alt stream 上排入 **EVENT_WAIT**；
- ACL 要求子 stream 最后一个任务是 **event record**，否则 `capture_end` 返回 **107025**。

**为什么 CUDA CI 没发现**

- 该 SBO 分支是 **NPU-only**（`if _is_npu and self._sbo_shared_overlap`）；
- NVIDIA 测试根本不执行这段代码。

**在哪个阶段**

```text
Scheduler.__init__ -> init_model_worker -> init_all_cuda_graphs
  -> capture_decode_graph -> DecodeCudaGraphRunner.capture
  -> capture_end -> 107025
```

- server 尚未启动，外部表现为 `/health` 3600s 超时。

**修复**

- #53214：NPU 上改用 `record_stream`，避免尾部 side-stream wait；单机 12 层 Kimi-K3、TP16/EP16、DSPARK 已验证。
- 状态：open，未合入。

---

## 第 4 页：NPU PR 流水线有没有拦截？

| PR | NPU run 结论 | NPU 用例执行 | 原因 |
|---|---|---|---|
| #51298（问题 1 根因） | success | **0 个** | 缺 `run-ci-npu-extra` 标签 |
| #50186（Inkling-2-Small） | success | **0 个** | 缺 `run-ci-npu-extra` 标签 |
| #51900（问题 2 根因） | success | **0 个** | 标签为空，缺 `run-ci` + `run-ci-npu-extra` |
| #53214（问题 2 修复） | success | 只有 CPU 单测 | 标签齐全，但 precision 推荐 0 个用例 -> `skip_all` |

- 前三个 run 成功的 4 个 job 是 `check-changes`、`pr-gate`、`set-image-config`、`Analyze failure report`，没有任何真实 NPU 用例。
- **false green**：0 个 NPU 用例执行，workflow 仍显示 success。

**为什么跳过**

1. 标签门禁：NPU 测试需要 `run-ci` + `run-ci-npu-extra`，三个 PR 都没带。
2. precision 选择器：即使带标签，`kimi_k3.py` 等模型代码推荐为 0 时会 `skip_all`；全量运行只对 3 个 CI 配置文件生效。

---

## 第 5 页：如果用例都运行，能不能拦截？

| 问题 | 套件里有没有对应用例 | 全量运行能否拦截 | 结论 |
|---|---|---|---|
| 问题 1（softcap） | 有：`extra-c-acc-test-16-npu-a3` 的 DeepSeek-V4-Flash W8A8 8p **DSPARK** 用例 | **能**（启动 warmup 就会报 TypeError） | 有能力，但没跑；precision 还可能漏检 |
| 问题 2（107025） | 无：Kimi-K3 32P 只在 nightly 多机；Kimi-K3 单测是 CUDA-only | **不能** | 结构性覆盖缺口 |

**关键判断**

- 问题 1 是“没跑”：PR 套件有拦截能力，被标签门禁挡住。
- 问题 2 是“跑不到”：PR 套件最大单机 16 卡，Kimi-K3 32P 是 4 节点，只有 nightly 才跑。
- 因此责任要分开：问题 1 主要是门禁未触发；问题 2 我们 CI 也有覆盖缺口。

---

## 第 6 页：下一步行动（建议）

**P0（本周）**

1. NPU 敏感路径（`kimi_k3.py`、`dspark/**`、`hardware_backend/npu/**`）自动触发 NPU CI 或自动加标签。
2. 在 `extra-b-test-1-npu-a3` 增加 `MarkovSampleStep` 关键字兼容单测（约 90s，可直接拦截问题 1）。

**P1（两周内）**

3. 增加 NPU 图捕获 side-stream 单测，或把 #53214 的 12 层 Kimi-K3 单机 TP16/EP16 DSPARK 配置固化为 PR smoke 用例（拦截问题 2）。
4. precision 推荐为 0 时不要 `skip_all`，至少跑一个固定 smoke 集。

**P2（流程）**

5. 0 个 NPU 用例执行时，check 返回 neutral/失败，而不是 success。
6. 增加 sglang 与 sgl-kernel-npu 的算子签名契约测试；推动 NPU 实现 `softcap` / `vocab_limit` 语义。

---

## 附：关键链接

- 问题 1 根因 PR：https://github.com/sgl-project/sglang/pull/51298
- 问题 1 关联 PR（Inkling-2-Small）：https://github.com/sgl-project/sglang/pull/50186
- 问题 1 修复 PR（sgl-kernel-npu）：https://github.com/sgl-project/sgl-kernel-npu/pull/1070
- 问题 1 首个失败 nightly：https://github.com/sgl-project/sglang/actions/runs/44562390109
- 问题 2 根因 PR：https://github.com/sgl-project/sglang/pull/51900
- 问题 2 首个失败 nightly：https://github.com/sgl-project/sglang/actions/runs/44922677219
- 问题 2 修复 PR：https://github.com/sgl-project/sglang/pull/53214
- #51900 的 NPU PR run（0 用例）：https://github.com/sgl-project/sglang/actions/runs/44845789645
