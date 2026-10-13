# test_npu_vlm_input_format.py CI 失败排查报告

> 更新时间：2026-10-13
> 用例路径：`test/registered/npu/vlm_models/test_npu_vlm_input_format.py`（suite=full-2-npu-a3，est_time=747）
> 关联 PR：#1234（全量测试）、#1327（A/B 验证）

---

## 1. 用例与失败现象

用例验证引擎对**多图输入**的三种处理路径（每个模型 3 个测试 × 3 个模型 = 9 个测试）：
Gemma-3-4B → Kimi-VL-A3B → Qwen2.5-VL-3B（按字母序执行）。

- 测试请求发送 **2 张本地图片**（图1=出租车街景，图2=SGL logo）+ prompt
  `"Describe both the first image and the second image in detail separately."`
- 失败点：`TestQwenVLUnderstandsImage.test_accepts_image` → `verify_response` 第 133 行
  `assert has_sg_or_logo_side`。模型输出**只详细描述了图1（出租车），随后自然 EOS（约 150/512 tokens）**，
  未提及图2。Gemma / Kimi 的 6 个测试在所有环境中均通过。
- 失败输出（B080 与 B100 上**逐字节一致**，597 字符）：
  > `**first image:** the image depicts a scene on a city street with two yellow taxis. the taxi on the left is partially visible, ...`

补充：CI 全量跑该文件时带 `-f`（failfast），Qwen 类按字母序最后执行，失败后剩余 2 个测试不跑，
故全量日志显示 `Ran 7 tests`；singletest 无 `-f`，`Ran 9 tests`。failfast 只是截断现象，不是失败原因。

## 2. 全部运行记录

| # | 日期 | 环境/镜像 | 通道 | 结果 | 关键观察 | 链接 |
|---|------|----------|------|------|---------|------|
| 1 | 09-09 | B060 (cann9.1.0-a3-20260904) | full-test | ⚠️ run 失败 | partition 因 **2 个 LoRA 用例**失败（check-run 注解可证）；**vlm 自身结果日志已被 GitHub 删除，不可考** | [run 34355596030](https://github.com/Ascend/sglang/actions/runs/34355596030) / [job 102481168489](https://github.com/Ascend/sglang/actions/jobs/102481168489) |
| 2 | 09-16 | B060 | singletest | ✅ PASS | `Ran 9 tests` 全过；watchdog×2 | [run 35064716320](https://github.com/Ascend/sglang/actions/runs/35064716320/job/104692354350)（PR #1245） |
| 3 | 09-20 | B070 | full-test | ❌ FAIL | 失败文本与后续逐字节一致 | [run 35516918693](https://github.com/Ascend/sglang/actions/runs/35516918693) |
| 4 | 09-24 | B071 | full-test | ❌ FAIL | 同上 | [run 35998267141](https://github.com/Ascend/sglang/actions/runs/35998267141) |
| 5 | 09-28 | B072 | full-test | ❌ FAIL | 同上 | [run 36457035749](https://github.com/Ascend/sglang/actions/runs/36457035749) |
| 6 | 09-29 | B080 | full-test | ❌ FAIL | `Ran 7 tests`（failfast 截断） | [run 36589487654](https://github.com/Ascend/sglang/actions/runs/36589487654/job/109478585845) |
| 7 | 09-30 | **B080（本地服务器）** | 手动单跑 | ✅ PASS | `Ran 9 tests in 868s OK`；watchdog×4；本地 CANN=标准版 cann-9.1.0 | 本地日志 `vlm_input_format_20260930_172528.log` |
| 8 | 10-11 | B100 | full-test | ❌ FAIL | 与 B080 失败文本逐字节一致 | [run 38133839963](https://github.com/Ascend/sglang/actions/runs/38133839963/job/114453527362)（PR #1234） |
| 9 | 10-13 | B100 | singletest | ❌ FAIL | **9/9 全跑，唯一失败=Qwen test_accepts_image**；watchdog×3；文本逐字节一致 | [run 38341968133](https://github.com/Ascend/sglang/actions/runs/38341968133/job/115114803146)（PR #1327） |
| 10 | 10-13 | B100 | full-2（仅保留 full-2 的裁剪版全量） | 见 PR | 用于 #9 的同镜像 A/B 对照 | [run 38341968416](https://github.com/Ascend/sglang/actions/runs/38341968416)（PR #1327） |

> 行 1 说明：run 34355596030 的 job 日志 blob 已被 GitHub 回收（BlobNotFound），vlm 的结果条目在已删除部分。
> "B060 全量失败"属实，但那是 **LoRA 用例**导致的失败，**不等于 vlm 失败**。此数据点请开发侧帮忙确认。

## 3. 已排除的因素（均有日志证据）

| 候选原因 | 证据 | 结论 |
|---------|------|------|
| full-test 独有环境变量（`PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` / `STREAMS_PER_DEVICE=32`） | B100 **singletest 同样失败**（干净环境，仅 `SGLANG_SET_CPU_AFFINITY=1`） | ❌ 排除（但见 §6 双因子讨论） |
| 测试文件被改动 | testcases-260813 上该文件最后修改 **2026-08-14**（commit 4fdf70f9b5），B060 通过与 B100 失败用的是同一份 | ❌ 排除 |
| 引擎启动参数差异 | 四路日志 server_args 全量（516 键）逐键 diff，mm/图像相关键（`mm_io_worker_num`、`image_processor_backend`、`disable_fast_image_processor` 等）**完全一致** | ❌ 排除 |
| qwen2_5_vl 模型代码变更 | 本地/B060/B100 三份日志在 `qwen2_5_vl.py:249` 打出同一告警，行号一致 | ❌ 基本排除（文件版本一致） |
| transformers `image_token` v5.11 语义切换 | 通过（B060）与失败（B100）日志打印**同一条**弃用警告 → 运行时行为一致 | ❌ 排除 |
| full-test pip 装包改了图像链路 | `Successfully installed` 全清单核查：numpy/Pillow/torchvision/transformers 均未动 | ❌ 排除 |
| watchdog 300s 超时截断输出 | 见 §5 | ❌ 排除 |
| failfast 截断 | 只是少跑了 2 个测试 | ❌ 排除 |

## 4. 关键事实矩阵

| 环境 | B060 | B070 | B071 | B072 | B080 | B100 |
|------|------|------|------|------|------|------|
| CI singletest | ✅ (09-16) | — | — | — | 未跑 | ❌ (10-13) |
| CI full-test | ⚠️ 不可考 (09-09) | ❌ | ❌ | ❌ | ❌ | ❌ |
| 本地（用户服务器） | — | — | — | — | ✅ (09-30) | 未跑 |

三个确定事实：
1. **本地 B080 通过** → B080 镜像的整套栈（sglang 快照 + CANN a3 + torch_npu）在 CI 之外可以正确处理双图输入。
2. **CI singletest：B060 过、B100 挂**（同一测试文件、同一 harness）→ 09-16 → 10-13 之间某个东西变了。
3. **CI full-test：B070→B100 全部失败且文本逐字节一致** → 确定性的输入处理分歧，非 flaky。

## 5. "默认超时逻辑导致失败"假说的验证

针对"某种默认超时（如 watchdog_timeout=300）导致失败"的猜想，逐项核实：

1. **TokenizerManager watchdog（soft=True, 300s）**：只打印 py-spy 栈 + 提示，**不 kill 任何东西**。
   且三边都出现：本地 PASS×4、B060 PASS×2、B100 FAIL×3 → 与成败无相关性。
   栈内容均为引擎生命周期等待（triton 编译、`_wait_for_scheduler_ready`、idle 的 mm-io worker），
   属于引擎启动/停止慢的噪音。
2. **Scheduler watchdog（会 kill 卡死的 forward batch）**：未触发（若触发会进程崩溃，测试直接 error 而非断言失败）。
3. **失败请求内部无 300s 级停滞**：B100 singletest 中 Qwen 三个测试的引擎启动时刻为
   03:42:05 / 03:42:44 / 03:43:35，03:43:57 出 FAIL → 每个 Qwen 测试仅约 **40 秒**，远小于任何默认超时。
4. **超时截断与"逐字节一致"矛盾**：若是超时打断生成，输出长度随当时速度漂移，不可能在
   B080/B100、full/singletest 四次失败中产出 597 字符完全相同的文本；且失败输出以**自然 EOS** 结束
   （模型自认"说完了"），是典型的"没看见第二张图"而非"说到一半被掐断"。

**结论：现有证据不支持"默认超时导致失败"。** 超时/阻塞现象与失败共现，但更可能是同一底层问题
（多图路径劣化）的伴生症状。建议复跑时给 Engine 加 `log_level="info"` 抓 `Prefill batch. #new-token`
与 mm 处理日志后复查一次。

## 6. 当前最优解释

```
 singletest       full-test
B060   ✅  →  ⚠️不可考
B070+  ?      ❌ (逐字节一致)
B100   ❌  →  ❌
本地B080 ✅     —
```

- **事实 2**（B060 singletest ✅ / B100 singletest ❌）要求 09-16→10-13 之间存在一个变化 S：
  - 候选 S1：B080→B100 镜像栈变化（bundled sglang 快照 / CANN a3 微构建 / torch_npu 微版本）
  - 候选 S2：CI a3-2 运行池自身变化（驱动/NPU 固件升级）——需运维确认 9-16~10-13 间是否有变更
- **事实 1**（本地 B080 ✅）说明 B080 镜像栈本身不坏，因此 B070~B080 的全量失败若非 S1 造成，
  就必须存在 full-test 特有因素 F（环境变量/资源竞争等）——但 F 已被"环境变量排除"证据削弱，
  剩下的是 full-test harness 特有但 singletest 没有的东西（待查：分片顺序导致的内存/残留状态等）。
- **注意**：本地 `/sgl-workspace/sglang` 的版本与来源尚未确认（是 B080 镜像自带，还是自建/软链）。
  若是自建，则"sglang 快照"仍未洗清；若确为 B080 镜像自带，则 S1 收窄为 CANN/torch_npu 微构建。

## 7. 决定性实验（按性价比排序）

| # | 实验 | 成本 | 结果判读 |
|---|------|------|---------|
| E1 | 把 PR #1327 的 singletest 镜像改成 **B080** 跑一次（改一行镜像 tag） | 一次 CI | ❌失败 → CI singletest 环境在 9-16 后变化（S2/框架论成立，镜像洗清）；✅通过 → B080→B100 镜像回归（S1） |
| E2 | 本地拉 **B100 镜像**跑同用例（默认环境） | 一次本地 | ❌失败 → B100 镜像内容回归实锤；✅通过 → CI 环境因素实锤 |
| E3 | 本地 B100 镜像 + `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True STREAMS_PER_DEVICE=32` | 一次本地 | 复现 → env 因素；不复现 → 排除 env |
| E4 | 任意复跑加 `log_level="info"`，抓 Qwen 请求 `Prefill batch. #new-token` | 顺带 | 两图 prompt 的 token 数在 B100 上若明显小于本地/B060 → image2 token 丢失实锤，排查范围缩到多图占位符展开/mrope |

E1 + E2 交叉即可把 S1/S2 二选一。

## 8. 附：PR #3（原 #1298）对 tokenizer 用例的改动解读

### 旧版（当前 patch-14-validate 内，失败版本）
```python
# setUpClass 起一个 带 --enable-tokenizer-batch-encode 的服务
def test_tokenizer_batch_encode_throughput_improvement(self):
    self.run_decode(16)                      # 预热
    max_tokens = 256
    res = self.run_decode(max_tokens)        # "The capital of France is", ignore_eos=True
    throughput = max_tokens / (tok - tic)
    self.assertGreaterEqual(throughput, 200 * 0.8)   # >= 160 tok/s
```
- **问题 1**：测的是"解码阶段吞吐 ≥160 tok/s"。`--enable-tokenizer-batch-encode` 只影响
  **输入 tokenization（prefill 编码）**，与解码速度无关——断言和参数完全脱钩。
- **问题 2**：绝对阈值 160 对硬件/镜像/负载敏感。B100 上实测两次 91.19、139.73 → 必挂。
- `test_gsm8k`（200 例，≥0.38）：与参数无强关联，但保留作正确性回归。

### 新版（#1298 重写，PR #3 携带）
```python
# 不起服务，直接拿 tokenizer 对比两种编码方式
def test_tokenizer_batch_encode_throughput_improvement(self):
    # 10 个请求 × 每请求 8 条 prompt × 每条 32000 随机 token（random.seed(0) 固定，
    # 规模对齐社区 v0.5.5 benchmark/benchmark_batch/benchmark_batch.py）
    off_avg = benchmark(逐条 encode)   # flag-off 的行为
    on_avg  = benchmark(批量 encode)   # tokenizer(texts) 一次编 8 条 = flag-on 的行为
    self.assertLess(on_avg, off_avg)   # 相对断言：批量必须更快
```
- **断言改为相对比较**（同一进程、同一批数据下 batch < sequential），
  不再依赖机器绝对速度 → 在任何硬件上语义稳定。
- **规模放大**（8×32k tokens/请求）：小 prompt 下批量优势不明显，大 prompt 才能测出
  `--enable-tokenizer-batch-encode` 的收益，同时与社区基准对齐。
- `test_gsm8k`：服务端启动移入测试方法（flag 开启），分数阈值 0.38 不变；
  `est_time` 400→900（新增的 80 条 32k-token prompt 编码 + 服务启动耗时）。
- 工程细节：`--dp 2`（a3-2 只可见 2 卡）、`--max-prefill-tokens 131072`
  （8×32k=256k 需要 chunked prefill 吞下）、`terminate_and_kill_process_tree` 替换旧的 kill。

**给开发的两个确认点**：① 旧断言 160 tok/s 是否本来就是误用（测了与参数无关的解码吞吐）？
② 新版 gsm8k 分数 0.38 在各镜像上的基线是否稳定。

## 9. 待开发/运维确认清单

1. 本地 09-30 那次的 `/sgl-workspace/sglang` 来源与版本（B080 镜像自带？自建？软链？）——决定 sglang 快照是否洗清
2. 09-09 B060 全量中 **vlm 的真实结果**（日志已删；如内部有留档请确认）——决定单因子/双因子
3. CI a3-2 运行池在 **09-16 → 09-20** 之间是否有驱动/固件/镜像缓存变更（候选 S2）
4. B080 → B100 两个镜像的构建 changelog（sglang commit 范围、CANN/torch_npu 是否 bump）（候选 S1）
5. 是否允许在 vlm 用例中增加输入层断言（prompt token 数 / 每图展开 token id dump）以便未来快速定位
