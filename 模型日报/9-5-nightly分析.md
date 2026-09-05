# Nightly NPU 分析（9-5）

## 一、本次 run 基础数据
- 本次：`runs/33894505909`，A3 使用默认镜像 `main-cann9.0.0-a3`（schedule 触发，无入参）。
- 对比：`runs/33777992960`（9/4 日报，run 1502）。

## 二、本次结果汇总
- kimi_k3：执行通过，实测精度 0.9293（低于基线 0.935）。较 9/4 的 0.9444 有所下降，仍在基线附近波动。
- glm_5_2：评测已完成（精度 0.5657，远低于基线 0.912），但用例最终报 "Monitoring ended but target pattern was not detected"（monitor_pod_logs 未检测到目标 pattern），标记失败。精度与 9/2（0.5303）、8/27（0.4949）同量级，精度问题未修复。
- DSV4-Flash 性能 8p_in32k / 8p_in8k：段错误（SIGSEGV）`aclnnQuantLightningIndexerGetWorkspaceSize`，即 npu_quant_lightning_indexer 段错误，与 9/4 同源（zbal 问题），未修复。
- DSV4-Flash 精度 8p_gpqa：server 启动时段错误（SIGSEGV）——CUDA graph 捕获阶段调用 DSV4 indexer 自定义算子（`ascend_dsv4_backend.py _forward_npu_fused → forward_c4_indexer`）时崩溃，server 进程退出（code -9），setUpClass 失败。与 9/4 的 indexer/量化段错误同一类问题。
- DSV4-Flash 性能 1p1d_16p（多节点 PD 分离）：decode 侧 pod 提前退出（status: Succeeded），monitor_pod_logs 未检测到就绪 pattern，server 未就绪，与 9/4 表现一致。

## 三、结论
- DSV4-Flash 的段错误（zbal / npu_quant_lightning_indexer）连续多日未修复，性能与精度用例均持续失败，需跟进开发定位 zbal。
- GLM5.2 精度长期远低于基线（0.49~0.57 vs 0.912），且本次监控逻辑判失败，精度本身未见改善。
- Kimi-K3 精度在基线附近波动（9/2 0.9192、9/4 0.9444、9/5 0.9293），需关注是否稳定达标 0.935。

## 四、关键用例耗时拆分（GLM5.2 / Kimi-K3，时间均为 UTC，北京 = UTC+8）

两个用例均在 `multi-node-mix-poc` 串行块的第 3、4 例，紧跟前一个 mix 用例结束即开跑，排队等待几乎为 0。

| 用例 | job 总时长 | 排队(创建→开跑) | Run test 步骤 | 有无重试 | 主耗时 |
| --- | --- | --- | --- | --- | --- |
| GLM5.2 | 3h02m42s | 36s | 2h57m28s | 有 1 次重试 | 评测重试拉长近一倍 |
| Kimi-K3 | 4h11m53s | 38s | 4h05m58s | 无 | pod 等 32 卡资源 |

### GLM5.2（2 节点 = 16 卡）
- pod 调度 Pending：04:13:49 → 04:14:34（45s，16 卡几乎无等待）
- server 启动 + 模型加载：04:15:01 → 04:20:37（5m36s）
- 评测第 1 轮：04:24:44 → 05:57:18（92m34s），精度 0.5657
- 重试第 2 轮：05:57:20 → 07:11:10（73m50s），报 "Monitoring ended but target pattern was not detected" 失败

关键：`05:57:18` 打印 `Accuracy 0.5657 below threshold 0.8867..., retrying (1/2)`，因精度低于阈值触发重跑评测，又跑约 74 分钟才失败。若无此次重试，约 105 分钟即可结束；重试使评测从 93 分钟接近翻倍到 166 分钟（92+74），是本次耗时偏高的直接原因。

### Kimi-K3（4 节点 = 32 卡）
- pod 调度 Pending：07:17:01 → 09:40:47（2h23m46s，等 32 卡资源，主耗时）
- server 启动 + 模型加载：09:40:47 → 09:59:00（18m13s）
- 评测（gpqa_diamond 198 题）：10:02:40 → 11:22:37（79m57s），精度 0.9293

`run_attempt=1`、`SGLANG_TEST_MAX_RETRY=0`，无 `retrying`，单轮跑完、无重试。主耗时是 pod 处于 `Pending` 约 144 分钟（等共享 NPU 资源池调度 4 节点 32 卡，不产生有效工作），真正评测仅约 80 分钟。