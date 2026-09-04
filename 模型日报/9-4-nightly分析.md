# Nightly NPU 分析（9-4）

## 一、本次与历史 run 基础数据
- 本次：`runs/33777992960`（run 1502）start=09-03 16:20:07Z，仍在进行中
- 前一次：`runs/33654410404`（1492）09-02 16:21:31Z → 09-03 11:51:57Z，约 19.5h
- 前两次：`runs/33533096253`（1487）09-01 16:39:31Z → 09-02 09:13:22Z，约 16.6h
- 本次 A3 使用默认镜像 main-cann9.0.0-a3（schedule 触发，无入参）

## 二、用例运行顺序与排队位置（依据 `nightly-test-npu.yml`）
1. `set-image-config`
2. 并行 A3 单机 suites（`needs: [set-image-config]`）：`nightly-1/2/4/8/16-npu-a3`、`nightly-perf-2/4/8/16-npu-a3`、`nightly-acc-2/4/16-npu-a3`
3. `nightly-poc-multi-node-tests`（`needs` 上述全部 perf-2/4/16 + acc-2/4/16；`max-parallel=1` 串行 5 例）：glm5_1_50ms(perf) → glm5_1_aime26(acc) → mimo ttft → mimo tpot → deepseek_v4_flash_1p1d_16p(perf)
4. `nightly-poc-multi-node-mix-tests`（依赖上一步，串行 4 例）：kimi_k2_6(perf) → kimi_k2_6_aime25(acc) → glm_5_2_w4a8_16p_gpqa(acc) → kimi_k3_w4a8_32p_gpqa(acc)
5. `check-all-jobs`

关注模型排队位置：
- DSV4-Flash 8p_in8k / 8p_in32k（性能）→ `nightly-perf-16-npu-a3`（单机并行，靠前）
- DSV4-Flash 8p_gpqa（精度）→ `nightly-acc-16-npu-a3`（单机并行，靠前）
- DSV4-Flash 1p1d_16p（性能，PD分离多节点）→ multi-node-tests 第 5 例（串行块末位）
- GLM5.2 → multi-node-mix 第 3 例（倒数第 2）
- Kimi-K3 → multi-node-mix 第 4 例（整条流水线绝对最后）

## 三、剩余时间估算
截至快照（约 09-04 10:33Z）GLM5.2 刚开跑，其后仅剩 GLM5.2 与 Kimi-K3 两例，历史单例耗时稳定：GLM5.2 ≈ 194~203min、Kimi-K3 ≈ 181~207min。
→ 还需约 6.5~7 小时，预计 09-04 17:00~17:30Z 完成（北京时间 09-05 01:00~01:30）。本次总时长预计约 25h（明显长于前两次 16.6h / 19.5h，见下）。

## 四、运行时间过长的原因
1. 多节点用例严格串行（`max-parallel=1`）且被安排在流水线末尾，同时 multi-node-tests 依赖全部 6 个单机 perf/acc job 完成后才开工，关键路径长。
2. 单机阶段最长板 `nightly-acc-2-npu-a3`（4 分片）各分片跑满约 5h（300min 超时），是 multi-node 开工的 gate；本次 acc-2 最晚 02:36 才结束。
3. A3 自托管 runner 排队：A2 job（16:21）立即执行，但所有 A3 job 直到约 21:15 才开始（约 5h 等待 A3 runner 空闲），三次 run 均存在此固定等待。
4. 本次 multi-node-poc 用例真实跑通（glm5_1 perf 182min、aime26 109min、mimo 28/43min）而非前两次 2~3min 快速失败，比前两次多出约 6h。

## 五、用例在代码仓中的精确位置（均已核实存在）
- `test/registered/npu/performance/deepseek_v4_flash/test_npu_deepseek_v4_flash_w8a8_1p1d_16p_in8k_out1k_50ms.py` — `register_npu_ci(est_time=3600, suite="", nightly=True, disabled="performance testcase")`，多节点 PD 分离
- `test/registered/npu/performance/deepseek_v4_flash/test_npu_deepseek_v4_flash_w8a8_8p_in32k_out1k_50ms.py` — `register_npu_ci(est_time=1800, suite="nightly-perf-16-npu-a3", nightly=True)`
- `test/registered/npu/performance/deepseek_v4_flash/test_npu_deepseek_v4_flash_w8a8_8p_in8k_out1k_50ms.py` — `register_npu_ci(est_time=1800, suite="nightly-perf-16-npu-a3", nightly=True)`
- `test/registered/npu/accuracy/deepseek_v4_flash/test_npu_deepseek_v4_flash_w8a8_8p_gpqa.py` — `register_npu_ci(est_time=3600, suite="nightly-acc-16-npu-a3", nightly=True)`；accuracy=0.874
- `test/registered/npu/accuracy/kimi_k3/test_npu_kimi_k3_w4a8_32p_gpqa.py` — `register_npu_ci(est_time=3600, suite="", nightly=True, disabled="accuracy testcase")`；accuracy=0.935，4 节点
- `test/registered/npu/accuracy/glm5_2/test_npu_glm_5_2_w4a8_16p_gpqa.py` — `register_npu_ci(est_time=3600, suite="", nightly=True, disabled="accuracy testcase")`；accuracy=0.912，2 节点

用例到 suite 的映射：单机 job 通过 `test/run_suite.py --suite` + `register_npu_ci(suite=...)` 发现；kimi_k3 / glm5_2 因 `suite=""` 且 `disabled`，交由 nightly-test-npu.yml 中 multi-node-mix 矩阵显式列出（脚本文件级执行）。

## 六、本次失败原因（与昨日对比）
- `nightly-perf-16-npu-a3`（8p 两例）：`Fatal Python error: Segmentation fault`，栈命中 `aclnnQuantLightningIndexerGetWorkspaceSize` —— 即 npu_quant_lightning_indexer 段错误，与 9/1 一致（zbal 问题）。
- `nightly-acc-16-npu-a3`（8p_gpqa）：同为 `aclnnQuantLightningIndexerGetWorkspaceSize` 段错误，与 9/2 一致。
- multi-node deepseek_v4_flash_1p1d_16p：`run_npu_e2e_test.py` 中 `monitor_pod_logs` 抛出 "Monitoring ended but target pattern was not detected"（server 未就绪），与 9/1 server 启动即崩溃同一表现。
- glm_5_2 / kimi_k3：截至快照尚未出结果（进行中 / 排队中）。