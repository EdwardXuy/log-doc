# sgl-project/sgl-kernel-npu CI/CD 流水线分析报告

**生成时间**: 2026-06-01 19:56:17
**仓库**: sgl-project/sgl-kernel-npu
**分析范围**: 所有已完成的运行（成功 + 失败）
**数据量**: 20 次运行, 72 个任务

---

## 1. 概览

### daily-build-test

| 指标 | 数值 |
|------|------|
| 总运行次数 | 10 |
| 成功 / 失败 | 10 / 0 |
| 运行成功率 | 100.0% |
| 总任务数 | 30 |
| 成功 / 失败 / 取消 / 跳过 | 30 / 0 / 0 / 0 |
| 任务通过率（不含取消/跳过） | 100.0% |

### pr-test-npu

| 指标 | 数值 |
|------|------|
| 总运行次数 | 10 |
| 成功 / 失败 | 4 / 6 |
| 运行成功率 | 40.0% |
| 总任务数 | 42 |
| 成功 / 失败 / 取消 / 跳过 | 29 / 10 / 0 / 3 |
| 任务通过率（不含取消/跳过） | 74.4% |

### 总计

| 指标 | 数值 |
|------|------|
| 总运行次数 | 20 |
| 成功 / 失败 | 14 / 6 |
| 运行成功率 | 70.0% |
| 总任务数 | 72 |
| 成功 / 失败 / 取消 / 跳过 | 59 / 10 / 0 / 3 |
| 任务通过率（不含取消/跳过） | 85.5% |

---

## 2. 每次运行的任务统计（最近20次）

### daily-build-test

| 运行 | 标题 | 结论 | 总任务 | 成功 | 失败 | 取消 | 通过率 |
|------|------|------|--------|------|------|------|--------|
| [26717965554](https://github.com/sgl-project/sgl-kernel-npu/actions/runs/26717965554) | ci: restore build cache for PR CI, with ... | 成功 | 3 | 3 | 0 | 0 | 100.0% |
| [26688781007](https://github.com/sgl-project/sgl-kernel-npu/actions/runs/26688781007) | ci: restore build cache for PR CI, with ... | 成功 | 3 | 3 | 0 | 0 | 100.0% |
| [26650989313](https://github.com/sgl-project/sgl-kernel-npu/actions/runs/26650989313) | added new commands to README for DeepEP ... | 成功 | 3 | 3 | 0 | 0 | 100.0% |
| [26590057200](https://github.com/sgl-project/sgl-kernel-npu/actions/runs/26590057200) | added new commands to README for DeepEP ... | 成功 | 3 | 3 | 0 | 0 | 100.0% |
| [26526055603](https://github.com/sgl-project/sgl-kernel-npu/actions/runs/26526055603) | refactor: separate submodule init and bu... | 成功 | 3 | 3 | 0 | 0 | 100.0% |
| [26462975507](https://github.com/sgl-project/sgl-kernel-npu/actions/runs/26462975507) | fix insert slice (#511) | 成功 | 3 | 3 | 0 | 0 | 100.0% |
| [26410672184](https://github.com/sgl-project/sgl-kernel-npu/actions/runs/26410672184) | fix insert slice (#511) | 成功 | 3 | 3 | 0 | 0 | 100.0% |
| [26366390149](https://github.com/sgl-project/sgl-kernel-npu/actions/runs/26366390149) | fix insert slice (#511) | 成功 | 3 | 3 | 0 | 0 | 100.0% |
| [26337599333](https://github.com/sgl-project/sgl-kernel-npu/actions/runs/26337599333) | fix insert slice (#511) | 成功 | 3 | 3 | 0 | 0 | 100.0% |
| [26300235624](https://github.com/sgl-project/sgl-kernel-npu/actions/runs/26300235624) | fix insert slice (#511) | 成功 | 3 | 3 | 0 | 0 | 100.0% |

### pr-test-npu

| 运行 | 标题 | 结论 | 总任务 | 成功 | 失败 | 取消 | 通过率 |
|------|------|------|--------|------|------|------|--------|
| [26748663367](https://github.com/sgl-project/sgl-kernel-npu/actions/runs/26748663367) | feat: support H=12,8 | 成功 | 6 | 3 | 0 | 0 | 50.0% |
| [26745218348](https://github.com/sgl-project/sgl-kernel-npu/actions/runs/26745218348) | fix review comments | 失败 | 6 | 2 | 4 | 0 | 33.3% |
| [26738525734](https://github.com/sgl-project/sgl-kernel-npu/actions/runs/26738525734) | Update pr-test-npu.yml | 成功 | 6 | 6 | 0 | 0 | 100.0% |
| [26734123735](https://github.com/sgl-project/sgl-kernel-npu/actions/runs/26734123735) | Update pr-test-npu.yml | 失败 | 6 | 3 | 3 | 0 | 50.0% |
| [26733952665](https://github.com/sgl-project/sgl-kernel-npu/actions/runs/26733952665) | Add HCCL_BUFFSIZE environment variable f... | 失败 | 0 | 0 | 0 | 0 | 0% |
| [26679309674](https://github.com/sgl-project/sgl-kernel-npu/actions/runs/26679309674) | Merge branch 'main' into test-pr | 失败 | 0 | 0 | 0 | 0 | 0% |
| [26678958665](https://github.com/sgl-project/sgl-kernel-npu/actions/runs/26678958665) | start test | 失败 | 0 | 0 | 0 | 0 | 0% |
| [26678533713](https://github.com/sgl-project/sgl-kernel-npu/actions/runs/26678533713) | Modification according to the review com... | 失败 | 6 | 3 | 3 | 0 | 50.0% |
| [26673018437](https://github.com/sgl-project/sgl-kernel-npu/actions/runs/26673018437) | ci: fix missing cache key and remove dup... | 成功 | 6 | 6 | 0 | 0 | 100.0% |
| [26671232298](https://github.com/sgl-project/sgl-kernel-npu/actions/runs/26671232298) | ci: fix YAML indentation in workflow fil... | 成功 | 6 | 6 | 0 | 0 | 100.0% |

---

## 3. 稳定性分析

### pr-test-npu 任务稳定性

| 任务名称 | 总次数 | 成功 | 失败 | 通过率 |
|----------|--------|------|------|--------|
| Check changed files | 7 | 7 | 0 | 100.0% |
| multi-node (test_internode_a2, 2, tests/python/deepep/test_internode_a2.py) / test_internode_a2 | 7 | 7 | 0 | 100.0% |
| test-build-deepep-a2 | 7 | 5 | 1 | 71.4% |
| test-all-build | 7 | 3 | 3 | 42.9% |
| test-build-deepep-a3 | 7 | 3 | 3 | 42.9% |
| finish | 7 | 4 | 3 | 57.1% |

### daily-build-test 任务稳定性

| 任务名称 | 总次数 | 成功 | 失败 | 通过率 |
|----------|--------|------|------|--------|
| daily-enumerate-intranode | 10 | 10 | 0 | 100.0% |
| daily-enumerate-low-latency | 10 | 10 | 0 | 100.0% |
| finish | 10 | 10 | 0 | 100.0% |

### PR 工作流端到端时长（目标: <= 60 分钟）

| 运行 | 标题 | 结论 | 核心任务最大时长 | 是否达标 |
|------|------|------|------------------|----------|
| [26748663367](https://github.com/sgl-project/sgl-kernel-npu/actions/runs/26748663367) | feat: support H=12,8 | success | 27.1 分钟 | 是 |
| [26745218348](https://github.com/sgl-project/sgl-kernel-npu/actions/runs/26745218348) | fix review comments | failure | 100.2 分钟 | 否 |
| [26738525734](https://github.com/sgl-project/sgl-kernel-npu/actions/runs/26738525734) | Update pr-test-npu.yml | success | 18.6 分钟 | 是 |
| [26734123735](https://github.com/sgl-project/sgl-kernel-npu/actions/runs/26734123735) | Update pr-test-npu.yml | failure | 94.2 分钟 | 否 |
| [26678533713](https://github.com/sgl-project/sgl-kernel-npu/actions/runs/26678533713) | Modification according to the ... | failure | 38.9 分钟 | 是 |
| [26673018437](https://github.com/sgl-project/sgl-kernel-npu/actions/runs/26673018437) | ci: fix missing cache key and ... | success | 67.3 分钟 | 否 |
| [26671232298](https://github.com/sgl-project/sgl-kernel-npu/actions/runs/26671232298) | ci: fix YAML indentation in wo... | success | 65.5 分钟 | 否 |

---

## 4. 执行时间分析

### 4.1 耗时最长的30个任务（所有工作流）

| 排名 | 工作流 | 任务名称 | 运行ID | 时长(分钟) | 结论 |
|------|--------|----------|--------|------------|------|
| 1 | pr-test-npu | multi-node (test_internode_a2, 2, tests/python/deepep/test_internode_a2.py) / test_internode_a2 | 26745218348 | 100.22 | success |
| 2 | pr-test-npu | test-build-deepep-a2 | 26734123735 | 94.17 | success |
| 3 | pr-test-npu | test-build-deepep-a2 | 26745218348 | 78.37 | failure |
| 4 | pr-test-npu | test-build-deepep-a3 | 26673018437 | 67.27 | success |
| 5 | pr-test-npu | test-build-deepep-a3 | 26671232298 | 65.47 | success |
| 6 | pr-test-npu | test-all-build | 26671232298 | 63.27 | success |
| 7 | pr-test-npu | test-all-build | 26673018437 | 62.15 | success |
| 8 | pr-test-npu | multi-node (test_internode_a2, 2, tests/python/deepep/test_internode_a2.py) / test_internode_a2 | 26734123735 | 46.37 | success |
| 9 | pr-test-npu | test-build-deepep-a2 | 26671232298 | 39.13 | success |
| 10 | pr-test-npu | test-build-deepep-a2 | 26678533713 | 38.88 | success |
| 11 | pr-test-npu | test-build-deepep-a2 | 26673018437 | 38.58 | success |
| 12 | pr-test-npu | multi-node (test_internode_a2, 2, tests/python/deepep/test_internode_a2.py) / test_internode_a2 | 26748663367 | 27.05 | success |
| 13 | daily-build-test | daily-enumerate-low-latency | 26688781007 | 24.2 | success |
| 14 | daily-build-test | daily-enumerate-low-latency | 26650989313 | 23.93 | success |
| 15 | daily-build-test | daily-enumerate-low-latency | 26366390149 | 23.93 | success |
| 16 | daily-build-test | daily-enumerate-low-latency | 26717965554 | 23.88 | success |
| 17 | daily-build-test | daily-enumerate-low-latency | 26526055603 | 23.88 | success |
| 18 | daily-build-test | daily-enumerate-low-latency | 26300235624 | 23.83 | success |
| 19 | daily-build-test | daily-enumerate-low-latency | 26462975507 | 23.73 | success |
| 20 | daily-build-test | daily-enumerate-low-latency | 26590057200 | 23.67 | success |
| 21 | daily-build-test | daily-enumerate-low-latency | 26410672184 | 23.65 | success |
| 22 | daily-build-test | daily-enumerate-low-latency | 26337599333 | 22.28 | success |
| 23 | pr-test-npu | test-all-build | 26738525734 | 18.62 | success |
| 24 | pr-test-npu | test-build-deepep-a3 | 26678533713 | 17.53 | failure |
| 25 | pr-test-npu | test-build-deepep-a2 | 26738525734 | 17.5 | success |
| 26 | pr-test-npu | test-build-deepep-a3 | 26738525734 | 16.8 | success |
| 27 | pr-test-npu | test-all-build | 26678533713 | 16.5 | failure |
| 28 | pr-test-npu | test-all-build | 26734123735 | 12.25 | failure |
| 29 | pr-test-npu | test-build-deepep-a3 | 26734123735 | 11.65 | failure |
| 30 | pr-test-npu | multi-node (test_internode_a2, 2, tests/python/deepep/test_internode_a2.py) / test_internode_a2 | 26738525734 | 10.03 | success |

### 4.2 按任务名称的平均时长

| 工作流 | 任务名称 | 次数 | 平均(分钟) | 最大(分钟) | 最小(分钟) |
|--------|----------|------|------------|------------|------------|
| pr-test-npu | multi-node (test_internode_a2, 2, tests/python/deepep/test_internode_a2.py) / test_internode_a2 | 7 | 30.2 | 100.2 | 8.9 |
| pr-test-npu | test-build-deepep-a2 | 6 | 51.1 | 94.2 | 17.5 |
| pr-test-npu | test-build-deepep-a3 | 6 | 30.4 | 67.3 | 3.7 |
| pr-test-npu | test-all-build | 6 | 29.4 | 63.3 | 3.9 |
| daily-build-test | daily-enumerate-low-latency | 10 | 23.7 | 24.2 | 22.3 |
| daily-build-test | daily-enumerate-intranode | 10 | 6.4 | 7.9 | 5.8 |
| pr-test-npu | Check changed files | 7 | 0.1 | 0.1 | 0.1 |
| daily-build-test | finish | 10 | 0.0 | 0.1 | 0.0 |
| pr-test-npu | finish | 7 | 0.1 | 0.1 | 0.0 |

---

## 5. 失败分析

### 5.1 按工作流统计失败任务

| 工作流 | 失败任务数 | 有效任务总数 | 失败率 |
|--------|------------|--------------|--------|
| build_and_release | 0 | 0 | 0% |
| daily-build-test | 0 | 30 | 0.0% |
| pr-test-npu | 10 | 42 | 23.8% |

### 5.2 按任务名称统计失败

| 工作流 | 任务名称 | 失败次数 | 常见失败步骤 |
|--------|----------|----------|--------------|
| pr-test-npu | test-build-deepep-a3 | 3 | Run test intranode, Run test low latency, Run test low latency (A3 - 16 processe |
| pr-test-npu | test-all-build | 3 | Run test intranode, Run test low latency, Run test low latency (A3 - 16 processe |
| pr-test-npu | finish | 3 | Check all dependent job statuses |
| pr-test-npu | test-build-deepep-a2 | 1 | Run test intranode |

---

*报告由 GitHub Actions 日志分析器生成 | 数据源: sgl-project/sgl-kernel-npu*