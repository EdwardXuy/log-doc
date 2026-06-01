# 错误日志分析报告

**生成时间**: 2026-06-01 19:56:07
**分析范围**:  至 
**错误记录总数**: 13

---

## 1. 错误类别汇总

| 类别 | 数量 | 占比 |
|------|------|------|
| 产品问题 | 7 | 53.8% |
| 用例设计问题 | 4 | 30.8% |
| 未知 | 2 | 15.4% |

### 子类别细分
| 类别 | 子类别 | 数量 |
|------|--------|------|
| 产品问题 | NPU算子错误 | 5 |
| 用例设计问题 | 断言失败 | 2 |
| 未知 | Python异常 | 2 |
| 用例设计问题 | 模块缺失 | 2 |
| 产品问题 | CANN框架错误 | 2 |

---

## 2. 根因分析

| 根因 | 数量 | 置信度 | 修复建议 |
|------|------|--------|----------|
| 未知 - 需人工复核 | 5 | 低 | 手动复核日志以确定根因 |
| 产品问题 - NPU算子Tiling缺陷 | 4 | 高 | 上报CANN/NPU团队：MoeDistributeDispatchV2 tiling失败 |
| 用例设计问题 - 精度阈值过严 | 2 | 中 | 调整测试精度阈值或深入调查数值差异原因 |
| 用例设计问题 - 构建产物缺失 | 2 | 高 | 确保DeepEP wheel已构建并在测试前安装 |

---

## 3. 详细错误记录

### pr-test-npu_run-26678533713 / test-all-build

| 测试 | 错误详情 | 类别 | 根因 | 置信度 |
|------|----------|------|------|--------|
| unknown | 2026-05-30T08:08:50.1511080Z git version 2.34.1 / 2026-05-30T08:08:50.1544060Z Copying '/root/.gitconfig' to '/__w/_temp | 用例设计问题 | 用例设计问题 - 精度阈值过严 | 中 |

### pr-test-npu_run-26678533713 / test-build-deepep-a3

| 测试 | 错误详情 | 类别 | 根因 | 置信度 |
|------|----------|------|------|--------|
| unknown | 2026-05-30T08:09:49.3887823Z git version 2.34.1 / 2026-05-30T08:09:49.3918917Z Copying '/root/.gitconfig' to '/__w/_temp | 用例设计问题 | 用例设计问题 - 精度阈值过严 | 中 |

### pr-test-npu_run-26734123735 / test-all-build

| 测试 | 错误详情 | 类别 | 根因 | 置信度 |
|------|----------|------|------|--------|
| unknown | 2026-06-01T04:07:45.3232765Z Exception raised from low_latency_dispatch at /__w/sgl-kernel-npu/sgl-kernel-npu/csrc/deepe | 产品问题 | 产品问题 - NPU算子Tiling缺陷 | 高 |
| unknown | 2026-06-01T04:07:45.3232765Z Exception raised from low_latency_dispatch at /__w/sgl-kernel-npu/sgl-kernel-npu/csrc/deepe | 产品问题 | 产品问题 - NPU算子Tiling缺陷 | 高 |

### pr-test-npu_run-26734123735 / test-build-deepep-a3

| 测试 | 错误详情 | 类别 | 根因 | 置信度 |
|------|----------|------|------|--------|
| unknown | 2026-06-01T04:02:16.1387949Z Exception raised from low_latency_dispatch at /__w/sgl-kernel-npu/sgl-kernel-npu/csrc/deepe | 产品问题 | 产品问题 - NPU算子Tiling缺陷 | 高 |
| unknown | 2026-06-01T04:02:16.1387949Z Exception raised from low_latency_dispatch at /__w/sgl-kernel-npu/sgl-kernel-npu/csrc/deepe | 产品问题 | 产品问题 - NPU算子Tiling缺陷 | 高 |

### pr-test-npu_run-26745218348 / test-all-build

| 测试 | 错误详情 | 类别 | 根因 | 置信度 |
|------|----------|------|------|--------|
| unknown | 2026-06-01T09:02:38.3310694Z                                                                ^ / 2026-06-01T09:02:38.3311 | 未知 | 未知 - 需人工复核 | 低 |
| unknown | 2026-06-01T09:02:38.3310694Z                                                                ^ / 2026-06-01T09:02:38.3311 | 未知 | 未知 - 需人工复核 | 低 |
| unknown | 2026-06-01T09:03:53.8760415Z ##[group]Run '/home/runner/k8s/index.js' / 2026-06-01T09:03:53.8761645Z shell: /home/runner | 用例设计问题 | 用例设计问题 - 构建产物缺失 | 高 |

### pr-test-npu_run-26745218348 / test-build-deepep-a2

| 测试 | 错误详情 | 类别 | 根因 | 置信度 |
|------|----------|------|------|--------|
| unknown | 2026-06-01T10:17:18.7944307Z         Check NnopbaseExecutorMatchCache(executor) failed / 2026-06-01T10:17:18.7944759Z    | 产品问题 | 未知 - 需人工复核 | 低 |

### pr-test-npu_run-26745218348 / test-build-deepep-a3

| 测试 | 错误详情 | 类别 | 根因 | 置信度 |
|------|----------|------|------|--------|
| unknown | 2026-06-01T10:07:51.7417777Z                                                          ^ / 2026-06-01T10:07:51.7418175Z / | 产品问题 | 未知 - 需人工复核 | 低 |
| unknown | 2026-06-01T10:07:51.7417777Z                                                          ^ / 2026-06-01T10:07:51.7418175Z / | 产品问题 | 未知 - 需人工复核 | 低 |
| unknown | 2026-06-01T10:09:07.0936459Z ##[group]Run '/home/runner/k8s/index.js' / 2026-06-01T10:09:07.0937575Z shell: /home/runner | 用例设计问题 | 用例设计问题 - 构建产物缺失 | 高 |

---

*报告由 GitHub Actions 日志分析器错误分析模块生成*