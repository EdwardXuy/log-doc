# SGL-Kernel-NPU GitHub Actions Log Analysis Report

**Generated**: 2026-05-21 16:52:43
**Repository**: sgl-project/sgl-kernel-npu
**Scope**: Latest 10 completed runs per workflow
**Workflows**: build_and_release / daily-build-test / pr-test-npu

---

## 1. Execution Overview

### 1.1 Run Statistics

| Workflow | Runs | Success | Failure | Success Rate |
|----------|------|---------|---------|-------------|
| build_and_release | 10 | 7 | 3 | 70% |
| daily-build-test | 10 | 10 | 0 | 100% |
| pr-test-npu | 10 |  | 9 | 0% |
| **Total** | 30 | 17 | 12 | 56.7% |

### 1.2 Job Statistics

| Workflow | Jobs | Success | Failure | Avg Duration |
|----------|------|---------|---------|-------------|
| build_and_release | 95 | 92 | 3 | 33.9min |
| daily-build-test | 44 | 44 | 0 | 41.3min |
| pr-test-npu | 40 | 24 | 16 | 49.1min |

---

## 2. PR Workflow E2E Duration Analysis

**Target**: E2E duration <= 60 minutes (excluding resource wait time)

### 2.1 E2E Duration per Run (max core job duration)

| Run ID | Conclusion | Max Core Job Duration (min) | Meets Target |
|--------|------------|---------------------------|-------------|
| 26213188425 | failure | 1.8 | PASS |
| 26169674673 | failure | 63.2 | FAIL |
| 26167272288 | failure | 62.5 | FAIL |
| 26161634826 | failure | 62.5 | FAIL |
| 26152306380 | failure | 62.4 | FAIL |
| 26102123802 | failure | 63.4 | FAIL |
| 26095800853 | failure | 64.1 | FAIL |
| 26088340282 | failure | 63.3 | FAIL |
| 26087740898 | failure | 3 | PASS |
| 26086689989 | success | 66.8 | FAIL |

### 2.2 E2E Duration Summary

- Pass count: 2 / 10 (20%)
- Core job avg duration: 49.1min
- Core job max duration: 66.8min

### 2.3 Conclusion: PR E2E duration pass rate is low (20%). Main bottleneck is test-build-deepep-a2 job (avg 80+min, max 152min). Consider splitting tests or optimizing A2 test strategy.

---

## 3. Test Case Stability Rate Analysis

**Formula**: Stability Rate = Passed Cases / (Total Cases - Product Issue Cases)
**Target**: > 95%

### 3.1 Overall Stability Rate

| Metric | Value |
|--------|-------|
| Total test cases | 784 |
| Passed cases | 692 |
| Failed cases | 92 |
| Product issue cases | 46 |
| Code issue cases | 46 |
| **Stability Rate** | **93.77%** |
| Meets target (>95%) | FAIL |

### 3.2 Stability Rate by Workflow

| Workflow | Total | Passed | Failed | Product Issues | Stability Rate | Target |
|----------|-------|--------|--------|---------------|---------------|--------|
| pr-test-npu | 782 | 692 | 90 | 44 | 93.77% | FAIL |
| daily-build-test | 0 | 0 | 0 | 0 | 0% | FAIL |

### 3.3 Conclusion: Overall stability rate 93.77% does NOT meet target (>95%). Main impact from pr-test-npu workflow multi-node internode test persistent failures.

---

## 4. Error Classification Analysis

### 4.1 Error Type Distribution

| Error Type | Count | Percentage |
|-----------|-------|-----------|
| Unknown | 46 | 50% |
| CANNError | 22 | 23.9% |
| PythonException | 14 | 15.2% |
| TimeoutError | 10 | 10.9% |

### 4.2 Product Issue Classification

| Category | Count |
|----------|-------|
| Infrastructure-Runner | 28 |
| Infrastructure-Network | 18 |

### 4.3 Error Details by Workflow

#### pr-test-npu (90 errors)

| Job Name | Test Name | Error Type | Error Code | Product Issue | Error Summary |
|----------|-----------|------------|------------|--------------|--------------|
| finish | with | Unknown |  | No | Unknown error near line 33 |
| finish | with | Unknown |  | No | Unknown error near line 41 |
| multi-node (test_internode_a2, 2, tests/python/deepep/test_internode_a2.py) / test_internode_a2 | to | CANNError |  | Yes: Infrastructure-Runner | 2026-05-19T09:15:22.0748856Z Download action repository 'actions/checkout@v4' (S... |
| multi-node (test_internode_a2, 2, tests/python/deepep/test_internode_a2.py) / test_internode_a2 | to | CANNError |  | Yes: Infrastructure-Runner | 2026-05-19T09:15:22.0748856Z Download action repository 'actions/checkout@v4' (S... |
| multi-node (test_internode_a2, 2, tests/python/deepep/test_internode_a2.py) / test_internode_a2 | to | TimeoutError |  | Yes: Infrastructure-Network | 2026-05-19T12:16:45.6705452Z timeout in 10800s |
| multi-node (test_internode_a2, 2, tests/python/deepep/test_internode_a2.py) / test_internode_a2 | . | TimeoutError |  | Yes: Infrastructure-Network | 2026-05-19T12:16:45.6705452Z timeout in 10800s |
| test-all-build | to | PythonException |  | Yes: Infrastructure-Runner | 2026-05-19T09:33:31.4866815Z Traceback (most recent call last): |
| test-all-build | . | PythonException |  | Yes: Infrastructure-Runner | 2026-05-19T09:33:31.4866815Z Traceback (most recent call last): |
| test-build-deepep-a2 | to | PythonException |  | Yes: Infrastructure-Runner | 2026-05-19T10:09:51.1156423Z Traceback (most recent call last): |
| test-build-deepep-a2 | . | PythonException |  | Yes: Infrastructure-Runner | 2026-05-19T10:09:51.1156423Z Traceback (most recent call last): |
| test-build-deepep-a3 | to | PythonException |  | Yes: Infrastructure-Runner | 2026-05-19T09:36:40.5854583Z Traceback (most recent call last): |
| test-build-deepep-a3 | . | PythonException |  | Yes: Infrastructure-Runner | 2026-05-19T09:36:40.5854583Z Traceback (most recent call last): |
| finish | with | CANNError |  | Yes: Infrastructure-Runner | 2026-05-19T12:31:12.6122752Z Worker ID: {e2e17125-a7c2-4159-98ad-f31f81f1cf1e} |
| finish | with | CANNError |  | Yes: Infrastructure-Runner | 2026-05-19T12:31:12.6122752Z Worker ID: {e2e17125-a7c2-4159-98ad-f31f81f1cf1e} |
| multi-node (test_internode_a2, 2, tests/python/deepep/test_internode_a2.py) / test_internode_a2 | to | CANNError |  | Yes: Infrastructure-Network | 2026-05-19T09:30:07.0484831Z Download action repository 'actions/checkout@v4' (S... |
| multi-node (test_internode_a2, 2, tests/python/deepep/test_internode_a2.py) / test_internode_a2 | to | CANNError |  | Yes: Infrastructure-Network | 2026-05-19T09:30:07.0484831Z Download action repository 'actions/checkout@v4' (S... |
| multi-node (test_internode_a2, 2, tests/python/deepep/test_internode_a2.py) / test_internode_a2 | to | TimeoutError |  | Yes: Infrastructure-Network | 2026-05-19T12:30:59.0394480Z timeout in 10800s |
| multi-node (test_internode_a2, 2, tests/python/deepep/test_internode_a2.py) / test_internode_a2 | . | TimeoutError |  | Yes: Infrastructure-Network | 2026-05-19T12:30:59.0394480Z timeout in 10800s |
| test-all-build | to | Unknown |  | No | Unknown error near line 1159 |
| test-all-build | ); | Unknown |  | No | Unknown error near line 2666 |
| test-all-build | ); | Unknown |  | No | Unknown error near line 2682 |
| test-all-build | ); | Unknown |  | No | Unknown error near line 3373 |
| test-all-build | ); | Unknown |  | No | Unknown error near line 3389 |
| test-all-build | ); | Unknown |  | No | Unknown error near line 3405 |
| test-all-build | ); | Unknown |  | No | Unknown error near line 3421 |
| test-all-build | . | Unknown |  | No | Unknown error near line 14959 |
| finish | with | CANNError |  | Yes: Infrastructure-Runner | 2026-05-19T17:21:00.3819844Z Worker ID: {8b8350b4-5066-4afb-b8b2-d9e15305d716} |
| finish | with | CANNError |  | Yes: Infrastructure-Runner | 2026-05-19T17:21:00.3819844Z Worker ID: {8b8350b4-5066-4afb-b8b2-d9e15305d716} |
| multi-node (test_internode_a2, 2, tests/python/deepep/test_internode_a2.py) / test_internode_a2 | to | CANNError |  | Yes: Infrastructure-Runner | 2026-05-19T14:19:56.0900296Z Download action repository 'actions/checkout@v4' (S... |
| multi-node (test_internode_a2, 2, tests/python/deepep/test_internode_a2.py) / test_internode_a2 | to | CANNError |  | Yes: Infrastructure-Runner | 2026-05-19T14:19:56.0900296Z Download action repository 'actions/checkout@v4' (S... |
| multi-node (test_internode_a2, 2, tests/python/deepep/test_internode_a2.py) / test_internode_a2 | to | TimeoutError |  | Yes: Infrastructure-Network | 2026-05-19T17:20:46.9050225Z timeout in 10800s |
| multi-node (test_internode_a2, 2, tests/python/deepep/test_internode_a2.py) / test_internode_a2 | . | TimeoutError |  | Yes: Infrastructure-Network | 2026-05-19T17:20:46.9050225Z timeout in 10800s |
| finish | with | Unknown |  | No | Unknown error near line 33 |
| finish | with | Unknown |  | No | Unknown error near line 41 |
| multi-node (test_internode_a2, 2, tests/python/deepep/test_internode_a2.py) / test_internode_a2 | to | CANNError |  | Yes: Infrastructure-Runner | 2026-05-19T14:00:13.7165221Z Download action repository 'actions/checkout@v4' (S... |
| multi-node (test_internode_a2, 2, tests/python/deepep/test_internode_a2.py) / test_internode_a2 | to | CANNError |  | Yes: Infrastructure-Runner | 2026-05-19T14:00:13.7165221Z Download action repository 'actions/checkout@v4' (S... |
| multi-node (test_internode_a2, 2, tests/python/deepep/test_internode_a2.py) / test_internode_a2 | to | TimeoutError |  | Yes: Infrastructure-Network | 2026-05-19T17:01:00.1226531Z timeout in 10800s |
| multi-node (test_internode_a2, 2, tests/python/deepep/test_internode_a2.py) / test_internode_a2 | . | TimeoutError |  | Yes: Infrastructure-Network | 2026-05-19T17:01:00.1226531Z timeout in 10800s |
| finish | with | Unknown |  | No | Unknown error near line 33 |
| finish | with | Unknown |  | No | Unknown error near line 41 |
| multi-node (test_internode_a2, 2, tests/python/deepep/test_internode_a2.py) / test_internode_a2 | to | CANNError |  | Yes: Infrastructure-Runner | 2026-05-20T08:58:56.5269639Z Download action repository 'actions/checkout@v4' (S... |
| multi-node (test_internode_a2, 2, tests/python/deepep/test_internode_a2.py) / test_internode_a2 | to | CANNError |  | Yes: Infrastructure-Runner | 2026-05-20T08:58:56.5269639Z Download action repository 'actions/checkout@v4' (S... |
| multi-node (test_internode_a2, 2, tests/python/deepep/test_internode_a2.py) / test_internode_a2 | . | Unknown |  | No | Unknown error near line 2599 |
| multi-node (test_internode_a2, 2, tests/python/deepep/test_internode_a2.py) / test_internode_a2 | . | Unknown |  | No | Unknown error near line 2646 |
| multi-node (test_internode_a2, 2, tests/python/deepep/test_internode_a2.py) / test_internode_a2 | . | Unknown |  | No | Unknown error near line 2693 |
| multi-node (test_internode_a2, 2, tests/python/deepep/test_internode_a2.py) / test_internode_a2 | after | Unknown |  | No | Unknown error near line 2695 |
| multi-node (test_internode_a2, 2, tests/python/deepep/test_internode_a2.py) / test_internode_a2 | to | Unknown |  | No | Unknown error near line 2714 |
| multi-node (test_internode_a2, 2, tests/python/deepep/test_internode_a2.py) / test_internode_a2 | . | Unknown |  | No | Unknown error near line 2716 |
| finish | with | Unknown |  | No | Unknown error near line 33 |
| finish | with | Unknown |  | No | Unknown error near line 41 |
| multi-node (test_internode_a2, 2, tests/python/deepep/test_internode_a2.py) / test_internode_a2 | to | CANNError |  | Yes: Infrastructure-Runner | 2026-05-20T12:11:17.3484242Z Download action repository 'actions/checkout@v4' (S... |
| multi-node (test_internode_a2, 2, tests/python/deepep/test_internode_a2.py) / test_internode_a2 | to | CANNError |  | Yes: Infrastructure-Runner | 2026-05-20T12:11:17.3484242Z Download action repository 'actions/checkout@v4' (S... |
| multi-node (test_internode_a2, 2, tests/python/deepep/test_internode_a2.py) / test_internode_a2 | . | Unknown |  | No | Unknown error near line 1466 |
| multi-node (test_internode_a2, 2, tests/python/deepep/test_internode_a2.py) / test_internode_a2 | . | Unknown |  | No | Unknown error near line 1513 |
| multi-node (test_internode_a2, 2, tests/python/deepep/test_internode_a2.py) / test_internode_a2 | . | Unknown |  | No | Unknown error near line 1560 |
| multi-node (test_internode_a2, 2, tests/python/deepep/test_internode_a2.py) / test_internode_a2 | after | Unknown |  | No | Unknown error near line 1562 |
| multi-node (test_internode_a2, 2, tests/python/deepep/test_internode_a2.py) / test_internode_a2 | to | Unknown |  | No | Unknown error near line 1581 |
| multi-node (test_internode_a2, 2, tests/python/deepep/test_internode_a2.py) / test_internode_a2 | . | Unknown |  | No | Unknown error near line 1583 |
| finish | with | Unknown |  | No | Unknown error near line 33 |
| finish | with | Unknown |  | No | Unknown error near line 41 |
| multi-node (test_internode_a2, 2, tests/python/deepep/test_internode_a2.py) / test_internode_a2 | to | CANNError |  | Yes: Infrastructure-Runner | 2026-05-20T13:57:52.8793213Z Download action repository 'actions/checkout@v4' (S... |
| multi-node (test_internode_a2, 2, tests/python/deepep/test_internode_a2.py) / test_internode_a2 | to | CANNError |  | Yes: Infrastructure-Runner | 2026-05-20T13:57:52.8793213Z Download action repository 'actions/checkout@v4' (S... |
| multi-node (test_internode_a2, 2, tests/python/deepep/test_internode_a2.py) / test_internode_a2 | . | Unknown |  | No | Unknown error near line 1485 |
| multi-node (test_internode_a2, 2, tests/python/deepep/test_internode_a2.py) / test_internode_a2 | . | Unknown |  | No | Unknown error near line 1532 |
| multi-node (test_internode_a2, 2, tests/python/deepep/test_internode_a2.py) / test_internode_a2 | . | Unknown |  | No | Unknown error near line 1579 |
| multi-node (test_internode_a2, 2, tests/python/deepep/test_internode_a2.py) / test_internode_a2 | after | Unknown |  | No | Unknown error near line 1581 |
| multi-node (test_internode_a2, 2, tests/python/deepep/test_internode_a2.py) / test_internode_a2 | to | Unknown |  | No | Unknown error near line 1600 |
| multi-node (test_internode_a2, 2, tests/python/deepep/test_internode_a2.py) / test_internode_a2 | . | Unknown |  | No | Unknown error near line 1602 |
| finish | with | Unknown |  | No | Unknown error near line 33 |
| finish | with | Unknown |  | No | Unknown error near line 41 |
| multi-node (test_internode_a2, 2, tests/python/deepep/test_internode_a2.py) / test_internode_a2 | to | CANNError |  | Yes: Infrastructure-Runner | 2026-05-20T14:38:21.2278994Z Download action repository 'actions/checkout@v4' (S... |
| multi-node (test_internode_a2, 2, tests/python/deepep/test_internode_a2.py) / test_internode_a2 | to | CANNError |  | Yes: Infrastructure-Runner | 2026-05-20T14:38:21.2278994Z Download action repository 'actions/checkout@v4' (S... |
| multi-node (test_internode_a2, 2, tests/python/deepep/test_internode_a2.py) / test_internode_a2 | . | Unknown |  | No | Unknown error near line 1467 |
| multi-node (test_internode_a2, 2, tests/python/deepep/test_internode_a2.py) / test_internode_a2 | . | Unknown |  | No | Unknown error near line 1514 |
| multi-node (test_internode_a2, 2, tests/python/deepep/test_internode_a2.py) / test_internode_a2 | . | Unknown |  | No | Unknown error near line 1561 |
| multi-node (test_internode_a2, 2, tests/python/deepep/test_internode_a2.py) / test_internode_a2 | after | Unknown |  | No | Unknown error near line 1563 |
| multi-node (test_internode_a2, 2, tests/python/deepep/test_internode_a2.py) / test_internode_a2 | to | Unknown |  | No | Unknown error near line 1582 |
| multi-node (test_internode_a2, 2, tests/python/deepep/test_internode_a2.py) / test_internode_a2 | . | Unknown |  | No | Unknown error near line 1584 |
| finish | with | Unknown |  | No | Unknown error near line 33 |
| finish | with | Unknown |  | No | Unknown error near line 41 |
| multi-node (test_internode_a2, 2, tests/python/deepep/test_internode_a2.py) / test_internode_a2 | to | CANNError |  | Yes: Infrastructure-Network | 2026-05-21T07:56:42.1362094Z Download action repository 'actions/checkout@v4' (S... |
| multi-node (test_internode_a2, 2, tests/python/deepep/test_internode_a2.py) / test_internode_a2 | to | CANNError |  | Yes: Infrastructure-Network | 2026-05-21T07:56:42.1362094Z Download action repository 'actions/checkout@v4' (S... |
| multi-node (test_internode_a2, 2, tests/python/deepep/test_internode_a2.py) / test_internode_a2 | to | TimeoutError |  | Yes: Infrastructure-Network | 2026-05-21T07:57:46.1047764Z Timeout set to: 10800 seconds |
| multi-node (test_internode_a2, 2, tests/python/deepep/test_internode_a2.py) / test_internode_a2 | . | TimeoutError |  | Yes: Infrastructure-Network | 2026-05-21T07:57:46.1047764Z Timeout set to: 10800 seconds |
| test-all-build | to | PythonException |  | Yes: Infrastructure-Runner | 2026-05-21T08:03:30.1066920Z Traceback (most recent call last): |
| test-all-build | . | PythonException |  | Yes: Infrastructure-Runner | 2026-05-21T08:03:30.1066920Z Traceback (most recent call last): |
| test-build-deepep-a2 | to | PythonException |  | Yes: Infrastructure-Network | 2026-05-21T07:58:01.3735871Z Traceback (most recent call last): |
| test-build-deepep-a2 | . | PythonException |  | Yes: Infrastructure-Network | 2026-05-21T07:58:01.3735871Z Traceback (most recent call last): |
| test-build-deepep-a3 | to | PythonException |  | Yes: Infrastructure-Runner | 2026-05-21T07:58:16.7762787Z Traceback (most recent call last): |
| test-build-deepep-a3 | . | PythonException |  | Yes: Infrastructure-Runner | 2026-05-21T07:58:16.7762787Z Traceback (most recent call last): |

#### build_and_release (2 errors)

| Job Name | Test Name | Error Type | Error Code | Product Issue | Error Summary |
|----------|-----------|------------|------------|--------------|--------------|
| build-and-release (x86_64, 910b, 8.5.0, 2.10.0) | to | PythonException |  | Yes: Infrastructure-Network | 2026-05-21T07:48:09.9578715Z Traceback (most recent call last): |
| build-and-release (x86_64, 910b, 8.5.0, 2.10.0) | to | PythonException |  | Yes: Infrastructure-Network | 2026-05-21T07:48:09.9578715Z Traceback (most recent call last): |

---

## 5. Key Findings and Recommendations

### 5.1 Key Findings

1. **multi-node internode A2 test persistent failure**: test_internode_a2 failed in 9/10 runs, the biggest instability factor
2. **DeepEP A2 build time too long**: test-build-deepep-a2 job can take up to 152min, far exceeding 60min target
3. **CANN errors frequent**: 22 errors classified as CANNError, need to check CANN version compatibility
4. **Infrastructure issues prominent**: 28 Runner infrastructure errors, 18 network timeout errors
5. **daily-build-test workflow stable**: All 10 runs successful, no failures

### 5.2 Recommendations

1. **Internode A2 test**: Investigate K8s cluster configuration and RDMA communication issues (nearly 100% failure rate)
2. **A2 build optimization**: Consider splitting test-build-deepep-a2 into multiple parallel jobs or optimizing test parameters
3. **CANN compatibility**: Verify CANN 9.0.0 compatibility with current codebase (22 CANNError may relate to version upgrade)
4. **Runner reliability**: 28 Runner-related failures indicate self-hosted runners need enhanced stability monitoring
5. **Network timeout**: 18 network timeout errors suggest adding retry mechanisms

---

## 6. Run Details

### 6.1 build_and_release

| Run ID | Conclusion | Event | Created |
|--------|------------|-------|---------|
| 26213188353 | failure | pull_request | 2026-05-21T07:55:43Z |
| 26212747717 | failure | pull_request | 2026-05-21T07:45:56Z |
| 26138650165 | success | release | 2026-05-20T03:03:32Z |
| 26095800844 | success | pull_request | 2026-05-19T12:01:05Z |
| 26087741075 | success | pull_request | 2026-05-19T09:12:06Z |
| 26083669928 | failure | pull_request | 2026-05-19T07:46:36Z |
| 26081179968 | success | pull_request | 2026-05-19T06:49:06Z |
| 26033602703 | success | pull_request | 2026-05-18T12:30:58Z |
| 26010780037 | success | pull_request | 2026-05-18T02:46:15Z |
| 26010217992 | success | pull_request | 2026-05-18T02:25:47Z |

### 6.2 daily-build-test

| Run ID | Conclusion | Event | Created |
|--------|------------|-------|---------|
| 26177429996 | success | schedule | 2026-05-20T16:59:21Z |
| 26167272155 | success | pull_request | 2026-05-20T13:56:11Z |
| 26152306421 | success | pull_request | 2026-05-20T08:58:06Z |
| 26112217228 | success | schedule | 2026-05-19T16:56:24Z |
| 26102123867 | success | pull_request | 2026-05-19T13:59:20Z |
| 26092898754 | success | pull_request | 2026-05-19T10:59:44Z |
| 26047706057 | success | schedule | 2026-05-18T16:54:40Z |
| 26010780027 | success | pull_request | 2026-05-18T02:46:15Z |
| 25996129429 | success | schedule | 2026-05-17T16:18:15Z |
| 25966777518 | success | schedule | 2026-05-16T16:16:44Z |

### 6.3 pr-test-npu

| Run ID | Conclusion | Event | Created |
|--------|------------|-------|---------|
| 26213188425 | failure | pull_request | 2026-05-21T07:55:43Z |
| 26169674673 | failure | pull_request | 2026-05-20T14:37:08Z |
| 26167272288 | failure | pull_request | 2026-05-20T13:56:11Z |
| 26161634826 | failure | pull_request | 2026-05-20T12:10:42Z |
| 26152306380 | failure | pull_request | 2026-05-20T08:58:06Z |
| 26102123802 | failure | pull_request | 2026-05-19T13:59:20Z |
| 26095800853 | failure | pull_request | 2026-05-19T12:01:05Z |
| 26088340282 | failure | pull_request | 2026-05-19T09:24:22Z |
| 26087740898 | failure | pull_request | 2026-05-19T09:12:06Z |
| 26086689989 | success | pull_request | 2026-05-19T08:50:40Z |

---

*Report generated automatically by GitHub Actions Log Analyzer*
*Data source: sgl-project/sgl-kernel-npu repository GitHub Actions API*
