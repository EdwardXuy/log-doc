---
name: gh-actions-log-analyzer
version: "3.3"
description: Analyze GitHub Actions workflow run logs for sgl-kernel-npu repository. Fetch job logs from API, identify failed tests, categorize error types with root cause analysis (product/test design/infrastructure), calculate stability rates and E2E durations, and generate detailed reports in Markdown/CSV/Excel format. Reports are generated in Chinese. Use when analyzing GitHub Actions logs, CI/CD failures, or workflow run statistics.
---

# GitHub Actions Log Analyzer v3.3

Multi-workflow GitHub Actions log analysis system with statistical reporting and error root cause analysis.

## Architecture

```
Step 1: scripts/fetch_data.ps1     -> analysis/logs/{timestamp}/ + analysis/reports/{timestamp}/all_runs.csv + all_jobs.csv
Step 2: scripts/analyze_errors.py  -> analysis/reports/{timestamp}/failed_tests_detail.csv + error_analysis_report.md (Chinese)
Step 3: scripts/generate_report.py -> analysis/reports/{timestamp}/full_analysis_report.md (Chinese) + analysis_report.xlsx
                                       analysis/optimization/{timestamp}/optimization_proposal.md
```

## Quick Start

### Prerequisites

- PowerShell 5.1+ (for data fetching)
- Python 3.8+ with openpyxl (for report generation)
- GitHub PAT with `repo` permission (stored in `token.txt`)

### Full Pipeline

```powershell
# 1. Prepare token file (DO NOT commit this file)
echo "ghp_your_token_here" > token.txt

# 2. Fetch data from GitHub API (full fetch - ignore incremental state, get exactly 10 runs per workflow)
$token = (Get-Content "token.txt" -Raw).Trim()
.\scripts\fetch_data.ps1 -Token $token -Repo "sgl-project/sgl-kernel-npu" -OutputDir ".\analysis" -ForceFullFetch

# 3. Analyze error logs (root cause classification, Chinese report)
python .\scripts\analyze_errors.py --timestamp <TIMESTAMP_FROM_STEP2> --base-dir "."

# 4. Generate reports (Chinese)
python .\scripts\generate_report.py --timestamp <TIMESTAMP_FROM_STEP2> --base-dir "."
```

### Example

```powershell
$token = (Get-Content "token.txt" -Raw).Trim()
.\scripts\fetch_data.ps1 -Token $token -SinceDate "2026-05-30"
# Output: "Timestamp: 20260601-155605"

python .\scripts\analyze_errors.py --timestamp 20260601-155605 --base-dir "."
# Output: failed_tests_detail.csv + error_analysis_report.md (Chinese)

python .\scripts\generate_report.py --timestamp 20260601-155605 --base-dir "."
# Output: full_analysis_report.md (Chinese) + analysis_report.xlsx + optimization_proposal.md
```

## Workflow Analysis Scope

| Workflow | Trigger | Analysis Scope |
|----------|---------|---------------|
| daily-build-test | schedule (cron: 0 16 * * *) | Latest 10 schedule event runs |
| pr-test-npu | pull_request | Latest 10 completed runs |

Note: `build_and_release` workflow is excluded from analysis.

## Input Parameters

### fetch_data.ps1

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| Token | Yes | - | GitHub PAT with `repo` permission |
| Repo | No | sgl-project/sgl-kernel-npu | Repository in `owner/repo` format |
| OutputDir | No | `.\analysis` | Output directory |
| MaxPages | No | 10 | Max pages to fetch per workflow (30 runs/page) |
| SinceDate | No | "" | Start date filter (e.g. "2026-05-30") |
| UntilDate | No | "" | End date filter (e.g. "2026-06-01") |
| ForceFullFetch | No | False | Ignore incremental state, force full fetch to get exactly 10 runs per workflow |

### analyze_errors.py

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| --timestamp | Yes | - | Timestamp folder name from fetch_data.ps1 |
| --base-dir | No | `.` | Base directory containing analysis folder |
| --since-date | No | "" | Start date for report header |
| --until-date | No | "" | End date for report header |

### generate_report.py

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| --timestamp | Yes | - | Timestamp folder name from fetch_data.ps1 |
| --base-dir | No | `.` | Base directory containing analysis folder |
| --repo | No | sgl-project/sgl-kernel-npu | Repository name (used in report title) |

## Output Files

All outputs are organized by timestamp:

```
analysis/
  logs/{timestamp}/
    {workflow}_run-{run_id}/
      summary.json          # Run summary
      {job_name}/
        job-info.json       # Job metadata with steps
        full-log.txt        # Full log (failed jobs only)
  reports/{timestamp}/
    all_runs.csv            # All runs metadata
    all_jobs.csv            # All jobs metadata
    all_runs.json           # All runs metadata (JSON)
    all_jobs.json           # All jobs metadata (JSON)
    full_analysis_report.md # Comprehensive analysis report (Chinese)
    error_analysis_report.md # Error root cause analysis report (Chinese)
    failed_tests_detail.csv # Failed test details with classification
    analysis_report.xlsx    # Multi-sheet Excel workbook
  optimization/{timestamp}/
    optimization_proposal.md # Time optimization proposals
  analysis_state.json       # State for incremental analysis
```

## Report Contents

### full_analysis_report.md (Chinese)

1. **Overview**: Per-workflow run/job success rates
2. **Per-Run Job Statistics**: Each run's job success/failure/cancelled counts with pass rates
3. **Stability Analysis**: Job-level stability rates, PR E2E duration analysis
4. **Execution Time Analysis**: Top longest jobs, average duration by job name
5. **Failure Analysis**: Failed jobs by workflow and job name, common failed steps

### error_analysis_report.md (Chinese)

1. **错误类别汇总**: 产品问题 vs 用例设计问题 vs 基础设施问题
2. **根因分析**: 详细根因，含置信度和修复建议
3. **详细错误记录**: 按Run/Job分组，含GitHub链接直达原始日志

### failed_tests_detail.csv

Columns: RunId, JobName, TestPath, TestName, ErrorDetail, Category, SubCategory, Description, RootCause, Confidence, Recommendation, LogLine, RunUrl

## Error Categories (Chinese)

The system classifies errors into three top-level categories:

### 产品问题

| 子类别 | 匹配模式 | 说明 |
|--------|---------|------|
| NPU算子错误 | `aclnn.*failed`, `EZ9999`, `NPU function error` | NPU算子执行失败（tiling/映射/运行时） |
| HCCL通信错误 | `HCCL.*error`, `HCCL_BUFFSIZE` | HCCL集合通信错误 |
| CANN框架错误 | `CANN.*error`, `E\d{5}` | CANN框架或驱动错误 |
| Triton兼容性 | `triton.language has no attribute` | Triton版本与NPU不兼容 |
| DeepEP运行时错误 | `deep_ep.*error`, `RuntimeError.*deep_ep` | DeepEP库运行时错误 |

### 用例设计问题

| 子类别 | 匹配模式 | 说明 |
|--------|---------|------|
| 断言失败 | `AssertionError`, `assert diff.*Error` | 测试断言失败（精度/行为不匹配） |
| 模块缺失 | `ModuleNotFoundError`, `ImportError: cannot import name` | Python模块缺失或导入失败 |
| 配置属性缺失 | `Config.*has no attribute` | 模型配置缺少必要属性 |

### 基础设施问题

| 子类别 | 匹配模式 | 说明 |
|--------|---------|------|
| K8s Pod调度超时 | `Pod.*Pending.*timeout` | K8s Pod调度超时 |
| K8s Pod崩溃 | `Pod logs ended but target pattern was not detected` | K8s Pod启动后测试进程崩溃 |
| Runner/环境问题 | `runner lost`, `No such file or directory.*set_env` | 自托管Runner或环境问题 |
| 超时 | `timeout`, `timed out` | 执行超时 |
| 内存不足 | `out of memory`, `OOM`, `Killed` | 内存不足或进程被杀死 |

## Root Cause Analysis (Chinese)

The system automatically determines root cause and provides recommendations:

| 根因 | 置信度 | 修复建议 |
|------|--------|----------|
| 产品问题 - NPU算子Tiling缺陷 | 高 | 上报CANN/NPU团队：算子tiling失败 |
| 产品问题 - NPU算子运行时缺陷 | 高 | 上报CANN/NPU团队：算子运行时失败 |
| 用例设计问题 - 精度阈值过严 | 中 | 调整测试精度阈值或深入调查数值差异原因 |
| 用例设计问题 - 构建产物缺失 | 高 | 确保DeepEP wheel已构建并在测试前安装 |
| 用例设计问题 - 依赖缺失 | 中 | 在requirements或测试环境中添加缺失的Python包 |
| 基础设施问题 - K8s资源不足 | 高 | 检查K8s集群资源可用性 |
| 基础设施问题 - 资源耗尽 | 高 | 增加Pod内存限制或减少batch size |

## Incremental Analysis

State file `analysis_state.json` tracks last analyzed run per workflow:

```json
{
  "last_analysis_time": "2026-06-01T15:56:05Z",
  "analysis_timestamp": "20260601-155605",
  "run_count": 11,
  "job_count": 39,
  "since_date": "2026-05-30",
  "until_date": "",
  "workflows": {
    "daily-build-test": { "latest_run_id": 26717965554 },
    "pr-test-npu": { "latest_run_id": 26738525734 }
  }
}
```

To run incrementally (analyze only new runs since last analysis):
1. Read `analysis_state.json`
2. Use `SinceDate` parameter or fetch runs newer than `latest_run_id`
3. Append new data to existing CSV files (not overwrite)
4. Update `analysis_state.json`
5. Regenerate reports

**Note**: Use `-ForceFullFetch` parameter to ignore incremental state and force fetching exactly 10 runs per workflow. This is useful when you need a consistent 10-run analysis regardless of previous analysis state.

## Workflow IDs

The following workflow IDs are hardcoded in `fetch_data.ps1`:

| Workflow | ID |
|----------|-----|
| daily-build-test | 204375811 |
| pr-test-npu | 179736185 |

To analyze a different repository, modify these IDs in the script.

## Statistical Metrics

### E2E Duration

- Definition: Maximum execution time of core jobs per run
- Target: <= 60 minutes (excluding resource wait time)
- Wait time: `started_at - created_at` at run level

### Stability Rate

- Formula: `Passed Jobs / (Total Jobs - Cancelled Jobs) * 100%`
- Target: > 95%
- Product issues: Build script bugs, K8s infrastructure issues

## Files

| File | Description |
|------|-------------|
| `scripts/fetch_data.ps1` | Fetch completed runs and jobs from GitHub API |
| `scripts/analyze_errors.py` | Analyze failed job logs, classify errors, determine root causes (Chinese output) |
| `scripts/generate_report.py` | Generate Chinese analysis reports and Excel workbook |
| `SKILL.md` | This documentation file |

## Requirements

- PowerShell 5.1+
- Python 3.8+ with openpyxl
- Git (for uploading reports to repository)
- GitHub PAT with `repo` permission

## Important Notes

1. **Language**: Reports are generated in Chinese. Technical terms (job names, error types) remain in English.
2. **Token Security**: NEVER commit `token.txt` or include tokens in code. Always read from a separate file.
3. **API Rate Limit**: PAT allows 5000 requests/hour. Fetching ~20 runs with ~100 jobs uses ~50 requests.
4. **Timestamp Organization**: Each analysis run creates a timestamped folder, allowing historical comparison.
5. **Error Analysis**: The error analysis module focuses on failed job logs and provides root cause classification to help distinguish product bugs from test design issues.
6. **Workflow Scope**: 
   - `daily-build-test` only analyzes `schedule` event runs (latest 10)
   - `pr-test-npu` analyzes latest 10 completed runs
   - `build_and_release` is excluded from analysis
7. **Path Handling**: Output directory is resolved to absolute path to avoid path concatenation issues. Use `-ForceFullFetch` when you need exactly 10 runs per workflow regardless of incremental analysis state.
