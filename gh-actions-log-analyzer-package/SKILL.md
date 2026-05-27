---
name: gh-actions-log-analyzer
version: "3.0"
description: Analyze GitHub Actions workflow run logs across multiple workflows, fetch job logs from API, identify failed tests, categorize error types, calculate stability rates and E2E durations, and generate detailed reports in Markdown/CSV/Excel format. Reports are generated in Chinese. Use when analyzing GitHub Actions logs, CI/CD failures, or workflow run statistics.
---

# GitHub Actions Log Analyzer v3

Multi-workflow GitHub Actions log analysis system with statistical reporting.

## Architecture

```
Step 1: scripts/fetch_data.ps1     -> analysis/logs/{timestamp}/ + analysis/reports/{timestamp}/all_runs.csv + all_jobs.csv
Step 2: scripts/generate_report.py -> analysis/reports/{timestamp}/full_analysis_report.md + analysis_report.xlsx
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

# 2. Fetch data from GitHub API
$token = (Get-Content "token.txt" -Raw).Trim()
.\scripts\fetch_data.ps1 -Token $token -Repo "sgl-project/sgl-kernel-npu" -OutputDir ".\analysis"

# 3. Generate reports (Chinese)
python .\scripts\generate_report.py --timestamp <TIMESTAMP_FROM_STEP2>
```

### Example

```powershell
$token = (Get-Content "token.txt" -Raw).Trim()
.\scripts\fetch_data.ps1 -Token $token
# Output: "Timestamp: 20260526-194128"

python .\scripts\generate_report.py --timestamp 20260526-194128
# Output: full_analysis_report.md + analysis_report.xlsx + optimization_proposal.md
```

## Input Parameters

### fetch_data.ps1

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| Token | Yes | - | GitHub PAT with `repo` permission |
| Repo | No | sgl-project/sgl-kernel-npu | Repository in `owner/repo` format |
| OutputDir | No | `.\analysis` | Output directory |
| MaxPages | No | 5 | Max pages to fetch per workflow (30 runs/page) |

### generate_report.py

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| --timestamp | Yes | - | Timestamp folder name from fetch_data.ps1 output |
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
    analysis_report.xlsx    # Multi-sheet Excel workbook
  optimization/{timestamp}/
    optimization_proposal.md # Time optimization proposals
```

## Report Contents (Chinese)

The `full_analysis_report.md` contains:

1. **Overview**: Per-workflow run/job success rates
2. **Per-Run Job Statistics**: Each run's job success/failure/cancelled counts with pass rates
3. **Stability Analysis**: Job-level stability rates, PR E2E duration analysis
4. **Execution Time Analysis**: Top longest jobs, average duration by job name
5. **Failure Analysis**: Failed jobs by workflow and job name, common failed steps

The `optimization_proposal.md` contains:

1. Current PR workflow time breakdown
2. Bottleneck analysis
3. Optimization proposals (split jobs, share artifacts, conditional internode, etc.)
4. Summary of expected improvements

## Workflow IDs

The following workflow IDs are hardcoded in `fetch_data.ps1`:

| Workflow | ID |
|----------|-----|
| build_and_release | 213514694 |
| daily-build-test | 204375811 |
| pr-test-npu | 179736185 |

To analyze a different repository, modify these IDs in the script.

## Error Categories

The system identifies the following failure patterns from logs:

| Error Type | Pattern | Description |
|------------|---------|-------------|
| Build Failure | `/set_env.sh: No such file or directory` | CANN environment init script missing |
| Import Failure | `ModuleNotFoundError: No module named 'deep_ep'` | DeepEP wheel not built/installed |
| K8s Pod Pending | `Pod.*Pending.*timeout` | K8s cluster resource insufficient |
| K8s Pod Error | `Pod logs ended but target pattern was not detected` | Pod started but test process crashed |
| Step Failure | Job step conclusion = failure | Individual step failure |

## Statistical Metrics

### E2E Duration

- Definition: Maximum execution time of core jobs per run
- Target: <= 60 minutes (excluding resource wait time)
- Wait time: `started_at - created_at` at run level

### Stability Rate

- Formula: `Passed Jobs / (Total Jobs - Cancelled Jobs) * 100%`
- Target: > 95%
- Product issues: Build script bugs, K8s infrastructure issues

## Incremental Analysis

To analyze only new runs since last analysis:

1. Check the latest RunId in the previous `all_runs.csv`
2. Modify `fetch_data.ps1` to add a `--since-run-id` parameter
3. Append new data to existing CSV files
4. Regenerate reports

## Files

| File | Description |
|------|-------------|
| `scripts/fetch_data.ps1` | Fetch all completed runs and jobs from GitHub API |
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
3. **API Rate Limit**: PAT allows 5000 requests/hour. Fetching ~370 runs with ~2273 jobs uses ~300 requests.
4. **Timestamp Organization**: Each analysis run creates a timestamped folder, allowing historical comparison.
