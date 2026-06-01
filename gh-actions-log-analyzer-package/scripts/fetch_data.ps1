param(
    [Parameter(Mandatory=$true)]
    [string]$Token,
    [string]$Repo = "sgl-project/sgl-kernel-npu",
    [string]$OutputDir = ".\analysis",
    [int]$MaxPages = 10,
    [string]$SinceDate = "",   # e.g. "2026-05-30"
    [string]$UntilDate = "",   # e.g. "2026-06-01"
    [switch]$ForceFullFetch    # Force full fetch, ignore incremental state
)

# Resolve OutputDir to absolute path to avoid path issues
$resolved = (Resolve-Path $OutputDir -ErrorAction SilentlyContinue)
if ($resolved) {
    $OutputDir = $resolved.Path
} else {
    $OutputDir = (Join-Path (Get-Location) $OutputDir)
}

$headers = @{
    "Authorization" = "token $Token"
    "Accept" = "application/vnd.github.v3+json"
    "User-Agent" = "GH-Log-Analyzer"
}

$apiBase = "https://api.github.com/repos/$Repo"

# Only analyze daily-build-test and pr-test-npu
$workflowDefs = @(
    @{ Id = 204375811; Name = "daily-build-test" },
    @{ Id = 179736185; Name = "pr-test-npu" }
)

$ts = Get-Date -Format "yyyyMMdd-HHmmss"
$logsDir = Join-Path $OutputDir "logs\$ts"
$reportsDir = Join-Path $OutputDir "reports\$ts"
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null
New-Item -ItemType Directory -Force -Path $reportsDir | Out-Null

Write-Host "Output directory: $OutputDir" -ForegroundColor Cyan
Write-Host "Logs directory: $logsDir" -ForegroundColor Cyan
Write-Host "Reports directory: $reportsDir" -ForegroundColor Cyan

# Parse date filters
$sinceDateTime = $null
$untilDateTime = $null
if ($SinceDate -ne "") { $sinceDateTime = [DateTime]$SinceDate }
if ($UntilDate -ne "") { $untilDateTime = [DateTime]$UntilDate }

# Load previous analysis state for incremental analysis
$stateFile = Join-Path $OutputDir "analysis_state.json"
$previousState = $null
if (Test-Path $stateFile) {
    try {
        $previousState = Get-Content $stateFile -Raw | ConvertFrom-Json
        Write-Host "Loaded previous analysis state from $stateFile" -ForegroundColor Cyan
    } catch {
        Write-Host "Warning: Could not parse analysis_state.json, will perform full fetch" -ForegroundColor Yellow
    }
}

$allRuns = @()
$allJobs = @()

foreach ($wf in $workflowDefs) {
    Write-Host "=== $($wf.Name) ===" -ForegroundColor Cyan
    $validRuns = @(); $page = 1
    while ($page -le $MaxPages) {
        try {
            $uri = "$apiBase/actions/workflows/$($wf.Id)/runs?per_page=30&page=$page&status=completed"
            $resp = Invoke-RestMethod -Uri $uri -Headers $headers
        } catch { Write-Host "  Error fetching page $page`: $_" -ForegroundColor Red; break }
        foreach ($r in $resp.workflow_runs) {
            $runCreated = [DateTime]$r.created_at
            # Date filter (client-side)
            if ($sinceDateTime -and $runCreated -lt $sinceDateTime) { continue }
            if ($untilDateTime -and $runCreated -gt $untilDateTime.AddDays(1)) { continue }
            # Incremental filter: skip runs already analyzed (if no explicit date range and not forced full fetch)
            if (-not $ForceFullFetch -and $SinceDate -eq "" -and $previousState -and $previousState.workflows.$($wf.Name)) {
                $prevLatest = $previousState.workflows.$($wf.Name).latest_run_id
                if ($r.id -le $prevLatest) { continue }
            }
            if ($r.conclusion -in @("success","failure")) { $validRuns += $r }
        }
        if ($resp.workflow_runs.Count -lt 30) { break }
        $page++
    }
    $sorted = $validRuns | Sort-Object created_at -Descending
    Write-Host "  Total valid runs: $($sorted.Count)" -ForegroundColor Green

    # For daily-build-test, filter to only schedule event, take last 10
    if ($wf.Name -eq "daily-build-test") {
        $scheduleRuns = $sorted | Where-Object { $_.event -eq "schedule" }
        if ($scheduleRuns.Count -eq 0) {
            Write-Host "  Warning: No schedule event runs found, falling back to all events" -ForegroundColor Yellow
            $scheduleRuns = $sorted
        }
        $sorted = $scheduleRuns | Select-Object -First 10
        Write-Host "  Schedule runs selected: $($sorted.Count)" -ForegroundColor Cyan
    }

    # For pr-test-npu, take latest 10 runs
    if ($wf.Name -eq "pr-test-npu") {
        $sorted = $sorted | Select-Object -First 10
        Write-Host "  Latest 10 runs selected" -ForegroundColor Cyan
    }

    foreach ($run in $sorted) {
        $title = ($run.head_commit.message -split "`n")[0]
        $prNum = if ($run.pull_requests.Count -gt 0) { $run.pull_requests[0].number } else { "" }
        $runObj = [PSCustomObject]@{
            RunId=$run.id; WorkflowName=$wf.Name; Conclusion=$run.conclusion;
            Event=$run.event; CreatedAt=$run.created_at; RunStartedAt=$run.run_started_at;
            UpdatedAt=$run.updated_at; HeadBranch=$run.head_branch;
            Title=$title; PR=$prNum; HtmlUrl=$run.html_url
        }
        $allRuns += $runObj

        $runDir = Join-Path $logsDir "$($wf.Name)_run-$($run.id)"
        New-Item -ItemType Directory -Force -Path $runDir | Out-Null

        $jobs = @(); $jp = 1
        do {
            try {
                $jr = Invoke-RestMethod -Uri "$apiBase/actions/runs/$($run.id)/jobs?per_page=100&page=$jp" -Headers $headers
                $jobs += $jr.jobs; $jp++
            } catch { break }
        } while ($jr.jobs.Count -eq 100)

        $runSummary = @{ run_id=$run.id; workflow=$wf.Name; conclusion=$run.conclusion;
            total_jobs=$jobs.Count; successful=($jobs|?{$_.conclusion -eq 'success'}).Count;
            failed=($jobs|?{$_.conclusion -eq 'failure'}).Count; cancelled=($jobs|?{$_.conclusion -eq 'cancelled'}).Count }
        $runSummary | ConvertTo-Json | Out-File (Join-Path $runDir "summary.json") -Encoding UTF8

        foreach ($job in $jobs) {
            $started = if($job.started_at){[DateTime]$job.started_at}else{$null}
            $completed = if($job.completed_at){[DateTime]$job.completed_at}else{$null}
            $dur = if($started -and $completed){[Math]::Round(($completed-$started).TotalMinutes,2)}else{0}

            $jobDir = Join-Path $runDir ($job.name -replace '[\\/:*?"<>|]','_')
            New-Item -ItemType Directory -Force -Path $jobDir | Out-Null

            $failedStepNames = @()
            foreach ($step in $job.steps) {
                if ($step.conclusion -eq "failure") { $failedStepNames += $step.name }
            }

            $jobInfo = @{ id=$job.id; name=$job.name; status=$job.status; conclusion=$job.conclusion;
                started_at=$job.started_at; completed_at=$job.completed_at; duration_minutes=$dur;
                runner_name=$job.runner_name; steps_count=$job.steps.Count;
                failed_steps_count=$failedStepNames.Count; failed_steps=$failedStepNames }
            $jobInfo | ConvertTo-Json -Depth 5 | Out-File (Join-Path $jobDir "job-info.json") -Encoding UTF8

            $jobRec = [PSCustomObject]@{
                RunId=$run.id; WorkflowName=$wf.Name; RunConclusion=$run.conclusion;
                JobId=$job.id; JobName=$job.name; JobConclusion=$job.conclusion;
                StartedAt=$job.started_at; CompletedAt=$job.completed_at;
                DurationMin=$dur; RunnerName=$job.runner_name;
                StepsCount=$job.steps.Count; FailedStepsCount=$failedStepNames.Count;
                FailedStepsNames=($failedStepNames -join "; ")
            }
            $allJobs += $jobRec

            if ($job.conclusion -eq "failure") {
                try {
                    $logResp = Invoke-WebRequest -Uri "$apiBase/actions/jobs/$($job.id)/logs" -Headers $headers -UseBasicParsing -TimeoutSec 120
                    $logPath = Join-Path $jobDir "full-log.txt"
                    [IO.File]::WriteAllText($logPath, $logResp.Content, [Text.Encoding]::UTF8)
                    Write-Host "    Log saved: $logPath" -ForegroundColor Gray
                } catch {
                    $errMsg = "Log fetch failed: $($_.Exception.Message)"
                    $logPath = Join-Path $jobDir "full-log.txt"
                    [IO.File]::WriteAllText($logPath, $errMsg, [Text.Encoding]::UTF8)
                    Write-Host "    $errMsg" -ForegroundColor Red
                }
                Start-Sleep -Milliseconds 200
            }
        }
        Write-Host "  Run $($run.id): $($jobs.Count) jobs" -ForegroundColor Gray
    }
}

$allRuns | ConvertTo-Json -Depth 5 | Out-File (Join-Path $reportsDir "all_runs.json") -Encoding UTF8
$allRuns | Select-Object RunId,WorkflowName,Conclusion,Event,CreatedAt,Title,PR,HtmlUrl | Export-Csv (Join-Path $reportsDir "all_runs.csv") -NoTypeInformation -Encoding UTF8
$allJobs | ConvertTo-Json -Depth 5 | Out-File (Join-Path $reportsDir "all_jobs.json") -Encoding UTF8
$allJobs | Export-Csv (Join-Path $reportsDir "all_jobs.csv") -NoTypeInformation -Encoding UTF8

# Save analysis state for incremental analysis
$state = @{
    last_analysis_time = (Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ")
    analysis_timestamp = $ts
    run_count = $allRuns.Count
    job_count = $allJobs.Count
    since_date = $SinceDate
    until_date = $UntilDate
    workflows = @{
    }
}
foreach ($wf in $workflowDefs) {
    # For incremental analysis, track the latest run ID across all runs (including previous ones)
    $wfRuns = $allRuns | Where-Object { $_.WorkflowName -eq $wf.Name }
    if ($wfRuns) {
        $latest = $wfRuns | Sort-Object RunId -Descending | Select-Object -First 1
        $state.workflows[$wf.Name] = @{ latest_run_id = $latest.RunId }
    } elseif ($previousState -and $previousState.workflows.$($wf.Name)) {
        # Preserve previous state if no new runs in this workflow
        $state.workflows[$wf.Name] = @{ latest_run_id = $previousState.workflows.$($wf.Name).latest_run_id }
    }
}
$state | ConvertTo-Json -Depth 5 | Out-File $stateFile -Encoding UTF8

Write-Host "`n=== Data Fetch Complete ===" -ForegroundColor Green
Write-Host "Timestamp: $ts"
Write-Host "Runs: $($allRuns.Count), Jobs: $($allJobs.Count)"
Write-Host "Logs dir: $logsDir"
Write-Host "Reports dir: $reportsDir"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. python scripts\analyze_errors.py --timestamp $ts --base-dir ."
Write-Host "  2. python scripts\generate_report.py --timestamp $ts --base-dir ."
