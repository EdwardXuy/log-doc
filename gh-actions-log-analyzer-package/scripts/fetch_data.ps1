param(
    [Parameter(Mandatory=$true)]
    [string]$Token,
    [string]$Repo = "sgl-project/sgl-kernel-npu",
    [string]$OutputDir = ".\analysis",
    [int]$MaxPages = 5
)

$headers = @{
    "Authorization" = "token $Token"
    "Accept" = "application/vnd.github.v3+json"
    "User-Agent" = "GH-Log-Analyzer"
}

$apiBase = "https://api.github.com/repos/$Repo"

$workflowDefs = @(
    @{ Id = 213514694; Name = "build_and_release" },
    @{ Id = 204375811; Name = "daily-build-test" },
    @{ Id = 179736185; Name = "pr-test-npu" }
)

$ts = Get-Date -Format "yyyyMMdd-HHmmss"
$logsDir = Join-Path $OutputDir "logs\$ts"
$reportsDir = Join-Path $OutputDir "reports\$ts"
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null
New-Item -ItemType Directory -Force -Path $reportsDir | Out-Null

$allRuns = @()
$allJobs = @()

foreach ($wf in $workflowDefs) {
    Write-Host "=== $($wf.Name) ===" -ForegroundColor Cyan
    $validRuns = @(); $page = 1
    while ($page -le $MaxPages) {
        try {
            $resp = Invoke-RestMethod -Uri "$apiBase/actions/workflows/$($wf.Id)/runs?per_page=30&page=$page&status=completed" -Headers $headers
        } catch { break }
        foreach ($r in $resp.workflow_runs) {
            if ($r.conclusion -in @("success","failure")) { $validRuns += $r }
        }
        $page++
    }
    $sorted = $validRuns | Sort-Object created_at -Descending
    Write-Host "  Total valid runs: $($sorted.Count)"

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
                    [IO.File]::WriteAllText((Join-Path $jobDir "full-log.txt"), $logResp.Content, [Text.Encoding]::UTF8)
                } catch {
                    "Log fetch failed: $($_.Exception.Message)" | Out-File (Join-Path $jobDir "full-log.txt") -Encoding UTF8
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

Write-Host "`n=== Data Fetch Complete ===" -ForegroundColor Green
Write-Host "Timestamp: $ts"
Write-Host "Runs: $($allRuns.Count), Jobs: $($allJobs.Count)"
Write-Host "Logs dir: $logsDir"
Write-Host "Reports dir: $reportsDir"
Write-Host ""
Write-Host "Next step: python scripts/generate_report.py --timestamp $ts"
