# Run the Prompt #7 regime grid in resumable steps, one CLI call at a time.
#
#   powershell -File scripts\regime_grid.ps1 -Timeframes 1h,30m,15m
#   powershell -File scripts\regime_grid.ps1 -Timeframes 5m -Heavy
#
# Every step writes its results as it finishes and --resume reuses current ones,
# so a stopped run loses at most the step in progress. Each step's wall time and
# the peak working set of all python.exe processes are appended to
# logs\regime_grid.log. Steps per timeframe:
#   kmeans    offline + expanding quarterly walk-forward + analysis
#   gmm_diag  offline only (the covariance-structure comparison, Step 12)
#   gmm       offline + expanding quarterly walk-forward + analysis
#   hmm       offline + expanding and rolling quarterly walk-forward, monthly for
#             causal.monthly_states, nulls, analysis (+ pipeline-null increments)
# -Heavy (5m): the full GMM offline only; the HMM runs every K expanding-only;
# rolling refits for K = 2 and 3, pipeline-null increments for K = 3; no monthly refits.
param(
    [string[]]$Timeframes = @("1h", "30m", "15m"),
    [switch]$Heavy
)
$Timeframes = @($Timeframes | ForEach-Object { $_ -split "," } | Where-Object { $_ })  # -File passes "a,b" as one string
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
$log = Join-Path $root "logs\regime_grid.log"

function Invoke-Step([string[]]$arguments) {
    $start = Get-Date
    "$($start.ToString('s')) START xq $($arguments -join ' ')" | Add-Content $log
    $out = Join-Path $root "logs\regime_step.out"
    $err = Join-Path $root "logs\regime_step.err"
    $proc = Start-Process -FilePath $python -ArgumentList (@("-m", "xauusd_quant.cli") + $arguments) `
        -WorkingDirectory $root -PassThru -NoNewWindow -RedirectStandardOutput $out -RedirectStandardError $err
    $peak = 0
    while (-not $proc.HasExited) {
        Start-Sleep -Seconds 5
        $ws = (Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Measure-Object WorkingSetSize -Sum).Sum
        if ($ws -gt $peak) { $peak = $ws }
    }
    $proc.WaitForExit()
    $elapsed = (Get-Date) - $start
    "$((Get-Date).ToString('s')) END code=$($proc.ExitCode) elapsed=$([math]::Round($elapsed.TotalMinutes,1))min peakMB=$([math]::Round($peak/1MB))" | Add-Content $log
    Get-Content $out -Tail 12 | Add-Content $log
}

foreach ($tf in $Timeframes) {
    Invoke-Step @("regime-research", "--timeframe", $tf, "--model", "kmeans", "--schemes", "expanding_quarterly", "--resume")
    Invoke-Step @("regime-research", "--timeframe", $tf, "--model", "gmm_diag", "--stages", "offline", "--resume")
    if ($Heavy) {    # 5m: a GMM walk-forward alone would take ~5 h; offline only
        Invoke-Step @("regime-research", "--timeframe", $tf, "--model", "gmm", "--stages", "offline", "--resume")
    } else {
        Invoke-Step @("regime-research", "--timeframe", $tf, "--model", "gmm", "--schemes", "expanding_quarterly", "--resume")
    }
    if ($Heavy) {
        Invoke-Step @("regime-research", "--timeframe", $tf, "--model", "hmm", "--schemes", "expanding_quarterly", "--resume", "--null-increment-states", "3")
        foreach ($k in 2, 3) {
            Invoke-Step @("regime-research", "--timeframe", $tf, "--model", "hmm", "--states", "$k", "--stages", "walk_forward,analysis", "--schemes", "expanding_quarterly,rolling_quarterly", "--resume", "--null-increment-states", "3")
        }
    } else {
        Invoke-Step @("regime-research", "--timeframe", $tf, "--model", "hmm", "--resume", "--monthly")
    }
}
"$((Get-Date).ToString('s')) GRID DONE $($Timeframes -join ',')" | Add-Content $log
