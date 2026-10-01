param(
    [Parameter(Mandatory = $true)][string]$Name,
    [Parameter(Mandatory = $true)][string]$ArgList,
    [string]$Exe = "",
    [int]$Every = 30
)
# Runs one command from the project root and logs it the way long runs are logged here:
#   logs/<Name>.log      START / END lines, exit code, elapsed minutes, peak private MB
#   logs/<Name>.mem.csv  memory of the whole process tree every $Every seconds
#   logs/<Name>.out/.err the command's stdout / stderr
# The venv's python.exe is a launcher that spawns the real interpreter, so the whole
# tree is measured. MIMALLOC_PURGE_DELAY=0 is set for the child (see CLAUDE.md).
#
#   powershell -File scripts\run_logged.ps1 -Name suite -ArgList "-m pytest -q"
#   powershell -File scripts\run_logged.ps1 -Name ens5m -ArgList "-m xauusd_quant.cli ensemble-research -t 5m"
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if (-not $Exe) { $Exe = Join-Path $root ".venv\Scripts\python.exe" }
$log = Join-Path $root "logs\$Name"
$utf8 = New-Object Text.UTF8Encoding($false)
[IO.File]::WriteAllText("$log.log", "$(Get-Date -Format s) START $Exe $ArgList`n", $utf8)
[IO.File]::WriteAllText("$log.mem.csv", "time,tree_ws_mb,tree_private_mb,procs,free_phys_mb,free_commit_mb`n", $utf8)
$env:MIMALLOC_PURGE_DELAY = "0"
$p = Start-Process -FilePath $Exe -ArgumentList $ArgList -WorkingDirectory $root `
    -RedirectStandardOutput "$log.out" -RedirectStandardError "$log.err" -PassThru -NoNewWindow
$null = $p.Handle          # keeps ExitCode readable after the process exits
[IO.File]::AppendAllText("$log.log", "root pid $($p.Id)`n", $utf8)
$peak = 0
$started = Get-Date
while (-not $p.HasExited) {
    try {
        $all = @(Get-CimInstance Win32_Process)
        $ids = New-Object 'System.Collections.Generic.HashSet[int]'
        [void]$ids.Add($p.Id)
        $changed = $true
        while ($changed) {
            $changed = $false
            foreach ($q in $all) {
                if ($ids.Contains([int]$q.ParentProcessId) -and -not $ids.Contains([int]$q.ProcessId)) {
                    [void]$ids.Add([int]$q.ProcessId); $changed = $true
                }
            }
        }
        $tree = @($all | Where-Object { $ids.Contains([int]$_.ProcessId) })
        $ws = [int](($tree | Measure-Object WorkingSetSize -Sum).Sum / 1MB)
        $priv = [int](($tree | Measure-Object PrivatePageCount -Sum).Sum / 1MB)
        if ($priv -gt $peak) { $peak = $priv }
        $os = Get-CimInstance Win32_OperatingSystem
        $line = "{0},{1},{2},{3},{4},{5}`n" -f (Get-Date -Format HH:mm:ss), $ws, $priv, $tree.Count, `
            [int]($os.FreePhysicalMemory / 1KB), [int]($os.FreeVirtualMemory / 1KB)
        [IO.File]::AppendAllText("$log.mem.csv", $line, $utf8)
    } catch { }
    Start-Sleep -Seconds $Every
}
$p.WaitForExit()
$elapsed = ((Get-Date) - $started).TotalMinutes
[IO.File]::AppendAllText("$log.log", ("{0} END code={1} elapsed={2:N1}min peakPrivateMB={3}`n" -f `
    (Get-Date -Format s), $p.ExitCode, $elapsed, $peak), $utf8)
exit $p.ExitCode
