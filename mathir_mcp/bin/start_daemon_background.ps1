# ============================================================================
# MATHIR Daemon — Non-Blocking Starter (PowerShell)
# ----------------------------------------------------------------------------
# Same as auto_start_helpers.ps1 but returns IMMEDIATELY after launching the
# daemon. The actual readiness check runs in the background; the PID is
# written to a file for later inspection.
#
# Why?
#   The standard helper polls the port with Test-NetConnection in a loop,
#   which blocks the calling shell for 5-20 seconds. When called from an AI
#   agent or a CI step, this looks like "the shell froze" to the user.
#
# Usage:
#   .\start_daemon_background.ps1              # launch + return immediately
#   .\start_daemon_background.ps1 -Wait        # block until port 7338 is up
#   .\start_daemon_background.ps1 -Verify      # just check the port and report
#
# Exit codes:
#   0 = daemon was launched (or already running)
#   1 = python or daemon script not found
# ============================================================================

[CmdletBinding()]
param(
    [ValidateSet('Start','Verify')]
    [string]$Action = 'Start',

    [switch]$Wait,
    [int]$WaitTimeout = 60,

    [string]$PythonPath = "$env:USERPROFILE\AppData\Local\Programs\Python\Python311\python.exe",
    [string]$BinDir     = "$env:USERPROFILE\.config\opencode\bin",
    [string]$DaemonPath = "$env:USERPROFILE\.config\opencode\bin\mathir_daemon.py",
    [string]$StatsPath  = "$env:USERPROFILE\.config\opencode\bin\mathir_stats_server.py",
    [int]$DaemonPort    = 7338,
    [int]$StatsPort     = 7420,
    [string]$LogPath    = "$env:USERPROFILE\.config\opencode\bin\mathir_daemon.log",
    [string]$StatsLogPath = "$env:USERPROFILE\.config\opencode\bin\mathir_stats_server.log",
    [string]$PidFile    = "$env:USERPROFILE\.config\opencode\bin\mathir_daemon.pid"
)

$ErrorActionPreference = 'Stop'

function Test-PortQuick {
    param([int]$PortToTest, [int]$TimeoutMs = 800)
    try {
        $client = New-Object System.Net.Sockets.TcpClient
        $iar    = $client.BeginConnect('127.0.0.1', $PortToTest, $null, $null)
        $ok     = $iar.AsyncWaitHandle.WaitOne($TimeoutMs, $false)
        if ($ok) { $client.EndConnect($iar); $client.Close(); return $true }
        $client.Close(); return $false
    } catch { return $false }
}

# Sanity
if (-not (Test-Path $PythonPath)) { Write-Error "Python not found: $PythonPath"; exit 1 }
if (-not (Test-Path $DaemonPath)) { Write-Error "Daemon script not found: $DaemonPath"; exit 1 }

# Verify mode
if ($Action -eq 'Verify') {
    $daemonUp  = Test-PortQuick -PortToTest $DaemonPort
    $statsUp   = Test-PortQuick -PortToTest $StatsPort
    Write-Host "Daemon (port $DaemonPort): $(if ($daemonUp) {'UP'} else {'DOWN'})"
    Write-Host "Stats  (port $StatsPort): $(if ($statsUp)  {'UP'} else {'DOWN'})"
    exit 0
}

# If already running, just report
if (Test-PortQuick -PortToTest $DaemonPort) {
    Write-Host "Daemon already listening on port $DaemonPort"
    exit 0
}

# Launch detached (true background — no parent shell blocking).
#
# IMPORTANT: do NOT redirect stdout/stderr here. With RedirectStandardOutput=True,
# the .NET Process object needs a live reader (BeginOutputReadLine + event handler
# that runs in the calling PowerShell process). When the script exits, the event
# handlers die, the pipe stops being drained, the daemon's stdout buffer fills,
# and the daemon crashes. We instead use CreateNoWindow=True with no redirect —
# the daemon's output goes to its own invisible console (no buffer issue).
Write-Host "Launching daemon (detached)..."
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName               = $PythonPath
$psi.Arguments              = "`"$DaemonPath`""
$psi.WorkingDirectory       = $env:USERPROFILE
$psi.UseShellExecute        = $false
$psi.CreateNoWindow         = $true
# No RedirectStandardOutput / RedirectStandardError — see comment above.

$proc = [System.Diagnostics.Process]::Start($psi)
Set-Content -LiteralPath $PidFile -Value $proc.Id -Encoding UTF8 -Force
Write-Host "Daemon launched: PID $($proc.Id), PID file: $PidFile"

# Optional stats server (also detached)
if ((Test-Path $StatsPath) -and -not (Test-PortQuick -PortToTest $StatsPort)) {
    $psiS = New-Object System.Diagnostics.ProcessStartInfo
    $psiS.FileName               = $PythonPath
    $psiS.Arguments              = "`"$StatsPath`""
    $psiS.WorkingDirectory       = $env:USERPROFILE
    $psiS.UseShellExecute        = $false
    $psiS.CreateNoWindow         = $true
    # No redirect — see comment above.
    $procS = [System.Diagnostics.Process]::Start($psiS)
    Write-Host "Stats server launched: PID $($procS.Id)"
}

# Optional blocking wait
if ($Wait) {
    $deadline = (Get-Date).AddSeconds($WaitTimeout)
    while ((Get-Date) -lt $deadline) {
        if (Test-PortQuick -PortToTest $DaemonPort) {
            Write-Host "Daemon ready on port $DaemonPort"
            exit 0
        }
        Start-Sleep -Seconds 1
    }
    Write-Warning "Daemon did not become ready within $WaitTimeout seconds (still launching in background)"
    exit 0  # don't fail — process is launched, just not ready yet
}

Write-Host "(Non-blocking: daemon is launching in the background. Use -Wait to block until ready.)"
exit 0
