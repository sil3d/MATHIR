# ============================================================================
# MATHIR Daemon Healthcheck
# ----------------------------------------------------------------------------
# Verifies the MATHIR daemon is listening on port 7338. If not, launches the
# robust auto_start.bat which will retry up to 3 times.
#
# Usage:
#   .\auto_start_healthcheck.ps1                    # check + auto-restart
#   .\auto_start_healthcheck.ps1 -CheckOnly         # just check, don't start
#   .\auto_start_healthcheck.ps1 -Quiet             # no console output (cron mode)
#
# Can be run manually or via Task Scheduler with -RunLevel Highest.
# Suggested schedule: every 5 minutes (Task Scheduler -> Trigger -> New...).
# ============================================================================

[CmdletBinding()]
param(
    [switch]$CheckOnly,
    [switch]$Quiet
)

$ErrorActionPreference = 'Stop'

# ---- Configuration ---------------------------------------------------------
$Port          = 7338
$BinDir        = Join-Path $env:USERPROFILE '.config\opencode\bin'
$LogPath       = Join-Path $BinDir 'mathir_healthcheck.log'
$AutoStartBat  = Join-Path $BinDir 'auto_start.bat'

# ---- Logging ---------------------------------------------------------------
function Write-HealthLog {
    param([string]$Level, [string]$Message)
    $ts = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
    $line = "[$ts] [$Level] $Message"
    Add-Content -LiteralPath $LogPath -Value $line -Encoding UTF8
    if (-not $Quiet) {
        switch ($Level) {
            'INFO'  { Write-Host    $line }
            'WARN'  { Write-Host    $line -ForegroundColor Yellow }
            'ERROR' { Write-Host    $line -ForegroundColor Red }
            'OK'    { Write-Host    $line -ForegroundColor Green }
        }
    }
}

# Ensure log directory exists
$logDir = Split-Path $LogPath -Parent
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}

# ---- Port check ------------------------------------------------------------
function Test-DaemonPort {
    param([int]$PortToTest, [int]$TimeoutMs = 1500)
    try {
        $client = New-Object System.Net.Sockets.TcpClient
        $iar    = $client.BeginConnect('127.0.0.1', $PortToTest, $null, $null)
        $ok     = $iar.AsyncWaitHandle.WaitOne($TimeoutMs, $false)
        if ($ok) {
            $client.EndConnect($iar)
            $client.Close()
            return $true
        }
        $client.Close()
        return $false
    } catch {
        return $false
    }
}

# ---- Main ------------------------------------------------------------------
Write-HealthLog 'INFO' "Healthcheck started (PID $PID, user $env:USERNAME)"

if (Test-DaemonPort -PortToTest $Port) {
    Write-HealthLog 'OK' "Daemon is healthy (port $Port open)"
    exit 0
}

Write-HealthLog 'WARN' "Port $Port is NOT open -- daemon is down or starting up."

if ($CheckOnly) {
    Write-HealthLog 'WARN' '-CheckOnly specified, not starting daemon.'
    exit 1
}

# Verify auto_start.bat exists
if (-not (Test-Path $AutoStartBat)) {
    Write-HealthLog 'ERROR' "auto_start.bat not found at $AutoStartBat"
    exit 2
}

Write-HealthLog 'INFO' "Launching $AutoStartBat ..."
try {
    # Start the bat in a hidden window. Capture exit code.
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName               = $AutoStartBat
    $psi.UseShellExecute        = $false
    $psi.CreateNoWindow         = $true
    $psi.RedirectStandardOutput = $false
    $psi.RedirectStandardError  = $false
    $psi.WorkingDirectory       = $BinDir

    $proc = [System.Diagnostics.Process]::Start($psi)
    $proc.WaitForExit() | Out-Null
    $exitCode = $proc.ExitCode
} catch {
    Write-HealthLog 'ERROR' "Failed to launch auto_start.bat: $_"
    exit 2
}

Write-HealthLog 'INFO' "auto_start.bat exited with code $exitCode"

# Final verification
if (Test-DaemonPort -PortToTest $Port) {
    Write-HealthLog 'OK' "Daemon is healthy after restart (port $Port open)"
    exit 0
} else {
    Write-HealthLog 'ERROR' "Daemon STILL not listening on port $Port after restart attempt"
    exit 1
}
