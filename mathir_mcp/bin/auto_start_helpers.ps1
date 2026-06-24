# ============================================================================
# MATHIR Daemon Auto-Start Helper (PowerShell)
# ----------------------------------------------------------------------------
# Robust starter with retry logic and port-health verification. This is the
# canonical entry point when running from PowerShell, Task Scheduler, or any
# environment where the .bat is invoked with `& auto_start.bat`.
#
# Why a separate helper?
#   The .bat file (auto_start.bat) intentionally contains NO embedded
#   `powershell -Command` blocks because PowerShell's `&` call operator
#   misinterprets `>>` and `2>&1` redirection in the .bat body as
#   outer-level redirection, producing:
#       "input redirection is not supported" (exit 255)
#   This helper is invoked via `powershell -File` (which is well-behaved)
#   or directly, and contains all the intelligence.
#
# Usage:
#   .\auto_start_helpers.ps1                            # start + verify
#   .\auto_start_helpers.ps1 -CheckOnly                 # port check only
#   .\auto_start_helpers.ps1 -Action Check -Port 7338   # used by .bat
#   .\auto_start_helpers.ps1 -Action Kill  -Port 7338   # used by .bat
#
# Exit codes:
#   0 = success (daemon listening on port)
#   1 = daemon failed to start after MaxRetries
#   2 = python or daemon script not found
# ============================================================================

[CmdletBinding()]
param(
    [ValidateSet('Start','Check','Kill','Restart')]
    [string]$Action = 'Start',

    [int]$Port = 7338,

    [int]$MaxRetries = 3,
    [int]$WaitSeconds = 3,

    [string]$PythonPath = "$env:USERPROFILE\AppData\Local\Programs\Python\Python311\python.exe",
    [string]$BinDir     = "$env:USERPROFILE\.config\opencode\bin",
    [string]$DaemonPath = "$env:USERPROFILE\.config\opencode\bin\mathir_daemon.py",
    [string]$StatsPath  = "$env:USERPROFILE\.config\opencode\bin\mathir_stats_server.py",
    [int]$StatsPort     = 7420,
    [string]$LogPath    = "$env:USERPROFILE\.config\opencode\bin\mathir_daemon.log",

    [switch]$CheckOnly,
    [switch]$Quiet
)

$ErrorActionPreference = 'Stop'

# ---- Logging ---------------------------------------------------------------
function Write-Log {
    param(
        [Parameter(Mandatory)][string]$Level,
        [Parameter(Mandatory)][string]$Message
    )
    $ts = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
    $line = "[$ts] [$Level] $Message"
    try {
        Add-Content -LiteralPath $LogPath -Value $line -Encoding UTF8 -ErrorAction SilentlyContinue
    } catch { }
    if (-not $Quiet) {
        switch ($Level) {
            'OK'    { Write-Host $line -ForegroundColor Green }
            'WARN'  { Write-Host $line -ForegroundColor Yellow }
            'ERROR' { Write-Host $line -ForegroundColor Red }
            'FATAL' { Write-Host $line -ForegroundColor Magenta }
            default { Write-Host $line }
        }
    }
}

# Ensure log directory exists
$logDir = Split-Path $LogPath -Parent
if ($logDir -and -not (Test-Path $logDir)) {
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

# ---- Kill process bound to port -------------------------------------------
function Stop-ProcessOnPort {
    param([int]$PortToKill)
    try {
        $owner = Get-NetTCPConnection -LocalPort $PortToKill -State Listen -ErrorAction SilentlyContinue |
                 Select-Object -First 1 -ExpandProperty OwningProcess
        if ($owner) {
            Stop-Process -Id $owner -Force -ErrorAction SilentlyContinue
            Write-Log 'INFO' "Killed PID $owner (was bound to port $PortToKill)"
        }
    } catch {
        Write-Log 'WARN' "Stop-ProcessOnPort: $_"
    }
}

# ---- Action: Check ---------------------------------------------------------
if ($Action -eq 'Check') {
    if (Test-DaemonPort -PortToTest $Port) { exit 0 } else { exit 1 }
}

# ---- Action: Kill ----------------------------------------------------------
if ($Action -eq 'Kill') {
    Stop-ProcessOnPort -PortToKill $Port
    exit 0
}

# ---- Action: Restart -------------------------------------------------------
if ($Action -eq 'Restart') {
    Write-Log 'INFO' "Restart requested for port $Port"
    Stop-ProcessOnPort -PortToKill $Port
    Start-Sleep -Seconds 1
    $Action = 'Start'
}

# ---- Action: Start (default) ----------------------------------------------
Write-Log 'INFO' "=========================================="
Write-Log 'INFO' "auto_start_helpers.ps1 invoked (PID $PID, user $env:USERNAME)"
Write-Log 'INFO' "Python:    $PythonPath"
Write-Log 'INFO' "Daemon:    $DaemonPath"
Write-Log 'INFO' "Port:      $Port"
Write-Log 'INFO' "Max tries: $MaxRetries (wait ${WaitSeconds}s between)"

# Sanity checks
if (-not (Test-Path $PythonPath)) {
    Write-Log 'FATAL' "Python not found at: $PythonPath"
    exit 2
}
if (-not (Test-Path $DaemonPath)) {
    Write-Log 'FATAL' "Daemon script not found at: $DaemonPath"
    exit 2
}

# Already running?
if (Test-DaemonPort -PortToTest $Port) {
    Write-Log 'OK' "Daemon already listening on port $Port, nothing to do."
    exit 0
}

if ($CheckOnly) {
    Write-Log 'WARN' "Port $Port is NOT open and -CheckOnly specified, exiting."
    exit 1
}

Write-Log 'WARN' "Port $Port is NOT open, starting daemon..."

# Retry loop
$attempt = 0
while ($attempt -lt $MaxRetries) {
    $attempt++
    Write-Log 'INFO' "Attempt $attempt/$MaxRetries`: launching daemon..."

    try {
        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName               = $PythonPath
        $psi.Arguments              = "`"$DaemonPath`""
        # WorkingDirectory MUST be the user home (or project root), NOT $BinDir.
        # The daemon resolves its DB path from CWD via .mathir/ — if CWD is bin/,
        # it creates bin/.mathir/ which has no project context, and worse, may
        # be read-only or write into a wrong dir. Use USERPROFILE so it falls
        # back to ~/.mathir/mathir_global/ when no .mathir exists in CWD.
        $psi.WorkingDirectory       = $env:USERPROFILE
        $psi.UseShellExecute        = $false
        $psi.CreateNoWindow         = $true
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError  = $true
        $psi.StandardOutputEncoding = [System.Text.Encoding]::UTF8
        $psi.StandardErrorEncoding  = [System.Text.Encoding]::UTF8

        $proc = [System.Diagnostics.Process]::Start($psi)
        # Don't WaitForExit — daemon runs forever. Just give it a moment.
        Start-Sleep -Milliseconds 500
        Write-Log 'INFO' "Launched daemon (PID $($proc.Id) assigned by OS)"

        # Async tail stdout/stderr into the log
        Register-ObjectEvent -InputObject $proc -EventName 'OutputDataReceived' -Action {
            if ($EventArgs.Data) { Add-Content -LiteralPath $using:LogPath -Value $EventArgs.Data -Encoding UTF8 }
        } | Out-Null
        Register-ObjectEvent -InputObject $proc -EventName 'ErrorDataReceived' -Action {
            if ($EventArgs.Data) { Add-Content -LiteralPath $using:LogPath -Value "[stderr] $($EventArgs.Data)" -Encoding UTF8 }
        } | Out-Null
        $proc.BeginOutputReadLine()
        $proc.BeginErrorReadLine()
    } catch {
        Write-Log 'ERROR' "Failed to launch daemon: $_"
    }

    # Wait for the embedder to load
    Start-Sleep -Seconds $WaitSeconds

    if (Test-DaemonPort -PortToTest $Port) {
        Write-Log 'OK' "SUCCESS: daemon is listening on port $Port (attempt $attempt)."

        # Also start the stats server (dashboard backend) if present
        if (Test-Path $StatsPath) {
            if (-not (Test-DaemonPort -PortToTest $StatsPort)) {
                Write-Log 'INFO' "Starting stats server (dashboard backend on port $StatsPort)..."
                $psiS = New-Object System.Diagnostics.ProcessStartInfo
                $psiS.FileName               = $PythonPath
                $psiS.Arguments              = "`"$StatsPath`""
                # Same as daemon: use USERPROFILE not BinDir (see comment above).
                $psiS.WorkingDirectory       = $env:USERPROFILE
                $psiS.UseShellExecute        = $false
                $psiS.CreateNoWindow         = $true
                $psiS.RedirectStandardOutput = $true
                $psiS.RedirectStandardError  = $true
                $psiS.StandardOutputEncoding = [System.Text.Encoding]::UTF8
                $psiS.StandardErrorEncoding  = [System.Text.Encoding]::UTF8
                $procS = [System.Diagnostics.Process]::Start($psiS)
                Start-Sleep -Seconds 3
                if (Test-DaemonPort -PortToTest $StatsPort) {
                    Write-Log 'OK' "Stats server is listening on port $StatsPort (PID $($procS.Id))"
                } else {
                    Write-Log 'WARN' "Stats server did not bind to port $StatsPort within 3s"
                }
            } else {
                Write-Log 'OK' "Stats server already listening on port $StatsPort"
            }
        }

        exit 0
    }

    Write-Log 'WARN' "Attempt $attempt failed -- port $Port still closed."

    if ($attempt -lt $MaxRetries) {
        Stop-ProcessOnPort -PortToKill $Port
        Start-Sleep -Seconds 1
    }
}

# All retries exhausted
Write-Log 'FATAL' "Daemon failed to start after $MaxRetries attempts."
Write-Log 'INFO' "Last 30 lines of log for diagnosis:"
if (Test-Path $LogPath) {
    Get-Content -LiteralPath $LogPath -Tail 30 -ErrorAction SilentlyContinue | ForEach-Object {
        Write-Log 'INFO' "[log] $_"
    }
}
exit 1
