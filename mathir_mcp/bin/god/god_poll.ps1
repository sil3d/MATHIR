# GOD POLLER — PowerShell one-shot poller for god-mode (Windows)
# Usage:
#   .\god_poll.ps1 -Mode worker -Name mimo-code -Interval 5
#   .\god_poll.ps1 -Mode orchestrator -Interval 5
#
# Press Ctrl+C to stop.

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][ValidateSet("worker","orchestrator","observer")][string]$Mode,
    [string]$Name = "",
    [string]$Daemon = "http://localhost:7338",
    [int]$Interval = 5,
    [string]$StateFile = "$HOME\.config\mathir\god_bridge_state.json"
)

$ErrorActionPreference = "Continue"
$LogFile = "$HOME\.config\mathir\god_bridge.log"
$null = New-Item -ItemType Directory -Path (Split-Path $LogFile -Parent) -Force

function Write-Log {
    param([Parameter(Position=0)][string]$Msg, [Parameter(Position=1)][string]$Level)
    if (-not $Level) { $Level = "INFO" }
    $ts = (Get-Date).ToString("o")
    $line = "[" + $ts + "] [" + $Level + "] " + $Msg
    Write-Host $line
    Add-Content -LiteralPath $LogFile -Value $line -Encoding UTF8
}

function Beep-Notify {
    [console]::Beep(800, 200)
    [console]::Beep(1000, 200)
}

function Invoke-Poll {
    param([string]$Url, [hashtable]$Body)
    try {
        $json = $Body | ConvertTo-Json -Depth 5 -Compress
        return Invoke-RestMethod -Method POST -Uri $Url -Body $json -ContentType "application/json" -TimeoutSec 5
    } catch {
        Write-Log "HTTP ERROR ${Url}: $($_.Exception.Message)" "ERROR"
        return $null
    }
}

$startMsg = "POLLER START - Mode=$Mode Name=$Name Interval=${Interval}s Daemon=$Daemon"
Write-Log -Msg $startMsg -Level INFO

if ($Mode -eq "worker" -and -not $Name) {
    Write-Log "--Name required in worker mode" "ERROR"
    exit 2
}

try {
    while ($true) {
        switch ($Mode) {
            "worker" {
                $resp = Invoke-Poll "$Daemon/api/god/poll" @{ agent = $Name; status = "pending" }
                if ($resp -and $resp.task) {
                    $label = $resp.task.label
                    $msg1 = "NEW TASK: " + $label
                    Write-Log -Msg $msg1 -Level TASK
                    $preview = $resp.task.content.Substring(0, [Math]::Min(200, $resp.task.content.Length))
                    $msg2 = "  content: " + $preview
                    Write-Log -Msg $msg2 -Level TASK
                    Beep-Notify
                }
            }
            "orchestrator" {
                $resp = Invoke-Poll "$Daemon/api/memory/audit" @{ limit = 50 }
                if ($resp -is [array]) {
                    foreach ($entry in $resp) {
                        if ($entry.label -like 'god:result:*') {
                            $lbl = $entry.label
                            $ag = $entry.agent
                            $msg1 = 'NEW RESULT: ' + $lbl
                            Write-Log -Msg $msg1 -Level RESULT
                            $msg2 = '  agent: ' + $ag
                            Write-Log -Msg $msg2 -Level RESULT
                        }
                    }
                    Beep-Notify
                }
            }
            "observer" {
                foreach ($prefix in @("god:task:","god:result:","god:reply:","god:reg:","god:shutdown:")) {
                    $resp = Invoke-Poll "$Daemon/api/memory/audit" @{ limit = 20 }
                    if ($resp -is [array]) {
                        foreach ($entry in $resp) {
                        if ($entry.label -like "${prefix}*") {
                            $kind = $prefix.TrimEnd(':')
                            $lbl = $entry.label
                            $msg = '[' + $kind + '] ' + $lbl
                            Write-Log -Msg $msg -Level OBS
                        }
                        }
                    }
                }
            }
        }
        Start-Sleep -Seconds $Interval
    }
} finally {
    Write-Log -Msg ("POLLER STOP") -Level INFO
}
