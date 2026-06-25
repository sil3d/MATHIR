# ============================================================================
# MATHIR Auto-Start Setup
# ----------------------------------------------------------------------------
# Registers the MATHIR daemon to start at every user logon. Three layers of
# defense are configured (only the ones that succeed will be active):
#
#   1. HKCU\Software\Microsoft\Windows\CurrentVersion\Run registry key
#        - NO admin rights required
#        - Fires at every user logon
#        - Primary method
#
#   2. .vbs in the Startup folder
#        - NO admin rights required
#        - Same effect as the registry key, but visible in Explorer
#        - Belt-and-suspenders fallback for the registry key
#
#   3. Task Scheduler
#        - REQUIRES admin rights
#        - More powerful (can run with -RunLevel Highest, restart on failure)
#        - Attempted last; silently skipped if user lacks elevation
#
# Optional: install a healthcheck on a schedule (recommended).
# ============================================================================

[CmdletBinding()]
param(
    [switch]$SkipTaskScheduler,
    [switch]$InstallHealthcheck
)

$ErrorActionPreference = 'Stop'

# ---- Configuration ---------------------------------------------------------
$BinDir          = Join-Path $env:USERPROFILE '.config\opencode\bin'
$VbsSource       = Join-Path $BinDir 'auto_start_vbs.vbs'
$StartupFolder   = Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs\Startup'
$VbsDest         = Join-Path $StartupFolder 'mathir_daemon.vbs'
$RegPath         = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Run'
$RegName         = 'MATHIR_Daemon'
$TaskName        = 'MATHIR_Daemon_AutoStart'
$PythonPath      = (Get-Command python).Source
$DaemonPath      = Join-Path $BinDir 'mathir_daemon.py'
$HealthcheckPath = Join-Path $BinDir 'auto_start_healthcheck.ps1'

function Write-Status {
    param([string]$Msg, [string]$Color = 'Cyan')
    Write-Host $Msg -ForegroundColor $Color
}

Write-Status '' 'White'
Write-Status '========================================' 'White'
Write-Status '  MATHIR Daemon Auto-Start Setup'        'White'
Write-Status '========================================' 'White'
Write-Status "User:      $env:USERNAME"
Write-Status "BinDir:    $BinDir"
Write-Status "Python:    $PythonPath"
Write-Status '' 'White'

# ============================================================================
# 1) Registry: HKCU\...\Run  (no admin needed)
# ============================================================================
Write-Status '[1/3] Registering HKCU Run key...' 'Yellow'
try {
    if (-not (Test-Path $RegPath)) {
        New-Item -Path $RegPath -Force | Out-Null
    }
    # The command must be wrapped in quotes because paths contain spaces.
    # Using wscript.exe on the .vbs keeps it hidden (no console flash).
    $regValue = "`"C:\Windows\System32\wscript.exe`" `"$VbsDest`""
    New-ItemProperty -LiteralPath $RegPath -Name $RegName -Value $regValue -PropertyType String -Force | Out-Null
    Write-Status "  OK: $RegPath\$RegName = $regValue" 'Green'
} catch {
    Write-Status "  FAILED: $_" 'Red'
    Write-Status "  (Registry is the preferred method. Fix this before continuing.)" 'Red'
}

# ============================================================================
# 2) Startup folder: copy the .vbs here
# ============================================================================
Write-Status '' 'White'
Write-Status '[2/3] Copying mathir_daemon.vbs to Startup folder...' 'Yellow'
try {
    if (-not (Test-Path $StartupFolder)) {
        New-Item -ItemType Directory -Path $StartupFolder -Force | Out-Null
    }
    if (-not (Test-Path $VbsSource)) {
        throw "Source VBS not found: $VbsSource"
    }
    Copy-Item -LiteralPath $VbsSource -Destination $VbsDest -Force
    Write-Status "  OK: $VbsDest" 'Green'
} catch {
    Write-Status "  FAILED: $_" 'Red'
}

# ============================================================================
# 3) Task Scheduler:  only if user is admin and -SkipTaskScheduler not set
# ============================================================================
Write-Status '' 'White'
Write-Status '[3/3] Registering Task Scheduler (admin only)...' 'Yellow'
if ($SkipTaskScheduler) {
    Write-Status '  SKIPPED (-SkipTaskScheduler specified)' 'Gray'
} else {
    $isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
    if (-not $isAdmin) {
        Write-Status '  SKIPPED (not running as Administrator)' 'Gray'
        Write-Status '  To enable, re-run this script from an elevated PowerShell.' 'Gray'
    } else {
        try {
            Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
            $action  = New-ScheduledTaskAction -Execute $PythonPath -Argument "`"$DaemonPath`"" -WorkingDirectory $BinDir
            $trigger = New-ScheduledTaskTrigger -AtLogOn
            $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
            $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Highest
            Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Description 'MATHIR cognitive memory daemon (auto-start on logon, auto-restart on failure)' | Out-Null
            Write-Status "  OK: Task '$TaskName' registered (logon + restart on failure)" 'Green'
        } catch {
            Write-Status "  FAILED: $_" 'Red'
        }
    }
}

# ============================================================================
# 4) Optional: healthcheck on a schedule (Task Scheduler, every 5 minutes)
# ============================================================================
if ($InstallHealthcheck) {
    Write-Status '' 'White'
    Write-Status '[opt] Installing healthcheck (every 5 minutes)...' 'Yellow'
    $isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
    if (-not $isAdmin) {
        Write-Status '  SKIPPED (not running as Administrator)' 'Gray'
        Write-Status "  To enable manually run this command as admin:" 'Gray'
        Write-Status "    Register-ScheduledTask -TaskName MATHIR_Daemon_Healthcheck -Action (New-ScheduledTaskAction -Execute 'powershell.exe' -Argument '-NoProfile -ExecutionPolicy Bypass -File `"$HealthcheckPath`" -Quiet' -WorkingDirectory '$BinDir') -Trigger (New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 5) -RepetitionDuration (New-TimeSpan -Days 3650)) -Settings (New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable) -Principal (New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Highest)" 'Gray'
    } else {
        try {
            $hcName = 'MATHIR_Daemon_Healthcheck'
            Unregister-ScheduledTask -TaskName $hcName -Confirm:$false -ErrorAction SilentlyContinue
            $hcAction  = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$HealthcheckPath`" -Quiet" -WorkingDirectory $BinDir
            $hcTrigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 5) -RepetitionDuration (New-TimeSpan -Days 3650)
            $hcSettings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
            $hcPrincipal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Highest
            Register-ScheduledTask -TaskName $hcName -Action $hcAction -Trigger $hcTrigger -Settings $hcSettings -Principal $hcPrincipal -Description 'MATHIR daemon healthcheck -- restarts daemon if port 7338 is not responding' | Out-Null
            Write-Status "  OK: Task '$hcName' registered (every 5 minutes)" 'Green'
        } catch {
            Write-Status "  FAILED: $_" 'Red'
        }
    }
}

Write-Status '' 'White'
Write-Status '========================================' 'Green'
Write-Status '  Setup complete.' 'Green'
Write-Status '========================================' 'Green'
Write-Status ''
Write-Status "To start now:        & '$BinDir\auto_start.bat'" 'White'
Write-Status "To check health:     & '$HealthcheckPath'" 'White'
Write-Status "To uninstall:        Remove-ItemProperty -Path '$RegPath' -Name '$RegName'" 'White'
Write-Status "                      Remove-Item '$VbsDest' -Force" 'White'
Write-Status "                      Unregister-ScheduledTask -TaskName '$TaskName' -Confirm:`$false" 'White'
