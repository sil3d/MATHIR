# start-config-server.ps1
# Lance le serveur OpenCode Config sur port 7337 si pas déjà actif.

$Port   = 7337
$ServerScript = "$env:USERPROFILE\.config\opencode\lib\config-server.py"
$Health = "http://localhost:$Port/health"

# Déjà actif ?
try {
  Invoke-RestMethod $Health -TimeoutSec 2 -ErrorAction Stop | Out-Null
  Write-Host "  [OK] Config Server déjà actif sur port $Port" -ForegroundColor Green
  exit 0
} catch {}

# Vérification Python
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
  Write-Host "  [ERREUR] Python introuvable dans PATH" -ForegroundColor Red
  exit 1
}
if (-not (Test-Path $ServerScript)) {
  Write-Host "  [ERREUR] config-server.py introuvable : $ServerScript" -ForegroundColor Red
  exit 1
}

# Démarrage en arrière-plan (non-blocking)
$pinfo = New-Object System.Diagnostics.ProcessStartInfo
$pinfo.FileName = "python"
$pinfo.Arguments = "`"$ServerScript`""
$pinfo.UseShellExecute = $false
$pinfo.CreateNoWindow = $true
$pinfo.RedirectStandardOutput = $false
$pinfo.RedirectStandardError = $false
$p = New-Object System.Diagnostics.Process
$p.StartInfo = $pinfo
$p.Start() | Out-Null

Start-Sleep -Milliseconds 1500

# Vérification post-démarrage
try {
  Invoke-RestMethod $Health -TimeoutSec 3 -ErrorAction Stop | Out-Null
  Write-Host "  [OK] Config Server démarré sur http://localhost:$Port" -ForegroundColor Green
  Write-Host "       Logs : $env:TEMP\opencode-config-server.log" -ForegroundColor DarkGray
} catch {
  Write-Host "  [ERREUR] Le serveur n'a pas répondu. Lance manuellement :" -ForegroundColor Red
  Write-Host "           python `"$ServerScript`"" -ForegroundColor Yellow
  exit 1
}