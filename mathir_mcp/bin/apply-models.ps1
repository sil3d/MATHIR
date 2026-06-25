# apply-models.ps1
# Lit model-assignments.json et INSÈRE ou MET À JOUR model: dans chaque agent .md
# Usage: powershell -File "$env:USERPROFILE\.config\opencode\apply-models.ps1"

$ConfigDir    = "$env:USERPROFILE\.config\opencode"
$AgentsDir    = "$ConfigDir\agents"
$DataDir      = "$ConfigDir\data"
$JsonFile     = "$DataDir\model-assignments.json"
$OpenCodeJson = "$ConfigDir\opencode.json"

# ── Vérification ─────────────────────────────────────────
if (-not (Test-Path $JsonFile)) {
    Write-Host ""
    Write-Host "  [ERREUR] model-assignments.json introuvable dans $ConfigDir" -ForegroundColor Red
    Write-Host "           Ouvre lib/config.html dans Chrome et clique Sauvegarder JSON" -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

$config  = Get-Content $JsonFile -Raw -Encoding UTF8 | ConvertFrom-Json
$changed = 0

Write-Host ""
Write-Host "  OpenCode · Apply Models" -ForegroundColor Cyan
Write-Host "  ──────────────────────────────────────────────" -ForegroundColor DarkGray
Write-Host ""

# ── Fonction : insérer ou remplacer model: dans le frontmatter ─
function Set-AgentModel {
    param([string]$FilePath, [string]$Model)

    $raw = [System.IO.File]::ReadAllText($FilePath, [System.Text.Encoding]::UTF8)

    if ($raw -match '(?m)^model: ') {
        # Remplacer la ligne existante
        $new = $raw -replace '(?m)^model: [^\r\n]+', "model: $Model"
    } else {
        # Insérer après la première ligne "---" (ouverture frontmatter)
        $new = $raw -replace '(?s)(^---\s*\r?\n)', "`$1model: $Model`n"
    }

    if ($new -ne $raw) {
        [System.IO.File]::WriteAllText($FilePath, $new, [System.Text.Encoding]::UTF8)
        return $true
    }
    return $false
}

# ── Modèle global → opencode.json ─────────────────────────
if ($config.globalModel -and (Test-Path $OpenCodeJson)) {
    $oc    = Get-Content $OpenCodeJson -Raw -Encoding UTF8
    $newOc = $oc -replace '"model":\s*"[^"]*"', "`"model`": `"$($config.globalModel)`""
    if ($newOc -ne $oc) {
        [System.IO.File]::WriteAllText($OpenCodeJson, $newOc, [System.Text.Encoding]::UTF8)
        $short = ($config.globalModel -split '/')[-1]
        Write-Host "  [global]  opencode.json            → $short" -ForegroundColor Blue
        $changed++
    }
}

# ── Agents ────────────────────────────────────────────────
$agentNames = $config.agents | Get-Member -MemberType NoteProperty | Select-Object -ExpandProperty Name

foreach ($name in $agentNames) {
    $model  = $config.agents.$name
    $mdFile = Join-Path $AgentsDir "$name.md"

    if (-not (Test-Path $mdFile)) {
        Write-Host "  [skip]    @$name — agent introuvable" -ForegroundColor DarkGray
        continue
    }

    $updated = Set-AgentModel -FilePath $mdFile -Model $model
    $short   = ($model -split '/')[-1]

    if ($updated) {
        Write-Host "  [ok]      @$($name.PadRight(24)) → $short" -ForegroundColor Green
        $changed++
    } else {
        Write-Host "  [=]       @$($name.PadRight(24))   $short" -ForegroundColor DarkGray
    }
}

# ── max_parallel (swarm) ────────────────────────────────
$swarmFile = Join-Path $AgentsDir "swarm.md"
if ($config.PSObject.Properties.Name -contains "maxParallel" -and (Test-Path $swarmFile)) {
    $mp = $config.maxParallel
    $raw = [System.IO.File]::ReadAllText($swarmFile, [System.Text.Encoding]::UTF8)
    if ($raw -match '(?m)^max_parallel:\s*\d+') {
        $newRaw = $raw -replace '(?m)^max_parallel:\s*\d+', "max_parallel: $mp"
    } else {
        $newRaw = $raw -replace '(?s)(^---\s*\r?\n)', "`$1max_parallel: $mp\n"
    }
    if ($newRaw -ne $raw) {
        [System.IO.File]::WriteAllText($swarmFile, $newRaw, [System.Text.Encoding]::UTF8)
        Write-Host "  [swarm]   max_parallel             → $mp" -ForegroundColor Magenta
        $changed++
    }
}

# ── Résumé ────────────────────────────────────────────────
Write-Host ""
Write-Host "  ──────────────────────────────────────────────" -ForegroundColor DarkGray

if ($changed -eq 0) {
    Write-Host "  Déjà synchronisé — rien à mettre à jour" -ForegroundColor DarkGray
} else {
    Write-Host "  $changed fichier(s) mis à jour" -ForegroundColor Green
}

Write-Host ""
Write-Host "  Si un modèle ne fonctionne pas :" -ForegroundColor DarkGray
Write-Host "  → Ouvre ~/.config/opencode/lib/config.html · change le modèle · lance ce script" -ForegroundColor DarkGray
Write-Host ""
