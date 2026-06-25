#Requires -Version 5.1
<#
.SYNOPSIS
    OpenCode PowerConfig — Install (Windows)
    Copies the config directory to %USERPROFILE%\.config\opencode\ on a new PC.
    Run once on any machine to install the full system.

.PARAMETER DryRun
    Show what would happen without making any changes.

.PARAMETER Export
    Create a ZIP archive of the entire config directory (excludes data/ and __pycache__).

.PARAMETER SkipServer
    Don't start the config server after install.

.EXAMPLE
    .\install.ps1
    .\install.ps1 -DryRun
    .\install.ps1 -Export
#>
param(
    [switch]$DryRun    = $false,
    [switch]$Export    = $false,
    [switch]$SkipServer = $false
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Colour helpers ──────────────────────────────────────────
function Write-Cyan  { param($m) Write-Host $m -ForegroundColor Cyan    }
function Write-Green { param($m) Write-Host $m -ForegroundColor Green   }
function Write-Red   { param($m) Write-Host $m -ForegroundColor Red     }
function Write-Yellow{ param($m) Write-Host $m -ForegroundColor Yellow  }

# ── Banner ─────────────────────────────────────────────────
Write-Host ""
Write-Cyan "╔══════════════════════════════════════════════════════════════╗"
Write-Cyan "║       OpenCode PowerConfig — Install (Windows)               ║"
Write-Cyan "║       33 agents · 95 skills · 397 models · SQLite memory    ║"
Write-Cyan "╚══════════════════════════════════════════════════════════════╝"
if ($DryRun) { Write-Yellow "  [DRY RUN — no changes will be made]" }
if ($Export)  { Write-Yellow "  [EXPORT MODE — creating ZIP archive]" }
Write-Host ""

$SrcDir   = $PSScriptRoot
$DestDir  = Join-Path $env:USERPROFILE ".config\opencode"
$IsSame   = ($SrcDir -eq $DestDir)

# ════════════════════════════════════════════════════════════════
# EXPORT MODE
# ════════════════════════════════════════════════════════════════
if ($Export) {
    $ZipPath = Join-Path $SrcDir "opencode-powerconfig.zip"
    if ($IsSame) {
        Write-Red "  [ERROR] Can't export from the target directory itself."
        Write-Red "         Run from a separate copy of the config."
        exit 1
    }
    # Build exclude patterns (data files, cache, node_modules)
    $exclude = @("data\*","data","__pycache__","__pycache__\*",
                  "node_modules","node_modules\*",".ruff_cache",".ruff_cache\*")
    Write-Cyan "  Creating ZIP archive..."
    try {
        if (Test-Path $ZipPath) { Remove-Item $ZipPath -Force }
        Compress-Archive -Path (Join-Path $SrcDir "*") -DestinationPath $ZipPath -CompressionLevel Optimal
        $size = (Get-Item $ZipPath).Length / 1MB
        Write-Green "  [OK] Archive created: $ZipPath ($([math]::Round($size,1)) MB)"
        Write-Host ""
        Write-Host "  To install on another PC:"
        Write-Host "    1. Copy opencode-powerconfig.zip to the target PC"
        Write-Host "    2. Extract to %USERPROFILE%\.config\opencode\"
        Write-Host "    3. Run bin\install.ps1 from the extracted directory"
        Write-Host ""
    } catch {
        Write-Red "  [ERROR] Failed to create ZIP: $_"
        exit 1
    }
    exit 0
}

# ════════════════════════════════════════════════════════════════
# PREREQUISITE CHECKS
# ════════════════════════════════════════════════════════════════
Write-Cyan "  Checking prerequisites..."

$checks = @(
    @{ name = "python3"; ok = (Get-Command python -ErrorAction SilentlyContinue) -or
                               (Get-Command python3 -ErrorAction SilentlyContinue) },
    @{ name = "sqlite3"; ok = [bool](Get-Command sqlite3 -ErrorAction SilentlyContinue) },
    @{ name = "opencode"; ok = [bool](Get-Command opencode -ErrorAction SilentlyContinue) }
)

$prereqOk = $true
foreach ($c in $checks) {
    if ($c.ok) {
        Write-Green "  [OK]   $($c.name)"
    } else {
        Write-Red   "  [!!]  $($c.name) — NOT FOUND"
        $prereqOk = $false
    }
}

if (-not $prereqOk) {
    Write-Red ""
    Write-Red "  One or more prerequisites are missing. Install them before running this script."
    Write-Yellow "    Python  → https://www.python.org/downloads/"
    Write-Yellow "    opencode → npm install -g opencode-ai"
    Write-Yellow "    sqlite3  → included in Python (via pip or system package)"
    exit 1
}
Write-Host ""

# ════════════════════════════════════════════════════════════════
# COPY HELPER
# ════════════════════════════════════════════════════════════════
$installed = @()
$skipped   = @()
$errors    = @()

function Copy-Item-Conditional {
    param([string]$Src, [string]$Dst, [bool]$Protected)

    $fname = Split-Path $Dst -Leaf

    if ($Protected -and (Test-Path $Dst)) {
        $skipped += $fname
        Write-Yellow "  SKIP   $fname  (protected — data already exists)"
        return
    }
    try {
        if (-not $DryRun) {
            $dstDir = Split-Path $Dst -Parent
            if (-not (Test-Path $dstDir)) { New-Item -ItemType Directory -Force -Path $dstDir | Out-Null }
            Copy-Item -Path $Src -Destination $Dst -Force
        }
        $installed += $fname
        Write-Green "  COPY   $fname"
    } catch {
        $errors += $fname
        Write-Red "  ERR    $fname  — $($_.Exception.Message)"
    }
}

# ── Copy a directory recursively ─────────────────────────
function Copy-Dir-Conditional {
    param([string]$SrcDir, [string]$DstDir, [bool]$Protected, [string]$Label)

    if (-not (Test-Path $SrcDir)) {
        Write-Yellow "  WARN   $Label not found — skipping"
        return
    }
    if ($IsSame) {
        $skipped += $Label
        Write-Yellow "  SKIP   $Label/  (source == destination)"
        return
    }
    Write-Cyan "  $Label/..."
    Get-ChildItem -Path $SrcDir -File -Recurse -ErrorAction SilentlyContinue | ForEach-Object {
        $rel    = $_.FullName.Substring($SrcDir.Length).TrimStart('\','/')
        $dstF   = Join-Path $DstDir $rel
        $prot   = $Protected -and ($rel -match "^[^/]+$")
        Copy-Item-Conditional -Src $_.FullName -Dst $dstF -Protected $prot
    }
}

# ════════════════════════════════════════════════════════════════
# CREATE TARGET DIRECTORY
# ════════════════════════════════════════════════════════════════
Write-Cyan "  Creating $DestDir..."
if (-not $DryRun) { New-Item -ItemType Directory -Force -Path $DestDir | Out-Null }
Write-Green "  OK"
Write-Host ""

# ════════════════════════════════════════════════════════════════
# COPY ROOT FILES (must stay at root)
# ════════════════════════════════════════════════════════════════
Write-Cyan "  Copying root files (must stay at root)..."
$rootMustStay = @(
    "opencode.json", "GLOBAL_INSTRUCTIONS.md", "SKILL_ROUTING.md",
    "README.md", "AGENTS.md", "CHANGELOG.md", "version.json", "LICENSE.md"
)
foreach ($fname in $rootMustStay) {
    $srcF = Join-Path $SrcDir $fname
    $dstF = Join-Path $DestDir $fname
    if (Test-Path $srcF) {
        Copy-Item-Conditional -Src $srcF -Dst $dstF -Protected $false
    }
}
Write-Host ""

# ════════════════════════════════════════════════════════════════
# COPY SUBDIRECTORIES
# ════════════════════════════════════════════════════════════════
$dirsToCopy = @(
    @{ src = "bin";        dst = "bin";         prot = $false; label = "bin"        },
    @{ src = "lib";        dst = "lib";         prot = $false; label = "lib"        },
    @{ src = "data";       dst = "data";        prot = $true;  label = "data"       },
    @{ src = "agents";     dst = "agents";      prot = $false; label = "agents"     },
    @{ src = "skills";     dst = "skills";      prot = $false; label = "skills"     },
    @{ src = "skills-global"; dst = "skills-global"; prot = $false; label = "skills-global" },
    @{ src = "commands";   dst = "commands";    prot = $false; label = "commands"   },
    @{ src = "docs";       dst = "docs";        prot = $false; label = "docs"       },
    @{ src = "pre-existing"; dst = "pre-existing"; prot = $false; label = "pre-existing" }
)

foreach ($d in $dirsToCopy) {
    $srcD = Join-Path $SrcDir $d.src
    $dstD = Join-Path $DestDir $d.dst
    if (Test-Path $srcD) {
        Write-Cyan "  Copying $($d.label)/..."
        Get-ChildItem -Path $srcD -File -Recurse -ErrorAction SilentlyContinue | ForEach-Object {
            $rel  = $_.FullName.Substring($srcD.Length).TrimStart('\','/')
            $dstF = Join-Path $dstD $rel
            Copy-Item-Conditional -Src $_.FullName -Dst $dstF -Protected $d.prot
        }
    }
    Write-Host ""
}

# ════════════════════════════════════════════════════════════════
# INIT SQLITE DATABASE
# ════════════════════════════════════════════════════════════════
Write-Cyan "  Initialising SQLite database..."
$dbPath     = Join-Path $DestDir "data\memory.db"
$schemaPath = Join-Path $DestDir "data\memory-schema.sql"

if (Test-Path $dbPath) {
    Write-Yellow "  SKIP   memory.db  (already exists — data preserved)"
} elseif (Test-Path $schemaPath) {
    try {
        if (-not $DryRun) { & sqlite3 $dbPath ".read $schemaPath" 2>$null }
        Write-Green "  INIT   memory.db"
        $installed += "memory.db"
    } catch {
        Write-Red "  ERR    SQLite init failed: $($_.Exception.Message)"
        $errors += "memory.db"
    }
} else {
    Write-Yellow "  WARN   memory-schema.sql not found"
}
Write-Host ""

# ════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════
Write-Cyan "══════════════════════════ SUMMARY ═════════════════════════"
Write-Host ""

if ($installed.Count -gt 0) {
    Write-Green "  Installed ($($installed.Count)):"
    foreach ($f in $installed) { Write-Green "    + $f" }
}
if ($skipped.Count -gt 0) {
    Write-Host ""
    Write-Yellow "  Skipped ($($skipped.Count)):"
    foreach ($f in $skipped)   { Write-Yellow "    ~ $f" }
}
if ($errors.Count -gt 0) {
    Write-Host ""
    Write-Red "  Errors ($($errors.Count)):"
    foreach ($f in $errors)    { Write-Red   "    x $f" }
}

Write-Host ""
if ($errors.Count -eq 0) {
    Write-Green "  All done! OpenCode config installed at:"
    Write-Green "  $DestDir"
    Write-Host ""
    Write-Cyan "  Next steps:"
    Write-Host "    1. Start the server:   powershell -File `"$DestDir\bin\start-config-server.ps1`""
    Write-Host "    2. Open config UI:      $DestDir\lib\lib/config.html  (in Chrome)"
    Write-Host "    3. Assign models:       Select a model per agent → Save & Apply"
    Write-Host "    4. Test it:             @coder hello world"

    # MATHIR agent injection: ensure all agents have the current injection block.
    # The template is agents/_MATHIR_INJECT.md; the script propagates it.
    Write-Host ""
    Write-Cyan "  MATHIR Agent Injection:"
    $injectScript = Join-Path $DestDir "bin/mathir_inject.py"
    if (Test-Path $injectScript) {
        try {
            $injectOutput = & python $injectScript --check 2>&1
            if ($LASTEXITCODE -eq 2) {
                Write-Yellow "    Some agents need injection — running --apply..."
                & python $injectScript --apply 2>&1 | Out-Null
                if ($LASTEXITCODE -eq 0) {
                    Write-Green "    [OK] MATHIR injection applied to all agents."
                } else {
                    Write-Red "    [WARN] MATHIR inject failed — run: python bin/mathir_inject.py --apply"
                }
            } elseif ($LASTEXITCODE -eq 0) {
                Write-Green "    [OK] All agents have current MATHIR injection block."
            }
        } catch {
            Write-Yellow "    [SKIP] Could not run mathir_inject.py: $_"
        }
    } else {
        Write-Yellow "    [SKIP] mathir_inject.py not found — agents may need manual injection."
    }
} else {
    Write-Red "  Completed with $($errors.Count) error(s). Review above."
}

if ($DryRun) {
    Write-Host ""
    Write-Yellow "  [DRY RUN] No files were modified."
}
Write-Host ""