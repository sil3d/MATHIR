# MATHIR Daemon Direct Socket (PowerShell, no Python)
# Usage: . ./mathir_daemon.ps1; Invoke-Mathir memory_recall @{query="Mycerise"; k=5}

function Invoke-Mathir {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Method,
        [hashtable]$Params = @{}
    )

    $client = New-Object System.Net.Sockets.TcpClient
    $client.Connect('127.0.0.1', 7338)
    $stream = $client.GetStream()

    $request = @{method = $Method; params = $Params} | ConvertTo-Json -Compress
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($request)
    $stream.Write($bytes, 0, $bytes.Length)
    $stream.Flush()

    # Read response
    $ms = New-Object System.IO.MemoryStream
    $buffer = New-Object byte[] 8192
    $stream.ReadTimeout = 5000
    try {
        while ($true) {
            $read = $stream.Read($buffer, 0, 8192)
            if ($read -le 0) { break }
            $ms.Write($buffer, 0, $read)
        }
    } catch { }

    $client.Close()
    $response = [System.Text.Encoding]::UTF8.GetString($ms.ToArray())
    try {
        return $response | ConvertFrom-Json
    } catch {
        return $response
    }
}

# Quick stats
function Get-MathirStats {
    $r = Invoke-Mathir -Method 'memory_stats'
    # Daemon returns either {result: ...} or the data directly
    if ($r.result) { return $r.result } else { return $r }
}

# Quick recall
function Search-Mathir {
    param([string]$Query, [int]$K = 5)
    $r = Invoke-Mathir -Method 'memory_recall' -Params @{query = $Query; k = $K}
    if ($r.result) { return $r.result } else { return $r }
}

# Quick save
function Save-Mathir {
    param(
        [string]$Content,
        [string]$Agent = 'powershell',
        [string]$BlockType = 'episodic',
        [string]$Label = '',
        [int]$Priority = 5
    )
    $r = Invoke-Mathir -Method 'memory_save' -Params @{
        content = $Content
        agent = $Agent
        block_type = $BlockType
        label = $Label
        priority = $Priority
    }
    if ($r.result) { return $r.result } else { return $r }
}

Write-Host "MATHIR PowerShell module loaded. Functions: Invoke-Mathir, Get-MathirStats, Search-Mathir, Save-Mathir"