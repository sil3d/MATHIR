# MATHIR Daemon Direct Socket (PowerShell, no Python)
# Usage: . ./mathir_daemon.ps1
#   Search-Mathir "Mycerise" -K 5
#   Get-MathirStats
#   Save-Mathir "hello world" -Label "greeting"
#
# Wire protocol: daemon reads until EOF on the socket, then sends ONE response.
# Key insight: we MUST call $client.Client.Shutdown(Send) after writing — otherwise
# the daemon's inner recv() loop never sees EOF, the connection stays open, and
# our Read() loop blocks until ReadTimeout fires. (Original bug: 5-second hang.)

$script:MATHIR_HOST = if ($env:MATHIR_HOST) { $env:MATHIR_HOST } else { '127.0.0.1' }
$script:MATHIR_PORT = if ($env:MATHIR_PORT) { [int]$env:MATHIR_PORT } else { 7338 }
$script:MATHIR_CONNECT_TIMEOUT_MS = 3000

function Invoke-Mathir {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Method,
        [hashtable]$Params = @{}
    )

    $request = @{method = $Method; params = $Params} | ConvertTo-Json -Compress -Depth 10
    $bytes   = [System.Text.Encoding]::UTF8.GetBytes($request)

    $client = New-Object System.Net.Sockets.TcpClient
    try {
        # Connect with a bounded timeout (default ctor blocks forever on filtered ports)
        $iar = $client.BeginConnect($script:MATHIR_HOST, $script:MATHIR_PORT, $null, $null)
        if (-not $iar.AsyncWaitHandle.WaitOne($script:MATHIR_CONNECT_TIMEOUT_MS)) {
            $client.Close()
            throw "Connect to $($script:MATHIR_HOST):$($script:MATHIR_PORT) timed out after $($script:MATHIR_CONNECT_TIMEOUT_MS)ms"
        }
        $client.EndConnect($iar)

        $stream = $client.GetStream()
        $stream.WriteTimeout = 5000

        # Write request
        $stream.Write($bytes, 0, $bytes.Length)
        $stream.Flush()

        # CRITICAL: shutdown write side → daemon's recv() returns 0 → daemon closes
        # connection → our Read() returns 0 → loop exits cleanly. Without this, we
        # block forever waiting for the daemon to close the connection it never
        # closes (the daemon's loop is `while True: data = conn.recv(...)`).
        $client.Client.Shutdown([System.Net.Sockets.SocketShutdown]::Send)

        # Read response until EOF. No timeout needed — daemon closes after sending.
        $ms     = New-Object System.IO.MemoryStream
        $buffer = New-Object byte[] 8192
        while ($true) {
            $read = $stream.Read($buffer, 0, 8192)
            if ($read -le 0) { break }
            $ms.Write($buffer, 0, $read)
        }

        if ($ms.Length -eq 0) {
            throw "Empty response from daemon (connection closed before any data)"
        }

        $response = [System.Text.Encoding]::UTF8.GetString($ms.ToArray())
        try {
            return $response | ConvertFrom-Json
        } catch {
            Write-Warning "Daemon returned non-JSON: $response"
            return $response
        }
    }
    finally {
        # Close socket (idempotent — already shutdown on write side)
        try { $client.Close() } catch { }
        $client.Dispose()
    }
}

# --- High-level convenience wrappers -----------------------------------------

function Get-MathirStats {
    Invoke-Mathir -Method 'memory_stats'
}

function Search-Mathir {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Query,
        [int]$K = 5,
        [string]$Agent,
        [string]$BlockType
    )
    $p = @{query = $Query; k = $K}
    if ($Agent)     { $p.agent      = $Agent }
    if ($BlockType) { $p.block_type = $BlockType }
    Invoke-Mathir -Method 'memory_recall' -Params $p
}

function Save-Mathir {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Content,
        [string]$Agent = 'powershell',
        [string]$BlockType = 'episodic',
        [string]$Label = '',
        [int]$Priority = 5
    )
    Invoke-Mathir -Method 'memory_save' -Params @{
        content    = $Content
        agent      = $Agent
        block_type = $BlockType
        label      = $Label
        priority   = $Priority
    }
}

function Remove-Mathir {
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$MemoryId)
    Invoke-Mathir -Method 'memory_delete' -Params @{memory_id = $MemoryId}
}

function Search-MathirHybrid {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Query,
        [int]$K = 5,
        [double]$VectorWeight = 1.0,
        [double]$Bm25Weight = 1.0
    )
    Invoke-Mathir -Method 'memory_hybrid_search' -Params @{
        query         = $Query
        k             = $K
        vector_weight = $VectorWeight
        bm25_weight   = $Bm25Weight
    }
}

function Get-MathirPing {
    Invoke-Mathir -Method 'ping'
}

Write-Host "MATHIR PowerShell module loaded. Commands:"
Write-Host "  Invoke-Mathir <method> [-Params @{...}]      # generic"
Write-Host "  Get-MathirStats"
Write-Host "  Get-MathirPing"
Write-Host "  Search-Mathir <query> [-K 5] [-Agent x] [-BlockType y]"
Write-Host "  Search-MathirHybrid <query> [-K 5]"
Write-Host "  Save-Mathir <content> [-Label x] [-Agent x] [-BlockType y] [-Priority 5]"
Write-Host "  Remove-Mathir <memory_id>"
Write-Host "  Daemon: $($script:MATHIR_HOST):$($script:MATHIR_PORT) (override with `\$env:MATHIR_HOST / `\$env:MATHIR_PORT)"
