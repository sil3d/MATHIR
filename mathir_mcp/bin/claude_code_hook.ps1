$hookPath = Join-Path $PSScriptRoot "claude_code_hook.py"
$bytes = [System.IO.Stream]::new([Console]::OpenStandardInput(), [Console]::InputEncoding).ReadToEnd()
$json = @{message = $bytes} | ConvertTo-Json -Compress
$json | python $hookPath
