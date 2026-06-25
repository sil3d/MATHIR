# setup-github-mcp.ps1
# Configure GitHub MCP server for OpenCode
# Usage: .\setup-github-mcp.ps1

Write-Host "=== GitHub MCP Server Setup ===" -ForegroundColor Cyan
Write-Host ""

# Check if GITHUB_TOKEN is already set
$currentToken = $env:GITHUB_TOKEN
if ($currentToken) {
    Write-Host "[OK] GITHUB_TOKEN is already set (length: $($currentToken.Length))" -ForegroundColor Green
    Write-Host ""
    Write-Host "To update, run:" -ForegroundColor Yellow
    Write-Host '  $env:GITHUB_TOKEN = "ghp_your_new_token_here"' -ForegroundColor White
    Write-Host ""
} else {
    Write-Host "[!] GITHUB_TOKEN is NOT set" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "To create a GitHub Personal Access Token (PAT):" -ForegroundColor Cyan
    Write-Host "  1. Go to https://github.com/settings/tokens" -ForegroundColor White
    Write-Host "  2. Click 'Generate new token' -> 'Generate new token (classic)'" -ForegroundColor White
    Write-Host "  3. Select scopes:" -ForegroundColor White
    Write-Host "     - repo (full control of private repositories)" -ForegroundColor Gray
    Write-Host "     - read:org (read organization membership)" -ForegroundColor Gray
    Write-Host "     - read:user (read user profile)" -ForegroundColor Gray
    Write-Host "     - read:project (read project boards)" -ForegroundColor Gray
    Write-Host "  4. Click 'Generate token' and copy it" -ForegroundColor White
    Write-Host ""
    Write-Host "Then set it in your PowerShell session:" -ForegroundColor Yellow
    Write-Host '  $env:GITHUB_TOKEN = "ghp_your_token_here"' -ForegroundColor White
    Write-Host ""
    Write-Host "To make it permanent (user-level):" -ForegroundColor Yellow
    Write-Host '  [System.Environment]::SetEnvironmentVariable("GITHUB_TOKEN", "ghp_your_token_here", "User")' -ForegroundColor White
    Write-Host ""
}

Write-Host "=== Verification ===" -ForegroundColor Cyan
Write-Host ""

# Test GitHub MCP server
Write-Host "Testing GitHub MCP server startup..." -ForegroundColor Gray
$result = npx -y @modelcontextprotocol/server-github --help 2>&1
if ($result -match "GitHub MCP Server") {
    Write-Host "[OK] GitHub MCP server is available" -ForegroundColor Green
} else {
    Write-Host "[FAIL] GitHub MCP server failed to start" -ForegroundColor Red
    Write-Host "Output: $result" -ForegroundColor Gray
}

Write-Host ""
Write-Host "=== Next Steps ===" -ForegroundColor Cyan
Write-Host "  1. Set GITHUB_TOKEN (see above)" -ForegroundColor White
Write-Host "  2. Restart OpenCode to load the new MCP server" -ForegroundColor White
Write-Host "  3. Use GitHub tools: clone, push, pull, create issues, etc." -ForegroundColor White
Write-Host ""
