# Integration Guide

## OpenCode Integration

### Step 1: Start MATHIR Daemon

```bash
python ~/.config/opencode/bin/mathir_daemon.py
```

### Step 2: Configure OpenCode

Edit `~/.config/opencode/opencode.json`:

```json
{
  "mcpServers": {
    "mathir": {
      "url": "http://127.0.0.1:7338/sse",
      "env": {}
    }
  }
}
```

Or use the config server:

```bash
powershell ~/.config/opencode/bin/start-config-server.ps1
# Open http://localhost:7337 in browser
# Add MCP server under MCPs tab
```

### Step 3: Verify

In OpenCode session:
```bash
python ~/.config/opencode/bin/mathir_client.py ping
# {"status": "ok", "model": "nomic-embed-text-v1.5", "dims": 768}
```

## Claude Desktop Integration

### Step 1: Start MATHIR Daemon

```bash
python bin/mathir_daemon.py
```

### Step 2: Configure Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json` (Mac)
or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "mathir": {
      "url": "http://127.0.0.1:7338/sse"
    }
  }
}
```

### Step 3: Restart Claude Desktop

MATHIR tools appear in Claude's tool list. Usage:
- "Remember this: [content]" → memory_save
- "What do you know about [topic]?" → memory_recall
- "Memory stats" → memory_stats

## Custom Integration via TCP Socket

Any language can connect via TCP:

```python
import socket
import json

def mathir_call(method, params=None):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(("127.0.0.1", 7338))
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": 1
        }
        s.sendall(json.dumps(request).encode() + b"\n")
        data = b""
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            data += chunk
            if b"\n" in data:
                break
        return json.loads(data.decode())

# Ping
print(mathir_call("ping"))

# Save
mathir_call("memory_save", {
    "content": "Uses React + TypeScript + Zustand",
    "agent": "coder",
    "block_type": "semantic",
    "label": "tech-stack",
    "priority": 7
})

# Recall
results = mathir_call("memory_recall", {
    "query": "what state management",
    "agent": "coder",
    "k": 5
})
for r in results["results"]:
    print(f"  {r['score']:.2f} — {r['content'][:80]}")
```

## Node.js Integration

```javascript
const net = require('net');

function mathirCall(method, params = {}) {
  return new Promise((resolve, reject) => {
    const client = net.createConnection({ port: 7338 }, () => {
      const request = JSON.stringify({
        jsonrpc: '2.0',
        method,
        params,
        id: 1
      }) + '\n';
      client.write(request);
    });
    
    let data = '';
    client.on('data', (chunk) => {
      data += chunk.toString();
      if (data.includes('\n')) {
        client.end();
        resolve(JSON.parse(data.trim()));
      }
    });
    
    client.on('error', reject);
  });
}

// Usage
mathirCall('ping').then(console.log);
mathirCall('memory_save', {
  content: 'Node.js project with Express',
  agent: 'coder',
  block_type: 'semantic',
  label: 'backend-stack',
  priority: 7
}).then(console.log);
```

## Bash Integration (curl)

```bash
# Ping
echo '{"jsonrpc":"2.0","method":"ping","id":1}' | nc -q 1 127.0.0.1 7338

# Save (escape JSON properly)
echo '{"jsonrpc":"2.0","method":"memory_save","params":{"content":"test","agent":"cli","block_type":"semantic","label":"cli-test","priority":5},"id":1}' | nc -q 1 127.0.0.1 7338

# Recall
echo '{"jsonrpc":"2.0","method":"memory_recall","params":{"query":"test","k":3},"id":1}' | nc -q 1 127.0.0.1 7338
```

## MCP SSE Transport

MATHIR supports Server-Sent Events (SSE) transport for MCP:

```bash
# Direct SSE connection
curl -N http://127.0.0.1:7338/sse

# POST messages to the endpoint
curl -X POST http://127.0.0.1:7338/messages \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"ping","id":1}'
```

## Python MCP SDK Integration

```python
import asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client

async def main():
    async with sse_client("http://127.0.0.1:7338/sse") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # List available tools
            tools = await session.list_tools()
            for tool in tools.tools:
                print(f"Tool: {tool.name} — {tool.description}")
            
            # Save memory
            result = await session.call_tool("memory_save", {
                "content": "Project uses Python 3.11 with FastAPI",
                "agent": "coder",
                "block_type": "semantic",
                "label": "backend-stack",
                "priority": 7
            })
            print(f"Saved: {result}")
            
            # Recall
            result = await session.call_tool("memory_recall", {
                "query": "what backend framework",
                "agent": "coder",
                "k": 5
            })
            for item in result.content:
                print(item.text)

asyncio.run(main())
```

## Installation Requirements

```bash
# For MATHIR server
pip install -r requirements.txt

# For MCP client (if using MCP SDK)
pip install mcp

# For SSE transport (if using SSE)
pip install httpx-sse
```

## Port Configuration

Default port: 7338

To change:
```bash
python bin/mathir_daemon.py --port 8080
```

Update client connections accordingly:
```bash
python bin/mathir_client.py ping --port 8080
```

## Firewall Note

On Windows, the first time you start the daemon, Windows Firewall may block it. Click "Allow access" to permit TCP connections on localhost.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Connection refused | Start daemon: `python bin/mathir_daemon.py` |
| Port already in use | Kill existing or use `--port 8080` |
| Firewall blocked | Allow access in Windows Firewall dialog |
| MCP tools not showing | Restart Claude Desktop / OpenCode after config |
| SSE not working | Ensure daemon started with `--sse` flag |
