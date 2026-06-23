"""
MATHIR MCP Server
Easy-to-use MCP server for MATHIR memory system.
Plug into any MCP-compatible client (Claude, OpenCode, etc.)
"""

import json
import sys
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add MATHIR to path
MATHIR_DIR = Path(__file__).parent
sys.path.insert(0, str(MATHIR_DIR))

from mathir_lib.providers import get_provider
from mathir_lib.config import load_config


class MATHIRMCPServer:
    """MCP Server for MATHIR memory system."""
    
    def __init__(self):
        self.config = load_config()
        self.provider = None
        self._memory = {}
    
    def _get_provider(self):
        """Get or create embedding provider."""
        if self.provider is None:
            provider_name = self.config.get("providers", {}).get("default", "direct")
            provider_config = self.config.get("providers", {}).get(provider_name, {})
            self.provider = get_provider(provider_name, provider_config)
        return self.provider
    
    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP JSON-RPC request."""
        method = request.get("method")
        params = request.get("params", {})
        request_id = request.get("id")
        
        try:
            if method == "initialize":
                return self._handle_initialize(request_id, params)
            elif method == "tools/list":
                return self._handle_tools_list(request_id)
            elif method == "tools/call":
                return self._handle_tools_call(request_id, params)
            else:
                return self._error_response(request_id, -32601, f"Method not found: {method}")
        except Exception as e:
            return self._error_response(request_id, -32603, str(e))
    
    def _handle_initialize(self, request_id: int, params: Dict) -> Dict:
        """Handle initialize request."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "mathir-memory",
                    "version": "1.0.0"
                }
            }
        }
    
    def _handle_tools_list(self, request_id: int) -> Dict:
        """Handle tools/list request."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "tools": [
                    {
                        "name": "memory_save",
                        "description": "Save a memory to MATHIR",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string", "description": "Memory content"},
                                "agent": {"type": "string", "description": "Agent name"},
                                "tier": {"type": "string", "enum": ["working", "episodic", "semantic", "immune"], "description": "Memory tier"},
                                "label": {"type": "string", "description": "Short label"}
                            },
                            "required": ["content", "agent", "tier", "label"]
                        }
                    },
                    {
                        "name": "memory_recall",
                        "description": "Recall similar memories from MATHIR",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "Search query"},
                                "k": {"type": "integer", "description": "Number of results", "default": 5},
                                "tier": {"type": "string", "description": "Filter by tier"}
                            },
                            "required": ["query"]
                        }
                    },
                    {
                        "name": "memory_stats",
                        "description": "Get MATHIR memory statistics",
                        "inputSchema": {
                            "type": "object",
                            "properties": {}
                        }
                    },
                    {
                        "name": "provider_info",
                        "description": "Get embedding provider info",
                        "inputSchema": {
                            "type": "object",
                            "properties": {}
                        }
                    }
                ]
            }
        }
    
    def _handle_tools_call(self, request_id: int, params: Dict) -> Dict:
        """Handle tools/call request."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        if tool_name == "memory_save":
            result = self._memory_save(arguments)
        elif tool_name == "memory_recall":
            result = self._memory_recall(arguments)
        elif tool_name == "memory_stats":
            result = self._memory_stats()
        elif tool_name == "provider_info":
            result = self._provider_info()
        else:
            return self._error_response(request_id, -32602, f"Unknown tool: {tool_name}")
        
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(result, indent=2)
                    }
                ]
            }
        }
    
    def _memory_save(self, args: Dict) -> Dict:
        """Save a memory."""
        content = args["content"]
        agent = args["agent"]
        tier = args["tier"]
        label = args["label"]
        
        # Generate embedding
        provider = self._get_provider()
        embedding = provider.embed_batch([content])[0]
        
        # Store in memory
        memory_id = f"mem_{len(self._memory)}"
        self._memory[memory_id] = {
            "id": memory_id,
            "content": content,
            "agent": agent,
            "tier": tier,
            "label": label,
            "embedding": embedding.tolist()
        }
        
        return {
            "success": True,
            "memory_id": memory_id,
            "agent": agent,
            "tier": tier,
            "label": label
        }
    
    def _memory_recall(self, args: Dict) -> Dict:
        """Recall similar memories."""
        query = args["query"]
        k = args.get("k", 5)
        tier_filter = args.get("tier")
        
        # Generate query embedding
        provider = self._get_provider()
        query_emb = provider.embed_batch([query])[0]
        
        # Search memories
        results = []
        for mem_id, mem in self._memory.items():
            if tier_filter and mem["tier"] != tier_filter:
                continue
            
            mem_emb = mem["embedding"]
            similarity = float(query_emb @ mem_emb)
            
            results.append({
                "id": mem_id,
                "content": mem["content"],
                "agent": mem["agent"],
                "tier": mem["tier"],
                "label": mem["label"],
                "similarity": similarity
            })
        
        # Sort by similarity
        results.sort(key=lambda x: x["similarity"], reverse=True)
        results = results[:k]
        
        return {
            "query": query,
            "results": results,
            "total": len(results)
        }
    
    def _memory_stats(self) -> Dict:
        """Get memory statistics."""
        stats = {
            "total_memories": len(self._memory),
            "by_tier": {},
            "by_agent": {}
        }
        
        for mem in self._memory.values():
            tier = mem["tier"]
            agent = mem["agent"]
            
            stats["by_tier"][tier] = stats["by_tier"].get(tier, 0) + 1
            stats["by_agent"][agent] = stats["by_agent"].get(agent, 0) + 1
        
        return stats
    
    def _provider_info(self) -> Dict:
        """Get provider info."""
        provider = self._get_provider()
        return {
            "provider_id": provider.provider_id(),
            "embedding_dim": provider.embedding_dim()
        }
    
    def _error_response(self, request_id: int, code: int, message: str) -> Dict:
        """Create error response."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": code,
                "message": message
            }
        }


def main():
    """Run MCP server."""
    server = MATHIRMCPServer()
    
    print("MATHIR MCP Server started", file=sys.stderr)
    print("Listening on stdin...", file=sys.stderr)
    
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        
        try:
            request = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON: {e}", file=sys.stderr)
            continue
        
        response = server.handle_request(request)
        print(json.dumps(response))


if __name__ == "__main__":
    main()
