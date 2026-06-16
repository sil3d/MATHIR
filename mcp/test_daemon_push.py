#!/usr/bin/env python3
"""
MATHIR Daemon Push Test -- Demonstrates automatic memory enrichment.

The "daemon push" concept:
  Instead of agents explicitly calling recall(), the client middleware
  analyzes the current context and automatically pulls relevant memories,
  enriching the agent's context without tool calls.

Architecture:
  Agent -> PushClient (this script) -> Daemon (port 7338)
  
  PushClient:
  1. Receives agent's conversation context
  2. Extracts key themes via NLP-lite heuristics
  3. Queries daemon for relevant memories (multiple queries, deduped)
  4. Returns enriched context with memories ranked by relevance
  5. Measures latency at each step

Usage:
  python test_daemon_push.py                    # Run all demo scenarios
  python test_daemon_push.py --context "..."    # Custom context
  python test_daemon_push.py --benchmark        # Run latency benchmarks
  python test_daemon_push.py --interactive      # Interactive REPL
"""

import sys
import os
import json
import time
import socket
import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict

# Fix Windows console encoding for Unicode output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        os.environ["PYTHONIOENCODING"] = "utf-8"

# --- Configuration -----------------------------------------------------------

DAEMON_HOST = "127.0.0.1"
DAEMON_PORT = 7338
DAEMON_TIMEOUT = 30

# --- Data Classes ------------------------------------------------------------

@dataclass
class PushResult:
    """Result of a daemon push operation."""
    memories: List[Dict[str, Any]]
    queries_used: List[str]
    total_latency_ms: float
    query_latency_ms: float
    embed_latency_ms: float
    dedup_count: int
    context_length: int
    relevance_scores: List[float] = field(default_factory=list)

    @property
    def avg_relevance(self) -> float:
        return sum(self.relevance_scores) / len(self.relevance_scores) if self.relevance_scores else 0.0

    def summary(self) -> str:
        lines = [
            f"  Memories found:     {len(self.memories)}",
            f"  Queries executed:   {len(self.queries_used)}",
            f"  Duplicates removed: {self.dedup_count}",
            f"  Total latency:      {self.total_latency_ms:.1f}ms",
            f"    Query time:       {self.query_latency_ms:.1f}ms",
            f"    Embed time:       {self.embed_latency_ms:.1f}ms",
            f"  Avg relevance:      {self.avg_relevance:.3f}",
            f"  Context length:     {self.context_length} chars",
        ]
        return "\n".join(lines)


# --- Daemon Connector --------------------------------------------------------

class DaemonConnector:
    """Low-level TCP connection to MATHIR daemon."""

    def __init__(self, host: str = DAEMON_HOST, port: int = DAEMON_PORT):
        self.host = host
        self.port = port

    def call(self, method: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Send JSON-RPC request to daemon, return response."""
        if params is None:
            params = {}

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(DAEMON_TIMEOUT)
            sock.connect((self.host, self.port))

            request = json.dumps({"method": method, "params": params})
            sock.sendall(request.encode("utf-8"))

            data = sock.recv(65536)
            sock.close()

            return json.loads(data.decode("utf-8"))
        except socket.error as e:
            return {"error": f"Daemon connection failed: {e}"}
        except json.JSONDecodeError as e:
            return {"error": f"Invalid JSON response: {e}"}

    def ping(self) -> bool:
        """Check if daemon is alive."""
        result = self.call("ping")
        return "error" not in result

    def recall(self, query: str, k: int = 5, agent: str = None,
               block_type: str = None) -> Dict[str, Any]:
        """Semantic memory recall via daemon."""
        params = {"query": query, "k": k}
        if agent:
            params["agent"] = agent
        if block_type:
            params["block_type"] = block_type
        return self.call("memory_recall", params)

    def save(self, content: str, agent: str, block_type: str,
             label: str, priority: int = 5) -> Dict[str, Any]:
        """Save a memory via daemon."""
        return self.call("memory_save", {
            "content": content,
            "agent": agent,
            "block_type": block_type,
            "label": label,
            "priority": priority,
        })


# --- Context Analyzer --------------------------------------------------------

class ContextAnalyzer:
    """
    Extracts search queries from agent conversation context.
    
    Uses NLP-lite heuristics:
    - Keywords extraction (frequency + position weighting)
    - Named entity patterns (file paths, function names, errors)
    - Topic phrases from sentence structure
    """

    # Common stop words to filter out
    STOP_WORDS = frozenset({
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above", "below",
        "between", "out", "off", "over", "under", "again", "further", "then",
        "once", "here", "there", "when", "where", "why", "how", "all", "both",
        "each", "few", "more", "most", "other", "some", "such", "no", "nor",
        "not", "only", "own", "same", "so", "than", "too", "very", "just",
        "don", "now", "and", "but", "or", "if", "while", "that", "this",
        "these", "those", "it", "its", "i", "me", "my", "myself", "we", "our",
        "you", "your", "he", "him", "his", "she", "her", "they", "them", "their",
        "what", "which", "who", "whom", "about", "up", "down", "s", "t", "re",
        "ve", "ll", "d", "m", "doesn", "isn", "wasn", "weren", "hasn", "haven",
        "hadn", "won", "wouldn", "couldn", "shouldn", "mustn", "let", "say",
        "said", "also", "like", "get", "got", "make", "made", "go", "going",
        "went", "come", "came", "take", "took", "give", "gave", "know", "knew",
        "think", "thought", "see", "saw", "want", "use", "find", "found",
        "tell", "told", "ask", "asked", "work", "seem", "feel", "felt", "try",
        "left", "right", "back", "still", "even", "new", "want", "because",
        "way", "well", "look", "now", "thing", "things", "much", "something",
        "anything", "nothing", "everything", "really", "quite", "already",
        "sure", "please", "thank", "thanks", "yes", "yeah", "ok", "okay",
        "right", "wrong", "one", "two", "first", "second", "last", "next",
    })

    @classmethod
    def extract_queries(cls, context: str, max_queries: int = 5) -> List[str]:
        """
        Extract search queries from context text.
        
        Returns list of queries ranked by expected relevance.
        """
        queries = []

        # 1. Extract error patterns (highest priority)
        error_patterns = cls._extract_errors(context)
        queries.extend(error_patterns)

        # 2. Extract file paths
        file_paths = cls._extract_file_paths(context)
        queries.extend(file_paths)

        # 3. Extract function/class names
        code_names = cls._extract_code_names(context)
        queries.extend(code_names)

        # 4. Extract key phrases
        key_phrases = cls._extract_key_phrases(context)
        queries.extend(key_phrases)

        # Deduplicate while preserving order
        seen = set()
        unique_queries = []
        for q in queries:
            q_lower = q.lower().strip()
            if q_lower not in seen and len(q_lower) > 3:
                seen.add(q_lower)
                unique_queries.append(q)

        return unique_queries[:max_queries]

    @classmethod
    def _extract_errors(cls, text: str) -> List[str]:
        """Extract error messages and stack traces."""
        queries = []
        # Python-style errors
        for match in re.finditer(r'(?:Error|Exception|Traceback)[^\n]{5,100}', text):
            queries.append(match.group(0).strip())
        # "error: ..." patterns
        for match in re.finditer(r'(?i)error[:\s]+(.{5,80})', text):
            queries.append(f"error {match.group(1).strip()}")
        return queries[:2]

    @classmethod
    def _extract_file_paths(cls, text: str) -> List[str]:
        """Extract file paths and module references."""
        queries = []
        for match in re.finditer(r'[\w/\\.-]+\.(?:py|ts|tsx|js|jsx|rs|go|java|rb|css|json|yaml|yml|toml)', text):
            path = match.group(0)
            # Use just the filename for search
            filename = path.split("/")[-1].split("\\")[-1]
            queries.append(filename)
        return queries[:2]

    @classmethod
    def _extract_code_names(cls, text: str) -> List[str]:
        """Extract function, class, and variable names."""
        queries = []
        # snake_case identifiers (likely functions/variables)
        for match in re.finditer(r'\b([a-z_][a-z0-9_]{3,30})\b', text):
            word = match.group(1)
            if word not in cls.STOP_WORDS:
                queries.append(word)
        # PascalCase (likely classes)
        for match in re.finditer(r'\b([A-Z][a-zA-Z0-9]{2,30})\b', text):
            queries.append(match.group(1))
        return queries[:3]

    @classmethod
    def _extract_key_phrases(cls, text: str) -> List[str]:
        """Extract meaningful phrases from natural language."""
        queries = []
        # Split into sentences and extract noun phrases
        sentences = re.split(r'[.!?\n]+', text)
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 10 or len(sent) > 200:
                continue
            # Extract 2-4 word chunks that aren't all stop words
            words = sent.split()
            for i in range(len(words) - 1):
                chunk = " ".join(words[i:i+2])
                chunk_words = set(w.lower() for w in words[i:i+2])
                if not chunk_words.issubset(cls.STOP_WORDS):
                    queries.append(chunk)
        return queries[:3]


# --- Push Client -------------------------------------------------------------

class PushClient:
    """
    Client that automatically enriches agent context with relevant memories.
    
    Flow:
    1. Agent sends conversation context
    2. Client extracts search queries from context
    3. Client queries daemon for each query (parallel-ready, sequential here)
    4. Results are deduplicated and ranked
    5. Enriched context returned to agent
    """

    def __init__(self, host: str = DAEMON_HOST, port: int = DAEMON_PORT):
        self.daemon = DaemonConnector(host, port)
        self._query_cache: OrderedDict[str, List[Dict]] = OrderedDict()
        self._cache_max = 100

    def push(self, context: str, k_per_query: int = 3,
             max_memories: int = 10) -> PushResult:
        """
        Analyze context and push relevant memories.
        
        Args:
            context: Agent's current conversation/task context
            k_per_query: Results per query sent to daemon
            max_memories: Maximum memories to return
            
        Returns:
            PushResult with memories, latency, and relevance data
        """
        total_start = time.perf_counter()

        # Step 1: Extract queries from context
        embed_start = time.perf_counter()
        queries = ContextAnalyzer.extract_queries(context)
        embed_latency = (time.perf_counter() - embed_start) * 1000

        if not queries:
            # Fallback: use the full context as a single query
            queries = [context[:200]]

        # Step 2: Query daemon for each query
        query_start = time.perf_counter()
        all_results = []
        seen_ids = set()

        for query in queries:
            # Check cache
            cache_key = hashlib.md5(query.encode()).hexdigest()
            if cache_key in self._query_cache:
                cached = self._query_cache[cache_key]
                for r in cached:
                    rid = r.get("memory_id", "")
                    if rid not in seen_ids:
                        seen_ids.add(rid)
                        all_results.append(r)
                continue

            result = self.daemon.recall(query, k=k_per_query)
            if "error" in result:
                continue

            memories = result.get("results", [])
            self._query_cache[cache_key] = memories
            if len(self._query_cache) > self._cache_max:
                self._query_cache.popitem(last=False)

            for r in memories:
                rid = r.get("memory_id", "")
                if rid not in seen_ids:
                    seen_ids.add(rid)
                    all_results.append(r)

        query_latency = (time.perf_counter() - query_start) * 1000

        # Step 3: Rank by relevance score and deduplicate
        all_results.sort(key=lambda r: r.get("score", 0), reverse=True)
        
        # Deduplicate by content hash (same content = same memory, even if different IDs)
        seen_contents = set()
        deduped = []
        for r in all_results:
            content = r.get("content", "").strip()
            content_hash = hashlib.md5(content.encode()).hexdigest() if content else r.get("memory_id", "")
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                deduped.append(r)
        
        dedup_count = len(all_results) - len(deduped)
        top_results = deduped[:max_memories]
        relevance_scores = [r.get("score", 0.0) for r in top_results]

        total_latency = (time.perf_counter() - total_start) * 1000

        return PushResult(
            memories=top_results,
            queries_used=queries,
            total_latency_ms=total_latency,
            query_latency_ms=query_latency,
            embed_latency_ms=embed_latency,
            dedup_count=dedup_count,
            context_length=len(context),
            relevance_scores=relevance_scores,
        )

    def push_formatted(self, context: str, **kwargs) -> Tuple[str, PushResult]:
        """
        Push and return formatted context string for the agent.
        
        Returns:
            (formatted_enrichment_text, raw_result)
        """
        result = self.push(context, **kwargs)

        if not result.memories:
            return "No relevant memories found.", result

        lines = ["## Relevant Memories (auto-pushed by MATHIR)\n"]
        for i, mem in enumerate(result.memories, 1):
            score = mem.get("score", 0)
            label = mem.get("label", "unknown")
            agent = mem.get("agent", "?")
            block_type = mem.get("block_type", "?")
            content = mem.get("content", "")
            lines.append(f"### {i}. [{block_type}/{agent}] {label} (score: {score:.3f})")
            lines.append(f"{content}\n")

        lines.append(f"---\n*{len(result.memories)} memories pushed "
                     f"in {result.total_latency_ms:.0f}ms "
                     f"across {len(result.queries_used)} queries*")

        return "\n".join(lines), result


# --- Demo Scenarios ----------------------------------------------------------

DEMO_CONTEXTS = [
    {
        "name": "Auth Bug Debug",
        "context": """
I'm debugging a null pointer error in the authentication module. 
The login endpoint at /api/auth/login is crashing when the session token 
isn't refreshed. The error happens in auth.py line 42 where it tries to 
access session.user without checking if session is None first. 
I need to add a mutex lock to prevent race conditions during token refresh.
The stack trace shows:
  File "auth.py", line 42, in login
    user = session.user  # <-- crashes here
TypeError: 'NoneType' object has no attribute 'user'
""",
    },
    {
        "name": "Project Setup",
        "context": """
Setting up a new Next.js project with TypeScript. Using Prisma for the 
database, Tailwind CSS for styling. The API routes go in /src/app/api/ 
and components in /src/components/. Need to configure the database 
connection in prisma/schema.prisma and set up environment variables 
in .env.local for DATABASE_URL and NEXTAUTH_SECRET.
""",
    },
    {
        "name": "Refactor Request",
        "context": """
I need to refactor the WebSocket connection handler. Currently it's a 
God class with 500 lines handling auth, message routing, reconnection, 
and rate limiting. Want to split it into separate services: 
WebSocketAuthService, MessageRouter, ReconnectionManager, and RateLimiter. 
Should follow the existing patterns in the codebase for dependency injection.
""",
    },
    {
        "name": "Performance Issue",
        "context": """
The dashboard page is loading slowly — LCP is 4.2s. I think the N+1 
query problem in the dashboard API is the bottleneck. Each widget makes 
a separate database query. Need to batch them or use a DataLoader pattern. 
Also the bundle size is large because we're importing all of lodash 
instead of individual functions. The page uses React Query for data 
fetching with staleTime of 30 seconds.
""",
    },
]


def run_demo_scenarios():
    """Run all demo scenarios and display results."""
    client = PushClient()

    print("=" * 70)
    print("MATHIR DAEMON PUSH -- DEMO")
    print("=" * 70)
    print()

    # Check daemon
    if not client.daemon.ping():
        print("ERROR: Daemon not running on port 7338")
        print("Start with: python ~/.config/opencode/bin/mathir_daemon.py")
        sys.exit(1)
    print("Daemon: OK (port 7338)\n")

    # Save some test memories first
    print("Saving test memories...")
    test_memories = [
        ("Updated: null pointer fixed in PR #42. Token refresh now uses mutex lock.", "coder", "working_memory", "auth-bug-fix", 9),
        ("Next.js 14, Prisma, Tailwind. API routes in /src/app/api. Components in /src/components/.", "coder", "semantic", "project-stack", 7),
        ("WebSocket refactor: split God class into AuthService, MessageRouter, ReconnectionManager.", "refactor", "semantic", "ws-refactor-plan", 8),
        ("N+1 query in dashboard API. Use DataLoader pattern. Bundle size from lodash imports.", "performance-optimizer", "working_memory", "perf-bottleneck", 8),
        ("Memory protocol: save with mathir_client.py save, recall with recall command.", "coder", "procedural", "mathir-protocol", 6),
    ]
    for content, agent, btype, label, priority in test_memories:
        result = client.daemon.save(content, agent, btype, label, priority)
        status = "OK" if result.get("saved") else f"FAIL: {result.get('error', 'unknown')}"
        print(f"  [{label}] {status}")

    print()
    print("-" * 70)

    all_results = []

    for scenario in DEMO_CONTEXTS:
        print(f"\n{'=' * 70}")
        print(f"SCENARIO: {scenario['name']}")
        print(f"{'=' * 70}")
        print(f"\nContext ({len(scenario['context'])} chars):")
        print(f"  {scenario['context'].strip()[:150]}...")
        print()

        enriched_text, result = client.push_formatted(scenario["context"])

        print(f"Extracted queries: {result.queries_used}")
        print()
        print(result.summary())
        print()
        print(enriched_text)
        print()

        all_results.append({
            "scenario": scenario["name"],
            "memories_found": len(result.memories),
            "avg_relevance": result.avg_relevance,
            "total_latency_ms": result.total_latency_ms,
            "queries_count": len(result.queries_used),
        })

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n{'Scenario':<25} {'Memories':>8} {'Avg Relevance':>14} {'Latency':>10} {'Queries':>8}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['scenario']:<25} {r['memories_found']:>8} {r['avg_relevance']:>14.3f} {r['total_latency_ms']:>9.1f}ms {r['queries_count']:>8}")

    total_memories = sum(r["memories_found"] for r in all_results)
    avg_latency = sum(r["total_latency_ms"] for r in all_results) / len(all_results)
    avg_relevance = sum(r["avg_relevance"] for r in all_results) / len(all_results)
    print("-" * 70)
    print(f"{'TOTALS':<25} {total_memories:>8} {avg_relevance:>14.3f} {avg_latency:>9.1f}ms")
    print()

    # Verdict
    print("=" * 70)
    print("VERDICT: Is Daemon Push Worth Implementing?")
    print("=" * 70)
    print()

    if avg_latency < 100:
        latency_verdict = "EXCELLENT -- sub-100ms latency makes real-time push feasible"
    elif avg_latency < 500:
        latency_verdict = "GOOD -- acceptable for async push (background thread)"
    elif avg_latency < 1000:
        latency_verdict = "MARGINAL -- noticeable delay, needs optimization"
    else:
        latency_verdict = "POOR -- too slow for real-time, needs caching/batching"

    if avg_relevance > 0.7:
        relevance_verdict = "HIGH -- memories are highly relevant to context"
    elif avg_relevance > 0.4:
        relevance_verdict = "MODERATE -- some relevant memories, room to improve"
    else:
        relevance_verdict = "LOW -- queries need tuning for better relevance"

    print(f"  Latency:   {latency_verdict}")
    print(f"  Relevance: {relevance_verdict}")
    print()

    if total_memories > 0 and avg_relevance > 0.3:
        print("  RECOMMENDATION: YES -- implement daemon push as background service")
        print("  Implementation approach:")
        print("    1. Client sends context on each conversation turn")
        print("    2. PushClient extracts queries (embedded in agent middleware)")
        print("    3. Daemon returns relevant memories in background")
        print("    4. Memories injected into system prompt before next LLM call")
        print("    5. Cache results for 5 minutes to avoid re-querying")
    else:
        print("  RECOMMENDATION: NOT YET — improve query extraction first")
        print("  The context analyzer needs better NLP to extract meaningful queries")
    print()


def run_benchmark():
    """Run latency benchmarks."""
    client = PushClient()

    print("=" * 70)
    print("MATHIR DAEMON PUSH -- LATENCY BENCHMARK")
    print("=" * 70)
    print()

    if not client.daemon.ping():
        print("ERROR: Daemon not running")
        sys.exit(1)

    # Benchmark: raw daemon recall
    print("1. Raw daemon recall latency (10 iterations):")
    latencies = []
    for i in range(10):
        start = time.perf_counter()
        result = client.daemon.recall("authentication bug fix", k=5)
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)
    print(f"   Mean: {sum(latencies)/len(latencies):.1f}ms | "
          f"Min: {min(latencies):.1f}ms | Max: {max(latencies):.1f}ms | "
          f"P95: {sorted(latencies)[int(len(latencies)*0.95)]:.1f}ms")
    print()

    # Benchmark: full push with context analysis
    print("2. Full push latency (context -> memories, 5 scenarios):")
    for scenario in DEMO_CONTEXTS:
        start = time.perf_counter()
        result = client.push(scenario["context"])
        elapsed = (time.perf_counter() - start) * 1000
        print(f"   [{scenario['name']:<25}] {elapsed:.1f}ms | "
              f"{len(result.memories)} memories | "
              f"queries: {len(result.queries_used)}")
    print()

    # Benchmark: cache hit
    print("3. Cache hit latency:")
    ctx = DEMO_CONTEXTS[0]["context"]
    # First call (cold)
    start = time.perf_counter()
    client.push(ctx)
    cold = (time.perf_counter() - start) * 1000
    # Second call (warm)
    start = time.perf_counter()
    client.push(ctx)
    warm = (time.perf_counter() - start) * 1000
    print(f"   Cold: {cold:.1f}ms -> Warm: {warm:.1f}ms ({(1-warm/cold)*100:.0f}% faster)")
    print()


def run_interactive():
    """Interactive REPL for testing push."""
    client = PushClient()

    print("=" * 70)
    print("MATHIR DAEMON PUSH -- INTERACTIVE MODE")
    print("=" * 70)
    print("Type your context and press Enter. Commands:")
    print("  /stats  — Show memory stats")
    print("  /save <content> — Save a test memory")
    print("  /quit   — Exit")
    print()

    if not client.daemon.ping():
        print("ERROR: Daemon not running")
        sys.exit(1)

    while True:
        try:
            context = input("\n> Context: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not context:
            continue

        if context == "/quit":
            break
        elif context == "/stats":
            result = client.daemon.call("memory_stats")
            print(json.dumps(result, indent=2))
            continue
        elif context.startswith("/save "):
            content = context[6:]
            result = client.daemon.save(content, "test", "episodic", "interactive-test", 5)
            print(f"Saved: {result.get('memory_id', 'failed')}")
            continue

        enriched, result = client.push_formatted(context)
        print(f"\n{result.summary()}")
        print(f"\n{enriched}")


# --- Main --------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="MATHIR Daemon Push Test")
    parser.add_argument("--context", help="Custom context to test")
    parser.add_argument("--benchmark", action="store_true", help="Run latency benchmarks")
    parser.add_argument("--interactive", action="store_true", help="Interactive REPL mode")
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark()
    elif args.interactive:
        run_interactive()
    elif args.context:
        client = PushClient()
        if not client.daemon.ping():
            print("ERROR: Daemon not running on port 7338")
            sys.exit(1)
        enriched, result = client.push_formatted(args.context)
        print(result.summary())
        print()
        print(enriched)
    else:
        run_demo_scenarios()


if __name__ == "__main__":
    main()
