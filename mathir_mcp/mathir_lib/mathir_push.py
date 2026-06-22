#!/usr/bin/env python3
"""
MATHIR Push — Context-aware memory retrieval.
Analyzes conversation context to extract relevant queries,
then fetches and deduplicates memories.
"""

import re
import hashlib
import time
import logging
import threading
from collections import OrderedDict
from typing import Optional

log = logging.getLogger("mathir-push")


class ContextAnalyzer:
    """Extracts relevant queries from a conversation context."""

    # Patterns for error detection
    ERROR_PATTERNS = [
        r'(?:error|exception|traceback|failed|failure)[:\s]+(.{10,120})',
        r'(\w+Error)\s*[:]',
        r'(\w+Exception)\s*[:]',
        r'Traceback \(most recent call last\)',
        r'FAILED\s+.*?(\S+\.py:\d+)',
    ]

    # Patterns for file paths
    FILE_PATTERNS = [
        r'[\w/\\.-]+\.(?:py|ts|tsx|js|jsx|rs|go|java|rb|cpp|c|h|hpp)',
        r'(?:src|lib|app|components?|utils?|tests?|modules?)[/\\][\w/\\.-]+',
        r'~/[\w/\\.-]+',
        r'\./[\w/\\.-]+',
    ]

    # Technical keywords that indicate domain
    TECH_KEYWORDS = {
        'auth': ['authentication', 'login', 'session', 'jwt', 'token', 'oauth', 'password'],
        'api': ['endpoint', 'route', 'request', 'response', 'http', 'rest', 'graphql'],
        'database': ['sql', 'query', 'migration', 'schema', 'table', 'postgres', 'sqlite', 'redis'],
        'testing': ['test', 'assert', 'mock', 'fixture', 'coverage', 'jest', 'pytest'],
        'security': ['vulnerability', 'xss', 'csrf', 'injection', 'owasp', 'encrypt'],
        'performance': ['optimize', 'slow', 'latency', 'cache', 'memory', 'cpu', 'benchmark'],
        'ui': ['component', 'render', 'css', 'style', 'layout', 'responsive', 'modal'],
        'build': ['compile', 'bundle', 'webpack', 'vite', 'cargo', 'npm', 'pip'],
    }

    # Intent keywords
    INTENT_KEYWORDS = {
        'fix': ['fix', 'patch', 'repair', 'resolve', 'debug', 'troubleshoot'],
        'implement': ['implement', 'build', 'create', 'add', 'write', 'develop'],
        'refactor': ['refactor', 'restructure', 'clean', 'simplify', 'reorganize'],
        'review': ['review', 'audit', 'check', 'verify', 'validate'],
        'understand': ['explain', 'understand', 'how does', 'why does', 'what is'],
    }

    def extract_queries(self, context: str, max_queries: int = 5) -> list[str]:
        """Extract relevant search queries from conversation context.

        Analyzes the context text for error messages, file names, technical
        keywords, function names, and intent to produce targeted search queries.

        Args:
            context: The conversation context text to analyze.
            max_queries: Maximum number of queries to return (default 5).

        Returns:
            List of search query strings, ordered by likely relevance.
        """
        if not context or not context.strip():
            return []

        queries: list[str] = []
        context_lower = context.lower()

        # 1. Extract error messages — highest priority
        for pattern in self.ERROR_PATTERNS:
            for match in re.finditer(pattern, context, re.IGNORECASE):
                text = match.group(1).strip() if match.lastindex else match.group(0).strip()
                # Clean up and use as query
                clean = re.sub(r'\s+', ' ', text)[:120]
                if clean and clean not in queries:
                    queries.append(clean)

        # 2. Extract file names — high priority
        for pattern in self.FILE_PATTERNS:
            for match in re.finditer(pattern, context):
                fname = match.group(0).strip()
                if fname and fname not in queries:
                    queries.append(fname)

        # 3. Extract function/class names (e.g. function_name(), Class.method())
        func_pattern = r'\b([A-Z][a-zA-Z0-9_]*(?:\.[a-z][a-zA-Z0-9_]*)?\([)]?)'
        for match in re.finditer(func_pattern, context):
            name = match.group(1).rstrip('(')
            if name and name not in queries and len(name) > 2:
                queries.append(name)

        # 4. Detect technical domain and add domain keywords
        detected_domains: list[str] = []
        for domain, keywords in self.TECH_KEYWORDS.items():
            if any(kw in context_lower for kw in keywords):
                detected_domains.append(domain)
                # Use the domain itself + a key keyword as query
                domain_query = f"{domain} {keywords[0]}"
                if domain_query not in queries:
                    queries.append(domain_query)

        # 5. Detect intent
        for intent, keywords in self.INTENT_KEYWORDS.items():
            if any(kw in context_lower for kw in keywords):
                # Combine intent with detected domain for specificity
                if detected_domains:
                    intent_query = f"{intent} {detected_domains[0]}"
                else:
                    intent_query = intent
                if intent_query not in queries:
                    queries.append(intent_query)

        # 6. If still very few queries, extract significant words
        if len(queries) < 2:
            # Tokenize and filter meaningful words
            words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_]{2,}\b', context)
            # Frequency-based selection
            freq: dict[str, int] = {}
            for w in words:
                wl = w.lower()
                # Skip common stop words
                if wl in {'the', 'and', 'for', 'this', 'that', 'with', 'from',
                          'are', 'was', 'were', 'have', 'has', 'had', 'not',
                          'you', 'your', 'its', 'can', 'will', 'just', 'but'}:
                    continue
                freq[wl] = freq.get(wl, 0) + 1
            # Take top 3 by frequency
            top_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:3]
            for word, _ in top_words:
                if word not in queries:
                    queries.append(word)

        result = queries[:max_queries]
        log.info(f"Extracted {len(result)} queries from context ({len(context)} chars)")
        return result


class PushCache:
    """LRU cache with TTL for push results."""

    def __init__(self, ttl_seconds: int = 300, max_size: int = 100):
        """Initialize the cache.

        Args:
            ttl_seconds: Time-to-live for cache entries in seconds (default 300).
            max_size: Maximum number of entries in the cache (default 100).
        """
        self._cache: OrderedDict[str, tuple[float, list]] = OrderedDict()
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._lock = threading.Lock()

    def get(self, context_hash: str) -> Optional[list]:
        """Get cached memories for a context hash.

        Args:
            context_hash: SHA256 hash of the context string.

        Returns:
            Cached memories list if found and not expired, else None.
        """
        with self._lock:
            if context_hash not in self._cache:
                return None

            ts, memories = self._cache[context_hash]
            if time.time() - ts > self._ttl:
                # Expired — remove it
                del self._cache[context_hash]
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(context_hash)
            return memories

    def set(self, context_hash: str, memories: list) -> None:
        """Store memories in cache.

        Args:
            context_hash: SHA256 hash of the context string.
            memories: List of memory dicts to cache.
        """
        with self._lock:
            if context_hash in self._cache:
                self._cache.move_to_end(context_hash)
            elif len(self._cache) >= self._max_size:
                # Evict oldest
                self._cache.popitem(last=False)

            self._cache[context_hash] = (time.time(), memories)

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()

    def stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dict with 'entries', 'max_size', and 'ttl_seconds' keys.
        """
        with self._lock:
            return {
                'entries': len(self._cache),
                'max_size': self._max_size,
                'ttl_seconds': self._ttl,
            }


def context_hash(context: str) -> str:
    """Compute SHA256 hash of a context string.

    Args:
        context: The context string to hash.

    Returns:
        Hex digest of the SHA256 hash.
    """
    return hashlib.sha256(context.encode('utf-8')).hexdigest()[:32]


def deduplicate_memories(memories: list[dict]) -> list[dict]:
    """Deduplicate memories by content hash.

    Uses full SHA256 hex digest (64 hex chars) to prevent hash collision attacks
    where an attacker crafts two different memories with the same truncated hash.

    Args:
        memories: List of memory dicts, each with a 'content' field.

    Returns:
        Deduplicated list preserving original order.
    """
    seen: set[str] = set()
    result: list[dict] = []
    for mem in memories:
        content = mem.get('content', '')
        # Use full SHA256 — 64 hex chars (256 bits) — no truncation
        h = hashlib.sha256(content.encode('utf-8')).hexdigest()
        if h not in seen:
            seen.add(h)
            result.append(mem)
    return result
