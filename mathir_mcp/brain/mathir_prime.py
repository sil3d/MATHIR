"""
MATHIR Pre-Cognitive Priming — Phase 5
======================================
Senses the context BEFORE the user even asks:
- Current working directory
- Open files (via lsof / PowerShell)
- Git branch + recent commits
- Recent file modifications

Builds a "priming context" that gets injected alongside the user message recall.

This is what your reticular activating system does — filters what matters
based on environmental cues, before conscious attention kicks in.
"""
import os
import sys
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Optional


def get_cwd() -> str:
    """Current working directory."""
    try:
        return os.getcwd()
    except Exception:
        return ""


def get_git_context(cwd: str) -> Dict:
    """Get git branch + last commit."""
    if not cwd:
        return {}
    try:
        # Branch
        r = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            cwd=cwd, capture_output=True, text=True, timeout=2
        )
        branch = r.stdout.strip() if r.returncode == 0 else None
        
        # Last commit
        r = subprocess.run(
            ['git', 'log', '-1', '--oneline'],
            cwd=cwd, capture_output=True, text=True, timeout=2
        )
        last_commit = r.stdout.strip() if r.returncode == 0 else None
        
        # Repo name (basename)
        r = subprocess.run(
            ['git', 'rev-parse', '--show-toplevel'],
            cwd=cwd, capture_output=True, text=True, timeout=2
        )
        repo = Path(r.stdout.strip()).name if r.returncode == 0 else None
        
        return {
            'branch': branch,
            'last_commit': last_commit,
            'repo': repo
        }
    except Exception:
        return {}


def get_recent_files(cwd: str, hours: int = 24) -> List[str]:
    """Files modified in last N hours."""
    if not cwd or not Path(cwd).exists():
        return []
    try:
        import time
        cutoff = time.time() - (hours * 3600)
        recent = []
        for root, dirs, files in os.walk(cwd):
            # Skip common ignore dirs
            dirs[:] = [d for d in dirs if d not in ('.git', 'node_modules', '__pycache__', '.venv', 'venv', 'target', 'dist', 'build')]
            for f in files:
                fp = Path(root) / f
                try:
                    if fp.stat().st_mtime > cutoff:
                        recent.append(str(fp.relative_to(cwd)))
                except (OSError, ValueError):
                    continue
            if len(recent) > 20:  # Cap
                break
        return recent[:20]
    except Exception:
        return []


def get_open_files() -> List[str]:
    """Currently open files (best-effort, OS-specific)."""
    try:
        if sys.platform == 'win32':
            # PowerShell: Get-Process | Where-Object {$_.MainWindowTitle}
            r = subprocess.run(
                ['powershell', '-NoProfile', '-Command', 
                 "(Get-Process | Where-Object {$_.MainWindowTitle -ne ''} | Select-Object -First 5 MainWindowTitle)"],
                capture_output=True, text=True, timeout=3
            )
            # Hard to extract file paths from window titles, skip for now
            return []
        else:
            # Linux: lsof
            r = subprocess.run(
                ['lsof', '-c', 'python', '-t'],
                capture_output=True, text=True, timeout=2
            )
            return [Path(f).name for f in r.stdout.split('\n') if f.endswith('.py')]
    except Exception:
        return []


def build_priming_context() -> Dict:
    """
    Build the priming context from environmental cues.
    Returns a dict that can be added to the recall query for better results.
    """
    cwd = get_cwd()
    
    context = {
        'cwd': cwd,
        'project': Path(cwd).name if cwd else None,
        'git': get_git_context(cwd) if cwd else {},
        'recent_files': get_recent_files(cwd) if cwd else [],
    }
    
    return context


def format_for_injection(context: Dict) -> str:
    """Format priming context as a short string for system prompt injection."""
    parts = []
    
    if context.get('project'):
        parts.append(f"**Project:** {context['project']}")
    
    git = context.get('git', {})
    if git.get('branch'):
        parts.append(f"**Git:** `{git['branch']}` — {git.get('last_commit', 'no commit')}")
    
    if context.get('recent_files'):
        files_str = ", ".join(context['recent_files'][:5])
        parts.append(f"**Recent files:** {files_str}")
    
    if not parts:
        return ""
    
    return "**📍 Context (auto-sensed):** " + " | ".join(parts)


if __name__ == "__main__":
    ctx = build_priming_context()
    print(json.dumps(ctx, indent=2, default=str))
    print()
    print(format_for_injection(ctx))
