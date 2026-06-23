#!/usr/bin/env python3
"""
Add GitHub topics to sil3d/MATHIR repository.
Run this after authenticating: gh auth login
Or set GH_TOKEN environment variable.
"""
import subprocess
import sys

TOPICS = [
    # Primary (most important)
    "llm-memory",
    "memory-augmented",
    "cognitive-memory",
    "mcp",
    "model-context-protocol",
    "ai-agent",
    "rag",
    "knowledge-graph",
    "open-source",
    "mit-license",
    # Technical
    "sqlite",
    "local-ai",
    "edge-ai",
    # Hardware
    "jetson",
    "raspberry-pi",
    # Conceptual
    "neuroscience",
    "ebbinghaus",
    # Security
    "prompt-injection-detection",
    "anomaly-detection",
    # Alternative (for SEO)
    "vector-database",
]

def main():
    print(f"Adding {len(TOPICS)} topics to sil3d/MATHIR...")
    cmd = ["gh", "repo", "edit", "sil3d/MATHIR"]
    for topic in TOPICS:
        cmd.extend(["--add-topic", topic])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Topics added successfully!")
        print("\nTopics added:")
        for t in TOPICS:
            print(f"  - {t}")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        print("\nMake sure you're authenticated:")
        print("  gh auth login")
        print("Or set the GH_TOKEN environment variable.")
        sys.exit(1)
    except FileNotFoundError:
        print("gh CLI not found. Install from: https://cli.github.com/")
        sys.exit(1)

if __name__ == "__main__":
    main()