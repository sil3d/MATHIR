"""Single shared defense against stored/recalled-content prompt injection.

Every place that puts MATHIR memory text into an LLM prompt -- the daemon's
/api/context (consumed directly by claude_code_hook.py's UserPromptSubmit
hook) and mathir_proxy.py's OpenAI/Anthropic injection routes -- MUST route
through this module instead of reimplementing the same guard. It already
drifted once: mathir_server.py had its own copy with a no-op bug
(`s.replace(tok, tok.strip())`, a no-op since none of these tokens have
whitespace to strip) that silently defeated the whole guard while
mathir_proxy.py's independent copy worked correctly. One implementation
means one place to fix, one place to test, no drift.

Threat model: a memory's content/label, or a user's own query text, may
contain an attacker-chosen string (a poisoned memory written by a
compromised agent, or just an adversarial prompt). That text gets wrapped
in a delimiter block (`<mathir-auto-injection>...</mathir-auto-injection>`
or an OpenAI/Anthropic system-message block) and handed to the model as
"quoted data". Without sanitization, the text could contain the closing
delimiter itself and break out of the block, making subsequent attacker
text read as real instructions/conversation instead of quoted memory.
"""

from __future__ import annotations

# Substrings that must never appear verbatim in injected memory text --
# they'd let stored content break out of the injection block, recurse into
# a template placeholder, or inject chat-template control tokens.
FORBIDDEN_SUBSTRINGS = (
    "</mathir-",           # close a real <mathir-...> injection block early
    "<mathir-",            # forge a fake opening tag / spoof attributes
    "{{MATHIR_CONTEXT}}",  # recurse into the mathir_inject_proxy placeholder
    "<|",                  # chat-template control tokens: <|im_start|>, ...
    "### ",                # forge a fake markdown section header
)

# Cap for the multi-line, quoted-block form (mathir_proxy.py's system-prompt
# injection). The single-line form used by /api/context truncates per-field
# instead (labels/content are already capped upstream), so this constant is
# only consumed by sanitize_block().
DEFAULT_MAX_BYTES = 8 * 1024


def _strip_forbidden(text: str) -> str:
    cleaned = text
    for bad in FORBIDDEN_SUBSTRINGS:
        cleaned = cleaned.replace(bad, "")
    return cleaned


def sanitize_line(text: str) -> str:
    """Single-line form: collapse CR/LF and strip forbidden substrings.

    Use for short fields (task text, memory labels) that get concatenated
    into a single header/summary line, e.g. /api/context's formatted
    "## MATHIR Auto-Context — N memories for: <task>" line.
    """
    if not text:
        return ""
    s = str(text).replace("\r", " ").replace("\n", " ")
    return _strip_forbidden(s)


def sanitize_block(text: str, max_bytes: int = DEFAULT_MAX_BYTES) -> str:
    """Multi-line form: strip forbidden substrings, quote every line as
    data (`> `) so the model treats it as quoted memory rather than fresh
    instructions, and cap total size so a runaway recall can't drown the
    prompt.

    Use for the full context body injected into a system prompt, e.g.
    mathir_proxy.py's <mathir-auto-injection> block.
    """
    if not text:
        return ""
    cleaned = _strip_forbidden(text)
    lines = cleaned.splitlines() or [""]
    quoted = "\n".join(f"> {ln}" if ln else ">" for ln in lines)
    encoded = quoted.encode("utf-8", errors="ignore")
    if len(encoded) > max_bytes:
        quoted = encoded[:max_bytes].decode("utf-8", errors="ignore")
    return quoted
