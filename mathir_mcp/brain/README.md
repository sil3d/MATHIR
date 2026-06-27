# brain/ — RETIRED (v9.0)

This directory has been retired as part of the v9.0 universal unification.

All canonical implementations now live in `mathir_mcp/mathir_lib/`:
- `mathir_brain.py` — merged from brain/ (portable paths, HTTP probe)
- `mathir_watchdog.py` — merged (backoff + PID lock + `--kill`)
- `mathir_inject_proxy.py` — canonical (includes prompt-injection sanitization)

The v8 brain/ originals are preserved in `/_deprecated/brain_v8/`.
