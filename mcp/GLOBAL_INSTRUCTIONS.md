# MATHIR Global Instructions — Agent Integration Guide

**Copy these instructions into your agent's global instructions file.**
MATHIR works with OpenCode, OpenClaude, Kilo Code, MiMo Code, Claude Code, and any MCP-compatible tool.

---

## MATHIR AUTO-MEMORY — FIRST ACTION EVERY SESSION

**At the very start of EVERY session, BEFORE any other action:**

### Step 1: Find MATHIR

```bash
# Try common locations (first one that works wins):
# - ./mathir_client.py (current project)
# - ~/.config/opencode/bin/mathir_client.py (OpenCode)
# - /path/to/MATHIR/bin/mathir_client.py (custom install)

# Quick check:
python ./mathir_client.py ping 2>/dev/null || \
python ~/.config/opencode/bin/mathir_client.py ping 2>/dev/null || \
echo "MATHIR not found — install from https://github.com/sil3d/MATHIR"
```

### Step 2: Start daemon if not running

```bash
# Find and start daemon:
python ./mathir_daemon.py &  # or full path
```

### Step 3: Recall existing memories

```bash
python [MATHIR_PATH]/mathir_client.py recall "project context" -k 10
```

### Step 4: If no memories exist, create initial project memory

```bash
python [MATHIR_PATH]/mathir_client.py save "PROJECT INIT — DOCTOR-GRADE CODE QUALITY STANDARDS (MANDATORY FOR ALL AGENTS): CORE RULES: 1. NEVER HARDCODE values. 2. NEVER GUESS information. 3. NEVER SKIP error handling. 4. NEVER COPY-PASTE without understanding. 5. NEVER IGNORE security. 6. NEVER COMMIT secrets. 7. NEVER SKIP type hints. 8. NEVER USE deprecated APIs. 9. NEVER ASSUME dependencies exist. 10. NEVER SKIP logging. VERIFICATION: ALWAYS verify code works, check for errors, run tests, validate inputs, check edge cases. BEFORE CLAIMING DONE: Run all tests, check linting, verify fix works, test edge cases, document changes." -a init -t semantic -l doctor-grade-standards -p 10
```

### Step 5: After EVERY task completion, save what was done

```bash
python [MATHIR_PATH]/mathir_client.py save "Task completed: [what was done, result, key decisions]" -a [your_agent_name] -t episodic -l task-[short-description] -p 7
```

### Step 6: At the END of the session, save a summary

```bash
python [MATHIR_PATH]/mathir_client.py save "Session summary: [all tasks done, overall progress, next steps]" -a [your_agent_name] -t episodic -l session-[date] -p 8
```

### Other Commands

```bash
# Fast text search (no embedding, instant)
python [MATHIR_PATH]/mathir_client.py search "query" -k 5

# Memory statistics
python [MATHIR_PATH]/mathir_client.py stats

# Push relevant memories (proactive delivery)
python [MATHIR_PATH]/mathir_client.py push "current context" --auto
```

**The database is created automatically in `.mathir/mathir.db` inside the project directory.**

---

## MCP INTEGRATION — If Your Tool Supports MCP

If you're using an MCP-compatible tool (OpenCode, OpenClaude, Kilo Code, MiMo Code, Claude Code), add MATHIR as an MCP server:

### OpenCode
```json
// ~/.config/opencode/opencode.json
{ "mcpServers": { "mathir": { "url": "http://127.0.0.1:7338/sse" } } }
```

### OpenClaude
```bash
/mcp add mathir http://127.0.0.1:7338/sse
```

### Kilo Code (VS Code)
```
Settings → MCP → Add Server → Remote (HTTP)
URL: http://127.0.0.1:7338/sse
```

### MiMo Code
```json
// ~/.config/mimocode/config.json
{ "mcp": { "mathir": { "type": "remote", "url": "http://127.0.0.1:7338/sse" } } }
```

### Claude Desktop
```json
// ~/Library/Application Support/Claude/claude_desktop_config.json (Mac)
// %APPDATA%\Claude\claude_desktop_config.json (Windows)
{ "mcpServers": { "mathir": { "url": "http://127.0.0.1:7338/sse" } } }
```

### Available MCP Tools (6)

| Tool | Description |
|------|-------------|
| `memory_save` | Save a memory block with embedding |
| `memory_recall` | Search memories by semantic similarity |
| `memory_smart_search` | Hybrid search (vector + full-text) |
| `memory_stats` | Get memory statistics |
| `memory_delete` | Soft-delete a memory |
| `memory_push` | Proactive memory delivery based on context |

---

## ERROR HANDLING RULES — MANDATORY FOR ALL AGENTS

### NEVER say "pre-existing error" and skip it
- If you find an error during your work, FIX IT or explain WHY you can't
- "Pre-existing error" is NOT an excuse — it's a cop-out
- If the error blocks your task, you MUST fix it or escalate with full context

### NEVER delete or comment out code to "fix" errors
- Deleting code is NOT fixing — it's hiding the problem
- Commenting out code is NOT fixing — it's technical debt
- You MUST find the ROOT CAUSE and fix it properly

### What to do when you find an error:
1. **Understand the error** — read the error message, find the source
2. **Try to fix it** — research, test different approaches
3. **If you can't fix it** — document what you tried and WHY it failed
4. **Escalate properly** — give the orchestrator full context, not just "pre-existing error"

### Forbidden phrases:
- "This is a pre-existing error, not mine"
- "I'll just comment this out"
- "This error was already there"
- "Not my problem"

### Required phrases:
- "I found this error: [error]. I tried: [solutions]. It failed because: [reason]. I need help with: [specific ask]"
- "This error blocks my task because: [reason]. Here's what I attempted: [list]"

---

## MATHIR EVOLUTION — SELF-IMPROVEMENT & SKILL CREATION

**MATHIR is not just storage. It's a BRAIN that learns and evolves.**

### The Evolution Protocol

Every agent MUST actively build the collective intelligence:

1. **Detect patterns** — when you see the same error, same problem, same solution 2+ times
2. **Create a skill** — document the pattern with FULL detail
3. **Save to MATHIR** — so future sessions benefit from this learning
4. **Improve existing skills** — when you find a better solution, UPDATE the skill

### When to Create a Skill

| Signal | Action |
|--------|--------|
| Same error appears 2+ times in different sessions | Create skill with root cause + fix |
| You spend > 5 min debugging something | Document the debugging process as a skill |
| You discover a non-obvious solution | Save it — others will hit this |
| A test reveals a recurring pattern | Create a skill for that pattern |
| You learn a new tool/API quirk | Document it with examples |
| You find a better way to do something | UPDATE the existing skill |
| You write a plan | Save the plan + why this approach |
| You make an architecture decision | Save decision + alternatives rejected |
| You do a code review | Save findings + fixes applied |
| You refactor something | Save before/after + why it's better |
| You set up a new tool | Save the setup steps + gotchas |
| You learn a new pattern | Save with examples + when to use |

### Skill Format — FULL DETAIL REQUIRED

When creating a skill, you MUST include ALL of these:

```bash
python [MATHIR_PATH]/mathir_client.py save "SKILL: [Problem Title]

PROBLEM:
- What: [Exact description]
- When: [When does it happen?]
- Symptoms: [Error look like?]
- Affected: [Files, components?]

ROOT CAUSE:
- Why: [Technical explanation]
- Mechanism: [Underlying mechanism?]

SOLUTION:
- Step 1: [Exact command or code]
- Step 2: [Next step]
- Verification: [How to verify]

PREVENTION:
- How to avoid: [What to do differently]
- Code pattern: [Pattern that prevents this]
- Checklist: [Quick check before shipping]

EXAMPLES:
- Bad: [Code that causes problem]
- Good: [Code that avoids problem]
- Fix: [Code that fixes problem]

TAGS: [error, rust, tauri, auth, etc.]" \
  -a [your_agent_name] -t semantic -l skill-[problem-slug] -p 9
```

### Skill Quality Rules

- **NO shortcuts** — write the FULL detail, even if it takes more tokens
- **Include code examples** — bad, good, and fix versions
- **Include the WHY** — not just "do X" but "do X because Y"
- **Include verification** — how to confirm the fix works
- **Include prevention** — how to avoid this in the future
- **Use tags** — so skills can be found by topic

### When Dispatching Agents — FULL CONTEXT

Before dispatching an agent, the orchestrator MUST:
1. Recall relevant skills from MATHIR
2. Include skill content in the agent's task description
3. Tell the agent: "Here are skills from past sessions: [skills]"

### Save Throughout the Session — NOT Just at the End

| When | What to save |
|------|-------------|
| **Start** | Recall existing skills, project context |
| **After discovering something new** | Save the discovery immediately |
| **After fixing a bug** | Save the skill (problem + solution + prevention) |
| **After a test reveals a pattern** | Save the pattern |
| **After completing a task** | Save what was done + what you learned |
| **After writing a plan** | Save the plan + why you chose this approach |
| **After an architecture decision** | Save the decision + alternatives considered |
| **After a code review** | Save what was found + how it was fixed |
| **After a failed attempt** | Save what you tried + why it failed (STILL valuable) |
| **At end** | Session summary + overall learnings |

**MATHIR learns from EVERYTHING** — mistakes, tests, discoveries, plans, decisions, architecture, even failed attempts.

### The Evolution Loop

```
Session N:
  1. Recall skills from MATHIR
  2. Work on task
  3. Detect new patterns
  4. Create/update skills
  5. Save everything important

Session N+1:
  1. Recall skills (including new ones from N)
  2. Work with MORE knowledge
  3. Detect MORE patterns
  4. Create/update MORE skills
  5. System gets SMARTER every session
```

---

## MEMORY TYPES — When to Use Each

| Type | When | Example |
|------|------|---------|
| `working_memory` | Active task, current bug | "Bug: null pointer in auth.py:42" |
| `episodic` | After completing a task | "Fixed login refresh token bug in PR #42" |
| `semantic` | Discovery about the project | "This project uses JWT tokens in Authorization header" |
| `procedural` | Workflow that works well | "How to debug auth issues: 1. Check token, 2. Check CORS..." |

---

## Anti-Patterns — NEVER DO THIS

- "I'll remember this for next time" → **SAVE IT TO MATHIR**
- "This is obvious, no need to document" → **Document it anyway**
- "I'll fix it and move on" → **Save the skill too**
- "The error is fixed, that's enough" → **Save prevention + root cause**
- "Others will figure it out" → **They won't — save the skill**

---

## Quick Reference

```bash
# Ping daemon
python [MATHIR_PATH]/mathir_client.py ping

# Save memory
python [MATHIR_PATH]/mathir_client.py save "content" -a agent -t semantic -l label -p 7

# Recall memories
python [MATHIR_PATH]/mathir_client.py recall "query" -k 5

# Search (fast, no embedding)
python [MATHIR_PATH]/mathir_client.py search "query" -k 5

# Stats
python [MATHIR_PATH]/mathir_client.py stats

# Push (proactive delivery)
python [MATHIR_PATH]/mathir_client.py push "context" --auto

# Delete
python [MATHIR_PATH]/mathir_client.py delete [id] --reason "outdated"
```

Replace `[MATHIR_PATH]` with the actual path to your MATHIR installation (e.g., `~/.config/opencode/bin` or `/path/to/MATHIR/bin`).
