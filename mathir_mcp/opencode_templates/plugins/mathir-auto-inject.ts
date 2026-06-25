import type { Plugin } from "@opencode-ai/plugin";

/**
 * MATHIR Auto-Injection Plugin — v8.5.0
 *
 * Hooks into experimental.chat.system.transform to inject relevant memories
 * into the system prompt at session start and during the session.
 *
 * Requires: MATHIR unified server running on port 7338.
 */

const MATHIR_URL = process.env.MATHIR_URL || "http://127.0.0.1:7338";
const DEBUG = process.env.MATHIR_PLUGIN_DEBUG === "1";

// Track which sessions already got the startup injection
const startupInjected = new Set<string>();
// Track last injection time per session to avoid re-injecting too often
const lastInjection = new Map<string, number>();
const INJECTION_COOLDOWN_MS = 30_000; // 30s between injections

let projectPath: string | null = null;

async function fetchContext(task: string, k = 8): Promise<string | null> {
  try {
    const url = `${MATHIR_URL}/api/context?task=${encodeURIComponent(task)}&k=${k}`;
    const res = await fetch(url, { signal: AbortSignal.timeout(5000) });
    if (!res.ok) return null;
    const data = await res.json() as { context?: string; error?: string };
    if (data.error) return null;
    return data.context || null;
  } catch (e) {
    if (DEBUG) console.error(`[mathir] fetchContext failed:`, (e as Error).message);
    return null;
  }
}

async function fetchStats(): Promise<string | null> {
  try {
    const res = await fetch(`${MATHIR_URL}/api/stats`, { signal: AbortSignal.timeout(5000) });
    if (!res.ok) return null;
    const data = await res.json() as { total_memories?: number; by_tier?: Record<string, number> };
    if (!data.total_memories) return null;
    const tiers = Object.entries(data.by_tier || {})
      .map(([t, n]) => `${t}: ${n}`)
      .join(", ");
    return `MATHIR: ${data.total_memories} memories stored (${tiers})`;
  } catch {
    return null;
  }
}

export default function mathirPlugin(): Plugin {
  return {
    name: "mathir-auto-inject",
    hooks: {
      // ── session.started: inject startup context ──
      "session.started": async (input) => {
        const sid = input.sessionID;
        if (!sid) return;
        if (startupInjected.has(sid)) return;
        startupInjected.add(sid);
        // Store project path for later
        if (input.projectPath) projectPath = input.projectPath;
      },

      // ── experimental.chat.system.transform: inject memories ──
      "experimental.chat.system.transform": async (input, output) => {
        const sid = input.sessionID;
        if (!sid) return;
        if (!Array.isArray(output.system)) return;

        const now = Date.now();
        const lastTime = lastInjection.get(sid) || 0;
        const isFirstTurn = !startupInjected.has(sid);

        // Always inject on first turn, then respect cooldown
        if (!isFirstTurn && (now - lastTime) < INJECTION_COOLDOWN_MS) return;
        lastInjection.set(sid, now);

        // Build the task description from user message
        const userMessage = input.messages
          ?.filter((m: any) => m.role === "user")
          .map((m: any) => m.content)
          .join(" ")
          .slice(0, 300) || "general context";

        // Fetch relevant memories
        const context = await fetchContext(userMessage, isFirstTurn ? 8 : 5);
        if (context) {
          output.system.push(`<mathir-auto-injection>\n${context}\n</mathir-auto-injection>`);
          if (DEBUG) console.log(`[mathir] Injected ${context.length} chars for: ${userMessage.slice(0, 50)}`);
        }
      },

      // ── session.destroyed: cleanup ──
      "session.destroyed": async (input) => {
        const sid = input.sessionID;
        if (sid) {
          startupInjected.delete(sid);
          lastInjection.delete(sid);
        }
      },
    },
  };
}
