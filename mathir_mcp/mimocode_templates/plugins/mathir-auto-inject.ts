import type { Plugin } from "@mimo-ai/plugin";
import { spawn } from "child_process";
import * as os from "os";
import * as path from "path";

/**
 * MATHIR Auto-Injection Plugin for mimocode — v8.5.0
 *
 * Port of the opencode plugin. Hooks into experimental.chat.system.transform
 * to inject relevant memories into the system prompt at session start and
 * during the session.
 *
 * Features:
 * - Health check on session start
 * - Auto-restart if server is down
 * - Retry logic with exponential backoff
 * - Graceful degradation if server unavailable
 * - Portable paths (resolves via $HOME, no hardcoded usernames)
 */

const HOME = os.homedir();
const MATHIR_URL = process.env.MATHIR_URL || "http://127.0.0.1:7338";
const MATHIR_SERVER = process.env.MATHIR_SERVER ||
  path.join(HOME, ".config", "mimocode", "bin", "mathir_server.py");
const MATHIR_WORKDIR = process.env.MATHIR_WORKDIR || HOME;
const DEBUG = process.env.MATHIR_PLUGIN_DEBUG === "1";

// Track which sessions already got the startup injection
const startupInjected = new Set<string>();
// Track last injection time per session to avoid re-injecting too often
const lastInjection = new Map<string, number>();
const INJECTION_COOLDOWN_MS = 30_000; // 30s between injections

// Server state
let serverRunning = false;
let serverStarting = false;
let lastHealthCheck = 0;
const HEALTH_CHECK_INTERVAL_MS = 60_000; // 1 min between health checks

let projectPath: string | null = null;

// ── Health check ──
async function checkHealth(): Promise<boolean> {
  const now = Date.now();
  if (now - lastHealthCheck < HEALTH_CHECK_INTERVAL_MS && serverRunning) {
    return true;
  }
  lastHealthCheck = now;

  try {
    const res = await fetch(`${MATHIR_URL}/health`, { signal: AbortSignal.timeout(3000) });
    if (res.ok) {
      serverRunning = true;
      return true;
    }
  } catch {
    // Server not responding
  }
  serverRunning = false;
  return false;
}

// ── Auto-start server ──
function startServer(): void {
  if (serverStarting) return;
  serverStarting = true;

  if (DEBUG) console.log("[mathir] Starting server...");

  try {
    const proc = spawn("python", [MATHIR_SERVER], {
      cwd: MATHIR_WORKDIR,
      detached: true,
      stdio: "ignore",
    });

    proc.on("error", (err) => {
      if (DEBUG) console.error("[mathir] Server start failed:", err.message);
      serverStarting = false;
    });

    proc.on("exit", () => {
      serverStarting = false;
    });

    // Detach so it survives plugin restarts
    proc.unref();

    // Server needs ~30s to load embedder
    if (DEBUG) console.log("[mathir] Server starting (PID: " + proc.pid + "), waiting 35s...");
  } catch (e) {
    if (DEBUG) console.error("[mathir] Spawn error:", (e as Error).message);
    serverStarting = false;
  }
}

// ── Fetch with retry ──
async function fetchWithRetry(url: string, retries = 2): Promise<Response | null> {
  for (let i = 0; i <= retries; i++) {
    try {
      const res = await fetch(url, { signal: AbortSignal.timeout(5000) });
      if (res.ok) return res;
      if (res.status >= 500 && i < retries) {
        // Server error, retry after delay
        await new Promise(r => setTimeout(r, 1000 * (i + 1)));
        continue;
      }
      return null;
    } catch (e) {
      if (i < retries) {
        await new Promise(r => setTimeout(r, 1000 * (i + 1)));
        continue;
      }
      return null;
    }
  }
  return null;
}

async function fetchContext(task: string, k = 8): Promise<string | null> {
  // Health check first
  const healthy = await checkHealth();
  if (!healthy) {
    if (DEBUG) console.log("[mathir] Server unhealthy, attempting restart...");
    startServer();
    // Wait for server to start (max 35s)
    for (let i = 0; i < 7; i++) {
      await new Promise(r => setTimeout(r, 5000));
      if (await checkHealth()) break;
    }
  }

  try {
    const url = `${MATHIR_URL}/api/context?task=${encodeURIComponent(task)}&k=${k}`;
    const res = await fetchWithRetry(url);
    if (!res) return null;
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
    const res = await fetchWithRetry(`${MATHIR_URL}/api/stats`);
    if (!res) return null;
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
      // ── session.started: health check + inject startup context ──
      "session.started": async (input) => {
        const sid = input.sessionID;
        if (!sid) return;

        // Store project path
        if (input.projectPath) projectPath = input.projectPath;

        // Health check on first session
        if (!startupInjected.has(sid)) {
          const healthy = await checkHealth();
          if (!healthy && DEBUG) {
            console.log("[mathir] Server not running on session start, attempting auto-start...");
            startServer();
          }
        }

        if (startupInjected.has(sid)) return;
        startupInjected.add(sid);
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
          .map((m: any) => contentToString(m.content))
          .join(" ")
          .slice(0, 300) || "general context";

        // Fetch relevant memories
        const context = await fetchContext(userMessage, isFirstTurn ? 8 : 5);
        if (context) {
          output.system.push(`<mathir-auto-injection>\n${context}\n</mathir-auto-injection>`);
          if (DEBUG) console.log(`[mathir] Injected ${context.length} chars for: ${userMessage.slice(0, 50)}`);
        } else if (DEBUG) {
          console.log(`[mathir] No context injected (server may be starting)`);
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

// Helper: message content may be string or array of parts
function contentToString(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((p: any) => (typeof p === "string" ? p : (p?.text ?? "")))
      .join(" ");
  }
  return "";
}
