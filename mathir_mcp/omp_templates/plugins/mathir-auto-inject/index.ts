// MATHIR Tier-A auto-injection hook for OMP.
//
// Calls the MATHIR daemon at MATHIR_DAEMON_URL (default http://127.0.0.1:7338)
// on every `before_agent_start` event and injects the recalled context + active
// guardrails as a pre-agent message. Mirrors the behaviour of the Claude Code
// Python hook (bin/claude_code_hook.py) but native to OMP's TypeScript hook
// subsystem, so it is loaded by the default extension runner with no extra
// wiring.
//
// Project scoping: forwards project=ctx.cwd basename + strict=true so the
// daemon's vector recall filters results to the active project (or 'global'
// for cross-project handoff memories). The OMP session launched from
// C:/Users/princ/.omp/dev/foo will only see foo- and global-tagged memories,
// never anything tagged with another project's name. Disabling strict
// reverts to the old broader recall behaviour. Set MATHIR_INJECT_STRICT=false
// to opt out.
//
// Fail-open: any error (daemon down, network, timeout, malformed payload) is
// logged via pi.logger and the agent proceeds without injection. Matches
// Claude Code's hook semantics, where a slow/missing hook must never block
// the user.

interface ContextResponse {
  context?: string;
  guardrails?: Array<{ label?: string; content?: string }>;
  guardrails_count?: number;
  tiers?: Record<string, number>;
  total?: number;
  task?: string;
}

interface PiAPI {
  on: (event: string, handler: (event: unknown, ctx: { cwd?: string }) => Promise<unknown> | unknown) => void;
  logger: {
    info?: (msg: string) => void;
    warn?: (msg: string) => void;
    error?: (msg: string) => void;
  };
}

const DAEMON = process.env.MATHIR_DAEMON_URL ?? "http://127.0.0.1:7338";
const K = parseInt(process.env.MATHIR_INJECT_K ?? "8", 10);
const TIMEOUT_MS = parseInt(process.env.MATHIR_HOOK_TIMEOUT_MS ?? "3000", 10);
const STRICT = (process.env.MATHIR_INJECT_STRICT ?? "true").toLowerCase() !== "false";

// MATHIR mini instructions block: always injected as a pre-agent message so the
// agent knows it must maintain the shared memory DB, even when the daemon is
// down (fail-open keeps the etiquette available). Mirrors the "DB Hygiene"
// section of GLOBAL_INSTRUCTIONS.md (v8.9.8) in compact form.
const MATHIR_INSTRUCTIONS = [
  "MATHIR memory etiquette — keep the shared memory DB clean:",
  "1. Save what you learn (memory_save via MCP tools if available).",
  "2. Dedupe before saving (memory_consolidate dry_run) — reuse existing memory_ids.",
  "3. Repair broken memories (memory_delete + corrected memory_save, or memory_promote).",
  "4. Orient via the latest final-conclusion/handoff memories before trusting older findings.",
  "5. Housekeep at session end (memory_consolidate + memory_build_links).",
].join("\n");

// God Mode: relay + registration nudge (mirrors the opencode plugin's
// checkGodRelay/checkGodRegistration and claude_code_hook.py). Runs on the
// same before_agent_start event as the context injection — OMP's hook API
// has no per-turn message event with a return value, so the poll happens at
// every agent start (session or subagent) instead of every chat turn.
const GOD_AGENT_NAME = process.env.MATHIR_GOD_AGENT_NAME ?? "omp";
const godSeenTasks = new Set<string>();
let godRegistered = false;

// v8.9.8 — per-turn re-injection while the agent is thinking.
// OMP only returns message-bearing results from before_agent_start, so the
// context is frozen for the whole turn. before_provider_request lets us
// re-inject it into EVERY provider request (including mid-thinking steps).
// We skip the re-injection right after a before_agent_start (its custom
// message is already in the turn history), then resume after the cooldown.
const PROVIDER_REINJECT_COOLDOWN_MS = parseInt(
  process.env.MATHIR_PROVIDER_REINJECT_COOLDOWN_MS ?? "5000", 10);
let lastAgentStart = 0;

function projectFromCwd(cwd: string | undefined): string {
  if (!cwd) return "global";
  const normalized = cwd.replace(/\\/g, "/").replace(/\/+$/, "");
  const last = normalized.split("/").filter(Boolean).pop();
  return last && last.length > 0 ? last : "global";
}

async function postJson(urlPath: string, body: Record<string, unknown>): Promise<unknown | null> {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), TIMEOUT_MS);
  try {
    const res = await fetch(`${DAEMON}${urlPath}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: ctrl.signal,
    });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  } finally {
    clearTimeout(t);
  }
}

async function fetchContext(task: string, project: string, strict: boolean): Promise<ContextResponse | null> {
  const body: Record<string, unknown> = { task, k: K, project };
  if (strict) body.strict = true;
  return (await postJson("/api/context", body)) as ContextResponse | null;
}

// ── God Mode: relay + registration nudge ──
async function checkGodRelay(project: string, cwd: string): Promise<string | null> {
  const data = (await postJson("/api/god/poll", {
    agent: GOD_AGENT_NAME,
    status: "pending",
    project,
    cwd,
  })) as { task?: { memory_id?: string; content?: string; label?: string } } | null;
  const task = data?.task;
  if (!task || !task.memory_id || !task.content) return null;
  if (godSeenTasks.has(task.memory_id)) return null;

  // Ack server-side immediately (see mathir_server.py's /api/god/ack docstring
  // for why: /api/god/poll always returns the single oldest pending task,
  // so without an ack the same message blocks every task behind it forever).
  await postJson("/api/god/ack", { memory_id: task.memory_id, status: "delivered", project, cwd });

  godSeenTasks.add(task.memory_id);
  // Same breakout-token strip as mathir_sanitize.py -- this content came
  // from another agent, treat it as untrusted the same way recalled
  // memory content is treated.
  const safe = task.content.replace(/<\/mathir-/g, "").replace(/<mathir-/g, "").slice(0, 8000);
  return (
    `<mathir-god-message to="${GOD_AGENT_NAME}" label="${task.label || ""}">\n` +
    `A message is waiting from another MATHIR God Mode agent:\n${safe}\n</mathir-god-message>`
  );
}

async function checkGodRegistration(project: string, cwd: string): Promise<string | null> {
  if (godRegistered) return null;
  const data = (await postJson("/api/god/agents", { project, cwd })) as { agents?: Array<{ name?: string }> } | null;
  const names = new Set((data?.agents || []).map((a) => a.name));
  if (names.has(GOD_AGENT_NAME)) {
    godRegistered = true;
    return null;
  }
  return (
    "<mathir-god-registration-needed>\n" +
    `You are not yet registered as a MATHIR God Mode worker under the name "${GOD_AGENT_NAME}". ` +
    "If the user asks you to coordinate with other agents/terminals, or if you want to be discoverable " +
    "by an orchestrator, call the mathir_god_agent MCP tool now with no arguments to start " +
    "self-identification, then call it again with your honest name/capabilities/introduction. " +
    "This is optional -- only do it if multi-agent coordination is relevant to what you are working on right now.\n" +
    "</mathir-god-registration-needed>"
  );
}

function extractTask(event: unknown): string {
  const e = event as Record<string, unknown>;
  const candidates = [e?.prompt, e?.userMessage, e?.message, e?.input];
  for (const c of candidates) {
    if (typeof c === "string" && c.trim()) return c.slice(0, 2000);
  }
  if (Array.isArray(e?.messages)) {
    for (let i = (e.messages as unknown[]).length - 1; i >= 0; i--) {
      const m = (e.messages as Array<Record<string, unknown>>)[i];
      const role = m?.role;
      const content = m?.content;
      if (role === "user" && typeof content === "string" && content.trim()) {
        return content.slice(0, 2000);
      }
      if (role === "user" && Array.isArray(content)) {
        const text = (content as Array<Record<string, unknown>>)
          .map((c) => (typeof c?.text === "string" ? c.text : ""))
          .join("\n")
          .trim();
        if (text) return text.slice(0, 2000);
      }
    }
  }
  return "agent turn";
}

export default function mathirAutoInject(pi: PiAPI): void {
  pi.on("before_agent_start", async (event, ctx) => {
    const task = extractTask(event);
    const cwd = ctx?.cwd ?? process.cwd();
    const project = projectFromCwd(cwd);
    lastAgentStart = Date.now(); // mark turn start for before_provider_request dedup

    const [ctx_res, godBlock, regBlock] = await Promise.all([
      fetchContext(task, project, STRICT),
      checkGodRelay(project, cwd),
      checkGodRegistration(project, cwd),
    ]);

    // Instructions always come first; god blocks are displayed as their own
    // message type, so keep them prominent right after.
    const parts: string[] = [MATHIR_INSTRUCTIONS];
    if (godBlock) parts.push(godBlock);
    if (regBlock) parts.push(regBlock);
    if (ctx_res?.context) parts.push(ctx_res.context);
    const content = parts.join("\n\n");

    const guardrailCount = ctx_res?.guardrails_count ?? ctx_res?.guardrails?.length ?? 0;
    const memCount = ctx_res?.total ?? 0;
    pi.logger.info?.(
      `mathir: injecting instructions + ${guardrailCount} guardrails + ${memCount} memories (project=${project}, cwd=${cwd}, strict=${STRICT})`,
    );

    return {
      message: {
        customType: "mathir-auto-injection",
        content,
        display: true,
        details: {
          source: "mathir-daemon",
          daemon: DAEMON,
          project,
          cwd,
          strict: STRICT,
          guardrails: ctx_res?.guardrails ?? [],
          tiers: ctx_res?.tiers ?? {},
          k: K,
        },
        attribution: "MATHIR auto-injection",
      },
    };
  });

  // ── v8.9.8: re-inject context into every provider request ──
  // Fires for EVERY request to the model provider, including the steps the
  // agent takes while "thinking" (tool calls → next request → ...). We
  // prepend a system message with the recalled context so the model keeps
  // relevant memory visible for the whole turn, not just at prompt submit.
  // Fail-open: any error or a payload without .messages leaves the request
  // untouched.
  pi.on("before_provider_request", async (event, ctx) => {
    const payload = (event as Record<string, unknown>)?.payload;
    if (!payload || typeof payload !== "object") return;
    const messages = (payload as Record<string, unknown>).messages;
    if (!Array.isArray(messages)) return; // not a request body we can enrich

    const now = Date.now();
    if (now - lastAgentStart < PROVIDER_REINJECT_COOLDOWN_MS) return; // custom msg already in history

    // Already injected into this exact request body?
    const already = (messages as Array<Record<string, unknown>>).some(
      (m) => typeof m?.content === "string" && (m.content as string).includes("<mathir-auto-injection>"),
    );
    if (already) return;

    const task = extractTask({ messages });
    const cwd = ctx?.cwd ?? process.cwd();
    const project = projectFromCwd(cwd);

    const res = await fetchContext(task, project, STRICT);
    if (!res?.context) return;

    (messages as Array<Record<string, unknown>>).unshift({
      role: "system",
      content: `<mathir-auto-injection>\n${res.context}\n</mathir-auto-injection>`,
    });
    pi.logger.info?.(
      `mathir: re-injected ${res.context.length} chars into provider request (project=${project})`,
    );
    return payload; // transformed request body replaces the original
  });
}
