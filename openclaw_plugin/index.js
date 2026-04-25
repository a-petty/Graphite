/**
 * Graphite Memory Capture — OpenClaw plugin (Phase 2c, PR β)
 *
 * One job: when an OpenClaw agent finishes a turn (or session — see TODO
 * below), reach into the local graphited daemon over its Unix socket and
 * call `remember` with the session text. The daemon owns batching,
 * extraction, and persistence; we just push raw text fast and exit.
 *
 * No external npm dependencies — only Node's built-in `net`, `fs`, `os`,
 * `path`. That keeps `openclaw plugins install -l <path>` working without
 * a build step.
 *
 * NOTE: OpenClaw's plugin event-payload shape is not fully documented. The
 * code below uses a defensive shape extractor (`extractMessages`) that
 * handles several plausible payload layouts. If your install of OpenClaw
 * surfaces a different shape, the plugin will log it via `api.log.warn`
 * and exit cleanly — the user can then adjust the extractor here.
 */

"use strict";

const crypto = require("crypto");
const fs = require("fs");
const net = require("net");
const os = require("os");
const path = require("path");

const PLUGIN_ID = "graphite-capture";
const OVERFLOW_FORMAT_VERSION = 1;

// --- Defaults; overridden from the OpenClaw config block we registered. -----
const DEFAULT_DAEMON_SOCKET = path.join(os.homedir(), ".graphite", "daemon.sock");
const DEFAULT_OVERFLOW_DIR = path.join(os.homedir(), ".graphite", "spool_overflow");
const DEFAULT_TIMEOUT_MS = 2000;

// --- Helpers -----------------------------------------------------------------

function expandHome(p) {
  if (!p) return p;
  if (p.startsWith("~/")) return path.join(os.homedir(), p.slice(2));
  if (p === "~") return os.homedir();
  return p;
}

/**
 * Best-effort extraction of a transcript-like array of messages from the
 * `agent_end` event payload. Tries several known shapes; returns an array
 * of `{role, text}` objects, possibly empty.
 *
 * TODO(verify-against-runtime): The OpenClaw docs we have describe
 * `agent_end` as a "conversation hook" with `allowConversationAccess`,
 * but the exact field names are not published. Adjust this function once
 * you've seen one real event in the daemon's hook log.
 */
function extractMessages(event) {
  if (!event || typeof event !== "object") return [];

  // Shape A: { conversation: { messages: [...] } }
  if (event.conversation && Array.isArray(event.conversation.messages)) {
    return event.conversation.messages.map(normalizeMessage).filter(Boolean);
  }
  // Shape B: { messages: [...] }
  if (Array.isArray(event.messages)) {
    return event.messages.map(normalizeMessage).filter(Boolean);
  }
  // Shape C: { transcript: [...] }
  if (Array.isArray(event.transcript)) {
    return event.transcript.map(normalizeMessage).filter(Boolean);
  }
  // Shape D: a single trailing assistant message under `output`
  if (typeof event.output === "string" && event.output.trim()) {
    return [{ role: "assistant", text: event.output }];
  }
  return [];
}

function normalizeMessage(m) {
  if (!m || typeof m !== "object") return null;
  const role = m.role || m.type || "unknown";
  // The body could be in any of these fields.
  let text = "";
  if (typeof m.text === "string") text = m.text;
  else if (typeof m.content === "string") text = m.content;
  else if (Array.isArray(m.content)) {
    text = m.content
      .map((part) => (typeof part === "string" ? part : part && part.text) || "")
      .filter(Boolean)
      .join("\n");
  } else if (typeof m.body === "string") text = m.body;
  if (!text.trim()) return null;
  return { role, text };
}

function joinTranscript(messages) {
  return messages
    .map(({ role, text }) => {
      const tag = role === "user" ? "User" : role === "assistant" ? "Assistant" : role;
      return `${tag}: ${text}`;
    })
    .join("\n\n");
}

/**
 * Send one JSON-RPC-shaped request to graphited and read one response.
 * No retries. Resolves to the parsed `result` field, or rejects with
 * the daemon's error message. Caller wraps in try/catch — we never throw
 * synchronously.
 */
function callDaemon({ socketPath, method, params, timeoutMs }) {
  return new Promise((resolve, reject) => {
    let settled = false;
    const conn = net.createConnection(socketPath);
    const timer = setTimeout(() => {
      if (settled) return;
      settled = true;
      conn.destroy();
      reject(new Error(`graphited timeout after ${timeoutMs}ms`));
    }, timeoutMs);

    let buf = "";
    conn.setEncoding("utf8");

    conn.on("error", (err) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      reject(err);
    });

    conn.on("data", (chunk) => {
      buf += chunk;
      const newlineIdx = buf.indexOf("\n");
      if (newlineIdx === -1) return;
      const line = buf.slice(0, newlineIdx);
      try {
        const resp = JSON.parse(line);
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        conn.destroy();
        if (resp && resp.error) {
          reject(new Error(`graphited error ${resp.error.code}: ${resp.error.message}`));
        } else {
          resolve(resp ? resp.result : null);
        }
      } catch (e) {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        conn.destroy();
        reject(new Error(`graphited returned malformed JSON: ${e.message}`));
      }
    });

    conn.on("connect", () => {
      const id = Math.floor(Math.random() * 1e9);
      const req = JSON.stringify({ id, method, params: params || {} }) + "\n";
      conn.write(req);
    });
  });
}

// --- Plugin entry ------------------------------------------------------------

/**
 * OpenClaw's documented entry pattern is `definePluginEntry({ id, name,
 * register(api) { ... } })` with `api.on("agent_end", handler)` for the
 * conversation hooks. We wrap that pattern but stay defensive: if the
 * harness exposes a different shape, the module also exports a CommonJS
 * default that takes `({on, log, config})` directly.
 */

let definePluginEntry;
try {
  // Optional dependency on @openclaw/plugin-sdk. If it's available,
  // we use it for the canonical registration shape.
  // eslint-disable-next-line global-require
  ({ definePluginEntry } = require("@openclaw/plugin-sdk"));
} catch (_e) {
  // Fall back to a passthrough so the same file works in environments
  // where the SDK is provided implicitly via the `register` callback.
  definePluginEntry = (def) => def;
}

function writeOverflow({ overflowDir, payload, log }) {
  try {
    fs.mkdirSync(overflowDir, { recursive: true, mode: 0o700 });
    const hash = crypto
      .createHash("sha256")
      .update(payload.source_id + ":" + payload.text)
      .digest("hex")
      .slice(0, 16);
    const fname = `${Date.now()}-${hash}.json`;
    const tmpPath = path.join(overflowDir, fname + ".tmp");
    const finalPath = path.join(overflowDir, fname);
    const body = {
      version: OVERFLOW_FORMAT_VERSION,
      source_id: payload.source_id,
      text: payload.text,
      category: payload.category,
      project: payload.project,
      entity_hints: payload.entity_hints,
      captured_at: Math.floor(Date.now() / 1000),
    };
    // Write to .tmp then rename so the reconciler never reads a partial file.
    fs.writeFileSync(tmpPath, JSON.stringify(body), { mode: 0o600 });
    fs.renameSync(tmpPath, finalPath);
    log && log.info && log.info(`[${PLUGIN_ID}] daemon unreachable; overflowed to ${finalPath}`);
    return finalPath;
  } catch (err) {
    log && log.warn && log.warn(`[${PLUGIN_ID}] overflow write failed: ${err.message}`);
    return null;
  }
}

function makeHandler({ socketPath, overflowDir, timeoutMs, log }) {
  return async function onAgentEnd(event) {
    try {
      const messages = extractMessages(event);
      if (messages.length === 0) {
        log && log.debug && log.debug(`[${PLUGIN_ID}] agent_end with no extractable messages — skipping`);
        return;
      }

      const text = joinTranscript(messages);
      const sessionId = (event && (event.sessionId || event.session_id || event.id)) || "unknown";
      const agentId = (event && (event.agentId || event.agent_id || event.agent)) || "main";
      const sourceId = `openclaw://${agentId}/${sessionId}`;
      const project = (event && (event.project || event.workspace || event.cwd)) || null;

      const payload = {
        text,
        source_id: sourceId,
        category: "Episodic",
        project: project ? path.basename(String(project)) : null,
        entity_hints: null,
      };

      try {
        await callDaemon({
          socketPath,
          method: "remember",
          params: payload,
          timeoutMs,
        });
      } catch (daemonErr) {
        // Live path failed — drop an overflow JSON so the daemon's
        // reconciler can replay it later. Phase 2c PR γ.
        writeOverflow({ overflowDir, payload, log });
        return; // skip the flush attempt — daemon is unreachable
      }

      // Best-effort flush so the captured session is processed before
      // it ages out of the spool's auto-trigger window. If this fails,
      // the auto-threshold and graphite spool flush are both fallbacks.
      try {
        await callDaemon({
          socketPath,
          method: "flush_spool",
          params: { source_filter: sourceId, limit: 100 },
          timeoutMs,
        });
      } catch (flushErr) {
        log && log.debug && log.debug(`[${PLUGIN_ID}] flush_spool failed (non-fatal): ${flushErr.message}`);
      }

      log && log.info && log.info(`[${PLUGIN_ID}] captured session ${sourceId} (${messages.length} messages)`);
    } catch (err) {
      // Never throw into the OpenClaw plugin host — that would risk
      // disrupting the user's agent loop. Log and move on.
      if (log && log.warn) log.warn(`[${PLUGIN_ID}] capture failed: ${err.message}`);
    }
  };
}

module.exports = definePluginEntry({
  id: PLUGIN_ID,
  name: "Graphite Memory Capture",
  register(api) {
    const cfg = (api && api.config) || {};
    const socketPath = expandHome(cfg.daemonSocket || DEFAULT_DAEMON_SOCKET);
    const overflowDir = expandHome(cfg.overflowDir || DEFAULT_OVERFLOW_DIR);
    const timeoutMs = Number(cfg.timeoutMs || DEFAULT_TIMEOUT_MS);
    const log = api && api.log;

    const handler = makeHandler({ socketPath, overflowDir, timeoutMs, log });

    // TODO(verify-against-runtime): OpenClaw's plugin SDK might expose
    // event subscription as `api.on(eventName, handler)`,
    // `api.subscribe(...)`, or via the manifest's `hooks.subscribe`
    // declaration alone. We try `api.on` first, then fall back to a
    // method registration so the plugin host can find us either way.
    if (api && typeof api.on === "function") {
      api.on("agent_end", handler);
    } else if (api && typeof api.subscribe === "function") {
      api.subscribe("agent_end", handler);
    } else {
      // Manifest-driven: nothing to register imperatively. The host
      // dispatches events to the exported `onAgentEnd` below.
    }

    if (api && typeof api.registerHandler === "function") {
      api.registerHandler("agent_end", handler);
    }

    // Expose the handler for hosts that look up an export rather than
    // a callback registration.
    module.exports.onAgentEnd = handler;
  },
});

// Standalone-export fallback for hosts that don't go through
// `definePluginEntry`. Same handler binding; reads config at call time.
module.exports.onAgentEnd = async function onAgentEndStandalone(event, ctx) {
  const cfg = (ctx && ctx.config) || {};
  const socketPath = expandHome(cfg.daemonSocket || DEFAULT_DAEMON_SOCKET);
  const overflowDir = expandHome(cfg.overflowDir || DEFAULT_OVERFLOW_DIR);
  const timeoutMs = Number(cfg.timeoutMs || DEFAULT_TIMEOUT_MS);
  const log = ctx && ctx.log;
  return makeHandler({ socketPath, overflowDir, timeoutMs, log })(event);
};

// Exported for unit testing from JS / for plugin hosts that introspect.
module.exports._internals = {
  extractMessages,
  joinTranscript,
  callDaemon,
  makeHandler,
  writeOverflow,
};
