---
name: graphite-capture
description: "Pushes OpenClaw session activity into the Graphite knowledge graph via the local graphited daemon."
metadata:
  openclaw:
    emoji: "🧠"
    events:
      - "agent_end"
    requires:
      bins:
        - "node"
    config:
      daemonSocket: "~/.graphite/daemon.sock"
      overflowDir: "~/.graphite/spool_overflow"
      timeoutMs: 2000
    hooks:
      allowConversationAccess: true
---

# Graphite Memory Capture

Subscribes to OpenClaw's `agent_end` lifecycle event with
`allowConversationAccess: true`, joins the session messages into a
transcript, and calls Graphite's `remember` RPC over the local
`graphited` Unix socket. The daemon batches and extracts asynchronously
so the agent loop is never blocked.

## What it does

* On `agent_end`, extracts user/assistant messages from the event
  payload and concatenates them into a single transcript.
* Calls `remember(text, source_id="openclaw://<agent>/<session>", category="Episodic")`.
* Best-effort `flush_spool(source_filter=...)` to drain the just-captured
  fragment immediately rather than waiting on the size threshold.
* If the daemon socket is unreachable, atomically writes the capture to
  `~/.graphite/spool_overflow/` so the daemon's overflow reconciler can
  replay it on next start.

## Configuration

Override defaults under `plugins.entries.graphite-capture.config` in
`~/.openclaw/openclaw.json`:

```json
{
  "plugins": {
    "entries": {
      "graphite-capture": {
        "enabled": true,
        "hooks": { "allowConversationAccess": true },
        "config": {
          "daemonSocket": "~/.graphite/daemon.sock",
          "overflowDir": "~/.graphite/spool_overflow",
          "timeoutMs": 2000
        }
      }
    }
  }
}
```

## Verifying capture is working

After running an OpenClaw session:

```bash
graphite spool status
# Expect: pending or extracting count > 0, then dropping as the daemon
# worker drains the batch.
```

## Caveats

* **Event payload shape.** The exact fields OpenClaw passes to
  `agent_end` are not fully published. The plugin's `extractMessages`
  helper handles four plausible layouts (`conversation.messages`,
  `messages`, `transcript`, top-level `output`); if your install uses a
  different shape, the plugin logs `[graphite-capture] agent_end with no
  extractable messages — skipping` and the user adjusts that one helper.
* **No retries.** If the daemon socket is unreachable, the capture goes
  to the overflow directory rather than retrying mid-event; the
  daemon's reconciler is the durability path.

## Source layout

* `HOOK.md` — this file (manifest)
* `index.js` — entry point, plain Node (no build step)
* `README.md` — installation and verification notes
