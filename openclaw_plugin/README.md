# Graphite Memory Capture — OpenClaw hook pack

OpenClaw's CLI calls these "hook packs" (the directory must contain
`HOOK.md` + a `handler`/`index` entry); the docs occasionally also
refer to them as "plugins". Same thing under both names.

Pushes OpenClaw session activity into the local Graphite knowledge graph
via the `graphited` daemon.

## What it does

Subscribes to OpenClaw's `agent_end` lifecycle hook with
`allowConversationAccess: true`, builds a transcript out of the session's
messages, and calls Graphite's `remember` RPC over the daemon's Unix
socket. The daemon batches and extracts asynchronously, so the agent
loop is never blocked.

## Installing

```bash
# 1. Make sure graphited is running.
graphite daemon status

# 2. Wire the MCP server (read access).
graphite mcp install --target openclaw

# 3. Install the plugin from this repo path. Plugin defs are local;
#    OpenClaw symlinks into them with `-l`.
openclaw plugins install -l /path/to/Graphite/openclaw_plugin

# 4. Verify the plugin is loaded.
openclaw plugins list
openclaw plugins inspect graphite-capture --json
```

`graphite mcp install --target openclaw --with-plugin` automates step
3 — see `python_shell/graphite/mcp_install.py`.

## Configuration

Override the defaults under `plugins.entries.graphite-capture.config`
in `~/.openclaw/openclaw.json`:

```json
{
  "plugins": {
    "entries": {
      "graphite-capture": {
        "enabled": true,
        "hooks": { "allowConversationAccess": true },
        "config": {
          "daemonSocket": "~/.graphite/daemon.sock",
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
# Expect: pending count > 0, then dropping as the worker drains.

graphite_status                  # via MCP, from another agent
# "Spool: pending=N extracting=0 extracted=M ..."
```

## Caveats / TODOs

- **Event payload shape.** The exact fields OpenClaw passes to
  `agent_end` are not fully documented at the time of this writing. The
  plugin's `extractMessages` helper handles several plausible layouts
  (`conversation.messages`, `messages`, `transcript`, `output`); if your
  install of OpenClaw uses a different shape, adjust that helper. The
  plugin logs `[graphite-capture] agent_end with no extractable messages
  — skipping` when nothing matches.
- **Plugin SDK fallback.** The file optionally requires
  `@openclaw/plugin-sdk` and falls back to a no-op shim if the SDK is
  unavailable. Either way the manifest's `hooks.subscribe` field tells
  OpenClaw which events to dispatch.
- **No retries.** If the daemon socket is unreachable, the capture is
  dropped on the floor. Intended fallback: PR γ's reconciler scans
  OpenClaw's own transcript directory and replays anything missed.

## Uninstall

```bash
openclaw plugins uninstall graphite-capture
graphite mcp uninstall --target openclaw
```
