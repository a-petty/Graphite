# Graphite Workflow Templates

Workflow templates teach Claude how to orchestrate multi-MCP-server data flows
into Graphite. Each workflow describes how to read from a source (Slack, email,
calendar, Notion) and ingest the content using `graphite_ingest_text()`.

## Available Workflows

| Workflow | Source | Description |
|----------|--------|-------------|
| `ingest-slack` | Slack MCP server | Ingest channel messages and threads |
| `ingest-email` | Gmail / Outlook MCP server | Ingest email messages |
| `ingest-calendar` | Google Calendar / Outlook MCP server | Ingest meeting events and notes |
| `ingest-notion` | Notion MCP server | Ingest Notion pages |

## How to Use

1. Ensure both the source MCP server (e.g. Slack) and Graphite MCP server are connected
2. Ask Claude to ingest from the source, e.g.:
   - "Ingest the last week of messages from #engineering into Graphite"
   - "Pull my recent emails about Project Alpha into memory"
   - "Ingest my calendar events from this month"
3. Claude will read the appropriate workflow template and follow the steps

## How It Works

These are **instruction files**, not executable code. Claude reads them to
understand the correct sequence of MCP tool calls for each data source.
The `graphite_ingest_text()` tool handles the actual ingestion — it accepts
raw text, a stable `source_id` for deduplication, and a memory category.

Re-running any workflow is safe. Content hashing makes `graphite_ingest_text()`
idempotent — unchanged content is skipped automatically.

## Adding New Workflows

Create a new `.md` file in this directory following the existing pattern:
1. Prerequisites (which MCP servers are needed)
2. Step-by-step instructions for Claude
3. Notes on source_id format, category choice, and edge cases
