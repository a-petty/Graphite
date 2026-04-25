# Workflow: Ingest Slack Channel into Graphite

## Prerequisites

- Slack MCP server connected (provides channel/message reading tools)
- Graphite MCP server connected

## Steps

1. **List channels** — Use the Slack MCP tool to list available channels. Let the user
   pick which channel(s) to ingest, or use the one they specified.

2. **Fetch messages** — Read messages from the target channel:
   - Include thread replies for full context
   - Filter by date range if the user specified one
   - Page through results if the channel has many messages

3. **Group into threads** — Organize messages into logical conversation threads:
   - Messages with thread replies form one group
   - Standalone messages close in time (< 5 min gap) about the same topic can be combined
   - Very short messages (< 50 chars) with no meaningful content can be skipped

4. **Format each thread as a document** — For each thread, create a text block:
   - First line: channel name, date, and topic summary
   - Each message: `[timestamp] Speaker Name: message text`
   - Include reactions or attachments if available
   - Example:
     ```
     #engineering — 2025-01-15 — API migration discussion

     [09:15] Alice Chen: Has anyone started looking at the v3 API migration?
     [09:17] Bob Park: I did a spike last week. The auth flow changed significantly.
     [09:18] Alice Chen: Can you share your notes? We should plan this for next sprint.
     [09:20] Bob Park: Sure, I'll post them in the thread.
     ```

5. **Ingest each thread** — Call `graphite_ingest_text()` for each formatted thread:
   - `text`: the formatted thread content
   - `source_id`: `"slack://{channel_id}-{thread_ts}"` (or `"slack://{channel_id}-{message_ts}"` for standalone messages)
   - `category`: `"Episodic"`

6. **Report results** — Summarize what was ingested:
   - Number of threads processed
   - Total entities found
   - Any errors or skipped messages

## Notes

- **Idempotent**: Re-running is safe. Content hashing skips unchanged threads.
- **source_id format**: Use `slack://{channel_id}-{thread_ts}` for stable deduplication.
  The channel ID + thread timestamp uniquely identifies a conversation.
- **Category**: Slack conversations are almost always `"Episodic"` (events in time).
  Exception: a channel that's a knowledge base or wiki — use `"Semantic"`.
- **Large channels**: For channels with thousands of messages, batch by week or month
  to avoid overwhelming the LLM extraction pipeline.
- **DMs**: Same pattern works for direct messages. Use `"slack://dm-{dm_id}-{thread_ts}"`.
