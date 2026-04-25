# Workflow: Ingest Calendar Events into Graphite

## Prerequisites

- Google Calendar or Outlook Calendar MCP server connected
- Graphite MCP server connected

## Steps

1. **List events** — Use the calendar MCP tool to fetch events:
   - Filter by date range (user-specified or default to last 30 days)
   - Include recurring event instances
   - Skip all-day events that are just placeholders (e.g. "OOO", "Focus Time")
     unless the user specifically requests them

2. **Fetch event details** — For each event, retrieve:
   - Title/subject
   - Start and end time
   - Attendees (names and roles if available)
   - Location (physical or video link)
   - Description/notes
   - Any attached meeting notes or agenda

3. **Format each event as a document** — Create a text block:
   - Header: title, date/time, attendees
   - Body: description, notes, agenda items
   - Example:
     ```
     Meeting: Weekly Product Sync
     Date: 2025-01-22 14:00-15:00
     Attendees: Alice Chen, Bob Park, Carol Davis, David Lee
     Location: Zoom

     Agenda:
     - Review sprint progress
     - Discuss customer feedback from beta launch
     - Plan Q2 roadmap priorities

     Notes:
     Alice presented the sprint metrics. 8 of 10 stories completed.
     Bob flagged a blocker on the payment integration — needs API access from vendor.
     Carol shared positive feedback from 3 beta customers. NPS score: 72.
     Decision: Push Q2 roadmap kickoff to Feb 1 to close out remaining beta items.
     ```

4. **Ingest each event** — Call `graphite_ingest_text()`:
   - `text`: the formatted event content
   - `source_id`: `"gcal://{event_id}"` or `"outlook-cal://{event_id}"`
   - `category`: `"Episodic"` (meetings are events in time)

5. **Report results** — Summarize what was ingested.

## Notes

- **Idempotent**: Re-running is safe. Same event ID + same content = skipped.
- **source_id format**: Use the calendar provider's stable event ID.
  Google Calendar: `"gcal://{event_id}"`. Outlook: `"outlook-cal://{event_id}"`.
- **Recurring events**: Each instance gets its own source_id (most calendar APIs
  provide unique IDs per instance). Don't deduplicate recurring events — each
  occurrence may have different notes.
- **Events without notes**: Events with only a title and attendee list still have
  value — they record who met and when. Ingest them with the available metadata.
- **Category**: Almost always `"Episodic"`. Exception: recurring reference events
  like "Office Hours" or "Team Standup Template" might be `"Procedural"`.
