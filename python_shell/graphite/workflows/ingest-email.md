# Workflow: Ingest Email into Graphite

## Prerequisites

- Gmail or Outlook MCP server connected (provides email reading tools)
- Graphite MCP server connected

## Steps

1. **Search or list emails** — Use the email MCP tool to find relevant messages:
   - Search by sender, subject, date range, or label/folder
   - Let the user specify the scope (e.g. "emails about Project Alpha from last month")

2. **Fetch full messages** — For each email, retrieve:
   - Subject line
   - From, To, CC recipients
   - Date sent
   - Body text (prefer plain text; strip HTML if needed)
   - Thread/conversation ID if available

3. **Group email threads** — If the source provides conversation threading:
   - Group replies together into a single document per thread
   - Order messages chronologically within the thread
   - If no threading, treat each email as a standalone document

4. **Format each email/thread as a document** — Create a text block:
   - Header: Subject, date, participants
   - Each message in the thread with sender and timestamp
   - Example:
     ```
     Subject: Q1 Budget Review — Final Numbers
     Date: 2025-01-20
     From: Sarah Kim
     To: Finance Team
     CC: James Wright

     Sarah Kim [2025-01-20 10:30]:
     Hi team, attached are the final Q1 numbers. Key takeaways:
     - Revenue exceeded forecast by 12%
     - Engineering headcount came in under budget
     - Marketing spend was 8% over due to the product launch

     James Wright [2025-01-20 11:15]:
     Thanks Sarah. The engineering savings offset the marketing overage nicely.
     Can you break down the revenue by product line for the board deck?
     ```

5. **Ingest each thread/message** — Call `graphite_ingest_text()`:
   - `text`: the formatted email content
   - `source_id`: `"gmail://{message_id}"` or `"outlook://{message_id}"`
     (use conversation ID if grouping threads: `"gmail://thread-{thread_id}"`)
   - `category`: Usually `"Episodic"`. Use `"Semantic"` for reference emails
     (org charts, policy docs), `"Procedural"` for how-to or process emails.

6. **Report results** — Summarize what was ingested.

## Notes

- **Idempotent**: Re-running is safe. Same source_id + same content = skipped.
- **source_id format**: Use the email provider's stable message/thread ID.
  Gmail: `"gmail://{message_id}"`. Outlook: `"outlook://{message_id}"`.
- **Attachments**: Text-based attachments can be appended to the email body.
  Binary attachments (PDFs, images) should be noted but not included.
- **Large threads**: Email threads with 20+ messages may be very long. Consider
  splitting at natural breakpoints (date boundaries, topic shifts).
- **Sensitivity**: Skip emails that appear to contain passwords, tokens, or
  other credentials. Flag to the user if unsure.
