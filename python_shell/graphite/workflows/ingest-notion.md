# Workflow: Ingest Notion Pages into Graphite

## Prerequisites

- Notion MCP server connected (provides page/database reading tools)
- Graphite MCP server connected

## Steps

1. **Find pages** — Use the Notion MCP tool to search or list pages:
   - Search by title, database, or workspace section
   - The user may specify a database, page, or search query
   - For databases, list entries and let the user confirm scope

2. **Fetch page content** — For each page, retrieve:
   - Title
   - Author / last edited by
   - Created date and last modified date
   - Full page content (text blocks, headings, lists, etc.)
   - Properties (if from a database: status, tags, assignees, etc.)

3. **Format each page as a document** — Create a text block:
   - Header: title, author, dates, properties
   - Body: full page content with structure preserved
   - Example:
     ```
     Title: API Migration Plan — v2 to v3
     Author: Bob Park
     Created: 2025-01-10
     Last Modified: 2025-01-18
     Status: In Progress
     Tags: engineering, api, migration

     ## Overview
     This document outlines the migration plan from API v2 to v3.
     Target completion: end of Q1 2025.

     ## Breaking Changes
     - Authentication: OAuth2 replaces API keys
     - Pagination: cursor-based replaces offset-based
     - Rate limits: per-endpoint instead of global

     ## Migration Steps
     1. Update auth flow in client SDK
     2. Migrate pagination in list endpoints
     3. Update rate limit handling
     4. Run integration tests against v3 staging
     ```

4. **Choose the right category** — Notion content varies widely:
   - Meeting notes, journals, logs → `"Episodic"`
   - People profiles, team directories, reference docs → `"Semantic"`
   - Project plans, process docs, runbooks, how-tos → `"Procedural"`
   - When in doubt, ask the user or default to `"Episodic"`

5. **Ingest each page** — Call `graphite_ingest_text()`:
   - `text`: the formatted page content
   - `source_id`: `"notion://{page_id}"`
   - `category`: chosen based on content type (see above)

6. **Report results** — Summarize what was ingested.

## Notes

- **Idempotent**: Re-running is safe. Same page ID + same content = skipped.
  If a page was edited since last ingestion, it will be re-processed automatically.
- **source_id format**: Use the Notion page ID: `"notion://{page_id}"`.
  Page IDs are stable UUIDs that don't change when pages are renamed or moved.
- **Databases**: For Notion databases, each row/entry is a separate page.
  Ingest each entry as its own document with its own source_id.
- **Nested pages**: Notion pages can contain sub-pages. Decide with the user
  whether to recurse into sub-pages or only ingest the top-level page.
- **Rich content**: Notion supports embeds, images, and code blocks. Include
  code blocks as-is. Note image/embed presence but don't include binary content.
- **Large pages**: Very long Notion pages (> 10,000 words) may benefit from
  being split. The structural parser will chunk them, but extremely large
  documents may reduce extraction quality.
