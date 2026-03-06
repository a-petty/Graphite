# Graphql Schema Design Review — 2024-07-08

## Attendees

Priya Patel, Marcus Johnson, Sophie Martin

## Discussion

**Priya Patel:** What's our fallback if this doesn't work? We should discuss our Django configuration. The ORM query optimization brought the endpoint from 800ms to 120ms.

**Priya Patel:** This connects to our earlier decision to adopted snowflake as primary data warehouse back in June 2024.

**Marcus Johnson:** This is going to be a big win for us. I've been working with Swagger and the performance characteristics are solid — The auto-generated docs are staying in sync with the API changes.

**Sophie Martin:** Marcus Johnson raises a good point. let me walk through the data on this. I ran benchmarks on GraphQL last week. Query complexity scoring is preventing the N+1 issues we saw before.

## Decisions

**Priya Patel:** Based on this discussion, we've decided: Chose Apollo Server over Hasura for GraphQL. Apollo provides more control over resolver logic and integrates better with existing Django auth middleware

## Action Items

- Priya Patel will investigate Django configuration and report findings by end of week
- Marcus Johnson will investigate GraphQL configuration and report findings by end of week
- Sophie Martin will investigate Swagger configuration and report findings by end of week
