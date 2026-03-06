# Incident Postmortem — GraphQL gateway timeout — 30 min degraded API responses — 2025-01-07

## Attendees

Priya Patel, Alex Petrov, Derek Washington

## Incident Summary

GraphQL gateway timeout — 30 min degraded API responses. This affected the API Migration service. The incident was detected by automated monitoring and the team responded within 5 minutes.

## Root Cause Analysis

**Priya Patel:** The root cause was identified: N+1 query in new Apollo Server resolver caused database connection pool exhaustion under load

The affected components included GraphQL, Apollo Server, PostgreSQL.

## Resolution

**Alex Petrov:** Added DataLoader batching for the affected resolver, implemented connection pool monitoring

## Lessons from Prior Incidents

**Derek Washington:** This is similar to the redis cluster failover incident from October 2024. We should review whether the mitigations from that incident apply here.

## Action Items

- Priya Patel will implement safeguards for GraphQL to prevent recurrence
- Alex Petrov will implement safeguards for Apollo Server to prevent recurrence
- Derek Washington will implement safeguards for PostgreSQL to prevent recurrence
