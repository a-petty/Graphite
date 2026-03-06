# ADR-004: Use DataLoader pattern for GraphQL N+1 prevention

**Status:** Accepted

**Date:** December 2024

**Author:** Priya Patel

## Context

The January GraphQL gateway timeout was caused by an N+1 query pattern in a new resolver. We need a systematic approach to prevent this class of bug.

This ADR was prompted by the GraphQL gateway timeout — 30 min degraded API responses in January 2025.

## Decision

Mandate DataLoader for all database-accessing GraphQL resolvers, with automated N+1 detection in CI

**Rationale:** DataLoader batches and caches database calls within a single request, eliminating N+1 queries by design

## Alternatives Considered

- Manual query optimization per resolver — doesn't scale
- Hasura-style auto-generated resolvers — too restrictive for our use case
- Query complexity limits only — treats symptoms not cause

## Consequences

- Additional boilerplate per resolver
- Need to train team on DataLoader patterns
- Prevents entire class of performance regressions
