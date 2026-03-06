# ADR-008: Feature store architecture for ML serving

**Status:** Proposed

**Date:** May 2025

**Author:** Emily Zhang

## Context

ML Recommendation Engine serving requires sub-50ms feature retrieval. Current approach of computing features at request time adds 200ms latency.

## Decision

Build two-tier feature store: Redis for online serving (<10ms) and Snowflake for offline training batch features

**Rationale:** Separation of online/offline feature paths optimizes for both latency and compute cost

## Alternatives Considered

- Feast — open-source but adds operational complexity we may not need yet
- Single Redis tier — insufficient for large historical feature sets
- Pre-computed feature tables in PostgreSQL — too slow for real-time serving

## Consequences

- Dual-write complexity for feature consistency
- Redis capacity planning for feature data
- Enables real-time personalization across all products
