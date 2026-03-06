# ADR-006: Adopt event sourcing for payment domain

**Status:** Accepted

**Date:** March 2025

**Author:** John Doe

## Context

Payment compliance requirements demand full audit trails. The existing Kafka-based event pipeline (ADR-001) provides the foundation, but we need true event sourcing with aggregate reconstruction.

This ADR formalizes the decision to chose event sourcing over cqrs-only for core domain services made in March 2025.

## Decision

Implement event sourcing with Kafka as the event log and PostgreSQL JSONB for event snapshots in the payment domain

**Rationale:** Full event log satisfies PCI audit requirements and enables temporal debugging of payment state transitions

## Alternatives Considered

- Change data capture from PostgreSQL WAL — loses domain event semantics
- CQRS without event sourcing — insufficient for compliance audit trail

## Consequences

- Significant refactoring of payment service write path
- Enables replay and temporal queries
- Team needs event sourcing training
