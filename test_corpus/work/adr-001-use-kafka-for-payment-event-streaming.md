# ADR-001: Use Kafka for payment event streaming

**Status:** Accepted

**Date:** October 2024

**Author:** John Doe

## Context

The July payment outage revealed that synchronous payment processing creates cascading failures. We need an async event-driven architecture for payment state transitions.

This ADR was prompted by the Payment service outage — 45 min downtime in July 2024.

## Decision

Adopt Kafka as the event backbone for all payment-related state changes, with PostgreSQL as the transactional store

**Rationale:** Kafka provides guaranteed delivery, replay capability, and natural backpressure — critical for payment reliability

## Alternatives Considered

- RabbitMQ — simpler but lacks replay and partitioned ordering
- AWS SQS — managed but no log semantics for audit trail
- Redis Streams — insufficient durability guarantees for payment data

## Consequences

- Team needs Kafka operational expertise
- Added infrastructure complexity
- Enables future event sourcing migration
