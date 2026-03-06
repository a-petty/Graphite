# ADR-007: Service mesh mTLS rollout strategy

**Status:** Accepted

**Date:** April 2025

**Author:** Ryan O'Brien

## Context

Istio is deployed but mTLS is in permissive mode. Need a strategy to enforce strict mTLS across all namespaces without breaking existing services.

## Decision

Phased namespace-by-namespace mTLS enforcement starting with non-critical services, with automated compliance checking

**Rationale:** Gradual rollout reduces blast radius of misconfigurations while still achieving full encryption within Q2

## Alternatives Considered

- Big-bang enforcement — too risky for production traffic
- Application-level TLS — inconsistent and harder to audit

## Consequences

- 3-month rollout timeline
- Need per-namespace PeerAuthentication policies
- Temporary mixed-mode traffic during transition
