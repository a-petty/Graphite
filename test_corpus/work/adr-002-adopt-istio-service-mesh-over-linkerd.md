# ADR-002: Adopt Istio service mesh over Linkerd

**Status:** Accepted

**Date:** November 2024

**Author:** Ryan O'Brien

## Context

With 40+ microservices, we need centralized traffic management, mTLS, and observability. The October service mesh planning meeting identified Istio and Linkerd as finalists.

This ADR formalizes the decision to approved istio for service mesh made in October 2024.

## Decision

Deploy Istio as our service mesh with Envoy sidecars across all Kubernetes namespaces

**Rationale:** Istio offers richer traffic policies, built-in fault injection for chaos testing, and better Prometheus integration

## Alternatives Considered

- Linkerd — lighter weight but fewer traffic management features
- Consul Connect — requires HashiCorp ecosystem buy-in
- No mesh — continue with manual mTLS and service discovery

## Consequences

- Higher memory overhead per pod (~50MB sidecar)
- Requires Istio expertise on infrastructure team
- Enables canary deployments and circuit breaking
