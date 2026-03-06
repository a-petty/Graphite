# Service Mesh Rollout — Kickoff Meeting — 2024-11-04

## Attendees

Ryan O'Brien, David Kim, Derek Washington

## Project Overview

**Ryan O'Brien:** Welcome everyone. Today we're kicking off the Service Mesh Rollout project. Our goal is to Implement service mesh for traffic management, security, and observability across microservices. Let me walk through the approach and timeline.

## Technical Approach

**Ryan O'Brien:** We'll be using Istio for this project. The rationale is mature service mesh with mTLS and traffic policies.

**Ryan O'Brien:** We'll be using Kubernetes for this project. The rationale is existing orchestration platform for sidecar injection.

**Derek Washington:** We'll be using Envoy for this project. The rationale is high-performance sidecar proxy with rich L7 features.

**Derek Washington:** We'll be using Prometheus for this project. The rationale is native metrics export from Envoy sidecars.

**Ryan O'Brien:** We should keep in mind the redis cluster failover incident from October 2024.

## Timeline

- Phase 1: Architecture and design (November 2024)
- Phase 2: Core implementation (following month)
- Phase 3: Testing and rollout (month after)

## Action Items

- Ryan O'Brien will draft the Kubernetes section of the technical design doc by Friday
- David Kim will draft the Kubernetes section of the technical design doc by Friday
- Derek Washington will draft the Istio section of the technical design doc by Friday
