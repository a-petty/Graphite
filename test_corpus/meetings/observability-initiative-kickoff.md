# Observability Initiative — Kickoff Meeting — 2024-07-24

## Attendees

Derek Washington, John Doe, David Kim

## Project Overview

**Derek Washington:** Welcome everyone. Today we're kicking off the Observability Initiative project. Our goal is to Implement distributed tracing and unified monitoring across all microservices. Let me walk through the approach and timeline.

## Technical Approach

**Derek Washington:** We'll be using Jaeger for this project. The rationale is distributed tracing with service dependency visualization.

**John Doe:** We'll be using OpenTelemetry for this project. The rationale is vendor-neutral instrumentation standard.

**Derek Washington:** We'll be using Datadog for this project. The rationale is unified dashboards and alerting across infrastructure.

**Derek Washington:** We'll be using Prometheus for this project. The rationale is metric collection for Kubernetes-native workloads.

**Derek Washington:** We'll be using Grafana for this project. The rationale is visualization layer for Prometheus metrics.

**Derek Washington:** This connects to our earlier decision to adopted snowflake as primary data warehouse back in June 2024.

## Timeline

- Phase 1: Architecture and design (July 2024)
- Phase 2: Core implementation (following month)
- Phase 3: Testing and rollout (month after)

## Action Items

- Derek Washington will draft the Datadog section of the technical design doc by Friday
- John Doe will draft the OpenTelemetry section of the technical design doc by Friday
- David Kim will draft the OpenTelemetry section of the technical design doc by Friday
