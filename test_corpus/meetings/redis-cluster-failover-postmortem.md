# Incident Postmortem — Redis cluster failover — 15 min elevated latency — 2024-10-21

## Attendees

Derek Washington, Priya Patel, David Kim

## Incident Summary

Redis cluster failover — 15 min elevated latency. This affected the ML Recommendation Engine service. The incident was detected by automated monitoring and the team responded within 5 minutes.

## Root Cause Analysis

**Derek Washington:** The root cause was identified: Primary Redis node ran out of memory due to unbounded cache growth in the recommendation feature cache, triggering automatic failover to replica

The affected components included Redis, Datadog, Kubernetes.

## Resolution

**Priya Patel:** Set maxmemory-policy to allkeys-lru, added Datadog memory alerts at 80% threshold, documented cache eviction strategy

## Lessons from Prior Incidents

**David Kim:** This is similar to the payment service outage incident from July 2024. We should review whether the mitigations from that incident apply here.

## Action Items

- Derek Washington will implement safeguards for Redis to prevent recurrence
- Priya Patel will implement safeguards for Datadog to prevent recurrence
- David Kim will implement safeguards for Kubernetes to prevent recurrence
