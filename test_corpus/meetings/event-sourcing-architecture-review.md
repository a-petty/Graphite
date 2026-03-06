# Event Sourcing Architecture Review — 2025-03-12

## Attendees

John Doe, Ryan O'Brien, Daniel Reeves, Aisha Hassan

## Discussion

**John Doe:** Let me walk through the data on this. I've been working with Python and the performance characteristics are solid — We're making good progress on the implementation.

**Ryan O'Brien:** Building on what John Doe said, what's our fallback if this doesn't work? We should discuss our Kafka configuration. The consumer lag is under 100ms even at peak throughput.

**Daniel Reeves:** Ryan O'Brien raises a good point. quick update: I ran benchmarks on PostgreSQL last week. The query planner is choosing index scans correctly after the ANALYZE.

**Daniel Reeves:** We should keep in mind the payment service outage incident from July 2024.

**Aisha Hassan:** I think we need to be practical here. I've been working with Kubernetes and the performance characteristics are solid — The HPA scaling is keeping pods between 2 and 8 replicas as expected.

## Decisions

**John Doe:** Based on this discussion, we've decided: Chose event sourcing over CQRS-only for core domain services. Full event log provides auditability for payment compliance and enables temporal queries for debugging

## Related Projects

The Observability Initiative project (in_progress) is relevant to this discussion.

The Payment Gateway Refactor project (in_progress) is relevant to this discussion.

## Action Items

- John Doe will investigate Python configuration and report findings by end of week
- Ryan O'Brien will investigate Kafka configuration and report findings by end of week
- Daniel Reeves will investigate PostgreSQL configuration and report findings by end of week
