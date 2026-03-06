# Q4 Planning — 2024-11-02

## Attendees

Aisha Hassan, Sarah Chen, Angela Rivera, Derek Washington

## Discussion

**Aisha Hassan:** The trade-off we need to evaluate is straightforward. I ran benchmarks on Grafana last week. The new dashboards are being used daily by the on-call team.

**Sarah Chen:** I've been analyzing the metrics and the Prometheus integration is coming along well. The scrape interval is at 15s which gives us good resolution without overhead.

**Sarah Chen:** We should keep in mind the redis cluster failover incident from October 2024.

**Angela Rivera:** Building on what Sarah Chen said, the trade-off we need to evaluate is straightforward. The Datadog integration is coming along well. The custom dashboards are giving us real-time visibility into the pipeline.

**Derek Washington:** Angela Rivera raises a good point. i think we need to be practical here. Our Grafana setup needs attention. The new dashboards are being used daily by the on-call team.

## Decisions

**John Doe:** Based on this discussion, we've decided: Scheduled CockroachDB migration for Q1 2025. Need horizontally scalable SQL database for payment service growth projections exceeding single-node PostgreSQL limits

**Derek Washington:** Based on this discussion, we've decided: Adopted PagerDuty for on-call rotation. Integrates with Datadog alerts and provides escalation policies needed for 24/7 coverage

## Related Projects

The Dashboard Redesign project (completed) is relevant to this discussion.

## Action Items

- Aisha Hassan will investigate Grafana configuration and report findings by end of week
- Sarah Chen will investigate Prometheus configuration and report findings by end of week
- Angela Rivera will investigate Datadog configuration and report findings by end of week
