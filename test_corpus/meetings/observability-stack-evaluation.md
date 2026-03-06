# Observability Stack Evaluation — 2024-09-14

## Attendees

Derek Washington, David Kim, Ryan O'Brien

## Discussion

**Derek Washington:** What's the simplest path to getting this done? We should discuss our Prometheus configuration. The scrape interval is at 15s which gives us good resolution without overhead.

**David Kim:** Derek Washington raises a good point. let's focus on what we can ship this quarter. I ran benchmarks on PagerDuty last week. The escalation policies are working and mean time to acknowledge is under 3 minutes.

**Ryan O'Brien:** Building on what David Kim said, before we commit, we should consider the risks. I've been working with Grafana and the performance characteristics are solid — The new dashboards are being used daily by the on-call team.

**Ryan O'Brien:** We should keep in mind the payment service outage incident from July 2024.

## Decisions

**Sarah Chen:** Based on this discussion, we've decided: Adopted trunk-based development. Reduce merge conflicts and enable continuous deployment by keeping branches short-lived

## Related Projects

The Observability Initiative project (in_progress) is relevant to this discussion.

## Action Items

- Derek Washington will investigate Grafana configuration and report findings by end of week
- David Kim will investigate Prometheus configuration and report findings by end of week
- Ryan O'Brien will investigate PagerDuty configuration and report findings by end of week
