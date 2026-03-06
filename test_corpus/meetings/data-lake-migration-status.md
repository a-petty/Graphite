# Data Lake Migration Status — 2024-11-24

## Attendees

James Liu, David Kim, Aisha Hassan

## Discussion

**James Liu:** I've been analyzing the metrics and our Airflow setup needs attention. The DAG execution times are consistent at about 12 minutes per run.

**James Liu:** This connects to our earlier decision to selected react over vue for dashboard redesign back in August 2024.

**David Kim:** James Liu raises a good point. what's the simplest path to getting this done? Our Snowflake setup needs attention. The warehouse auto-scaling is keeping costs predictable.

**Aisha Hassan:** To add to David Kim's point, the trade-off we need to evaluate is straightforward. I ran benchmarks on dbt last week. The test coverage for data models is at 85% with schema tests on all tables.

## Decisions

**John Doe:** Based on this discussion, we've decided: Scheduled CockroachDB migration for Q1 2025. Need horizontally scalable SQL database for payment service growth projections exceeding single-node PostgreSQL limits

**Derek Washington:** Based on this discussion, we've decided: Adopted PagerDuty for on-call rotation. Integrates with Datadog alerts and provides escalation policies needed for 24/7 coverage

## Related Projects

The Data Lake Migration project (completed) is relevant to this discussion.

The ML Recommendation Engine project (in_progress) is relevant to this discussion.

The Service Mesh Rollout project (in_progress) is relevant to this discussion.

## Action Items

- James Liu will investigate Airflow configuration and report findings by end of week
- David Kim will investigate dbt configuration and report findings by end of week
- Aisha Hassan will investigate Snowflake configuration and report findings by end of week
