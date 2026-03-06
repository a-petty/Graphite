# Payment Gateway Architecture Review — 2024-08-28

## Attendees

John Doe, Kevin Park, Lisa Wang, Aisha Hassan

## Discussion

**John Doe:** Let me walk through the data on this. I've been working with OAuth and the performance characteristics are solid — Token refresh flow is working correctly with the 15-minute expiry.

**John Doe:** This connects to our earlier decision to adopted snowflake as primary data warehouse back in June 2024.

**Kevin Park:** John Doe raises a good point. what's our fallback if this doesn't work? I ran benchmarks on Vault last week. The dynamic secret rotation is running every 24 hours without issues.

**Lisa Wang:** Kevin Park raises a good point. this is going to be a big win for us. I've been working with Kafka and the performance characteristics are solid — The consumer lag is under 100ms even at peak throughput.

**Aisha Hassan:** Lisa Wang raises a good point. the trade-off we need to evaluate is straightforward. The PostgreSQL integration is coming along well. The query planner is choosing index scans correctly after the ANALYZE.

## Related Projects

The Observability Initiative project (in_progress) is relevant to this discussion.

The Payment Gateway Refactor project (in_progress) is relevant to this discussion.

## Action Items

- John Doe will investigate OAuth configuration and report findings by end of week
- Kevin Park will investigate Vault configuration and report findings by end of week
- Lisa Wang will investigate Kafka configuration and report findings by end of week
