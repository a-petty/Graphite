# Payment Gateway Security Review — 2024-11-09

## Attendees

Kevin Park, John Doe, Tom Brennan

## Discussion

**Kevin Park:** Before we commit, we should consider the risks. We should discuss our OAuth configuration. Token refresh flow is working correctly with the 15-minute expiry.

**John Doe:** Let me walk through the data on this. I ran benchmarks on Vault last week. The dynamic secret rotation is running every 24 hours without issues.

**Tom Brennan:** I think we need to be practical here. Our PostgreSQL setup needs attention. The query planner is choosing index scans correctly after the ANALYZE.

**Tom Brennan:** We should keep in mind the payment service outage incident from July 2024.

## Decisions

**John Doe:** Based on this discussion, we've decided: Scheduled CockroachDB migration for Q1 2025. Need horizontally scalable SQL database for payment service growth projections exceeding single-node PostgreSQL limits

## Action Items

- Kevin Park will investigate OAuth configuration and report findings by end of week
- John Doe will investigate Vault configuration and report findings by end of week
- Tom Brennan will investigate PostgreSQL configuration and report findings by end of week
