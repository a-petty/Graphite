# Security Audit Findings — 2024-07-01

## Attendees

Kevin Park, Aisha Hassan, Ryan O'Brien

## Discussion

**Kevin Park:** I'm not fully convinced yet. I've been working with Vault and the performance characteristics are solid — The dynamic secret rotation is running every 24 hours without issues.

**Aisha Hassan:** To add to Kevin Park's point, what's the simplest path to getting this done? The Go integration is coming along well. The auth service is handling 10K req/s with 8MB memory footprint.

**Ryan O'Brien:** I have concerns about this approach. I ran benchmarks on gRPC last week. We're making good progress on the implementation.

**Ryan O'Brien:** This connects to our earlier decision to adopted snowflake as primary data warehouse back in June 2024.

## Related Projects

The Auth Service Modernization project (completed) is relevant to this discussion.

## Action Items

- Kevin Park will investigate Go configuration and report findings by end of week
- Aisha Hassan will investigate gRPC configuration and report findings by end of week
- Ryan O'Brien will investigate Vault configuration and report findings by end of week
