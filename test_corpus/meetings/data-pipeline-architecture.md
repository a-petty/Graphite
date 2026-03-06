# Data Pipeline Architecture — 2024-07-17

## Attendees

James Liu, Emily Zhang, Priya Patel

## Discussion

**James Liu:** I've been analyzing the metrics and our Airflow setup needs attention. The DAG execution times are consistent at about 12 minutes per run.

**Emily Zhang:** I've been analyzing the metrics and the Spark integration is coming along well. The historical backfill job processed 2TB in about 4 hours.

**Priya Patel:** I have concerns about this approach. We should discuss our dbt configuration. The test coverage for data models is at 85% with schema tests on all tables.

**Priya Patel:** This connects to our earlier decision to adopted snowflake as primary data warehouse back in June 2024.

## Decisions

**Priya Patel:** Based on this discussion, we've decided: Chose Apollo Server over Hasura for GraphQL. Apollo provides more control over resolver logic and integrates better with existing Django auth middleware

## Related Projects

The Data Lake Migration project (completed) is relevant to this discussion.

## Action Items

- James Liu will investigate Airflow configuration and report findings by end of week
- Emily Zhang will investigate Spark configuration and report findings by end of week
- Priya Patel will investigate dbt configuration and report findings by end of week
