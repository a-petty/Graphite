# Feature Store Design — 2024-10-03

## Attendees

Emily Zhang, James Liu, John Doe

## Discussion

**Emily Zhang:** Let me walk through the data on this. I've been working with Airflow and the performance characteristics are solid — The DAG execution times are consistent at about 12 minutes per run.

**James Liu:** I've been analyzing the metrics and our PyTorch setup needs attention. The model training is converging in about 45 epochs on the current dataset.

**John Doe:** James Liu raises a good point. i've been analyzing the metrics and we should discuss our dbt configuration. The test coverage for data models is at 85% with schema tests on all tables.

**John Doe:** We can build on the lessons from the Auth Service Modernization project which shipped in September 2024.

## Related Projects

The Data Lake Migration project (completed) is relevant to this discussion.

The ML Recommendation Engine project (in_progress) is relevant to this discussion.

## Action Items

- Emily Zhang will investigate Airflow configuration and report findings by end of week
- James Liu will investigate PyTorch configuration and report findings by end of week
- John Doe will investigate dbt configuration and report findings by end of week
