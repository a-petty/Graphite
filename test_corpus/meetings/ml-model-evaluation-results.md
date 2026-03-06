# Ml Model Evaluation Results — 2024-11-15

## Attendees

Emily Zhang, James Liu, Angela Rivera

## Discussion

**Emily Zhang:** I want to make sure we're systematic about this. The Airflow integration is coming along well. The DAG execution times are consistent at about 12 minutes per run.

**James Liu:** Building on what Emily Zhang said, i want to make sure we're systematic about this. The PyTorch integration is coming along well. The model training is converging in about 45 epochs on the current dataset.

**Angela Rivera:** I think we need to be practical here. I ran benchmarks on dbt last week. The test coverage for data models is at 85% with schema tests on all tables.

**Angela Rivera:** We can build on the lessons from the Auth Service Modernization project which shipped in September 2024.

## Related Projects

The Developer Portal project (completed) is relevant to this discussion.

## Action Items

- Emily Zhang will investigate Airflow configuration and report findings by end of week
- James Liu will investigate PyTorch configuration and report findings by end of week
- Angela Rivera will investigate dbt configuration and report findings by end of week
