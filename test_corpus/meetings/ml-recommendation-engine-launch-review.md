# Ml Recommendation Engine Launch Review — 2025-05-19

## Attendees

Emily Zhang, James Liu, Angela Rivera, Aisha Hassan

## Discussion

**Emily Zhang:** I've been analyzing the metrics and we should discuss our Airflow configuration. The DAG execution times are consistent at about 12 minutes per run.

**Emily Zhang:** We should keep in mind the graphql gateway timeout incident from January 2025.

**James Liu:** I've been analyzing the metrics and i've been working with PyTorch and the performance characteristics are solid — The model training is converging in about 45 epochs on the current dataset.

**Angela Rivera:** I think we need to be practical here. I ran benchmarks on Python last week. We're making good progress on the implementation.

**Aisha Hassan:** To add to Angela Rivera's point, what's the simplest path to getting this done? Our dbt setup needs attention. The test coverage for data models is at 85% with schema tests on all tables.

## Related Projects

The ML Recommendation Engine project (in_progress) is relevant to this discussion.

## Action Items

- Emily Zhang will investigate Airflow configuration and report findings by end of week
- James Liu will investigate PyTorch configuration and report findings by end of week
- Angela Rivera will investigate dbt configuration and report findings by end of week
