# Ml Model Training Pipeline Design — 2024-08-24

## Attendees

Emily Zhang, James Liu

## Discussion

**Emily Zhang:** Let me walk through the data on this. I've been working with Airflow and the performance characteristics are solid — The DAG execution times are consistent at about 12 minutes per run.

**James Liu:** I want to make sure we're systematic about this. We should discuss our PyTorch configuration. The model training is converging in about 45 epochs on the current dataset.

**James Liu:** This connects to our earlier decision to adopted snowflake as primary data warehouse back in June 2024.

**Emily Zhang:** Let me walk through the data on this. Our Python setup needs attention. We're making good progress on the implementation.

## Related Projects

The Data Lake Migration project (completed) is relevant to this discussion.

## Action Items

- Emily Zhang will investigate Airflow configuration and report findings by end of week
- James Liu will investigate PyTorch configuration and report findings by end of week
