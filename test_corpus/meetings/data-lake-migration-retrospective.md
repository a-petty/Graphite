# Data Lake Migration Retrospective — 2025-02-12

## Attendees

James Liu, Emily Zhang, David Kim, Aisha Hassan

## Discussion

**James Liu:** I've been analyzing the metrics and we should discuss our Airflow configuration. The DAG execution times are consistent at about 12 minutes per run.

**James Liu:** This connects to our earlier decision to selected react over vue for dashboard redesign back in August 2024.

**Emily Zhang:** Based on my review of the documentation, i've been working with ArgoCD and the performance characteristics are solid — The GitOps sync is completing within 90 seconds of merge.

**David Kim:** What's the simplest path to getting this done? The Python integration is coming along well. We're making good progress on the implementation.

**Aisha Hassan:** Let's focus on what we can ship this quarter. We should discuss our GitHub Actions configuration. The CI pipeline runs in about 8 minutes end to end.

## Decisions

**Emily Zhang:** Based on this discussion, we've decided: Selected MLflow over Weights & Biases for experiment tracking. Open-source, self-hosted option avoids vendor lock-in and integrates with existing Airflow pipelines

## Related Projects

The Data Lake Migration project (completed) is relevant to this discussion.

## Action Items

- James Liu will investigate Airflow configuration and report findings by end of week
- Emily Zhang will investigate ArgoCD configuration and report findings by end of week
- David Kim will investigate Python configuration and report findings by end of week
