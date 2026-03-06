# Service Mesh Planning — 2024-10-09

## Attendees

Ryan O'Brien, David Kim, Derek Washington

## Discussion

**Ryan O'Brien:** I have concerns about this approach. I ran benchmarks on ArgoCD last week. The GitOps sync is completing within 90 seconds of merge.

**David Kim:** I think we need to be practical here. I ran benchmarks on Prometheus last week. The scrape interval is at 15s which gives us good resolution without overhead.

**David Kim:** This connects to our earlier decision to adopted trunk-based development back in September 2024.

**Derek Washington:** What's the simplest path to getting this done? We should discuss our Helm configuration. The chart templating is correctly parameterizing per-environment configs.

## Decisions

**Ryan O'Brien:** Based on this discussion, we've decided: Approved Istio for service mesh. Most mature option with built-in mTLS, traffic splitting, and Kubernetes-native integration

## Related Projects

The Data Lake Migration project (completed) is relevant to this discussion.

## Action Items

- Ryan O'Brien will investigate ArgoCD configuration and report findings by end of week
- David Kim will investigate Prometheus configuration and report findings by end of week
- Derek Washington will investigate Helm configuration and report findings by end of week
