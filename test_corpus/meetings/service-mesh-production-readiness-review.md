# Service Mesh Production Readiness Review — 2025-04-05

## Attendees

Ryan O'Brien, David Kim, Derek Washington, Kevin Park

## Discussion

**Ryan O'Brien:** What's our fallback if this doesn't work? The ArgoCD integration is coming along well. The GitOps sync is completing within 90 seconds of merge.

**Ryan O'Brien:** We can build on the lessons from the Developer Portal project which shipped in January 2025.

**David Kim:** To add to Ryan O'Brien's point, what's the simplest path to getting this done? I've been working with GitHub Actions and the performance characteristics are solid — The CI pipeline runs in about 8 minutes end to end.

**Derek Washington:** I agree with David Kim. the trade-off we need to evaluate is straightforward. Our Prometheus setup needs attention. The scrape interval is at 15s which gives us good resolution without overhead.

**Kevin Park:** To add to Derek Washington's point, i'm not fully convinced yet. Our Helm setup needs attention. The chart templating is correctly parameterizing per-environment configs.

## Related Projects

The Observability Initiative project (in_progress) is relevant to this discussion.

The Service Mesh Rollout project (in_progress) is relevant to this discussion.

## Action Items

- Ryan O'Brien will investigate ArgoCD configuration and report findings by end of week
- David Kim will investigate GitHub Actions configuration and report findings by end of week
- Derek Washington will investigate Helm configuration and report findings by end of week
