# Engineering Retrospective — 2024-10-05

## Attendees

Sarah Chen, Marcus Johnson, Priya Patel, David Kim

## Discussion

**Sarah Chen:** I want to make sure we're systematic about this. Our ArgoCD setup needs attention. The GitOps sync is completing within 90 seconds of merge.

**Marcus Johnson:** Sarah Chen raises a good point. this is going to be a big win for us. Our GitHub Actions setup needs attention. The CI pipeline runs in about 8 minutes end to end.

**Marcus Johnson:** This connects to our earlier decision to selected react over vue for dashboard redesign back in August 2024.

**Priya Patel:** Building on what Marcus Johnson said, what's our fallback if this doesn't work? Our Helm setup needs attention. The chart templating is correctly parameterizing per-environment configs.

**David Kim:** I agree with Priya Patel. the trade-off we need to evaluate is straightforward. We should discuss our Playwright configuration. We're making good progress on the implementation.

## Related Projects

The Dashboard Redesign project (completed) is relevant to this discussion.

## Action Items

- Sarah Chen will investigate Playwright configuration and report findings by end of week
- Marcus Johnson will investigate ArgoCD configuration and report findings by end of week
- Priya Patel will investigate GitHub Actions configuration and report findings by end of week
