# ADR-003: Standardize on PagerDuty for incident management

**Status:** Accepted

**Date:** November 2024

**Author:** Derek Washington

## Context

On-call rotation has been managed via a shared Google Sheet, leading to missed pages and unclear escalation paths. The payment and Redis incidents highlighted the need for proper tooling.

This ADR formalizes the decision to adopted pagerduty for on-call rotation made in November 2024.

## Decision

Adopt PagerDuty for all on-call scheduling, alert routing, and incident escalation

**Rationale:** PagerDuty integrates natively with Datadog and provides mobile push, SMS, and phone escalation required for 24/7 coverage

## Alternatives Considered

- Opsgenie — cheaper but weaker Datadog integration
- VictorOps — being sunset in favor of Splunk On-Call
- Custom Slack bot — insufficient for compliance requirements

## Consequences

- $500/month licensing cost
- Need to migrate existing Datadog alert rules to PagerDuty services
- Enables SLA tracking and incident analytics
