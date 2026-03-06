# Incident Postmortem — Airflow scheduler crash — 2 hour data pipeline delay — 2025-03-15

## Attendees

James Liu, David Kim, Derek Washington

## Incident Summary

Airflow scheduler crash — 2 hour data pipeline delay. This affected the Data Lake Migration service. The incident was detected by automated monitoring and the team responded within 5 minutes.

## Root Cause Analysis

**James Liu:** The root cause was identified: Airflow scheduler process exhausted file descriptors due to excessive DAG file parsing in the data lake migration DAGs

The affected components included Airflow, Kubernetes, S3.

## Resolution

**David Kim:** Increased file descriptor limits, consolidated DAG definitions, added health check alerting for scheduler process

## Lessons from Prior Incidents

**Derek Washington:** This is similar to the graphql gateway timeout incident from January 2025. We should review whether the mitigations from that incident apply here.

## Action Items

- James Liu will implement safeguards for Airflow to prevent recurrence
- David Kim will implement safeguards for Kubernetes to prevent recurrence
- Derek Washington will implement safeguards for S3 to prevent recurrence
