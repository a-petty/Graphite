# Incident Postmortem — Redis cluster failover — 15 min elevated latency — 2024-10-22

## Attendees
Derek Washington, Priya Patel, David Kim

## Incident Summary

Redis cluster failover — 15 min elevated latency. The incident was detected by automated monitoring and the team responded within 5 minutes.


## Root Cause Analysis

**Derek Washington:** The root cause was a configuration issue in PagerDuty. A recent deployment changed the connection pool settings without updating the timeout values.


## Resolution

**Priya Patel:** We rolled back the configuration change and added monitoring alerts to catch similar issues in the future.


## Action Items

- Derek Washington will implement additional safeguards to prevent recurrence

- Priya Patel will implement additional safeguards to prevent recurrence
