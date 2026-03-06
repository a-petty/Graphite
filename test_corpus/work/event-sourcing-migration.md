# Event Sourcing Migration — Project Summary

## Project Overview

The Event Sourcing Migration project was initiated in April 2025 and is led by John Doe. The core team includes Daniel Reeves, Lisa Wang, Ryan O'Brien.

Goal: Migrate core domain services to event sourcing for auditability and replay capability.

## Technical Architecture

The technical stack includes Kafka, PostgreSQL, Go, Python.

Kafka was selected for durable event log with replay capability.

PostgreSQL was selected for event store with JSONB for flexible event schemas.

Go was selected for high-throughput event consumer services.

Python integrates well with the existing infrastructure.

## Key Decisions

1. Approved Istio for service mesh (October 2024)
1. Scheduled CockroachDB migration for Q1 2025 (November 2024)

## Timeline

- April 2025: Project kickoff and initial architecture design
- Ongoing: Active development and iteration

## Status

The project is in the planning phase with design work underway.
