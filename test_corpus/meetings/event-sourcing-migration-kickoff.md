# Event Sourcing Migration — Kickoff Meeting — 2025-04-18

## Attendees

John Doe, Daniel Reeves, Lisa Wang, Ryan O'Brien

## Project Overview

**John Doe:** Welcome everyone. Today we're kicking off the Event Sourcing Migration project. Our goal is to Migrate core domain services to event sourcing for auditability and replay capability. Let me walk through the approach and timeline.

## Technical Approach

**John Doe:** We'll be using Kafka for this project. The rationale is durable event log with replay capability.

**John Doe:** We'll be using PostgreSQL for this project. The rationale is event store with JSONB for flexible event schemas.

**John Doe:** We'll be using Go for this project. The rationale is high-throughput event consumer services.

**John Doe:** We'll be using Python for this project. The rationale is proven track record and team familiarity.

**John Doe:** This connects to our earlier decision to chose apollo server over hasura for graphql back in July 2024.

## Timeline

- Phase 1: Architecture and design (April 2025)
- Phase 2: Core implementation (following month)
- Phase 3: Testing and rollout (month after)

## Action Items

- John Doe will draft the Kafka section of the technical design doc by Friday
- Daniel Reeves will draft the PostgreSQL section of the technical design doc by Friday
- Lisa Wang will draft the PostgreSQL section of the technical design doc by Friday
