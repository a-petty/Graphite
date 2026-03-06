# API Migration — Kickoff Meeting — 2024-07-13

## Attendees

Priya Patel, Marcus Johnson, Alex Petrov

## Project Overview

**Priya Patel:** Welcome everyone. Today we're kicking off the API Migration project. Our goal is to Migrate internal services from REST to GraphQL for flexible querying and reduced over-fetching. Let me walk through the approach and timeline.

## Technical Approach

**Priya Patel:** We'll be using GraphQL for this project. The rationale is reduces over-fetching and enables client-driven queries.

**Marcus Johnson:** We'll be using Apollo Server for this project. The rationale is mature ecosystem with caching and federation support.

**Alex Petrov:** We'll be using REST for this project. The rationale is proven track record and team familiarity.

**Priya Patel:** We'll be using Swagger for this project. The rationale is documentation for legacy REST endpoints during transition.

**Priya Patel:** This connects to our earlier decision to adopted snowflake as primary data warehouse back in June 2024.

## Timeline

- Phase 1: Architecture and design (July 2024)
- Phase 2: Core implementation (following month)
- Phase 3: Testing and rollout (month after)

## Action Items

- Priya Patel will draft the Apollo Server section of the technical design doc by Friday
- Marcus Johnson will draft the GraphQL section of the technical design doc by Friday
- Alex Petrov will draft the GraphQL section of the technical design doc by Friday
