# Data Lake Migration — Kickoff Meeting — 2024-06-09

## Attendees

James Liu, Emily Zhang, David Kim

## Project Overview

**James Liu:** Welcome everyone. Today we're kicking off the Data Lake Migration project. Our goal is to Consolidate disparate data stores into unified analytics warehouse. Let me walk through the approach and timeline.

## Technical Approach

**James Liu:** We'll be using Snowflake for this project. The rationale is scalable cloud warehouse with separation of compute and storage.

**James Liu:** We'll be using dbt for this project. The rationale is version-controlled data transformations and testing.

**James Liu:** We'll be using Airflow for this project. The rationale is orchestration for complex ETL dependency graphs.

**James Liu:** We'll be using Spark for this project. The rationale is large-scale data processing for historical backfill.

**Emily Zhang:** We'll be using S3 for this project. The rationale is cost-effective raw data landing zone.

## Timeline

- Phase 1: Architecture and design (June 2024)
- Phase 2: Core implementation (following month)
- Phase 3: Testing and rollout (month after)

## Action Items

- James Liu will draft the Snowflake section of the technical design doc by Friday
- Emily Zhang will draft the Snowflake section of the technical design doc by Friday
- David Kim will draft the Snowflake section of the technical design doc by Friday
