# ADR-005: Select MLflow over Weights & Biases for experiment tracking

**Status:** Accepted

**Date:** February 2025

**Author:** Emily Zhang

## Context

The ML Recommendation Engine needs experiment tracking for model development. Team evaluated self-hosted vs SaaS options.

This ADR formalizes the decision to selected mlflow over weights & biases for experiment tracking made in February 2025.

## Decision

Deploy self-hosted MLflow for experiment tracking, model registry, and artifact storage

**Rationale:** Open-source avoids vendor lock-in, integrates with existing Airflow pipelines, and keeps experiment data on our infrastructure

## Alternatives Considered

- Weights & Biases — better UX but SaaS-only with data residency concerns
- Neptune.ai — good collaboration features but expensive at scale
- ClearML — open-source but less mature model registry

## Consequences

- Need to maintain MLflow infrastructure
- Self-service model deployment through MLflow model registry
- Training data stays on-premises
