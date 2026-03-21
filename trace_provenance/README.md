# TRACE Provenance Logs

This directory contains machine-readable provenance documentation for AI-human collaboration in this project, following the [TRACE](https://trace-protocol.org) (Transparent Research AI Collaboration Environment) protocol v0.3.0.

## What Is TRACE?

TRACE is a protocol for documenting AI contributions to research with full attribution transparency. It records decisions, contributions, corrections, and annotations with structured metadata about who proposed, directed, and executed each action.

## Coverage

These logs cover the **manuscript preparation phase** (February-March 2026), including:

- Venue selection decisions
- Manuscript revision for multiple submission targets
- Reference verification and correction
- Supplementary material restructuring

The original data analysis pipeline (topic modeling, consensus UMAP, etc.) was completed before TRACE was integrated and is not covered by these logs.

## Files

- `project_summary.json` — Aggregated metrics across all sessions
- `sessions/` — Individual session logs (JSON)

## Key Statistics

| Metric | Value |
|---|---|
| Sessions | 17 |
| Total events | 33 |
| Contributions | 11 |
| Decisions | 5 |
| Annotations | 15 |
| Human corrections | 1 |
| Intervention rate | 3% |

## How to Read the Logs

Each session file contains timestamped events. Key fields:

- `event_type`: decision, contribution, annotation, state_change, tool_call
- `proposed_by_type` / `resolved_by_type`: human or ai
- `direction`: who had the idea (human_directed, ai_directed, collaborative)
- `execution`: who did the work (human, ai, collaborative)

## TRACE Protocol

- **Version**: 0.3.0
- **Documentation**: [trace-protocol.org](https://trace-protocol.org)
- **Package**: `trace-mcp` on PyPI
