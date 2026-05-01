# TRACE Logs

Machine-readable provenance documentation for AI–human collaboration on this project, following the TRACE (Transparent Research AI Collaboration Environment) protocol v0.3.

## What's here

- **`sessions/`** — 25 individual session logs (JSON), covering the manuscript-preparation phase from 2026-03-21 through 2026-04-30. Each file is one TRACE session corresponding to one focused work block (typically a few hours).
- **`project_summary.json`** — aggregated metrics across all sessions: decision counts, contribution attribution, annotation categories, intervention rates, participant list, and a session manifest.
- **`build_summary.py`** — the script used to produce `project_summary.json` from `sessions/`. Run with `python build_summary.py` from this directory's parent (the repo root). Documented inline; takes no arguments.

## How to read a session

Each session JSON has a top-level `metadata` block (project, participants, environment) and an `events` array. Event types you'll encounter:

| Type | What it records |
|------|-----------------|
| **decision** | A methodology choice (which approach, which parameters, how to handle ambiguous data). Includes `rationale`, `proposed_by` (`human` or `ai`), and `disposition` (`accepted` / `revised` / `rejected` / `proposed`-pending). |
| **contribution** | A concrete deliverable (code, document, analysis). Includes `direction` (whose idea it was) and `execution` (who did the work) — both ∈ {`human`, `ai`, `collaborative`}. |
| **annotation** | A `category`-tagged note: `correction` (human caught and fixed an AI mistake, with `corrects_event_ids` linking to the corrected event), `gotcha` (surprising discovery), `learning` (a generalizable insight), `observation`, `todo`, or `question`. |
| **tool_call** | Calls to domain MCP tools that produce results used in the workflow. Not heavily used in this project. |
| **state_change** | Environment / model / configuration changes (e.g., switching Claude model versions). |

## Reading `project_summary.json`

The `_definitions` block at the top of the file documents every metric inline so the summary can be read standalone, including the formulas for `acceptance_rate` and the two `intervention_rate` variants.

Two important conventions:
- **`acceptance_rate`** denominator is *resolved* decisions (accepted + revised + rejected). Pending decisions are excluded. Pending = the AI proposed something and the human never explicitly resolved it before the session ended; the protocol forbids the AI from resolving its own proposals.
- **`intervention_rate_per_event`** mirrors the upstream `trace_project_summary` tool, and uses total events as the denominator — useful for cross-project comparison. A second metric, **`intervention_rate_per_decision`**, divides by total decisions and is the more intuitive measure of "how often did the human override the AI."

## Coverage

These logs document AI–human collaboration during the **manuscript-preparation phase** (2026-03-21 → 2026-04-30). The original data analysis pipeline (chunking, embedding, consensus UMAP, hyperparameter validation, clustering) was completed before TRACE was integrated and is **not** covered here.

This is to our knowledge the first project to publish TRACE-format provenance logs alongside a peer-reviewed submission. Conventions for how AI–human collaboration should be documented in academic publishing are still emerging; we have erred toward transparency and inclusion (e.g., open pending decisions are surfaced rather than hidden) over polish.

## Caveats and known imperfections

- **Schema-version coverage.** This archive uses TRACE v0.3 exclusively, the first stable schema for the protocol. Earlier sessions from the project's setup phase (Feb–Mar 2026) used draft v0.1 / v0.2 schemas with inconsistent field names and were excluded from this archive in favor of the stable record. The TRACE protocol specification is at https://trace-protocol.org/v0.3 (reference implementation: `trace-mcp` on PyPI).
- **Path references in early sessions** reflect the directory naming at the time of logging — for example, several events refer to `jan_2026_manuscript/...`, which was the previous name of what is now `manuscripts_and_presentations/manuscript/...`. We have not retroactively rewritten paths in the logs because doing so would conflict with the protocol's "never alter past data" rule. One personal absolute path (`/Users/echoes/...`) was redacted to `<TRACE-repo>/...` in `trace_20260413_799f12.json`; no other content was modified.
- **Pending decisions and still-active sessions are surfaced explicitly** in `project_summary.json` under `open_items`. Several sessions ended with proposals the human never explicitly accepted or rejected; those are recorded as `disposition: "proposed"` rather than silently resolved. A few sessions are also marked `status: "active"` despite being weeks old — these are sessions that were not formally closed at the end of a work block.
- **Participant aliases** (e.g., `oliver` / `oliver-muellerklein` / generic `human`) were collapsed to canonical IDs in `project_summary.json` for readability; the raw session files preserve the original IDs as logged. AI-side actor IDs distinguish between specific Claude model versions where the model was explicitly recorded (`claude-opus-4-6`, `claude-opus-4-7`) versus the generic `claude` label used when the model wasn't recorded.
- **Tool-call events are absent** from the project's TRACE record. Domain MCP tools (other than TRACE itself) were not systematically logged via `trace_log_tool_call`. This is a coverage gap, not a sign that no tools were used; it reflects the early-adopter state of the project's TRACE integration.

## Citation

If you use these logs or the TRACE protocol in your own work, please cite the protocol specification at https://trace-protocol.org/v0.3 and reference this dataset by the repository URL.
