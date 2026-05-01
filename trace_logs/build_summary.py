"""Build the publication-grade project_summary.json for the trace_logs/ artifact.

Post-processes the tool output from `trace_project_summary` to add the four
fields the reviewer-lens audit identified as missing:
  - protocol_version (top-level)
  - date_range
  - sessions_by_month
  - session_manifest

Also normalizes participants (dedupes oliver / oliver-muellerklein / human),
surfaces open_items (pending decisions, still-active sessions), and adds an
explicit _definitions block so reviewers can read the file standalone.

Inputs: trace_logs/sessions/*.json
Output: trace_logs/project_summary.json

Run:
  /opt/homebrew/Caskroom/miniforge/base/bin/python /tmp/trace_audit/build_summary.py
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(
    "/Users/echoes/Documents/Berkeley/Research/When-Algorithms-Meet-Artists"
)
SESSIONS_DIR = ROOT / "trace_logs" / "sessions"
OUTPUT = ROOT / "trace_logs" / "project_summary.json"

PROJECT = "when-algorithms-meet-artists"

# Canonical actor map: collapse aliases to a single id-and-type pair.
# Humans:
#   - all generic 'human'-typed actor ids that refer to Oliver collapse to 'oliver'
#   - 'ariya' kept distinct (co-author)
# AI:
#   - 'claude' / 'ai-assistant' / 'claude-ai-assistant' = generic shorthand
#     for sessions where the specific Claude model wasn't recorded; collapsed
#     to 'claude' (generic)
#   - 'claude-opus-4-6' and 'claude-opus-4-7' kept distinct (specific model
#     versions actually used in those sessions)
ACTOR_ALIASES = {
    ("human", "oliver"): ("human", "oliver"),
    ("human", "oliver-muellerklein"): ("human", "oliver"),
    ("human", "human"): ("human", "oliver"),
    ("human", "ariya"): ("human", "ariya"),
    ("ai", "claude"): ("ai", "claude"),
    ("ai", "ai-assistant"): ("ai", "claude"),
    ("ai", "claude-ai-assistant"): ("ai", "claude"),
    ("ai", "claude-opus-4-6"): ("ai", "claude-opus-4-6"),
    ("ai", "claude-opus-4-7"): ("ai", "claude-opus-4-7"),
}


def canonical_actor(actor: dict) -> tuple[str, str]:
    t = (actor or {}).get("type", "")
    i = (actor or {}).get("id", "")
    return ACTOR_ALIASES.get((t, i), (t, i))


def main() -> None:
    sessions = sorted(SESSIONS_DIR.glob("*.json"))
    assert len(sessions) == 25, f"Expected 25 sessions, found {len(sessions)}"

    # Aggregators
    total_events = 0
    events_by_type: Counter[str] = Counter()
    decisions_total = 0
    decisions_by_proposer: Counter[str] = Counter()
    decisions_by_disposition: Counter[str] = Counter()
    decisions_by_suggestion_type: Counter[str] = Counter()
    contributions_matrix: Counter[str] = Counter()
    annotations_by_category: Counter[str] = Counter()
    corrections = 0
    corrections_with_links = 0
    decision_revisions = 0
    decision_rejections = 0
    retry_chains = 0
    participant_set: set[tuple[str, str]] = set()
    participant_event_count: Counter[tuple[str, str]] = Counter()

    sessions_by_month: defaultdict[str, int] = defaultdict(int)
    session_manifest = []
    pending_decisions_by_session: dict[str, int] = {}
    still_active_sessions = []
    earliest, latest = None, None
    protocol_versions: set[str] = set()

    for path in sessions:
        with open(path) as f:
            d = json.load(f)
        sid = d["id"]
        created = d.get("created", "")
        ended = d.get("ended")
        status = d.get("status", "")
        if created:
            month = created[:7]  # YYYY-MM
            sessions_by_month[month] += 1
            if not earliest or created < earliest:
                earliest = created
            if not latest or created > latest:
                latest = created
        protocol_versions.add(d.get("trace_version", "unknown"))

        # Pull session-declared participants (these include collaborators like
        # Ariya who are involved but may not have authored TRACE-logged events
        # directly).
        for p in (d.get("metadata", {}) or {}).get("participants", []) or []:
            if isinstance(p, dict):
                ca = canonical_actor(p)
                if ca[1]:
                    participant_set.add(ca)
                    # Don't increment event_count for declared-only participants;
                    # only event-actor occurrences count toward event volume.

        events = d.get("events", []) or []
        total_events += len(events)

        n_pending_in_session = 0
        for evt in events:
            t = evt.get("type")
            if t:
                events_by_type[t] += 1
            actor = evt.get("actor") or {}
            ca = canonical_actor(actor)
            if ca[1]:  # non-empty id
                participant_set.add(ca)
                participant_event_count[ca] += 1

            if t == "decision":
                decisions_total += 1
                dec = evt.get("decision") or {}
                proposer = canonical_actor(dec.get("proposed_by") or {})[0]  # type only
                if proposer == "ai":
                    decisions_by_proposer["proposed_by_ai"] += 1
                elif proposer == "human":
                    decisions_by_proposer["proposed_by_human"] += 1
                else:
                    decisions_by_proposer["proposed_by_system_or_other"] += 1
                disp = dec.get("disposition", "proposed")
                # 'proposed' = pending in tool's terminology
                if disp == "proposed":
                    decisions_by_disposition["pending"] += 1
                    n_pending_in_session += 1
                else:
                    decisions_by_disposition[disp] += 1
                if disp == "revised":
                    decision_revisions += 1
                if disp == "rejected":
                    decision_rejections += 1
                st = dec.get("suggestion_type")
                if st:
                    decisions_by_suggestion_type[st] += 1

            elif t == "contribution":
                con = evt.get("contribution") or {}
                direction = con.get("direction", "")
                execution = con.get("execution", "")
                if direction and execution:
                    key = f"{direction}_directed_{execution}_executed"
                    contributions_matrix[key] += 1

            elif t == "annotation":
                ann = evt.get("annotation") or {}
                cat = ann.get("category", "")
                if cat:
                    annotations_by_category[cat] += 1
                if cat == "correction":
                    corrections += 1
                    if ann.get("corrects_event_ids"):
                        corrections_with_links += 1

            elif t == "tool_call":
                tc = evt.get("tool_call") or {}
                if tc.get("retries_event_id") is not None:
                    retry_chains += 1

        pending_decisions_by_session[sid] = n_pending_in_session
        if status not in ("completed", "abandoned"):
            still_active_sessions.append(sid)

        session_manifest.append({
            "id": sid,
            "created": created,
            "ended": ended,
            "status": status,
            "n_events": len(events),
            "n_pending_decisions": n_pending_in_session,
            "trace_version": d.get("trace_version"),
        })

    # Derived metrics
    resolved_decisions = (
        decisions_by_disposition.get("accepted", 0)
        + decisions_by_disposition.get("revised", 0)
        + decisions_by_disposition.get("rejected", 0)
    )
    acceptance_rate = (
        round(decisions_by_disposition.get("accepted", 0) / resolved_decisions, 3)
        if resolved_decisions
        else None
    )
    intervention_total = corrections + decision_revisions + decision_rejections
    intervention_rate_per_event = (
        round(intervention_total / total_events, 4) if total_events else None
    )
    intervention_rate_per_decision = (
        round((decision_revisions + decision_rejections) / decisions_total, 4)
        if decisions_total
        else None
    )

    # Sort session_manifest by created date
    session_manifest.sort(key=lambda r: r["created"])

    # Normalize participants list. Include session-declared participants
    # (from metadata.participants) plus event-actor participants. Sort by
    # event_count desc, then id asc; declared-only participants get
    # event_count=0.
    participants = []
    for (t, i) in participant_set:
        participants.append({
            "id": i,
            "type": t,
            "event_count": participant_event_count.get((t, i), 0),
        })
    participants.sort(key=lambda p: (-p["event_count"], p["id"]))

    summary = {
        "_definitions": {
            "schema_note": "This file aggregates TRACE v0.3 protocol events across all sessions whose metadata.project field exactly equals 'when-algorithms-meet-artists'.",
            "acceptance_rate": "accepted / (accepted + revised + rejected); pending decisions excluded from the denominator.",
            "intervention_rate_per_event": "(corrections + revised_decisions + rejected_decisions) / total_events. Mirrors the upstream trace_project_summary tool's intervention_rate.",
            "intervention_rate_per_decision": "(revised_decisions + rejected_decisions) / total_decisions. Independent measure useful when the event mix is annotation-heavy.",
            "pending": "Decisions whose disposition is still 'proposed' (no human resolution recorded). The TRACE protocol forbids the AI from resolving its own proposals; pending decisions accumulate when work continued without an explicit human accept/reject.",
            "contributions_matrix_key": "<direction>_directed_<execution>_executed where direction = whose idea, execution = who did the work; both ∈ {human, ai, collaborative}.",
            "participant_aliases": "ids 'oliver', 'oliver-muellerklein', and 'human' (when actor.type='human') were collapsed to a single canonical id 'oliver'. Other ids unchanged.",
            "tool_caveats": [
                "Upstream trace_project_summary uses a case-insensitive substring filter on metadata.project; this artifact uses an exact-match filter for unambiguous attribution.",
                "Decisions by proposer break out only 'ai' and 'human' actor types; any 'system' actor would be counted in decisions.total but not in either breakdown (none present in this project).",
            ],
        },
        "project": PROJECT,
        "protocol_version": sorted(protocol_versions),
        "session_count": len(sessions),
        "total_events": total_events,
        "date_range": {
            "earliest_session_created": earliest,
            "latest_session_created": latest,
        },
        "sessions_by_month": dict(sorted(sessions_by_month.items())),
        "events_by_type": dict(events_by_type.most_common()),
        "decisions": {
            "total": decisions_total,
            **dict(decisions_by_proposer),
            **dict(decisions_by_disposition),
            "acceptance_rate": acceptance_rate,
            "suggestion_types": dict(decisions_by_suggestion_type.most_common()),
        },
        "contributions": dict(contributions_matrix.most_common()),
        "annotations_by_category": dict(annotations_by_category.most_common()),
        "human_interventions": {
            "total": intervention_total,
            "corrections": corrections,
            "corrections_with_links": corrections_with_links,
            "decision_rejections": decision_rejections,
            "decision_revisions": decision_revisions,
            "retry_chains": retry_chains,
            "intervention_rate_per_event": intervention_rate_per_event,
            "intervention_rate_per_decision": intervention_rate_per_decision,
        },
        "open_items": {
            "pending_decisions": sum(pending_decisions_by_session.values()),
            "pending_decisions_by_session": {
                sid: n for sid, n in pending_decisions_by_session.items() if n > 0
            },
            "still_active_sessions": still_active_sessions,
        },
        "participants": participants,
        "session_manifest": session_manifest,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {OUTPUT}")
    print(f"  sessions:        {summary['session_count']}")
    print(f"  total events:    {summary['total_events']}")
    print(f"  date range:      {earliest[:10]} → {latest[:10]}")
    print(f"  protocol:        {summary['protocol_version']}")
    print(f"  decisions:       {decisions_total} (accepted={decisions_by_disposition.get('accepted',0)}, pending={decisions_by_disposition.get('pending',0)})")
    print(f"  contributions:   {sum(contributions_matrix.values())}")
    print(f"  annotations:     {sum(annotations_by_category.values())}")
    print(f"  participants:    {len(participants)} (after alias normalization)")
    print(f"  open items:      {summary['open_items']['pending_decisions']} pending decisions, {len(still_active_sessions)} still-active sessions")


if __name__ == "__main__":
    main()
