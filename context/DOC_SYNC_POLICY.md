# Design Doc ↔ Decision Log Sync Policy

Goal: design docs and decision log must not drift.

## Source of truth
- Design docs define intended architecture/behavior.
- The decision log records approved choices.

They must be consistent. If a conflict is detected, do NOT resolve it locally.

## Hard Rule: Conflict Escalation
If any agent finds a conflict between:
- design docs and an approved decision, OR
- design docs and an approved Spec Packet, OR
- decision log and approved Spec Packet,

the agent MUST:
1) STOP
2) Capture a Conflict Packet (see below)
3) Invoke the design-doc-sync agent to reconcile by updating the design docs
4) Ensure the reconciliation is logged in the changelog

No work proceeds until the conflict is resolved.

## Conflict Packet (verbatim)
Include:
- conflict_summary: 1 sentence
- detected_by: agent name
- decision_ids: [DEC-####] (if applicable)
- doc_refs: file paths + headings/sections if possible
- spec_refs: (spec packet name/id if applicable)
- why_it_matters: impact if unresolved
- proposed_doc_change: what should change in docs (high-level)
- questions_for_user: only if the doc change requires user intent clarification
