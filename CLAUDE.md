# CLAUDE.md (repo contract / human-readable project guide)

## Project Overview
This project is to try a bunch of different RL algorithms and configurations to ultimately find one that will play the game 2048 at a high level

## Source of Truth
- Design docs are the source of truth.
- When in doubt, check design docs first, then ask.

## Operating Model
This repo uses an agentic workflow orchestrated by a separate **workflow-orchestrator** agent.
- `CLAUDE.md` is the project’s stable contract: purpose, constraints, and conventions.
- Workflow logic (steps, gating, escalation rules) lives in the orchestrator prompt.

---

## Design Doc Drafting Workflow (Main Conversation)

This workflow governs how the **main Claude conversation** collaborates with the user
to create or update design documentation.

This workflow is intentionally interactive and user-driven.

### Principles
- The user remains the decision-maker at all times.
- Claude assists by drafting, organizing, and identifying gaps.
- Claude must NOT take control or commit to decisions without user approval.
- No repository files are created or modified during drafting.
- No subagents are invoked during drafting.

### Step 1 — Establish Direction (Optional)
Claude may ask up to 1–3 targeted questions to clarify:
- document type (ADR, design-spec, architecture, note)
- primary goal or scope
- known constraints

If the user prefers to start drafting immediately, skip questions.

### Step 2 — Draft & Flesh Out
Claude produces a concrete draft or structured outline that:
- reflects the user’s stated intent
- includes explicit invariants (MUST / SHALL / NEVER)
- clearly labels assumptions and suggestions
- calls out open questions instead of deciding

### Step 3 — Iterate (User-Led)
Repeat as needed:
- user feedback → revise draft
- Claude proposes alternatives or tradeoffs **as suggestions**
- ambiguous choices are surfaced as questions, not decisions

Claude must not finalize or commit the document without explicit user approval.

### Step 4 — Finalize Doc Draft Packet
When the user says the design is settled, Claude outputs ONLY a
**Doc Draft Packet (verbatim)** suitable for handoff to a documentation agent.

### Step 5 — Handoff (Explicit)
Only after explicit user approval may Claude invoke a documentation agent
(e.g., `design-doc-author`) to apply the document, validate alignment, and log decisions.

---

## Conventions
We are working in a single main branch.  different RL algorithms will be separated into subfolders.
Feature work shall be done in feature branches and PR's are made to merge into main

## Design Docs
- Updating docs: handled via Design Doc Drafting Workflow + documentation agents
- If docs conflict with code: follow docs and flag the inconsistency

## Testing
### Philosophy
- Don't fix unit tests just to pass
- Look at what they are actually testing and the code
- Determine if it's really a bug
- If uncertain, ask

## Agents (Reference Only)
These are the agents the orchestrator may invoke:
- feature-implementation-planner
- code-change-advisor
- spec-alignment-reviewer
- test-planner
- pr-review-architect
- design-doc-author
- design-doc-sync

## Decisions
Follow context/DECISION_LOGGING.md. Log qualifying decisions via python3 scripts/decision_log.py.

Before proposing or making a decision, query via:
- `python3 scripts/decision_log.py search ...`
- `python3 scripts/decision_log.py latest ...`

If query fails, STOP.

**Note:** During the Design Doc Drafting Workflow, suggestions, questions,
and alternative proposals are allowed.
Only finalized, user-approved commitments are considered decisions.
