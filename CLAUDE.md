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

## Governance Model (Decision Master)
- `decision-master` is the central authority for decision lookup, decision writing, and design-doc compliance.
- Other agents must NOT write decisions or change governance logs directly.
- Proposed decisions may be treated as effective immediately when marked effective by decision-master.
- The only user-involved governance step is reviewing pending governance changes (see below).

---

## Design Doc Drafting Workflow (Main Conversation)

Use this workflow when creating or updating design documentation.

### Principles
- Drafting is interactive in the main conversation.
- The user controls the *design intent* during drafting.
- No repo files are created/modified during drafting.
- No governance files (decisions/changelog) are written during drafting.
- Finalization + application is handled by agents, routed through decision-master.

### Step 1 — Establish Direction (Optional)
Claude may ask up to 1–3 targeted questions to clarify:
- document type (ADR, design-spec, architecture, note)
- doc location/path
- primary scope and constraints

If the user prefers to start drafting immediately, skip questions.

### Step 2 — Draft & Iterate (User-Led)
Claude produces a concrete draft or outline and iterates based on user feedback.
Drafts MUST clearly label:
- invariants (MUST / SHALL / NEVER)
- assumptions
- open questions
- any decision-like rules implied by the doc text

### Step 3 — Finalize Doc Draft Packet (Verbatim)
When the user says the design draft is settled, Claude outputs ONLY:

## Doc Draft Packet (verbatim)

doc_path: <path/to/doc.md>
doc_type: <ADR | design-spec | design-note | architecture>
intent:
- <1–3 sentences>

proposed_content:
<full doc text OR patch-style edits>

governance_requests:
- <bullets: “this doc implies a rule/constraint such as …”>
- <include removals/replacements explicitly, if applicable>

open_questions:
- None

change_summary:
- <what changed vs current doc>

### Step 4 — Apply Draft to Docs (design-doc-author)
Invoke `design-doc-author` with the Doc Draft Packet verbatim.

`design-doc-author` responsibilities:
- apply the doc content to the repository
- identify governance-impacting rules implied by the doc
- produce a Governance Request Packet for decision-master (verbatim)
- do NOT write decisions or changelog entries directly

### Step 5 — Governance Write (decision-master)
Invoke `decision-master` in GOVERNANCE_WRITE mode with the Governance Request Packet.

Decision-master responsibilities:
- write new/updated decisions to `context/decisions_active.jsonl`
- delete disapproved/replaced decisions from `context/decisions_active.jsonl`
- copy deleted decisions into `context/decisions_graveyard.jsonl` with kill metadata
- append doc-change record to `context/CHANGELOG_PENDING.md`

### Step 6 — Report Results
Claude reports:
- doc paths changed
- decisions added/updated/removed (IDs + 1-line summaries)
- changelog entry reference (if present)

No additional analysis. No further steps unless requested.

---

## Pending Governance Review Workflow (Main Conversation)

This is the only governance step that requires user input.

### What gets reviewed
- Pending/effective decisions written by decision-master (status: proposed, effective=true)
- Pending documentation changes recorded in `context/CHANGELOG_PENDING.md`

### Review actions
For each pending item, the user may:
- approve (ratify)
- alter
- decline
- replace

### After review
The workflow-orchestrator must:
- identify impacted code/docs/tests for altered/declined/replaced items
- fix the repository to match the reviewed outcomes
- re-run decision-master PR gate when applicable

---

## Proposed Decision Review Workflow (Manual Governance Review)

Use this workflow when you want to review, approve, disapprove, or replace
proposed governance decisions.

This workflow is **manual** and user-driven.
It is the ONLY place where governance decisions are explicitly approved or rejected.

### Scope
- Applies to decisions in `context/decisions_active.jsonl` with `status: proposed`
- May also include related pending design-doc changes

### Step 1 — Enumerate Proposed Decisions
Invoke `decision-master` in GOVERNANCE_REVIEW mode.

Decision-master MUST return:
- a list of all proposed decisions
- for each:
  - DEC-#### id
  - decision text
  - source (doc / PR / agent)
  - impacted areas (code paths, design docs if known)

No filtering or interpretation by Claude at this step.

### Step 2 — User Review
For each proposed decision, the user may choose to:

- **Approve**
- **Disapprove**
- **Replace** (approve an alternative rule instead)

The user may also:
- approve multiple decisions together
- disapprove groups of related decisions
- replace one decision with multiple new ones

### Step 3 — Apply Governance Outcome (decision-master)
Invoke `decision-master` in GOVERNANCE_WRITE mode with the review outcomes.

Decision-master responsibilities:
- For **approved** decisions:
  - mark decision as `approved` in `decisions_active.jsonl`
- For **disapproved** decisions:
  - remove decision from `decisions_active.jsonl`
  - copy full entry to `decisions_graveyard.jsonl` with:
    - killed_ts
    - killed_by (user)
    - killed_reason
- For **replaced** decisions:
  - remove original decision(s) from active
  - copy originals to graveyard with replacement metadata
  - write replacement decision(s) as `approved`

All changes MUST be atomic.

### Step 4 — Determine Remediation Scope
If any decision was **disapproved or replaced**, decision-master MUST emit
a **Remediation Required Packet** including:

- removed_decisions: [DEC-####]
- added_decisions: [DEC-####]
- enforcement_scope: ENTIRE_REPO
- remediation_intent:
  - <plain-language description of what must now be true>
  - e.g. “No Python loops are allowed anywhere in the repository”

### Step 5 — Repo-Wide Remediation
Invoke `workflow-orchestrator` with the Remediation Required Packet.
Governance is NOT re-reviewed during remediation.

### Step 6 — Final Confirmation
After remediation completes:
- decision-master may optionally run an audit check
- Claude reports:
  - decisions approved / removed / replaced
  - remediation summary
  - files modified

No further action unless user requests another review.

---

## Decision & Design Audit Workflow (Manual Governance Audit)

Use this workflow to audit the **current state of governance**:
- decisions
- design documents
- their mutual alignment
- and their compliance with the *current governance standards* of the system.

This workflow is **manual** and user-initiated.

### Goals
This workflow exists to ensure that:
- design docs and decisions do not contradict each other
- obsolete or superseded rules are removed
- governance artifacts evolve as system standards evolve
- the decision set remains coherent, minimal, and enforceable

This workflow may result in:
- decision deletions (to graveyard)
- decision updates or replacements
- design doc updates
- governance metadata updates (e.g. future tags)

### Step 1 — Collect Governance State (decision-master)
Invoke `decision-master` in GOVERNANCE_AUDIT mode.

### Step 2 — Audit Findings Output
Decision-master MUST output an **Audit Findings Packet** including:

- conflicts:
  - decision vs design doc conflicts
  - doc vs decision conflicts
- obsolete_decisions:
  - decisions that should be removed
- upgrade_required:
  - decisions/docs that must be updated to meet current standards
- missing_governance:
  - rules implied by docs or agents but not captured as decisions
- recommended_actions:
  - ADD_DECISION
  - UPDATE_DECISION
  - DELETE_DECISION
  - UPDATE_DESIGN_DOC

Each item MUST include:
- affected_decisions (DEC-####)
- affected_docs (paths)
- rationale
- recommended fix

### Step 3 — User Review
Claude presents the Audit Findings Packet to the user.

For each recommended action, the user chooses:
- **Accept**
- **Reject**
- **Modify**

No changes are applied in this step.

### Step 4 — Apply Audit Outcomes (decision-master)
Invoke `decision-master` in GOVERNANCE_WRITE mode with the accepted audit outcomes.

### Step 5 — Apply Design Doc Updates (if any)
If design doc updates are required:
- invoke `design-doc-author` with Doc Draft Packets produced from the audit
- route any resulting governance changes back through decision-master

### Step 6 — Remediation (if required)
If any audit action introduces or changes enforcement rules:
- decision-master MUST emit a Remediation Required Packet
- invoke `workflow-orchestrator` to perform repo-wide remediation
- remediation MUST continue until audit rules are satisfied

### Step 7 — Final Report
Claude reports:
- decisions deleted / updated / added
- design docs updated
- remediation summary (if applicable)

No further action unless the user initiates another audit.

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
- decision-master

## Decisions
- Decision writing and updates are routed through `decision-master`.
- Other agents may query decisions, but they do not write to `context/decisions_active.jsonl`.

Effective decision rules:
- `approved` is effective
- `proposed` is effective only if `context.effective=true`
- `rejected` is forbidden

Note: During the Design Doc Drafting Workflow, suggestions/questions are allowed.
Only user-approved commitments become Doc Draft Packets and then get applied via doc agents + decision-master.


**Note:** During the Design Doc Drafting Workflow, suggestions, questions,
and alternative proposals are allowed.
Only finalized, user-approved commitments are considered decisions.
