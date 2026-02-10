#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional, Tuple
import time

ACTIVE_PATH = os.path.join("context", "decisions_active.jsonl")
GRAVEYARD_PATH = os.path.join("context", "decisions_graveyard.jsonl")
ID_RE = re.compile(r"^DEC-(\d{4,})$")
OBS_LOG_PATH = os.path.join("context", "observability.jsonl")

def _append_observability(event: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(OBS_LOG_PATH), exist_ok=True)
    with open(OBS_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")

def _caller_hint() -> Dict[str, Any]:
    # Best-effort: depends on what Claude Code exports into the environment.
    keys = [
        "CLAUDE_CODE_ENTRYPOINT",
        "CLAUDE_AGENT",
        "CLAUDE_SUBAGENT",
        "CLAUDE_ROLE",
    ]
    out = {k: os.environ.get(k) for k in keys if os.environ.get(k)}
    return out or {"caller": "unknown"}


def _now_iso_chicago() -> str:
    return datetime.now(ZoneInfo("America/Chicago")).isoformat(timespec="seconds")


def _ensure_file(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        open(path, "a", encoding="utf-8").close()


def _load_entries(path: str) -> List[Dict[str, Any]]:
    _ensure_file(path)
    entries: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    entries.append(obj)
            except Exception:
                # tolerate malformed historical lines
                continue
    return entries


def _write_entries_atomic(path: str, entries: List[Dict[str, Any]]) -> None:
    # rewrite the entire file atomically
    _ensure_file(path)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _max_dec_id(entries: List[Dict[str, Any]]) -> int:
    max_id = 0
    for e in entries:
        _id = e.get("id")
        if isinstance(_id, str):
            m = ID_RE.match(_id)
            if m:
                max_id = max(max_id, int(m.group(1)))
    return max_id


def _get_next_id() -> str:
    active = _load_entries(ACTIVE_PATH)
    grave = _load_entries(GRAVEYARD_PATH)
    next_num = max(_max_dec_id(active), _max_dec_id(grave)) + 1
    return f"DEC-{next_num:04d}"


def _emit(data: Any, fmt: str) -> None:
    if fmt == "json":
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return

    # text format (human-friendly)
    if isinstance(data, dict) and "entries" in data and isinstance(data["entries"], list):
        entries = data["entries"]
    elif isinstance(data, dict):
        entries = [data]
    elif isinstance(data, list):
        entries = data
    else:
        print(str(data))
        return

    for e in entries:
        _id = e.get("id", "?")
        ts = e.get("ts", "?")
        actor = e.get("actor", "?")
        status = e.get("status", "?")
        decision = e.get("decision", "")
        wf = (e.get("context") or {}).get("workflow", "?")
        step = (e.get("context") or {}).get("step", "?")
        print(f"{_id} [{status}] {ts} actor={actor} wf={wf} step={step}")
        if decision:
            print(f"  {decision}")
        reason = (e.get("context") or {}).get("reason")
        if reason:
            print(f"  reason: {reason}")
        refs = (e.get("context") or {}).get("refs") or []
        if refs:
            print(f"  refs: {', '.join(refs)}")
        # graveyard metadata (if present)
        killed_ts = e.get("killed_ts")
        if killed_ts:
            print(f"  killed_ts: {killed_ts}")
            print(f"  killed_by: {e.get('killed_by')}")
            print(f"  killed_reason: {e.get('killed_reason')}")
            rb = e.get("replaced_by")
            if rb:
                print(f"  replaced_by: {rb}")
        print("")


def _haystack(e: Dict[str, Any]) -> str:
    ctx = e.get("context") or {}
    parts = [
        str(e.get("id", "")),
        str(e.get("actor", "")),
        str(e.get("status", "")),
        str(e.get("decision", "")),
        str(ctx.get("reason", "")),
        " ".join(ctx.get("refs") or []),
        " ".join(ctx.get("options") or []),
        str(ctx.get("chosen", "")),
        " ".join(ctx.get("supersedes") or []),
        str(e.get("killed_reason", "")),
        str(e.get("killed_by", "")),
        str(e.get("replaced_by", "")),
    ]
    return " ".join(parts).lower()


def add_entry(args: argparse.Namespace) -> int:
    entry_id = _get_next_id()
    ts = _now_iso_chicago()

    ctx: Dict[str, Any] = {
        "workflow": args.workflow,
        "step": args.step,
        "reason": args.reason,
        "refs": args.refs or [],
    }
    if args.options:
        ctx["options"] = args.options
    if args.chosen:
        ctx["chosen"] = args.chosen
    if args.supersedes:
        ctx["supersedes"] = args.supersedes
    
    entry: Dict[str, Any] = {
        "id": entry_id,
        "ts": ts,
        "actor": args.actor,
        "status": args.status,
        "decision": args.decision,
        "context": ctx,
    }
    
    if args.tags:
        tags_registry = _load_tags()
        for t in args.tags:
            if t not in tags_registry:
                print(f"[HARD STOP] unknown tag: {t}", file=sys.stderr)
                return 2
        entry["tags"] = sorted(set(args.tags))

    try:
        _ensure_file(ACTIVE_PATH)
        with open(ACTIVE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[HARD STOP] Failed to append to active decisions: {e}", file=sys.stderr)
        return 2

    print(entry_id)
    return 0


def latest(args: argparse.Namespace) -> int:
    path = GRAVEYARD_PATH if args.graveyard else ACTIVE_PATH
    entries = _load_entries(path)
    n = max(1, int(args.n))
    out = {"entries": entries[-n:]}
    _emit(out, args.format)
    return 0


def get_by_id(args: argparse.Namespace) -> int:
    target = args.id.strip()
    if not ID_RE.match(target):
        print(f"[HARD STOP] Invalid id format: {target}", file=sys.stderr)
        return 2

    # search active first unless explicitly graveyard-only
    if not args.graveyard_only:
        for e in reversed(_load_entries(ACTIVE_PATH)):
            if e.get("id") == target:
                _emit(e, args.format)
                return 0

    for e in reversed(_load_entries(GRAVEYARD_PATH)):
        if e.get("id") == target:
            _emit(e, args.format)
            return 0

    print(f"[HARD STOP] Decision not found: {target}", file=sys.stderr)
    return 2


def search(args: argparse.Namespace) -> int:
    q = (args.q or "").strip().lower()
    if not q:
        print("[HARD STOP] search requires --q", file=sys.stderr)
        return 2

    entries: List[Dict[str, Any]] = []
    if args.graveyard_only:
        entries = _load_entries(GRAVEYARD_PATH)
    elif args.both:
        entries = _load_entries(ACTIVE_PATH) + _load_entries(GRAVEYARD_PATH)
    else:
        entries = _load_entries(ACTIVE_PATH)

    hits = [e for e in entries if q in _haystack(e)]
    n = max(1, int(args.n))
    out = {"query": args.q, "entries": hits[-n:]}
    _emit(out, args.format)
    return 0


def list_proposed(args: argparse.Namespace) -> int:
    entries = _load_entries(ACTIVE_PATH)
    proposed = [e for e in entries if e.get("status") == "proposed"]
    out = {"entries": proposed}
    _emit(out, args.format)
    return 0


def delete_decision(args: argparse.Namespace) -> int:
    target = args.id.strip()
    if not ID_RE.match(target):
        print(f"[HARD STOP] Invalid id format: {target}", file=sys.stderr)
        return 2

    active = _load_entries(ACTIVE_PATH)
    idx = None
    for i, e in enumerate(active):
        if e.get("id") == target:
            idx = i
            break

    if idx is None:
        print(f"[HARD STOP] Decision not found in active: {target}", file=sys.stderr)
        return 2

    entry = active.pop(idx)

    killed_ts = _now_iso_chicago()
    grave_entry = dict(entry)  # shallow copy ok (context is nested but fine)
    grave_entry["killed_ts"] = killed_ts
    grave_entry["killed_by"] = args.killed_by
    grave_entry["killed_reason"] = args.killed_reason
    if args.replaced_by:
        grave_entry["replaced_by"] = args.replaced_by

    try:
        # rewrite active atomically
        _write_entries_atomic(ACTIVE_PATH, active)
    except Exception as e:
        print(f"[HARD STOP] Failed to rewrite active decisions file: {e}", file=sys.stderr)
        return 2

    try:
        _ensure_file(GRAVEYARD_PATH)
        with open(GRAVEYARD_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(grave_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        # this is part of "delete" semantics; treat as HARD STOP so we don't lose the decision
        print(f"[HARD STOP] Deleted from active but failed to write graveyard: {e}", file=sys.stderr)
        return 2

    print(target)
    return 0

TAGS_PATH = os.path.join("context", "decision_tags.json")

def _ensure_tags_file() -> None:
    os.makedirs(os.path.dirname(TAGS_PATH), exist_ok=True)
    if not os.path.exists(TAGS_PATH):
        with open(TAGS_PATH, "w", encoding="utf-8") as f:
            f.write("{}")

def _load_tags() -> Dict[str, Dict[str, str]]:
    _ensure_tags_file()
    with open(TAGS_PATH, "r", encoding="utf-8") as f:
        try:
            obj = json.load(f)
        except Exception:
            obj = {}
    return obj if isinstance(obj, dict) else {}

def _save_tags(tags: Dict[str, Dict[str, str]]) -> None:
    _ensure_tags_file()
    tmp = TAGS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(tags, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, TAGS_PATH)

def tag_create(args: argparse.Namespace) -> int:
    tag = args.tag.strip()
    if not tag or " " in tag:
        print("[HARD STOP] tag must be non-empty and contain no spaces", file=sys.stderr)
        return 2
    tags = _load_tags()
    if tag in tags:
        print(f"[HARD STOP] tag already exists: {tag}", file=sys.stderr)
        return 2
    tags[tag] = {"description": args.description.strip()}
    _save_tags(tags)
    print(tag)
    return 0

def tag_list(args: argparse.Namespace) -> int:
    tags = _load_tags()
    if args.format == "json":
        print(json.dumps(tags, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    for k in sorted(tags.keys()):
        print(f"{k}: {tags[k].get('description','')}")
    return 0

def search_by_tag(args: argparse.Namespace) -> int:
    tag = args.tag.strip()
    tags = _load_tags()
    if tag not in tags:
        print(f"[HARD STOP] unknown tag: {tag}", file=sys.stderr)
        return 2

    entries: List[Dict[str, Any]] = []
    if args.graveyard_only:
        entries = _load_entries(GRAVEYARD_PATH)
    elif args.both:
        entries = _load_entries(ACTIVE_PATH) + _load_entries(GRAVEYARD_PATH)
    else:
        entries = _load_entries(ACTIVE_PATH)

    hits = [e for e in entries if tag in (e.get("tags") or [])]
    n = max(1, int(args.n))
    out = {"tag": tag, "entries": hits[-n:]}
    _emit(out, args.format)
    return 0

def tag_assign(args: argparse.Namespace) -> int:
    target = args.id.strip()
    if not ID_RE.match(target):
        print(f"[HARD STOP] Invalid id format: {target}", file=sys.stderr)
        return 2

    tags_registry = _load_tags()
    new_tags = [t.strip() for t in (args.tags or []) if t.strip()]
    for t in new_tags:
        if t not in tags_registry:
            print(f"[HARD STOP] unknown tag: {t}", file=sys.stderr)
            return 2

    active = _load_entries(ACTIVE_PATH)
    found = False
    for e in active:
        if e.get("id") == target:
            existing = e.get("tags") or []
            # stable union
            merged = sorted(set(existing) | set(new_tags))
            e["tags"] = merged
            found = True
            break
    if not found:
        print(f"[HARD STOP] Decision not found in active: {target}", file=sys.stderr)
        return 2

    try:
        _write_entries_atomic(ACTIVE_PATH, active)
    except Exception as ex:
        print(f"[HARD STOP] Failed to rewrite active decisions file: {ex}", file=sys.stderr)
        return 2

    print(target)
    return 0

def main() -> int:
    start = time.monotonic()
    ts = _now_iso_chicago()

    # Parse first so we know which subcommand it is.
    p = argparse.ArgumentParser(prog="decision_log")
    sub = p.add_subparsers(dest="cmd", required=True)

    # add
    addp = sub.add_parser("add", help="Append a decision entry to context/decisions_active.jsonl")
    addp.add_argument("--actor", required=True)
    addp.add_argument("--status", required=True, choices=["proposed", "approved"])
    addp.add_argument("--decision", required=True)
    addp.add_argument("--workflow", required=True, choices=["Small Changes", "Large Changes", "Iteration Loop", "meta"])
    addp.add_argument("--step", required=True)
    addp.add_argument("--reason", required=True)
    addp.add_argument("--refs", nargs="*", default=[])
    addp.add_argument("--options", nargs="*")
    addp.add_argument("--chosen")
    addp.add_argument("--supersedes", nargs="*")
    addp.set_defaults(func=add_entry)

    # latest
    lp = sub.add_parser("latest", help="Show the last N decisions from active (or graveyard)")
    lp.add_argument("--n", type=int, default=20)
    lp.add_argument("--format", choices=["json", "text"], default="json")
    lp.add_argument("--graveyard", action="store_true", help="Read from graveyard instead of active")
    lp.set_defaults(func=latest)

    # get
    gp = sub.add_parser("get", help="Get a decision by id (search active then graveyard by default)")
    gp.add_argument("--id", required=True)
    gp.add_argument("--format", choices=["json", "text"], default="json")
    gp.add_argument("--graveyard-only", action="store_true", help="Search graveyard only")
    gp.set_defaults(func=get_by_id)

    # search
    sp = sub.add_parser("search", help="Search decisions by keyword (default: active only)")
    sp.add_argument("--q", required=True)
    sp.add_argument("--n", type=int, default=20)
    sp.add_argument("--format", choices=["json", "text"], default="json")
    sp.add_argument("--both", action="store_true", help="Search active + graveyard")
    sp.add_argument("--graveyard-only", action="store_true", help="Search graveyard only")
    sp.set_defaults(func=search)

    # list proposed
    pp = sub.add_parser("list-proposed", help="List all proposed decisions in active")
    pp.add_argument("--format", choices=["json", "text"], default="json")
    pp.set_defaults(func=list_proposed)

    # delete
    dp = sub.add_parser("delete", help="Delete a decision from active and append it to graveyard")
    dp.add_argument("--id", required=True)
    dp.add_argument("--killed-by", required=True)
    dp.add_argument("--killed-reason", required=True)
    dp.add_argument("--replaced-by")
    dp.set_defaults(func=delete_decision)
    
    addp.add_argument("--tags", nargs="*", default=[])
    tc = sub.add_parser("tag-create", help="Create a new decision tag")
    tc.add_argument("--tag", required=True)
    tc.add_argument("--description", required=True)
    tc.add_argument("--format", choices=["json", "text"], default="text")
    tc.set_defaults(func=tag_create)

    tl = sub.add_parser("tag-list", help="List all tags")
    tl.add_argument("--format", choices=["json", "text"], default="text")
    tl.set_defaults(func=tag_list)

    st = sub.add_parser("search-tag", help="Search decisions by tag")
    st.add_argument("--tag", required=True)
    st.add_argument("--n", type=int, default=200)
    st.add_argument("--format", choices=["json", "text"], default="json")
    st.add_argument("--both", action="store_true")
    st.add_argument("--graveyard-only", action="store_true")
    st.set_defaults(func=search_by_tag)

    ta = sub.add_parser("tag-assign", help="Assign tag(s) to an existing active decision")
    ta.add_argument("--id", required=True)
    ta.add_argument("--tags", nargs="*", required=True)
    ta.set_defaults(func=tag_assign)


    args = p.parse_args()
    
    # Log the call (safe, concise subset)
    call_event: Dict[str, Any] = {
        "ts": ts,
        "event_type": "decision_log_call",
        "cmd": args.cmd,
        "argv": sys.argv[1:],  # full argv for forensics; remove if too noisy
        **_caller_hint(),
    }

    # Add some common fields when present (best-effort)
    for k in ["actor", "status", "workflow", "step", "id", "tag"]:
        if hasattr(args, k):
            call_event[k] = getattr(args, k)

    rc = 0
    err: Optional[str] = None
    try:
        rc = args.func(args)
        return rc
    except Exception as ex:
        rc = 2
        err = f"{type(ex).__name__}: {ex}"
        raise
    finally:
        end_ts = _now_iso_chicago()
        dur_ms = int((time.monotonic() - start) * 1000)

        call_event["end_ts"] = end_ts
        call_event["duration_ms"] = dur_ms
        call_event["rc"] = rc
        if err:
            call_event["error"] = err

        _append_observability(call_event)
        
    return args.func(args)

if __name__ == "__main__":
    raise SystemExit(main())
