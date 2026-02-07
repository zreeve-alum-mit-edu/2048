#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional

DECISIONS_PATH = os.path.join("context", "decisions.jsonl")
ID_RE = re.compile(r"^DEC-(\d{4,})$")


def _now_iso_chicago() -> str:
    tz = ZoneInfo("America/Chicago")
    return datetime.now(tz=tz).isoformat(timespec="seconds")


def _ensure_file() -> None:
    os.makedirs(os.path.dirname(DECISIONS_PATH), exist_ok=True)
    if not os.path.exists(DECISIONS_PATH):
        open(DECISIONS_PATH, "a", encoding="utf-8").close()


def _load_all_entries() -> List[Dict[str, Any]]:
    _ensure_file()
    entries: List[Dict[str, Any]] = []
    with open(DECISIONS_PATH, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    entries.append(obj)
            except Exception:
                # Hard-stop philosophy is for *decision operations*,
                # but we tolerate a bad historical line so reads still work.
                # (You can tighten this later if you want.)
                continue
    return entries


def _get_next_id() -> str:
    entries = _load_all_entries()
    last_id_num = 0
    for e in entries:
        _id = e.get("id")
        if isinstance(_id, str):
            m = ID_RE.match(_id)
            if m:
                last_id_num = max(last_id_num, int(m.group(1)))
    return f"DEC-{last_id_num + 1:04d}"


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
        print("")


def add_entry(args: argparse.Namespace) -> int:
    # HARD STOP if we can't write
    _ensure_file()

    entry_id = _get_next_id()
    ts = _now_iso_chicago()

    context: Dict[str, Any] = {
        "workflow": args.workflow,
        "step": args.step,
        "reason": args.reason,
        "refs": args.refs or [],
    }
    if args.options:
        context["options"] = args.options
    if args.chosen:
        context["chosen"] = args.chosen
    if args.supersedes:
        context["supersedes"] = args.supersedes

    entry: Dict[str, Any] = {
        "id": entry_id,
        "ts": ts,
        "actor": args.actor,
        "status": args.status,
        "decision": args.decision,
        "context": context,
    }

    try:
        with open(DECISIONS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[HARD STOP] Failed to append decision: {e}", file=sys.stderr)
        return 2

    # Print the id so callers can reference it immediately
    print(entry_id)
    return 0


def latest(args: argparse.Namespace) -> int:
    entries = _load_all_entries()
    n = max(1, int(args.n))
    out = {"entries": entries[-n:]}
    _emit(out, args.format)
    return 0


def get_by_id(args: argparse.Namespace) -> int:
    target = args.id.strip()
    if not ID_RE.match(target):
        print(f"[HARD STOP] Invalid id format: {target}", file=sys.stderr)
        return 2

    entries = _load_all_entries()
    for e in reversed(entries):
        if e.get("id") == target:
            _emit(e, args.format)
            return 0

    print(f"[HARD STOP] Decision not found: {target}", file=sys.stderr)
    return 2


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
    ]
    return " ".join(parts).lower()


def search(args: argparse.Namespace) -> int:
    q = (args.q or "").strip().lower()
    if not q:
        print("[HARD STOP] search requires --q", file=sys.stderr)
        return 2

    entries = _load_all_entries()
    hits = [e for e in entries if q in _haystack(e)]
    n = max(1, int(args.n))
    out = {"query": args.q, "entries": hits[-n:]}
    _emit(out, args.format)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="decision_log")
    sub = p.add_subparsers(dest="cmd", required=True)

    # add
    addp = sub.add_parser("add", help="Append a decision entry to context/decisions.jsonl")
    addp.add_argument("--actor", required=True)
    addp.add_argument("--status", required=True, choices=["proposed", "approved", "rejected"])
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
    lp = sub.add_parser("latest", help="Show the last N decisions (default JSON)")
    lp.add_argument("--n", type=int, default=20)
    lp.add_argument("--format", choices=["json", "text"], default="json")
    lp.set_defaults(func=latest)

    # get
    gp = sub.add_parser("get", help="Get a decision by id (default JSON)")
    gp.add_argument("--id", required=True)
    gp.add_argument("--format", choices=["json", "text"], default="json")
    gp.set_defaults(func=get_by_id)

    # search
    sp = sub.add_parser("search", help="Search decisions by keyword (default JSON)")
    sp.add_argument("--q", required=True)
    sp.add_argument("--n", type=int, default=20)
    sp.add_argument("--format", choices=["json", "text"], default="json")
    sp.set_defaults(func=search)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
