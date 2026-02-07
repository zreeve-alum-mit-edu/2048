#!/usr/bin/env python3
"""
Best-effort pending rule application logger.

This script MUST NOT block progress:
- If it fails to write, it prints a warning to stderr and exits non-zero.
- Callers should treat failure as a WARN-only condition.

Usage:
  python3 scripts/rule_app_log.py \
    --actor "design-doc-author" \
    --rule_id "RULE-0001" \
    --rule_summary "Docs are binding; extract approved decisions from new MUST/SHALL/NEVER constraints" \
    --applied_to "file:docs/design/foo.md" "DEC:DEC-0123" \
    --notes "Logged decisions: DEC-0123, DEC-0124"
"""

import argparse
import json
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

LOG_PATH = os.path.join("context", "RULE_APPLICATION_LOG.jsonl")


def now_iso_chicago() -> str:
    return datetime.now(ZoneInfo("America/Chicago")).isoformat(timespec="seconds")


def ensure_file() -> None:
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    if not os.path.exists(LOG_PATH):
        open(LOG_PATH, "a", encoding="utf-8").close()


def main() -> int:
    p = argparse.ArgumentParser(prog="rule_app_log")
    p.add_argument("--actor", required=True)
    p.add_argument("--rule_id", required=True)
    p.add_argument("--rule_summary", required=True)
    p.add_argument("--applied_to", nargs="*", default=[])
    p.add_argument("--notes", default="")

    args = p.parse_args()

    entry = {
        "ts": now_iso_chicago(),
        "actor": args.actor,
        "rule_id": args.rule_id,
        "rule_summary": args.rule_summary,
        "applied_to": args.applied_to,
        "notes": args.notes,
    }

    try:
        ensure_file()
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return 0
    except Exception as e:
        print(f"[WARN] Failed to append {LOG_PATH}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
