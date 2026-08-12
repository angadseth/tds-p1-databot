"""Health + grading watchdog for the P1 Q5 data-analyst Telegram bot.

Run it any time to see, in one screen, whether the bot is alive and whether the
grader has been in touch:

    uv run --with requests python watch_bot.py

Exit code 0 = all clear, 1 = something needs attention (so it can drive a cron).
"""

import datetime
import json
import os
import sys

import requests

BASE = "https://tds-p1-databot-v2.onrender.com"
SERVICE = "srv-d9l2eg3m8hqs7394h8ig"
# Optional: set RENDER_API_KEY to also check whether the host suspended the
# service. Never hardcode it — this repository is public.
RENDER_KEY = os.environ.get("RENDER_API_KEY", "")
OWNER_CHAT = 6239214943  # Angad's own Telegram chat — anything else is an outsider
STATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".watch_state.json")

# An "answer" that is really the agent giving up.
BAD_WORDS = ("fail", "unable", "cannot", "could not", "couldn't", "no data",
             "not available", "unavailable", "sorry", "error", "unknown")

problems: list[str] = []
notes: list[str] = []


def check_health():
    try:
        r = requests.get(f"{BASE}/health", timeout=45)
        if r.status_code == 200:
            notes.append(f"health 200 - {r.json().get('model')}")
        else:
            problems.append(f"health returned {r.status_code}")
    except Exception as e:
        problems.append(f"health unreachable: {e}")


def check_service():
    if not RENDER_KEY:
        notes.append("render api check skipped (no RENDER_API_KEY set)")
        return
    try:
        s = requests.get(
            f"https://api.render.com/v1/services/{SERVICE}",
            headers={"Authorization": f"Bearer {RENDER_KEY}"}, timeout=45,
        ).json()
        if s.get("suspended") != "not_suspended":
            problems.append(f"RENDER SUSPENDED: {s.get('suspended')} {s.get('suspenders')}")
        else:
            notes.append("render not suspended")
    except Exception as e:
        notes.append(f"render api check skipped: {e}")


def check_log():
    try:
        txt = requests.get(f"{BASE}/run.jsonl", timeout=45).text
    except Exception as e:
        problems.append(f"run.jsonl unreachable: {e}")
        return
    events = [json.loads(l) for l in txt.splitlines() if l.strip()]
    notes.append(f"log has {len(events)} lines")

    seen = {}
    if os.path.exists(STATE):
        try:
            seen = json.load(open(STATE))
        except Exception:
            seen = {}
    last_n = seen.get("lines", 0)
    fresh = events[last_n:] if len(events) >= last_n else events

    # 1. Anyone other than Angad talking to the bot = the grader.
    outsiders = sorted({e["chat_id"] for e in events
                        if e.get("chat_id") and e["chat_id"] != OWNER_CHAT})
    for cid in outsiders:
        qs = [e for e in events if e.get("chat_id") == cid and e["event"] == "question"]
        ans = [e for e in events if e.get("chat_id") == cid and e["event"] == "answer"]
        problems.append(
            f"GRADER ACTIVITY chat_id={cid}: {len(qs)} question(s), {len(ans)} answer(s)"
        )
        for a in ans:
            body = json.loads(a["reply"]).get("answer")
            flat = json.dumps(body, ensure_ascii=False).lower()
            verdict = "LOOKS LIKE A FAILURE" if any(w in flat for w in BAD_WORDS) else "looks like a real answer"
            problems.append(f"    {a['ts']} -> {json.dumps(body, ensure_ascii=False)[:160]}  [{verdict}]")
        for q in qs:
            problems.append(f"    asked: {q['text'][:200]}")

    # 2. Agent-level trouble in the new lines.
    for e in fresh:
        if e["event"] in ("giveup_unrecovered", "agent_crash", "llm_error_final"):
            problems.append(f"{e['event']} at {e['ts']}: {str(e)[:200]}")
        if e["event"] == "giveup_retry":
            notes.append(f"giveup_retry at {e['ts']} (guard fired - it recovered)")

    # 3. Has the process restarted? (a restart wipes /tmp/run.jsonl)
    starts = [e for e in events if e["event"] == "startup"]
    if starts:
        notes.append(f"process up since {starts[-1]['ts']}")
    if events:
        last = max(datetime.datetime.fromisoformat(e["ts"]) for e in events if e.get("ts"))
        idle = (datetime.datetime.now(datetime.timezone.utc) - last).total_seconds() / 60
        notes.append(f"last event {idle:.0f} min ago")
        notes.append("SAFE TO REDEPLOY" if idle > 15 else "DO NOT REDEPLOY - a run may be in flight")

    json.dump({"lines": len(events)}, open(STATE, "w"))


check_health()
check_service()
check_log()

print(f"=== p1 databot watch - {datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M} UTC ===")
for n in notes:
    print(f"  ok   {n}")
if problems:
    print("\n!!! NEEDS ATTENTION !!!")
    for p in problems:
        print(f"  >>>  {p}")
else:
    print("\nAll clear - no grader activity, no agent failures.")
sys.exit(1 if problems else 0)
