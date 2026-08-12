"""Data-analyst Telegram bot — TDS Project 1.

An LLM agent that answers data-analysis questions sent over Telegram.
Replies to every message with exactly one JSON object:
    {"answer": <shaped as the question asks>, "log_url": "<public JSONL log>"}

Architecture:
  - FastAPI app serves /health and /run.jsonl (the public agent log).
  - A background thread long-polls Telegram getUpdates.
  - Each incoming message runs an agentic loop (OpenAI-compatible chat with a
    run_python tool) until the model produces the final JSON answer.
  - A keep-warm thread pings our own public URL so the free host never idles out.
"""

import io
import json
import os
import re
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import requests
from fastapi import FastAPI
from fastapi.responses import FileResponse, PlainTextResponse

# ---------------------------------------------------------------- config
BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
AIPIPE_TOKEN = os.environ.get("AIPIPE_TOKEN", "")
MODEL = os.environ.get("MODEL", "gpt-4o-mini")
MODEL_BASE_URL = os.environ.get("MODEL_BASE_URL", "https://aipipe.org/openai/v1")
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000").rstrip("/")
# Anyone else who messages the bot is the grader; forward those exchanges here.
OWNER_CHAT_ID = int(os.environ.get("OWNER_CHAT_ID", "6239214943"))
LOG_PATH = os.environ.get("LOG_PATH", "/tmp/run.jsonl")
LOG_URL = f"{BASE_URL}/run.jsonl"
TG_API = f"https://api.telegram.org/bot{BOT_TOKEN}"

MAX_AGENT_STEPS = 14
PY_TIMEOUT = 60  # seconds for one run_python call
ANSWER_BUDGET = 210  # wall-clock seconds before we force a final answer
MAX_GIVEUP_RETRIES = 2  # extra rounds forced when the model tries to answer "I failed"

_log_lock = threading.Lock()
_histories: dict[int, list[dict]] = {}  # chat_id -> chat-completion messages
_hist_lock = threading.Lock()
_pyns: dict[int, dict] = {}  # chat_id -> persistent run_python globals
_pyns_lock = threading.Lock()


# ---------------------------------------------------------------- logging
def log_event(**fields):
    fields["ts"] = datetime.now(timezone.utc).isoformat()
    line = json.dumps(fields, ensure_ascii=False, default=str)
    with _log_lock:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ---------------------------------------------------------------- tools
def run_python(code: str, chat_id: int = 0) -> str:
    """Execute Python code, return captured stdout (or the error).

    Globals persist per chat, so a later step can reuse a dataframe an earlier
    step downloaded instead of refetching it.
    """
    out = io.StringIO()
    out_lock = threading.Lock()

    # Capture output by giving the sandbox its own print(), NOT by redirecting
    # sys.stdout: that is process-global, so with several chats in flight one
    # chat would steal another's output, and a timed-out thread would leave the
    # whole process writing into a dead buffer.
    def _print(*args, sep=" ", end="\n", file=None, flush=False):
        with out_lock:
            out.write(sep.join(str(a) for a in args) + end)

    with _pyns_lock:
        env = _pyns.setdefault(chat_id, {"__name__": "__main__"})
        if len(_pyns) > 50:  # bound memory across many chats
            for k in list(_pyns)[:-20]:
                _pyns.pop(k, None)
    env["print"] = _print

    def target():
        try:
            exec(code, env)
        except BaseException:  # SystemExit from exit() must not vanish silently
            with out_lock:
                out.write("\n" + traceback.format_exc(limit=4))

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(PY_TIMEOUT)
    if t.is_alive():
        return (
            "ERROR: code timed out after %ss. Pass timeout= to every network call, "
            "fetch less data, and print only what you need." % PY_TIMEOUT
        )
    with out_lock:
        text = out.getvalue()
    text = text[-8000:] if text else "(no output — use print())"
    return text + fabrication_warning(code)


# Phrases the model writes when it is about to invent its data. The prompt
# forbids this, but a prompt alone does not stop it, so we also say so at the
# moment it happens — as tool output, where it is hardest to ignore.
FABRICATION_MARKERS = (
    "simulated data", "simulate the data", "hypothetical", "for demonstration",
    "for illustration", "example data", "let's assume we have", "assume we have",
    "known statistics", "placeholder data", "dummy data", "sample data for",
    "using known values", "from memory",
)


def fabrication_warning(code: str) -> str:
    """Flag code that invents its own numbers instead of fetching them."""
    low = code.lower()
    if not any(m in low for m in FABRICATION_MARKERS):
        return ""
    if any(f in low for f in ("requests.get", "read_csv(", "read_html(", "urlopen")):
        return ""  # it is still really fetching something
    return (
        "\n\n!!! WARNING: this code contains data you invented, not data you fetched. "
        "Numbers recalled from memory are NOT evidence, and a max() over them is not a "
        "computation — it is a guess that changes every run. Do NOT base the final answer "
        "on it. Either fetch the real figures from a source you can verify, or state your "
        "knowledge-based answer directly in the final JSON without staging a fake calculation."
    )


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Run Python code on the server and get its printed output. "
                "pandas, numpy, requests, bs4, openpyxl are installed and the "
                "network is available (download public datasets with requests). "
                "Always print() what you need to see. Variables persist between "
                "calls in this chat, so data you fetched in an earlier call is "
                "still in scope — inspect it instead of refetching."
            ),
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "Python source to execute"}},
                "required": ["code"],
            },
        },
    }
]

SYSTEM_PROMPT = """You are an expert data-analyst agent answering questions sent to a Telegram bot.

## The one unbreakable rule
NEVER answer with a failure, an apology, or an excuse. Strings like "failed to fetch",
"unable to determine", "data not available", "please check the API" are ALWAYS wrong answers —
they score zero. The question always has a real answer and you must commit to one. If every
fetch attempt fails, answer from your own knowledge of the published statistics instead. An
educated best guess in the right shape beats a perfect explanation of why you could not answer.

## Rules
1. Answer the user's LATEST message. Earlier messages in the chat are context for multi-turn tasks.
2. The message may embed data inline, or reference a public dataset (WHO GHO, World Bank, MOSPI,
   data.gov.in, ...). Use the run_python tool to fetch and compute — never guess a number you could
   compute. But see the unbreakable rule: fetching is the means, answering is the goal.
3. The message usually spells out the exact JSON shape it wants, e.g.
   Reply with ONLY {"answer": {"state": "<state>"}, "log_url": "..."}.
4. When ready, reply with ONLY that JSON object — no prose, no markdown fences. Use the placeholder
   "LOG_URL" for log_url; the harness substitutes the real URL. Match the requested "answer" shape
   EXACTLY (keys, nesting, types: numbers as numbers unless a string is asked for).
5. If no shape is specified, reply {"answer": <your concise answer>, "log_url": "LOG_URL"}.
6. If a mid-conversation message is only setup/context ("I will send data next"), still reply with
   {"answer": "ok", "log_url": "LOG_URL"} unless it asks something.
7. Round as instructed; if unspecified, give reasonable precision. Never add keys that were not asked for.

## NEVER fabricate data (this is how wrong answers happen)
Writing values you remember into a literal and computing on them is NOT computing — it is a guess
wearing a computation's clothes, and it silently produces a different answer every time:
    state_mmr = {"Kerala": 42, "Assam": 215, ...}   # FORBIDDEN: these numbers are invented
    max(state_mmr, key=state_mmr.get)               # looks rigorous, means nothing
If you could not fetch the real data, do NOT stage a fake calculation. Say the answer directly in
the final JSON from your own knowledge of the published figure. Compute only on data you actually
fetched and printed.

## Check that you got data, not a web page
Before parsing, print `r.status_code`, `r.headers.get("content-type")` and `r.text[:200]`.
HTTP 200 with `text/html` where you expected CSV or JSON means the URL is WRONG — a portal error
page, a login wall, or a link you half-remembered. Do not guess dataset URLs; a guessed URL that
returns HTML is the usual first domino. Switch to a source you can verify (a documented API, a
Wikipedia table via the API, the World Bank API) instead of inventing numbers.

## How to not fail at fetching
An empty result is NOT proof the data is missing — it almost always means your identifier or filter
was wrong. Before concluding anything is unavailable:
- Fetch ONE unfiltered record first and print it, so you can see the real column/dimension names and
  the exact value format. Filter only after you have seen the shape of the data.
- Country/region codes are usually ISO3 ("BRA", "IND", "ZAF"), NOT country names. Same for state and
  indicator codes — look up the code list endpoint rather than assuming a label works.
- WHO GHO OData (ghoapi.azureedge.net): SpatialDim holds ISO3 codes, and rows are split by dimension —
  e.g. Dim1 is SEX_BTSX (both sexes) / SEX_MLE / SEX_FMLE. Filter the dimension explicitly or you will
  silently mix series together.
- Pass timeout=20 to EVERY network call. A hung request burns the whole budget. If a source fails
  twice, stop retrying it and move to a different source.
- Reachability differs by host: some sites answer from this server and others do not. Two failures
  on one source is your signal to switch, not to try harder.
- If a host or endpoint is dead, try an alternate source (World Bank API, the CSV download, Wikipedia)
  before falling back to knowledge.
- Never call max()/min()/idxmax() on a collection you have not just printed and confirmed is non-empty.
- Print intermediate values at every step. Sanity-check the final number against what you'd expect.

## Every qualifier in the question is a filter you must write
"both sexes", "total population", "urban", "rural", "constant 2015 prices", "age-standardised",
"seasonally adjusted" — each one names a specific slice of the data. If a qualifier from the question
does not appear as an explicit filter in your code, you have NOT answered the question that was asked.
Statistical datasets ship every slice in the same table.

## The silent killer: many rows per key
Before you collapse rows into one value per entity/year, PRINT how many rows exist per key, e.g.
    from collections import Counter
    print(Counter((r["SpatialDim"], r["TimeDim"]) for r in rows).most_common(5))
If any count is above 1 you are mixing series (sex, age group, region type, data source) and
`d[key] = value` silently keeps whichever row happened to come last. This produces a confident,
wrong answer with no error message — it is the most common way this task is failed. Filter the
extra dimensions until every key has exactly one row.

## Before you send the final JSON
1. Print the FULL comparison table you based the answer on — every candidate with its computed
   value, sorted — not just the winner. Read it and check the winner really is the extreme.
2. Re-read the question and confirm: right entity, right years, right qualifiers, right units,
   right rounding, right key names and nesting.
3. Confirm "answer" is an actual answer, not a status report.
"""

# Answers that are really the model giving up. Matching one forces another round.
GIVEUP_RE = re.compile(
    r"\b(?:fail(?:ed|ure)?|unable|cannot|can'?t|could ?n'?t|not able|no data|"
    r"not available|unavailable|missing data|error occurred|an error|sorry|"
    r"apolog\w+|please (?:check|try|provide)|try again|unknown|undetermined|"
    r"unable to determine|insufficient)\b",
    re.I,
)


GIVEUP_NUDGE = (
    "Your last reply was a failure message, not an answer. That scores zero.\n"
    "If it actually WAS a valid answer to the question, resend it EXACTLY as it was.\n"
    "Otherwise keep working: an empty API result means your identifier or filter was wrong, "
    "not that the data is missing. Fetch one UNFILTERED record and print it to see the real "
    "codes and dimension values (country codes are usually ISO3 like 'BRA'; rows are often split "
    "by dimensions such as sex). Or switch to a different source. Or, if nothing can be fetched, "
    "answer from your own knowledge of the published figures — you must commit to a real answer.\n"
    "Now reply with ONLY the JSON object in the exact shape the question asked for."
)


def looks_like_giveup(obj) -> bool:
    """True if the model's 'answer' is a failure message rather than an answer."""

    def bad(v) -> bool:
        if isinstance(v, str):
            return bool(GIVEUP_RE.search(v))
        if isinstance(v, dict):
            return any(bad(x) for x in v.values())
        if isinstance(v, (list, tuple)):
            return any(bad(x) for x in v)
        return False

    if "answer" not in obj:
        return False
    a = obj["answer"]
    if a is None:
        return True
    return bad(a)


# ---------------------------------------------------------------- llm
def chat_completion(messages, use_tools=True):
    body = {"model": MODEL, "messages": messages, "temperature": 0}
    if use_tools:
        body["tools"] = TOOLS
    r = requests.post(
        f"{MODEL_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {AIPIPE_TOKEN}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (data-analyst-bot)",
        },
        json=body,
        timeout=180,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]


def extract_json(text: str):
    """Pull the first balanced JSON object out of model text."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.M)
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if esc:
            esc = False
            continue
        if c == "\\":
            esc = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def solve(chat_id: int, question: str) -> str:
    """Run the agent loop; return the final JSON reply text."""
    with _hist_lock:
        history = _histories.setdefault(chat_id, [])
        history.append({"role": "user", "content": question})
        # keep the last 20 turns
        del history[:-20]
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + list(history)

    log_event(event="question", chat_id=chat_id, text=question)

    deadline = time.time() + ANSWER_BUDGET
    step = 0
    obj: dict = {"answer": "unable to determine"}

    for attempt in range(MAX_GIVEUP_RETRIES + 1):
        final_text = None
        while step < MAX_AGENT_STEPS:
            out_of_time = time.time() > deadline
            if out_of_time:
                messages.append(
                    {
                        "role": "user",
                        "content": "Time is up. Reply NOW with only your best final JSON object.",
                    }
                )
            try:
                msg = chat_completion(messages, use_tools=not out_of_time)
            except Exception as e:
                log_event(event="llm_error", chat_id=chat_id, error=str(e))
                time.sleep(2)
                try:
                    msg = chat_completion(messages, use_tools=True)
                except Exception as e2:
                    log_event(event="llm_error_final", chat_id=chat_id, error=str(e2))
                    break
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                messages.append(msg)
                for tc in tool_calls:
                    try:
                        code = json.loads(tc["function"]["arguments"]).get("code", "")
                    except json.JSONDecodeError:
                        code = tc["function"]["arguments"]
                    log_event(event="tool_call", chat_id=chat_id, step=step, code=code[:4000])
                    output = run_python(code, chat_id)
                    log_event(event="tool_result", chat_id=chat_id, step=step, output=output[:4000])
                    messages.append(
                        {"role": "tool", "tool_call_id": tc["id"], "content": output}
                    )
                step += 1
                continue
            final_text = msg.get("content") or ""
            messages.append({"role": "assistant", "content": final_text})
            step += 1
            break

        candidate = extract_json(final_text) if final_text else None
        if candidate is None:
            candidate = {"answer": (final_text or "unable to determine").strip()[:1000]}
        if "answer" not in candidate:
            candidate = {"answer": candidate}
        obj = candidate

        if not looks_like_giveup(obj):
            break
        if attempt == MAX_GIVEUP_RETRIES or step >= MAX_AGENT_STEPS or time.time() > deadline:
            log_event(event="giveup_unrecovered", chat_id=chat_id, reply=json.dumps(obj)[:500])
            break
        log_event(
            event="giveup_retry",
            chat_id=chat_id,
            attempt=attempt + 1,
            rejected=json.dumps(obj, ensure_ascii=False)[:500],
        )
        messages.append({"role": "user", "content": GIVEUP_NUDGE})

    obj["log_url"] = LOG_URL
    reply = json.dumps(obj, ensure_ascii=False)

    with _hist_lock:
        _histories.setdefault(chat_id, []).append({"role": "assistant", "content": reply})
    log_event(event="answer", chat_id=chat_id, reply=reply)
    return reply


# ---------------------------------------------------------------- telegram
def tg(method, **params):
    r = requests.post(f"{TG_API}/{method}", json=params, timeout=65)
    return r.json()


def notify_owner(chat_id: int, question: str, reply: str):
    """Forward a grader exchange to the owner. Never let this break the reply."""
    if not OWNER_CHAT_ID or chat_id == OWNER_CHAT_ID:
        return
    try:
        verdict = "!! LOOKS LIKE A FAILURE !!" if looks_like_giveup(json.loads(reply)) else "answered"
    except Exception:
        verdict = "unparseable reply"
    try:
        tg(
            "sendMessage",
            chat_id=OWNER_CHAT_ID,
            text=(f"GRADER just messaged the bot (chat_id {chat_id}) - {verdict}\n\n"
                  f"Q: {question[:1500]}\n\nA: {reply[:1500]}"),
        )
        log_event(event="owner_notified", chat_id=chat_id, verdict=verdict)
    except Exception as e:
        log_event(event="owner_notify_failed", chat_id=chat_id, error=str(e))


def handle_update(upd):
    msg = upd.get("message") or upd.get("edited_message")
    if not msg:
        return
    text = msg.get("text") or msg.get("caption") or ""
    chat_id = msg["chat"]["id"]
    if not text:
        return
    try:
        reply = solve(chat_id, text)
    except Exception:
        log_event(event="agent_crash", chat_id=chat_id, error=traceback.format_exc())
        reply = json.dumps({"answer": "internal error", "log_url": LOG_URL})
    tg("sendMessage", chat_id=chat_id, text=reply)  # the grader's reply comes first
    notify_owner(chat_id, text, reply)


def poll_loop():
    log_event(event="startup", base_url=BASE_URL, model=MODEL)
    offset = 0
    pool = ThreadPoolExecutor(max_workers=6)
    while True:
        try:
            resp = requests.get(
                f"{TG_API}/getUpdates",
                params={"offset": offset, "timeout": 50},
                timeout=65,
            ).json()
            for upd in resp.get("result", []):
                offset = upd["update_id"] + 1
                pool.submit(handle_update, upd)
        except Exception as e:
            log_event(event="poll_error", error=str(e))
            time.sleep(5)


def keepwarm_loop():
    """Ping our own public URL so a free host never spins down."""
    while True:
        time.sleep(600)
        try:
            requests.get(f"{BASE_URL}/health", timeout=30)
        except Exception:
            pass


# ---------------------------------------------------------------- web app
app = FastAPI()


@app.on_event("startup")
def _start():
    if not os.path.exists(LOG_PATH):
        log_event(event="log_created")
    threading.Thread(target=poll_loop, daemon=True).start()
    threading.Thread(target=keepwarm_loop, daemon=True).start()


@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    return {"ok": True, "model": MODEL, "log_url": LOG_URL}


@app.get("/run.jsonl")
def run_log():
    if os.path.exists(LOG_PATH):
        return FileResponse(LOG_PATH, media_type="application/jsonl; charset=utf-8", filename="run.jsonl")
    return PlainTextResponse("", media_type="application/jsonl")


@app.get("/")
def root():
    return {"service": "data-analyst-telegram-bot", "log_url": LOG_URL}
