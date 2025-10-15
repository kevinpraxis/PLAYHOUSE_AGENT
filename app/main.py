import os, json, time, re, random, csv  
import requests
from requests_oauthlib import OAuth1
from datetime import datetime, timezone
from dateutil.parser import isoparse
from pathlib import Path  # ✅
import google.generativeai as genai

# ---------- Config ----------
QUERIES = [
    '("recommend" OR "any suggestions" OR "which app" OR "what app") ("AI chat" OR "AI twin" OR "AI companion") lang:en -is:retweet',
    '("tried" OR "using" OR "experience with" OR "review of") ("AI twin" OR "AI bestie" OR "AI app") lang:en -is:retweet',
    '(college OR campus OR dorm OR freshman OR student) ("AI app" OR "AI twin" OR "AI companion") lang:en -is:retweet',
    '("make life easier" OR "help me with" OR "AI for" OR "AI assistant") lang:en -is:retweet',
    '("AI friend" OR "virtual friend" OR "AI bestie" OR "AI girlfriend" OR "AI boyfriend") lang:en -is:retweet',
    '("new app" OR "found this app" OR "best AI app" OR "AI social app" OR "AI texting app") lang:en -is:retweet'
]

MAX_PER_QUERY = 25
FRESH_HOURS   = 12
MIN_SCORE     = 2
SEEN_FILE     = "seen_ids.json"
POST_OR_REPLY = "reply"

GEMINI_CANDIDATES = [
    os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash"),
    "models/gemini-2.5-pro",
    "models/gemini-2.0-flash",
    "models/gemini-2.5-flash-lite",
]

GEMINI_CANDIDATES = list(dict.fromkeys(GEMINI_CANDIDATES))


# CSV log
CSV_LOG = "data/run_log.csv"
CSV_COLUMNS = [
    "ts_iso","action","endpoint","http_status","limit","remaining","reset_epoch","reset_iso",
    "query","page_size","tweet_id","tweet_created_at","conversation_id","score",
    "selected","drafted","posted","dry_run","reply_to","post_id",
    "text","draft_text"
]

def _ensure_csv():
    Path("data").mkdir(parents=True, exist_ok=True)
    if not os.path.exists(CSV_LOG):
        with open(CSV_LOG, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(CSV_COLUMNS)

def _str(x):
    if x is None:
        return ""
    s = str(x)
    return s.replace("\r", " ").replace("\n", " ").strip()

def log_event(**kwargs):

    _ensure_csv()
    row = {k: "" for k in CSV_COLUMNS}
    row.update({k: kwargs.get(k, "") for k in CSV_COLUMNS})
    row["ts_iso"] = datetime.now(timezone.utc).isoformat()
    # Unified stringification
    for k in CSV_COLUMNS:
        row[k] = _str(row[k])
    with open(CSV_LOG, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([row[c] for c in CSV_COLUMNS])

# PURPOSE: Utility helpers for dedup and timestamp handling.
def load_seen(path=SEEN_FILE):
    if os.path.exists(path):
        with open(path, "r") as f:
            return set(json.load(f))
    return set()

def save_seen(seen, path=SEEN_FILE):
    with open(path, "w") as f:
        json.dump(list(seen), f)

SEEN = load_seen()

def hours_ago(iso_ts: str) -> float:
    try:
        dt_obj = isoparse(iso_ts)
        if dt_obj.tzinfo is None:
            dt_obj = dt_obj.replace(tzinfo=timezone.utc)
    except Exception:
        return float("inf")
    return (datetime.now(timezone.utc) - dt_obj).total_seconds() / 3600.0

def _is_retweet_text(t):
    txt = (t.get("text") or "").lstrip()
    return txt.startswith("RT @")

# rate headers -> dict
def _rate_info_from_response(r, endpoint, query="", page_size=None):
    rem  = r.headers.get("x-rate-limit-remaining")
    rs   = r.headers.get("x-rate-limit-reset")
    lim  = r.headers.get("x-rate-limit-limit")
    try:
        reset_epoch = int(rs) if rs is not None else ""
    except Exception:
        reset_epoch = ""
    reset_iso = ""
    if isinstance(reset_epoch, int):
        try:
            reset_iso = datetime.fromtimestamp(reset_epoch, timezone.utc).isoformat()
        except Exception:
            reset_iso = ""
    info = {
        "action": "search" if "search" in endpoint else "post",
        "endpoint": endpoint,
        "http_status": r.status_code,
        "limit": lim or "",
        "remaining": rem or "",
        "reset_epoch": reset_epoch,
        "reset_iso": reset_iso,
        "query": query or "",
        "page_size": page_size or "",
        "dry_run": "",  
    }
    return info

def recent_search(query: str, max_results: int = 10, run_id: str | None = None):
    url_a = "https://api.x.com/2/tweets/search/recent"
    url_b = "https://api.twitter.com/2/tweets/search/recent"  # fallback

    # limit [10, 100]
    page_size = max(10, min(int(max_results or 10), 100))
    params = {
        "query": query,
        "max_results": page_size,
        "tweet.fields": "created_at,public_metrics,conversation_id",
    }

    def _get(url):
        return requests.get(url, headers=HEADERS_READ, params=params, timeout=20)

    r = _get(url_a)
    if r.status_code == 404:
        r = _get(url_b)

    # record first search info
    rate_info = _rate_info_from_response(r, endpoint="/2/tweets/search/recent", query=query, page_size=page_size)
    log_event(**rate_info)

    rem = r.headers.get("x-rate-limit-remaining")
    rs  = r.headers.get("x-rate-limit-reset")
    now_epoch = int(time.time())
    try:
        reset_epoch = int(rs) if rs is not None else None
    except Exception:
        reset_epoch = None

    if r.status_code == 429:
        if reset_epoch and reset_epoch <= now_epoch:   
            return [], False, rate_info
        return [], True, rate_info

    if r.status_code == 400:
        if rem == "0" and reset_epoch and reset_epoch > now_epoch:
            return [], True, rate_info
        return [], False, rate_info

    if 401 <= r.status_code < 500 or r.status_code >= 500:
        return [], False, rate_info

    # parsing data
    try:
        data = r.json().get("data", []) or []
    except Exception:
        data = []

    # store data
    if data:
        Path("data").mkdir(parents=True, exist_ok=True)
        if not run_id:
            # If run_id is not passed, the timestamp is still used to avoid overwriting
            run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        dump_path = f"data/search_raw_{run_id}.ndjson"
        # Use append mode to facilitate aggregating multiple requests into the same file in one run
        with open(dump_path, "a", encoding="utf-8") as f:
            fetched_at = datetime.now(timezone.utc).isoformat()
            for row in data:
                row_out = dict(row)
                row_out["_query"] = query
                row_out["_fetched_at"] = fetched_at
                f.write(json.dumps(row_out, ensure_ascii=False) + "\n")
        log_event(
            action="search_dump",
            endpoint="/2/tweets/search/recent",
            text=dump_path,
            query=query,
            page_size=len(data)
        )

    return data, False, rate_info

# PURPOSE: A minimal, explainable scoring function to filter candidates before LLM drafting.
def score_tweet(t: dict) -> int:
    s = 0
    text = t.get("text", "")

    if "?" in text or re.search(r"\b(recommend|suggestion|which app)\b", text, re.I):
        s += 1
    if t.get("created_at") and hours_ago(t["created_at"]) <= FRESH_HOURS:
        s += 1
    pm = t.get("public_metrics", {})
    if (pm.get("reply_count", 0) + pm.get("like_count", 0)) >= 1:
        s += 1
    return s

def text_trim(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + "…"

def hard_limit(s: str, n: int) -> str:
    return s[:n]

def gen_gemini_reply(original_text: str) -> str:
    """
    Generate a concise (<=260 chars) reply suitable for X/Twitter.

    Behavior:
    - Tries Gemini models in the order defined by GEMINI_CANDIDATES
      (the first entry can be overridden via env var GEMINI_MODEL).
    - If GOOGLE_API_KEY is not present in the environment, attempts to load it
      from the project root `.env` so this function works in offline tests too.
    - Never raises on model errors: if all candidates fail, returns a small,
      rule-based fallback string so the pipeline doesn't break.

    Requirements:
    - `google-generativeai` is available and imported as `genai` at module top.
    - `text_trim` and `hard_limit` helpers exist in this module.
    - Optional: `log_event` exists (if not, the call is wrapped in try/except).
    """
    import os
    from pathlib import Path

    # 1) Ensure the API key is configured (supports offline calls without init_clients()).
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        # Try loading from repo root `.env` (…/app/main.py -> parents[1] is the repo root).
        try:
            from dotenv import load_dotenv
            load_dotenv(Path(__file__).resolve().parents[1] / ".env")
            key = os.getenv("GOOGLE_API_KEY")
        except Exception:
            pass
    assert key, "Missing GOOGLE_API_KEY (set it in .env or environment variables)"
    try:
        genai.configure(api_key=key)  # idempotent; re-configuring is harmless
    except Exception:
        pass

    # 2) Build a minimal, focused prompt.
    system_msg = (
        "You are a helpful, concise peer on X. "
        "Constraints: <=260 characters; no mass hashtags; no aggressive promotion; "
        "1 short empathy line + 1–2 concrete tips tailored to the post; natural tone."
    )
    prompt = f"""Original post:
---
{text_trim(original_text or "", 800)}
---

Write a single reply (<=260 chars). Avoid templated or spammy phrasing. Provide practical tips."""

    # 3) Try candidate models in order; return on first success.
    last_err = None
    for model_name in GEMINI_CANDIDATES:
        try:
            out = genai.GenerativeModel(model_name).generate_content(
                [system_msg, prompt],
                request_options={"timeout": 30}
            )
            text = (getattr(out, "text", "") or "").strip()
            if text:
                # Optional: record which model produced the output (ignore if logger absent).
                try:
                    log_event(action="gemini_model", endpoint="gemini", text=model_name)
                except Exception:
                    pass
                return hard_limit(text, 260)
        except Exception as e:
            last_err = e
            continue

    # 4) All candidates failed → return a deterministic, safe fallback.
    print("[gemini] all candidates failed; fallback:", last_err)
    base = (original_text or "").replace("\n", " ")[:140]
    fallback = (
        "Got you. Try 1–2 apps for a week, note what truly helps, and mute the noisy stuff. "
        "If you share your goal, I can suggest a tiny checklist. "
    )
    return hard_limit(fallback + base, 260)

def human_review(draft: str, original_text: str) -> str | None:
    print("\n--- Original ---")
    print(text_trim(original_text, 500))
    print("\n--- Draft ---")
    print(draft)
    print("\nActions: [y] post, [e] edit then post, [s] skip")

    while True:
        act = input("> ").strip().lower()
        if act == "y":
            return draft
        elif act == "e":
            print("Enter your edited reply (single line):")
            edited = input("> ").strip()
            return hard_limit(edited, 260)
        elif act == "s":
            return None

def post_tweet(text: str):
    url = "https://api.x.com/2/tweets"
    payload = {"text": text}
    r = requests.post(url, auth=auth_oauth1, json=payload)
    if r.status_code == 404:
        url = "https://api.twitter.com/2/tweets"
        r = requests.post(url, auth=auth_oauth1, json=payload)
    # Bucket for logging posts/responses
    rate_info = _rate_info_from_response(r, endpoint="/2/tweets")
    log_event(**rate_info, drafted=True, posted=True, action="post")
    return r

def reply_to(tweet_id: str, text: str):
    url = "https://api.x.com/2/tweets"
    payload = {"text": text, "reply": {"in_reply_to_tweet_id": tweet_id}}
    r = requests.post(url, auth=auth_oauth1, json=payload)
    if r.status_code == 404:
        url = "https://api.twitter.com/2/tweets"
        r = requests.post(url, auth=auth_oauth1, json=payload)
    # Bucket/response for recording replies
    rate_info = _rate_info_from_response(r, endpoint="/2/tweets")
    log_event(**rate_info, drafted=True, posted=True, action="reply", reply_to=tweet_id)
    return r

def init_clients():
    bearer = os.getenv("X_BEARER_TOKEN")
    api_key = os.getenv("X_API_KEY")
    api_secret = os.getenv("X_API_SECRET")
    access_token = os.getenv("X_ACCESS_TOKEN")
    access_secret = os.getenv("X_ACCESS_SECRET")
    google_key = os.getenv("GOOGLE_API_KEY")

    assert bearer, "Missing X_BEARER_TOKEN"
    assert api_key and api_secret and access_token and access_secret, "Missing OAuth1 creds"
    assert google_key, "Missing GOOGLE_API_KEY"

    genai.configure(api_key=google_key)
    auth = OAuth1(api_key, api_secret, access_token, access_secret)
    headers_read = {"Authorization": f"Bearer {bearer}"}
    return headers_read, auth

def run_once(query: str, target: int, dry_run: bool = True, interactive: bool = True):
    global SEEN
    headers, auth = init_clients()

    global HEADERS_READ, auth_oauth1
    HEADERS_READ, auth_oauth1 = headers, auth

    #  generate run_id for every move
    RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    Path("data").mkdir(parents=True, exist_ok=True)
    prepared_path = f"data/replies_prepared_{RUN_ID}.ndjson"
    sent_path     = f"data/replies_sent_{RUN_ID}.ndjson"

    log_event(action="run_start", endpoint="local", text=f"run_id={RUN_ID}", dry_run=dry_run)

    queries = [query] if query else QUERIES
    candidates = []
    fetched = 0
    rate_exhausted = False

    for q in queries:
        if fetched >= target or rate_exhausted:
            break

        page_size = min(MAX_PER_QUERY, max(10, max(1, target - fetched)))

        try:
            #  pass run_id to recent_search, Aggregate the original data into the same file
            tweets, exhausted, rate_info = recent_search(q, page_size, run_id=RUN_ID)
            rate_exhausted = rate_exhausted or exhausted
        except Exception as e:
            print("Search error:", e)
            log_event(action="error", endpoint="/2/tweets/search/recent", http_status="EXC",
                      text=str(e), query=q, page_size=page_size)
            continue

        # Also log the dry_run flag for this request
        log_event(action="search_meta", endpoint="/2/tweets/search/recent",
                  http_status=rate_info.get("http_status",""),
                  limit=rate_info.get("limit",""),
                  remaining=rate_info.get("remaining",""),
                  reset_epoch=rate_info.get("reset_epoch",""),
                  reset_iso=rate_info.get("reset_iso",""),
                  query=q, page_size=page_size, dry_run=dry_run)

        for t in tweets:
            if fetched >= target:
                break
            if _is_retweet_text(t):
                continue
            tid = t["id"]
            if tid in SEEN:
                continue

            sc = score_tweet(t)
            log_event(action="found", endpoint="/2/tweets/search/recent",
                      tweet_id=tid, tweet_created_at=t.get("created_at",""),
                      conversation_id=t.get("conversation_id",""),
                      score=sc, query=q, page_size=page_size, text=t.get("text",""))

            if sc >= MIN_SCORE:
                candidates.append((sc, t))
            fetched += 1

        # The Free mode can be broken here to limit a run to a single request
        # break

    # 4) sort and cut
    candidates.sort(key=lambda x: (-x[0], x[1].get("created_at", "")))
    candidates = candidates[:target]

    # Whether it is a dry run or a formal run, first "prepare and put the gear prepared"
    queued = 0
    with open(prepared_path, "a", encoding="utf-8") as pf:
        for _, t in candidates:
            try:
                draft = gen_gemini_reply(t.get("text", ""))
                drafted_flag = True
            except Exception as e:
                draft = ""
                drafted_flag = False
                log_event(action="draft_error", endpoint="local",
                          tweet_id=t["id"], text=str(e))

            pf.write(json.dumps(
                {
                    "run_id": RUN_ID,
                    "tweet_id": t["id"],
                    "tweet_created_at": t.get("created_at",""),
                    "conversation_id": t.get("conversation_id",""),
                    "score": score_tweet(t),
                    "original_text": t.get("text",""),
                    "draft_text": draft,
                    "dry_run": dry_run,
                },
                ensure_ascii=False
            ) + "\n")

            SEEN.add(t["id"])
            queued += 1

            log_event(action="queue", endpoint="local",
                      tweet_id=t["id"],
                      tweet_created_at=t.get("created_at",""),
                      conversation_id=t.get("conversation_id",""),
                      score=score_tweet(t),
                      drafted=drafted_flag, draft_text=draft, selected=True,
                      dry_run=dry_run, text=t.get("text",""))

    if dry_run:
        # Compatible with existing prompts
        print(f"[dry-run] queued {queued} replies -> {prepared_path}")
        if rate_exhausted:
            print("[rate] stopped early: current window exhausted.")
        save_seen(SEEN)
        return

    # 6) Formal sending: Success/failure of sending is recorded in the sent file
    for sc, t in candidates:
        tid = t["id"]
        original = t.get("text", "")

        # The draft has been written from the prepared; it is generated again here to prevent manual switching (it can also be reused)
        try:
            draft = gen_gemini_reply(original)
        except Exception as e:
            draft = ""
            log_event(action="draft_error", endpoint="local",
                      tweet_id=tid, text=str(e))

        to_send = draft
        if interactive:
            approved = human_review(draft, original)
            if not approved:
                print("Skipped.")
                SEEN.add(tid); save_seen(SEEN)
                log_event(action="skip", endpoint="local",
                          tweet_id=tid, drafted=True, draft_text=draft,
                          selected=True, dry_run=False, text=original)
                continue
            to_send = approved

        if POST_OR_REPLY == "post":
            resp = post_tweet(to_send)
        else:
            resp = reply_to(tid, to_send)

        try:
            post_id = resp.json().get("data", {}).get("id", "")
        except Exception:
            post_id = ""

        # ✅ Landing sent mapping
        with open(sent_path, "a", encoding="utf-8") as sf:
            sf.write(json.dumps(
                {
                    "run_id": RUN_ID,
                    "tweet_id": tid,
                    "reply_text": to_send,
                    "post_id": post_id,
                    "status_code": getattr(resp, "status_code", None),
                    "response_snippet": (resp.text or "")[:400],
                },
                ensure_ascii=False
            ) + "\n")

        log_event(action="reply_done" if POST_OR_REPLY=="reply" else "post_done",
                  endpoint="/2/tweets",
                  http_status=getattr(resp, "status_code", ""),
                  posted=True, reply_to=(tid if POST_OR_REPLY=="reply" else ""),
                  post_id=post_id, draft_text=to_send, text=original, dry_run=False)

        print("API:", resp.status_code, (resp.text or "")[:200])
        SEEN.add(tid); save_seen(SEEN)
        time.sleep(random.uniform(8, 20))

    print(f"[sent] wrote mapping -> {sent_path}")
