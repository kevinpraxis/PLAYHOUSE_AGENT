
import os, json, time, re
import requests
from requests_oauthlib import OAuth1
from datetime import datetime, timezone
from dateutil.parser import isoparse
import google.generativeai as genai

# ---------- Config ----------
QUERIES = [
    # --- Advice seeking / recommendations ---
    '("recommend" OR "any suggestions" OR "which app" OR "what app") ("AI chat" OR "AI twin" OR "AI companion") lang:en -is:retweet',

    # --- Reviews / experiences ---
    '("tried" OR "using" OR "experience with" OR "review of") ("AI twin" OR "AI bestie" OR "AI app") lang:en -is:retweet',

    # --- Social / lifestyle context ---
    '(college OR campus OR dorm OR freshman OR student) ("AI app" OR "AI twin" OR "AI companion") lang:en -is:retweet',

    # --- Productivity / life improvement context ---
    '("make life easier" OR "help me with" OR "AI for" OR "AI assistant") lang:en -is:retweet',

    # --- Emotional / relationship context ---
    '("AI friend" OR "virtual friend" OR "AI bestie" OR "AI girlfriend" OR "AI boyfriend") lang:en -is:retweet',

    # --- App discovery / comparison ---
    '("new app" OR "found this app" OR "best AI app" OR "AI social app" OR "AI texting app") lang:en -is:retweet'
]

MAX_PER_QUERY = 5           # how many tweets per query (subject to your Free quota)
FRESH_HOURS   = 12           # freshness window (hours)
MIN_SCORE     = 2            # rule-based score threshold to draft a reply
SEEN_FILE     = "seen_ids.json"  # simple local dedup store
POST_OR_REPLY = "reply"      # "reply" or "post"

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

# Patch: make hours_ago robust against import shadowing

def hours_ago(iso_ts: str) -> float:
    try:
        dt_obj = isoparse(iso_ts)
        if dt_obj.tzinfo is None:
            dt_obj = dt_obj.replace(tzinfo=timezone.utc)
    except Exception:
        return float("inf")
    return (datetime.now(timezone.utc) - dt_obj).total_seconds() / 3600.0

# PURPOSE: Query the Recent Search endpoint with a given operator string.

def _is_retweet_text(t):
    txt = (t.get("text") or "").lstrip()
    return txt.startswith("RT @")


def recent_search(query: str, max_results: int = 10):
    url_a = "https://api.x.com/2/tweets/search/recent"
    url_b = "https://api.twitter.com/2/tweets/search/recent"  # fallback
    params = {
        "query": query,
        "max_results": min(max_results, 100),
        "tweet.fields": "created_at,public_metrics,conversation_id",
    }

    def _get(url):
        return requests.get(url, headers=HEADERS_READ, params=params, timeout=20)

    r = _get(url_a)
    if r.status_code == 404:
        r = _get(url_b)

    remain = r.headers.get("x-rate-limit-remaining")
    reset  = r.headers.get("x-rate-limit-reset")

    # Hit Window: Stop this round
    if r.status_code == 429 or (r.status_code == 400 and remain == "0"):
        try:
            ts = int(reset)
            utc = datetime.fromtimestamp(ts, timezone.utc)
            print(f"[rate] window exhausted; resets at {utc.isoformat()}")
        except Exception:
            print("[rate] window exhausted; reset unknown")
        return [], True  # The second return value: whether it consumes

    if r.status_code >= 400:
        print("[http]", r.status_code, r.text[:200])
        return [], False

    data = r.json().get("data", []) or []
    return data, False

# PURPOSE: A minimal, explainable scoring function to filter candidates before LLM drafting.

def score_tweet(t: dict) -> int:
    s = 0
    text = t.get("text", "")

    # Rule 1: intent-like phrasing (questions / recommendations)
    if "?" in text or re.search(r"\b(recommend|suggestion|which app)\b", text, re.I):
        s += 1

    # Rule 2: freshness
    if t.get("created_at") and hours_ago(t["created_at"]) <= FRESH_HOURS:
        s += 1

    # Rule 3: minimal engagement signal
    pm = t.get("public_metrics", {})
    if (pm.get("reply_count", 0) + pm.get("like_count", 0)) >= 1:
        s += 1

    return s

# PURPOSE: Use Gemini to generate a concise, actionable reply (<=260 chars).

def text_trim(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + "…"

def hard_limit(s: str, n: int) -> str:
    return s[:n]

def gen_gemini_reply(original_text: str) -> str:
    system_msg = (
        "You are a helpful, concise peer on X. "
        "Constraints: <=260 characters; no mass hashtags; no aggressive promotion; "
        "1 short empathy line + 1–2 concrete tips tailored to the post; natural tone."
    )
    prompt = f"""Original post:
---
{text_trim(original_text, 800)}
---

Write a single reply (<=260 chars). Avoid templated or spammy phrasing. Provide practical tips."""
    model = genai.GenerativeModel("models/gemini-1.5-flash-8b-latest")
    out = model.generate_content([system_msg, prompt])
    draft = (out.text or "").strip()
    return hard_limit(draft, 260)

# PURPOSE: Human review to approve/edit/skip the LLM draft before posting.

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

# PURPOSE: Minimal wrappers to create a tweet or reply using OAuth 1.0a user context.

def post_tweet(text: str):
    url = "https://api.x.com/2/tweets"
    payload = {"text": text}
    r = requests.post(url, auth=auth_oauth1, json=payload)
    if r.status_code == 404:  # fallback domain
        url = "https://api.twitter.com/2/tweets"
        r = requests.post(url, auth=auth_oauth1, json=payload)
    return r

def reply_to(tweet_id: str, text: str):
    url = "https://api.x.com/2/tweets"
    payload = {"text": text, "reply": {"in_reply_to_tweet_id": tweet_id}}
    r = requests.post(url, auth=auth_oauth1, json=payload)
    if r.status_code == 404:
        url = "https://api.twitter.com/2/tweets"
        r = requests.post(url, auth=auth_oauth1, json=payload)
    return r

# PURPOSE: One pass: search -> score -> draft -> human review -> post/reply.
# Respects a simple dedup store to avoid interacting with the same tweet twice.
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
    # 1) initialize
    global SEEN
    headers, auth = init_clients()
    
    global HEADERS_READ, auth_oauth1
    HEADERS_READ, auth_oauth1 = headers, auth

    # 2) Select the query to run (single or multiple)
    queries = [query] if query else QUERIES
    candidates = []
    fetched = 0
    rate_exhausted = False

    # 3) Crawling + filtering + scoring (strictly respecting the target,
    # stopping when the upper limit is reached)
    for q in queries:
        if fetched >= target or rate_exhausted:
            break
        page_size = min(MAX_PER_QUERY, max(1, target - fetched))
        try:
            tweets, exhausted = recent_search(q, page_size)
            rate_exhausted = rate_exhausted or exhausted
        except Exception as e:
            print("Search error:", e)
            continue

        for t in tweets:
            if fetched >= target:
                break
            # Lightweight RT filtration
            txt = (t.get("text") or "").lstrip()
            if txt.startswith("RT @"):
                continue
            tid = t["id"]
            if tid in SEEN:
                continue

            sc = score_tweet(t)
            if sc >= MIN_SCORE:
                candidates.append((sc, t))
            fetched += 1

    # 4) sort and cut
    candidates.sort(key=lambda x: (-x[0], x[1].get("created_at", "")))
    candidates = candidates[:target]

    # 5) dry-run
    if dry_run:
        os.makedirs("data", exist_ok=True)
        out = "data/replies_queued.ndjson"
        with open(out, "w", encoding="utf-8") as f:
            for _, t in candidates:
                draft = gen_gemini_reply(t.get("text", ""))
                f.write(json.dumps({"tweet_id": t["id"], "reply_text": draft}, ensure_ascii=False) + "\n")
                SEEN.add(t["id"])
        save_seen(SEEN)
        print(f"[dry-run] queued {len(candidates)} replies -> {out}")
        if rate_exhausted:
            print("[rate] stopped early: current window exhausted.")
        return

    # 6) Post (with manual review + dithering + idempotence)
    for sc, t in candidates:
        tid = t["id"]
        original = t.get("text", "")
        draft = gen_gemini_reply(original)

        to_send = draft
        if interactive:
            approved = human_review(draft, original)
            if not approved:
                print("Skipped.")
                SEEN.add(tid); save_seen(SEEN)
                continue
            to_send = approved

        resp = post_tweet(to_send) if POST_OR_REPLY == "post" else reply_to(tid, to_send)
        print("API:", resp.status_code, resp.text[:200])

        SEEN.add(tid); save_seen(SEEN)
        time.sleep(random.uniform(8, 20))  # Jitter to avoid being like a robot