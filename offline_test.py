import os, json
from dotenv import load_dotenv

# 显式指定 .env 的路径，避免 stdin 场景下的断言问题
load_dotenv(dotenv_path=".env", override=False)

# 用你刚测通的模型
os.environ.setdefault("GEMINI_MODEL", "models/gemini-1.5-flash-8b-latest")

from app.main import _is_retweet_text, score_tweet, gen_gemini_reply, MIN_SCORE, SEEN, save_seen

with open("data/sample_tweets.json", "r", encoding="utf-8") as f:
    items = json.load(f)

candidates = []
for t in items:
    if _is_retweet_text(t):
        continue
    tid = t.get("id")
    if not tid or tid in SEEN:
        continue
    sc = score_tweet(t)
    if sc >= MIN_SCORE:
        candidates.append((sc, t))

candidates.sort(key=lambda x: (-x[0], x[1].get("created_at","")))

os.makedirs("data", exist_ok=True)
out = "data/replies_queued.ndjson"
with open(out, "w", encoding="utf-8") as f:
    for _, t in candidates:
        draft = gen_gemini_reply(t.get("text",""))
        f.write(json.dumps({"tweet_id": t["id"], "reply_text": draft}, ensure_ascii=False) + "\n")
        SEEN.add(t["id"])

save_seen(SEEN)
print(f"[offline] queued {len(candidates)} replies -> {out}")
