import os
import time
import requests
from datetime import datetime, timezone

# ====== НАСТРОЙКИ ======
SUBGRAPH_URL = "https://api.thegraph.com/subgraphs/name/polymarket/matic-markets"

MIN_USD = 3_000
MAX_PRICE = 0.15
POLL_SEC = 30
EARLY_WINDOW_HOURS = 6  # "ранний" бонус в скоринге

# Telegram (обязательно через env)
TG_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TG_CHAT_ID = os.environ.get("CHAT_ID")

if not TG_TOKEN or not TG_CHAT_ID:
    raise RuntimeError("Missing TELEGRAM_TOKEN or CHAT_ID env vars")

TG_API = f"https://api.telegram.org/bot{TG_TOKEN}"

# защита от повторных отправок
seen_trade_ids = set()

def iso(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

def safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def safe_int(x, default=0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default

def tg_send(text: str) -> None:
    # Telegram ограничивает длину сообщения; на всякий случай обрежем
    if len(text) > 3500:
        text = text[:3500] + "\n...[truncated]"
    r = requests.post(
        f"{TG_API}/sendMessage",
        data={"chat_id": TG_CHAT_ID, "text": text},
        timeout=20,
    )
    # если Telegram вернёт ошибку, увидишь в логах
    if r.status_code != 200:
        print("Telegram error:", r.status_code, r.text, flush=True)

def graphql(query: str) -> dict:
    r = requests.post(SUBGRAPH_URL, json={"query": query}, timeout=25)
    r.raise_for_status()
    data = r.json()
    if "errors" in data:
        raise RuntimeError(data["errors"])
    return data.get("data", {})

def fetch_recent_trades(limit: int = 200):
    # Поля createdAt* могут отсутствовать — тогда "ранний" скоринг будет n/a, но фильтр продолжит работать.
    query = f"""
    {{
      trades(first: {limit}, orderBy: timestamp, orderDirection: desc) {{
        id
        trader
        collateralAmountUSD
        price
        timestamp
        market {{
          id
          question
          createdAtTimestamp
          creationTimestamp
          createdAt
          createdTime
        }}
      }}
    }}
    """
    return graphql(query).get("trades", [])

def market_created_ts(market: dict) -> int | None:
    for k in ("createdAtTimestamp", "creationTimestamp", "createdAt", "createdTime"):
        ts = safe_int(market.get(k)) if market else 0
        if ts > 0:
            return ts
    return None

def score_trade(usd: float, price: float, trade_ts: int, mkt_ts: int | None) -> tuple[int, str]:
    score = 0
    tags = []

    # базовые условия
    if usd >= MIN_USD:
        score += 10; tags.append("usd>=min")
    if 0 < price <= MAX_PRICE:
        score += 10; tags.append("cheap<0.15")

    # бонус за размер
    if usd >= 25_000:
        score += 8; tags.append("usd>=25k")
    elif usd >= 10_000:
        score += 5; tags.append("usd>=10k")

    # ранний вход (если есть created timestamp рынка)
    if mkt_ts:
        age_sec = trade_ts - mkt_ts
        if age_sec >= 0:
            age_h = age_sec / 3600
            if age_h <= EARLY_WINDOW_HOURS:
                score += 20; tags.append(f"early<= {EARLY_WINDOW_HOURS}h")
            elif age_h <= 24:
                score += 8; tags.append("early<=24h")

    return score, ",".join(tags)

def format_signal(s: dict) -> str:
    created = iso(s["marketCreated"]) if s["marketCreated"] else "n/a"
    return (
        "SIGNAL (whale/cheap)\n"
        f"score: {s['score']} | tags: {s['tags']}\n"
        f"time:  {iso(s['time'])}\n"
        f"usd:   ${s['usd']:,.0f}\n"
        f"price: {s['price']}\n"
        f"mkt_created: {created}\n"
        f"market: {s['question']}\n"
        f"wallet: {s['wallet']}\n"
    )

def main():
    print("Scanner started", flush=True)
    tg_send("Scanner started (MIN_USD=3000, price<0.15).")

    while True:
        try:
            trades = fetch_recent_trades(250)
            signals = []

            for t in trades:
                tid = t.get("id")
                if not tid or tid in seen_trade_ids:
                    continue

                usd = safe_float(t.get("collateralAmountUSD"))
                price = safe_float(t.get("price"))
                trade_ts = safe_int(t.get("timestamp"))

                # фильтр: все рынки, дешево + объем
                if usd >= MIN_USD and (0 < price < MAX_PRICE):
                    market = t.get("market") or {}
                    q = market.get("question", "unknown market")
                    trader = t.get("trader", "unknown")
                    mkt_ts = market_created_ts(market)

                    score, tags = score_trade(usd, price, trade_ts, mkt_ts)

                    signals.append({
                        "score": score,
                        "time": trade_ts,
                        "usd": usd,
                        "price": price,
                        "wallet": trader,
                        "question": q,
                        "marketCreated": mkt_ts,
                        "tags": tags,
                        "tradeId": tid,
                    })

                seen_trade_ids.add(tid)

            # приоритет: сначала самые "ранние/сильные"
            signals.sort(key=lambda x: (x["score"], x["usd"]), reverse=True)

            # шлём ВСЕ сигналы (но только новые, благодаря seen_trade_ids)
            for s in signals:
                msg = format_signal(s)
                tg_send(msg)
                print("Sent:", s["tradeId"], flush=True)

            if len(seen_trade_ids) > 10_000:
                seen_trade_ids.clear()

        except Exception as e:
            print("Error:", repr(e), flush=True)

        time.sleep(POLL_SEC)

if __name__ == "__main__":
    main()
