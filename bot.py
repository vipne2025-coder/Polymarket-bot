import os
import time
import json
import threading
from datetime import datetime, timezone

import requests
import websocket

# =======================
# ENV
# =======================
TG_TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")
if not TG_TOKEN or not CHAT_ID:
    raise RuntimeError("Set TELEGRAM_TOKEN and CHAT_ID env vars")

TG_API = f"https://api.telegram.org/bot{TG_TOKEN}"

# =======================
# SETTINGS
# =======================
# Whale-trade (Subgraph) filter
SUBGRAPH_URL = "https://api.thegraph.com/subgraphs/name/polymarket/matic-markets"
MIN_USD = 3000
MAX_PRICE = 0.15
SUBGRAPH_POLL_SEC = 30

# Fast-move (WebSocket) filter
PRICE_LIMIT = 0.15          # only watch moves while price < 0.15
MOVE_PCT = 0.02             # 2% jump
MOVE_WINDOW_SEC = 5         # approximate window by comparing last seen values

WS_URL = "wss://clob.polymarket.com/ws"

# =======================
# Telegram
# =======================
def tg_send(text: str) -> None:
    if len(text) > 3500:
        text = text[:3500] + "\n...[truncated]"
    r = requests.post(
        f"{TG_API}/sendMessage",
        data={"chat_id": CHAT_ID, "text": text},
        timeout=20,
    )
    if r.status_code != 200:
        print("Telegram error:", r.status_code, r.text, flush=True)

# =======================
# Helpers
# =======================
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

# =======================
# Subgraph whale scanner
# =======================
seen_trade_ids = set()

def graphql(query: str) -> dict:
    r = requests.post(SUBGRAPH_URL, json={"query": query}, timeout=25)
    r.raise_for_status()
    data = r.json()
    if "errors" in data:
        raise RuntimeError(data["errors"])
    return data.get("data", {})

def fetch_recent_trades(limit: int = 250):
    query = f"""
    {{
      trades(first: {limit}, orderBy: timestamp, orderDirection: desc) {{
        id
        trader
        collateralAmountUSD
        price
        timestamp
        market {{ question }}
      }}
    }}
    """
    return graphql(query).get("trades", [])

def run_subgraph_loop():
    tg_send(f"Scanner started: trades usd>={MIN_USD}, price<{MAX_PRICE}; fast-move price<{PRICE_LIMIT}, move>={MOVE_PCT*100:.1f}%")
    print("Subgraph loop started", flush=True)

    while True:
        try:
            trades = fetch_recent_trades(300)
            trades = list(reversed(trades))  # old->new

            for t in trades:
                tid = t.get("id")
                if not tid or tid in seen_trade_ids:
                    continue

                usd = safe_float(t.get("collateralAmountUSD"))
                price = safe_float(t.get("price"))
                ts = safe_int(t.get("timestamp"))
                trader = t.get("trader", "unknown")
                q = (t.get("market") or {}).get("question", "unknown market")

                if usd >= MIN_USD and (0 < price < MAX_PRICE):
                    msg = (
                        "WHALE TRADE\n"
                        f"time:  {iso(ts)}\n"
                        f"usd:   ${usd:,.0f}\n"
                        f"price: {price}\n"
                        f"market: {q}\n"
                        f"trader: {trader}\n"
                    )
                    tg_send(msg)
                    print("Sent whale:", tid, flush=True)

                seen_trade_ids.add(tid)

            if len(seen_trade_ids) > 20000:
                seen_trade_ids.clear()

        except Exception as e:
            print("Subgraph error:", repr(e), flush=True)

        time.sleep(SUBGRAPH_POLL_SEC)

# =======================
# WebSocket fast-move scanner
# =======================
# We store last mid price per market + timestamp
last_mid = {}  # market_id -> (mid, ts)

def extract_book_like(payload: dict):
    """
    Try to extract:
      - market identifier
      - bids/asks arrays or best bid/ask directly

    Because schemas vary, we attempt multiple common patterns.
    Return (market, best_bid, best_ask) or (None, None, None).
    """
    market = payload.get("market") or payload.get("market_id") or payload.get("asset_id") or payload.get("token_id")

    # Pattern A: payload has bids/asks arrays of dicts {price, size}
    bids = payload.get("bids")
    asks = payload.get("asks")

    def best_price(side):
        if not isinstance(side, list) or not side:
            return None
        # elements might be dicts or [price, size]
        first = side[0]
        if isinstance(first, dict):
            return safe_float(first.get("price"), None)
        if isinstance(first, (list, tuple)) and len(first) >= 1:
            return safe_float(first[0], None)
        return None

    best_bid = best_price(bids) if bids is not None else None
    best_ask = best_price(asks) if asks is not None else None

    # Pattern B: payload has bestBid/bestAsk or bid/ask
    if best_bid is None:
        for k in ("bestBid", "best_bid", "bid"):
            if k in payload:
                best_bid = safe_float(payload.get(k), None)
                break
    if best_ask is None:
        for k in ("bestAsk", "best_ask", "ask"):
            if k in payload:
                best_ask = safe_float(payload.get(k), None)
                break

    if market is None or best_bid is None or best_ask is None:
        return None, None, None
    return str(market), best_bid, best_ask

def on_ws_message(ws, message):
    try:
        data = json.loads(message)
    except Exception:
        return

    # Some WS send wrapper objects; try unwrap common patterns
    payload = data.get("data") if isinstance(data, dict) and "data" in data else data

    # Optional: keep one-line debug of unknown messages (rate-limited)
    # print("WS msg keys:", list(payload.keys())[:10], flush=True)

    market, best_bid, best_ask = extract_book_like(payload)
    if not market:
        return

    mid = (best_bid + best_ask) / 2.0
    now = int(time.time())

    prev = last_mid.get(market)
    last_mid[market] = (mid, now)

    if not prev:
        return

    prev_mid, prev_ts = prev
    if prev_mid <= 0:
        return

    # approximate window: only compare if last update was recent
    if now - prev_ts > MOVE_WINDOW_SEC:
        return

    move = (mid - prev_mid) / prev_mid

    # Only alert when in "cheap zone"
    if mid < PRICE_LIMIT and move >= MOVE_PCT:
        msg = (
            "FAST MOVE\n"
            f"market_id: {market}\n"
            f"mid: {mid:.4f}\n"
            f"move: +{move*100:.2f}% in ~{now-prev_ts}s\n"
            f"hint: possible large buy"
        )
        tg_send(msg)
        print("Sent fast-move:", market, flush=True)

def on_ws_open(ws):
    print("WebSocket connected", flush=True)
    # Subscription message depends on Polymarket WS schema.
    # We try a common generic subscribe pattern; if it doesn't work, logs will show server responses.
    try:
        ws.send(json.dumps({"type": "subscribe", "channel": "book"}))
    except Exception:
        pass

def on_ws_error(ws, error):
    print("WebSocket error:", repr(error), flush=True)

def on_ws_close(ws, code, reason):
    print("WebSocket closed:", code, reason, flush=True)

def run_ws_loop():
    while True:
        try:
            ws = websocket.WebSocketApp(
                WS_URL,
                on_open=on_ws_open,
                on_message=on_ws_message,
                on_error=on_ws_error,
                on_close=on_ws_close,
            )
            ws.run_forever(ping_interval=20, ping_timeout=10)
        except Exception as e:
            print("WS loop error:", repr(e), flush=True)

        time.sleep(3)  # reconnect backoff

# =======================
# Main
# =======================
if __name__ == "__main__":
    print("Starting combined scanner...", flush=True)

    t1 = threading.Thread(target=run_subgraph_loop, daemon=True)
    t2 = threading.Thread(target=run_ws_loop, daemon=True)
    t1.start()
    t2.start()

    # keep process alive
    while True:
        time.sleep(60)
