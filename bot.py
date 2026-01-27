import os
import time
import requests
from typing import Any, Dict, List, Optional, Tuple

# =========================
# Config (через env)
# =========================
TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TG_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

if not TG_TOKEN or not TG_CHAT_ID:
    raise RuntimeError("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID env vars")

MIN_USD = float(os.environ.get("MIN_CASH_USD", "3000"))

# Метка CHEAP (как раньше; это НЕ фильтр, а просто тег)
CHEAP_PRICE = float(os.environ.get("MAX_CHEAP_PRICE", "0.15"))

# НОВОЕ: фильтр по цене сделки (всё дороже — игнорируем)
MAX_ENTRY_PRICE = float(os.environ.get("MAX_ENTRY_PRICE", "0.40"))

POLL_SECONDS = int(os.environ.get("POLL_SECONDS", "15"))
TRADES_LIMIT = int(os.environ.get("TRADES_LIMIT", "100"))

DATA_API_TRADES = "https://data-api.polymarket.com/trades"
TG_SEND_URL = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"

# дедуп по уникальному ключу сделки
seen: set[Tuple[Any, ...]] = set()

# =========================
# Telegram
# =========================
def tg_send(text: str) -> None:
    if len(text) > 3500:
        text = text[:3500] + "\n...[truncated]"
    r = requests.post(
        TG_SEND_URL,
        data={
            "chat_id": TG_CHAT_ID,
            "text": text,
            "disable_web_page_preview": "true",
        },
        timeout=20,
    )
    if r.status_code != 200:
        print("TG_ERROR", r.status_code, r.text[:300], flush=True)

# =========================
# Polymarket Data API
# =========================
def request_json(url: str, params: dict, retries: int = 3) -> Any:
    last_err: Optional[str] = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=20)
            if r.status_code != 200:
                last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                time.sleep(2 * (i + 1))
                continue
            return r.json()
        except Exception as e:
            last_err = repr(e)
            time.sleep(2 * (i + 1))
    raise RuntimeError(last_err or "unknown error")

def fetch_trades() -> List[Dict[str, Any]]:
    # API-фильтр по денежной сумме сделки (но мы всё равно пересчитываем notional)
    params = {
        "limit": TRADES_LIMIT,
        "filterType": "CASH",
        "filterAmount": MIN_USD,
    }
    data = request_json(DATA_API_TRADES, params=params, retries=3)
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected /trades response type: {type(data)}")
    return data

def to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0

def trade_key(t: Dict[str, Any]) -> Tuple[Any, ...]:
    # Устойчивый ключ (поля могут быть пустыми/разными)
    return (
        t.get("transactionHash"),
        t.get("tradeId"),
        t.get("timestamp"),
        t.get("price"),
        t.get("size"),
        t.get("outcome"),
        t.get("side"),
        t.get("slug"),
    )

def format_signal(t: Dict[str, Any], price: float, size: float, notional: float) -> str:
    title = (t.get("title") or "").strip()
    outcome = (t.get("outcome") or "").strip()
    side = (t.get("side") or "").strip()
    slug = (t.get("slug") or "").strip()

    tags = ["BIG"]
    if 0 < price < CHEAP_PRICE:
        tags.insert(0, "CHEAP")  # CHEAP,BIG

    link = f"https://polymarket.com/market/{slug}" if slug else "https://polymarket.com/"

    return (
        f"[{','.join(tags)}] {title}\n"
        f"{outcome} | {side}\n"
        f"price={price:.4f} size={size:.2f} notional≈${notional:,.0f}\n"
        f"{link}"
    )

# =========================
# Main loop
# =========================
def main() -> None:
    tg_send(
        f"Bot started. MIN_USD={MIN_USD}, MAX_ENTRY_PRICE={MAX_ENTRY_PRICE}, "
        f"CHEAP<{CHEAP_PRICE}, poll={POLL_SECONDS}s"
    )

    while True:
        try:
            trades = fetch_trades()
            trades = list(reversed(trades))  # старые -> новые

            sent_count = 0
            for t in trades:
                k = trade_key(t)
                if k in seen:
                    continue

                price = to_float(t.get("price"))
                size = to_float(t.get("size"))
                notional = price * size  # приближённо, но стабильно

                # ФИЛЬТР: сумма + цена сделки <= 0.40
                if notional >= MIN_USD and 0 < price <= MAX_ENTRY_PRICE:
                    tg_send(format_signal(t, price, size, notional))
                    sent_count += 1

                seen.add(k)

            print(
                f"Tick: fetched={len(trades)} sent={sent_count} seen={len(seen)}",
                flush=True
            )

            if len(seen) > 8000:
                seen.clear()

        except Exception as e:
            print("ERROR", repr(e), flush=True)
            try:
                tg_send(f"Scanner error: {repr(e)[:900]}")
            except Exception:
                pass

        time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    main()
