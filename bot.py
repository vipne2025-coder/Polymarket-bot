import os
import time
import json
import sqlite3
import requests
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

# =========================
# Telegram
# =========================
TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TG_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
if not TG_TOKEN or not TG_CHAT_ID:
    raise RuntimeError("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID env vars")

TG_SEND_URL = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"

def tg_send(text: str) -> None:
    if len(text) > 3500:
        text = text[:3500] + "\n...[truncated]"
    r = requests.post(
        TG_SEND_URL,
        data={"chat_id": TG_CHAT_ID, "text": text, "disable_web_page_preview": "true"},
        timeout=20,
    )
    if r.status_code != 200:
        print("TG_ERROR", r.status_code, r.text[:300], flush=True)

# =========================
# Polymarket Data API
# =========================
DATA_API_TRADES = "https://data-api.polymarket.com/trades"

def request_json(url: str, params: dict, retries: int = 3) -> Any:
    last_err: Optional[str] = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=20)
            if r.status_code != 200:
                last_err = f"HTTP {r.status_code}: {r.text[:250]}"
                time.sleep(2 * (i + 1))
                continue
            return r.json()
        except Exception as e:
            last_err = repr(e)
            time.sleep(2 * (i + 1))
    raise RuntimeError(last_err or "unknown error")

def fetch_trades(min_usd: float, limit: int) -> List[Dict[str, Any]]:
    params = {"limit": limit}
    # фильтруем по WATCH_MIN_USD на стороне API (меньше мусора)
    if min_usd > 0:
        params.update({"filterType": "CASH", "filterAmount": min_usd})
    data = request_json(DATA_API_TRADES, params=params, retries=3)
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected /trades response type: {type(data)}")
    return data

# =========================
# Config (balanced, not too strict)
# =========================
# 2 уровня сигналов
INSIDER_MIN_USD = float(os.environ.get("INSIDER_MIN_USD", "3000"))
WATCH_MIN_USD = float(os.environ.get("WATCH_MIN_USD", "1000"))

# фильтр по цене сделки
MAX_ENTRY_PRICE = float(os.environ.get("MAX_ENTRY_PRICE", "0.40"))
MIN_PRICE = float(os.environ.get("MIN_PRICE", "0.05"))  # режем копеечный арбитраж

# пометки
EARLY_PRICE = float(os.environ.get("EARLY_PRICE", "0.20"))
CHEAP_PRICE = float(os.environ.get("MAX_CHEAP_PRICE", "0.15"))

# скорость/свежесть
POLL_SECONDS = int(os.environ.get("POLL_SECONDS", "12"))
TRADES_LIMIT = int(os.environ.get("TRADES_LIMIT", "140"))
MAX_TRADE_AGE_SEC = int(os.environ.get("MAX_TRADE_AGE_SEC", "600"))  # 10 минут — чтобы сигналы были

# анти-спам
MARKET_COOLDOWN_SEC = int(os.environ.get("MARKET_COOLDOWN_SEC", "180"))  # 3 минуты
WALLET_COOLDOWN_SEC = int(os.environ.get("WALLET_COOLDOWN_SEC", "120"))  # 2 минуты

# анти-вилочник (работает только когда wallet есть)
FLIP_WINDOW_SEC = int(os.environ.get("FLIP_WINDOW_SEC", "1800"))         # 30 минут
MARKET_WINDOW_SEC = int(os.environ.get("MARKET_WINDOW_SEC", "900"))      # 15 минут
MAX_TRADES_PER_MARKET_WINDOW = int(os.environ.get("MAX_TRADES_PER_MARKET_WINDOW", "8"))
BLACKLIST_TTL_SEC = int(os.environ.get("BLACKLIST_TTL_SEC", "43200"))    # 12 часов (не сутки)

# ВАЖНО: разрешаем сигналы без wallet (иначе часто "тишина")
ALLOW_NO_WALLET = int(os.environ.get("ALLOW_NO_WALLET", "1"))  # 1 = да

# Отладка: печатать статистику отсева в логах
DEBUG_STATS = int(os.environ.get("DEBUG_STATS", "1"))
DEBUG_EVERY_SEC = int(os.environ.get("DEBUG_EVERY_SEC", "120"))

DB_PATH = os.environ.get("DB_PATH", "scanner.db")

# =========================
# Helpers
# =========================
def now_ts() -> int:
    return int(time.time())

def to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0

def to_int(x: Any) -> int:
    try:
        return int(float(x))
    except Exception:
        return 0

def extract_wallet(t: Dict[str, Any]) -> str:
    for k in ("maker", "taker", "trader", "user", "wallet", "address"):
        v = t.get(k)
        if isinstance(v, str) and v:
            return v
    return ""

def short_addr(addr: str) -> str:
    if not addr or len(addr) < 12:
        return "n/a"
    return f"{addr[:6]}...{addr[-4:]}"

def side_to_yes_no(side: str) -> str:
    s = (side or "").upper()
    if s == "BUY":
        return "YES"
    if s == "SELL":
        return "NO"
    return s or "?"

def market_link(slug: str) -> str:
    return f"https://polymarket.com/market/{slug}" if slug else "https://polymarket.com"

def trade_key(t: Dict[str, Any]) -> str:
    # строковый ключ для SQLite
    parts = {
        "tx": t.get("transactionHash"),
        "id": t.get("tradeId"),
        "ts": t.get("timestamp"),
        "p": t.get("price"),
        "s": t.get("size"),
        "o": t.get("outcome"),
        "side": t.get("side"),
        "slug": t.get("slug"),
    }
    return json.dumps(parts, sort_keys=True, ensure_ascii=False)

# =========================
# SQLite
# =========================
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def init_db() -> None:
    conn = db()
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS seen_trades (k TEXT PRIMARY KEY, ts INTEGER)")
    cur.execute("CREATE TABLE IF NOT EXISTS last_alert (key TEXT PRIMARY KEY, ts INTEGER)")
    cur.execute("""
      CREATE TABLE IF NOT EXISTS wallet_blacklist (
        wallet TEXT PRIMARY KEY,
        reason TEXT,
        until_ts INTEGER
      )
    """)
    cur.execute("""
      CREATE TABLE IF NOT EXISTS wallet_market_actions (
        wallet TEXT, slug TEXT, side TEXT, ts INTEGER
      )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_wma ON wallet_market_actions(wallet, slug, ts)")
    conn.commit()
    conn.close()

def seen_trade(k: str) -> bool:
    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM seen_trades WHERE k=?", (k,))
    row = cur.fetchone()
    conn.close()
    return row is not None

def mark_seen_trade(k: str, ts: int) -> None:
    conn = db()
    conn.execute("INSERT OR REPLACE INTO seen_trades(k, ts) VALUES(?, ?)", (k, ts))
    conn.commit()
    conn.close()

def get_last_alert(key: str) -> int:
    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT ts FROM last_alert WHERE key=?", (key,))
    row = cur.fetchone()
    conn.close()
    return int(row[0]) if row else 0

def set_last_alert(key: str, ts: int) -> None:
    conn = db()
    conn.execute("INSERT OR REPLACE INTO last_alert(key, ts) VALUES(?, ?)", (key, ts))
    conn.commit()
    conn.close()

def blacklist_wallet(wallet: str, reason: str, ttl_sec: int = BLACKLIST_TTL_SEC) -> None:
    until_ts = now_ts() + ttl_sec
    conn = db()
    conn.execute(
        "INSERT OR REPLACE INTO wallet_blacklist(wallet, reason, until_ts) VALUES(?, ?, ?)",
        (wallet, reason, until_ts),
    )
    conn.commit()
    conn.close()

def is_blacklisted(wallet: str) -> Tuple[bool, str]:
    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT reason, until_ts FROM wallet_blacklist WHERE wallet=?", (wallet,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return False, ""
    reason, until_ts = str(row[0]), int(row[1])
    if until_ts <= now_ts():
        conn = db()
        conn.execute("DELETE FROM wallet_blacklist WHERE wallet=?", (wallet,))
        conn.commit()
        conn.close()
        return False, ""
    return True, reason

def record_action(wallet: str, slug: str, side: str, ts: int) -> None:
    conn = db()
    conn.execute(
        "INSERT INTO wallet_market_actions(wallet, slug, side, ts) VALUES(?, ?, ?, ?)",
        (wallet, slug, side, ts),
    )
    conn.execute(
        "DELETE FROM wallet_market_actions WHERE ts < ?",
        (ts - max(FLIP_WINDOW_SEC, MARKET_WINDOW_SEC) - 60,),
    )
    conn.commit()
    conn.close()

def detect_flip_and_frequency(wallet: str, slug: str, side: str, ts: int) -> Tuple[bool, str]:
    conn = db()
    cur = conn.cursor()
    cur.execute("""
      SELECT side FROM wallet_market_actions
      WHERE wallet=? AND slug=? AND ts>=?
      ORDER BY ts DESC
      LIMIT 50
    """, (wallet, slug, ts - FLIP_WINDOW_SEC))
    sides = [r[0] for r in cur.fetchall()]

    cur.execute("""
      SELECT COUNT(*) FROM wallet_market_actions
      WHERE wallet=? AND slug=? AND ts>=?
    """, (wallet, slug, ts - MARKET_WINDOW_SEC))
    cnt = int(cur.fetchone()[0])
    conn.close()

    opp = "SELL" if side == "BUY" else "BUY"
    if opp in sides:
        return True, "flip BUY<->SELL (arb/mm)"
    if cnt >= MAX_TRADES_PER_MARKET_WINDOW:
        return True, f"too frequent in market ({cnt}/{MARKET_WINDOW_SEC}s)"
    return False, ""

# =========================
# Cooldowns
# =========================
def should_alert_by_cooldown(slug: str, wallet: str, ts: int) -> bool:
    if slug and ts - get_last_alert(f"m:{slug}") < MARKET_COOLDOWN_SEC:
        return False
    if wallet and ts - get_last_alert(f"w:{wallet}") < WALLET_COOLDOWN_SEC:
        return False
    return True

def set_cooldown(slug: str, wallet: str, ts: int) -> None:
    if slug:
        set_last_alert(f"m:{slug}", ts)
    if wallet:
        set_last_alert(f"w:{wallet}", ts)

# =========================
# Two-level classifier
# =========================
def classify(price: float, notional: float, wallet_present: bool) -> str:
    # базовые правила: INSIDER/WATCH
    if notional >= INSIDER_MIN_USD:
        # если кошелька нет — всё равно разрешаем INSIDER (иначе будет тишина),
        # но требуем более “раннюю/адекватную” цену, чтобы не ловить мусор.
        if (not wallet_present) and (not ALLOW_NO_WALLET):
            return "DROP"
        if (not wallet_present) and price > 0.30:
            return "WATCH"
        return "INSIDER"

    if notional >= WATCH_MIN_USD:
        if (not wallet_present) and (not ALLOW_NO_WALLET):
            return "DROP"
        return "WATCH"

    return "DROP"

def format_msg(level: str, t: Dict[str, Any], price: float, size: float, notional: float, wallet: str) -> str:
    title = (t.get("title") or "Unknown market").strip()
    outcome = (t.get("outcome") or "Unknown outcome").strip()
    side = (t.get("side") or "").upper()
    slug = (t.get("slug") or "").strip()

    early_tag = " ⚡EARLY" if 0 < price < EARLY_PRICE else ""
    cheap_tag = " 💎CHEAP" if 0 < price < CHEAP_PRICE else ""

    header = "🔥 INSIDER" if level == "INSIDER" else "👀 WATCH"

    msg = (
        f"{header} SIGNAL{early_tag}{cheap_tag}\n\n"
        f"📊 {title}\n"
        f"📌 СТАВКА: {outcome} — {side_to_yes_no(side)}\n\n"
        f"💵 Цена: ${price:.3f}\n"
        f"📦 Акции: {size:,.0f}\n"
        f"💰 Сумма: ~${notional:,.0f}\n"
        f"👛 Wallet: {short_addr(wallet)}\n\n"
        f"🔗 {market_link(slug)}"
    )
    if wallet and wallet.startswith("0x"):
        msg = msg.replace("👛 Wallet:", f"👛 Wallet: {short_addr(wallet)}\nhttps://polygonscan.com/address/{wallet}\n")
        # аккуратно: чтобы не было дубля "wallet"
        msg = msg.replace(f"👛 Wallet: {short_addr(wallet)}\nhttps://polygonscan.com/address/{wallet}\n{short_addr(wallet)}", f"👛 Wallet: {short_addr(wallet)}\nhttps://polygonscan.com/address/{wallet}")
    return msg

# =========================
# Main
# =========================
def main() -> None:
    init_db()

    tg_send(
        "✅ Scanner started (balanced)\n"
        f"INSIDER≥${INSIDER_MIN_USD}, WATCH≥${WATCH_MIN_USD}\n"
        f"price∈[{MIN_PRICE},{MAX_ENTRY_PRICE}], age≤{MAX_TRADE_AGE_SEC}s\n"
        f"ALLOW_NO_WALLET={ALLOW_NO_WALLET}"
    )

    last_debug = 0
    counters = defaultdict(int)

    while True:
        try:
            now = now_ts()
            trades = fetch_trades(WATCH_MIN_USD, limit=TRADES_LIMIT)
            trades = list(reversed(trades))

            sent = 0

            for t in trades:
                k = trade_key(t)
                if seen_trade(k):
                    counters["skip_seen"] += 1
                    continue

                ts = to_int(t.get("timestamp")) or now
                age = now - ts
                if age > MAX_TRADE_AGE_SEC:
                    counters["skip_age"] += 1
                    mark_seen_trade(k, ts)
                    continue

                price = to_float(t.get("price"))
                size = to_float(t.get("size"))
                notional = price * size

                if not (MIN_PRICE <= price <= MAX_ENTRY_PRICE):
                    counters["skip_price"] += 1
                    mark_seen_trade(k, ts)
                    continue

                if notional < WATCH_MIN_USD:
                    counters["skip_small"] += 1
                    mark_seen_trade(k, ts)
                    continue

                slug = (t.get("slug") or "").strip()
                if not slug:
                    counters["skip_no_slug"] += 1
                    mark_seen_trade(k, ts)
                    continue

                wallet = extract_wallet(t)
                wallet_present = bool(wallet)

                # если wallet нет и запрещено — режем
                if (not wallet_present) and (not ALLOW_NO_WALLET):
                    counters["skip_no_wallet"] += 1
                    mark_seen_trade(k, ts)
                    continue

                # blacklist / anti-viloch (только если wallet есть)
                if wallet_present:
                    bl, _ = is_blacklisted(wallet)
                    if bl:
                        counters["skip_blacklist"] += 1
                        mark_seen_trade(k, ts)
                        continue

                    side = (t.get("side") or "").upper()
                    if side in ("BUY", "SELL"):
                        record_action(wallet, slug, side, ts)
                        bad, reason = detect_flip_and_frequency(wallet, slug, side, ts)
                        if bad:
                            blacklist_wallet(wallet, reason)
                            counters["skip_viloch"] += 1
                            mark_seen_trade(k, ts)
                            continue

                # cooldown (для уменьшения шума)
                if not should_alert_by_cooldown(slug, wallet, ts):
                    counters["skip_cooldown"] += 1
                    mark_seen_trade(k, ts)
                    continue

                level = classify(price, notional, wallet_present)
                if level == "DROP":
                    counters["skip_class"] += 1
                    mark_seen_trade(k, ts)
                    continue

                tg_send(format_msg(level, t, price, size, notional, wallet))
                set_cooldown(slug, wallet, ts)
                sent += 1
                counters["sent"] += 1

                mark_seen_trade(k, ts)

            print(f"Tick: fetched={len(trades)} sent={sent}", flush=True)

            # debug summary раз в DEBUG_EVERY_SEC
            if DEBUG_STATS and (now - last_debug >= DEBUG_EVERY_SEC):
                last_debug = now
                summary = " | ".join(f"{k}={v}" for k, v in sorted(counters.items()))
                print("DEBUG:", summary, flush=True)
                # сбрасываем, чтобы видеть динамику
                counters.clear()

        except Exception as e:
            print("ERROR:", repr(e), flush=True)
            try:
                tg_send(f"⚠️ Scanner error: {repr(e)[:900]}")
            except Exception:
                pass

        time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    main()
