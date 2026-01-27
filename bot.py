import os
import time
import json
import sqlite3
import requests
from typing import Any, Dict, List, Optional, Tuple

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

def fetch_trades(filter_min_usd: float, limit: int) -> List[Dict[str, Any]]:
    params = {"limit": limit}
    if filter_min_usd > 0:
        params.update({"filterType": "CASH", "filterAmount": filter_min_usd})
    data = request_json(DATA_API_TRADES, params=params, retries=3)
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected /trades response type: {type(data)}")
    return data

# =========================
# Config (Railway Variables)
# =========================
MIN_USD = float(os.environ.get("MIN_CASH_USD", "3000"))

# твой фильтр по цене сделки
MAX_ENTRY_PRICE = float(os.environ.get("MAX_ENTRY_PRICE", "0.40"))

# метки
EARLY_PRICE = float(os.environ.get("EARLY_PRICE", "0.20"))
CHEAP_PRICE = float(os.environ.get("MAX_CHEAP_PRICE", "0.15"))

# анти-вилочники: режем совсем “копейки”
MIN_PRICE = float(os.environ.get("MIN_PRICE", "0.05"))

# скорость/свежесть
POLL_SECONDS = int(os.environ.get("POLL_SECONDS", "12"))
TRADES_LIMIT = int(os.environ.get("TRADES_LIMIT", "140"))
MAX_TRADE_AGE_SEC = int(os.environ.get("MAX_TRADE_AGE_SEC", "120"))

# меньше спама
MARKET_COOLDOWN_SEC = int(os.environ.get("MARKET_COOLDOWN_SEC", "240"))
WALLET_COOLDOWN_SEC = int(os.environ.get("WALLET_COOLDOWN_SEC", "120"))

# анти-арб/анти-бот
FLIP_WINDOW_SEC = int(os.environ.get("FLIP_WINDOW_SEC", "1800"))
MARKET_WINDOW_SEC = int(os.environ.get("MARKET_WINDOW_SEC", "900"))
MAX_TRADES_PER_MARKET_WINDOW = int(os.environ.get("MAX_TRADES_PER_MARKET_WINDOW", "6"))
BLACKLIST_TTL_SEC = int(os.environ.get("BLACKLIST_TTL_SEC", "86400"))

# режим отправки
SEND_WATCH = int(os.environ.get("SEND_WATCH", "0"))  # 0=только STRONG, 1=+WATCH

# анти-FOMO подтверждение
CONFIRM_DELAY_SEC = int(os.environ.get("CONFIRM_DELAY_SEC", "25"))
MAX_PRICE_SLIPPAGE_PCT = float(os.environ.get("MAX_PRICE_SLIPPAGE_PCT", "0.15"))  # 15%

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

def side_to_yes_no(side: str) -> str:
    s = (side or "").upper()
    if s == "BUY":
        return "YES"
    if s == "SELL":
        return "NO"
    return s or "?"

def short_addr(addr: str) -> str:
    if not addr or len(addr) < 12:
        return "n/a"
    return f"{addr[:6]}...{addr[-4:]}"

def extract_wallet(t: Dict[str, Any]) -> str:
    for k in ("maker", "taker", "trader", "user", "wallet", "address"):
        v = t.get(k)
        if isinstance(v, str) and v:
            return v
    return ""

def trade_key(t: Dict[str, Any]) -> str:
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

def market_link(slug: str) -> str:
    return f"https://polymarket.com/market/{slug}" if slug else "https://polymarket.com"

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
    cur.execute("""
      CREATE TABLE IF NOT EXISTS wallet_stats (
        wallet TEXT PRIMARY KEY,
        total INTEGER DEFAULT 0,
        big INTEGER DEFAULT 0,
        early INTEGER DEFAULT 0,
        flips INTEGER DEFAULT 0,
        updated_ts INTEGER DEFAULT 0
      )
    """)
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
    cur.execute("CREATE TABLE IF NOT EXISTS last_alert (key TEXT PRIMARY KEY, ts INTEGER)")
    cur.execute("""
      CREATE TABLE IF NOT EXISTS pending_confirm (
        trade_k TEXT PRIMARY KEY,
        slug TEXT,
        wallet TEXT,
        side TEXT,
        outcome TEXT,
        title TEXT,
        first_price REAL,
        first_notional REAL,
        first_size REAL,
        created_ts INTEGER,
        confirm_after_ts INTEGER
      )
    """)
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

def bump_wallet_stats(wallet: str, notional: float, price: float) -> None:
    ts = now_ts()
    big = 1 if notional >= 10000 else 0
    early = 1 if 0 < price < EARLY_PRICE else 0
    conn = db()
    conn.execute("""
      INSERT INTO wallet_stats(wallet, total, big, early, flips, updated_ts)
      VALUES(?, 1, ?, ?, 0, ?)
      ON CONFLICT(wallet) DO UPDATE SET
        total = total + 1,
        big = big + ?,
        early = early + ?,
        updated_ts = ?
    """, (wallet, big, early, ts, big, early, ts))
    conn.commit()
    conn.close()

def bump_flip(wallet: str) -> None:
    ts = now_ts()
    conn = db()
    conn.execute("""
      INSERT INTO wallet_stats(wallet, total, big, early, flips, updated_ts)
      VALUES(?, 0, 0, 0, 1, ?)
      ON CONFLICT(wallet) DO UPDATE SET
        flips = flips + 1,
        updated_ts = ?
    """, (wallet, ts, ts))
    conn.commit()
    conn.close()

def wallet_score(wallet: str) -> int:
    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT total, big, early, flips FROM wallet_stats WHERE wallet=?", (wallet,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return 0
    total, big, early, flips = map(int, row)
    # скоринг: опыт + ранние + большие, flips сильно штрафуют
    return min(total, 40) + (big * 4) + (early * 6) - (flips * 10)

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

def enqueue_confirm(trade_k: str, slug: str, wallet: str, side: str, outcome: str, title: str,
                    first_price: float, first_notional: float, first_size: float, created_ts: int) -> None:
    conn = db()
    conn.execute("""
      INSERT OR REPLACE INTO pending_confirm
      (trade_k, slug, wallet, side, outcome, title, first_price, first_notional, first_size, created_ts, confirm_after_ts)
      VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (trade_k, slug, wallet, side, outcome, title, first_price, first_notional, first_size, created_ts,
          created_ts + CONFIRM_DELAY_SEC))
    conn.commit()
    conn.close()

def pop_due_confirms(ts: int) -> List[Dict[str, Any]]:
    conn = db()
    cur = conn.cursor()
    cur.execute("""
      SELECT trade_k, slug, wallet, side, outcome, title, first_price, first_notional, first_size, created_ts, confirm_after_ts
      FROM pending_confirm
      WHERE confirm_after_ts <= ?
      ORDER BY confirm_after_ts ASC
      LIMIT 50
    """, (ts,))
    rows = cur.fetchall()
    cur.execute("DELETE FROM pending_confirm WHERE confirm_after_ts <= ?", (ts,))
    conn.commit()
    conn.close()

    out = []
    for r in rows:
        out.append({
            "trade_k": r[0], "slug": r[1], "wallet": r[2], "side": r[3],
            "outcome": r[4], "title": r[5],
            "first_price": float(r[6]), "first_notional": float(r[7]), "first_size": float(r[8]),
            "created_ts": int(r[9]), "confirm_after_ts": int(r[10]),
        })
    return out

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
# Signal classification + formatting
# =========================
def classify(price: float, notional: float, score: int, wallet_present: bool) -> str:
    # без кошелька — супер строго
    if not wallet_present:
        if notional >= 20000 and price < EARLY_PRICE:
            return "WATCH"
        return "DROP"

    # STRONG (максимально безопасно)
    if price < EARLY_PRICE and notional >= 7000 and score >= 12:
        return "STRONG"
    if price < CHEAP_PRICE and notional >= 10000 and score >= 8:
        return "STRONG"

    # WATCH (опционально)
    if price <= 0.30 and notional >= MIN_USD and score >= 8:
        return "WATCH"

    return "DROP"

def tags_line(price: float, notional: float, score: int) -> str:
    tags = []
    if price < EARLY_PRICE:
        tags.append("EARLY")
    if price < CHEAP_PRICE:
        tags.append("CHEAP")
    tags.append("WHALE" if notional >= 10000 else "BIG")
    if score >= 25:
        tags.append("TRUST+")
    elif score >= 12:
        tags.append("TRUST")
    else:
        tags.append("NEW")
    return " + ".join(tags)

def format_signal(level: str, title: str, outcome: str, side: str,
                  price: float, size: float, notional: float,
                  wallet: str, score: int, slug: str,
                  note: str = "") -> str:
    lvl_emoji = "🟢" if level == "STRONG" else "🟡"
    msg = (
        f"{lvl_emoji} {level} SIGNAL — {tags_line(price, notional, score)}\n\n"
        f"📊 {title}\n"
        f"📌 СТАВКА: {outcome} — {side_to_yes_no(side)}\n\n"
        f"💵 Цена входа: ${price:.3f}\n"
        f"📦 Акции: {size:,.0f}\n"
        f"💰 Сумма: ~${notional:,.0f}\n\n"
    )
    if wallet:
        msg += f"👛 Wallet: {short_addr(wallet)} | score={score}\n"
        if wallet.startswith("0x"):
            msg += f"https://polygonscan.com/address/{wallet}\n\n"
        else:
            msg += "\n"
    else:
        msg += "👛 Wallet: n/a\n\n"

    if note:
        msg += f"⚠️ {note}\n\n"

    msg += f"🔗 {market_link(slug)}"
    return msg

def latest_price_for_slug(slug: str) -> Optional[float]:
    # пере-проверка цены через свежие трейды (быстро и без других API)
    trades = fetch_trades(0.0, limit=80)
    for t in trades:
        if (t.get("slug") or "").strip() == slug:
            p = to_float(t.get("price"))
            if p > 0:
                return p
    return None

# =========================
# Main
# =========================
def main() -> None:
    init_db()
    tg_send(
        "✅ SAFE PRO scanner started\n"
        f"Filters: notional≥{MIN_USD}, price∈[{MIN_PRICE},{MAX_ENTRY_PRICE}], age≤{MAX_TRADE_AGE_SEC}s\n"
        f"EARLY<{EARLY_PRICE}, CHEAP<{CHEAP_PRICE}, poll={POLL_SECONDS}s\n"
        f"Confirm={CONFIRM_DELAY_SEC}s, SEND_WATCH={SEND_WATCH}"
    )
    print("Started", flush=True)

    while True:
        try:
            # 1) Confirm queue (anti-FOMO)
            due = pop_due_confirms(now_ts())
            for item in due:
                slug = item["slug"]
                first_price = item["first_price"]

                note = ""
                current_price = latest_price_for_slug(slug)
                if current_price and first_price > 0:
                    move = (current_price - first_price) / first_price
                    if move > MAX_PRICE_SLIPPAGE_PCT:
                        note = f"Цена уже выше примерно на +{move*100:.0f}% — вход может быть поздним"

                wallet = item["wallet"]
                score = wallet_score(wallet) if wallet else 0

                tg_send(format_signal(
                    level="STRONG",
                    title=item["title"],
                    outcome=item["outcome"],
                    side=item["side"],
                    price=first_price,
                    size=item["first_size"],
                    notional=item["first_notional"],
                    wallet=wallet,
                    score=score,
                    slug=item["slug"],
                    note=note
                ))

            # 2) New trades
            trades = fetch_trades(MIN_USD, limit=TRADES_LIMIT)
            trades = list(reversed(trades))  # old->new

            queued = 0
            for t in trades:
                k = trade_key(t)
                if seen_trade(k):
                    continue

                ts = to_int(t.get("timestamp")) or now_ts()
                if now_ts() - ts > MAX_TRADE_AGE_SEC:
                    mark_seen_trade(k, ts)
                    continue

                price = to_float(t.get("price"))
                size = to_float(t.get("size"))
                notional = price * size

                slug = (t.get("slug") or "").strip()
                title = (t.get("title") or "Unknown market").strip()
                outcome = (t.get("outcome") or "Unknown outcome").strip()
                side = (t.get("side") or "").upper()

                # hard filters
                if notional < MIN_USD:
                    mark_seen_trade(k, ts); continue
                if not (MIN_PRICE <= price <= MAX_ENTRY_PRICE):
                    mark_seen_trade(k, ts); continue
                if not slug:
                    # ради безопасности: без ссылки на рынок чаще мусор
                    mark_seen_trade(k, ts); continue

                wallet = extract_wallet(t)
                wallet_present = bool(wallet)

                # blacklist
                if wallet_present:
                    bl, _ = is_blacklisted(wallet)
                    if bl:
                        mark_seen_trade(k, ts); continue

                # anti-arb
                if wallet_present and side in ("BUY", "SELL"):
                    record_action(wallet, slug, side, ts)
                    bad, reason = detect_flip_and_frequency(wallet, slug, side, ts)
                    if bad:
                        bump_flip(wallet)
                        blacklist_wallet(wallet, reason)
                        mark_seen_trade(k, ts); continue

                # scoring
                score = 0
                if wallet_present:
                    bump_wallet_stats(wallet, notional, price)
                    score = wallet_score(wallet)

                level = classify(price, notional, score, wallet_present)
                if level == "DROP":
                    mark_seen_trade(k, ts); continue
                if level == "WATCH" and not SEND_WATCH:
                    mark_seen_trade(k, ts); continue

                # cooldown
                if not should_alert_by_cooldown(slug, wallet, ts):
                    mark_seen_trade(k, ts); continue

                # STRONG -> confirm flow (safe), WATCH -> direct
                if level == "STRONG":
                    enqueue_confirm(
                        trade_k=k, slug=slug, wallet=wallet, side=side,
                        outcome=outcome, title=title,
                        first_price=price, first_notional=notional, first_size=size,
                        created_ts=ts
                    )
                    queued += 1
                else:
                    tg_send(format_signal(
                        level=level, title=title, outcome=outcome, side=side,
                        price=price, size=size, notional=notional,
                        wallet=wallet, score=score, slug=slug
                    ))

                set_cooldown(slug, wallet, ts)
                mark_seen_trade(k, ts)

            print(f"Tick: fetched={len(trades)} queued={queued}", flush=True)

        except Exception as e:
            print("ERROR", repr(e), flush=True)
            try:
                tg_send(f"⚠️ Scanner error: {repr(e)[:900]}")
            except Exception:
                pass

        time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    main()
