import os
import time
import math
import csv
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests

# ============================================================
# Bybit 15m Signals Bot (Telegram alerts only)
# - Signals: 15m
# - Entry confirmation: 5m
# - Trend filter: 1h (EMA200 + EMA50 bias)
# - Setups:
#   A) 15m impulse -> 30-60% pullback -> 5m continuation confirm
#   B) 15m level rejection at robust range edges -> 5m confirm
#
# Quality & UX upgrades:
# - Structural stop (never "repaint" stop to fit %). If stop too wide -> reject signal.
# - Hard gates: MAX_STOP_PCT and MIN_RR (to TP1)
# - Two-stage API flow to reduce rate-limit risk:
#   Stage-1 (cheap): 15m scan, candidates only
#   Stage-2 (expensive): 5m, 1h, orderbook on candidates
# - Signal TTL + per-symbol cooldown
# - "Price entered entry zone" notification (optional)
# - Signal outcome logging (CSV) with MFE/MAE at 15/30/60m (optional)
# ============================================================

# =========================
# Telegram (required)
# =========================
TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TG_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
if not TG_TOKEN or not TG_CHAT_ID:
    raise RuntimeError("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in environment variables.")

def tg_send(text: str) -> None:
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {
        "chat_id": TG_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    r = requests.post(url, json=payload, timeout=20)
    r.raise_for_status()

# =========================
# Config (env)
# =========================
BYBIT_BASE = os.environ.get("BYBIT_BASE", "https://api.bybit.com")

TOP_N = int(os.environ.get("TOP_N", "25"))
POLL_SECONDS = int(os.environ.get("POLL_SECONDS", "20"))

# User sizing (message hints only)
EQUITY_USD = float(os.environ.get("EQUITY_USD", "200"))
NOTIONAL_USD = float(os.environ.get("NOTIONAL_USD", "50"))
LEVERAGE = float(os.environ.get("LEVERAGE", "10"))

# Timeframes (minutes)
TF_SIGNAL = os.environ.get("TF_SIGNAL", "15")     # 15m
TF_CONFIRM = os.environ.get("TF_CONFIRM", "5")    # 5m
TF_TREND = os.environ.get("TF_TREND", "60")       # 1h

# Mechanics
SIGNAL_TTL_SEC = int(os.environ.get("SIGNAL_TTL_SEC", "2400"))           # 40 min
SYMBOL_COOLDOWN_SEC = int(os.environ.get("SYMBOL_COOLDOWN_SEC", "1200")) # 20 min per symbol

# Setup A: impulse candle on 15m
IMPULSE_ATR_MULT = float(os.environ.get("IMPULSE_ATR_MULT", "1.6"))      # range >= ATR*mult
IMPULSE_BODY_FRAC = float(os.environ.get("IMPULSE_BODY_FRAC", "0.55"))   # abs(body)/range >= frac
CLOSE_NEAR_EXT_FRAC = float(os.environ.get("CLOSE_NEAR_EXT_FRAC", "0.25")) # close within 25% of high/low

# Pullback on impulse BODY (percent)
PULLBACK_MIN_PCT = float(os.environ.get("PULLBACK_MIN_PCT", "30"))  # 30% retrace
PULLBACK_MAX_PCT = float(os.environ.get("PULLBACK_MAX_PCT", "60"))  # 60% retrace
MAX_CHASE_PCT = float(os.environ.get("MAX_CHASE_PCT", "0.20"))      # don't chase if moved away > this %

# Setup B: rejection at levels
LEVEL_LOOKBACK = int(os.environ.get("LEVEL_LOOKBACK", "40"))  # 40 candles of 15m (~10h)
LEVEL_TOL_PCT = float(os.environ.get("LEVEL_TOL_PCT", "0.20")) # 0.20% tolerance to "touch" level
REJECT_WICK_FRAC = float(os.environ.get("REJECT_WICK_FRAC", "0.45"))  # wick >= 45% of range
ROBUST_LEVEL_Q = float(os.environ.get("ROBUST_LEVEL_Q", "0.05"))      # ignore extreme 5% spikes for range edges (0.05 -> 5%)

# Hard risk/quality gates (crucial for your bankroll style)
MAX_STOP_PCT = float(os.environ.get("MAX_STOP_PCT", "0.60"))           # reject if structural stop wider than this
MIN_RR = float(os.environ.get("MIN_RR", "2.0"))                        # reject if RR to TP1 < this

# Stop/TP (R-multiples around structural stop distance)
STOP_BUFFER_ATR = float(os.environ.get("STOP_BUFFER_ATR", "0.20"))     # ATR buffer behind swing/level
TP1_R = float(os.environ.get("TP1_R", "1.2"))
TP2_R = float(os.environ.get("TP2_R", "2.0"))

# Market filters
USE_ORDERBOOK = int(os.environ.get("USE_ORDERBOOK", "1")) == 1
OB_LIMIT = int(os.environ.get("OB_LIMIT", "25"))
MAX_SPREAD_PCT = float(os.environ.get("MAX_SPREAD_PCT", "0.10"))

# Candidate scanning controls (to avoid rate limits)
MAX_CANDIDATES = int(os.environ.get("MAX_CANDIDATES", "8"))            # stage-2 symbols per tick
CANDIDATE_SCORE_MIN = float(os.environ.get("CANDIDATE_SCORE_MIN", "1"))# internal; keep >=1

# Notifications UX
REMIND_SEC = int(os.environ.get("REMIND_SEC", "600"))          # 10 min
MAX_REMINDERS = int(os.environ.get("MAX_REMINDERS", "1"))
SEND_IN_ZONE = int(os.environ.get("SEND_IN_ZONE", "1")) == 1   # notify when price enters entry zone

# Quiet hours (optional) Europe/Amsterdam default
QUIET_HOURS = os.environ.get("QUIET_HOURS", "").strip()        # e.g. "01:00-08:00" local time; empty disables

# Logging (optional)
ENABLE_SIGNAL_LOG = os.environ.get("ENABLE_SIGNAL_LOG", "false").lower() in ("1","true","yes","y")
SIGNAL_LOG_FILE = os.environ.get("SIGNAL_LOG_FILE", "signals_log.csv").strip()
EVAL_MINUTES = [int(x.strip()) for x in os.environ.get("EVAL_MINUTES", "15,30,60").split(",") if x.strip().isdigit()]

# =========================
# Bybit HTTP helpers (V5)
# =========================
_session = requests.Session()

def bybit_get(path: str, params: dict, retries: int = 2) -> dict:
    """
    Minimal retry/backoff on transient failures / rate limiting.
    Bybit may respond with HTTP 403 "access too frequent" when IP rate limit is exceeded.
    """
    url = f"{BYBIT_BASE}{path}"
    last_err = None
    for attempt in range(retries + 1):
        try:
            r = _session.get(url, params=params, timeout=20)
            # Basic backoff on 403/429
            if r.status_code in (403, 429):
                time.sleep(0.8 + attempt * 0.8)
                continue
            r.raise_for_status()
            data = r.json()
            # retCode 10016: rate limit exceeded (common)
            if data.get("retCode") in (10016, 10006):
                time.sleep(0.8 + attempt * 0.8)
                continue
            return data
        except Exception as e:
            last_err = e
            time.sleep(0.4 + attempt * 0.5)
    raise RuntimeError(f"Bybit GET failed: {path} {params} err={repr(last_err)}")

def get_linear_tickers() -> List[dict]:
    data = bybit_get("/v5/market/tickers", {"category": "linear"})
    if data.get("retCode") != 0:
        raise RuntimeError(f"Bybit tickers retCode={data.get('retCode')} msg={data.get('retMsg')}")
    return data["result"]["list"]

def get_kline(symbol: str, interval: str, limit: int) -> List[list]:
    data = bybit_get("/v5/market/kline", {
        "category": "linear",
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    })
    if data.get("retCode") != 0:
        raise RuntimeError(f"Bybit kline retCode={data.get('retCode')} msg={data.get('retMsg')}")
    return data["result"]["list"]

def get_orderbook(symbol: str) -> dict:
    data = bybit_get("/v5/market/orderbook", {
        "category": "linear",
        "symbol": symbol,
        "limit": OB_LIMIT
    })
    if data.get("retCode") != 0:
        raise RuntimeError(f"Bybit orderbook retCode={data.get('retCode')} msg={data.get('retMsg')}")
    return data["result"]

# =========================
# Utilities
# =========================
def now_ts() -> int:
    return int(time.time())

def f(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def spread_pct(bid: float, ask: float) -> float:
    if bid <= 0 or ask <= 0:
        return 999.0
    mid = (bid + ask) / 2.0
    return abs(ask - bid) / mid * 100.0

def fmt_pct(x: float, nd: int = 2) -> str:
    s = f"{x:.{nd}f}%"
    return s.replace("-", "−")

def fmt_price(x: float) -> str:
    if x >= 1000:
        return f"{x:,.2f}"
    if x >= 100:
        return f"{x:.3f}"
    if x >= 10:
        return f"{x:.4f}"
    return f"{x:.6f}"

def parse_klines(raw: List[list]) -> List[dict]:
    # Bybit returns newest-first: [startTime, open, high, low, close, volume, turnover]
    out = []
    for row in reversed(raw):  # oldest -> newest
        out.append({
            "ts": int(row[0]) // 1000 if str(row[0]).isdigit() else int(float(row[0]) / 1000),
            "open": f(row[1]),
            "high": f(row[2]),
            "low": f(row[3]),
            "close": f(row[4]),
            "volume": f(row[5]),
        })
    return out

def ema(values: List[float], period: int) -> Optional[float]:
    if len(values) < period:
        return None
    k = 2 / (period + 1)
    e = values[0]
    for v in values[1:]:
        e = v * k + e * (1 - k)
    return e

def atr(candles: List[dict], period: int = 14) -> Optional[float]:
    if len(candles) < period + 1:
        return None
    trs = []
    for i in range(1, len(candles)):
        h = candles[i]["high"]
        l = candles[i]["low"]
        pc = candles[i - 1]["close"]
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr)
    trs = trs[-period:]
    return sum(trs) / period if trs else None

def candle_stats(c: dict) -> Tuple[float, float, float, float, float, float]:
    o, h, l, cl = c["open"], c["high"], c["low"], c["close"]
    rng = max(1e-12, h - l)
    body = cl - o
    body_abs = abs(body)
    upper_wick = h - max(o, cl)
    lower_wick = min(o, cl) - l
    return rng, body, body_abs, upper_wick, lower_wick, cl

def quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    vs = sorted(values)
    idx = int(round((len(vs)-1) * q))
    idx = max(0, min(len(vs)-1, idx))
    return vs[idx]

def orderbook_imbalance(symbol: str) -> Optional[float]:
    if not USE_ORDERBOOK:
        return None
    try:
        ob = get_orderbook(symbol)
        bids = ob.get("b", []) or []
        asks = ob.get("a", []) or []
        if not bids or not asks:
            return None
        bid_vol = sum(f(x[1]) for x in bids)
        ask_vol = sum(f(x[1]) for x in asks)
        denom = bid_vol + ask_vol
        if denom <= 0:
            return None
        return (bid_vol - ask_vol) / denom
    except Exception:
        return None

def in_quiet_hours() -> bool:
    if not QUIET_HOURS:
        return False
    try:
        import datetime
        start_s, end_s = QUIET_HOURS.split("-")
        sh, sm = [int(x) for x in start_s.split(":")]
        eh, em = [int(x) for x in end_s.split(":")]
        now = datetime.datetime.now()
        start = now.replace(hour=sh, minute=sm, second=0, microsecond=0)
        end = now.replace(hour=eh, minute=em, second=0, microsecond=0)
        if end <= start:
            # crosses midnight
            return now >= start or now < end
        return start <= now < end
    except Exception:
        return False

# =========================
# Signal model
# =========================
@dataclass
class Signal:
    symbol: str
    setup: str                 # "A" or "B"
    direction: str             # "LONG" / "SHORT"
    created_ts: int
    score: float
    entry_low: float
    entry_high: float
    stop: float
    tp1: float
    tp2: float
    reason: str
    spread: float
    ob_imb: Optional[float]
    trend_dir: Optional[str]
    entry_mid: float
    stop_pct: float
    rr_to_tp1: float
    in_zone_notified: bool = False
    reminders_sent: int = 0

# =========================
# Logging: CSV + delayed evaluation (optional)
# =========================
_log_lock = threading.Lock()
_pending_evals: Dict[str, dict] = {}  # key -> eval state

def _log_init() -> None:
    if not ENABLE_SIGNAL_LOG:
        return
    header = [
        "timestamp","symbol","side","setup","score",
        "entry_low","entry_high","entry_mid",
        "stop","stop_pct","tp1","tp2","rr_to_tp1",
        "trend_1h","ob_imb","reason",
    ]
    for m in EVAL_MINUTES:
        header += [f"mfe_{m}", f"mae_{m}", f"hit_tp1_{m}", f"hit_tp2_{m}", f"hit_sl_{m}"]
    try:
        exists = os.path.exists(SIGNAL_LOG_FILE)
        if not exists:
            with _log_lock, open(SIGNAL_LOG_FILE, "w", newline="", encoding="utf-8") as fcsv:
                csv.writer(fcsv).writerow(header)
    except Exception as e:
        print("LOG_INIT_ERROR", repr(e), flush=True)

def _log_row(base: dict) -> None:
    if not ENABLE_SIGNAL_LOG:
        return
    try:
        with _log_lock, open(SIGNAL_LOG_FILE, "a", newline="", encoding="utf-8") as fcsv:
            w = csv.writer(fcsv)
            row = [base.get(k,"") for k in [
                "timestamp","symbol","side","setup","score",
                "entry_low","entry_high","entry_mid",
                "stop","stop_pct","tp1","tp2","rr_to_tp1",
                "trend_1h","ob_imb","reason",
            ]]
            for m in EVAL_MINUTES:
                row += [base.get(f"mfe_{m}",""), base.get(f"mae_{m}",""),
                        base.get(f"hit_tp1_{m}",""), base.get(f"hit_tp2_{m}",""), base.get(f"hit_sl_{m}","")]
            w.writerow(row)
    except Exception as e:
        print("LOG_WRITE_ERROR", repr(e), flush=True)

def _schedule_evals(sig: Signal) -> None:
    """
    Store minimal state; later we will evaluate using 1m klines for the horizon.
    Keyed by symbol+created_ts.
    """
    if not ENABLE_SIGNAL_LOG or not EVAL_MINUTES:
        return
    key = f"{sig.symbol}:{sig.created_ts}"
    _pending_evals[key] = {
        "sig": sig,
        "logged": False,
        "base": {
            "timestamp": sig.created_ts,
            "symbol": sig.symbol,
            "side": sig.direction,
            "setup": sig.setup,
            "score": int(round(sig.score)),
            "entry_low": sig.entry_low,
            "entry_high": sig.entry_high,
            "entry_mid": sig.entry_mid,
            "stop": sig.stop,
            "stop_pct": sig.stop_pct,
            "tp1": sig.tp1,
            "tp2": sig.tp2,
            "rr_to_tp1": sig.rr_to_tp1,
            "trend_1h": sig.trend_dir or "",
            "ob_imb": f"{sig.ob_imb:+.3f}" if sig.ob_imb is not None else "",
            "reason": sig.reason,
        }
    }

def _eval_pending() -> None:
    """
    Every loop: check if any scheduled horizon is due; if yes, compute MFE/MAE up to that horizon.
    Uses 1m klines for accuracy.
    """
    if not ENABLE_SIGNAL_LOG or not _pending_evals:
        return
    now = now_ts()
    done_keys = []
    for key, st in list(_pending_evals.items()):
        sig: Signal = st["sig"]
        base = st["base"]
        # determine which horizons are due and missing
        updated = False
        for m in EVAL_MINUTES:
            if f"mfe_{m}" in base:
                continue
            due_ts = sig.created_ts + m*60
            if now < due_ts:
                continue
            try:
                # Fetch 1m candles covering the horizon (+2 for safety)
                limit = min(1000, m + 5)
                c1 = parse_klines(get_kline(sig.symbol, "1", limit))
                # Keep candles with ts >= created_ts
                window = [c for c in c1 if c["ts"] >= sig.created_ts]
                if not window:
                    continue
                highs = [c["high"] for c in window]
                lows  = [c["low"] for c in window]
                entry = sig.entry_mid

                if sig.direction == "LONG":
                    mfe = (max(highs) - entry) / entry * 100.0
                    mae = (entry - min(lows)) / entry * 100.0
                    hit_tp1 = 1 if max(highs) >= sig.tp1 else 0
                    hit_tp2 = 1 if max(highs) >= sig.tp2 else 0
                    hit_sl  = 1 if min(lows) <= sig.stop else 0
                else:
                    mfe = (entry - min(lows)) / entry * 100.0
                    mae = (max(highs) - entry) / entry * 100.0
                    hit_tp1 = 1 if min(lows) <= sig.tp1 else 0
                    hit_tp2 = 1 if min(lows) <= sig.tp2 else 0
                    hit_sl  = 1 if max(highs) >= sig.stop else 0

                base[f"mfe_{m}"] = round(mfe, 4)
                base[f"mae_{m}"] = round(mae, 4)
                base[f"hit_tp1_{m}"] = hit_tp1
                base[f"hit_tp2_{m}"] = hit_tp2
                base[f"hit_sl_{m}"]  = hit_sl
                updated = True
            except Exception as e:
                print("EVAL_ERROR", key, m, repr(e), flush=True)

        # If we have at least one eval and haven't logged yet, log now; later updates will append another line?
        # Better: log once, only when all horizons are done OR after first horizon and then update by re-writing is hard.
        # We choose: log when the last horizon becomes available.
        if all((f"mfe_{m}" in base) for m in EVAL_MINUTES):
            _log_row(base)
            done_keys.append(key)
        else:
            # keep pending until complete; avoid spamming file
            pass

    for k in done_keys:
        _pending_evals.pop(k, None)

# =========================
# Symbol selection
# =========================
def pick_top_symbols(tickers: List[dict], top_n: int) -> List[dict]:
    def key(t: dict) -> float:
        return f(t.get("turnover24h"), 0.0)
    out = []
    for t in tickers:
        sym = str(t.get("symbol", ""))
        if not sym.endswith("USDT"):
            continue
        if key(t) <= 0:
            continue
        out.append(t)
    return sorted(out, key=key, reverse=True)[:top_n]

# =========================
# Trend cache (1h)
# =========================
_trend_cache: Dict[str, Tuple[int, Optional[str]]] = {}  # sym -> (ts, trend_dir)
TREND_CACHE_SEC = int(os.environ.get("TREND_CACHE_SEC", "240"))         # 4 min

def trend_filter_1h(symbol: str) -> Optional[str]:
    """
    Returns "LONG", "SHORT", or None (no clear trend).
    Simple: price vs EMA200 on 1h and EMA50>=EMA200 bias.
    Cached.
    """
    now = now_ts()
    cached = _trend_cache.get(symbol)
    if cached and (now - cached[0] < TREND_CACHE_SEC):
        return cached[1]

    raw = get_kline(symbol, TF_TREND, 230)
    c = parse_klines(raw)
    closes = [x["close"] for x in c]
    e200 = ema(closes[-210:], 200)
    e50 = ema(closes[-80:], 50) if len(closes) >= 80 else None
    if e200 is None or e50 is None:
        _trend_cache[symbol] = (now, None)
        return None
    last = closes[-1]
    trend = None
    if last > e200 and e50 >= e200:
        trend = "LONG"
    elif last < e200 and e50 <= e200:
        trend = "SHORT"
    _trend_cache[symbol] = (now, trend)
    return trend

# =========================
# Levels
# =========================
def compute_levels_15m_robust(c15: List[dict]) -> Tuple[float, float]:
    """
    Robust range edges over last LEVEL_LOOKBACK candles:
    use quantiles to ignore rare wicks/spikes.
    """
    window = c15[-LEVEL_LOOKBACK:]
    lows = [x["low"] for x in window]
    highs = [x["high"] for x in window]
    q = clamp(ROBUST_LEVEL_Q, 0.0, 0.2)
    lo = quantile(lows, q)
    hi = quantile(highs, 1.0 - q)
    # safety: keep within actual extremes
    lo = max(min(lows), lo)
    hi = min(max(highs), hi)
    return lo, hi

# =========================
# Micro confirmation (5m)
# =========================
def confirm_5m_break(symbol: str, c5: List[dict], direction: str, lookback: int = 6) -> bool:
    """
    Stronger confirmation than EMA+color:
    - candle closes in direction
    - closes beyond micro pivot (break of last N highs/lows)
    """
    if len(c5) < lookback + 5:
        return False
    last5 = c5[-2]  # last closed
    window = c5[-(lookback+2):-2]
    hi = max(x["high"] for x in window)
    lo = min(x["low"] for x in window)
    closes5 = [x["close"] for x in c5[:-1]]
    e20 = ema(closes5[-60:], 20) if len(closes5) >= 25 else None
    if e20 is None:
        return False

    if direction == "LONG":
        return (last5["close"] > last5["open"]) and (last5["close"] > e20) and (last5["close"] >= hi)
    else:
        return (last5["close"] < last5["open"]) and (last5["close"] < e20) and (last5["close"] <= lo)

# =========================
# Setup detection
# =========================
def detect_setup_A_candidate(c15: List[dict]) -> Optional[Tuple[str, dict]]:
    """
    Cheap stage-1 candidate check using only 15m.
    Returns (dir, payload) if impulse exists on last closed candle.
    """
    if len(c15) < 30:
        return None
    impulse = c15[-2]
    atr15 = atr(c15[:-1], 14)
    if atr15 is None or atr15 <= 0:
        return None
    rng, body, body_abs, uw, lw, close = candle_stats(impulse)
    if rng < atr15 * IMPULSE_ATR_MULT:
        return None
    if body_abs / rng < IMPULSE_BODY_FRAC:
        return None
    near_high = (impulse["high"] - impulse["close"]) / rng <= CLOSE_NEAR_EXT_FRAC
    near_low = (impulse["close"] - impulse["low"]) / rng <= CLOSE_NEAR_EXT_FRAC
    if body > 0 and near_high:
        return ("LONG", {"atr15": atr15, "impulse": impulse})
    if body < 0 and near_low:
        return ("SHORT", {"atr15": atr15, "impulse": impulse})
    return None

def detect_setup_A(symbol: str, c15: List[dict], c5: List[dict], trend_dir: str, ob_imb: Optional[float]) -> Optional[Signal]:
    """
    Full Setup A validation with 5m confirm + structural stop + RR gates.
    """
    cand = detect_setup_A_candidate(c15)
    if cand is None:
        return None
    dir_, payload = cand
    if dir_ != trend_dir:
        return None

    atr15 = payload["atr15"]
    impulse = payload["impulse"]
    imp_o = impulse["open"]
    imp_c = impulse["close"]
    imp_body = abs(imp_c - imp_o)
    if imp_body <= 0:
        return None

    pb_min = PULLBACK_MIN_PCT / 100.0
    pb_max = PULLBACK_MAX_PCT / 100.0
    if pb_max < pb_min:
        pb_min, pb_max = pb_max, pb_min

    if dir_ == "LONG":
        z_high = imp_c - imp_body * pb_min
        z_low  = imp_c - imp_body * pb_max
    else:
        z_low  = imp_c + imp_body * pb_min
        z_high = imp_c + imp_body * pb_max

    entry_low, entry_high = (min(z_low, z_high), max(z_low, z_high))
    entry_mid = (entry_low + entry_high) / 2.0
    if entry_mid <= 0:
        return None

    # Anti-chase: current price must not be far away from the zone
    last_price = c5[-1]["close"]
    if last_price > 0:
        if dir_ == "LONG":
            chase = max(0.0, (last_price - entry_high) / last_price * 100.0)
        else:
            chase = max(0.0, (entry_low - last_price) / last_price * 100.0)
        if chase > MAX_CHASE_PCT:
            return None

    # Zone must have been touched recently
    touched = any(cc["low"] <= entry_high and cc["high"] >= entry_low for cc in c5[-6:-1])
    if not touched:
        return None

    # Stronger 5m confirmation
    if not confirm_5m_break(symbol, c5, dir_):
        return None

    # Score base
    score = 82.0
    if ob_imb is not None:
        if dir_ == "LONG" and ob_imb < -0.20:
            score -= 10
        if dir_ == "SHORT" and ob_imb > 0.20:
            score -= 10

    # Structural stop: behind local 15m swing + ATR buffer (DO NOT repaint)
    swing_window = c15[-8:-1]  # last 7 completed candles
    if dir_ == "LONG":
        swing = min(x["low"] for x in swing_window)
        stop = swing - atr15 * STOP_BUFFER_ATR
        r = entry_mid - stop
    else:
        swing = max(x["high"] for x in swing_window)
        stop = swing + atr15 * STOP_BUFFER_ATR
        r = stop - entry_mid

    if r <= 0:
        return None

    stop_pct = abs(entry_mid - stop) / entry_mid * 100.0
    if stop_pct > MAX_STOP_PCT:
        return None

    # Targets based on R
    if dir_ == "LONG":
        tp1 = entry_mid + r * TP1_R
        tp2 = entry_mid + r * TP2_R
    else:
        tp1 = entry_mid - r * TP1_R
        tp2 = entry_mid - r * TP2_R

    # RR gate to TP1 (here RR=TP1_R, but we keep it explicit in case you later change TP definition)
    rr_to_tp1 = (abs(tp1 - entry_mid) / abs(entry_mid - stop)) if abs(entry_mid - stop) > 0 else 0.0
    if rr_to_tp1 < MIN_RR:
        return None

    reason = f"Setup A: 15m impulse (ATR×{IMPULSE_ATR_MULT:g}) → pullback {PULLBACK_MIN_PCT:.0f}-{PULLBACK_MAX_PCT:.0f}% → 5m BOS confirm"
    return Signal(
        symbol=symbol, setup="A", direction=dir_, created_ts=now_ts(),
        score=score, entry_low=entry_low, entry_high=entry_high,
        stop=stop, tp1=tp1, tp2=tp2, reason=reason,
        spread=0.0, ob_imb=ob_imb, trend_dir=trend_dir,
        entry_mid=entry_mid, stop_pct=stop_pct, rr_to_tp1=rr_to_tp1
    )

def detect_setup_B_candidate(c15: List[dict]) -> Optional[Tuple[str, dict]]:
    """
    Cheap stage-1 check for level rejection on 15m.
    Uses robust range edges.
    """
    if len(c15) < LEVEL_LOOKBACK + 5:
        return None
    range_low, range_high = compute_levels_15m_robust(c15)
    last_closed = c15[-2]
    rng, body, body_abs, uw, lw, close = candle_stats(last_closed)
    if rng <= 0:
        return None

    tol = close * (LEVEL_TOL_PCT / 100.0)
    touch_low = (last_closed["low"] <= range_low + tol)
    touch_high = (last_closed["high"] >= range_high - tol)

    long_reject = touch_low and (lw / rng >= REJECT_WICK_FRAC) and (last_closed["close"] > last_closed["open"])
    short_reject = touch_high and (uw / rng >= REJECT_WICK_FRAC) and (last_closed["close"] < last_closed["open"])
    if not (long_reject or short_reject):
        return None
    dir_ = "LONG" if long_reject else "SHORT"
    return (dir_, {"range_low": range_low, "range_high": range_high})

def detect_setup_B(symbol: str, c15: List[dict], c5: List[dict], trend_dir: Optional[str], ob_imb: Optional[float]) -> Optional[Signal]:
    """
    Full Setup B with 5m BOS confirm, structural stop at range edge, RR gates.
    """
    cand = detect_setup_B_candidate(c15)
    if cand is None:
        return None
    dir_, payload = cand
    range_low = payload["range_low"]
    range_high = payload["range_high"]
    last_closed = c15[-2]
    entry_low: float
    entry_high: float

    # Trend bias: if 1h trend exists and we go against it, require stronger orderbook alignment
    if trend_dir is not None and dir_ != trend_dir:
        if ob_imb is None:
            return None
        if dir_ == "LONG" and ob_imb < 0.12:
            return None
        if dir_ == "SHORT" and ob_imb > -0.12:
            return None

    # Entry zone around the level + last close (tight enough for fast trades)
    if dir_ == "LONG":
        entry_low = min(range_low, last_closed["low"])
        entry_high = min(last_closed["close"], range_low * (1.0 + 0.002))  # small cap above level
    else:
        entry_high = max(range_high, last_closed["high"])
        entry_low = max(last_closed["close"], range_high * (1.0 - 0.002))

    entry_mid = (entry_low + entry_high) / 2.0
    if entry_mid <= 0:
        return None

    # Anti-chase (use current 5m last price)
    last_price = c5[-1]["close"]
    if last_price > 0:
        if dir_ == "LONG":
            chase = max(0.0, (last_price - entry_high) / last_price * 100.0)
        else:
            chase = max(0.0, (entry_low - last_price) / last_price * 100.0)
        if chase > MAX_CHASE_PCT:
            return None

    # Stronger 5m confirmation
    if not confirm_5m_break(symbol, c5, dir_):
        return None

    # Score
    score = 74.0
    if trend_dir is not None and dir_ == trend_dir:
        score += 6
    if ob_imb is not None:
        if dir_ == "LONG" and ob_imb > 0.10:
            score += 4
        if dir_ == "SHORT" and ob_imb < -0.10:
            score += 4  # <-- fixed bug (was noop)

    atr15 = atr(c15[:-1], 14)
    if atr15 is None:
        return None

    # Structural stop behind the range edge + ATR buffer (DO NOT repaint)
    if dir_ == "LONG":
        stop = range_low - atr15 * STOP_BUFFER_ATR
        r = entry_mid - stop
    else:
        stop = range_high + atr15 * STOP_BUFFER_ATR
        r = stop - entry_mid

    if r <= 0:
        return None

    stop_pct = abs(entry_mid - stop) / entry_mid * 100.0
    if stop_pct > MAX_STOP_PCT:
        return None

    # Targets
    if dir_ == "LONG":
        tp1 = entry_mid + r * TP1_R
        tp2 = entry_mid + r * TP2_R
    else:
        tp1 = entry_mid - r * TP1_R
        tp2 = entry_mid - r * TP2_R

    rr_to_tp1 = (abs(tp1 - entry_mid) / abs(entry_mid - stop)) if abs(entry_mid - stop) > 0 else 0.0
    if rr_to_tp1 < MIN_RR:
        return None

    reason = f"Setup B: 15m robust range-edge rejection (q={ROBUST_LEVEL_Q:g}, lookback={LEVEL_LOOKBACK}) → 5m BOS confirm"
    return Signal(
        symbol=symbol, setup="B", direction=dir_, created_ts=now_ts(),
        score=score, entry_low=entry_low, entry_high=entry_high,
        stop=stop, tp1=tp1, tp2=tp2, reason=reason,
        spread=0.0, ob_imb=ob_imb, trend_dir=trend_dir,
        entry_mid=entry_mid, stop_pct=stop_pct, rr_to_tp1=rr_to_tp1
    )

# =========================
# Messaging
# =========================
def format_signal(sig: Signal, bid: float, ask: float) -> str:
    dir_ru = "ЛОНГ" if sig.direction == "LONG" else "ШОРТ"
    ob_line = ""
    if sig.ob_imb is not None:
        ob_line = f"• Стакан (imb): {sig.ob_imb:+.2f}\n"
    trend_line = f"• Тренд 1h: {sig.trend_dir}\n" if sig.trend_dir else "• Тренд 1h: нейтрально\n"

    hint = (
        f"Плечо x{LEVERAGE:.0f}, ставка ≈${NOTIONAL_USD:.0f}. "
        f"Стоп ≈{fmt_pct(sig.stop_pct,2)} по цене (структурный)."
    )

    return (
        f"🚦 <b>{sig.symbol}</b> | {dir_ru} | <b>15m</b> | Setup <b>{sig.setup}</b>\n"
        f"🧩 {sig.reason}\n"
        f"🧠 Score: {sig.score:.0f}/100\n\n"
        f"🎯 Вход (лимит): {fmt_price(sig.entry_low)} — {fmt_price(sig.entry_high)} (mid {fmt_price(sig.entry_mid)})\n"
        f"🛑 SL: {fmt_price(sig.stop)} ({fmt_pct(sig.stop_pct,2)})\n"
        f"✅ TP1: {fmt_price(sig.tp1)} | TP2: {fmt_price(sig.tp2)}\n"
        f"📏 RR до TP1: {sig.rr_to_tp1:.2f}\n\n"
        f"📌 Спред: {fmt_pct(spread_pct(bid, ask),3)}\n"
        f"{trend_line}"
        f"{ob_line}"
        f"⏳ TTL: {SIGNAL_TTL_SEC//60} мин | cooldown: {SYMBOL_COOLDOWN_SEC//60} мин\n\n"
        f"⚠️ Вход только при касании зоны, без догоняния.\n"
        f"💡 {hint}"
    )

def format_reminder(sig: Signal) -> str:
    dir_ru = "ЛОНГ" if sig.direction == "LONG" else "ШОРТ"
    ttl_left = max(0, SIGNAL_TTL_SEC - (now_ts() - sig.created_ts)) // 60
    return (
        f"⏰ Напоминание: <b>{sig.symbol}</b> {dir_ru} | Setup {sig.setup}\n"
        f"Зона: {fmt_price(sig.entry_low)} — {fmt_price(sig.entry_high)} | SL: {fmt_price(sig.stop)}\n"
        f"TTL осталось: {ttl_left} мин"
    )

def format_in_zone(sig: Signal, last: float) -> str:
    dir_ru = "ЛОНГ" if sig.direction == "LONG" else "ШОРТ"
    return (
        f"📍 Цена в зоне входа: <b>{sig.symbol}</b> {dir_ru} | Setup {sig.setup}\n"
        f"Last: {fmt_price(last)} | Зона: {fmt_price(sig.entry_low)} — {fmt_price(sig.entry_high)}\n"
        f"SL: {fmt_price(sig.stop)} | TP1: {fmt_price(sig.tp1)}"
    )

# =========================
# State
# =========================
cooldown: Dict[str, int] = {}          # symbol -> last_sent_ts
active: Dict[str, Signal] = {}         # symbol -> active signal

def cooldown_ok(symbol: str) -> bool:
    last = cooldown.get(symbol, 0)
    return (now_ts() - last) >= SYMBOL_COOLDOWN_SEC

def mark_sent(symbol: str) -> None:
    cooldown[symbol] = now_ts()

def in_entry_zone(sig: Signal, last: float) -> bool:
    return (sig.entry_low <= last <= sig.entry_high)

# =========================
# Candidate scan (stage-1)
# =========================
def scan_candidates(top: List[dict]) -> List[str]:
    """
    Cheap scan: for each symbol only pull 15m klines and detect candidate setups.
    Returns a list of symbols to run stage-2 on.
    """
    candidates: List[Tuple[int,str]] = []
    for t in top:
        sym = str(t.get("symbol"))
        bid = f(t.get("bid1Price"))
        ask = f(t.get("ask1Price"))
        sp = spread_pct(bid, ask)
        if sp > MAX_SPREAD_PCT:
            continue
        if sym in active:
            continue
        if not cooldown_ok(sym):
            continue

        try:
            c15 = parse_klines(get_kline(sym, TF_SIGNAL, max(LEVEL_LOOKBACK + 60, 140)))
        except Exception:
            continue

        score = 0
        if detect_setup_A_candidate(c15) is not None:
            score += 2
        if detect_setup_B_candidate(c15) is not None:
            score += 1

        if score >= CANDIDATE_SCORE_MIN:
            candidates.append((score, sym))

    # prioritize A candidates
    candidates.sort(reverse=True, key=lambda x: x[0])
    return [sym for _, sym in candidates[:MAX_CANDIDATES]]

# =========================
# Main loop
# =========================
def main() -> None:
    _log_init()

    tg_send(
        "✅ Bybit 15m Signals Bot (v2) запущен\n"
        f"Top={TOP_N}, poll={POLL_SECONDS}s | TF=15m (confirm 5m, trend 1h)\n"
        f"Плечо x{LEVERAGE:.0f} | Депо≈${EQUITY_USD:.0f} | Ставка≈${NOTIONAL_USD:.0f}\n"
        f"Сетапы: A (Impulse→Pullback→BOS), B (Robust Rejection). "
        f"Gates: stop≤{MAX_STOP_PCT:.2f}%, RR≥{MIN_RR:.1f}."
    )

    while True:
        try:
            # Background eval for logging (if enabled)
            _eval_pending()

            tickers = get_linear_tickers()
            top = pick_top_symbols(tickers, TOP_N)

            now = now_ts()

            # Clean expired actives
            for sym in list(active.keys()):
                sig = active[sym]
                if now - sig.created_ts > SIGNAL_TTL_SEC:
                    active.pop(sym, None)

            # Active signal UX: reminders + in-zone alerts
            for sym in list(active.keys()):
                t = next((x for x in top if str(x.get("symbol")) == sym), None)
                if not t:
                    continue
                last = f(t.get("lastPrice"))
                sig = active[sym]
                if SEND_IN_ZONE and (not sig.in_zone_notified) and in_entry_zone(sig, last):
                    if not in_quiet_hours():
                        tg_send(format_in_zone(sig, last))
                    sig.in_zone_notified = True
                    active[sym] = sig

                if sig.reminders_sent < MAX_REMINDERS and (now - sig.created_ts) >= REMIND_SEC * (sig.reminders_sent + 1):
                    if not in_quiet_hours():
                        tg_send(format_reminder(sig))
                    sig.reminders_sent += 1
                    active[sym] = sig

            # Stage-1 candidates
            candidates = scan_candidates(top)

            created = 0
            for sym in candidates:
                # Stage-2 data pulls
                t = next((x for x in top if str(x.get("symbol")) == sym), None)
                if not t:
                    continue
                bid = f(t.get("bid1Price"))
                ask = f(t.get("ask1Price"))
                last = f(t.get("lastPrice"))
                sp = spread_pct(bid, ask)
                if sp > MAX_SPREAD_PCT:
                    continue
                if sym in active or (not cooldown_ok(sym)):
                    continue

                try:
                    c15 = parse_klines(get_kline(sym, TF_SIGNAL, max(LEVEL_LOOKBACK + 60, 140)))
                    c5 = parse_klines(get_kline(sym, TF_CONFIRM, 180))
                except Exception:
                    continue

                try:
                    trend_dir = trend_filter_1h(sym)
                except Exception:
                    trend_dir = None

                ob_imb = orderbook_imbalance(sym)

                sig: Optional[Signal] = None

                # Prefer Setup A only when 1h trend is clear (higher quality)
                if trend_dir is not None:
                    sig = detect_setup_A(sym, c15, c5, trend_dir, ob_imb)

                if sig is None:
                    sig = detect_setup_B(sym, c15, c5, trend_dir, ob_imb)

                if sig is None:
                    continue

                # Final sanity: don't signal if price already moved too far away
                if last > 0:
                    if sig.direction == "LONG" and last > sig.entry_high * (1 + MAX_CHASE_PCT / 100.0):
                        continue
                    if sig.direction == "SHORT" and last < sig.entry_low * (1 - MAX_CHASE_PCT / 100.0):
                        continue

                sig.spread = sp

                if not in_quiet_hours():
                    tg_send(format_signal(sig, bid, ask))

                active[sym] = sig
                mark_sent(sym)
                created += 1

                # schedule logging evaluation
                _schedule_evals(sig)

            print(f"Tick: top={len(top)} cand={len(candidates)} created={created} active={len(active)} pending_eval={len(_pending_evals)}", flush=True)

        except Exception as e:
            print("ERROR", repr(e), flush=True)
            try:
                if not in_quiet_hours():
                    tg_send(f"⚠️ Ошибка бота: {repr(e)[:900]}")
            except Exception:
                pass

        time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    main()
