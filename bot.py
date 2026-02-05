#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bybit Alt Scalping Bot (Async) — Baseline v14 (Railway-safe)

Внедрено (14 пунктов, без переноса SL в BE):
1) Один корректный main()
2) Округления qty/price/SL/TP под шаги инструмента
3) Lifecycle лимиток: wait/TTL/cancel
4) Реальная реализация TP2 (через partial TP ордера)
5) Partials TP1+TP2 (reduceOnly), без переноса SL
6) Ротация кандидатов по TOP_N
7) positionIdx параметром (one-way/hedge)
8) Динамический SL на ATR (MAX_STOP_PCT как предохранитель)
9) ADX-фильтр
10) Дедупликация сигналов
11) Trend logic EMA20/EMA50 + slope
12) Daily risk guards: max trades/day + max daily risk budget (по планируемому риску)
13) Market Regime Detector: TREND/RANGE/HIGH_VOL (ADX+ATR%) + лёгкая адаптация фильтров
14) Orderbook walls: детект крупных стенок возле уровней (подтверждение lo/hi)

Railway:
- booleans в env только 0/1
- graceful shutdown (SIGINT/SIGTERM)
"""

import os
import json
import time
import math
import csv
import hmac
import hashlib
import asyncio
import logging
import signal
import urllib.parse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import aiohttp
import numpy as np
from cachetools import TTLCache
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


# ==============================
# ENV HELPERS (Railway-friendly)
# ==============================
def env_str(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return default if v is None or str(v).strip() == "" else str(v).strip()


def env_int(name: str, default: int) -> int:
    try:
        return int(env_str(name, str(default)))
    except Exception:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(env_str(name, str(default)))
    except Exception:
        return default


def env_bool_01(name: str, default: int = 0) -> bool:
    """Railway: используем только 0/1."""
    v = env_str(name, str(default))
    return v == "1"


# ==============================
# LOGGING
# ==============================
LOG_LEVEL = env_str("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("bybit-scalp")


# ==============================
# SHUTDOWN HANDLER
# ==============================
shutdown_event = asyncio.Event()


def _signal_handler(signum, frame):
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    shutdown_event.set()


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ==============================
# CONFIG
# ==============================
TG_TOKEN = env_str("TELEGRAM_BOT_TOKEN", "")
TG_CHAT_ID = env_str("TELEGRAM_CHAT_ID", "")

BYBIT_API_KEY = env_str("BYBIT_API_KEY", "")
BYBIT_API_SECRET = env_str("BYBIT_API_SECRET", "")
BYBIT_TESTNET = env_bool_01("BYBIT_TESTNET", 0)
BASE_URL = "https://api-testnet.bybit.com" if BYBIT_TESTNET else "https://api.bybit.com"
RECV_WINDOW = env_int("BYBIT_RECV_WINDOW", 5000)

AUTO_TRADE = env_bool_01("AUTO_TRADE", 0)
ENTRY_MODE = env_str("ENTRY_MODE", "auto").lower()  # auto|market|limit
SET_LEVERAGE = env_bool_01("SET_LEVERAGE", 0)
POSITION_IDX = env_int("POSITION_IDX", 0)  # 0 one-way, 1/2 hedge

NOTIONAL_USD = env_float("NOTIONAL_USD", 10.0)
LEVERAGE = env_int("LEVERAGE", 5)

TOP_N = env_int("TOP_N", 30)
MAX_CANDIDATES = env_int("MAX_CANDIDATES", 8)
POLL_SECONDS = env_int("POLL_SECONDS", 15)
SYMBOL_COOLDOWN_SEC = env_int("SYMBOL_COOLDOWN_SEC", 600)

TF_CONTEXT_MIN = env_int("TF_CONTEXT_MIN", 5)
TF_TREND_MIN = env_int("TF_TREND_MIN", 15)
TF_CONFIRM_MIN = env_int("TF_CONFIRM_MIN", 1)
CONFIRM_LOOKBACK = env_int("CONFIRM_LOOKBACK", 6)

ATR_LEN = env_int("ATR_LEN", 14)
ATR_MIN_PCT = env_float("ATR_MIN_PCT", 0.10)
ATR_MAX_PCT = env_float("ATR_MAX_PCT", 3.50)

LEVEL_LOOKBACK = env_int("LEVEL_LOOKBACK", 80)
ROBUST_LEVEL_Q = env_float("ROBUST_LEVEL_Q", 0.05)
LEVEL_TOL_PCT = env_float("LEVEL_TOL_PCT", 0.20)  # percent
REJECT_WICK_FRAC = env_float("REJECT_WICK_FRAC", 0.45)

# Orderbook filters
USE_ORDERBOOK = env_bool_01("USE_ORDERBOOK", 1)
OB_LIMIT = env_int("OB_LIMIT", 25)
OB_IMB_MIN = env_float("OB_IMB_MIN", 0.10)
MAX_SPREAD_PCT = env_float("MAX_SPREAD_PCT", 0.12)


# Flow / derivatives context filters (optional)
USE_FUNDING_FILTER = env_bool_01("USE_FUNDING_FILTER", 0)
FUNDING_MAX_LONG = env_float("FUNDING_MAX_LONG", 0.0006)   # 0.06%
FUNDING_MIN_SHORT = env_float("FUNDING_MIN_SHORT", -0.0006)

USE_OI_FILTER = env_bool_01("USE_OI_FILTER", 0)
OI_INTERVAL = os.getenv("OI_INTERVAL", "15min")            # 5min/15min/30min/1h/4h/1d
OI_LOOKBACK_MIN = env_int("OI_LOOKBACK_MIN", 30)           # lookback window in minutes
OI_MIN_DELTA_PCT = env_float("OI_MIN_DELTA_PCT", 1.0)      # require |ΔOI| >= X% to confirm

USE_CVD_FILTER = env_bool_01("USE_CVD_FILTER", 0)
CVD_WINDOW_SEC = env_int("CVD_WINDOW_SEC", 120)            # seconds
CVD_MIN_QUOTE = env_float("CVD_MIN_QUOTE", 5000.0)         # quote volume delta threshold
# Walls (optional)
ENABLE_WALL_FILTER = env_bool_01("ENABLE_WALL_FILTER", 0)
WALL_TOP_N = env_int("WALL_TOP_N", 10)
WALL_MULT = env_float("WALL_MULT", 5.0)
WALL_DIST_PCT = env_float("WALL_DIST_PCT", 0.10)  # within 0.10% of level

# Risk / RR
MAX_STOP_PCT = env_float("MAX_STOP_PCT", 1.0)  # safety cap
MIN_RR = env_float("MIN_RR", 1.2)
RR1 = env_float("RR1", 1.3)
RR2 = env_float("RR2", 1.8)

# ATR stop multiplier
ATR_STOP_MULT = env_float("ATR_STOP_MULT", 1.5)

# ADX / Regime
ADX_LEN = env_int("ADX_LEN", 14)
ADX_MIN = env_float("ADX_MIN", 20.0)  # below => skip
REGIME_TREND_ADX = env_float("REGIME_TREND_ADX", 25.0)
REGIME_HIGHVOL_ATR_PCT = env_float("REGIME_HIGHVOL_ATR_PCT", 1.50)

# Regime adaptations (light)
RANGE_MIN_RR = env_float("RANGE_MIN_RR", 1.4)
RANGE_REJECT_WICK_FRAC = env_float("RANGE_REJECT_WICK_FRAC", 0.50)
RANGE_CONFIRM_LOOKBACK = env_int("RANGE_CONFIRM_LOOKBACK", 8)

HIGHVOL_MIN_RR = env_float("HIGHVOL_MIN_RR", 1.3)
HIGHVOL_ATR_STOP_MULT = env_float("HIGHVOL_ATR_STOP_MULT", 1.8)

TREND_MIN_RR = env_float("TREND_MIN_RR", 1.2)

# Signal lifecycle
SIGNAL_TTL_SEC = env_int("SIGNAL_TTL_SEC", 900)
SIGNAL_DEDUP_SEC = env_int("SIGNAL_DEDUP_SEC", 60)

# Limit order lifecycle
LIMIT_WAIT_SEC = env_int("LIMIT_WAIT_SEC", 25)
LIMIT_POLL_SEC = env_int("LIMIT_POLL_SEC", 2)

# Partials (no SL move)
ENABLE_PARTIAL_TP = env_bool_01("ENABLE_PARTIAL_TP", 1)
TP1_SHARE = env_float("TP1_SHARE", 0.50)

# Daily risk guards
USE_DAILY_GUARDS = env_bool_01("USE_DAILY_GUARDS", 1)
MAX_TRADES_PER_DAY = env_int("MAX_TRADES_PER_DAY", 10)
MAX_DAILY_RISK_USD = env_float("MAX_DAILY_RISK_USD", 50.0)

# Logging
ENABLE_SIGNAL_LOG = env_bool_01("ENABLE_SIGNAL_LOG", 1)
SIGNAL_LOG_FILE = env_str("SIGNAL_LOG_FILE", "signals_log.csv")

# Universe
ALLOWED_CATEGORY = env_str("ALLOWED_CATEGORY", "linear")
ALLOWED_QUOTE = env_str("ALLOWED_QUOTE", "USDT")


# ==============================
# STATE / CACHES
# ==============================
KLINE_CACHE = TTLCache(maxsize=800, ttl=60)
OB_RAW_CACHE = TTLCache(maxsize=800, ttl=30)
TREND_CACHE = TTLCache(maxsize=800, ttl=180)
FUNDING_CACHE = TTLCache(maxsize=800, ttl=300)
OI_CACHE = TTLCache(maxsize=800, ttl=60)
CVD_CACHE = TTLCache(maxsize=800, ttl=5)
INSTR_CACHE: Dict[str, Dict[str, float]] = {}

COOLDOWN: Dict[str, float] = {}
RECENT_SIGNALS: Dict[str, int] = {}  # key -> unix ts

_ROT_OFFSET = 0  # candidate rotation

_daily_date = None
_daily_trades = 0
_daily_risk_used = 0.0


# ==============================
# DATA
# ==============================
@dataclass
class Candle:
    ts: int
    o: float
    h: float
    l: float
    c: float
    v: float


@dataclass
class Signal:
    signal_id: str
    ts: int
    symbol: str
    side: str
    setup: str
    entry_mid: float
    stop: float
    tp1: float
    tp2: float
    ttl_sec: int
    reason: str
    atr_pct: float
    adx: float
    spread_pct: float
    ob_imb: float
    regime: str
    bid_wall: int
    ask_wall: int
    funding_rate: float = 0.0
    oi_delta_pct: float = 0.0
    cvd_delta_quote: float = 0.0


# ==============================
# UTILS
# ==============================
def now_ms() -> int:
    return int(time.time() * 1000)


def utc_date_str() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())


def robust_range(prices: List[float], q: float) -> Tuple[float, float]:
    if not prices:
        return 0.0, 0.0
    ps = sorted(prices)
    lo_i = int(max(0, min(len(ps) - 1, int(len(ps) * q))))
    hi_i = int(max(0, min(len(ps) - 1, int(len(ps) * (1 - q)) - 1)))
    if hi_i < lo_i:
        hi_i = lo_i
    return ps[lo_i], ps[hi_i]


def wick_fraction(c: Candle) -> float:
    rng = max(1e-12, c.h - c.l)
    body = abs(c.c - c.o)
    return (rng - body) / rng


def candle_rejection_at_edge(c: Candle, edge: float, is_low_edge: bool, wick_min: float) -> bool:
    if wick_fraction(c) < wick_min:
        return False
    if is_low_edge:
        return (c.l <= edge) and (c.c > edge)
    return (c.h >= edge) and (c.c < edge)


def ema(values: List[float], period: int) -> List[float]:
    if not values or period <= 1:
        return values[:]
    k = 2.0 / (period + 1.0)
    out = []
    e = values[0]
    out.append(e)
    for v in values[1:]:
        e = e + k * (v - e)
        out.append(e)
    return out


def round_step(x: float, step: float, mode: str = "down") -> float:
    if step <= 0:
        return x
    k = x / step
    return (math.ceil(k) if mode == "up" else math.floor(k)) * step


def fmt_qty(q: float) -> str:
    return f"{q:.8f}".rstrip("0").rstrip(".")


def fmt_price(p: float) -> str:
    return f"{p:.8f}".rstrip("0").rstrip(".")


def should_cooldown(symbol: str) -> bool:
    return time.time() < COOLDOWN.get(symbol, 0.0)


def set_cooldown(symbol: str) -> None:
    COOLDOWN[symbol] = time.time() + SYMBOL_COOLDOWN_SEC


def can_generate_signal(symbol: str, side: str) -> bool:
    key = f"{symbol}_{side}"
    last_ts = RECENT_SIGNALS.get(key, 0)
    now_s = int(time.time())
    if now_s - last_ts < SIGNAL_DEDUP_SEC:
        return False
    RECENT_SIGNALS[key] = now_s
    return True


def _reset_daily_guards_if_needed() -> None:
    global _daily_date, _daily_trades, _daily_risk_used
    d = utc_date_str()
    if _daily_date != d:
        _daily_date = d
        _daily_trades = 0
        _daily_risk_used = 0.0


def daily_guards_allow(risk_usd: float) -> bool:
    if not USE_DAILY_GUARDS:
        return True
    _reset_daily_guards_if_needed()
    if _daily_trades >= MAX_TRADES_PER_DAY:
        return False
    if _daily_risk_used + risk_usd > MAX_DAILY_RISK_USD:
        return False
    return True


def daily_guards_commit(risk_usd: float) -> None:
    global _daily_trades, _daily_risk_used
    if not USE_DAILY_GUARDS:
        return
    _reset_daily_guards_if_needed()
    _daily_trades += 1
    _daily_risk_used += risk_usd


# ==============================
# INDICATORS
# ==============================
def atr_value(candles: List[Candle], length: int) -> float:
    if len(candles) < length + 1:
        return 0.0
    highs = np.array([c.h for c in candles], dtype=float)
    lows = np.array([c.l for c in candles], dtype=float)
    closes = np.array([c.c for c in candles], dtype=float)
    tr1 = highs[1:] - lows[1:]
    tr2 = np.abs(highs[1:] - closes[:-1])
    tr3 = np.abs(lows[1:] - closes[:-1])
    tr = np.maximum.reduce([tr1, tr2, tr3])
    return float(np.mean(tr[-length:]))


def atr_pct_fast(candles: List[Candle], length: int) -> float:
    atr = atr_value(candles, length)
    price = float(candles[-1].c) if candles else 0.0
    return (atr / price) * 100.0 if price > 0 else 0.0


def adx_value(candles: List[Candle], length: int) -> float:
    """Correct Wilder ADX (last value), bounded to 0..100.

    Uses Wilder's RMA (EMA with alpha=1/n) for TR, +DM, -DM and DX.
    """
    if len(candles) < length + 2:
        return 0.0

    highs = np.array([c.h for c in candles], dtype=float)
    lows = np.array([c.l for c in candles], dtype=float)
    closes = np.array([c.c for c in candles], dtype=float)

    up = highs[1:] - highs[:-1]
    down = lows[:-1] - lows[1:]

    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    tr1 = highs[1:] - lows[1:]
    tr2 = np.abs(highs[1:] - closes[:-1])
    tr3 = np.abs(lows[1:] - closes[:-1])
    tr = np.maximum.reduce([tr1, tr2, tr3])

    def wilder_rma(x: np.ndarray, n: int) -> np.ndarray:
        out = np.full_like(x, np.nan, dtype=float)
        if len(x) < n:
            return out
        out[n - 1] = np.mean(x[:n])
        alpha = 1.0 / n
        for i in range(n, len(x)):
            out[i] = out[i - 1] + alpha * (x[i] - out[i - 1])
        return out

    atr = wilder_rma(tr, length)
    p_dm = wilder_rma(plus_dm, length)
    m_dm = wilder_rma(minus_dm, length)

    with np.errstate(divide="ignore", invalid="ignore"):
        p_di = 100.0 * (p_dm / atr)
        m_di = 100.0 * (m_dm / atr)
        denom = (p_di + m_di)
        dx = 100.0 * np.abs(p_di - m_di) / denom

    dx = np.nan_to_num(dx, nan=0.0, posinf=0.0, neginf=0.0)
    adx = wilder_rma(dx, length)

    last = float(adx[-1]) if len(adx) else 0.0
    if not np.isfinite(last):
        return 0.0
    return float(max(0.0, min(100.0, last)))



def detect_regime(adx: float, atrp: float) -> str:
    if atrp >= REGIME_HIGHVOL_ATR_PCT:
        return "HIGH_VOL"
    if adx >= REGIME_TREND_ADX:
        return "TREND"
    return "RANGE"


# ==============================
# TELEGRAM
# ==============================
async def tg_send(session: aiohttp.ClientSession, text: str) -> None:
    if not TG_TOKEN or not TG_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        async with session.post(
            url,
            json={"chat_id": TG_CHAT_ID, "text": text, "disable_web_page_preview": True},
            timeout=aiohttp.ClientTimeout(total=10),
        ):
            return
    except Exception as e:
        logger.warning(f"Telegram send failed: {e}")


# ==============================
# BYBIT API (v5 signType=2, Railway-safe)
# ==============================
def _build_query(params: dict) -> str:
    """Stable query string for signing (sorted, urlencode)."""
    if not params:
        return ""
    return urllib.parse.urlencode(sorted(params.items()))

def sign_bybit_v5(api_key: str, api_secret: str, timestamp: str, recv_window: str, payload: str) -> str:
    """Bybit v5 signature (signType=2): sha256_hmac(timestamp+api_key+recv_window+payload)."""
    msg = f"{timestamp}{api_key}{recv_window}{payload}"
    return hmac.new(api_secret.encode(), msg.encode(), hashlib.sha256).hexdigest()

def _headers_v5(signature: str, timestamp: str, recv_window: str) -> dict:
    return {
        "Content-Type": "application/json",
        "X-BAPI-SIGN": signature,
        "X-BAPI-API-KEY": BYBIT_API_KEY,
        "X-BAPI-TIMESTAMP": timestamp,
        "X-BAPI-RECV-WINDOW": recv_window,
        "X-BAPI-SIGN-TYPE": "2",
    }

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
)
async def bybit_get(session: aiohttp.ClientSession, endpoint: str, params: dict) -> dict:
    timestamp = str(now_ms())
    recv_window = str(RECV_WINDOW)
    query = _build_query(params)
    sign = sign_bybit_v5(BYBIT_API_KEY, BYBIT_API_SECRET, timestamp, recv_window, query)
    headers = _headers_v5(sign, timestamp, recv_window)

    url = f"{BASE_URL}{endpoint}"
    async with session.get(url, params=params, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
        data = await resp.json()
        if data.get("retCode") != 0:
            raise ValueError(f"API error: {data}")
        return data.get("result", {})

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
)
async def bybit_post(session: aiohttp.ClientSession, endpoint: str, params: dict) -> dict:
    timestamp = str(now_ms())
    recv_window = str(RECV_WINDOW)

    # IMPORTANT: payload for signing must be the exact JSON string that is sent
    body_json = json.dumps(params or {}, separators=(",", ":"), ensure_ascii=False)

    sign = sign_bybit_v5(BYBIT_API_KEY, BYBIT_API_SECRET, timestamp, recv_window, body_json)
    headers = _headers_v5(sign, timestamp, recv_window)

    url = f"{BASE_URL}{endpoint}"
    async with session.post(url, data=body_json, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
        data = await resp.json()
        if data.get("retCode") != 0:
            raise ValueError(f"API error: {data}")
        return data.get("result", {})


# ==============================
# MARKET DATA

# ==============================
async def bybit_tickers_top(session: aiohttp.ClientSession, n: int) -> List[dict]:
    try:
        url = f"{BASE_URL}/v5/market/tickers"
        params = {"category": ALLOWED_CATEGORY}
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            data = await resp.json()
            if data.get("retCode") != 0:
                return []
            items = data.get("result", {}).get("list", []) or []
            valid = [
                t for t in items
                if str(t.get("symbol", "")).endswith(ALLOWED_QUOTE) and float(t.get("turnover24h", 0) or 0) > 0
            ]
            valid.sort(key=lambda x: float(x.get("turnover24h", 0) or 0), reverse=True)
            return valid[:n]
    except Exception as e:
        logger.error(f"Failed to fetch tickers: {e}")
        return []


async def bybit_kline(session: aiohttp.ClientSession, symbol: str, interval_min: int, limit: int) -> List:
    key = f"kline_{symbol}_{interval_min}_{limit}"
    if key in KLINE_CACHE:
        return KLINE_CACHE[key]
    try:
        url = f"{BASE_URL}/v5/market/kline"
        params = {"category": ALLOWED_CATEGORY, "symbol": symbol, "interval": str(interval_min), "limit": str(limit)}
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            data = await resp.json()
            if data.get("retCode") != 0:
                return []
            rows = data.get("result", {}).get("list", []) or []
            candles = [
                Candle(ts=int(r[0]), o=float(r[1]), h=float(r[2]), l=float(r[3]), c=float(r[4]), v=float(r[5]))
                for r in reversed(rows)
            ]
            KLINE_CACHE[key] = candles
            return candles
    except Exception as e:
        logger.error(f"Failed to fetch klines for {symbol}: {e}")
        return []


async def bybit_orderbook_raw(session: aiohttp.ClientSession, symbol: str, limit: int) -> Tuple[List[List[str]], List[List[str]]]:
    key = f"obraw_{symbol}_{limit}"
    if key in OB_RAW_CACHE:
        return OB_RAW_CACHE[key]
    try:
        url = f"{BASE_URL}/v5/market/orderbook"
        params = {"category": ALLOWED_CATEGORY, "symbol": symbol, "limit": str(limit)}
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            data = await resp.json()
            if data.get("retCode") != 0:
                return [], []
            ob = data.get("result", {}) or {}
            bids = ob.get("b", []) or []
            asks = ob.get("a", []) or []
            OB_RAW_CACHE[key] = (bids, asks)
            return bids, asks
    except Exception as e:
        logger.error(f"Failed to fetch orderbook for {symbol}: {e}")
        return [], []

async def bybit_funding_rate(session: aiohttp.ClientSession, symbol: str) -> float:
    """Latest funding rate (float). Uses /v5/market/history-fund-rate."""
    ck = f"fund:{symbol}"
    if ck in FUNDING_CACHE:
        return FUNDING_CACHE[ck]
    params = {"category": ALLOWED_CATEGORY, "symbol": symbol, "limit": 1}
    res = await bybit_get(session, "/v5/market/history-fund-rate", params)
    # result.list: [{fundingRate, fundingRateTimestamp, ...}]
    lst = res.get("list") or []
    fr = 0.0
    if lst:
        try:
            fr = float(lst[0].get("fundingRate", 0.0))
        except Exception:
            fr = 0.0
    FUNDING_CACHE[ck] = fr
    return fr

async def bybit_open_interest_delta(session: aiohttp.ClientSession, symbol: str) -> float:
    """Open interest % change over the last OI_LOOKBACK_MIN (float, can be negative)."""
    ck = f"oi:{symbol}:{OI_INTERVAL}:{OI_LOOKBACK_MIN}"
    if ck in OI_CACHE:
        return OI_CACHE[ck]
    end = now_ms()
    start = end - int(OI_LOOKBACK_MIN * 60 * 1000)
    params = {
        "category": ALLOWED_CATEGORY,
        "symbol": symbol,
        "intervalTime": OI_INTERVAL,
        "startTime": start,
        "endTime": end,
        "limit": 2,
    }
    res = await bybit_get(session, "/v5/market/open-interest", params)
    lst = res.get("list") or []
    delta_pct = 0.0
    if len(lst) >= 2:
        # list is usually newest-first; handle both
        try:
            oi_vals = []
            for it in lst:
                oi_vals.append(float(it.get("openInterest", 0.0)))
            oi_old = oi_vals[-1]
            oi_new = oi_vals[0]
            if oi_old > 0:
                delta_pct = (oi_new - oi_old) / oi_old * 100.0
        except Exception:
            delta_pct = 0.0
    OI_CACHE[ck] = delta_pct
    return delta_pct

async def bybit_cvd_delta_quote(session: aiohttp.ClientSession, symbol: str) -> float:
    """CVD-like delta using recent public trades (quote volume buy - sell) over CVD_WINDOW_SEC."""
    ck = f"cvd:{symbol}:{CVD_WINDOW_SEC}"
    if ck in CVD_CACHE:
        return CVD_CACHE[ck]
    params = {"category": ALLOWED_CATEGORY, "symbol": symbol, "limit": 200}
    res = await bybit_get(session, "/v5/market/recent-trade", params)
    lst = res.get("list") or []
    cutoff = now_ms() - int(CVD_WINDOW_SEC * 1000)
    buy_q = 0.0
    sell_q = 0.0
    for t in lst:
        try:
            ts = int(t.get("time", 0))
            if ts < cutoff:
                continue
            side = t.get("side", "")
            price = float(t.get("price", 0.0))
            size = float(t.get("size", 0.0))
            q = price * size
            if side == "Buy":
                buy_q += q
            else:
                sell_q += q
        except Exception:
            continue
    delta = buy_q - sell_q
    CVD_CACHE[ck] = delta
    return delta

def passes_funding_filter(side: str, funding_rate: float) -> bool:
    if not USE_FUNDING_FILTER:
        return True
    if side == "Buy":
        return funding_rate <= FUNDING_MAX_LONG
    else:
        return funding_rate >= FUNDING_MIN_SHORT

def passes_oi_filter(side: str, oi_delta_pct: float) -> bool:
    if not USE_OI_FILTER:
        return True
    # require absolute delta
    if abs(oi_delta_pct) < OI_MIN_DELTA_PCT:
        return False
    # optional directional sanity: in strong continuation, OI often grows with move
    # We keep it light: just require magnitude unless you want directional constraints.
    return True

def passes_cvd_filter(side: str, cvd_delta_quote: float) -> bool:
    if not USE_CVD_FILTER:
        return True
    if side == "Buy":
        return cvd_delta_quote >= CVD_MIN_QUOTE
    else:
        return cvd_delta_quote <= -CVD_MIN_QUOTE

def _wall_near_level(level: float, rows: List[List[str]], top_n: int) -> int:
    if not ENABLE_WALL_FILTER or top_n <= 0 or level <= 0:
        return 0
    sl = rows[:min(top_n, len(rows))]
    if not sl:
        return 0
    sizes = np.array([float(x[1]) for x in sl], dtype=float)
    avg = float(np.mean(sizes)) if len(sizes) else 0.0
    if avg <= 0:
        return 0
    for p_str, s_str in sl:
        p = float(p_str)
        s = float(s_str)
        if s >= avg * WALL_MULT:
            dist_pct = abs(p - level) / level * 100.0
            if dist_pct <= WALL_DIST_PCT:
                return 1
    return 0


# ==============================
# INSTRUMENT INFO / SIZING
# ==============================
async def fetch_instrument_info(session: aiohttp.ClientSession, symbol: str) -> Dict[str, float]:
    if symbol in INSTR_CACHE:
        return INSTR_CACHE[symbol]
    try:
        url = f"{BASE_URL}/v5/market/instruments-info"
        params = {"category": ALLOWED_CATEGORY, "symbol": symbol}
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            data = await resp.json()
            if data.get("retCode") != 0:
                return {}
            items = data.get("result", {}).get("list", []) or []
            if not items:
                return {}
            info = items[0]
            lot = info.get("lotSizeFilter", {}) or {}
            pricef = info.get("priceFilter", {}) or {}
            result = {
                "qty_step": float(lot.get("qtyStep", 0.001)),
                "min_qty": float(lot.get("minOrderQty", 0.001)),
                "price_tick": float(pricef.get("tickSize", 0.01)),
            }
            INSTR_CACHE[symbol] = result
            return result
    except Exception as e:
        logger.error(f"Failed to fetch instrument info for {symbol}: {e}")
        return {}


async def compute_qty_float(session: aiohttp.ClientSession, symbol: str, price: float) -> Tuple[float, float, float]:
    info = await fetch_instrument_info(session, symbol)
    qty_step = info.get("qty_step", 0.001)
    min_qty = info.get("min_qty", 0.001)
    raw_qty = (NOTIONAL_USD * LEVERAGE) / max(price, 1e-12)
    qty = round_step(raw_qty, qty_step, "down")
    qty = max(qty, min_qty)
    return float(qty), float(qty_step), float(min_qty)


# ==============================
# ORDERS
# ==============================
async def set_leverage_if_needed(session: aiohttp.ClientSession, symbol: str, leverage: int) -> None:
    if not SET_LEVERAGE:
        return
    try:
        await bybit_post(
            session,
            "/v5/position/set-leverage",
            {"category": ALLOWED_CATEGORY, "symbol": symbol, "buyLeverage": str(leverage), "sellLeverage": str(leverage)},
        )
        logger.info(f"Leverage set to {leverage}x for {symbol}")
    except Exception as e:
        logger.warning(f"Failed to set leverage for {symbol}: {e}")


async def has_open_position(session: aiohttp.ClientSession) -> bool:
    try:
        result = await bybit_get(session, "/v5/position/list", {"category": ALLOWED_CATEGORY, "settleCoin": ALLOWED_QUOTE})
        positions = result.get("list", []) or []
        return any(float(p.get("size", 0) or 0) > 0 for p in positions)
    except Exception as e:
        logger.warning(f"Failed to check positions: {e}")
        return False


async def get_order_status(session: aiohttp.ClientSession, symbol: str, order_id: str) -> str:
    try:
        res = await bybit_get(session, "/v5/order/realtime", {"category": ALLOWED_CATEGORY, "symbol": symbol, "orderId": order_id})
        items = res.get("list", []) or []
        if not items:
            return "Unknown"
        return str(items[0].get("orderStatus", "Unknown"))
    except Exception as e:
        logger.warning(f"Order status failed: {e}")
        return "Unknown"


async def cancel_order(session: aiohttp.ClientSession, symbol: str, order_id: str) -> bool:
    try:
        await bybit_post(session, "/v5/order/cancel", {"category": ALLOWED_CATEGORY, "symbol": symbol, "orderId": order_id})
        return True
    except Exception as e:
        logger.warning(f"Cancel failed: {e}")
        return False


async def place_entry_order(
    session: aiohttp.ClientSession,
    symbol: str,
    side: str,
    qty: float,
    entry_price: Optional[float],
    stop: float,
) -> Tuple[bool, str]:
    try:
        info = await fetch_instrument_info(session, symbol)
        tick = info.get("price_tick", 0.01)

        sl_mode = "down" if side == "Buy" else "up"
        stop_r = round_step(stop, tick, sl_mode)

        params = {
            "category": ALLOWED_CATEGORY,
            "symbol": symbol,
            "side": side,
            "orderType": "Market" if entry_price is None else "Limit",
            "qty": fmt_qty(qty),
            "stopLoss": fmt_price(stop_r),
            "timeInForce": "GTC",
            "positionIdx": POSITION_IDX,
        }

        if entry_price is not None:
            p_mode = "down" if side == "Buy" else "up"
            price_r = round_step(entry_price, tick, p_mode)
            params["price"] = fmt_price(price_r)

        result = await bybit_post(session, "/v5/order/create", params)
        oid = str(result.get("orderId", ""))
        return (oid != ""), oid
    except Exception as e:
        return False, str(e)


async def place_tp_order(
    session: aiohttp.ClientSession,
    symbol: str,
    side: str,
    qty: float,
    price: float,
) -> Tuple[bool, str]:
    try:
        info = await fetch_instrument_info(session, symbol)
        tick = info.get("price_tick", 0.01)

        p_mode = "up" if side == "Sell" else "down"
        price_r = round_step(price, tick, p_mode)

        params = {
            "category": ALLOWED_CATEGORY,
            "symbol": symbol,
            "side": side,
            "orderType": "Limit",
            "qty": fmt_qty(qty),
            "price": fmt_price(price_r),
            "timeInForce": "GTC",
            "reduceOnly": True,
            "positionIdx": POSITION_IDX,
        }
        result = await bybit_post(session, "/v5/order/create", params)
        oid = str(result.get("orderId", ""))
        return (oid != ""), oid
    except Exception as e:
        return False, str(e)


async def place_partials_if_enabled(session: aiohttp.ClientSession, sig: Signal, qty: float, qty_step: float, min_qty: float) -> None:
    if not ENABLE_PARTIAL_TP:
        return

    share = min(max(TP1_SHARE, 0.0), 1.0)
    q1 = round_step(qty * share, qty_step, "down")
    q2 = round_step(qty - q1, qty_step, "down")

    if q1 < min_qty or q2 < min_qty:
        q1 = qty
        q2 = 0.0

    opp = "Sell" if sig.side == "Buy" else "Buy"

    ok1, r1 = await place_tp_order(session, sig.symbol, opp, q1, sig.tp1)
    if ok1:
        logger.info(f"TP1 reduceOnly placed: {sig.symbol} qty={fmt_qty(q1)} oid={r1}")
    else:
        logger.warning(f"TP1 place failed: {sig.symbol} | {r1}")

    if q2 > 0:
        ok2, r2 = await place_tp_order(session, sig.symbol, opp, q2, sig.tp2)
        if ok2:
            logger.info(f"TP2 reduceOnly placed: {sig.symbol} qty={fmt_qty(q2)} oid={r2}")
        else:
            logger.warning(f"TP2 place failed: {sig.symbol} | {r2}")


# ==============================
# SIGNAL DETECTION
# ==============================
async def trend_filter(session: aiohttp.ClientSession, symbol: str) -> Tuple[int, float]:
    key = f"trend_{symbol}_{TF_TREND_MIN}"
    if key in TREND_CACHE:
        return TREND_CACHE[key]

    candles = await bybit_kline(session, symbol, TF_TREND_MIN, 60)
    if len(candles) < 60:
        TREND_CACHE[key] = (0, 0.0)
        return 0, 0.0

    closes = [c.c for c in candles]
    e_fast = ema(closes, 20)
    e_slow = ema(closes, 50)

    def slope(series: List[float], k: int = 5) -> float:
        if len(series) < k + 1 or series[-k] == 0:
            return 0.0
        return (series[-1] - series[-k]) / series[-k]

    current_fast = e_fast[-1]
    current_slow = e_slow[-1]
    fast_slope = slope(e_fast, 5)
    slow_slope = slope(e_slow, 5)
    slope_score = float(abs(fast_slope) + abs(slow_slope))

    if current_fast > current_slow and fast_slope > 0 and slow_slope > 0:
        out = (1, slope_score)
    elif current_fast < current_slow and fast_slope < 0 and slow_slope < 0:
        out = (-1, slope_score)
    else:
        out = (0, slope_score)

    TREND_CACHE[key] = out
    return out


def confirm_bos(candles: List[Candle], side: str, lookback: int) -> bool:
    if len(candles) < lookback:
        return False
    closes = [c.c for c in candles[-lookback:]]
    last = closes[-1]
    prev = closes[:-1]
    if side == "Buy":
        return last > max(prev)
    return last < min(prev)


async def detect_signal(session: aiohttp.ClientSession, symbol: str) -> Optional[Signal]:
    need = max(LEVEL_LOOKBACK, ATR_LEN) + 10
    ctx = await bybit_kline(session, symbol, TF_CONTEXT_MIN, need)
    if len(ctx) < need - 2:
        return None

    atrp = atr_pct_fast(ctx, ATR_LEN)
    if atrp < ATR_MIN_PCT or atrp > ATR_MAX_PCT:
        return None

    closes = [c.c for c in ctx[-LEVEL_LOOKBACK:]]
    lo, hi = robust_range(closes, ROBUST_LEVEL_Q)
    if lo <= 0 or hi <= 0 or hi <= lo:
        return None

    last = ctx[-1]
    mid = last.c

    trend, slope_score = await trend_filter(session, symbol)
    trend_candles = await bybit_kline(session, symbol, TF_TREND_MIN, 80)
    adx = adx_value(trend_candles, ADX_LEN) if trend_candles else 0.0
    regime = detect_regime(adx, atrp)

    wick_min = REJECT_WICK_FRAC
    min_rr_eff = MIN_RR
    confirm_lb = CONFIRM_LOOKBACK
    atr_mult = ATR_STOP_MULT

    if regime == "RANGE":
        if adx < ADX_MIN:
            return None
        wick_min = max(wick_min, RANGE_REJECT_WICK_FRAC)
        min_rr_eff = max(min_rr_eff, RANGE_MIN_RR)
        confirm_lb = max(confirm_lb, RANGE_CONFIRM_LOOKBACK)
    elif regime == "HIGH_VOL":
        min_rr_eff = max(min_rr_eff, HIGHVOL_MIN_RR)
        atr_mult = max(atr_mult, HIGHVOL_ATR_STOP_MULT)
    else:
        min_rr_eff = max(min_rr_eff, TREND_MIN_RR)

    tol = (LEVEL_TOL_PCT / 100.0) * mid
    near_low = (abs(mid - lo) <= tol) or (last.l <= lo + tol)
    near_high = (abs(mid - hi) <= tol) or (last.h >= hi - tol)

    conf = await bybit_kline(session, symbol, TF_CONFIRM_MIN, max(confirm_lb + 2, 20))
    if len(conf) < confirm_lb:
        return None

    spread_pct = 0.0
    imb = 0.0
    bid_wall = 0
    ask_wall = 0
    if USE_ORDERBOOK:
        bids, asks = await bybit_orderbook_raw(session, symbol, OB_LIMIT)
        if not bids or not asks:
            return None
        bid1 = float(bids[0][0])
        ask1 = float(asks[0][0])
        if bid1 <= 0 or ask1 <= 0:
            return None

        spread_pct = ((ask1 - bid1) / ((ask1 + bid1) / 2.0)) * 100.0
        if spread_pct > MAX_SPREAD_PCT:
            return None

        bid_vol = sum(float(b[1]) for b in bids)
        ask_vol = sum(float(a[1]) for a in asks)
        total = bid_vol + ask_vol
        imb = (bid_vol - ask_vol) / total if total > 0 else 0.0

        bid_wall = _wall_near_level(lo, bids, WALL_TOP_N)
        ask_wall = _wall_near_level(hi, asks, WALL_TOP_N)

    atr_abs = atr_value(ctx, ATR_LEN)
    if atr_abs <= 0:
        return None

    if near_low and candle_rejection_at_edge(last, lo, True, wick_min):
        if not can_generate_signal(symbol, "Buy"):
            return None

        look = max(confirm_lb, 7) if trend == -1 else confirm_lb
        if not confirm_bos(conf, "Buy", look):
            return None

        if USE_ORDERBOOK and (not USE_CVD_FILTER) and imb < OB_IMB_MIN:
            return None
        if ENABLE_WALL_FILTER and bid_wall == 0:
            return None

        stop = min(last.l, lo) - (atr_abs * atr_mult)
        stop_pct = ((mid - stop) / mid) * 100.0
        if stop_pct <= 0 or stop_pct > MAX_STOP_PCT:
            return None

        risk = mid - stop
        tp1 = mid + risk * RR1
        tp2 = mid + risk * RR2
        rr = (tp1 - mid) / risk if risk > 0 else 0.0
        if rr < min_rr_eff:
            return None

        funding_rate = 0.0
        oi_delta_pct = 0.0
        cvd_delta_quote = 0.0

        if USE_FUNDING_FILTER:
            funding_rate = await bybit_funding_rate(session, symbol)
            if not passes_funding_filter("Buy", funding_rate):
                return None

        if USE_OI_FILTER:
            oi_delta_pct = await bybit_open_interest_delta(session, symbol)
            if not passes_oi_filter("Buy", oi_delta_pct):
                return None

        if USE_CVD_FILTER:
            cvd_delta_quote = await bybit_cvd_delta_quote(session, symbol)
            if not passes_cvd_filter("Buy", cvd_delta_quote):
                return None

        return Signal(
            signal_id=f"{symbol}_{now_ms()}",
            ts=now_ms(),
            symbol=symbol,
            side="Buy",
            setup="reject_low",
            entry_mid=mid,
            stop=stop,
            tp1=tp1,
            tp2=tp2,
            ttl_sec=SIGNAL_TTL_SEC,
            reason=f"lo={lo:.6g} atr%={atrp:.2f} adx={adx:.1f} reg={regime} tr={trend} imb={imb:.3f} slope={slope_score:.4f}",
            atr_pct=atrp,
            adx=adx,
            spread_pct=spread_pct,
            ob_imb=imb,
            regime=regime,
            bid_wall=bid_wall,
            ask_wall=ask_wall,
            funding_rate=funding_rate,
            oi_delta_pct=oi_delta_pct,
            cvd_delta_quote=cvd_delta_quote,
        )

    if near_high and candle_rejection_at_edge(last, hi, False, wick_min):
        if not can_generate_signal(symbol, "Sell"):
            return None

        look = max(confirm_lb, 7) if trend == 1 else confirm_lb
        if not confirm_bos(conf, "Sell", look):
            return None

        if USE_ORDERBOOK and (not USE_CVD_FILTER) and imb > -OB_IMB_MIN:
            return None
        if ENABLE_WALL_FILTER and ask_wall == 0:
            return None

        stop = max(last.h, hi) + (atr_abs * atr_mult)
        stop_pct = ((stop - mid) / mid) * 100.0
        if stop_pct <= 0 or stop_pct > MAX_STOP_PCT:
            return None

        risk = stop - mid
        tp1 = mid - risk * RR1
        tp2 = mid - risk * RR2
        rr = (mid - tp1) / risk if risk > 0 else 0.0
        if rr < min_rr_eff:
            return None

        funding_rate = 0.0
        oi_delta_pct = 0.0
        cvd_delta_quote = 0.0

        if USE_FUNDING_FILTER:
            funding_rate = await bybit_funding_rate(session, symbol)
            if not passes_funding_filter("Sell", funding_rate):
                return None

        if USE_OI_FILTER:
            oi_delta_pct = await bybit_open_interest_delta(session, symbol)
            if not passes_oi_filter("Sell", oi_delta_pct):
                return None

        if USE_CVD_FILTER:
            cvd_delta_quote = await bybit_cvd_delta_quote(session, symbol)
            if not passes_cvd_filter("Sell", cvd_delta_quote):
                return None

        return Signal(
            signal_id=f"{symbol}_{now_ms()}",
            ts=now_ms(),
            symbol=symbol,
            side="Sell",
            setup="reject_high",
            entry_mid=mid,
            stop=stop,
            tp1=tp1,
            tp2=tp2,
            ttl_sec=SIGNAL_TTL_SEC,
            reason=f"hi={hi:.6g} atr%={atrp:.2f} adx={adx:.1f} reg={regime} tr={trend} imb={imb:.3f} slope={slope_score:.4f}",
            atr_pct=atrp,
            adx=adx,
            spread_pct=spread_pct,
            ob_imb=imb,
            regime=regime,
            bid_wall=bid_wall,
            ask_wall=ask_wall,
            funding_rate=funding_rate,
            oi_delta_pct=oi_delta_pct,
            cvd_delta_quote=cvd_delta_quote,
        )

    return None


# ==============================
# LOGGING
# ==============================
def append_log(sig: Signal, order_id: str) -> None:
    if not ENABLE_SIGNAL_LOG:
        return
    try:
        file_exists = os.path.isfile(SIGNAL_LOG_FILE)
        with open(SIGNAL_LOG_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "timestamp", "signal_id", "symbol", "side", "setup",
                    "entry_mid", "stop", "tp1", "tp2",
                    "atr_pct", "adx", "regime", "spread_pct", "ob_imb",
                    "bid_wall", "ask_wall",
                    "reason", "order_id"
                ])
            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(sig.ts // 1000)),
                sig.signal_id,
                sig.symbol,
                sig.side,
                sig.setup,
                fmt_price(sig.entry_mid),
                fmt_price(sig.stop),
                fmt_price(sig.tp1),
                fmt_price(sig.tp2),
                f"{sig.atr_pct:.3f}",
                f"{sig.adx:.2f}",
                sig.regime,
                f"{sig.spread_pct:.3f}",
                f"{sig.ob_imb:.3f}",
                str(sig.bid_wall),
                str(sig.ask_wall),
                sig.reason,
                order_id,
            ])
    except Exception as e:
        logger.error(f"Failed to write log: {e}")


def format_signal(sig: Signal) -> str:
    side = "🟢 LONG" if sig.side == "Buy" else "🔴 SHORT"
    risk_pct = abs((sig.entry_mid - sig.stop) / sig.entry_mid) * 100.0
    reward1_pct = abs((sig.tp1 - sig.entry_mid) / sig.entry_mid) * 100.0
    return (
        f"{side} | {sig.symbol}\n"
        f"📊 Setup: {sig.setup} | Regime: {sig.regime}\n"
        f"💰 Entry: {sig.entry_mid:.6g}\n"
        f"🛑 Stop: {sig.stop:.6g} (-{risk_pct:.2f}%)\n"
        f"🎯 TP1: {sig.tp1:.6g} (+{reward1_pct:.2f}%)\n"
        f"🎯 TP2: {sig.tp2:.6g}\n"
        f"📈 ATR: {sig.atr_pct:.2f}% | ADX: {sig.adx:.1f} | Spread: {sig.spread_pct:.3f}%\n"
        f"📚 OB imb: {sig.ob_imb:+.3f} | Walls: bid={sig.bid_wall} ask={sig.ask_wall}\n"
        f"⏱ TTL: {sig.ttl_sec}s"
    )


# ==============================
# SCAN / TRADE
# ==============================
def rotate_candidates(symbols: List[str]) -> List[str]:
    global _ROT_OFFSET
    if not symbols:
        return []
    n = len(symbols)
    k = max(1, min(MAX_CANDIDATES, n))
    start = _ROT_OFFSET % n
    out = []
    for i in range(k):
        out.append(symbols[(start + i) % n])
    _ROT_OFFSET = (start + k) % n
    return out


async def wait_limit_fill(session: aiohttp.ClientSession, symbol: str, order_id: str, ttl: int) -> bool:
    max_wait = min(LIMIT_WAIT_SEC, max(5, ttl))
    t_end = time.time() + max_wait
    while time.time() < t_end and not shutdown_event.is_set():
        st = await get_order_status(session, symbol, order_id)
        if st == "Filled":
            return True
        if st in ("Cancelled", "Rejected"):
            return False
        await asyncio.sleep(LIMIT_POLL_SEC)
    await cancel_order(session, symbol, order_id)
    return False


async def scan_and_trade(session: aiohttp.ClientSession) -> None:
    tickers = await bybit_tickers_top(session, TOP_N)
    if not tickers:
        logger.warning("No tickers fetched")
        return

    symbols = [t.get("symbol", "") for t in tickers if t.get("symbol")]
    symbols = symbols[:TOP_N]
    candidates = rotate_candidates(symbols)

    if AUTO_TRADE:
        if await has_open_position(session):
            logger.info("Open position exists -> skipping new entries")
            return

    for sym in candidates:
        if shutdown_event.is_set():
            break
        if should_cooldown(sym):
            continue

        try:
            sig = await detect_signal(session, sym)
            if not sig:
                continue

            msg = format_signal(sig)
            logger.info(msg.replace("\n", " | "))
            await tg_send(session, msg)

            order_id = ""
            if AUTO_TRADE:
                qty, qty_step, min_qty = await compute_qty_float(session, sym, sig.entry_mid)
                planned_risk_usd = abs(sig.entry_mid - sig.stop) * qty

                if not daily_guards_allow(planned_risk_usd):
                    logger.warning("Daily guards block trade")
                    await tg_send(session, "⛔ Daily guards: trade blocked (trades/risk limit).")
                    set_cooldown(sym)
                    append_log(sig, "")
                    return

                await set_leverage_if_needed(session, sym, LEVERAGE)

                use_market = ENTRY_MODE in ("auto", "market")
                limit_price = sig.entry_mid if ENTRY_MODE == "limit" else None

                ok, oid_or_err = await place_entry_order(
                    session=session,
                    symbol=sym,
                    side=sig.side,
                    qty=qty,
                    entry_price=None if use_market else limit_price,
                    stop=sig.stop,
                )
                if not ok:
                    fail_msg = f"❌ Entry failed: {sym} {sig.side} | {oid_or_err}"
                    logger.warning(fail_msg)
                    await tg_send(session, fail_msg)
                    set_cooldown(sym)
                    append_log(sig, "")
                    return

                order_id = oid_or_err

                if not use_market:
                    filled = await wait_limit_fill(session, sym, order_id, sig.ttl_sec)
                    if not filled:
                        msg2 = f"🟡 Limit not filled, cancelled: {sym} oid={order_id}"
                        logger.info(msg2)
                        await tg_send(session, msg2)
                        set_cooldown(sym)
                        append_log(sig, order_id)
                        return

                await place_partials_if_enabled(session, sig, qty, qty_step, min_qty)
                daily_guards_commit(planned_risk_usd)

                success_msg = f"✅ Entry placed: {sym} {sig.side} qty={fmt_qty(qty)} oid={order_id}"
                logger.info(success_msg)
                await tg_send(session, success_msg)

                set_cooldown(sym)

            append_log(sig, order_id)
            return

        except Exception as e:
            logger.error(f"Error processing {sym}: {e}", exc_info=True)
            continue


# ==============================
# MAIN
# ==============================
async def main() -> None:
    global BASE_URL
    BASE_URL = "https://api-testnet.bybit.com" if BYBIT_TESTNET else "https://api.bybit.com"

    if AUTO_TRADE and (not BYBIT_API_KEY or not BYBIT_API_SECRET):
        raise SystemExit("AUTO_TRADE=1 but BYBIT_API_KEY/BYBIT_API_SECRET missing")

    logger.info(f"Starting bot | testnet={BYBIT_TESTNET} | auto_trade={AUTO_TRADE} | entry_mode={ENTRY_MODE}")
    logger.info(f"Notional=${NOTIONAL_USD} | Lev={LEVERAGE} | TOP_N={TOP_N} | MAX_CANDIDATES={MAX_CANDIDATES}")
    if USE_DAILY_GUARDS:
        logger.info(f"Daily guards: max_trades={MAX_TRADES_PER_DAY} | max_risk_usd={MAX_DAILY_RISK_USD}")

    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        await tg_send(session, f"🟢 Bot started | testnet={BYBIT_TESTNET} | auto={AUTO_TRADE}")

        cycle = 0
        while not shutdown_event.is_set():
            t0 = time.perf_counter()
            cycle += 1
            try:
                await scan_and_trade(session)
            except Exception as e:
                logger.error(f"Cycle #{cycle} error: {e}", exc_info=True)

            dt = time.perf_counter() - t0
            logger.info(f"✓ Cycle #{cycle} done in {dt:.2f}s")

            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=POLL_SECONDS)
            except asyncio.TimeoutError:
                pass

        logger.info("Bot stopped")
        await tg_send(session, "🔴 Bot stopped")


if __name__ == "__main__":
    asyncio.run(main())
