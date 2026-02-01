#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bybit Alt Scalping Bot (Async) — WORKING v5 (Orders + Signals)

Что делает:
- Сканирует Top-N USDT perpetual (linear)
- Ищет сетап: rejection от robust-границ диапазона + 15m EMA-тренд + 1m BOS подтверждение
- Фильтры: ATR%, спред, дисбаланс стакана (imbalance)
- При AUTO_TRADE=true: открывает ордер (Market/Limit) с TP1 и SL (встроенными)
- Telegram уведомления + CSV лог сигналов

Безопасность:
- category=linear, quote=USDT
- API ключи: только Read + Trade. Withdraw НЕ включать.
"""

import os
import time
import math
import csv
import hmac
import json
import hashlib
import asyncio
import logging
import urllib.parse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import aiohttp
import numpy as np
from cachetools import TTLCache
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ==============================
# LOGGING
# ==============================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("bybit-scalp")

# ==============================
# ENV HELPERS
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

def env_bool(name: str, default: bool) -> bool:
    v = env_str(name, "1" if default else "0").lower()
    return v in ("1", "true", "yes", "y", "on")

# ==============================
# CONFIG
# ==============================
TG_TOKEN = env_str("TELEGRAM_BOT_TOKEN", "")
TG_CHAT_ID = env_str("TELEGRAM_CHAT_ID", "")

BYBIT_API_KEY = env_str("BYBIT_API_KEY", "")
BYBIT_API_SECRET = env_str("BYBIT_API_SECRET", "")
BYBIT_TESTNET = env_bool("BYBIT_TESTNET", False)
BASE_URL = "https://api-testnet.bybit.com" if BYBIT_TESTNET else "https://api.bybit.com"
RECV_WINDOW = env_int("BYBIT_RECV_WINDOW", 5000)

AUTO_TRADE = env_bool("AUTO_TRADE", False)
ENTRY_MODE = env_str("ENTRY_MODE", "auto").lower()  # auto|market|limit
SET_LEVERAGE = env_bool("SET_LEVERAGE", False)

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
LEVEL_TOL_PCT = env_float("LEVEL_TOL_PCT", 0.20)
REJECT_WICK_FRAC = env_float("REJECT_WICK_FRAC", 0.45)

USE_ORDERBOOK = env_bool("USE_ORDERBOOK", True)
OB_LIMIT = env_int("OB_LIMIT", 25)
OB_IMB_MIN = env_float("OB_IMB_MIN", 0.10)
MAX_SPREAD_PCT = env_float("MAX_SPREAD_PCT", 0.12)

MAX_STOP_PCT = env_float("MAX_STOP_PCT", 0.8)
MIN_RR = env_float("MIN_RR", 1.2)
RR1 = env_float("RR1", 1.3)
RR2 = env_float("RR2", 1.8)

SIGNAL_TTL_SEC = env_int("SIGNAL_TTL_SEC", 900)

ENABLE_SIGNAL_LOG = env_bool("ENABLE_SIGNAL_LOG", True)
SIGNAL_LOG_FILE = env_str("SIGNAL_LOG_FILE", "signals_log.csv")

ALLOWED_CATEGORY = env_str("ALLOWED_CATEGORY", "linear")
ALLOWED_QUOTE = env_str("ALLOWED_QUOTE", "USDT")

# ==============================
# CACHES / STATE
# ==============================
KLINE_CACHE = TTLCache(maxsize=600, ttl=90)
OB_CACHE = TTLCache(maxsize=600, ttl=45)
TREND_CACHE = TTLCache(maxsize=600, ttl=240)
INSTR_CACHE: Dict[str, Dict[str, float]] = {}
COOLDOWN: Dict[str, float] = {}

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
    side: str              # "Buy" or "Sell"
    setup: str
    entry_low: float
    entry_high: float
    entry_mid: float
    stop: float
    tp1: float
    tp2: float
    ttl_sec: int
    reason: str
    atr_pct: float
    spread_pct: float
    ob_imb: float

# ==============================
# UTILS
# ==============================
def now_ms() -> int:
    return int(time.time() * 1000)

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

def candle_rejection_at_edge(c: Candle, edge: float, is_low_edge: bool) -> bool:
    if wick_fraction(c) < REJECT_WICK_FRAC:
        return False
    if is_low_edge:
        return (c.l <= edge) and (c.c > edge)
    return (c.h >= edge) and (c.c < edge)

def atr_pct_fast(candles: List[Candle], length: int) -> float:
    if len(candles) < length + 1:
        return 0.0
    highs = np.array([c.h for c in candles], dtype=float)
    lows  = np.array([c.l for c in candles], dtype=float)
    closes = np.array([c.c for c in candles], dtype=float)

    tr1 = highs[1:] - lows[1:]
    tr2 = np.abs(highs[1:] - closes[:-1])
    tr3 = np.abs(lows[1:]  - closes[:-1])
    tr = np.maximum.reduce([tr1, tr2, tr3])

    atr = float(np.mean(tr[-length:]))
    price = float(closes[-1])
    return (atr / price) * 100.0 if price > 0 else 0.0

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

def should_cooldown(symbol: str) -> bool:
    return time.time() < COOLDOWN.get(symbol, 0.0)

def set_cooldown(symbol: str) -> None:
    COOLDOWN[symbol] = time.time() + SYMBOL_COOLDOWN_SEC

def round_step(x: float, step: float, mode: str = "down") -> float:
    if step <= 0:
        return x
    k = x / step
    return (math.ceil(k) if mode == "up" else math.floor(k)) * step

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
            timeout=10,
        ):
            return
    except Exception:
        return

# ==============================
# BYBIT SIGNING (v5)
# ==============================
class BybitTempError(RuntimeError):
    pass

RETRY_EXC = (aiohttp.ClientError, asyncio.TimeoutError, BybitTempError)

def _sign(secret: str, payload: str) -> str:
    return hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()

def _json_compact(obj: dict) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)

def _encode_query(params: dict) -> str:
    # encode consistently with sorted keys
    items = []
    for k in sorted(params.keys()):
        v = params[k]
        if v is None:
            continue
        items.append((k, str(v)))
    return urllib.parse.urlencode(items)

def _headers_private(ts: str, payload: str) -> Dict[str, str]:
    prehash = ts + BYBIT_API_KEY + str(RECV_WINDOW) + payload
    sign = _sign(BYBIT_API_SECRET, prehash)
    return {
        "X-BAPI-API-KEY": BYBIT_API_KEY,
        "X-BAPI-SIGN": sign,
        "X-BAPI-SIGN-TYPE": "2",
        "X-BAPI-TIMESTAMP": ts,
        "X-BAPI-RECV-WINDOW": str(RECV_WINDOW),
        "Content-Type": "application/json",
    }

@retry(stop=stop_after_attempt(5), wait=wait_exponential(min=0.8, max=12),
       retry=retry_if_exception_type(RETRY_EXC), reraise=True)
async def api_get(session: aiohttp.ClientSession, path: str, params: dict = None, timeout: int = 12) -> dict:
    url = BASE_URL + path
    async with session.get(url, params=params or {}, timeout=timeout) as r:
        j = await r.json()
        code = int(j.get("retCode", 0))
        if code in (10006, 10018) or r.status in (429, 500, 502, 503, 504):
            raise BybitTempError(f"temporary bybit error {code}: {j.get('retMsg')}")
        return j

@retry(stop=stop_after_attempt(5), wait=wait_exponential(min=0.8, max=12),
       retry=retry_if_exception_type(RETRY_EXC), reraise=True)
async def api_get_private(session: aiohttp.ClientSession, path: str, params: dict, timeout: int = 12) -> dict:
    if not BYBIT_API_KEY or not BYBIT_API_SECRET:
        raise RuntimeError("Missing BYBIT_API_KEY/BYBIT_API_SECRET")

    query = _encode_query(params or {})
    ts = str(now_ms())
    prehash = ts + BYBIT_API_KEY + str(RECV_WINDOW) + query
    sign = _sign(BYBIT_API_SECRET, prehash)

    headers = {
        "X-BAPI-API-KEY": BYBIT_API_KEY,
        "X-BAPI-SIGN": sign,
        "X-BAPI-SIGN-TYPE": "2",
        "X-BAPI-TIMESTAMP": ts,
        "X-BAPI-RECV-WINDOW": str(RECV_WINDOW),
    }
    url = BASE_URL + path

    async with session.get(url, params=params, headers=headers, timeout=timeout) as r:
        j = await r.json()
        code = int(j.get("retCode", 0))
        if code in (10006, 10018) or r.status in (429, 500, 502, 503, 504):
            raise BybitTempError(f"temporary bybit error {code}: {j.get('retMsg')}")
        return j

@retry(stop=stop_after_attempt(5), wait=wait_exponential(min=0.8, max=12),
       retry=retry_if_exception_type(RETRY_EXC), reraise=True)
async def api_post_private(session: aiohttp.ClientSession, path: str, body: dict, timeout: int = 12) -> dict:
    if not BYBIT_API_KEY or not BYBIT_API_SECRET:
        raise RuntimeError("Missing BYBIT_API_KEY/BYBIT_API_SECRET")

    payload = _json_compact(body)
    ts = str(now_ms())
    headers = _headers_private(ts, payload)

    url = BASE_URL + path
    async with session.post(url, data=payload, headers=headers, timeout=timeout) as r:
        j = await r.json()
        code = int(j.get("retCode", 0))
        if code in (10006, 10018) or r.status in (429, 500, 502, 503, 504):
            raise BybitTempError(f"temporary bybit error {code}: {j.get('retMsg')}")
        return j

# ==============================
# MARKET DATA
# ==============================
async def bybit_kline(session: aiohttp.ClientSession, symbol: str, interval_min: int, limit: int) -> List[Candle]:
    key = f"k_{symbol}_{interval_min}_{limit}"
    if key in KLINE_CACHE:
        return KLINE_CACHE[key]

    j = await api_get(session, "/v5/market/kline", {
        "category": ALLOWED_CATEGORY,
        "symbol": symbol,
        "interval": str(interval_min),
        "limit": str(limit),
    })

    rows = j.get("result", {}).get("list", []) or []
    out: List[Candle] = []
    for r in reversed(rows):
        try:
            out.append(Candle(
                ts=int(r[0]),
                o=float(r[1]),
                h=float(r[2]),
                l=float(r[3]),
                c=float(r[4]),
                v=float(r[5]),
            ))
        except Exception:
            continue

    KLINE_CACHE[key] = out
    return out

async def bybit_orderbook(session: aiohttp.ClientSession, symbol: str, limit: int) -> Tuple[float, float, float]:
    key = f"ob_{symbol}_{limit}"
    if key in OB_CACHE:
        return OB_CACHE[key]

    j = await api_get(session, "/v5/market/orderbook", {
        "category": ALLOWED_CATEGORY,
        "symbol": symbol,
        "limit": str(limit),
    })

    res = j.get("result", {}) or {}
    bids = res.get("b", []) or []
    asks = res.get("a", []) or []

    if not bids or not asks:
        return 0.0, 0.0, 0.0

    bid1 = float(bids[0][0])
    ask1 = float(asks[0][0])

    bid_sum = sum(float(q) for _, q in bids)
    ask_sum = sum(float(q) for _, q in asks)
    tot = bid_sum + ask_sum
    imb = (bid_sum - ask_sum) / tot if tot > 0 else 0.0

    OB_CACHE[key] = (bid1, ask1, imb)
    return bid1, ask1, imb

async def bybit_tickers_top(session: aiohttp.ClientSession, top_n: int) -> List[dict]:
    j = await api_get(session, "/v5/market/tickers", {"category": ALLOWED_CATEGORY})
    lst = j.get("result", {}).get("list", []) or []

    out = []
    for t in lst:
        sym = t.get("symbol", "")
        if not sym.endswith(ALLOWED_QUOTE):
            continue
        try:
            turn = float(t.get("turnover24h", 0) or 0)
            last = float(t.get("lastPrice", 0) or 0)
        except Exception:
            continue
        if turn <= 0 or last <= 0:
            continue
        out.append(t)

    out.sort(key=lambda x: float(x.get("turnover24h", 0) or 0), reverse=True)
    return out[:max(1, top_n)]

async def bybit_instrument_steps(session: aiohttp.ClientSession, symbol: str) -> Dict[str, float]:
    if symbol in INSTR_CACHE:
        return INSTR_CACHE[symbol]

    j = await api_get(session, "/v5/market/instruments-info", {
        "category": ALLOWED_CATEGORY,
        "symbol": symbol,
    })

    lst = j.get("result", {}).get("list", []) or []
    if not lst:
        raise RuntimeError(f"instrument info missing for {symbol}")

    info = lst[0]
    lot = info.get("lotSizeFilter", {}) or {}
    pricef = info.get("priceFilter", {}) or {}

    steps = {
        "qtyStep": float(lot.get("qtyStep", 0.0) or 0.0) or 0.001,
        "minQty": float(lot.get("minOrderQty", 0.0) or 0.0) or 0.001,
        "minNotional": float(lot.get("minOrderAmt", 0.0) or 0.0) or 0.0,
        "priceStep": float(pricef.get("tickSize", 0.0) or 0.0) or 0.0001,
    }

    INSTR_CACHE[symbol] = steps
    return steps

# ==============================
# TRADING
# ==============================
async def has_open_position(session: aiohttp.ClientSession) -> bool:
    j = await api_get_private(session, "/v5/position/list", {"category": ALLOWED_CATEGORY})
    lst = j.get("result", {}).get("list", []) or []
    for p in lst:
        try:
            size = float(p.get("size", 0) or 0)
            if size != 0.0:
                return True
        except Exception:
            continue
    return False

async def set_leverage_if_needed(session: aiohttp.ClientSession, symbol: str, lev: int) -> None:
    if not SET_LEVERAGE:
        return
    body = {
        "category": ALLOWED_CATEGORY,
        "symbol": symbol,
        "buyLeverage": str(lev),
        "sellLeverage": str(lev),
    }
    j = await api_post_private(session, "/v5/position/set-leverage", body)
    if int(j.get("retCode", 0)) != 0:
        logger.warning(f"set leverage failed {symbol}: {j.get('retMsg')}")

async def compute_qty(session: aiohttp.ClientSession, symbol: str, price: float) -> float:
    steps = await bybit_instrument_steps(session, symbol)
    qty_step = steps["qtyStep"]
    min_qty = steps["minQty"]
    min_notional = steps["minNotional"]

    pos_value = NOTIONAL_USD * float(LEVERAGE)
    raw_qty = pos_value / price
    qty = round_step(raw_qty, qty_step, "down")

    if qty < min_qty:
        qty = min_qty

    if min_notional > 0 and qty * price < min_notional:
        qty = round_step(min_notional / price, qty_step, "up")

    return float(qty)

async def place_order(session: aiohttp.ClientSession, symbol: str, side: str, qty: float,
                      price: Optional[float], sl: float, tp: float) -> Tuple[bool, str]:
    body = {
        "category": ALLOWED_CATEGORY,
        "symbol": symbol,
        "side": side,  # "Buy"/"Sell"
        "orderType": "Market" if price is None else "Limit",
        "qty": str(qty),
        "timeInForce": "IOC" if price is None else "GTC",
        "reduceOnly": False,
        "closeOnTrigger": False,
        "positionIdx": 0,
        "takeProfit": str(tp),
        "stopLoss": str(sl),
        "tpTriggerBy": "LastPrice",
        "slTriggerBy": "LastPrice",
    }
    if price is not None:
        body["price"] = str(price)

    j = await api_post_private(session, "/v5/order/create", body)
    if int(j.get("retCode", 0)) != 0:
        return False, f"{j.get('retMsg')} (code {j.get('retCode')})"
    oid = j.get("result", {}).get("orderId", "")
    return True, oid

# ==============================
# SIGNAL LOG
# ==============================
def ensure_log_header() -> None:
    if not ENABLE_SIGNAL_LOG:
        return
    if os.path.exists(SIGNAL_LOG_FILE):
        return
    header = [
        "ts","signal_id","symbol","side","setup",
        "entry_low","entry_high","entry_mid",
        "stop","tp1","tp2","ttl_sec",
        "atr_pct","spread_pct","ob_imb",
        "auto_trade","order_id","reason"
    ]
    with open(SIGNAL_LOG_FILE, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(header)

def append_log(sig: Signal, order_id: str) -> None:
    if not ENABLE_SIGNAL_LOG:
        return
    ensure_log_header()
    row = [
        sig.ts, sig.signal_id, sig.symbol, sig.side, sig.setup,
        sig.entry_low, sig.entry_high, sig.entry_mid,
        sig.stop, sig.tp1, sig.tp2, sig.ttl_sec,
        sig.atr_pct, sig.spread_pct, sig.ob_imb,
        int(AUTO_TRADE), order_id, sig.reason
    ]
    with open(SIGNAL_LOG_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)

# ==============================
# STRATEGY
# ==============================
async def trend_filter(session: aiohttp.ClientSession, symbol: str) -> int:
    key = f"trend_{symbol}_{TF_TREND_MIN}"
    if key in TREND_CACHE:
        return TREND_CACHE[key]

    candles = await bybit_kline(session, symbol, TF_TREND_MIN, 120)
    closes = [c.c for c in candles]
    if len(closes) < 60:
        TREND_CACHE[key] = 0
        return 0

    e_fast = ema(closes, 20)[-1]
    e_slow = ema(closes, 50)[-1]
    trend = 1 if e_fast > e_slow else -1 if e_fast < e_slow else 0

    TREND_CACHE[key] = trend
    return trend

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

    # стакан + спред
    spread_pct = 0.0
    imb = 0.0
    if USE_ORDERBOOK:
        bid1, ask1, imb = await bybit_orderbook(session, symbol, OB_LIMIT)
        if bid1 <= 0 or ask1 <= 0:
            return None
        spread_pct = ((ask1 - bid1) / ((ask1 + bid1) / 2.0)) * 100.0
        if spread_pct > MAX_SPREAD_PCT:
            return None

    trend = await trend_filter(session, symbol)

    tol = (LEVEL_TOL_PCT / 100.0) * mid
    near_low = (abs(mid - lo) <= tol) or (last.l <= lo + tol)
    near_high = (abs(mid - hi) <= tol) or (last.h >= hi - tol)

    conf = await bybit_kline(session, symbol, TF_CONFIRM_MIN, max(CONFIRM_LOOKBACK + 2, 20))
    if len(conf) < CONFIRM_LOOKBACK:
        return None

    # LONG
    if near_low and candle_rejection_at_edge(last, lo, True):
        look = max(CONFIRM_LOOKBACK, 7) if trend == -1 else CONFIRM_LOOKBACK
        if not confirm_bos(conf, "Buy", look):
            return None
        if USE_ORDERBOOK and imb < OB_IMB_MIN:
            return None

        stop = min(last.l, lo) * (1 - 0.001)  # буфер 0.1%
        stop_pct = ((mid - stop) / mid) * 100.0
        if stop_pct <= 0 or stop_pct > MAX_STOP_PCT:
            return None

        risk = mid - stop
        tp1 = mid + risk * RR1
        tp2 = mid + risk * RR2
        rr = (tp1 - mid) / risk if risk > 0 else 0.0
        if rr < MIN_RR:
            return None

        return Signal(
            signal_id=f"{symbol}_{now_ms()}",
            ts=now_ms(),
            symbol=symbol,
            side="Buy",
            setup="A_reject_low",
            entry_low=mid * 0.999,
            entry_high=mid * 1.001,
            entry_mid=mid,
            stop=stop,
            tp1=tp1,
            tp2=tp2,
            ttl_sec=SIGNAL_TTL_SEC,
            reason=f"reject_low lo={lo:.6g} atr%={atrp:.2f} trend={trend} ob_imb={imb:.3f}",
            atr_pct=atrp,
            spread_pct=spread_pct,
            ob_imb=imb,
        )

    # SHORT
    if near_high and candle_rejection_at_edge(last, hi, False):
        look = max(CONFIRM_LOOKBACK, 7) if trend == 1 else CONFIRM_LOOKBACK
        if not confirm_bos(conf, "Sell", look):
            return None
        if USE_ORDERBOOK and imb > -OB_IMB_MIN:
            return None

        stop = max(last.h, hi) * (1 + 0.001)
        stop_pct = ((stop - mid) / mid) * 100.0
        if stop_pct <= 0 or stop_pct > MAX_STOP_PCT:
            return None

        risk = stop - mid
        tp1 = mid - risk * RR1
        tp2 = mid - risk * RR2
        rr = (mid - tp1) / risk if risk > 0 else 0.0
        if rr < MIN_RR:
            return None

        return Signal(
            signal_id=f"{symbol}_{now_ms()}",
            ts=now_ms(),
            symbol=symbol,
            side="Sell",
            setup="A_reject_high",
            entry_low=mid * 0.999,
            entry_high=mid * 1.001,
            entry_mid=mid,
            stop=stop,
            tp1=tp1,
            tp2=tp2,
            ttl_sec=SIGNAL_TTL_SEC,
            reason=f"reject_high hi={hi:.6g} atr%={atrp:.2f} trend={trend} ob_imb={imb:.3f}",
            atr_pct=atrp,
            spread_pct=spread_pct,
            ob_imb=imb,
        )

    return None

def format_signal(sig: Signal) -> str:
    side = "LONG" if sig.side == "Buy" else "SHORT"
    return (
        f"📌 {sig.symbol} | {side} | {sig.setup}\n"
        f"Entry: {sig.entry_low:.6g}–{sig.entry_high:.6g} (mid {sig.entry_mid:.6g})\n"
        f"SL: {sig.stop:.6g}\n"
        f"TP1: {sig.tp1:.6g} | TP2: {sig.tp2:.6g}\n"
        f"ATR%: {sig.atr_pct:.2f} | Spread%: {sig.spread_pct:.3f} | OB_imb: {sig.ob_imb:.3f}\n"
        f"TTL: {sig.ttl_sec}s\n"
        f"{sig.reason}"
    )

# ==============================
# MAIN LOOP
# ==============================
async def scan_and_trade(session: aiohttp.ClientSession) -> None:
    # 1) тикеры по обороту
    tickers = await bybit_tickers_top(session, TOP_N)
    symbols = [t["symbol"] for t in tickers][:TOP_N]
    candidates = symbols[:max(1, MAX_CANDIDATES)]

    # 2) правило: одна позиция
    if AUTO_TRADE:
        try:
            if await has_open_position(session):
                logger.info("Open position detected -> skip new entries")
                return
        except Exception as e:
            logger.warning(f"position check failed: {e}")

    for sym in candidates:
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
                await set_leverage_if_needed(session, sym, LEVERAGE)

                use_market = ENTRY_MODE in ("auto", "market")
                limit_price = sig.entry_mid if ENTRY_MODE == "limit" else None

                qty = await compute_qty(session, sym, sig.entry_mid)

                ok, oid_or_err = await place_order(
                    session=session,
                    symbol=sym,
                    side=sig.side,
                    qty=qty,
                    price=None if use_market else limit_price,
                    sl=sig.stop,
                    tp=sig.tp1,  # TP2 оставляем как “дальний” ориентир
                )

                if ok:
                    order_id = oid_or_err
                    await tg_send(session, f"✅ Order placed: {sym} {sig.side} qty={qty} oid={order_id}")
                    logger.info(f"Order placed: {sym} {sig.side} qty={qty} oid={order_id}")
                else:
                    await tg_send(session, f"❌ Order failed: {sym} {sig.side} | {oid_or_err}")
                    logger.warning(f"Order failed: {sym} {sig.side} | {oid_or_err}")

                set_cooldown(sym)

            append_log(sig, order_id)
            return  # максимум 1 сделка/сигнал за цикл

        except Exception as e:
            logger.error(f"Error processing {sym}: {e}", exc_info=True)
            continue

async def main() -> None:
    global BASE_URL
    BASE_URL = "https://api-testnet.bybit.com" if BYBIT_TESTNET else "https://api.bybit.com"

    if AUTO_TRADE and (not BYBIT_API_KEY or not BYBIT_API_SECRET):
        raise SystemExit("AUTO_TRADE=true but BYBIT_API_KEY/BYBIT_API_SECRET are missing.")

    logger.info(f"Start | testnet={BYBIT_TESTNET} auto_trade={AUTO_TRADE} notional={NOTIONAL_USD} lev={LEVERAGE}")

    timeout = aiohttp.ClientTimeout(total=25)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        await tg_send(session, "🟢 Bybit bot запущен")
        while True:
            t0 = time.perf_counter()
            try:
                await scan_and_trade(session)
            except Exception as e:
                logger.error(f"Main loop error: {e}", exc_info=True)
            dt = time.perf_counter() - t0
            logger.info(f"cycle done in {dt:.2f}s")
            await asyncio.sleep(POLL_SECONDS)

if __name__ == "__main__":
    asyncio.run(main())

if __name__ == "__main__":
    main()


