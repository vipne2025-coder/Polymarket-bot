# ==============================
# BYBIT ALT SCALPING BOT (FINAL PATCHED)
# ==============================

import os
import time
import math
import csv
import hmac
import hashlib
import asyncio
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import aiohttp
import numpy as np
from cachetools import TTLCache
from tenacity import retry, stop_after_attempt, wait_exponential

# ==============================
# LOGGING
# ==============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("scalp-bot")

# ==============================
# ENV HELPERS
# ==============================
def env_str(name, default=""):
    v = os.getenv(name)
    return default if v is None or str(v).strip() == "" else str(v).strip()

def env_int(name, default=0):
    try:
        return int(env_str(name, default))
    except:
        return default

def env_float(name, default=0.0):
    try:
        return float(env_str(name, default))
    except:
        return default

def env_bool(name, default=False):
    return env_str(name, "1" if default else "0").lower() in ("1", "true", "yes")

# ==============================
# CONFIG
# ==============================
TG_TOKEN = env_str("TELEGRAM_BOT_TOKEN")
TG_CHAT_ID = env_str("TELEGRAM_CHAT_ID")

BYBIT_API_KEY = env_str("BYBIT_API_KEY")
BYBIT_API_SECRET = env_str("BYBIT_API_SECRET")
BYBIT_TESTNET = env_bool("BYBIT_TESTNET", False)

BASE_URL = "https://api.bybit.com"
CATEGORY = "linear"
QUOTE = "USDT"

AUTO_TRADE = env_bool("AUTO_TRADE", True)
ENTRY_MODE = env_str("ENTRY_MODE", "auto")

NOTIONAL_USD = env_float("NOTIONAL_USD", 10)
LEVERAGE = env_int("LEVERAGE", 5)

POLL_SECONDS = env_int("POLL_SECONDS", 15)
TOP_N = env_int("TOP_N", 30)
MAX_CANDIDATES = env_int("MAX_CANDIDATES", 8)

TF_CONTEXT_MIN = env_int("TF_CONTEXT_MIN", 5)
TF_TREND_MIN = env_int("TF_TREND_MIN", 15)
TF_CONFIRM_MIN = env_int("TF_CONFIRM_MIN", 1)
CONFIRM_LOOKBACK = env_int("CONFIRM_LOOKBACK", 6)

ATR_LEN = env_int("ATR_LEN", 14)
MAX_STOP_PCT = env_float("MAX_STOP_PCT", 0.8)
MIN_RR = env_float("MIN_RR", 1.2)
RR1 = env_float("RR1", 1.3)
RR2 = env_float("RR2", 1.8)

USE_ORDERBOOK = env_bool("USE_ORDERBOOK", True)
OB_IMB_MIN = env_float("OB_IMB_MIN", 0.10)
MAX_SPREAD_PCT = env_float("MAX_SPREAD_PCT", 0.12)

ENABLE_SIGNAL_LOG = env_bool("ENABLE_SIGNAL_LOG", True)
SIGNAL_LOG_FILE = env_str("SIGNAL_LOG_FILE", "signals_log.csv")

# ==============================
# CACHES
# ==============================
KLINE_CACHE = TTLCache(maxsize=300, ttl=90)
OB_CACHE = TTLCache(maxsize=300, ttl=45)

# ==============================
# DATA STRUCTURES
# ==============================
@dataclass
class Candle:
    o: float
    h: float
    l: float
    c: float

# ==============================
# UTILS
# ==============================
def atr_pct(candles: List[Candle], length: int) -> float:
    if len(candles) < length + 1:
        return 0.0
    trs = []
    for i in range(1, len(candles)):
        h, l, pc = candles[i].h, candles[i].l, candles[i-1].c
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    atr = np.mean(trs[-length:])
    return (atr / candles[-1].c) * 100.0

# ==============================
# TELEGRAM (ASYNC)
# ==============================
async def tg_send(session, text):
    if not TG_TOKEN or not TG_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    await session.post(url, json={"chat_id": TG_CHAT_ID, "text": text})

# ==============================
# MAIN LOOP (SKELETON)
# ==============================
async def main():
    logger.info("🟢 Bybit Scalp Bot started")
    async with aiohttp.ClientSession() as session:
        await tg_send(session, "🟢 Bybit Scalp Bot запущен")
        while True:
            try:
                # Здесь твоя логика сканирования и входов
                logger.info("Цикл сканирования выполнен")
                await asyncio.sleep(POLL_SECONDS)
            except Exception as e:
                logger.error(f"Ошибка цикла: {e}")
                await asyncio.sleep(10)

if __name__ == "__main__":
    asyncio.run(main())

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

