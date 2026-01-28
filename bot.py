import os
import time
import requests
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# =========================
# Telegram
# =========================
TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TG_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
if not TG_TOKEN or not TG_CHAT_ID:
    raise RuntimeError("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")

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
        print("TG_ERROR", r.status_code, r.text[:200], flush=True)


# =========================
# User settings
# =========================
EQUITY_USD = float(os.environ.get("EQUITY_USD", "200"))
LEVERAGE = float(os.environ.get("LEVERAGE", "15"))

# Рекомендуемый notional по типу
NOTIONAL_TREND = float(os.environ.get("NOTIONAL_TREND", "20"))
NOTIONAL_BOOST = float(os.environ.get("NOTIONAL_BOOST", "50"))

# =========================
# Bot settings
# =========================
BYBIT_BASE = os.environ.get("BYBIT_BASE", "https://api.bybit.com").rstrip("/")
POLL_SECONDS = float(os.environ.get("POLL_SECONDS", "10"))
TOP_N = int(os.environ.get("TOP_N", "20"))

CONFIRM_DELAY_SEC = int(os.environ.get("CONFIRM_DELAY_SEC", "20"))
SYMBOL_COOLDOWN_SEC = int(os.environ.get("SYMBOL_COOLDOWN_SEC", "1200"))  # 20 мин

# Lookbacks from tickers history
OI_LOOKBACK_SEC = int(os.environ.get("OI_LOOKBACK_SEC", "300"))       # ~5m
PRICE_LOOKBACK_SEC = int(os.environ.get("PRICE_LOOKBACK_SEC", "120")) # ~2m

MAX_SPREAD_PCT = float(os.environ.get("MAX_SPREAD_PCT", "0.08"))      # 0.08% максимум

# Funding risk flags (не блокируем, помечаем)
FUNDING_HIGH = float(os.environ.get("FUNDING_HIGH", "0.0002"))
FUNDING_LOW = float(os.environ.get("FUNDING_LOW", "-0.0002"))

# Kline
USE_KLINE = int(os.environ.get("USE_KLINE", "1"))
KLINE_INTERVAL = os.environ.get("KLINE_INTERVAL", "1")   # 1m
KLINE_LIMIT = int(os.environ.get("KLINE_LIMIT", "120"))   # 120 минут истории
VOL_SPIKE_MULT = float(os.environ.get("VOL_SPIKE_MULT", "1.8"))

# Сигналы (FAST отключён — специально)
TREND_PRICE_MOVE_PCT = float(os.environ.get("TREND_PRICE_MOVE_PCT", "1.2"))
TREND_OI_CHANGE_PCT = float(os.environ.get("TREND_OI_CHANGE_PCT", "2.0"))

BOOST_PRICE_MOVE_PCT = float(os.environ.get("BOOST_PRICE_MOVE_PCT", "1.8"))
BOOST_OI_CHANGE_PCT = float(os.environ.get("BOOST_OI_CHANGE_PCT", "3.0"))

# Вход по откату (зона)
# LONG: вход ниже текущей цены на 0.3–0.7%
# SHORT: вход выше текущей цены на 0.3–0.7%
PULLBACK_MIN_PCT = float(os.environ.get("PULLBACK_MIN_PCT", "0.30"))
PULLBACK_MAX_PCT = float(os.environ.get("PULLBACK_MAX_PCT", "0.70"))

# Чтобы не входить “в догонку” — если цена уже улетела, сигнал пропускаем
MAX_CHASE_PCT = float(os.environ.get("MAX_CHASE_PCT", "1.20"))  # 1.2% от цены на момент кандидата

# Стоп за структуру + буфер от ATR
STRUCT_WINDOW_MIN = int(os.environ.get("STRUCT_WINDOW_MIN", "20"))  # берём low/high за последние 20 минут
ATR_MULT_BUFFER = float(os.environ.get("ATR_MULT_BUFFER", "0.25"))  # буфер = 0.25*ATR к стопу

# Ограничения SL%
SL_PCT_MIN = float(os.environ.get("SL_PCT_MIN", "1.20"))
SL_PCT_MAX = float(os.environ.get("SL_PCT_MAX", "3.50"))

# Take-profit в R (от расстояния до SL)
TP1_R = float(os.environ.get("TP1_R", "1.6"))
TP2_R = float(os.environ.get("TP2_R", "3.0"))


# =========================
# Bybit V5
# =========================
def bybit_get(path: str, params: dict) -> dict:
    url = f"{BYBIT_BASE}{path}"
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def get_linear_tickers() -> List[dict]:
    data = bybit_get("/v5/market/tickers", {"category": "linear"})
    if data.get("retCode") != 0:
        raise RuntimeError(f"Bybit tickers retCode={data.get('retCode')} msg={data.get('retMsg')}")
    return data["result"]["list"]


def get_kline(symbol: str) -> List[list]:
    data = bybit_get("/v5/market/kline", {
        "category": "linear",
        "symbol": symbol,
        "interval": KLINE_INTERVAL,
        "limit": KLINE_LIMIT
    })
    if data.get("retCode") != 0:
        raise RuntimeError(f"Bybit kline retCode={data.get('retCode')} msg={data.get('retMsg')}")
    return data["result"]["list"]


# =========================
# Helpers
# =========================
def f(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def pct_change(cur: float, prev: float) -> float:
    if prev == 0:
        return 0.0
    return (cur - prev) / prev * 100.0


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def fmt_pct(x: float, nd: int = 2) -> str:
    return f"{x:.{nd}f}%"


def fmt_price(x: float) -> str:
    if x >= 100:
        return f"{x:.2f}"
    if x >= 1:
        return f"{x:.4f}"
    return f"{x:.6f}"


def spread_pct(bid: float, ask: float) -> float:
    if bid <= 0 or ask <= 0:
        return 999.0
    mid = (bid + ask) / 2.0
    if mid <= 0:
        return 999.0
    return abs(ask - bid) / mid * 100.0


def funding_flag_ru(funding: float, direction: str) -> str:
    if direction == "LONG" and funding >= FUNDING_HIGH:
        return "⚠️ funding высокий (лонги перегреты)"
    if direction == "SHORT" and funding <= FUNDING_LOW:
        return "⚠️ funding низкий (шорты перегреты)"
    return "ok"


def decide_direction(price_pct: float) -> str:
    return "LONG" if price_pct >= 0 else "SHORT"


def pick_top_symbols(tickers: List[dict], n: int) -> List[dict]:
    lst = []
    for t in tickers:
        sym = str(t.get("symbol", ""))
        if sym.endswith("USDT"):
            lst.append(t)
    lst.sort(key=lambda x: f(x.get("turnover24h", 0.0)), reverse=True)
    return lst[:n]


# =========================
# Kline analytics: volume spike, ATR, structure low/high
# =========================
def kline_metrics(symbol: str) -> Tuple[Optional[bool], Optional[float], Optional[float], Optional[float]]:
    """
    returns:
      vol_ok: last 1m volume >= VOL_SPIKE_MULT * avg(prev)
      atr_pct: ATR(14) in %
      struct_low: low of last STRUCT_WINDOW_MIN candles
      struct_high: high of last STRUCT_WINDOW_MIN candles
    """
    try:
        kl = get_kline(symbol)
        if not kl or len(kl) < max(30, STRUCT_WINDOW_MIN + 5):
            return None, None, None, None

        rows = []
        for r in kl:
            if len(r) < 7:
                continue
            ts = int(r[0]) // 1000
            o, h, l, c, v = map(float, [r[1], r[2], r[3], r[4], r[5]])
            rows.append((ts, o, h, l, c, v))
        rows.sort(key=lambda x: x[0])
        if len(rows) < max(30, STRUCT_WINDOW_MIN + 5):
            return None, None, None, None

        # volume spike (last vs avg of previous 30)
        vols = [x[5] for x in rows[-31:]]
        last_v = vols[-1]
        avg_v = sum(vols[:-1]) / max(1, (len(vols) - 1))
        vol_ok = (avg_v > 0) and (last_v >= VOL_SPIKE_MULT * avg_v)

        # ATR(14)
        tr = []
        for i in range(1, len(rows)):
            prev_c = rows[i - 1][4]
            h = rows[i][2]
            l = rows[i][3]
            tr.append(max(h - l, abs(h - prev_c), abs(l - prev_c)))
        atr = sum(tr[-14:]) / max(1, len(tr[-14:]))
        last_price = rows[-1][4]
        atr_pct = (atr / last_price) * 100.0 if last_price > 0 else None

        window = rows[-STRUCT_WINDOW_MIN:]
        struct_low = min(x[3] for x in window)
        struct_high = max(x[2] for x in window)

        return vol_ok, atr_pct, struct_low, struct_high
    except Exception as e:
        print("kline_metrics_error", symbol, repr(e), flush=True)
        return None, None, None, None


# =========================
# State
# =========================
@dataclass
class Point:
    ts: int
    price: float
    oi_value: float
    funding: float
    bid: float
    ask: float
    turnover24h: float


@dataclass
class Candidate:
    symbol: str
    direction: str          # LONG/SHORT
    tier: str               # TREND/BOOST
    created_ts: int
    ref_price: float
    oi_pct: float
    price_pct: float
    funding: float
    spread_pct: float
    vol_ok: Optional[bool] = None
    atr_pct: Optional[float] = None
    struct_low: Optional[float] = None
    struct_high: Optional[float] = None


history: Dict[str, List[Point]] = {}
pending: Dict[str, Candidate] = {}
last_sent: Dict[str, int] = {}


def prune(sym: str, keep_sec: int = 1800) -> None:
    now = int(time.time())
    arr = history.get(sym, [])
    history[sym] = [p for p in arr if now - p.ts <= keep_sec]


def get_lookback_point(points: List[Point], lookback_sec: int) -> Optional[Point]:
    if not points:
        return None
    now = points[-1].ts
    target = now - lookback_sec
    best = None
    for p in points:
        if p.ts <= target:
            best = p
    return best or points[0]


def cooldown_ok(sym: str) -> bool:
    now = int(time.time())
    return (now - last_sent.get(sym, 0)) >= SYMBOL_COOLDOWN_SEC


def mark_sent(sym: str) -> None:
    last_sent[sym] = int(time.time())


def classify_tier(price_pct: float, oi_pct: float) -> Optional[str]:
    ap = abs(price_pct)
    ao = abs(oi_pct)
    if ap >= BOOST_PRICE_MOVE_PCT and ao >= BOOST_OI_CHANGE_PCT:
        return "BOOST"
    if ap >= TREND_PRICE_MOVE_PCT and ao >= TREND_OI_CHANGE_PCT:
        return "TREND"
    return None


def tier_notional(tier: str) -> float:
    return NOTIONAL_BOOST if tier == "BOOST" else NOTIONAL_TREND


def entry_zone_from_price(cur_price: float, direction: str) -> Tuple[float, float]:
    """
    Возвращает (entry_hi, entry_lo) — диапазон для лимитного входа.
    LONG: зона ниже текущей цены
    SHORT: зона выше текущей цены
    """
    pb_min = PULLBACK_MIN_PCT / 100.0
    pb_max = PULLBACK_MAX_PCT / 100.0

    if direction == "LONG":
        entry_hi = cur_price * (1.0 - pb_min)
        entry_lo = cur_price * (1.0 - pb_max)
    else:
        entry_lo = cur_price * (1.0 + pb_min)
        entry_hi = cur_price * (1.0 + pb_max)

    # нормализуем (hi всегда >= lo)
    hi = max(entry_hi, entry_lo)
    lo = min(entry_hi, entry_lo)
    return hi, lo


def compute_structure_sl(entry: float, direction: str, struct_low: Optional[float], struct_high: Optional[float], atr_pct: Optional[float]) -> float:
    """
    Стоп за структуру + ATR-буфер.
    LONG: SL ниже struct_low
    SHORT: SL выше struct_high
    Если структуры нет — fallback на entry +/- SL_PCT_MIN
    """
    # буфер в цене = ATR_MULT_BUFFER * ATR (в цене)
    atr_price = None
    if atr_pct is not None:
        atr_price = entry * (atr_pct / 100.0)

    buffer_price = (ATR_MULT_BUFFER * atr_price) if atr_price is not None else (entry * (0.25 / 100.0))

    if direction == "LONG" and struct_low is not None:
        sl = struct_low - buffer_price
    elif direction == "SHORT" and struct_high is not None:
        sl = struct_high + buffer_price
    else:
        # fallback
        sl = entry * (1.0 - SL_PCT_MIN / 100.0) if direction == "LONG" else entry * (1.0 + SL_PCT_MIN / 100.0)

    # Ограничим SL% (чтобы не было слишком узко/широко)
    sl_pct = abs(entry - sl) / entry * 100.0 if entry > 0 else SL_PCT_MIN
    sl_pct = clamp(sl_pct, SL_PCT_MIN, SL_PCT_MAX)

    # Пересоберём SL строго под ограниченный sl_pct
    if direction == "LONG":
        return entry * (1.0 - sl_pct / 100.0)
    else:
        return entry * (1.0 + sl_pct / 100.0)


def compute_tp(entry: float, sl: float, direction: str) -> Tuple[float, float]:
    """
    TP по R: TP = entry +/- (entry-sl)*R
    """
    risk = abs(entry - sl)
    if direction == "LONG":
        tp1 = entry + risk * TP1_R
        tp2 = entry + risk * TP2_R
    else:
        tp1 = entry - risk * TP1_R
        tp2 = entry - risk * TP2_R
    return tp1, tp2


def build_message_ru(c: Candidate, cur_price: float) -> str:
    tier_text = "🚀 BOOST" if c.tier == "BOOST" else "📈 TREND"
    notional = tier_notional(c.tier)
    margin = notional / LEVERAGE

    dir_ru = "🟢 ЛОНГ" if c.direction == "LONG" else "🔴 ШОРТ"
    fund_note = funding_flag_ru(c.funding, c.direction)

    # Вход по откату — зона
    entry_hi, entry_lo = entry_zone_from_price(cur_price, c.direction)
    entry_mid = (entry_hi + entry_lo) / 2.0

    # Стоп за структуру
    sl = compute_structure_sl(entry_mid, c.direction, c.struct_low, c.struct_high, c.atr_pct)
    sl_pct = abs(entry_mid - sl) / entry_mid * 100.0 if entry_mid > 0 else SL_PCT_MIN

    tp1, tp2 = compute_tp(entry_mid, sl, c.direction)

    risk_usd = notional * (sl_pct / 100.0)

    vol_line = ""
    if c.vol_ok is True:
        vol_line = "• Объём: всплеск ✅"
    elif c.vol_ok is False:
        vol_line = "• Объём: нет всплеска ⚠️"

    factors = (
        f"• ΔOI (~5м): {fmt_pct(c.oi_pct, 2)}\n"
        f"• ΔЦена (~2м): {fmt_pct(c.price_pct, 2)}\n"
        f"• Funding: {c.funding:.6f} ({fund_note})\n"
        f"• Спред: {fmt_pct(c.spread_pct, 3)}\n"
        f"{vol_line}"
    )

    return (
        f"{tier_text} | {c.symbol} (Bybit USDT-PERP)\n"
        f"{dir_ru}\n\n"
        f"✅ ВХОД ТОЛЬКО ЛИМИТКОЙ (по откату)\n"
        f"🎯 Зона входа: {fmt_price(entry_lo)} – {fmt_price(entry_hi)}\n"
        f"⛔ Стоп-лосс: {fmt_price(sl)}  (~{fmt_pct(sl_pct,2)} | риск ≈ ${risk_usd:.2f})\n"
        f"✅ Тейк 1: {fmt_price(tp1)}  (~{fmt_pct(TP1_R*sl_pct,2)})\n"
        f"✅ Тейк 2: {fmt_price(tp2)}  (~{fmt_pct(TP2_R*sl_pct,2)})\n\n"
        f"💼 Рекомендация: ${notional:.0f} (плечо x{LEVERAGE:.0f}, маржа ≈ ${margin:.2f})\n\n"
        f"📌 Факторы:\n{factors}\n\n"
        f"⚠️ Если вход не дали — ПРОПУСКАЕШЬ (это нормально)."
    )


def main() -> None:
    tg_send(
        "✅ Bybit Futures Bot (работающий режим)\n"
        f"TOP {TOP_N}, опрос={POLL_SECONDS}s, подтверждение={CONFIRM_DELAY_SEC}s\n"
        f"Режим: вход по откату лимиткой + стоп за структуру\n"
        f"Классы: TREND ${NOTIONAL_TREND:.0f} | BOOST ${NOTIONAL_BOOST:.0f} | плечо x{LEVERAGE:.0f}"
    )

    while True:
        now = int(time.time())
        try:
            tickers = get_linear_tickers()
            top = pick_top_symbols(tickers, TOP_N)

            # 1) update history
            for t in top:
                sym = str(t.get("symbol"))
                price = f(t.get("lastPrice"))
                oi_val = f(t.get("openInterestValue"))
                funding = f(t.get("fundingRate"))
                bid = f(t.get("bid1Price"))
                ask = f(t.get("ask1Price"))
                turnover = f(t.get("turnover24h"))

                p = Point(
                    ts=now, price=price, oi_value=oi_val, funding=funding,
                    bid=bid, ask=ask, turnover24h=turnover
                )
                history.setdefault(sym, []).append(p)
                prune(sym)

            # 2) create candidates
            created = 0
            for t in top:
                sym = str(t.get("symbol"))
                pts = history.get(sym, [])
                if len(pts) < 3:
                    continue

                cur = pts[-1]
                sp = spread_pct(cur.bid, cur.ask)
                if sp > MAX_SPREAD_PCT:
                    continue

                p_lb = get_lookback_point(pts, PRICE_LOOKBACK_SEC)
                oi_lb = get_lookback_point(pts, OI_LOOKBACK_SEC)
                if not p_lb or not oi_lb:
                    continue

                price_pct = pct_change(cur.price, p_lb.price)
                oi_pct = pct_change(cur.oi_value, oi_lb.oi_value)

                tier = classify_tier(price_pct, oi_pct)
                if tier is None:
                    continue

                # cooldown
                if not cooldown_ok(sym):
                    continue

                # already pending
                if sym in pending:
                    continue

                direction = decide_direction(price_pct)

                cand = Candidate(
                    symbol=sym,
                    direction=direction,
                    tier=tier,
                    created_ts=now,
                    ref_price=cur.price,
                    oi_pct=oi_pct,
                    price_pct=price_pct,
                    funding=cur.funding,
                    spread_pct=sp,
                )

                # Kline metrics
                if USE_KLINE:
                    vol_ok, atr_pct, struct_low, struct_high = kline_metrics(sym)
                    cand.vol_ok = vol_ok
                    cand.atr_pct = atr_pct
                    cand.struct_low = struct_low
                    cand.struct_high = struct_high

                    # Если объёма нет — это часто “шум”. Мы не блокируем TREND полностью,
                    # но BOOST без объёма превращаем в TREND.
                    if vol_ok is False and cand.tier == "BOOST":
                        cand.tier = "TREND"
                    # TREND без объёма оставляем, но он будет реже из-за порогов.

                pending[sym] = cand
                created += 1

            # 3) confirm and send
            to_delete = []
            confirmed = 0

            for sym, cand in list(pending.items()):
                if now - cand.created_ts < CONFIRM_DELAY_SEC:
                    continue

                pts = history.get(sym, [])
                if not pts:
                    to_delete.append(sym)
                    continue

                cur = pts[-1]

                # 3.1) если цена слишком “убежала” с момента кандидата — пропускаем (не догоняем)
                chase = abs(cur.price - cand.ref_price) / cand.ref_price * 100.0 if cand.ref_price > 0 else 999.0
                if chase > MAX_CHASE_PCT:
                    to_delete.append(sym)
                    continue

                # 3.2) отправляем сигнал с зоной входа (лимитка) и стопом за структуру
                tg_send(build_message_ru(cand, cur.price))
                mark_sent(sym)
                confirmed += 1
                to_delete.append(sym)

            for sym in to_delete:
                pending.pop(sym, None)

            print(f"Tick: top={len(top)} created={created} confirmed={confirmed} pending={len(pending)}", flush=True)

        except Exception as e:
            print("ERROR", repr(e), flush=True)
            try:
                tg_send(f"⚠️ Ошибка бота: {repr(e)[:900]}")
            except Exception:
                pass

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
