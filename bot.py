import os
import time
import math
import requests
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# =========================
# Telegram (обязательно)
# =========================
TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TG_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
if not TG_TOKEN or not TG_CHAT_ID:
    raise RuntimeError("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")

TG_SEND_URL = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"

def tg_send(text: str) -> None:
    # телега лимит ~4096
    if len(text) > 3800:
        text = text[:3800] + "\n...[truncated]"
    r = requests.post(
        TG_SEND_URL,
        data={"chat_id": TG_CHAT_ID, "text": text, "disable_web_page_preview": "true"},
        timeout=20,
    )
    if r.status_code != 200:
        print("TG_ERROR", r.status_code, r.text[:200], flush=True)

# =========================
# Настройки под тебя
# =========================
EQUITY_USD = float(os.environ.get("EQUITY_USD", "200"))
LEVERAGE = float(os.environ.get("LEVERAGE", "15"))

# Две “ставки”
NOTIONAL_TREND = float(os.environ.get("NOTIONAL_TREND", "20"))   # спокойнее
NOTIONAL_BOOST = float(os.environ.get("NOTIONAL_BOOST", "50"))   # агрессивнее

# =========================
# Бот / опрос
# =========================
BYBIT_BASE = os.environ.get("BYBIT_BASE", "https://api.bybit.com").rstrip("/")
POLL_SECONDS = float(os.environ.get("POLL_SECONDS", "10"))
TOP_N = int(os.environ.get("TOP_N", "20"))

# подтверждение / сроки
CONFIRM_DELAY_SEC = int(os.environ.get("CONFIRM_DELAY_SEC", "20"))
SIGNAL_TTL_SEC = int(os.environ.get("SIGNAL_TTL_SEC", "1800"))
SYMBOL_COOLDOWN_SEC = int(os.environ.get("SYMBOL_COOLDOWN_SEC", "1200"))

# =========================
# Пороговые значения (баланс “есть сигналы / меньше мусора”)
# =========================
# TREND (мягче)
TREND_PRICE_MOVE_PCT = float(os.environ.get("TREND_PRICE_MOVE_PCT", "1.2"))  # ~2m
TREND_OI_CHANGE_PCT = float(os.environ.get("TREND_OI_CHANGE_PCT", "2.0"))    # ~5m

# BOOST (жестче)
BOOST_PRICE_MOVE_PCT = float(os.environ.get("BOOST_PRICE_MOVE_PCT", "1.8"))
BOOST_OI_CHANGE_PCT = float(os.environ.get("BOOST_OI_CHANGE_PCT", "3.0"))

# вход по откату (лимитка)
PULLBACK_MIN_PCT = float(os.environ.get("PULLBACK_MIN_PCT", "0.30"))  # от импульса
PULLBACK_MAX_PCT = float(os.environ.get("PULLBACK_MAX_PCT", "0.70"))
MAX_CHASE_PCT = float(os.environ.get("MAX_CHASE_PCT", "1.20"))        # если убежало — пропуск

# структура/тренд (простая, но рабочая)
STRUCT_WINDOW_MIN = int(os.environ.get("STRUCT_WINDOW_MIN", "20"))  # минут истории (kline)
SCORE_MIN = float(os.environ.get("SCORE_MIN", "65"))
SCORE_MIN_FLAT = float(os.environ.get("SCORE_MIN_FLAT", "75"))

# риск/SL/TP (ATR)
ATR_MULT_BUFFER = float(os.environ.get("ATR_MULT_BUFFER", "0.25"))  # добавка к ATR%
SL_PCT_MIN = float(os.environ.get("SL_PCT_MIN", "1.20"))
SL_PCT_MAX = float(os.environ.get("SL_PCT_MAX", "3.50"))

TP1_R = float(os.environ.get("TP1_R", "1.6"))
TP2_R = float(os.environ.get("TP2_R", "3.0"))

# ликвидность
MAX_SPREAD_PCT = float(os.environ.get("MAX_SPREAD_PCT", "0.08"))  # 0.08%

# kline/volume
USE_KLINE = int(os.environ.get("USE_KLINE", "1"))
KLINE_INTERVAL = os.environ.get("KLINE_INTERVAL", "1")
KLINE_LIMIT = int(os.environ.get("KLINE_LIMIT", "240"))
VOL_SPIKE_MULT = float(os.environ.get("VOL_SPIKE_MULT", "1.8"))

# orderbook (очень помогает выкинуть “мусор”)
USE_ORDERBOOK = int(os.environ.get("USE_ORDERBOOK", "1"))
OB_LIMIT = int(os.environ.get("OB_LIMIT", "25"))

# lookbacks (строим из тикеров)
OI_LOOKBACK_SEC = int(os.environ.get("OI_LOOKBACK_SEC", "300"))       # 5m
PRICE_LOOKBACK_SEC = int(os.environ.get("PRICE_LOOKBACK_SEC", "120")) # 2m

# funding только метка (не блокируем)
FUNDING_HIGH = float(os.environ.get("FUNDING_HIGH", "0.0002"))
FUNDING_LOW = float(os.environ.get("FUNDING_LOW", "-0.0002"))

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

def fmt_pct(x: float, nd=2) -> str:
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

def decide_direction(price_pct: float) -> str:
    return "ЛОНГ" if price_pct >= 0 else "ШОРТ"

def funding_flag(funding: float, direction_ru: str) -> str:
    # direction_ru: ЛОНГ/ШОРТ
    if direction_ru == "ЛОНГ" and funding >= FUNDING_HIGH:
        return "⚠️ высокое (лонги перегреты)"
    if direction_ru == "ШОРТ" and funding <= FUNDING_LOW:
        return "⚠️ низкое (шорты перегреты)"
    return "ok"

def now_ts() -> int:
    return int(time.time())

# =========================
# Data structures
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
class Signal:
    symbol: str
    mode: str              # TREND / BOOST
    direction_ru: str      # ЛОНГ / ШОРТ
    created_ts: int

    # impulse metrics
    oi_pct_5m: float
    price_pct_2m: float
    funding: float
    spread: float

    # confirm details
    entry_low: float
    entry_high: float
    stop_price: float
    tp1: float
    tp2: float
    sl_pct: float

    # filters/factors
    vol_spike: Optional[bool]
    ob_imbalance: Optional[float]   # + = bids stronger, - = asks stronger
    structure_ok: Optional[bool]
    score: float

# =========================
# State
# =========================
history: Dict[str, List[Point]] = {}
pending: Dict[str, Tuple[Signal, float]] = {}  # Signal + ref_price_at_create
last_sent: Dict[str, int] = {}

def prune(sym: str, keep_sec: int = 3600) -> None:
    now = now_ts()
    arr = history.get(sym, [])
    history[sym] = [p for p in arr if now - p.ts <= keep_sec]

def pick_top_symbols(tickers: List[dict], n: int) -> List[dict]:
    lst = []
    for t in tickers:
        sym = str(t.get("symbol", ""))
        if not sym.endswith("USDT"):
            continue
        lst.append(t)
    lst.sort(key=lambda x: f(x.get("turnover24h", 0.0)), reverse=True)
    return lst[:n]

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
    return (now_ts() - last_sent.get(sym, 0)) >= SYMBOL_COOLDOWN_SEC

def mark_sent(sym: str) -> None:
    last_sent[sym] = now_ts()

# =========================
# Kline analytics: Volume spike + ATR% + structure
# =========================
def kline_features(symbol: str) -> Tuple[Optional[bool], Optional[float], Optional[bool], float]:
    """
    Returns:
      vol_spike: last volume >= VOL_SPIKE_MULT * avg(prev)
      atr_pct: ATR(14) in %
      structure_ok: direction aligned with local structure (set later with direction)
      trend_strength_score: 0..100 (rough)
    """
    if not USE_KLINE:
        return None, None, None, 0.0

    try:
        kl = get_kline(symbol)
        if not kl or len(kl) < 40:
            return None, None, None, 0.0

        rows = []
        for r in kl:
            if len(r) < 7:
                continue
            ts = int(r[0]) // 1000
            o, h, l, c, v = map(float, [r[1], r[2], r[3], r[4], r[5]])
            rows.append((ts, o, h, l, c, v))

        rows.sort(key=lambda x: x[0])
        if len(rows) < 40:
            return None, None, None, 0.0

        # volume spike (last 1m vs avg of previous 30)
        vols = [x[5] for x in rows[-31:]]
        last_v = vols[-1]
        avg_v = sum(vols[:-1]) / max(1, (len(vols) - 1))
        vol_spike = (avg_v > 0) and (last_v >= VOL_SPIKE_MULT * avg_v)

        # ATR(14)
        tr = []
        for i in range(1, len(rows)):
            prev_c = rows[i - 1][4]
            h = rows[i][2]
            l = rows[i][3]
            tr.append(max(h - l, abs(h - prev_c), abs(l - prev_c)))
        atr = sum(tr[-14:]) / 14.0
        last_price = rows[-1][4]
        atr_pct = (atr / last_price) * 100.0 if last_price > 0 else None

        # structure / trend strength (simple):
        # - slope of last STRUCT_WINDOW_MIN closes
        w = min(STRUCT_WINDOW_MIN, len(rows) - 1)
        closes = [x[4] for x in rows[-w:]]
        # linear slope approx:
        xmean = (w - 1) / 2.0
        ymean = sum(closes) / w
        num = sum((i - xmean) * (closes[i] - ymean) for i in range(w))
        den = sum((i - xmean) ** 2 for i in range(w)) + 1e-9
        slope = num / den  # price units per minute
        slope_pct = (slope / closes[-1]) * 100.0 if closes[-1] else 0.0

        # normalize into score 0..100
        # bigger |slope_pct| => stronger structure
        strength = min(100.0, abs(slope_pct) * 800.0)  # tune
        return vol_spike, atr_pct, None, strength

    except Exception as e:
        print("kline_error", symbol, repr(e), flush=True)
        return None, None, None, 0.0

# =========================
# Orderbook analytics
# =========================
def orderbook_imbalance(symbol: str) -> Optional[float]:
    """
    Returns imbalance in [-1..+1]:
      +1 => bids dominate
      -1 => asks dominate
    """
    if not USE_ORDERBOOK:
        return None
    try:
        ob = get_orderbook(symbol)
        bids = ob.get("b", []) or []
        asks = ob.get("a", []) or []
        if not bids or not asks:
            return None

        bid_val = 0.0
        ask_val = 0.0
        # sum price*size (approx depth)
        for p, q in bids[:OB_LIMIT]:
            bid_val += float(p) * float(q)
        for p, q in asks[:OB_LIMIT]:
            ask_val += float(p) * float(q)

        s = bid_val + ask_val
        if s <= 0:
            return None
        return (bid_val - ask_val) / s
    except Exception as e:
        print("ob_error", symbol, repr(e), flush=True)
        return None

# =========================
# Trade plan
# =========================
def calc_sl_tp(entry_mid: float, direction_ru: str, sl_pct: float) -> Tuple[float, float, float]:
    tp1_pct = sl_pct * TP1_R
    tp2_pct = sl_pct * TP2_R
    if direction_ru == "ЛОНГ":
        sl = entry_mid * (1.0 - sl_pct / 100.0)
        tp1 = entry_mid * (1.0 + tp1_pct / 100.0)
        tp2 = entry_mid * (1.0 + tp2_pct / 100.0)
    else:
        sl = entry_mid * (1.0 + sl_pct / 100.0)
        tp1 = entry_mid * (1.0 - tp1_pct / 100.0)
        tp2 = entry_mid * (1.0 - tp2_pct / 100.0)
    return sl, tp1, tp2

def entry_zone_from_impulse(cur_price: float, direction_ru: str, impulse_pct: float) -> Tuple[float, float]:
    """
    Вход только по откату:
    - impulse_pct: |ΔPrice| за 2м
    Для ЛОНГ: ждём откат вниз от текущей цены
    Для ШОРТ: ждём откат вверх от текущей цены
    """
    # откат как доля от импульса
    pb_min = impulse_pct * (PULLBACK_MIN_PCT / 100.0)
    pb_max = impulse_pct * (PULLBACK_MAX_PCT / 100.0)

    if direction_ru == "ЛОНГ":
        # зона ниже текущей
        high = cur_price * (1.0 - pb_min / 100.0)
        low = cur_price * (1.0 - pb_max / 100.0)
        return min(low, high), max(low, high)
    else:
        # зона выше текущей
        low = cur_price * (1.0 + pb_min / 100.0)
        high = cur_price * (1.0 + pb_max / 100.0)
        return min(low, high), max(low, high)

def score_signal(mode: str,
                 oi_pct: float,
                 price_pct: float,
                 vol_spike: Optional[bool],
                 ob_imb: Optional[float],
                 struct_strength: float,
                 direction_ru: str) -> float:
    """
    Score 0..100
    """
    s = 0.0

    # base: thresholds
    if mode == "BOOST":
        s += min(35.0, abs(price_pct) / BOOST_PRICE_MOVE_PCT * 20.0)
        s += min(35.0, abs(oi_pct) / BOOST_OI_CHANGE_PCT * 20.0)
    else:
        s += min(35.0, abs(price_pct) / TREND_PRICE_MOVE_PCT * 20.0)
        s += min(35.0, abs(oi_pct) / TREND_OI_CHANGE_PCT * 20.0)

    # volume
    if vol_spike is True:
        s += 12.0
    elif vol_spike is False:
        s -= 6.0

    # orderbook (aligned only)
    if ob_imb is not None:
        aligned = (ob_imb > 0 and direction_ru == "ЛОНГ") or (ob_imb < 0 and direction_ru == "ШОРТ")
        if aligned:
            s += 10.0 + min(8.0, abs(ob_imb) * 40.0)
        else:
            s -= 10.0

    # structure strength
    s += min(18.0, struct_strength / 100.0 * 18.0)

    return clamp(s, 0.0, 100.0)

def format_signal(sig: Signal) -> str:
    # “пустые строки” перед сигналом — чтобы не сливалось
    pad = "\n\n\n"

    notional = NOTIONAL_BOOST if sig.mode == "BOOST" else NOTIONAL_TREND
    margin = notional / LEVERAGE
    risk_usd = notional * (sig.sl_pct / 100.0)

    # метки
    dir_emoji = "🟢" if sig.direction_ru == "ЛОНГ" else "🔴"
    mode_emoji = "⚡" if sig.mode == "BOOST" else "📈"
    vol_line = "✅ объём: всплеск" if sig.vol_spike is True else ("⚠️ объём: без всплеска" if sig.vol_spike is False else "ℹ️ объём: n/a")

    ob_line = ""
    if sig.ob_imbalance is not None:
        obp = sig.ob_imbalance * 100.0
        ob_line = f"• Ордербук: дисбаланс {obp:+.1f}%"

    struct_line = ""
    if sig.structure_ok is not None:
        struct_line = "✅ структура: ок" if sig.structure_ok else "⚠️ структура: сомнительно"

    funding_line = funding_flag(sig.funding, sig.direction_ru)

    # ВАЖНО: “вход только лимиткой”
    # (текстом, как на твоём скрине)
    return (
        pad +
        f"{mode_emoji} {sig.mode} | {sig.symbol} (Bybit USDT-PERP)\n"
        f"{dir_emoji} {sig.direction_ru}\n\n"
        f"✅ ВХОД ТОЛЬКО ЛИМИТКОЙ (по откату)\n"
        f"🎯 Зона входа: {fmt_price(sig.entry_low)} – {fmt_price(sig.entry_high)}\n"
        f"⛔ Стоп-лосс: {fmt_price(sig.stop_price)} ({fmt_pct(sig.sl_pct,2)} | risk≈${risk_usd:.2f})\n"
        f"✅ Тейк 1: {fmt_price(sig.tp1)} ({fmt_pct(sig.sl_pct*TP1_R,2)})\n"
        f"✅ Тейк 2: {fmt_price(sig.tp2)} ({fmt_pct(sig.sl_pct*TP2_R,2)})\n\n"
        f"💵 Рекомендация: ${notional:.0f} (плечо x{LEVERAGE:.0f}, маржа≈${margin:.2f})\n\n"
        f"📌 Факторы:\n"
        f"• ΔOI (~5м): {fmt_pct(sig.oi_pct_5m,2)}\n"
        f"• ΔЦена (~2м): {fmt_pct(sig.price_pct_2m,2)}\n"
        f"• Funding: {sig.funding:.6f} ({funding_line})\n"
        f"• Спред: {fmt_pct(sig.spread,3)}\n"
        f"• {vol_line}\n"
        f"{(ob_line + chr(10)) if ob_line else ''}"
        f"{(struct_line + chr(10)) if struct_line else ''}\n"
        f"🧠 Score: {sig.score:.0f}/100\n\n"
        f"⚠️ Если вход не дали — ПРОПУСКАЕШЬ (это нормально).\n"
    )

# =========================
# Main logic
# =========================
def main() -> None:
    tg_send(
        "✅ Bybit Futures STRONG Bot запущен\n"
        f"Top={TOP_N}, poll={POLL_SECONDS}s, confirm={CONFIRM_DELAY_SEC}s\n"
        f"Плечо x{LEVERAGE:.0f} | Депо≈${EQUITY_USD:.0f} | TREND=${NOTIONAL_TREND:.0f} | BOOST=${NOTIONAL_BOOST:.0f}\n"
        "Вход: только лимиткой по откату (зона входа)."
    )

    while True:
        now = now_ts()
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

                history.setdefault(sym, []).append(Point(
                    ts=now, price=price, oi_value=oi_val, funding=funding,
                    bid=bid, ask=ask, turnover24h=turnover
                ))
                prune(sym)

            # 2) create candidates
            created = 0
            for t in top:
                sym = str(t.get("symbol"))
                pts = history.get(sym, [])
                if len(pts) < 5:
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

                # mode detect
                mode = None
                if abs(price_pct) >= BOOST_PRICE_MOVE_PCT and abs(oi_pct) >= BOOST_OI_CHANGE_PCT:
                    mode = "BOOST"
                elif abs(price_pct) >= TREND_PRICE_MOVE_PCT and abs(oi_pct) >= TREND_OI_CHANGE_PCT:
                    mode = "TREND"
                else:
                    continue

                if sym in pending:
                    continue
                if not cooldown_ok(sym):
                    continue

                direction_ru = decide_direction(price_pct)

                # ---- extra confirms (kline + orderbook) ----
                vol_spike, atr_pct, _, struct_strength = kline_features(sym)
                ob_imb = orderbook_imbalance(sym)

                # SL from ATR%
                if atr_pct is None:
                    # fallback if kline not available
                    sl_pct = 1.8 if mode == "BOOST" else 1.4
                else:
                    sl_pct = atr_pct + ATR_MULT_BUFFER

                sl_pct = clamp(sl_pct, SL_PCT_MIN, SL_PCT_MAX)

                # entry zone from impulse
                impulse = abs(price_pct)
                entry_low, entry_high = entry_zone_from_impulse(cur.price, direction_ru, impulse_pct=impulse)

                # chase check: если цена уже улетела, вход не догоняем
                # (для лонга: если текущая выше верхней границы зоны больше чем MAX_CHASE_PCT импульса — пропуск)
                # (для шорта: аналогично)
                # сделаем проще: если расстояние от цены до ближайшей границы > MAX_CHASE_PCT% — пропуск
                nearest = entry_high if direction_ru == "ЛОНГ" else entry_low
                chase_pct = abs(cur.price - nearest) / cur.price * 100.0
                if chase_pct > MAX_CHASE_PCT:
                    continue

                # structure ok (align with slope sign, using struct_strength only as proxy)
                # Если тренд слабый — требуем выше SCORE_MIN_FLAT
                # Если тренд норм — достаточно SCORE_MIN
                # (структуру как bool зададим по направлению импульса; иначе будет слишком сложно без slope sign)
                structure_ok = True if struct_strength >= 25 else None

                score = score_signal(mode, oi_pct, price_pct, vol_spike, ob_imb, struct_strength, direction_ru)

                min_score = SCORE_MIN_FLAT if (struct_strength < 25) else SCORE_MIN
                if score < min_score:
                    continue

                # kline volume не режем “в ноль”, но штрафуем через score (уже сделано)
                # orderbook против направления — тоже режет через score

                # build plan: entry_mid for sl/tp
                entry_mid = (entry_low + entry_high) / 2.0
                sl, tp1, tp2 = calc_sl_tp(entry_mid, direction_ru, sl_pct)

                sig = Signal(
                    symbol=sym,
                    mode=mode,
                    direction_ru=direction_ru,
                    created_ts=now,
                    oi_pct_5m=oi_pct,
                    price_pct_2m=price_pct,
                    funding=cur.funding,
                    spread=sp,
                    entry_low=entry_low,
                    entry_high=entry_high,
                    stop_price=sl,
                    tp1=tp1,
                    tp2=tp2,
                    sl_pct=sl_pct,
                    vol_spike=vol_spike,
                    ob_imbalance=ob_imb,
                    structure_ok=structure_ok,
                    score=score
                )

                pending[sym] = (sig, cur.price)
                created += 1

            # 3) confirm + send
            confirmed = 0
            to_delete = []

            for sym, (sig, ref_price) in list(pending.items()):
                # ttl
                if now - sig.created_ts > SIGNAL_TTL_SEC:
                    to_delete.append(sym)
                    continue

                if now - sig.created_ts < CONFIRM_DELAY_SEC:
                    continue

                pts = history.get(sym, [])
                if not pts:
                    to_delete.append(sym)
                    continue

                cur = pts[-1]

                # confirm window: не должно “убежать” от зоны
                # если цена ушла сильно от ref — значит поздно
                move_pct = abs(cur.price - ref_price) / ref_price * 100.0 if ref_price > 0 else 0.0
                # для BOOST разрешим чуть больше
                max_move = 0.8 if sig.mode == "TREND" else 1.1
                if move_pct > max_move:
                    to_delete.append(sym)
                    continue

                # финальная проверка спреда
                sp = spread_pct(cur.bid, cur.ask)
                if sp > MAX_SPREAD_PCT:
                    to_delete.append(sym)
                    continue

                tg_send(format_signal(sig))
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
