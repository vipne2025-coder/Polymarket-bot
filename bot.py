import os
import time
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
# User settings (под тебя)
# =========================
EQUITY_USD = float(os.environ.get("EQUITY_USD", "200"))
LEVERAGE = float(os.environ.get("LEVERAGE", "15"))  # ты хотел 15

# Рекомендованные объёмы (notional)
NOTIONAL_FAST = float(os.environ.get("NOTIONAL_FAST", "20"))
NOTIONAL_TREND = float(os.environ.get("NOTIONAL_TREND", "20"))
NOTIONAL_BOOST = float(os.environ.get("NOTIONAL_BOOST", "50"))

# Late-entry: не входить, если цена убежала слишком далеко за время подтверждения
MAX_ENTRY_SLIPPAGE_PCT = float(os.environ.get("MAX_ENTRY_SLIPPAGE_PCT", "0.006"))  # 0.6%

# =========================
# Bot settings
# =========================
BYBIT_BASE = os.environ.get("BYBIT_BASE", "https://api.bybit.com").rstrip("/")
POLL_SECONDS = float(os.environ.get("POLL_SECONDS", "10"))
TOP_N = int(os.environ.get("TOP_N", "20"))

CONFIRM_DELAY_SEC = int(os.environ.get("CONFIRM_DELAY_SEC", "25"))
SYMBOL_COOLDOWN_SEC = int(os.environ.get("SYMBOL_COOLDOWN_SEC", "600"))  # 10 мин

# Lookbacks (строим из накопленной истории тикеров)
OI_LOOKBACK_SEC = int(os.environ.get("OI_LOOKBACK_SEC", "300"))         # 5м
PRICE_LOOKBACK_SEC = int(os.environ.get("PRICE_LOOKBACK_SEC", "120"))   # 2м

MAX_SPREAD_PCT = float(os.environ.get("MAX_SPREAD_PCT", "0.12"))         # 0.12% спред максимум

# Funding risk flags (не блокируем, но помечаем)
FUNDING_HIGH = float(os.environ.get("FUNDING_HIGH", "0.0002"))           # ~0.02%
FUNDING_LOW = float(os.environ.get("FUNDING_LOW", "-0.0002"))

# Kline confirm (объём + ATR% для SL/TP)
USE_KLINE_CONFIRM = int(os.environ.get("USE_KLINE_CONFIRM", "1"))
KLINE_INTERVAL = os.environ.get("KLINE_INTERVAL", "1")                  # 1m
KLINE_LIMIT = int(os.environ.get("KLINE_LIMIT", "60"))                  # 60 минут
VOL_SPIKE_MULT = float(os.environ.get("VOL_SPIKE_MULT", "1.6"))          # всплеск объёма

# =========================
# Классы сигналов (FAST / TREND / BOOST)
# Пороги можно менять env'ом, но дефолты рабочие.
# =========================
FAST_PRICE_MOVE_PCT = float(os.environ.get("FAST_PRICE_MOVE_PCT", "0.70"))    # ΔPrice% за ~2м
FAST_OI_CHANGE_PCT = float(os.environ.get("FAST_OI_CHANGE_PCT", "1.20"))      # ΔOI% за ~5м

TREND_PRICE_MOVE_PCT = float(os.environ.get("TREND_PRICE_MOVE_PCT", "1.00"))
TREND_OI_CHANGE_PCT = float(os.environ.get("TREND_OI_CHANGE_PCT", "1.80"))

BOOST_PRICE_MOVE_PCT = float(os.environ.get("BOOST_PRICE_MOVE_PCT", "1.20"))
BOOST_OI_CHANGE_PCT = float(os.environ.get("BOOST_OI_CHANGE_PCT", "2.50"))

# SL/TP в % (именно проценты от цены, не плечо)
FAST_SL_PCT = float(os.environ.get("FAST_SL_PCT", "0.70"))
FAST_TP1_PCT = float(os.environ.get("FAST_TP1_PCT", "1.20"))
FAST_TP2_PCT = float(os.environ.get("FAST_TP2_PCT", "2.20"))

TREND_SL_PCT = float(os.environ.get("TREND_SL_PCT", "1.20"))
TREND_TP1_PCT = float(os.environ.get("TREND_TP1_PCT", "2.50"))
TREND_TP2_PCT = float(os.environ.get("TREND_TP2_PCT", "5.00"))

BOOST_SL_PCT = float(os.environ.get("BOOST_SL_PCT", "1.50"))
BOOST_TP1_PCT = float(os.environ.get("BOOST_TP1_PCT", "3.50"))
BOOST_TP2_PCT = float(os.environ.get("BOOST_TP2_PCT", "7.00"))

# Ограничения SL (если ATR отдаст странное значение)
SL_PCT_MIN = float(os.environ.get("SL_PCT_MIN", "0.35"))
SL_PCT_MAX = float(os.environ.get("SL_PCT_MAX", "2.00"))

# Если объёма нет (vol_ok=False) — не блокируем полностью, а понижаем класс.
DOWNGRADE_ON_NO_VOLUME = int(os.environ.get("DOWNGRADE_ON_NO_VOLUME", "1"))  # 1 = да

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


def fmt_pct_val(x: float, nd: int = 2) -> str:
    return f"{x:.{nd}f}%"


def fmt_price(x: float) -> str:
    if x >= 100:
        return f"{x:.2f}"
    if x >= 1:
        return f"{x:.4f}"
    return f"{x:.6f}"


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
    tier: str               # FAST/TREND/BOOST
    created_ts: int
    ref_price: float
    oi_pct: float
    price_pct: float
    funding: float
    spread_pct: float
    turnover24h: float
    vol_ok: Optional[bool] = None
    atr_sl_pct: Optional[float] = None  # ATR-derived SL%


history: Dict[str, List[Point]] = {}
pending: Dict[str, Candidate] = {}
last_sent: Dict[str, int] = {}


def prune(sym: str, keep_sec: int = 1800) -> None:
    now = int(time.time())
    arr = history.get(sym, [])
    history[sym] = [p for p in arr if now - p.ts <= keep_sec]


def pick_top_symbols(tickers: List[dict], n: int) -> List[dict]:
    lst = []
    for t in tickers:
        sym = str(t.get("symbol", ""))
        if sym.endswith("USDT"):
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


def decide_direction(price_pct: float) -> str:
    return "LONG" if price_pct >= 0 else "SHORT"


def funding_flag_ru(funding: float, direction: str) -> str:
    if direction == "LONG" and funding >= FUNDING_HIGH:
        return "⚠️ funding высокий (лонги перегреты)"
    if direction == "SHORT" and funding <= FUNDING_LOW:
        return "⚠️ funding низкий (шорты перегреты)"
    return "ok"


def kline_volume_and_atr_pct(symbol: str) -> Tuple[Optional[bool], Optional[float]]:
    """
    Возвращает:
    - vol_ok: последняя 1m свеча по объёму >= VOL_SPIKE_MULT * avg(предыдущие N)
    - atr_pct: ATR(14) в % (можно использовать как SL%)
    """
    try:
        kl = get_kline(symbol)
        if not kl or len(kl) < 20:
            return None, None

        rows = []
        for r in kl:
            if len(r) < 7:
                continue
            ts = int(r[0]) // 1000
            o, h, l, c, v = map(float, [r[1], r[2], r[3], r[4], r[5]])
            rows.append((ts, o, h, l, c, v))
        rows.sort(key=lambda x: x[0])
        if len(rows) < 20:
            return None, None

        vols = [x[5] for x in rows[-30:]]
        last_v = vols[-1]
        avg_v = sum(vols[:-1]) / max(1, (len(vols) - 1))
        vol_ok = (avg_v > 0) and (last_v >= VOL_SPIKE_MULT * avg_v)

        tr = []
        for i in range(1, len(rows)):
            prev_c = rows[i - 1][4]
            h = rows[i][2]
            l = rows[i][3]
            tr.append(max(h - l, abs(h - prev_c), abs(l - prev_c)))
        atr = sum(tr[-14:]) / max(1, len(tr[-14:]))
        last_price = rows[-1][4]
        atr_pct = (atr / last_price) * 100.0 if last_price > 0 else None

        return vol_ok, atr_pct
    except Exception as e:
        print("kline_error", symbol, repr(e), flush=True)
        return None, None


def spread_pct(bid: float, ask: float) -> float:
    if bid <= 0 or ask <= 0:
        return 999.0
    mid = (bid + ask) / 2.0
    if mid <= 0:
        return 999.0
    return abs(ask - bid) / mid * 100.0


def cooldown_ok(sym: str) -> bool:
    now = int(time.time())
    return (now - last_sent.get(sym, 0)) >= SYMBOL_COOLDOWN_SEC


def mark_sent(sym: str) -> None:
    last_sent[sym] = int(time.time())


def classify_tier(price_pct: float, oi_pct: float) -> Optional[str]:
    ap = abs(price_pct)
    ao = abs(oi_pct)

    # BOOST
    if ap >= BOOST_PRICE_MOVE_PCT and ao >= BOOST_OI_CHANGE_PCT:
        return "BOOST"
    # TREND
    if ap >= TREND_PRICE_MOVE_PCT and ao >= TREND_OI_CHANGE_PCT:
        return "TREND"
    # FAST
    if ap >= FAST_PRICE_MOVE_PCT and ao >= FAST_OI_CHANGE_PCT:
        return "FAST"

    return None


def downgrade_tier(tier: str) -> str:
    if tier == "BOOST":
        return "TREND"
    if tier == "TREND":
        return "FAST"
    return tier


def tier_params(tier: str) -> Tuple[str, float, float, float, float]:
    """
    Возвращает:
    tag_text, notional, sl_pct, tp1_pct, tp2_pct
    """
    if tier == "BOOST":
        return "🚀 BOOST", NOTIONAL_BOOST, BOOST_SL_PCT, BOOST_TP1_PCT, BOOST_TP2_PCT
    if tier == "TREND":
        return "📈 TREND", NOTIONAL_TREND, TREND_SL_PCT, TREND_TP1_PCT, TREND_TP2_PCT
    return "⚡ FAST", NOTIONAL_FAST, FAST_SL_PCT, FAST_TP1_PCT, FAST_TP2_PCT


def calc_levels(entry: float, direction: str, sl_pct: float, tp1_pct: float, tp2_pct: float) -> Tuple[float, float, float]:
    if direction == "LONG":
        sl = entry * (1.0 - sl_pct / 100.0)
        tp1 = entry * (1.0 + tp1_pct / 100.0)
        tp2 = entry * (1.0 + tp2_pct / 100.0)
    else:
        sl = entry * (1.0 + sl_pct / 100.0)
        tp1 = entry * (1.0 - tp1_pct / 100.0)
        tp2 = entry * (1.0 - tp2_pct / 100.0)
    return sl, tp1, tp2


def build_message_ru(c: Candidate, entry_price: float) -> str:
    tag_text, notional, base_sl_pct, tp1_pct, tp2_pct = tier_params(c.tier)

    # SL%: если ATR есть — используем как основу, но в пределах
    sl_pct = c.atr_sl_pct if c.atr_sl_pct is not None else base_sl_pct
    sl_pct = clamp(sl_pct, SL_PCT_MIN, SL_PCT_MAX)

    sl, tp1, tp2 = calc_levels(entry_price, c.direction, sl_pct, tp1_pct, tp2_pct)

    margin = notional / LEVERAGE
    risk_usd = notional * (sl_pct / 100.0)

    dir_ru = "🟢 ЛОНГ" if c.direction == "LONG" else "🔴 ШОРТ"
    fund_note = funding_flag_ru(c.funding, c.direction)

    vol_line = ""
    if c.vol_ok is True:
        vol_line = "• Объём: всплеск ✅"
    elif c.vol_ok is False:
        vol_line = "• Объём: нет всплеска ⚠️"

    factors = (
        f"• ΔOI (~5м): {fmt_pct_val(c.oi_pct, 2)}\n"
        f"• ΔЦена (~2м): {fmt_pct_val(c.price_pct, 2)}\n"
        f"• Funding: {c.funding:.6f} ({fund_note})\n"
        f"• Спред: {fmt_pct_val(c.spread_pct, 3)}\n"
        f"{vol_line}"
    )

    return (
        f"{tag_text} | {c.symbol} (Bybit USDT-PERP)\n"
        f"{dir_ru}\n\n"
        f"💼 Рекомендация: ${notional:.0f} (плечо x{LEVERAGE:.0f}, маржа ≈ ${margin:.2f})\n\n"
        f"🎯 Вход: {fmt_price(entry_price)}\n"
        f"⛔ Стоп-лосс: {fmt_price(sl)} ({fmt_pct_val(sl_pct, 2)} | риск ≈ ${risk_usd:.2f})\n"
        f"✅ Тейк-профит 1: {fmt_price(tp1)} ({fmt_pct_val(tp1_pct, 2)})\n"
        f"✅ Тейк-профит 2: {fmt_price(tp2)} ({fmt_pct_val(tp2_pct, 2)})\n\n"
        f"📌 Факторы:\n{factors}\n\n"
        f"⚠️ Правило: пропустить вход, если цена сдвинулась > {MAX_ENTRY_SLIPPAGE_PCT*100:.2f}% за окно подтверждения"
    )


def main() -> None:
    tg_send(
        "✅ Bybit Futures Signal Bot запущен\n"
        f"TOP {TOP_N}, опрос={POLL_SECONDS}s, подтверждение={CONFIRM_DELAY_SEC}s\n"
        f"Депозит=${EQUITY_USD:.0f}, плечо=x{LEVERAGE:.0f}\n"
        f"Классы: FAST ${NOTIONAL_FAST:.0f} | TREND ${NOTIONAL_TREND:.0f} | BOOST ${NOTIONAL_BOOST:.0f}"
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
                    ts=now,
                    price=price,
                    oi_value=oi_val,
                    funding=funding,
                    bid=bid,
                    ask=ask,
                    turnover24h=turnover
                )
                history.setdefault(sym, []).append(p)
                prune(sym)

            # 2) create candidates (FAST/TREND/BOOST), но отправка только после подтверждения
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

                # cooldown (чтобы не спамить одним символом)
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
                    turnover24h=cur.turnover24h
                )

                # kline confirm (volume + ATR)
                if USE_KLINE_CONFIRM:
                    vol_ok, atr_pct = kline_volume_and_atr_pct(sym)
                    cand.vol_ok = vol_ok
                    if atr_pct is not None:
                        cand.atr_sl_pct = atr_pct

                    # если объёма нет — понижаем класс (а не “в ноль”), чтобы сигналы не исчезали
                    if DOWNGRADE_ON_NO_VOLUME and (vol_ok is False):
                        cand.tier = downgrade_tier(cand.tier)

                pending[sym] = cand
                created += 1

            # 3) confirm candidates and send
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

                # late-entry check
                move = abs(cur.price - cand.ref_price) / cand.ref_price if cand.ref_price > 0 else 0.0
                if move > MAX_ENTRY_SLIPPAGE_PCT:
                    to_delete.append(sym)
                    continue

                tg_send(build_message_ru(cand, cur.price))
                mark_sent(sym)
                confirmed += 1
                to_delete.append(sym)

            for sym in to_delete:
                pending.pop(sym, None)

            print(
                f"Tick: top={len(top)} created={created} confirmed={confirmed} pending={len(pending)}",
                flush=True
            )

        except Exception as e:
            print("ERROR", repr(e), flush=True)
            try:
                tg_send(f"⚠️ Ошибка бота: {repr(e)[:900]}")
            except Exception:
                pass

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
