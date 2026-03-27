# =============================================================================
# backtest.py — EURUSD Strategy Backtester (canonical clean version)
#
# Usage:
#   python backtest.py                   # last 180 days via MT5
#   python backtest.py --days 365        # last 12 months
#   python backtest.py --no-mt5          # synthetic data, no MT5 needed
#   python backtest.py --from 2024-01-01 --to 2024-12-31
# =============================================================================

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("backtest")

# =============================================================================
# STRATEGY PARAMETERS
# =============================================================================

SYMBOL            = "XAUUSD"   # Gold — change back to EURUSD to switch

# Instrument-specific pip config
# EURUSD: pip = 0.0001, pip_value = $10/lot
# XAUUSD: pip = 0.01,   pip_value = $1/lot  (contract = 100oz, $0.01 move = $1)
PIP_SIZE          = 0.01    # XAUUSD: 1 pip = $0.01 (1 cent)
PIP_VALUE_PER_LOT = 1.0     # XAUUSD: $1 per pip per lot
                             # EURUSD: set PIP_SIZE=0.0001, PIP_VALUE_PER_LOT=10.0

# Indicators
EMA_FAST          = 9
EMA_MED           = 50
EMA_SLOW          = 200
RSI_PERIOD        = 14
ATR_PERIOD        = 14
ADX_PERIOD        = 14

# Signal filters
RSI_LONG_MIN      = 45       # RSI floor for longs
RSI_LONG_MAX      = 65       # RSI ceiling for longs
RSI_SHORT_MIN     = 35       # RSI floor for shorts
RSI_SHORT_MAX     = 55       # RSI ceiling for shorts
ADX_MIN           = 25       # H1 ADX minimum — 25 is the classic "trending" threshold
                              # Below 25 = ranging market, EMA crosses unreliable
EMA_SEP_PIPS      = 0.5      # Min pip separation — slightly stricter cross confirmation

# Weekly trend alignment — H4 EMA stack must agree with H1 direction
# This prevents trading counter to the higher timeframe bias
USE_H4_FILTER     = True     # Require H4 EMA50 > EMA200 for longs, < for shorts

# Trade sizing — pure 1:3 RR, no partial close, no breakeven
SL_ATR_MULT       = 2.0      # SL = ATR × 2.0  (Gold is spikier than EURUSD — needs more room)
REWARD_RATIO      = 3.0      # TP = SL × 3  →  1:3 RR
RISK_PER_TRADE    = 0.015    # 1.5% risk per trade

# Session (UTC hours) — London open through NY close
SESSION_START     = 7
SESSION_END       = 20

# Daily limits
MAX_DAILY_LOSS    = 0.02     # Kill switch at -2% daily drawdown
MAX_OPEN_TRADES   = 3        # Max concurrent positions

# Costs
COMMISSION_USD    = 5.0      # Round-trip commission per lot (Gold typical)
SPREAD_PIPS       = 40       # XAUUSD typical spread in pips ($0.40 = 40 cent-pips)
                             # EURUSD: use 1.5
INITIAL_BALANCE   = 10_000.0

# Output
OUTPUT_FILE       = "backtest_results.json"

# =============================================================================
# INDICATORS
# =============================================================================

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hi, lo = df["high"], df["low"]
    up       = hi - hi.shift(1)
    down     = lo.shift(1) - lo
    plus_dm  = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    atr_s    = atr(df, period)
    plus_di  = (100 * pd.Series(plus_dm,  index=df.index)
                    .ewm(alpha=1 / period, adjust=False).mean() / atr_s)
    minus_di = (100 * pd.Series(minus_dm, index=df.index)
                    .ewm(alpha=1 / period, adjust=False).mean() / atr_s)
    dx = (100 * (plus_di - minus_di).abs()
              / (plus_di + minus_di).replace(0, np.nan))
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["ema9"]       = ema(d["close"], EMA_FAST)
    d["ema50"]      = ema(d["close"], EMA_MED)
    d["ema200"]     = ema(d["close"], EMA_SLOW)
    d["rsi"]        = rsi(d["close"], RSI_PERIOD)
    d["atr"]        = atr(d, ATR_PERIOD)
    d["adx"]        = adx(d, ADX_PERIOD)
    d["cross_up"]   = (d["ema9"] >  d["ema50"]) & (d["ema9"].shift(1) <= d["ema50"].shift(1))
    d["cross_down"] = (d["ema9"] <  d["ema50"]) & (d["ema9"].shift(1) >= d["ema50"].shift(1))
    return d

# =============================================================================
# DATA
# =============================================================================

def fetch_mt5(start: datetime, end: datetime, days: int = 180) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        import MetaTrader5 as mt5
    except ImportError:
        logger.error("MetaTrader5 not installed. Use --no-mt5 for synthetic data.")
        sys.exit(1)

    if not mt5.initialize():
        logger.error(f"MT5 init failed: {mt5.last_error()}")
        sys.exit(1)

    def _get(tf, label, min_bars):
        rates = mt5.copy_rates_range(SYMBOL, tf, start, end)
        if rates is None or len(rates) < min_bars:
            logger.warning(
                f"{label}: only {len(rates) if rates is not None else 0} bars from date range "
                f"— falling back to copy_rates_from_pos"
            )
            needed = {mt5.TIMEFRAME_H1: 24 * days, mt5.TIMEFRAME_H4: 6 * days, mt5.TIMEFRAME_M15: 96 * days}.get(tf, days * 24)
            rates = mt5.copy_rates_from_pos(SYMBOL, tf, 0, needed + 50)

        if rates is None or len(rates) == 0:
            logger.error(f"No {label} data: {mt5.last_error()}")
            mt5.shutdown()
            sys.exit(1)

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("time")

        # Trim to requested date window — prevents H4 fallback returning years of extra data
        df = df[df.index >= pd.Timestamp(start).tz_localize("UTC") if start.tzinfo is None
                else df.index >= pd.Timestamp(start)]
        df = df[df.index <= pd.Timestamp(end).tz_localize("UTC") if end.tzinfo is None
                else df.index <= pd.Timestamp(end)]

        actual_days = (df.index[-1] - df.index[0]).days if len(df) > 1 else 0
        if actual_days < days * 0.7:
            logger.warning(f"{label}: only {actual_days} days available (requested {days})")
        else:
            logger.info(f"{label}: {len(df):,} bars  {df.index[0].date()} → {df.index[-1].date()}")
        return df

    h1  = _get(mt5.TIMEFRAME_H1,  "H1",  min_bars=days * 16)
    h4  = _get(mt5.TIMEFRAME_H4,  "H4",  min_bars=days * 6)
    m15 = _get(mt5.TIMEFRAME_M15, "M15", min_bars=days * 67)
    mt5.shutdown()
    return h1, h4, m15


def make_synthetic(days: int = 180) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info(f"Generating {days} days of synthetic EURUSD data...")
    np.random.seed(42)

    def _build(n, freq_mins, sigma=0.0003):
        dt     = freq_mins / (252 * 24 * 60)
        noise  = np.random.normal(0, sigma * np.sqrt(dt), n)
        trend  = 0.00003 * np.sin(np.linspace(0, 10 * np.pi, n))
        prices = 1.0850 * np.exp(np.cumsum(noise + trend))
        end_t  = datetime.now(timezone.utc)
        start_t = end_t - timedelta(days=days)
        idx    = pd.date_range(start_t, periods=n, freq=f"{freq_mins}min", tz="UTC")
        hi_n   = np.abs(np.random.normal(0, sigma * 0.8, n))
        lo_n   = np.abs(np.random.normal(0, sigma * 0.8, n))
        df = pd.DataFrame({
            "open":        prices * (1 + np.random.normal(0, sigma * 0.3, n)),
            "high":        prices * (1 + hi_n),
            "low":         prices * (1 - lo_n),
            "close":       prices,
            "tick_volume": np.random.randint(200, 2000, n),
        }, index=idx)
        df["high"] = df[["open", "high", "close"]].max(axis=1)
        df["low"]  = df[["open", "low",  "close"]].min(axis=1)
        return df

    h1  = _build(days * 24, 60)
    h4  = _build(days * 6,  240)
    m15 = _build(days * 96, 15)
    logger.info(f"Synthetic: {len(h1):,} H1 bars, {len(h4):,} H4 bars, {len(m15):,} M15 bars")
    return h1, h4, m15

# =============================================================================
# SIGNAL GENERATION
# =============================================================================

def generate_signals(h1_raw: pd.DataFrame, h4_raw: pd.DataFrame, m15_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Compute indicators on both timeframes, align H1 values to M15 bars
    using merge_asof (no double ffill), then apply all signal conditions.
    """
    h1  = compute_indicators(h1_raw)
    h4  = compute_indicators(h4_raw)
    m15 = compute_indicators(m15_raw)

    # ── Build H4 trend columns (highest timeframe bias) ───────────────────
    h4_signals = pd.DataFrame(index=h4.index)
    h4_signals["h4_bull"] = h4["ema50"] > h4["ema200"]   # H4 EMA50 above EMA200
    h4_signals["h4_bear"] = h4["ema50"] < h4["ema200"]   # H4 EMA50 below EMA200

    # ── Build H1 signal columns ────────────────────────────────────────────
    h1_signals = pd.DataFrame(index=h1.index)
    h1_signals["h1_bull"] = (
        (h1["close"] > h1["ema50"])  &
        (h1["close"] > h1["ema200"]) &
        (h1["ema50"] > h1["ema200"])
    )
    h1_signals["h1_bear"] = (
        (h1["close"] < h1["ema50"])  &
        (h1["close"] < h1["ema200"]) &
        (h1["ema50"] < h1["ema200"])
    )
    h1_signals["h1_adx_ok"] = h1["adx"] > ADX_MIN

    # ── Align H4 → M15 via merge_asof ────────────────────────────────────
    h4_reset  = h4_signals.reset_index()
    m15_reset = m15.reset_index()

    merged = pd.merge_asof(
        m15_reset.sort_values("time"),
        h4_reset.sort_values("time"),
        on="time",
        direction="backward",
    ).set_index("time")

    for col in ["h4_bull", "h4_bear"]:
        merged[col] = merged[col].fillna(False).astype(bool)

    # ── Align H1 → M15 via merge_asof ─────────────────────────────────────
    h1_reset  = h1_signals.reset_index()
    m15_with_h4 = merged.reset_index()

    merged = pd.merge_asof(
        m15_with_h4.sort_values("time"),
        h1_reset.sort_values("time"),
        on="time",
        direction="backward",
    ).set_index("time")

    for col in ["h1_bull", "h1_bear", "h1_adx_ok"]:
        merged[col] = merged[col].fillna(False).astype(bool)

    # ── Session filter ─────────────────────────────────────────────────────
    merged["in_session"] = (
        (merged.index.hour >= SESSION_START) &
        (merged.index.hour <  SESSION_END)
    )

    # ── EMA separation — same-bar confirmation ───────────────────────────
    # Check on the cross bar itself. Even 0.5 pip clearance on bar N
    # is enough to confirm the cross isn't a fleeting touch-and-reverse.
    # Using shift(-1) was a lookahead that killed ~95% of valid crosses.
    pip = PIP_SIZE
    sep = EMA_SEP_PIPS * pip    # e.g. 0.3 pip minimum — removes false touches
    merged["sep_ok_up"]   = (merged["ema9"] - merged["ema50"]) >= sep
    merged["sep_ok_down"] = (merged["ema50"] - merged["ema9"]) >= sep

    # ── Final signal conditions ────────────────────────────────────────────
    # H4 alignment — only trade in direction of higher timeframe trend
    h4_long_ok  = merged["h4_bull"] if USE_H4_FILTER else pd.Series(True, index=merged.index)
    h4_short_ok = merged["h4_bear"] if USE_H4_FILTER else pd.Series(True, index=merged.index)

    long_cond = (
        h4_long_ok           &    # H4 trend aligned (highest TF)
        merged["h1_bull"]    &    # H1 trend aligned
        merged["h1_adx_ok"]  &    # H1 ADX > 25 (trending market)
        merged["cross_up"]   &    # M15 EMA9 crossed above EMA50
        merged["sep_ok_up"]  &    # Cross is clean, not a false touch
        merged["rsi"].between(RSI_LONG_MIN, RSI_LONG_MAX) &
        merged["in_session"]
    )
    short_cond = (
        h4_short_ok          &    # H4 trend aligned
        merged["h1_bear"]    &    # H1 trend aligned
        merged["h1_adx_ok"]  &    # H1 ADX > 25
        merged["cross_down"] &    # M15 EMA9 crossed below EMA50
        merged["sep_ok_down"] &   # Clean cross
        merged["rsi"].between(RSI_SHORT_MIN, RSI_SHORT_MAX) &
        merged["in_session"]
    )

    merged["signal"] = 0
    merged.loc[long_cond,  "signal"] = 1
    merged.loc[short_cond, "signal"] = -1

    # ── Pre-calculate entry / SL / TP levels ──────────────────────────────
    spread = SPREAD_PIPS * PIP_SIZE
    merged["sl_dist"]     = merged["atr"] * SL_ATR_MULT
    merged["entry_long"]  = merged["close"] + spread
    merged["entry_short"] = merged["close"]
    merged["sl_long"]     = merged["entry_long"]  - merged["sl_dist"]
    merged["tp_long"]     = merged["entry_long"]  + merged["sl_dist"] * REWARD_RATIO
    merged["sl_short"]    = merged["entry_short"] + merged["sl_dist"]
    merged["tp_short"]    = merged["entry_short"] - merged["sl_dist"] * REWARD_RATIO

    # ── Debug: layered filter funnel ─────────────────────────────────────
    n = len(merged)
    rsi_long  = merged["rsi"].between(RSI_LONG_MIN,  RSI_LONG_MAX)
    rsi_short = merged["rsi"].between(RSI_SHORT_MIN, RSI_SHORT_MAX)

    # Long funnel — each line shows how many bars survive after adding that filter
    l1 = merged["cross_up"]
    l2 = l1 & merged["sep_ok_up"]
    l3 = l2 & merged["in_session"]
    l4 = l3 & merged["h1_bull"]
    l5 = l4 & merged["h1_adx_ok"]
    l6 = l5 & rsi_long
    l7 = l6 & h4_long_ok

    # Short funnel
    s1 = merged["cross_down"]
    s2 = s1 & merged["sep_ok_down"]
    s3 = s2 & merged["in_session"]
    s4 = s3 & merged["h1_bear"]
    s5 = s4 & merged["h1_adx_ok"]
    s6 = s5 & rsi_short
    s7 = s6 & h4_short_ok

    logger.info("── Signal filter funnel ─────────────────────────────")
    logger.info(f"  Total M15 bars        : {n:,}")
    logger.info(f"  H4 bullish bars       : {merged['h4_bull'].sum():,}")
    logger.info(f"  H4 bearish bars       : {merged['h4_bear'].sum():,}")
    logger.info(f"  H1 bullish bars       : {merged['h1_bull'].sum():,}")
    logger.info(f"  H1 bearish bars       : {merged['h1_bear'].sum():,}")
    logger.info(f"  H1 ADX ok (>{ADX_MIN})    : {merged['h1_adx_ok'].sum():,}")
    logger.info(f"  --- LONG funnel ---")
    logger.info(f"  Cross up              : {l1.sum():,}")
    logger.info(f"  + Sep ok              : {l2.sum():,}")
    logger.info(f"  + In session          : {l3.sum():,}")
    logger.info(f"  + H1 bull             : {l4.sum():,}")
    logger.info(f"  + ADX ok              : {l5.sum():,}")
    logger.info(f"  + RSI zone            : {l6.sum():,}")
    logger.info(f"  + H4 aligned → SIGNALS: {l7.sum():,}  ← final long signals")
    logger.info(f"  --- SHORT funnel ---")
    logger.info(f"  Cross down            : {s1.sum():,}")
    logger.info(f"  + Sep ok              : {s2.sum():,}")
    logger.info(f"  + In session          : {s3.sum():,}")
    logger.info(f"  + H1 bear             : {s4.sum():,}")
    logger.info(f"  + ADX ok              : {s5.sum():,}")
    logger.info(f"  + RSI zone            : {s6.sum():,}")
    logger.info(f"  + H4 aligned → SIGNALS: {s7.sum():,}  ← final short signals")
    logger.info("─────────────────────────────────────────────────────")

    return merged

# =============================================================================
# TRADE SIMULATION
# =============================================================================

def simulate(bars: pd.DataFrame) -> dict:
    """
    Walk bar-by-bar. Apply kill switch, position sizing, partial close,
    and breakeven. Returns dict with trades, equity_curve, drawdown_curve.
    """
    equity         = INITIAL_BALANCE
    balance        = INITIAL_BALANCE
    peak           = INITIAL_BALANCE
    trades         = []
    open_pos       = []
    equity_curve   = []
    drawdown_curve = []

    day_start_bal  = balance
    current_day    = None
    day_killed     = False

    rows = bars.reset_index()

    for i, row in rows.iterrows():
        bar_time = row["time"]
        bar_day  = bar_time.date()

        # ── Day rollover ───────────────────────────────────────────────────
        if bar_day != current_day:
            current_day   = bar_day
            day_start_bal = balance
            day_killed    = False

        hi, lo, cl = row["high"], row["low"], row["close"]

        # ── Manage open positions — direct 1:3 SL/TP, no partials ──────
        still_open = []
        for pos in open_pos:
            d = pos["dir"]   # 1=long, -1=short

            # TP priority: if both SL and TP hit on the same bar, credit TP
            tp_hit = (d ==  1 and hi >= pos["tp"]) or                      (d == -1 and lo <= pos["tp"])
            sl_hit = (d ==  1 and lo <= pos["sl"]) or                      (d == -1 and hi >= pos["sl"])

            if tp_hit:
                outcome  = "tp"
                close_px = pos["tp"]
            elif sl_hit:
                outcome  = "sl"
                close_px = pos["sl"]
            else:
                still_open.append(pos)
                continue

            pip_gain = (close_px - pos["entry"]) * d
            pnl      = pip_gain / PIP_SIZE * PIP_VALUE_PER_LOT * pos["lots"] \
                       - COMMISSION_USD * pos["lots"]   # full round-trip on close

            balance += pnl
            equity  += pnl

            trades.append({
                "entry_time":    rows.at[pos["entry_i"], "time"].strftime("%Y-%m-%d %H:%M"),
                "exit_time":     bar_time.strftime("%Y-%m-%d %H:%M"),
                "direction":     "BUY" if d == 1 else "SELL",
                "entry_price":   round(pos["entry"],     5),
                "sl_price":      round(pos["sl"],        5),
                "tp_price":      round(pos["tp"],        5),
                "lots":          round(pos["orig_lots"], 2),
                "risk_amount":   pos.get("risk_amount",  0),
                "pnl":           round(pnl,              2),
                "outcome":       outcome,
                "balance_after": round(balance,          2),
            })

        open_pos = still_open

        # ── Daily P&L check ────────────────────────────────────────────────
        daily_pnl_pct = (balance - day_start_bal) / day_start_bal if day_start_bal else 0

        if not day_killed and daily_pnl_pct <= -MAX_DAILY_LOSS:
            day_killed = True
            for pos in open_pos:
                d        = pos["dir"]
                pip_gain = (cl - pos["entry"]) * d
                pnl      = pip_gain / PIP_SIZE * PIP_VALUE_PER_LOT * pos["lots"] - COMMISSION_USD * pos["lots"] * 0.5
                balance += pnl
                equity  += pnl
                trades.append({
                    "entry_time":    rows.at[pos["entry_i"], "time"].strftime("%Y-%m-%d %H:%M"),
                    "exit_time":     bar_time.strftime("%Y-%m-%d %H:%M"),
                    "direction":     "BUY" if d == 1 else "SELL",
                    "entry_price":   round(pos["entry"], 5),
                    "sl_price":      round(pos["sl"],    5),
                    "tp_price":      round(pos["tp"],    5),
                    "lots":          round(pos["orig_lots"], 2),
                    "risk_amount":   pos.get("risk_amount", 0),
                    "pnl":           round(pnl, 2),
                    "outcome":       "kill_switch",
                    "balance_after": round(balance, 2),
                })
            open_pos = []

        # ── Entry ──────────────────────────────────────────────────────────
        sig = row.get("signal", 0)
        can_trade = (
            not day_killed and
            len(open_pos) < MAX_OPEN_TRADES and
            sig != 0
        )

        if can_trade:
            d = int(sig)
            if d == 1:
                entry = row["entry_long"]
                sl    = row["sl_long"]
                tp    = row["tp_long"]
            else:
                entry = row["entry_short"]
                sl    = row["sl_short"]
                tp    = row["tp_short"]

            sl_dist  = abs(entry - sl)
            sl_pips  = sl_dist / PIP_SIZE
            if sl_pips > 0:
                # ── Compounding lot size ───────────────────────────────────
                # Lot size is recalculated from CURRENT equity every trade.
                # As balance grows from profitable trades, risk_amount grows
                # too, so each new position is automatically larger.
                # On a losing streak the opposite applies — lots shrink,
                # protecting the account from ruin (anti-martingale).
                #
                # Formula: lots = (equity × risk%) / (sl_pips × $10/pip/lot)
                # Example: $10,000 × 1% = $100 risk
                #          $100 / (20 pips × $10) = 0.50 lots
                # After growing to $11,000:
                #          $110 / (20 pips × $10) = 0.55 lots  ← auto scales
                risk_amount = equity * RISK_PER_TRADE
                raw_lots    = risk_amount / (sl_pips * PIP_VALUE_PER_LOT)
                lots        = round(raw_lots / 0.01) * 0.01   # snap to broker lot step
                lots        = max(0.01, min(lots, 100.0))      # enforce broker min/max

                # Entry commission (half on open, half on close)
                balance -= COMMISSION_USD * lots * 0.5
                equity  -= COMMISSION_USD * lots * 0.5

                open_pos.append({
                    "entry_i":     i,
                    "dir":         d,
                    "lots":        lots,
                    "orig_lots":   lots,
                    "risk_amount": round(risk_amount, 2),
                    "entry":       entry,
                    "sl":          sl,
                    "tp":          tp,
                })

        # ── Equity curve ───────────────────────────────────────────────────
        floating = sum(
            (cl - p["entry"]) * p["dir"] / PIP_SIZE * PIP_VALUE_PER_LOT * p["lots"]
            for p in open_pos
        )
        curr_eq = balance + floating
        peak    = max(peak, curr_eq)
        dd      = (curr_eq - peak) / peak * 100

        equity_curve.append({
            "time":    bar_time.strftime("%Y-%m-%d %H:%M"),
            "equity":  round(curr_eq, 2),
            "balance": round(balance, 2),
        })
        drawdown_curve.append(round(dd, 3))

    return {
        "trades":          trades,
        "equity_curve":    equity_curve,
        "drawdown_curve":  drawdown_curve,
        "final_balance":   round(balance, 2),
        "initial_balance": INITIAL_BALANCE,
    }

# =============================================================================
# STATISTICS
# =============================================================================

def compute_stats(results: dict) -> dict:
    trades = results["trades"]
    if not trades:
        return {"total_trades": 0}

    pnls   = [t["pnl"] for t in trades]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    n      = len(pnls)

    gross_win  = sum(wins)          if wins   else 0
    gross_loss = abs(sum(losses))   if losses else 0
    pf         = gross_win / gross_loss if gross_loss > 0 else float("inf")

    net_pnl   = results["final_balance"] - INITIAL_BALANCE
    total_ret = net_pnl / INITIAL_BALANCE * 100

    eq_vals = [e["equity"] for e in results["equity_curve"]]
    peak    = INITIAL_BALANCE
    max_dd  = 0.0
    for eq in eq_vals:
        peak   = max(peak, eq)
        max_dd = min(max_dd, (eq - peak) / peak * 100)

    # Sharpe (per-trade)
    if len(pnls) > 1:
        arr    = np.array(pnls) / INITIAL_BALANCE
        sharpe = (arr.mean() / arr.std(ddof=1) * np.sqrt(252)) if arr.std(ddof=1) > 0 else 0
    else:
        sharpe = 0

    # Consecutive losses
    max_cl, cl = 0, 0
    for p in pnls:
        cl = cl + 1 if p <= 0 else 0
        max_cl = max(max_cl, cl)

    # Monthly breakdown
    monthly: dict = {}
    for t in trades:
        m = t["entry_time"][:7]
        monthly.setdefault(m, {"pnl": 0, "trades": 0, "wins": 0})
        monthly[m]["pnl"]    += t["pnl"]
        monthly[m]["trades"] += 1
        if t["pnl"] > 0:
            monthly[m]["wins"] += 1

    monthly_list = [
        {
            "month":    k,
            "pnl":      round(v["pnl"], 2),
            "trades":   v["trades"],
            "win_rate": round(v["wins"] / v["trades"] * 100, 1),
        }
        for k, v in sorted(monthly.items())
    ]

    return {
        "total_trades":      n,
        "win_rate":          round(len(wins) / n * 100, 2),
        "net_pnl":           round(net_pnl, 2),
        "total_return_pct":  round(total_ret, 2),
        "profit_factor":     round(pf, 2),
        "sharpe_ratio":      round(sharpe, 2),
        "max_drawdown_pct":  round(max_dd, 2),
        "avg_win":           round(np.mean(wins),   2) if wins   else 0,
        "avg_loss":          round(np.mean(losses), 2) if losses else 0,
        "largest_win":       round(max(wins),       2) if wins   else 0,
        "largest_loss":      round(min(losses),     2) if losses else 0,
        "max_consec_losses": max_cl,
        "total_tp":          sum(1 for t in trades if t["outcome"] == "tp"),
        "total_sl":          sum(1 for t in trades if t["outcome"] == "sl"),
        "total_kill":        sum(1 for t in trades if t["outcome"] == "kill_switch"),
        "avg_tp_pnl":        round(np.mean([t["pnl"] for t in trades if t["outcome"] == "tp"]), 2)
                             if any(t["outcome"] == "tp" for t in trades) else 0,
        "avg_sl_pnl":        round(np.mean([t["pnl"] for t in trades if t["outcome"] == "sl"]), 2)
                             if any(t["outcome"] == "sl" for t in trades) else 0,
        "monthly":           monthly_list,
        # Compounding stats
        "first_lot_size":    round(trades[0]["lots"],  2) if trades else 0,
        "last_lot_size":     round(trades[-1]["lots"], 2) if trades else 0,
        "min_lot_size":      round(min(t["lots"] for t in trades), 2) if trades else 0,
        "max_lot_size":      round(max(t["lots"] for t in trades), 2) if trades else 0,
        "lot_growth_pct":    round(
            (trades[-1]["lots"] / trades[0]["lots"] - 1) * 100, 1
        ) if trades and trades[0]["lots"] > 0 else 0,
    }

# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="EURUSD Strategy Backtester")
    parser.add_argument("--days",   type=int, default=180)
    parser.add_argument("--from",   dest="date_from", default=None)
    parser.add_argument("--to",     dest="date_to",   default=None)
    parser.add_argument("--no-mt5", action="store_true")
    parser.add_argument("--output", default=OUTPUT_FILE)
    args = parser.parse_args()

    if args.date_from and args.date_to:
        start = datetime.strptime(args.date_from, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end   = datetime.strptime(args.date_to,   "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        end   = datetime.now(timezone.utc)
        start = end - timedelta(days=args.days)

    logger.info(f"Backtest: {start.date()} → {end.date()}")

    if args.no_mt5:
        h1, h4, m15 = make_synthetic(days=args.days)
    else:
        h1, h4, m15 = fetch_mt5(start, end, days=args.days)

    logger.info("Generating signals...")
    signals = generate_signals(h1, h4, m15)

    logger.info("Simulating trades...")
    results = simulate(signals)
    stats   = compute_stats(results)

    # ── Print summary ──────────────────────────────────────────────────────
    print("\n" + "═" * 50)
    print("  BACKTEST RESULTS")
    print("═" * 50)
    print(f"  Period          : {start.date()} → {end.date()}")
    print(f"  Total trades    : {stats.get('total_trades', 0)}")
    print(f"  Win rate        : {stats.get('win_rate', 0):.1f}%")
    print(f"  Net P&L         : ${stats.get('net_pnl', 0):+,.2f}")
    print(f"  Total return    : {stats.get('total_return_pct', 0):+.2f}%")
    print(f"  Profit factor   : {stats.get('profit_factor', 0):.2f}")
    print(f"  Sharpe ratio    : {stats.get('sharpe_ratio', 0):.2f}")
    print(f"  Max drawdown    : {stats.get('max_drawdown_pct', 0):.2f}%")
    print(f"  Avg win         : ${stats.get('avg_win', 0):+.2f}")
    print(f"  Avg loss        : ${stats.get('avg_loss', 0):+.2f}")
    print(f"  Max consec loss : {stats.get('max_consec_losses', 0)}")
    print(f"  TP hits         : {stats.get('total_tp', 0)}  avg ${stats.get('avg_tp_pnl', 0):+.2f}")
    print(f"  SL hits         : {stats.get('total_sl', 0)}  avg ${stats.get('avg_sl_pnl', 0):+.2f}")
    print("─" * 50)
    print(f"  Lot size (first): {stats.get('first_lot_size', 0):.2f}")
    print(f"  Lot size (last) : {stats.get('last_lot_size',  0):.2f}")
    print(f"  Lot size (min)  : {stats.get('min_lot_size',   0):.2f}")
    print(f"  Lot size (max)  : {stats.get('max_lot_size',   0):.2f}")
    print(f"  Lot growth      : {stats.get('lot_growth_pct', 0):+.1f}%  (compounding effect)")
    print("═" * 50)

    # ── Save JSON for dashboard ────────────────────────────────────────────
    payload = {"stats": stats, **results}
    Path(args.output).write_text(json.dumps(payload, indent=2))
    logger.info(f"Saved → {args.output}  (open dashboard.html to visualise)")


if __name__ == "__main__":
    main()