# =============================================================================
# signal_engine.py — Trade Signal Generation
# Combines all indicator conditions into a single BUY / SELL / NONE decision.
# This is the only module that decides whether a trade should be opened.
# =============================================================================

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import pandas as pd

import config
import indicators as ind

logger = logging.getLogger(__name__)


# =============================================================================
# SIGNAL TYPES
# =============================================================================

class SignalType(Enum):
    BUY  = "BUY"
    SELL = "SELL"
    NONE = "NONE"


@dataclass
class Signal:
    """
    A trade signal returned by the engine.
    Contains the direction plus all the calculated levels the
    order_manager needs to place the trade.
    """
    type:          SignalType
    entry_price:   float = 0.0       # Ask for BUY, Bid for SELL
    sl_price:      float = 0.0       # Stop-loss level
    tp_price:      float = 0.0       # Full take-profit level
    partial_tp:    float = 0.0       # Partial TP at 1:1 R
    sl_distance:   float = 0.0       # SL distance in price (used for position sizing)
    atr_value:     float = 0.0       # ATR at signal time (for reference/logging)
    reasons:       list  = field(default_factory=list)   # Why each condition passed/failed

    def __str__(self):
        if self.type == SignalType.NONE:
            return f"Signal(NONE | {' | '.join(self.reasons)})"
        return (
            f"Signal({self.type.value} | "
            f"entry={self.entry_price:.5f} "
            f"sl={self.sl_price:.5f} "
            f"tp={self.tp_price:.5f} "
            f"sl_dist={self.sl_distance:.5f})"
        )


# =============================================================================
# SL / TP CALCULATION
# =============================================================================

def _calculate_levels(
    direction: SignalType,
    entry_price: float,
    atr_value: float,
    sl_buffer_atr: float = 1.5,
) -> tuple[float, float, float, float]:
    """
    Calculate SL, full TP, partial TP, and SL distance.

    SL is placed at 1.5× ATR from entry — beyond recent noise.
    TP is placed at sl_distance × REWARD_RATIO from entry.
    Partial TP is at 1× sl_distance (1:1 R) for the breakeven trigger.

    Returns: (sl_price, tp_price, partial_tp_price, sl_distance)
    """
    sl_distance = round(atr_value * sl_buffer_atr, 5)

    if direction == SignalType.BUY:
        sl_price    = round(entry_price - sl_distance, 5)
        tp_price    = round(entry_price + sl_distance * config.REWARD_RATIO, 5)
        partial_tp  = round(entry_price + sl_distance * config.PARTIAL_CLOSE_R, 5)
    else:  # SELL
        sl_price    = round(entry_price + sl_distance, 5)
        tp_price    = round(entry_price - sl_distance * config.REWARD_RATIO, 5)
        partial_tp  = round(entry_price - sl_distance * config.PARTIAL_CLOSE_R, 5)

    return sl_price, tp_price, partial_tp, sl_distance


# =============================================================================
# MAIN SIGNAL FUNCTION
# =============================================================================

def evaluate(
    h1_df: pd.DataFrame,
    m15_df: pd.DataFrame,
    current_ask: float,
    current_bid: float,
) -> Signal:
    """
    Run every confluence condition and return a Signal.

    Conditions for a BUY:
      1. H1 bullish trend  — close > EMA50 > EMA200
      2. M15 EMA9 crossed above EMA50 on the last closed bar
      3. M15 RSI(14) between RSI_LONG_MIN and RSI_LONG_MAX

    Conditions for a SELL:
      1. H1 bearish trend  — close < EMA50 < EMA200
      2. M15 EMA9 crossed below EMA50 on the last closed bar
      3. M15 RSI(14) between RSI_SHORT_MIN and RSI_SHORT_MAX

    All three conditions must pass for a valid signal.
    Returns Signal(NONE) with reasons logged when any condition fails.
    """
    reasons = []

    # --- Prepare indicators ---
    h1, m15 = ind.prepare_data(h1_df, m15_df)
    atr_val  = ind.current_atr(m15)

    # === CHECK LONG CONDITIONS ===
    bull_trend  = ind.is_bullish_trend(h1)
    bull_cross  = ind.ema9_crossed_above_ema50(m15)
    rsi_long_ok = ind.rsi_in_long_zone(m15)

    if bull_trend and bull_cross and rsi_long_ok:
        sl, tp, partial, sl_dist = _calculate_levels(SignalType.BUY, current_ask, atr_val)
        sig = Signal(
            type         = SignalType.BUY,
            entry_price  = current_ask,
            sl_price     = sl,
            tp_price     = tp,
            partial_tp   = partial,
            sl_distance  = sl_dist,
            atr_value    = atr_val,
            reasons      = ["H1 bullish", "EMA9 bull cross", f"RSI={m15['rsi_14'].iloc[-1]:.1f}"],
        )
        logger.info(f"✅  BUY signal | {sig}")
        return sig

    # === CHECK SHORT CONDITIONS ===
    bear_trend   = ind.is_bearish_trend(h1)
    bear_cross   = ind.ema9_crossed_below_ema50(m15)
    rsi_short_ok = ind.rsi_in_short_zone(m15)

    if bear_trend and bear_cross and rsi_short_ok:
        sl, tp, partial, sl_dist = _calculate_levels(SignalType.SELL, current_bid, atr_val)
        sig = Signal(
            type         = SignalType.SELL,
            entry_price  = current_bid,
            sl_price     = sl,
            tp_price     = tp,
            partial_tp   = partial,
            sl_distance  = sl_dist,
            atr_value    = atr_val,
            reasons      = ["H1 bearish", "EMA9 bear cross", f"RSI={m15['rsi_14'].iloc[-1]:.1f}"],
        )
        logger.info(f"✅  SELL signal | {sig}")
        return sig

    # === NO SIGNAL — build diagnostic reasons ===
    if not bull_trend and not bear_trend:
        reasons.append("No clear H1 trend (choppy market)")
    elif bull_trend:
        if not bull_cross:
            reasons.append("Bullish trend but no EMA9 cross yet")
        elif not rsi_long_ok:
            rsi_val = m15["rsi_14"].iloc[-1]
            reasons.append(f"Bullish trend + cross but RSI={rsi_val:.1f} outside zone")
    elif bear_trend:
        if not bear_cross:
            reasons.append("Bearish trend but no EMA9 cross yet")
        elif not rsi_short_ok:
            rsi_val = m15["rsi_14"].iloc[-1]
            reasons.append(f"Bearish trend + cross but RSI={rsi_val:.1f} outside zone")

    logger.debug(f"No signal | {' | '.join(reasons)}")
    return Signal(type=SignalType.NONE, reasons=reasons)


# =============================================================================
# QUICK TEST (run directly with synthetic data)
# =============================================================================

if __name__ == "__main__":
    import numpy as np
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s | %(message)s")

    def _make_df(prices, freq):
        idx = pd.date_range("2024-01-01", periods=len(prices), freq=freq, tz="UTC")
        return pd.DataFrame({
            "open":  prices - 0.0002,
            "high":  prices + 0.0005,
            "low":   prices - 0.0005,
            "close": prices,
            "tick_volume": np.random.randint(100, 1000, len(prices)),
        }, index=idx)

    np.random.seed(7)
    n = 300

    # --- Scenario 1: Bullish — uptrend + fresh cross ---
    print("\n" + "=" * 55)
    print("  Scenario 1: Bullish market")
    print("=" * 55)
    # Build an uptrend: EMA9 will be > EMA50 > EMA200
    bull_prices_h1  = 1.07 + np.linspace(0, 0.03, n) + np.random.randn(n) * 0.0002
    # Force a cross on the last bar: dip below EMA50 then pop back
    bull_prices_m15 = 1.07 + np.linspace(0, 0.02, n) + np.random.randn(n) * 0.0001
    bull_prices_m15[-2] = bull_prices_m15[-2] - 0.006   # previous bar dips below
    bull_prices_m15[-1] = bull_prices_m15[-1] + 0.001   # last bar pops above

    h1  = _make_df(bull_prices_h1,  "1h")
    m15 = _make_df(bull_prices_m15, "15min")
    sig = evaluate(h1, m15, current_ask=1.095, current_bid=1.0948)
    print(f"\nResult: {sig}")

    # --- Scenario 2: Bearish ---
    print("\n" + "=" * 55)
    print("  Scenario 2: Bearish market")
    print("=" * 55)
    bear_prices_h1  = 1.10 - np.linspace(0, 0.03, n) + np.random.randn(n) * 0.0002
    bear_prices_m15 = 1.10 - np.linspace(0, 0.02, n) + np.random.randn(n) * 0.0001
    bear_prices_m15[-2] = bear_prices_m15[-2] + 0.006
    bear_prices_m15[-1] = bear_prices_m15[-1] - 0.001

    h1  = _make_df(bear_prices_h1,  "1h")
    m15 = _make_df(bear_prices_m15, "15min")
    sig = evaluate(h1, m15, current_ask=1.075, current_bid=1.0748)
    print(f"\nResult: {sig}")

    # --- Scenario 3: Choppy — no signal ---
    print("\n" + "=" * 55)
    print("  Scenario 3: Choppy (no signal expected)")
    print("=" * 55)
    chop_prices_h1  = 1.085 + np.random.randn(n) * 0.001
    chop_prices_m15 = 1.085 + np.random.randn(n) * 0.0005

    h1  = _make_df(chop_prices_h1,  "1h")
    m15 = _make_df(chop_prices_m15, "15min")
    sig = evaluate(h1, m15, current_ask=1.085, current_bid=1.0848)
    print(f"\nResult: {sig}")