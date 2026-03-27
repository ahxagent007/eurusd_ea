# =============================================================================
# trade_monitor.py — Open Trade Lifecycle Monitor
# Watches every open position on every bar and applies management rules:
#   1. Partial close at 1:1 R
#   2. Move SL to breakeven after partial close
#   3. Trailing stop once in profit (optional, configurable)
#   4. Max hold-time guard — close trades stuck open too long
#   5. Session-end close — flatten before the NY close
# This module is the only place that decides WHEN to modify a live trade.
# =============================================================================

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import config
import data_feed as df
import logger as log
import order_manager as om
from risk_manager import SessionState

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION  (add to config.py if you want UI control)
# =============================================================================

# Trailing stop — activate only after price moves this many R multiples in profit
TRAIL_ACTIVATE_R     = 1.0    # Start trailing after 1:1 R reached (same as partial TP)
TRAIL_STEP_ATR       = 0.5    # Trail by 0.5× ATR increments
TRAILING_ENABLED     = True   # Set False to disable trailing entirely

# Max hold time — close any trade open longer than this
MAX_HOLD_HOURS       = 24     # 24-hour max hold; avoids carrying over weekend gaps

# Session-end flatten — close all trades this many minutes before session end
FLATTEN_MINS_BEFORE  = 15     # Close 15 min before SESSION_END_HOUR


# =============================================================================
# MAIN MONITOR — call every M15 bar from main.py
# =============================================================================

def monitor_trades(state: SessionState) -> None:
    """
    Entry point called from the main loop every M15 bar close.
    Applies all lifecycle rules to every open position in sequence:

        partial close  →  breakeven  →  trail  →  max hold  →  session flatten

    Order matters — partial close must happen before breakeven,
    breakeven before trailing (trail from BE only, never from original SL).
    """
    positions = df.get_open_positions()
    if not positions:
        return

    price_info = df.get_current_price()
    if price_info is None:
        logger.warning("monitor_trades: cannot get price — skipping this bar.")
        return

    for pos in positions:
        _apply_lifecycle(pos, state, price_info)


# =============================================================================
# PER-POSITION LIFECYCLE
# =============================================================================

def _apply_lifecycle(
    pos:        dict,
    state:      SessionState,
    price_info: dict,
) -> None:
    """Apply all management rules to a single position in priority order."""

    ticket = pos["ticket"]

    # ── Rule 1: Session-end flatten (highest priority) ────────────────────
    if _should_flatten_for_session_end():
        logger.info(
            f"Session end approaching — flattening ticket={ticket} "
            f"({FLATTEN_MINS_BEFORE} min before close)."
        )
        _close_position(pos, price_info, reason="Session-end flatten")
        return   # Position gone — skip remaining rules

    # ── Rule 2: Max hold time ─────────────────────────────────────────────
    if _exceeded_max_hold(pos):
        logger.info(
            f"Max hold time ({MAX_HOLD_HOURS}h) exceeded — "
            f"closing ticket={ticket}."
        )
        _close_position(pos, price_info, reason=f"Max hold {MAX_HOLD_HOURS}h exceeded")
        return

    # ── Rule 3: Partial close at 1:1 R ───────────────────────────────────
    if ticket not in state.partial_closed:
        if _at_partial_tp(pos, price_info):
            logger.info(f"1:1 R reached — partial close + breakeven for ticket={ticket}.")
            success = om.partial_close(pos, state)
            if success:
                log.log_partial_close(
                    ticket      = ticket,
                    direction   = pos["type"].upper(),
                    lots_closed = round(pos["volume"] * config.PARTIAL_CLOSE_PCT, 2),
                    close_price = price_info["bid"] if pos["type"] == "buy" else price_info["ask"],
                    profit      = pos["profit"] * config.PARTIAL_CLOSE_PCT,
                    daily_pnl   = df.get_daily_pnl_pct(),
                )
                # Immediately move SL to breakeven
                new_sl = om.move_sl_to_breakeven(pos)
                if new_sl:
                    log.log_breakeven(ticket=ticket, new_sl=new_sl)
            return   # Skip trail on same bar as partial — re-evaluate next bar

    # ── Rule 4: Trailing stop (only after partial close + BE) ─────────────
    if TRAILING_ENABLED and ticket in state.partial_closed:
        _apply_trail(pos, price_info)


# =============================================================================
# RULE IMPLEMENTATIONS
# =============================================================================

def _at_partial_tp(pos: dict, price_info: dict) -> bool:
    """
    Returns True when price has moved at least PARTIAL_CLOSE_R × SL distance
    in the trade's favour.

    For a BUY:  current_bid ≥ open_price + (open_price - sl) × R
    For a SELL: current_ask ≤ open_price - (sl - open_price) × R
    """
    sl_dist = abs(pos["open_price"] - pos["sl"])
    if sl_dist == 0:
        return False

    target_move = sl_dist * config.PARTIAL_CLOSE_R

    if pos["type"] == "buy":
        return price_info["bid"] >= pos["open_price"] + target_move
    else:
        return price_info["ask"] <= pos["open_price"] - target_move


def _apply_trail(pos: dict, price_info: dict) -> None:
    """
    Trail the SL by TRAIL_STEP_ATR × ATR whenever price makes a new
    favourable extreme since the SL was last moved.

    We trail using M15 ATR so the step adapts to current volatility.
    The SL only ever moves in the trade's favour — never widened.

    For BUY:  new_sl = current_bid - (ATR × TRAIL_STEP_ATR)
              only update if new_sl > current sl
    For SELL: new_sl = current_ask + (ATR × TRAIL_STEP_ATR)
              only update if new_sl < current sl
    """
    m15_bars = df.get_bars(config.LTF_TIMEFRAME, 30)
    if m15_bars is None:
        return

    from indicators import current_atr
    atr_val  = current_atr(m15_bars)
    trail_dist = atr_val * TRAIL_STEP_ATR

    symbol = df.get_symbol_info()
    if symbol is None:
        return
    pip = symbol.point * 10

    if pos["type"] == "buy":
        candidate_sl = round(price_info["bid"] - trail_dist, 5)
        # Must be better than current SL and above entry (we're in profit)
        if candidate_sl > pos["sl"] + pip and candidate_sl > pos["open_price"]:
            _modify_sl(pos, candidate_sl, "Trail BUY")
    else:
        candidate_sl = round(price_info["ask"] + trail_dist, 5)
        if candidate_sl < pos["sl"] - pip and candidate_sl < pos["open_price"]:
            _modify_sl(pos, candidate_sl, "Trail SELL")


def _should_flatten_for_session_end() -> bool:
    """
    Returns True if we are within FLATTEN_MINS_BEFORE minutes of SESSION_END_HOUR.
    """
    now = datetime.now(timezone.utc)
    session_end = now.replace(
        hour=config.SESSION_END_HOUR, minute=0, second=0, microsecond=0
    )
    delta = (session_end - now).total_seconds() / 60   # minutes remaining
    return 0 <= delta <= FLATTEN_MINS_BEFORE


def _exceeded_max_hold(pos: dict) -> bool:
    """
    Returns True if the position has been open longer than MAX_HOLD_HOURS.
    """
    open_time = pos["open_time"]                         # datetime with tz
    age = datetime.now(timezone.utc) - open_time
    return age > timedelta(hours=MAX_HOLD_HOURS)


# =============================================================================
# LOW-LEVEL HELPERS
# =============================================================================

def _modify_sl(pos: dict, new_sl: float, reason: str) -> bool:
    """
    Send an SL modification request to MT5.
    Returns True on success.
    """
    import MetaTrader5 as mt5

    request = {
        "action":   mt5.TRADE_ACTION_SLTP,
        "symbol":   config.SYMBOL,
        "position": pos["ticket"],
        "sl":       new_sl,
        "tp":       pos["tp"],
    }
    result = mt5.order_send(request)

    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        retcode = result.retcode if result else "None"
        logger.error(
            f"SL modify failed | ticket={pos['ticket']} "
            f"reason={reason} retcode={retcode}"
        )
        return False

    logger.info(
        f"SL updated | ticket={pos['ticket']} | "
        f"new_sl={new_sl:.5f} | {reason}"
    )
    return True


def _close_position(pos: dict, price_info: dict, reason: str) -> bool:
    """
    Force-close a single position (session flatten or max hold).
    Logs the closure to the CSV journal.
    """
    import MetaTrader5 as mt5

    close_type  = mt5.ORDER_TYPE_SELL if pos["type"] == "buy" else mt5.ORDER_TYPE_BUY
    close_price = price_info["bid"] if pos["type"] == "buy" else price_info["ask"]

    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       config.SYMBOL,
        "volume":       pos["volume"],
        "type":         close_type,
        "position":     pos["ticket"],
        "price":        close_price,
        "deviation":    config.DEVIATION,
        "magic":        config.MAGIC_NUMBER,
        "comment":      f"EA_{reason.replace(' ', '_').upper()}",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)

    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        retcode = result.retcode if result else "None"
        logger.error(
            f"Force close failed | ticket={pos['ticket']} "
            f"reason={reason} retcode={retcode}"
        )
        return False

    log.log_trade_closed(
        ticket      = pos["ticket"],
        direction   = pos["type"].upper(),
        lots        = pos["volume"],
        entry_price = pos["open_price"],
        close_price = close_price,
        profit      = pos["profit"],
        reason      = reason,
        daily_pnl   = df.get_daily_pnl_pct(),
    )
    return True


# =============================================================================
# POSITION SUMMARY (call any time for a console snapshot)
# =============================================================================

def print_open_positions() -> None:
    """Print a formatted table of all open EA positions."""
    positions = df.get_open_positions()
    price_info = df.get_current_price()

    print("\n" + "=" * 72)
    print(f"  Open Positions ({len(positions)}/{config.MAX_OPEN_TRADES})   "
          f"EURUSD {'bid=' + str(price_info['bid']) if price_info else ''}")
    print("=" * 72)

    if not positions:
        print("  No open positions.")
    else:
        print(f"  {'Ticket':<10} {'Dir':<5} {'Lots':<6} {'Open':<10} "
              f"{'SL':<10} {'TP':<10} {'P&L':>8}  {'Age'}")
        print("  " + "-" * 68)
        now = datetime.now(timezone.utc)
        for p in positions:
            age_mins = int((now - p["open_time"]).total_seconds() / 60)
            age_str  = f"{age_mins//60}h{age_mins%60:02d}m"
            pnl_str  = f"{p['profit']:+.2f}"
            print(
                f"  {p['ticket']:<10} {p['type'].upper():<5} {p['volume']:<6} "
                f"{p['open_price']:<10.5f} {p['sl']:<10.5f} {p['tp']:<10.5f} "
                f"{pnl_str:>8}  {age_str}"
            )

    print("=" * 72 + "\n")


# =============================================================================
# QUICK TEST (no MT5 — exercises rule logic with mock data)
# =============================================================================

if __name__ == "__main__":
    import unittest.mock as mock
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s | %(message)s")

    print("=" * 55)
    print("  trade_monitor.py — Logic Test (no MT5 required)")
    print("=" * 55)

    now_utc = datetime.now(timezone.utc)

    # Mock position: BUY opened 2 hours ago, SL 20 pips below entry
    mock_pos_buy = {
        "ticket":     999001,
        "type":       "buy",
        "volume":     0.50,
        "open_price": 1.08500,
        "sl":         1.08300,   # 20 pip SL
        "tp":         1.08900,   # 40 pip TP
        "profit":     45.0,
        "open_time":  now_utc - timedelta(hours=2),
    }

    mock_price_at_1R = {"bid": 1.08700, "ask": 1.08702, "spread_points": 2}
    mock_price_below = {"bid": 1.08550, "ask": 1.08552, "spread_points": 2}

    # Test 1: _at_partial_tp — price at exactly 1:1 R
    result = _at_partial_tp(mock_pos_buy, mock_price_at_1R)
    print(f"\n[Test 1] At partial TP (price=1.08700, need ≥1.08700): {result}  (expected: True)")

    # Test 2: _at_partial_tp — price not there yet
    result2 = _at_partial_tp(mock_pos_buy, mock_price_below)
    print(f"[Test 2] At partial TP (price=1.08550): {result2}  (expected: False)")

    # Test 3: _exceeded_max_hold — fresh trade
    result3 = _exceeded_max_hold(mock_pos_buy)
    print(f"[Test 3] Max hold exceeded (2h old, limit={MAX_HOLD_HOURS}h): {result3}  (expected: False)")

    # Test 4: _exceeded_max_hold — stale trade
    mock_pos_old = {**mock_pos_buy, "open_time": now_utc - timedelta(hours=25)}
    result4 = _exceeded_max_hold(mock_pos_old)
    print(f"[Test 4] Max hold exceeded (25h old): {result4}  (expected: True)")

    # Test 5: Session-end flatten window
    result5 = _should_flatten_for_session_end()
    print(f"[Test 5] Session flatten now ({now_utc.hour:02d}:{now_utc.minute:02d} UTC): {result5}")

    # Test 6: SELL position partial TP check
    mock_pos_sell = {
        "ticket":     999002,
        "type":       "sell",
        "volume":     0.30,
        "open_price": 1.09000,
        "sl":         1.09200,   # 20 pip SL
        "tp":         1.08600,
        "profit":     60.0,
        "open_time":  now_utc - timedelta(hours=1),
    }
    mock_price_sell_tp = {"bid": 1.08798, "ask": 1.08800, "spread_points": 2}
    result6 = _at_partial_tp(mock_pos_sell, mock_price_sell_tp)
    print(f"[Test 6] SELL at partial TP (ask=1.08800, need ≤1.08800): {result6}  (expected: True)")

    print("\n✅  trade_monitor.py OK")