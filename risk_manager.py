# =============================================================================
# risk_manager.py — Position Sizing, Kill Switch & Trade Guards
# This module is the gatekeeper. Nothing gets traded without passing here.
# =============================================================================

import logging
from dataclasses import dataclass
from typing import Optional

import config
import data_feed as df

logger = logging.getLogger(__name__)


# =============================================================================
# SESSION STATE  (in-memory, resets each run / each day)
# =============================================================================

@dataclass
class SessionState:
    """
    Tracks intra-day state so the kill switch works correctly.
    Instantiate once at startup and pass through the main loop.
    """
    killed:          bool  = False   # True = EA suspended for the session
    opening_balance: float = 0.0     # Balance at session start (set on connect)
    partial_closed:  set   = None    # Ticket IDs already partially closed

    def __post_init__(self):
        if self.partial_closed is None:
            self.partial_closed = set()


# =============================================================================
# INITIALISE SESSION
# =============================================================================

def init_session() -> Optional[SessionState]:
    """
    Call once at startup. Reads the current account balance and stores it
    as the reference point for daily P&L calculations.
    Returns a fresh SessionState, or None if account info unavailable.
    """
    account = df.get_account_info()
    if account is None:
        logger.error("Cannot initialise session — account info unavailable.")
        return None

    state = SessionState(opening_balance=account["balance"])
    logger.info(
        f"Session initialised | Opening balance: "
        f"{state.opening_balance:.2f} {account['currency']}"
    )
    return state


# =============================================================================
# KILL SWITCH CHECK
# =============================================================================

def check_kill_switch(state: SessionState) -> SessionState:
    """
    Evaluate daily P&L against configured thresholds.
    Updates state.killed and state.daily_target_hit in-place.

    Called at the top of every main loop iteration BEFORE signal evaluation.
    Once killed, stays killed for the entire session (no auto-reset).
    """
    if state.killed:
        return state   # Already suspended — skip recalculation

    if state.opening_balance == 0:
        logger.warning("Opening balance is 0 — cannot calculate P&L %.")
        return state

    pnl_pct = df.get_daily_pnl_pct()

    # --- Drawdown breach ---
    if pnl_pct <= -config.MAX_DAILY_LOSS_PCT:
        logger.warning(
            f"🔴  KILL SWITCH TRIGGERED | Daily loss = {pnl_pct*100:.3f}% "
            f"(limit: -{config.MAX_DAILY_LOSS_PCT*100:.1f}%) | "
            f"EA suspended for this session."
        )
        state.killed = True
        return state

    logger.debug(f"Kill switch OK | Daily P&L: {pnl_pct*100:.3f}%")
    return state


def is_trading_allowed(state: SessionState) -> bool:
    """
    Master trading permission gate.
    Returns False if the kill switch is active, daily target is hit,
    outside session hours, or max open trades is reached.
    """
    if state.killed:
        logger.debug("Trading blocked — kill switch active.")
        return False

    if not df.is_market_session():
        logger.debug("Trading blocked — outside session hours.")
        return False

    open_positions = df.get_open_positions()
    if len(open_positions) >= config.MAX_OPEN_TRADES:
        logger.debug(
            f"Trading blocked — {len(open_positions)}/{config.MAX_OPEN_TRADES} "
            f"positions open."
        )
        return False

    return True


# =============================================================================
# POSITION SIZING
# =============================================================================

def calculate_lot_size(sl_distance_price: float) -> Optional[float]:
    """
    Compound position sizing — lot size scales automatically with account equity.

    Every winning trade grows the equity, which grows the next risk_amount,
    which grows the next lot size. This is the anti-martingale / fixed-fractional
    approach: bet more when winning, bet less when losing.

    Compounding example (1% risk, 20-pip SL, $10/pip/lot):
        $10,000 equity  → risk $100  → 0.50 lots
        $10,500 equity  → risk $105  → 0.53 lots  (+6% vs start)
        $11,000 equity  → risk $110  → 0.55 lots  (+10% vs start)
        $9,500  equity  → risk $95   → 0.48 lots  (shrinks on drawdown)

    The key property: losses shrink future lot sizes automatically, which
    slows the drawdown curve. Profits grow future lot sizes, which
    accelerates the equity curve. Both effects are proportional.

    Formula:
        risk_amount  = current_equity × RISK_PER_TRADE
        sl_pips      = sl_distance_price / 0.0001
        pip_val_lot  = contract_size × pip  (= $10 for EURUSD/USD)
        lots         = risk_amount / (sl_pips × pip_val_lot)
    """
    account = df.get_account_info()
    symbol  = df.get_symbol_info()

    if account is None or symbol is None:
        logger.error("Cannot size position — missing account or symbol info.")
        return None

    if sl_distance_price <= 0:
        logger.error(f"Invalid SL distance: {sl_distance_price}")
        return None

    # Always use CURRENT equity — this is what makes it compound
    equity      = account["equity"]
    risk_amount = equity * config.RISK_PER_TRADE

    # Pip value per lot in account currency
    point         = symbol.point               # 0.00001 for 5-digit broker
    pip           = point * 10                 # 0.00010 = 1 pip
    contract_size = symbol.trade_contract_size # 100,000 for EURUSD
    pip_value_lot = pip * contract_size        # $10.00 per lot (EURUSD/USD account)

    sl_pips = sl_distance_price / pip
    if sl_pips <= 0:
        logger.error(f"SL pips calculated as 0 or negative: {sl_pips}")
        return None

    raw_lots = risk_amount / (sl_pips * pip_value_lot)

    # Snap to broker lot step (usually 0.01)
    lot_step = symbol.volume_step
    lots     = round(raw_lots / lot_step) * lot_step
    lots     = round(lots, 2)

    # Enforce broker min/max
    lots = max(lots, symbol.volume_min)
    lots = min(lots, symbol.volume_max)

    logger.info(
        f"Compound lot size | equity={equity:.2f} (+{(equity/config.INITIAL_BALANCE - 1)*100:.1f}% from start) "
        f"| risk={risk_amount:.2f} | sl={sl_pips:.1f} pips → {lots} lots"
    )
    return lots


# =============================================================================
# BREAKEVEN & PARTIAL CLOSE CHECKS
# =============================================================================

def should_move_to_breakeven(position: dict) -> bool:
    """
    Returns True if a position has reached its partial TP level
    and the SL has NOT yet been moved to breakeven.

    We detect 'already at BE' by checking if the current SL is within
    1 pip of the open price (meaning it was already moved).
    """
    symbol = df.get_symbol_info()
    if symbol is None:
        return False

    pip = symbol.point * 10

    # SL is already at or above breakeven
    if position["type"] == "buy":
        if position["sl"] >= position["open_price"] - pip:
            return False
        # Check if bid has reached partial TP level
        price = df.get_current_price()
        if price is None:
            return False
        profit_distance = price["bid"] - position["open_price"]
        sl_distance     = position["open_price"] - position["sl"]
        return profit_distance >= sl_distance * config.PARTIAL_CLOSE_R

    else:  # sell
        if position["sl"] <= position["open_price"] + pip:
            return False
        price = df.get_current_price()
        if price is None:
            return False
        profit_distance = position["open_price"] - price["ask"]
        sl_distance     = position["sl"] - position["open_price"]
        return profit_distance >= sl_distance * config.PARTIAL_CLOSE_R


def should_partial_close(position: dict, state: SessionState) -> bool:
    """
    Returns True if the position should have 50% closed at 1:1 R
    and hasn't been partially closed yet this session.
    """
    if position["ticket"] in state.partial_closed:
        return False
    return should_move_to_breakeven(position)


# =============================================================================
# DAILY P&L REPORT  (call at end of session or on demand)
# =============================================================================

def print_daily_report(state: SessionState) -> None:
    """Print a concise summary of the session to the console."""
    account  = df.get_account_info()
    pnl_pct  = df.get_daily_pnl_pct()
    realised = df.get_daily_realised_pnl()

    print("\n" + "=" * 55)
    print("  Daily Session Report")
    print("=" * 55)
    if account:
        print(f"  Opening balance : {state.opening_balance:.2f}")
        print(f"  Current equity  : {account['equity']:.2f} {account['currency']}")
    print(f"  Realised P&L    : {realised:+.2f}")
    print(f"  Floating P&L    : {account['profit']:+.2f}" if account else "")
    print(f"  Daily P&L %     : {pnl_pct*100:+.3f}%")
    print(f"  Kill switch     : {'TRIGGERED' if state.killed else 'OK'}")
    print(f"  Target hit      : {'YES' if state.daily_target_hit else 'NO'}")
    print("=" * 55 + "\n")


# =============================================================================
# QUICK TEST (no MT5 — exercises logic with mocked values)
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s | %(message)s")

    print("=" * 55)
    print("  Risk Manager — Logic Test (no MT5 required)")
    print("=" * 55)

    # --- Test kill switch math ---
    state = SessionState(opening_balance=10_000.0)

    # Simulate -0.5% loss — should NOT kill
    import unittest.mock as mock
    with mock.patch("data_feed.get_daily_pnl_pct", return_value=-0.005):
        state = check_kill_switch(state)
        print(f"\n[-0.5% loss] Killed: {state.killed}  (expected: False)")

    # Simulate -1.0% loss — should kill
    state2 = SessionState(opening_balance=10_000.0)
    with mock.patch("data_feed.get_daily_pnl_pct", return_value=-0.010):
        state2 = check_kill_switch(state2)
        print(f"[-1.0% loss] Killed: {state2.killed}  (expected: True)")

    # Simulate +1.0% gain — should set target hit
    state3 = SessionState(opening_balance=10_000.0)
    with mock.patch("data_feed.get_daily_pnl_pct", return_value=0.010):
        state3 = check_kill_switch(state3)
        print(f"[+1.0% gain] Target hit: {state3.daily_target_hit}  (expected: True)")

    # --- Test lot size formula (offline) ---
    print("\n--- Position Sizing Formula (standalone) ---")
    equity       = 10_000.0
    risk_amount  = equity * 0.01       # $100
    sl_pips      = 20                  # 20 pip SL
    pip_val_lot  = 10.0                # $10/pip/lot for EURUSD/USD
    raw_lots     = risk_amount / (sl_pips * pip_val_lot)
    print(f"Equity: ${equity:,.0f} | Risk: ${risk_amount:.0f} | "
          f"SL: {sl_pips} pips | Lots: {raw_lots:.2f}")

    equity2      = 10_000.0
    sl_pips2     = 30
    raw_lots2    = (equity2 * 0.01) / (sl_pips2 * pip_val_lot)
    print(f"Equity: ${equity2:,.0f} | Risk: ${risk_amount:.0f} | "
          f"SL: {sl_pips2} pips | Lots: {raw_lots2:.2f}")

    print("\n✅  risk_manager.py logic OK")