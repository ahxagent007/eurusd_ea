# =============================================================================
# main.py — EURUSD Expert Advisor — Main Loop
# Entry point. Ties every module together into a live trading loop.
#
# Usage:
#   python main.py            — run live on demo
#   python main.py --dry-run  — signal detection only, no orders sent
# =============================================================================

import argparse
import logging
import sys
import time
from datetime import datetime, timezone

import config
import data_feed as df
import logger as log
import order_manager as om
import risk_manager as rm
import trade_monitor as tm
from signal_engine import SignalType, evaluate

# Module-level logger (set up after log.setup_logging() is called)
logger = logging.getLogger("main")


# =============================================================================
# STARTUP
# =============================================================================

def startup(dry_run: bool) -> rm.SessionState:
    """
    Initialise logging, connect to MT5, validate symbol, return SessionState.
    Exits the process if any critical step fails.
    """
    log.setup_logging()

    mode = "DRY-RUN (no orders)" if dry_run else "LIVE DEMO"
    logger.info("=" * 55)
    logger.info(f"  EURUSD EA starting — {mode}")
    logger.info(f"  Symbol   : {config.SYMBOL}")
    logger.info(f"  HTF      : {config.HTF_TIMEFRAME}  EMA{config.EMA_MED}/{config.EMA_SLOW}")
    logger.info(f"  LTF      : {config.LTF_TIMEFRAME}  EMA{config.EMA_FAST} cross + RSI{config.RSI_PERIOD}")
    logger.info(f"  Risk     : {config.RISK_PER_TRADE*100:.1f}% / trade  |  RR {config.REWARD_RATIO}:1")
    logger.info(f"  DD limit : -{config.MAX_DAILY_LOSS_PCT*100:.1f}%  (no daily profit cap)")
    logger.info("=" * 55)

    # Connect to MT5
    if not df.connect():
        logger.critical("MT5 connection failed — exiting.")
        sys.exit(1)

    # Validate symbol is available
    sym = df.get_symbol_info()
    if sym is None:
        logger.critical(f"Symbol {config.SYMBOL} not found — check broker. Exiting.")
        df.disconnect()
        sys.exit(1)

    logger.info(
        f"Symbol OK | spread={sym.spread} pts | "
        f"min_lot={sym.volume_min} | lot_step={sym.volume_step}"
    )

    # Initialise session state
    state = rm.init_session()
    if state is None:
        logger.critical("Could not read account info — exiting.")
        df.disconnect()
        sys.exit(1)

    return state


# =============================================================================
# LAST-CANDLE TRACKER  (fire signal once per M15 bar, not every second)
# =============================================================================

def _get_last_bar_time() -> datetime:
    """Return the timestamp of the most recent closed M15 bar."""
    bars = df.get_bars(config.LTF_TIMEFRAME, 2)
    if bars is None or bars.empty:
        return datetime.min.replace(tzinfo=timezone.utc)
    return bars.index[-1].to_pydatetime()


# =============================================================================
# CLOSED TRADE JOURNAL SYNC
# =============================================================================

def _sync_closed_trades(
    prev_positions: list[dict],
    curr_positions: list[dict],
    state: rm.SessionState,
) -> None:
    """
    Detect trades that closed since the last loop tick (SL/TP hit or manual).
    Logs them to the CSV journal.
    """
    prev_tickets = {p["ticket"]: p for p in prev_positions}
    curr_tickets  = {p["ticket"] for p in curr_positions}

    closed = set(prev_tickets.keys()) - curr_tickets
    for ticket in closed:
        p = prev_tickets[ticket]
        log.log_trade_closed(
            ticket      = ticket,
            direction   = p["type"].upper(),
            lots        = p["volume"],
            entry_price = p["open_price"],
            close_price = p["open_price"],   # Exact close price not available here;
                                              # check journal or MT5 history for precise value
            profit      = p["profit"],
            reason      = "Closed by SL/TP or manually",
            daily_pnl   = df.get_daily_pnl_pct(),
        )


# =============================================================================
# MAIN LOOP
# =============================================================================

def run(dry_run: bool = False) -> None:
    """
    The main trading loop. Runs until interrupted (Ctrl+C) or kill switch fires.

    Loop cycle (every ~5 seconds):
      1. Check kill switch — suspend if daily loss ≥ 1%
      2. If kill switch just fired → close all positions
      3. Wait for a new M15 bar to close (prevents signal re-firing)
      4. Fetch H1 + M15 data and evaluate signal confluence
      5. If valid signal and trading allowed → open trade
      6. Manage open trades (partial close + breakeven)
      7. Log any trades that closed since last iteration
    """
    state = startup(dry_run)

    last_bar_time  = _get_last_bar_time()   # Track last processed bar
    prev_positions = df.get_open_positions() # For closed-trade detection
    kill_announced = False                   # Prevent log spam after kill

    logger.info("Main loop running. Press Ctrl+C to stop.")

    try:
        while True:

            # ── 1. KILL SWITCH CHECK ───────────────────────────────────────
            prev_killed = state.killed
            state = rm.check_kill_switch(state)

            # ── 2. CLOSE ALL ON KILL ───────────────────────────────────────
            if state.killed and not prev_killed:
                # Kill switch just triggered this iteration
                pnl = df.get_daily_pnl_pct()
                n_closed = om.close_all_positions()
                log.log_kill_switch(pnl, n_closed)
                kill_announced = True

            if state.killed:
                if not kill_announced:
                    logger.warning("EA suspended — waiting for session end.")
                    kill_announced = True
                time.sleep(30)
                continue

            # ── 3. NEW BAR GATE ───────────────────────────────────────────
            # Only process once per M15 bar close — avoids firing the
            # same signal multiple times within one candle.
            current_bar_time = _get_last_bar_time()
            if current_bar_time <= last_bar_time:
                time.sleep(5)
                continue

            last_bar_time = current_bar_time
            logger.info(f"New M15 bar: {current_bar_time.strftime('%H:%M')} UTC")

            # ── 4. FETCH DATA ─────────────────────────────────────────────
            h1_bars  = df.get_bars(config.HTF_TIMEFRAME, config.HTF_BARS)
            m15_bars = df.get_bars(config.LTF_TIMEFRAME, config.LTF_BARS)

            if h1_bars is None or m15_bars is None:
                logger.warning("Data fetch failed — skipping this bar.")
                time.sleep(5)
                continue

            price = df.get_current_price()
            if price is None:
                logger.warning("Price fetch failed — skipping this bar.")
                time.sleep(5)
                continue

            # ── 5. SIGNAL EVALUATION ──────────────────────────────────────
            signal = evaluate(
                h1_df       = h1_bars,
                m15_df      = m15_bars,
                current_ask = price["ask"],
                current_bid = price["bid"],
            )

            # ── 6. TRADE ENTRY ────────────────────────────────────────────
            if signal.type != SignalType.NONE:

                if not rm.is_trading_allowed(state):
                    logger.info(
                        f"Signal {signal.type.value} detected but trading blocked — "
                        f"max {config.MAX_OPEN_TRADES} trades already open."
                    )
                else:
                    if dry_run:
                        # Dry-run: log signal but don't send order
                        logger.info(
                            f"[DRY-RUN] Would open {signal.type.value} | "
                            f"entry={signal.entry_price:.5f} "
                            f"sl={signal.sl_price:.5f} "
                            f"tp={signal.tp_price:.5f}"
                        )
                    else:
                        ticket = om.open_trade(signal, state)
                        if ticket:
                            log.log_trade_open(
                                ticket      = ticket,
                                direction   = signal.type.value,
                                lots        = 0.0,        # Actual lots fetched below
                                entry_price = signal.entry_price,
                                sl_price    = signal.sl_price,
                                tp_price    = signal.tp_price,
                                atr_value   = signal.atr_value,
                                reasons     = signal.reasons,
                                daily_pnl   = df.get_daily_pnl_pct(),
                            )

            else:
                log.log_no_signal(" | ".join(signal.reasons))

            # ── 7. TRADE MANAGEMENT ───────────────────────────────────────
            # Check every open position for partial close / breakeven
            if not dry_run:
                tm.monitor_trades(state)

            # ── 8. CLOSED TRADE SYNC ──────────────────────────────────────
            curr_positions = df.get_open_positions()
            _sync_closed_trades(prev_positions, curr_positions, state)
            prev_positions = curr_positions

            # ── 9. SESSION SUMMARY (every bar) ────────────────────────────
            pnl_pct = df.get_daily_pnl_pct()
            open_count = len(curr_positions)
            logger.info(
                f"Status | daily_pnl={pnl_pct*100:+.3f}% | "
                f"open_trades={open_count}/{config.MAX_OPEN_TRADES} | "
                f"kill={'🔴 ACTIVE' if state.killed else '✅ OK'} | "
                f"session={'✅' if df.is_market_session() else '⏸ outside hours'}"
            )

            time.sleep(5)   # Brief pause before next tick check

    except KeyboardInterrupt:
        logger.info("\nKeyboard interrupt — shutting down cleanly.")

    finally:
        _shutdown(state)


# =============================================================================
# CLEAN SHUTDOWN
# =============================================================================

def _shutdown(state: rm.SessionState) -> None:
    """Print daily report, disconnect from MT5."""
    logger.info("Running shutdown sequence...")
    rm.print_daily_report(state)
    df.disconnect()
    logger.info("EA stopped.")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EURUSD Expert Advisor")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Detect signals and log them but do NOT send any orders to MT5.",
    )
    args = parser.parse_args()
    run(dry_run=args.dry_run)