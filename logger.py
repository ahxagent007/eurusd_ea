# =============================================================================
# logger.py — Trade Journal & Console Logging Setup
# Writes every trade event to a CSV file and sets up structured console logs.
# =============================================================================

import csv
import logging
import logging.handlers
import os
from datetime import datetime, timezone
from typing import Optional

import config


# =============================================================================
# CONSOLE + FILE LOGGING SETUP
# =============================================================================

def setup_logging() -> None:
    """
    Configure the root logger.
    - Console: coloured level prefix, human-readable timestamps
    - Rotating file: ea.log, max 5MB × 3 backups (keeps logs manageable)
    Call once at the very start of main.py before anything else.
    """
    level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)

    fmt = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    # Root logger
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    # --- Console handler ---
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(_ColouredFormatter(fmt, datefmt=datefmt))
    root.addHandler(ch)

    # --- Rotating file handler ---
    fh = logging.handlers.RotatingFileHandler(
        "ea.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)   # Always log DEBUG to file, even if console is INFO
    fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.addHandler(fh)

    logging.info("Logging initialised — console + ea.log")


class _ColouredFormatter(logging.Formatter):
    """Add ANSI colour codes to console log levels."""
    COLOURS = {
        "DEBUG":    "\033[36m",    # Cyan
        "INFO":     "\033[32m",    # Green
        "WARNING":  "\033[33m",    # Yellow
        "ERROR":    "\033[31m",    # Red
        "CRITICAL": "\033[35m",    # Magenta
    }
    RESET = "\033[0m"

    def format(self, record):
        colour = self.COLOURS.get(record.levelname, "")
        record.levelname = f"{colour}{record.levelname}{self.RESET}"
        return super().format(record)


# =============================================================================
# CSV TRADE JOURNAL
# =============================================================================

_CSV_HEADERS = [
    "timestamp_utc",
    "event",           # OPEN | PARTIAL_CLOSE | BREAKEVEN | CLOSED | KILL_SWITCH
    "ticket",
    "symbol",
    "direction",       # BUY | SELL
    "lots",
    "entry_price",
    "sl_price",
    "tp_price",
    "close_price",
    "profit",
    "daily_pnl_pct",
    "atr_at_signal",
    "reason",          # Signal reasons or close reason
]


def _ensure_journal(filepath: str) -> None:
    """Create the CSV file with headers if it doesn't exist yet."""
    if not os.path.exists(filepath):
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_HEADERS)
            writer.writeheader()
        logging.info(f"Trade journal created: {filepath}")


def _write_row(row: dict) -> None:
    """Append one row to the CSV journal."""
    _ensure_journal(config.LOG_FILE)
    # Fill any missing keys with empty string
    full_row = {k: row.get(k, "") for k in _CSV_HEADERS}
    with open(config.LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_HEADERS)
        writer.writerow(full_row)


# =============================================================================
# JOURNAL EVENT FUNCTIONS
# =============================================================================

def log_trade_open(
    ticket:      int,
    direction:   str,
    lots:        float,
    entry_price: float,
    sl_price:    float,
    tp_price:    float,
    atr_value:   float,
    reasons:     list,
    daily_pnl:   float = 0.0,
) -> None:
    """Call immediately after a successful order_send."""
    _write_row({
        "timestamp_utc": _now(),
        "event":         "OPEN",
        "ticket":        ticket,
        "symbol":        config.SYMBOL,
        "direction":     direction,
        "lots":          lots,
        "entry_price":   round(entry_price, 5),
        "sl_price":      round(sl_price, 5),
        "tp_price":      round(tp_price, 5),
        "atr_at_signal": round(atr_value, 5),
        "daily_pnl_pct": f"{daily_pnl*100:.3f}%",
        "reason":        " | ".join(reasons),
    })
    logging.info(
        f"JOURNAL OPEN | ticket={ticket} {direction} {lots}L "
        f"@ {entry_price:.5f} SL={sl_price:.5f} TP={tp_price:.5f}"
    )


def log_partial_close(
    ticket:      int,
    direction:   str,
    lots_closed: float,
    close_price: float,
    profit:      float,
    daily_pnl:   float = 0.0,
) -> None:
    """Call after a successful partial close."""
    _write_row({
        "timestamp_utc": _now(),
        "event":         "PARTIAL_CLOSE",
        "ticket":        ticket,
        "symbol":        config.SYMBOL,
        "direction":     direction,
        "lots":          lots_closed,
        "close_price":   round(close_price, 5),
        "profit":        round(profit, 2),
        "daily_pnl_pct": f"{daily_pnl*100:.3f}%",
        "reason":        "1:1 R reached",
    })
    logging.info(
        f"JOURNAL PARTIAL | ticket={ticket} closed {lots_closed}L "
        f"@ {close_price:.5f} profit≈{profit:.2f}"
    )


def log_breakeven(ticket: int, new_sl: float) -> None:
    """Call after SL is moved to breakeven."""
    _write_row({
        "timestamp_utc": _now(),
        "event":         "BREAKEVEN",
        "ticket":        ticket,
        "symbol":        config.SYMBOL,
        "sl_price":      round(new_sl, 5),
        "reason":        "SL moved to entry",
    })
    logging.info(f"JOURNAL BREAKEVEN | ticket={ticket} new_sl={new_sl:.5f}")


def log_trade_closed(
    ticket:      int,
    direction:   str,
    lots:        float,
    entry_price: float,
    close_price: float,
    profit:      float,
    reason:      str,
    daily_pnl:   float = 0.0,
) -> None:
    """
    Call when a trade is fully closed (by SL, TP, or kill switch).
    The main loop detects closure by comparing open positions each bar.
    """
    _write_row({
        "timestamp_utc": _now(),
        "event":         "CLOSED",
        "ticket":        ticket,
        "symbol":        config.SYMBOL,
        "direction":     direction,
        "lots":          lots,
        "entry_price":   round(entry_price, 5),
        "close_price":   round(close_price, 5),
        "profit":        round(profit, 2),
        "daily_pnl_pct": f"{daily_pnl*100:.3f}%",
        "reason":        reason,
    })
    emoji = "✅" if profit >= 0 else "❌"
    logging.info(
        f"JOURNAL CLOSED {emoji} | ticket={ticket} {direction} "
        f"@ {close_price:.5f} | profit={profit:+.2f} | {reason}"
    )


def log_kill_switch(daily_pnl: float, positions_closed: int) -> None:
    """Call when the kill switch fires."""
    _write_row({
        "timestamp_utc": _now(),
        "event":         "KILL_SWITCH",
        "symbol":        config.SYMBOL,
        "daily_pnl_pct": f"{daily_pnl*100:.3f}%",
        "reason":        f"Daily loss limit hit — {positions_closed} positions closed",
    })
    logging.warning(
        f"JOURNAL KILL_SWITCH | daily_pnl={daily_pnl*100:.3f}% "
        f"| closed {positions_closed} positions"
    )


def log_no_signal(reason: str) -> None:
    """Optional — log why no signal fired (DEBUG level only, not written to CSV)."""
    logging.debug(f"No signal: {reason}")


# =============================================================================
# CLOSED TRADE DETECTION HELPER
# =============================================================================

def detect_closed_trades(
    previous_tickets: set,
    current_tickets:  set,
) -> set:
    """
    Compare ticket sets between two loop iterations.
    Returns the set of tickets that were open last bar but are now gone
    (closed by SL, TP, or manually).
    """
    return previous_tickets - current_tickets


# =============================================================================
# UTILITIES
# =============================================================================

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    setup_logging()
    import os

    # Use a temp journal for testing
    config.LOG_FILE = "test_journal.csv"

    log_trade_open(
        ticket=123456, direction="BUY", lots=0.50,
        entry_price=1.08500, sl_price=1.08300, tp_price=1.08900,
        atr_value=0.00085, reasons=["H1 bullish", "EMA9 bull cross", "RSI=52.4"],
        daily_pnl=0.002,
    )
    log_partial_close(
        ticket=123456, direction="BUY", lots_closed=0.25,
        close_price=1.08700, profit=50.0, daily_pnl=0.005,
    )
    log_breakeven(ticket=123456, new_sl=1.08510)
    log_trade_closed(
        ticket=123456, direction="BUY", lots=0.25,
        entry_price=1.08500, close_price=1.08900, profit=100.0,
        reason="TP hit", daily_pnl=0.010,
    )
    log_kill_switch(daily_pnl=-0.010, positions_closed=1)

    print(f"\nJournal written to: {config.LOG_FILE}")
    with open(config.LOG_FILE, "r") as f:
        print(f.read())

    os.remove(config.LOG_FILE)
    print("✅  logger.py OK")