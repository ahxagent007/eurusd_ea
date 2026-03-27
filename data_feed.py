# =============================================================================
# data_feed.py — MT5 Connection & Data Fetching
# Handles all communication with the MetaTrader 5 terminal.
# Every other module imports from here — nothing else touches MT5 directly.
# =============================================================================

import logging
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timezone
from typing import Optional

import config

logger = logging.getLogger(__name__)

# Map human-readable timeframe strings → MT5 constants
TIMEFRAME_MAP = {
    "M1":  mt5.TIMEFRAME_M1,
    "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1":  mt5.TIMEFRAME_H1,
    "H4":  mt5.TIMEFRAME_H4,
    "D1":  mt5.TIMEFRAME_D1,
}


# =============================================================================
# CONNECTION
# =============================================================================

from dotenv import load_dotenv
import os

load_dotenv()   # reads .env from the current directory

MT5_LOGIN    = int(os.getenv("MT5_LOGIN", "0"))
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
MT5_SERVER   = os.getenv("MT5_SERVER",   "")

def connect() -> bool:
    """
    Initialise the MT5 terminal and log in to the configured demo account.
    Returns True on success, False on failure.
    """
    if not mt5.initialize():
        logger.error(f"MT5 initialize() failed — {mt5.last_error()}")
        return False

    # If LOGIN is 0 we trust the already-logged-in terminal session
    if config.MT5_LOGIN == 0:
        logger.info("MT5 connected (using existing terminal session)")
        return True

    authorised = mt5.login(
        login=config.MT5_LOGIN,
        password=config.MT5_PASSWORD,
        server=config.MT5_SERVER,
    )
    if not authorised:
        logger.error(f"MT5 login failed — {mt5.last_error()}")
        mt5.shutdown()
        return False

    info = mt5.account_info()
    logger.info(
        f"MT5 connected | Account: {info.login} | "
        f"Broker: {info.company} | "
        f"Balance: {info.balance:.2f} {info.currency}"
    )
    return True


def disconnect() -> None:
    """Cleanly shut down the MT5 connection."""
    mt5.shutdown()
    logger.info("MT5 connection closed.")


# =============================================================================
# MARKET DATA
# =============================================================================

def get_bars(timeframe_str: str, count: int) -> Optional[pd.DataFrame]:
    """
    Fetch the last `count` closed OHLCV bars for EURUSD on the given timeframe.

    Returns a DataFrame with columns:
        time, open, high, low, close, tick_volume, spread, real_volume
    The most recent bar (index -1) is the last CLOSED bar.
    Returns None on error.
    """
    tf = TIMEFRAME_MAP.get(timeframe_str)
    if tf is None:
        logger.error(f"Unknown timeframe: {timeframe_str}")
        return None

    # Fetch count+1 bars and drop the still-forming current bar
    rates = mt5.copy_rates_from_pos(config.SYMBOL, tf, 0, count + 1)
    if rates is None or len(rates) == 0:
        logger.error(
            f"copy_rates_from_pos failed for {timeframe_str} — {mt5.last_error()}"
        )
        return None

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)

    # Drop the last row (incomplete / forming candle)
    df = df.iloc[:-1]

    logger.debug(f"Fetched {len(df)} {timeframe_str} bars. Last close: {df['close'].iloc[-1]:.5f}")
    return df


def get_current_price() -> Optional[dict]:
    """
    Return the latest bid/ask for EURUSD.
    Returns dict with keys: bid, ask, spread_points
    """
    tick = mt5.symbol_info_tick(config.SYMBOL)
    if tick is None:
        logger.error(f"symbol_info_tick failed — {mt5.last_error()}")
        return None
    spread = round((tick.ask - tick.bid) / mt5.symbol_info(config.SYMBOL).point)
    return {
        "bid": tick.bid,
        "ask": tick.ask,
        "spread_points": spread,
        "time": datetime.fromtimestamp(tick.time, tz=timezone.utc),
    }


def get_symbol_info() -> Optional[mt5.SymbolInfo]:
    """Return full symbol specification for EURUSD (lot size, pip value, etc.)."""
    info = mt5.symbol_info(config.SYMBOL)
    if info is None:
        logger.error(f"symbol_info failed — {mt5.last_error()}")
    return info


# =============================================================================
# ACCOUNT DATA
# =============================================================================

def get_account_info() -> Optional[dict]:
    """
    Return key account metrics needed by the risk manager.
    Keys: balance, equity, margin, free_margin, profit, currency
    """
    acc = mt5.account_info()
    if acc is None:
        logger.error(f"account_info failed — {mt5.last_error()}")
        return None
    return {
        "balance":     acc.balance,
        "equity":      acc.equity,
        "margin":      acc.margin,
        "free_margin": acc.margin_free,
        "profit":      acc.profit,       # Floating P&L of open positions
        "currency":    acc.currency,
    }


# =============================================================================
# OPEN POSITIONS
# =============================================================================

def get_open_positions() -> list[dict]:
    """
    Return all open positions belonging to this EA (filtered by MAGIC_NUMBER
    and SYMBOL).

    Each dict contains:
        ticket, type ('buy'|'sell'), volume, open_price,
        sl, tp, profit, open_time
    """
    positions = mt5.positions_get(symbol=config.SYMBOL)
    if positions is None:
        return []

    result = []
    for p in positions:
        if p.magic != config.MAGIC_NUMBER:
            continue
        result.append({
            "ticket":     p.ticket,
            "type":       "buy" if p.type == mt5.ORDER_TYPE_BUY else "sell",
            "volume":     p.volume,
            "open_price": p.price_open,
            "sl":         p.sl,
            "tp":         p.tp,
            "profit":     p.profit,
            "open_time":  datetime.fromtimestamp(p.time, tz=timezone.utc),
        })
    return result


# =============================================================================
# DAILY P&L
# =============================================================================

def get_daily_realised_pnl() -> float:
    """
    Calculate total REALISED P&L for today (closed trades only).
    Queries MT5 deal history from midnight UTC to now.
    Returns the sum in account currency.
    """
    today_midnight = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    now = datetime.now(timezone.utc)

    deals = mt5.history_deals_get(today_midnight, now)
    if deals is None:
        return 0.0

    total = sum(
        d.profit
        for d in deals
        if d.magic == config.MAGIC_NUMBER and d.symbol == config.SYMBOL
    )
    return total


def get_daily_pnl_pct() -> float:
    """
    Return today's total P&L percentage relative to starting balance.
    Combines realised (closed trades) + floating (open positions).
    Positive = gain, Negative = loss.
    """
    acc = get_account_info()
    if acc is None:
        return 0.0

    realised  = get_daily_realised_pnl()
    floating  = acc["profit"]          # Current floating P&L
    total_pnl = realised + floating

    # Use balance as the denominator (balance excludes floating)
    balance = acc["balance"]
    if balance == 0:
        return 0.0

    return total_pnl / balance         # e.g. -0.008 = -0.8%


# =============================================================================
# SESSION CHECK
# =============================================================================

def is_market_session() -> bool:
    """
    Returns True if the current UTC time falls within the configured
    trading session window (London + NY overlap by default).
    """
    current_hour = datetime.now(timezone.utc).hour
    return config.SESSION_START_HOUR <= current_hour < config.SESSION_END_HOUR


# =============================================================================
# QUICK CONNECTIVITY TEST  (run this file directly to verify your setup)
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    print("=" * 60)
    print("  MT5 Data Feed — Connection Test")
    print("=" * 60)

    if not connect():
        print("❌  Connection failed. Check config.py credentials.")
        exit(1)

    # Account snapshot
    acc = get_account_info()
    print(f"\n✅  Account: {acc}")

    # Latest price
    price = get_current_price()
    print(f"\n✅  EURUSD price: bid={price['bid']} ask={price['ask']} spread={price['spread_points']}pts")

    # H1 bars
    h1 = get_bars(config.HTF_TIMEFRAME, 5)
    print(f"\n✅  Last 5 H1 bars:\n{h1[['open','high','low','close']].tail()}")

    # M15 bars
    m15 = get_bars(config.LTF_TIMEFRAME, 5)
    print(f"\n✅  Last 5 M15 bars:\n{m15[['open','high','low','close']].tail()}")

    # Daily P&L
    pnl_pct = get_daily_pnl_pct()
    print(f"\n✅  Daily P&L: {pnl_pct*100:.3f}%")

    # Open positions
    positions = get_open_positions()
    print(f"\n✅  Open positions (this EA): {len(positions)}")

    # Session check
    print(f"\n✅  In trading session: {is_market_session()}")

    disconnect()
    print("\n" + "=" * 60)
    print("  All checks passed. Ready to build indicators.py next.")
    print("=" * 60)