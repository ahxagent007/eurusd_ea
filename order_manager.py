# =============================================================================
# order_manager.py — Order Execution & Trade Management
# Handles sending, modifying, and partially closing trades via MT5.
# Only this module calls mt5.order_send() and mt5.positions_get().
# =============================================================================

import logging
import MetaTrader5 as mt5
from typing import Optional

import config
import data_feed as df
from risk_manager import SessionState, calculate_lot_size
from signal_engine import Signal, SignalType

logger = logging.getLogger(__name__)


# =============================================================================
# OPEN TRADE
# =============================================================================

def open_trade(signal: Signal, state: SessionState) -> Optional[int]:
    """
    Place a market order based on the Signal object.
    Returns the ticket number on success, None on failure.

    Steps:
      1. Calculate lot size (1% equity risk)
      2. Build the MT5 trade request
      3. Send via order_send()
      4. Validate the result
    """
    # --- Position sizing ---
    lots = calculate_lot_size(signal.sl_distance)
    if lots is None:
        logger.error("open_trade aborted — could not calculate lot size.")
        return None

    # --- Build request ---
    order_type = mt5.ORDER_TYPE_BUY if signal.type == SignalType.BUY else mt5.ORDER_TYPE_SELL

    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       config.SYMBOL,
        "volume":       lots,
        "type":         order_type,
        "price":        signal.entry_price,
        "sl":           signal.sl_price,
        "tp":           signal.tp_price,
        "deviation":    config.DEVIATION,
        "magic":        config.MAGIC_NUMBER,
        "comment":      f"EA_{signal.type.value}",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    # --- Send order ---
    result = mt5.order_send(request)

    if result is None:
        logger.error(f"order_send returned None — {mt5.last_error()}")
        return None

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(
            f"Order failed | retcode={result.retcode} "
            f"({_retcode_str(result.retcode)}) | comment={result.comment}"
        )
        return None

    ticket = result.order
    logger.info(
        f"✅  Trade opened | ticket={ticket} | {signal.type.value} "
        f"{lots} lots @ {signal.entry_price:.5f} | "
        f"SL={signal.sl_price:.5f} TP={signal.tp_price:.5f}"
    )
    return ticket


# =============================================================================
# PARTIAL CLOSE (50% at 1:1 R)
# =============================================================================

def partial_close(position: dict, state: SessionState) -> bool:
    """
    Close PARTIAL_CLOSE_PCT of the position volume (default 50%).
    Called when price reaches the 1:1 R level.
    Marks the ticket in state.partial_closed to prevent repeat firing.

    Returns True on success.
    """
    symbol = df.get_symbol_info()
    if symbol is None:
        return False

    close_volume = round(
        position["volume"] * config.PARTIAL_CLOSE_PCT / symbol.volume_step
    ) * symbol.volume_step
    close_volume = round(close_volume, 2)
    close_volume = max(close_volume, symbol.volume_min)

    # Remaining volume must also be >= min lot
    remaining = round(position["volume"] - close_volume, 2)
    if remaining < symbol.volume_min:
        logger.warning(
            f"Partial close skipped — remaining volume {remaining} "
            f"< min lot {symbol.volume_min}"
        )
        return False

    price = df.get_current_price()
    if price is None:
        return False

    close_price = (
        price["bid"] if position["type"] == "buy" else price["ask"]
    )
    close_type = (
        mt5.ORDER_TYPE_SELL if position["type"] == "buy" else mt5.ORDER_TYPE_BUY
    )

    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       config.SYMBOL,
        "volume":       close_volume,
        "type":         close_type,
        "position":     position["ticket"],
        "price":        close_price,
        "deviation":    config.DEVIATION,
        "magic":        config.MAGIC_NUMBER,
        "comment":      "EA_PARTIAL_TP",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)

    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        retcode = result.retcode if result else "None"
        logger.error(
            f"Partial close failed | ticket={position['ticket']} "
            f"retcode={retcode}"
        )
        return False

    state.partial_closed.add(position["ticket"])
    logger.info(
        f"✅  Partial close | ticket={position['ticket']} | "
        f"closed {close_volume} lots @ {close_price:.5f} | "
        f"{remaining} lots remaining"
    )
    return True


# =============================================================================
# MOVE SL TO BREAKEVEN
# =============================================================================

def move_sl_to_breakeven(position: dict) -> bool:
    """
    Move the stop-loss to the trade's open price (breakeven).
    Called immediately after a successful partial close.

    Returns True on success.
    """
    symbol = df.get_symbol_info()
    if symbol is None:
        return False

    # Add 1 pip buffer above entry for buys, below for sells
    pip = symbol.point * 10
    if position["type"] == "buy":
        new_sl = round(position["open_price"] + pip, 5)   # 1 pip above entry
    else:
        new_sl = round(position["open_price"] - pip, 5)   # 1 pip below entry

    request = {
        "action":   mt5.TRADE_ACTION_SLTP,
        "symbol":   config.SYMBOL,
        "position": position["ticket"],
        "sl":       new_sl,
        "tp":       position["tp"],          # Keep existing TP unchanged
    }

    result = mt5.order_send(request)

    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        retcode = result.retcode if result else "None"
        logger.error(
            f"Breakeven move failed | ticket={position['ticket']} "
            f"retcode={retcode}"
        )
        return False

    logger.info(
        f"✅  SL → Breakeven | ticket={position['ticket']} | "
        f"new SL={new_sl:.5f} (open was {position['open_price']:.5f})"
    )
    return True


# =============================================================================
# CLOSE ALL POSITIONS (kill switch)
# =============================================================================

def close_all_positions() -> int:
    """
    Emergency close — called when the daily drawdown kill switch fires.
    Closes every open position belonging to this EA.
    Returns the number of positions successfully closed.
    """
    positions = df.get_open_positions()
    if not positions:
        logger.info("close_all_positions: nothing to close.")
        return 0

    price_info = df.get_current_price()
    if price_info is None:
        logger.error("close_all_positions: cannot get current price.")
        return 0

    closed = 0
    for pos in positions:
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
            "comment":      "EA_KILL_SWITCH",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.warning(
                f"🔴  Force closed ticket={pos['ticket']} "
                f"| {pos['type'].upper()} {pos['volume']} lots "
                f"| profit={pos['profit']:.2f}"
            )
            closed += 1
        else:
            retcode = result.retcode if result else "None"
            logger.error(f"Failed to force close ticket={pos['ticket']} retcode={retcode}")

    logger.warning(f"Kill switch closed {closed}/{len(positions)} positions.")
    return closed


# =============================================================================
# MANAGE OPEN TRADES (call every tick / every M15 bar)
# =============================================================================

def manage_open_trades(state: SessionState) -> None:
    """
    Loop through all open positions and apply trade management rules:
      1. If price hit 1:1 R and not yet partially closed → partial close + BE
      2. (Future extension point: trailing stop logic goes here)
    """
    from risk_manager import should_partial_close

    positions = df.get_open_positions()
    for pos in positions:
        if should_partial_close(pos, state):
            logger.info(
                f"Price at 1:1 R for ticket={pos['ticket']} — "
                f"executing partial close + breakeven."
            )
            if partial_close(pos, state):
                move_sl_to_breakeven(pos)


# =============================================================================
# RETCODE LOOKUP (human-readable errors)
# =============================================================================

_RETCODES = {
    10004: "Requote",
    10006: "Request rejected",
    10007: "Request cancelled",
    10008: "Order placed",
    10009: "Request completed",
    10010: "Only part of request completed",
    10011: "Request processing error",
    10012: "Request timeout",
    10013: "Invalid request",
    10014: "Invalid volume",
    10015: "Invalid price",
    10016: "Invalid stops",
    10017: "Trade disabled",
    10018: "Market closed",
    10019: "Insufficient funds",
    10020: "Prices changed",
    10021: "No quotes",
    10022: "Invalid expiration",
    10023: "Order state changed",
    10024: "Too frequent requests",
    10025: "No changes",
    10026: "Autotrading disabled by server",
    10027: "Autotrading disabled by client",
    10028: "Request locked",
    10029: "Order or position frozen",
    10030: "Unsupported filling type",
    10031: "No connection",
    10032: "Not allowed",
    10033: "Limit orders exceeded",
    10034: "Volume limit exceeded",
    10035: "Invalid order",
    10036: "Position already closed",
}

def _retcode_str(code: int) -> str:
    return _RETCODES.get(code, f"Unknown code {code}")