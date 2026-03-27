# =============================================================================
# config.py — EURUSD EA Configuration
# All tunable parameters live here. Never hardcode values in other modules.
# =============================================================================

# --- MT5 Connection ---
MT5_LOGIN    = 0          # Your demo account number (int)
MT5_PASSWORD = ""         # Your demo account password
MT5_SERVER   = ""         # Your broker server name e.g. "ICMarkets-Demo"
                          # Leave LOGIN=0 if you want MT5 to use the
                          # already-logged-in terminal without re-auth.

# --- Instrument ---
SYMBOL    = "EURUSD"
DEVIATION = 10            # Max price deviation in points for order execution

# --- Timeframes (MT5 constants are imported in data_feed.py) ---
HTF_TIMEFRAME = "H1"      # Higher timeframe for trend filter
LTF_TIMEFRAME = "M15"     # Lower timeframe for entry trigger
HTF_BARS      = 300       # How many H1 bars to fetch
LTF_BARS      = 100       # How many M15 bars to fetch

# --- Indicators ---
EMA_SLOW   = 200          # Slow EMA period (H1 trend)
EMA_MED    = 50           # Medium EMA period (H1 trend confirmation)
EMA_FAST   = 9            # Fast EMA period (M15 entry trigger)
RSI_PERIOD = 14           # RSI period (M15)

# RSI zones — only trade inside these bands to avoid chasing
RSI_LONG_MIN  = 45        # RSI minimum for a long signal
RSI_LONG_MAX  = 65        # RSI maximum for a long signal
RSI_SHORT_MIN = 35        # RSI minimum for a short signal
RSI_SHORT_MAX = 55        # RSI maximum for a short signal

# --- Risk Management ---
RISK_PER_TRADE   = 0.015  # 1.5% of account equity risked per trade (compounding)
REWARD_RATIO     = 3.5    # Risk:Reward — TP = SL distance × this
MAX_OPEN_TRADES  = 3      # Maximum number of concurrent open trades
PARTIAL_CLOSE_R  = 1.2    # Close 50% of position at 1.2R (breakeven trigger)
PARTIAL_CLOSE_PCT= 0.5    # Fraction of position to close at partial TP

# --- Daily Drawdown Kill Switch ---
MAX_DAILY_LOSS_PCT = 0.01  # 1% — EA suspends if daily floating+realised loss hits this
# MAX_DAILY_GAIN_PCT removed — no daily profit cap, let all valid signals trade

# --- Session Filter (UTC times) ---
# Trade only during London + NY overlap for best EURUSD liquidity
SESSION_START_HOUR = 7    # 07:00 UTC (London open)
SESSION_END_HOUR   = 20   # 20:00 UTC (NY close)

# --- Magic Number ---
# Unique ID stamped on every order so the EA only manages its own trades
MAGIC_NUMBER = 20240101

# --- Logging ---
LOG_FILE   = "trades_journal.csv"
LOG_LEVEL  = "INFO"       # DEBUG | INFO | WARNING