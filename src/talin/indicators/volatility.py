import numpy as np
import pandas as pd


__all__ = [
    "trange", "atr", "natr"
]


def trange(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Calculates the True Range given the highs, lows and the closes (period-1).

    Args:
        high (pd.Series): Series of highs
        low (pd.Series): Series of lows
        close (pd.Series): Series of closes

    Returns:
        pd.Series: Series of the True range
    """
    return pd.Series(
        np.r_[
            [high-low],
            [(high-close.shift(1)).abs()],
            [(low-close.shift(1)).abs()]
        ].T.max(axis=1)
    )


def atr(high: pd.Series, low: pd.Series, close: pd.Series, periods=14) -> pd.Series:
    """Calculates the Average True Range given the highs, lows, closes with a set loopback.

    Args:
        high (pd.Series): Series of highs
        low (pd.Series): Series of lows
        close (pd.Series): Series of closes
        periods (int, optional): The number of periods to look back (window for simple moving average). Defaults to 14.

    Returns:
        pd.Series: The Average True Range series
    """

    return trange(high, low, close).rolling(periods).mean()


def natr(high: pd.Series, low: pd.Series, close: pd.Series, periods=14) -> pd.Series:
    """Calculates the Normalized True Range.

    Args:
        high (pd.Series): Series of highs
        low (pd.Series): Series of lows
        close (pd.Series): Series of closes
        periods (int, optional): N periods to loop back. Defaults to 14.

    Returns:
        pd.Series: Normalized True Range series
    """

    return atr(high, low, close, periods) / close * 100
