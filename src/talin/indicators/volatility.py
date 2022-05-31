import numpy as np
import pandas as pd
"""

# TODO refactor to use predominantly pandas series
# TODO consider replacing loopback with periods 

NATR                 Normalized Average True Range

"""

__all__ = [
    "trange", "atr"
]


def trange(high, low, prev_close):
    # TODO consider if prev_close should become close
    """Calculates the True Range given the highs, lows and the previous closes (period-1). Note that inputs must be given as numpy arrays or similar objects such as pandas series. 

    Args:
        high (numpy array): array of highs
        low (numpy array): array of lows
        prev_close (numpy array): array of previous closes (period-1)

    Returns:
        numpy array: array of the True range for each period
    """
    return np.r_[high-low, high-prev_close, prev_close-low].T.max(axis=1)


def atr(high, low, prev_close, loopback=14):
    # TODO consider if prev_close should become close
    """Calculates the Average True Range given the highs, lows and the previous closes (period-1) with a set loopback. Note that the input arrays (high, low and prev_close) must be given as numpy arrays or similar objects such as pandas series.

    Args:
        high (numpy array): array of highs
        low (numpy array): array of lows
        prev_close (numpy array): array of pervious closes (period-1)
        loopback (int, optional): The number of periods to use as the window for the simple moving average. Defaults to 14.

    Returns:
        numpy array: The Average True Range for each period
    """

    return pd.Series(trange(high, low, prev_close)).rolling(loopback).mean().values


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

    return atr(high, low, close.shift(1), periods) / close * 100
