import numpy as np
import pandas as pd
"""

TODO IMPLEMENT

NATR                 Normalized Average True Range

"""

__all__ = [
    "trange", "atr"
]


def trange(high, low, prev_close):
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
