import pandas as pd
import numpy as np

__all__ = [
    "ad", "adosc", "obv"
]


def ad(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculates the Chaikin Accumulation Distribution Line.

    Args:
        high (pd.Series): Series of highs
        low (pd.Series): Series of lows
        close (pd.Series): Series of closes
        volume (pd.Series): Series of volumes

    Returns:
        pd.Series: Chaikin A/D Line series
    """

    # money flow multiplier
    n = ((close - low) - (high - close)) / (high - low)

    # money flow volume
    m = n * volume

    return m.shift(1) + m


def adosc(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
          short_periods=3, long_periods=10) -> pd.Series:
    """Calculates the Chaikin A/D Oscillator.

    Args:
        high (pd.Series): Series of highs
        low (pd.Series): Series of lows
        close (pd.Series): Series of closes
        volume (pd.Series): Series of volume
        short_periods (int, optional): N periods to use for short ema. Defaults to 3.
        long_periods (int, optional): N periods to use for long ema. Defaults to 10.

    Returns:
        pd.Series: Chaikin A/D Oscillator series
    """

    ad_line = ad(high, low, close, volume)
    return ad_line.ewm(span=short_periods, adjust=False).mean() - ad_line.ewm(span=long_periods, adjust=False).mean()


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculates the On-Balance Volume Indicator

    Args:
        close (pd.Serie): Series of closes
        volume (pd.Series): Series of Volumes

    Returns:
        pd.Series: OBV series
    """

    obvol = np.where(
        close > close.shift(1), volume, np.where(
            close < close.shift(1), -volume, 0  # if NaN or equal, will be zero
        )
    ).cumsum()

    obvol[0] = np.NaN  # first value should be NaN because difference is NaN
    return obvol
