import numpy as np
import pandas as pd
from src.talin.stats import simple_stats
from src.talin.indicators import volatility

__all__ = [
    "plus_dm", "minus_dm", "di", "dx", "adx", "adxr",
    "apo", "aroon", "aroon_osc", "bop", "cci", "cmo",
    "macd", "mfi", "mom", "ppo", "roc", "rsi", "stochf",
    "stoch", "stoch_rsi", "trix", "ultosc", "willr",
]

# TODO refactor to explicityly use predominantly use pd Series


def plus_dm(high: pd.Series) -> pd.Series:
    """Calculates the Positive Directional Movement

    Args:
        high (pd.Series): highs series

    Returns:
        pd.Series: positive directional movement
    """

    pDM = high - high.shift()
    pDM[pDM < 0] = 0

    return pDM


def minus_dm(low: pd.Series) -> pd.Series:
    """Calculates the Negative Directional Movement.
    Args:
        low (pd.Series): lows series

    Returns:
        pd.Series: negative directional movement
    """

    nDM = low.shift() - low
    nDM[nDM < 0] = 0

    return nDM


def di(high: pd.Series, low: pd.Series, close: pd.Series, periods=14) -> tuple[pd.Series, pd.Series]:
    """Calculates the (postive or negative) Directional indicator.

    Args:
        high (pd.Series): highs series
        low (pd.Series): lows series
        close (pd.Series): closes series
        periods (int, optional): The number of periods to lookback/use as the rolling average window. Defaults to 14.

    Returns:
        tuple[pd.Series, pd.Series]: returns a tuple of (positive DI, negative DI) series.
    """

    atr = volatility.atr(high, low, close, periods)
    pDI = plus_dm(high).rolling(periods).mean() / atr * 100
    nDI = minus_dm(low).rolling(periods).mean() / atr * 100

    return pDI, nDI


def dx(pDI: pd.Series, nDI: pd.Series) -> pd.Series:
    """Calculates the Directional Movement Index given the postive and negative Directional Indicators.
    Args:
        pDI (pd.Series): positive Directional Indicator
        nDI (pd.Series): negative Directional Indicator

    Returns:
        pd.Series: Directional Movement Index
    """

    pDI, nDI = (pd.Series(pDI), pd.Series(nDI))
    return (pDI - nDI).abs() / (pDI + nDI).abs() * 100


def adx(high: pd.Series, low: pd.Series, close: pd.Series, periods=14) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculates the Average Directional Index given the highs, lows, closes, and the loopback (window).

    Args:
        high (pd.Series): series of highs
        low (pd.Series): series of lows
        close (pd.Series): series of closes 
        periods (int, optional): The number of periods to lookback/use as the window. Defaults to 14.

    Returns:
        (pd.Series, pd.Series, pd.Series): Returns a tuple of the positive directional indicator, the negative direction indicator, and the average directional index
    """

    pDI, nDI = di(high, low, close, periods=14)
    dx_ = dx(pDI, nDI)

    # TODO determine if there is a more efficient way to do this
    # without using cython or numba

    first_adx = np.mean(dx_.values[:periods])
    adx_ = [np.NaN] * (periods-1) + [first_adx]

    for i in range(periods, dx_.size):
        adx_.append(
            ((adx_[i-1] * (periods-1)) + dx_[i]) / periods
        )

    return pDI, nDI, pd.Series(adx_)


def adxr(adx, lookback=2):
    """Calculates the Average Direction Movement Rating given the ADX and the number of periods to look backwards. ADX input must be given as a numpy array or similar object like pandas series. 

    Args:
        adx (numpy array): array/series of Average Directional Movement Indexes
        lookback (int): Number of periods to look backwards. Defaults to 2.

    Returns:
        pandas Series: The Average Direction Movement Rating
    """

    return (adx - pd.Series(adx).shift(lookback)) / 2


def apo(close, short_periods=14, long_periods=30):
    """Calculates Absolute Price Oscilliator given the a list of closing values and the number of periods for the short term and long term

    Args:
        close (Numerical List): List of closing values
        short_periods (int, optional): Number of periods for the short term. Defaults to 14.
        long_periods (int, optional): Number of periods for the long term. Defaults to 30.

    Returns:
        pandas Series: Absolute Price Oscillator series
    """

    close = pd.Series(close)
    return close.ewm(alpha=1/short_periods) - close.ewm(alpha=1/long_periods)


def aroon(high, low, periods=25):
    """Calculates the AROON indicator given the lists of highs and lows and the number of periods to use in the calculation.

    Args:
        high (Numerical List): List of highs
        low (Numerical List): List of lows
        periods (int, optional): Number of periods. Defaults to 25.

    Returns:
        (pandas Series, pandas Series): Returns the AROON up and AROON down series.
    """

    high, low = (pd.Series(high), pd.Series(low))

    aroon_up = 100 * \
        high.rolling(periods + 1).apply(lambda x: x.argmax()) / periods
    aroon_down = 100 * \
        low.rolling(periods + 1).apply(lambda x: x.argmin()) / periods

    return aroon_up, aroon_down


def aroon_osc(high, low, periods=25):
    """Calculates the AROON Oscillator given the lists of high and lows and the number of periods to use in the calculation.

    Args:
        high (Numerical List): List of highs
        low (Numerical List): List of lows
        periods (int, optional): Number of periods. Defaults to 25.
    Returns:
        pandas Series: AROON Oscillator series
    """

    aroon_up, aroon_down = aroon(high, low, periods)
    return aroon_up - aroon_down


def bop(high, low, open, close):
    """Calculates the Balance of Power indicator, given the input arrays of high, low, open, and close. Note that these input arrays must be numpy arrays or similar objects such as pandas Series

    Args:
        high (numpy array): array of highs
        low (numpy array): array of lows
        open (numpy array): array of opens
        close (numpy array): array of closes

    Returns:
        numpy array/same as input type: Balance of Power
    """

    return (close - open) / (high - low)


def cci(high, low, close, periods=20):
    """Calculates the Commodity Channel Index, given the input arrays of high, low, open and close. Note that these input arrays must be numpy arrays or similar objects such as pandas Series.

    Args:
        high (numpy array): array of highs
        low (numpy array): array of lows
        close (numpy array): array of closes
        periods (int, optional): Number of periods for the moving average. Defaults to 20.

    Returns:
        pandas Series: Commodity Channel Index series.
    """

    typicalP = pd.Series(simple_stats.typical_price(high, low, close))
    movingAvg = typicalP.rolling(periods).mean()
    meanDeviation = (
        typicalP - movingAvg).abs().rolling(periods).mean() / periods

    return (typicalP - movingAvg) / (0.015 * meanDeviation)


def cmo(close, periods=9):
    """Calculates the Chande Momentum Oscillator given the input list of closes and the number of periods to look back.

    Args:
        close (Numerical List): List of closes
        periods (int, optional): Number of periods to look back. Defaults to 9.

    Returns:
        pandas Series: Chande Momenum Oscillator series
    """

    close = pd.Series(close).diff()

    up_down = close.diff().apply(lambda x: 1 if x > 0 else 0 if x < 0 else x)
    nHigher = up_down.roling(periods).sum()
    nLower = (1-up_down).rolling(periods).sum()

    return (nHigher - nLower) / (nHigher + nLower) * 100


def macd(close, short_periods=12, long_periods=26):
    """Calculates the Moving Average Convergence Divergence given the input of a list of closes, number of period for short term ema, and number periods for long term ema.

    Args:
        close (Numerical List): List of closes
        short_periods (int, optional): Number of periods for short term ema. Defaults to 12.
        long_periods (int, optional): Number of periods for long term ema. Defaults to 26.

    Returns:
        pandas Series: Moving Average Convergence Diveregence
    """

    close = pd.Series(close)
    return close.ewm(alpha=1/short_periods) - close.ewm(alpha=1/long_periods)


def mfi(high, low, close, volume, periods=14):
    """Calculates the Money Flow Index given the input arrays high, low, close, and volume, as well as the number of periods to look back. Note that the input arrays must be given as numpy arrays or similar objects such as pandas Series.

    Args:
        high (numpy array): Array of highs
        low (numpy array): Array of lows
        close (numpy array): Array of closes
        volume (numpy array): Array of volumes
        periods (int, optional): Number of periods to look back. Defaults to 14.

    Returns:
        pandas Series: Money Flow Index series
    """

    typicalP = pd.Series(simple_stats.typical_price(high, low, close))
    typicalDiff = typicalP.diff()
    moneyFlow = typicalP * volume

    # the final else accomidates for NaN
    positiveNegative = typicalDiff.apply(
        lambda x: 1 if x > 0 else 0 if x < 0 else x)
    positiveFlow = moneyFlow * positiveNegative
    negativeFlow = moneyFlow * (1-positiveNegative)

    moneyFlowRatio = positiveFlow.rolling(
        periods).sum() / negativeFlow.rolling(periods).sum()

    return 100 - (100 / (1 + moneyFlowRatio))


def mom(price, periods=10):
    """Calculates the Momentum of the given price based on the number of periods specified.

    Args:
        price (Numerical List): List of prices (for example. closing)
        periods (int, optional): N-period momentum. Defaults to 10.

    Returns:
        pandas Series: N-period momentum of the price list
    """

    price = pd.Series(price)
    return price - price.shift(periods)


def ppo(price, short_periods=12, long_periods=26):
    """Calculates the Percentage Price Oscillator based on given price list and number of periods in the short and long term.

    Args:
        price (Numerical List): List of prices
        short_periods (int, optional): N periods to use for short term. Defaults to 12.
        long_periods (int, optional): N periods to use for long term. Defaults to 26.

    Returns:
        pandas Series: Percentage Price Oscillator series
    """

    price = pd.Series(price)
    short_ema = price.ewm(alpha=1/short_periods)
    long_ema = price.ewm(alpha=1/long_periods)

    return (short_ema - long_ema) / long_ema * 100


def roc(price):
    """Calculates Rate of Change given list of prices.

    Args:
        price (Numerical List): list of prices

    Returns:
        pandas Series: Rate of Change series
    """

    price = pd.Series(price)
    return (price / price.shift(1) - 1) * 100


def rsi(price, periods=14):
    """Calculates the Relative Strength Index based on the given price series and the number of periods to look back.

    Args:
        price (Numeric List): list of prices
        periods (int, optional): Number of periods to look back. Defaults to 14.

    Returns:
        pandas Series: RSI series
    """

    price = pd.Series(price)
    diff = price.diff()

    positiveNegative = diff.apply(
        lambda x: 1 if x > 0 else 0 if x < 0 else x)
    gain = diff * positiveNegative
    loss = diff.abs() * (1-positiveNegative)

    rs = gain.rolling(periods).mean() / loss.rolling(periods).mean()
    return 100 - (100 / (1+rs))


def stochf(high: pd.Series, low: pd.Series, close: pd.Series, period=14) -> pd.Series:
    """Calculates the Fast Stochastic Oscillator given an input of the highes, lows, and closes.

    Args:
        high (pd.Series): Series of highs
        low (pd.Series): Series of lows
        close (pd.Series): Series of closes
        period (int, optional): Number of periods to look back. Defaults to 14.

    Returns:
        pd.Series: Fast Stochasic Oscillator
    """

    highest = high.rolling(period).max()
    lowest = low.rolling(period).min()

    return (close - lowest) / (highest - lowest) * 100


def stoch(high: pd.Series, low: pd.Series, close: pd.Series,
          period=14, slow_window=3) -> pd.Series:
    """Calculates the Slow Stochastic Oscillator given an input of the highes, lows, and closes. The slow stochastic has a 3 period moving average applied to the stochastic indicator.

    Args:
        high (pd.Series): Series of highs
        low (pd.Series): Series of lows
        close (pd.Series): Series of closes
        period (int, optional): Number of periods to look back. Defaults to 14.
        slow_window (int, optional): The smoothing window on the stochastic indicator (1 -> fast stochastic). Defaults to 3.

    Returns:
        pd.Series: Slow Stochasic Oscillator
    """

    return stochf(high, low, close, period).rolling(3).mean()


def stoch_rsi(price: pd.Series, periods=14, rsi_periods=14) -> pd.Series:
    """Calculates the Stochastic RSI given an input of a price series.

    Args:
        price (pd.Series): Series of prices
        periods (int, optional): Number of periods to look back for the stochastic RSI indicator. Defaults to 14.
        rsi_periods (int, optional): Number of periods to look back for the underlying RSI indicator. Defaults to 14.

    Returns:
        pd.Series: Stochastic RSI indicator
    """

    rsi_indicator = rsi(price, rsi_periods)
    rsi_high = rsi_indicator.rolling(periods).max()
    rsi_low = rsi_indicator.rolling(periods).min()

    return (rsi_indicator - rsi_low) / (rsi_high - rsi_low)


def trix(price: pd.Series, first_periods=15, second_periods=15, third_periods=15) -> pd.Series:
    """Calculates the Triple Exponential Average give a price series.

    Args:
        price (pd.Series): Series of prices
        first_periods (int, optional): N Periods to use for first ema. Defaults to 15.
        second_periods (int, optional): N Periods to use for second ema. Defaults to 15.
        third_periods (int, optional): N Periods to use for third ema. Defaults to 15.

    Returns:
        pd.Series: _description_
    """

    triple = price.ewm(alpha=1/first_periods)\
        .ewm(alpha=1/second_periods)\
        .ewm(alpha=1/third_periods)

    diff = triple.diff()

    return (triple - diff) / diff


def ultosc(high: pd.Series, low: pd.Series, close: pd.Series,
           periods1=7, periods2=14, periods3=28) -> pd.Series:  # Ultimate Oscillator
    """Calculates the Ultimate Oscillator given series of high, low, and close.

    Args:
        high (pd.Series): Series of high
        low (pd.Series): Series of low
        close (pd.Series): Series of close
        periods1 (int, optional): N periods to look back for first average. Defaults to 7.
        periods2 (int, optional): N periods to look back for second average. Defaults to 14.
        periods3 (int, optional): N periods to look back for third average. Defaults to 28.

    Returns:
        pd.Series: Ultimate Oscillator series
    """

    minLowPC = np.r_[low.values, close.shift(1).values].T.min(axis=1)
    true_range = pd.Series(volatility.trange(high, low, close.shift(1)))

    buying_pressure = close - minLowPC

    avg1 = buying_pressure.rolling(
        periods1).sum() / true_range.rolling(periods1).sum()
    avg2 = buying_pressure.rolling(
        periods2).sum() / true_range.rolling(periods2).sum()
    avg3 = buying_pressure.rolling(
        periods3).sum() / true_range.rolling(periods3).sum()

    return (avg1*4 + avg2*2 + avg3) / 5 * 100


def willr(high: pd.Series, low: pd.Series, close: pd.Series, periods=14) -> pd.Series:  # Williams' %R
    """Calculates Williams' %R given series of highs, lows, and closes. 

    Args:
        high (pd.Series): Series of highs
        low (pd.Series): Series of lows
        close (pd.Series): Series of Closes
        periods (int, optional): Number of periods to look back. Defaults to 14.

    Returns:
        pd.Series: Williams' %R series
    """

    highestH = high.rolling(periods).max()
    lowestL = low.rolling(periods).min()

    return (highestH - close) / (highestH - lowestL)
