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


def plus_dm(high: pd.Series) -> pd.Series:
    """Calculates the Positive Directional Movement

    Args:
        high (pd.Series): highs series

    Returns:
        pd.Series: positive directional movement

    Source: https://www.investopedia.com/terms/a/adx.asp
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

    Source: https://www.investopedia.com/terms/a/adx.asp
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

    Source: https://www.investopedia.com/terms/a/adx.asp
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

    Source: https://www.investopedia.com/terms/a/adx.asp
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

    Source: https://www.investopedia.com/terms/a/adx.asp
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


def adxr(adx: pd.Series, periods=2) -> pd.Series:
    """Calculates the Average Direction Movement Rating given the ADX and the number of periods to look backwards.

    Args:
        adx (pd.Series): array/series of Average Directional Movement Indexes
        periods (int): Number of periods to look backwards. Defaults to 2.

    Returns:
        pd.Series: The Average Direction Movement Rating

    Source: https://www.marketvolume.com/technicalanalysis/adxr.asp
    """

    return (adx - adx.shift(periods)) / 2


def apo(close: pd.Series, short_periods=14, long_periods=30) -> pd.Series:
    """Calculates Absolute Price Oscilliator given the a list of closing values and the number of periods for the short term and long term.

    Args:
        close (pd.Series): Series of closing values
        short_periods (int, optional): Number of periods for the short term. Defaults to 14.
        long_periods (int, optional): Number of periods for the long term. Defaults to 30.

    Returns:
        pd.Series: Absolute Price Oscillator series

    Source: https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/apo
    """

    return close.ewm(span=short_periods, adjust=False).mean() - \
        close.ewm(span=long_periods, adjust=False).mean()


def aroon(high: pd.Series, low: pd.Series, periods=25) -> tuple[pd.Series, pd.Series]:
    """Calculates the AROON indicator given the lists of highs and lows and the number of periods to use in the calculation.

    Args:
        high (pd.Series): Series of highs
        low (pd.Series): Series of lows
        periods (int, optional): Number of periods. Defaults to 25.

    Returns:
        (pd.Series, pd.Series): Returns the AROON up and AROON down series.

    Source: https://www.investopedia.com/terms/a/aroon.asp
    """

    up_dist = high.rolling(periods).apply(
        lambda x: (periods - 1) - x.argmax())
    down_dist = low.rolling(periods).apply(
        lambda x: (periods - 1) - x.argmin())

    aroon_up = (periods - up_dist) / periods * 100
    aroon_down = (periods - down_dist) / periods * 100

    return aroon_up, aroon_down


def aroon_osc(high: pd.Series, low: pd.Series, periods=25) -> pd.Series:
    """Calculates the AROON Oscillator given the lists of high and lows and the number of periods to use in the calculation.

    Args:
        high (pd.Series): Series of highs
        low (pd.Series): Series of lows
        periods (int, optional): Number of periods. Defaults to 25.
    Returns:
        pd.Series: AROON Oscillator series

    Source: https://school.stockcharts.com/doku.php?id=technical_indicators:aroon_oscillator
    """

    aroon_up, aroon_down = aroon(high, low, periods)
    return aroon_up - aroon_down


def bop(high: pd.Series, low: pd.Series, open: pd.Series, close: pd.Series) -> pd.Series:
    """Calculates the Balance of Power indicator, given the input arrays of high, low, open, and close.

    Args:
        high (pd.Series): Series of highs
        low (pd.Series): Series of lows
        open (pd.Series): Series of opens
        close (pd.Series): Series of closes

    Returns:
        pd.Series: Balance of Power series

    Source: https://school.stockcharts.com/doku.php?id=technical_indicators:balance_of_power
    """

    return (close - open) / (high - low)


def cci(high: pd.Series, low: pd.Series, close: pd.Series, periods=20) -> pd.Series:
    """Calculates the Commodity Channel Index, given the input arrays of high, low, open and close.

    Args:
        high (pd.Series): array of highs
        low (pd.Series): array of lows
        close (pd.Series): array of closes
        periods (int, optional): Number of periods for the moving average. Defaults to 20.

    Returns:
        pd.Series: Commodity Channel Index series.

    Source: https://www.investopedia.com/terms/c/commoditychannelindex.asp
    """

    typicalP = simple_stats.typical_price(high, low, close)
    mAvg = typicalP.rolling(periods).mean()
    meanDeviation = (typicalP - mAvg).abs().rolling(periods).mean()

    return (typicalP - mAvg) / (0.015 * meanDeviation)


def cmo(close: pd.Series, periods=9) -> pd.Series:
    """Calculates the Chande Momentum Oscillator.

    Args:
        close (pd.Series): Series of closes
        periods (int, optional): Number of periods to look back. Defaults to 9.

    Returns:
        pd.Series: Chande Momenum Oscillator series

    Source: https://www.investopedia.com/terms/c/chandemomentumoscillator.asp
    """

    diff = close.diff()

    gains = diff.mask(close < close.shift(1), 0)
    losses = diff.mask(close > close.shift(1), 0).abs()

    sH = gains.rolling(periods).sum()
    sL = losses.rolling(periods).sum()

    return (sH - sL) / (sH + sL) * 100


def macd(close: pd.Series, short_periods=12, long_periods=26) -> pd.Series:
    """Calculates the Moving Average Convergence Divergence.

    Args:
        close (pd.Series): Series of closes
        short_periods (int, optional): Number of periods for short term ema. Defaults to 12.
        long_periods (int, optional): Number of periods for long term ema. Defaults to 26.

    Returns:
        pd.Series: Moving Average Convergence Diveregence

    Source: https://www.investopedia.com/terms/m/macd.asp
    """

    return close.ewm(span=short_periods, adjust=False).mean() - \
        close.ewm(span=long_periods, adjust=False).mean()


def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, periods=14) -> pd.Series:
    """Calculates the Money Flow Index/

    Args:
        high (pd.Series): Series of highs
        low (pd.Series): Series of lows
        close (pd.Series): Series of closes
        volume (pd.Series): Series of volumes
        periods (int, optional): Number of periods to look back. Defaults to 14.

    Returns:
        pd.Series: Money Flow Index series

    Source: https://www.investopedia.com/terms/m/mfi.asp
    """

    typicalP = pd.Series(simple_stats.typical_price(high, low, close))
    rawMoney = typicalP * volume

    posFlow = rawMoney.mask(typicalP < typicalP.shift(1), 0)
    negFlow = rawMoney.mask(typicalP > typicalP.shift(1), 0)

    # first value is negative since cannot compare previous of first typ price
    posFlow.loc[posFlow.index[0]] = np.NaN
    negFlow.loc[negFlow.index[0]] = np.NaN

    moneyFlowRatio = posFlow.rolling(periods).sum() / \
        negFlow.rolling(periods).sum()

    return 100 - (100 / (1 + moneyFlowRatio))


def mom(price: pd.Series, periods=10) -> pd.Series:
    """Calculates the Momentum of the given price based on the number of periods specified.

    Args:
        price (pd.Series): List of prices (for example. closing)
        periods (int, optional): N-period momentum. Defaults to 10.

    Returns:
        pd.Series: N-period momentum of the price list

    Source: https://www.warriortrading.com/momentum-indicator/
    """

    return price - price.shift(periods)


def ppo(price: pd.Series, short_periods=12, long_periods=26) -> pd.Series:
    """Calculates the Percentage Price Oscillator based on given price series and number of periods in the short and long term.

    Args:
        price (pd.Series): Series of prices
        short_periods (int, optional): N periods to use for short term. Defaults to 12.
        long_periods (int, optional): N periods to use for long term. Defaults to 26.

    Returns:
        pandas Series: Percentage Price Oscillator series

    Source: https://www.investopedia.com/articles/investing/051214/use-percentage-price-oscillator-elegant-indicator-picking-stocks.asp
    """

    short_ema = price.ewm(span=short_periods, adjust=False).mean()
    long_ema = price.ewm(span=long_periods, adjust=False).mean()

    return (short_ema - long_ema) / long_ema


def roc(close: pd.Series, periods=12) -> pd.Series:
    """Calculates Rate of Change.

    Args:
        close (pd.Series): list of closes
        periods (int, optional): Number of periods to look back. This value depends on the strategy, for instance a long term trader could use periods=200 while a short term trader might use 12. Defaults to 12

    Returns:
        pd.Series: Rate of Change series

    Source: https://www.investopedia.com/terms/p/pricerateofchange.asp
    """

    return (close - close.shift(periods)) / close.shift(periods) * 100


def rsi(close: pd.Series, periods=14) -> pd.Series:
    """Calculates the Relative Strength Index.

    Args:
        close (pd.Series): series of closes
        periods (int, optional): Number of periods to look back. Defaults to 14.

    Returns:
        pd.Series: RSI series

    Source: https://www.investopedia.com/terms/r/rsi.asp
    """

    diff = close.diff()

    gains = diff.mask(close < close.shift(1), 0)
    losses = diff.mask(close > close.shift(1), 0).abs()

    # TODO figure out if a there is a better implementation for this
    # This is a WSM average which has no implemenation in pandas

    avgGain = [np.NaN] * (periods-1) + [np.mean(gains.values[:periods])]
    avgLoss = [np.NaN] * (periods-1) + [np.mean(losses.values[:periods])]

    for i in range(periods, diff.size):
        avgGain.append(
            (avgGain[i-1] * (periods-1)) + gains.values[i]
        )
        avgLoss.append(
            (avgLoss[i-1] * (periods-1)) + losses.values[i]
        )

    avgGain = pd.Series(avgGain)
    avgLoss = pd.Series(avgLoss)

    return 100 - (100 / (1 + (avgGain/avgLoss)))


def stochf(high: pd.Series, low: pd.Series, close: pd.Series, periods=14) -> pd.Series:
    """Calculates the Fast Stochastic Oscillator given an input of the highes, lows, and closes.

    Args:
        high (pd.Series): Series of highs
        low (pd.Series): Series of lows
        close (pd.Series): Series of closes
        periods (int, optional): Number of periods to look back. Defaults to 14.

    Returns:
        pd.Series: Fast Stochasic Oscillator

    Source: https://www.investopedia.com/ask/answers/05/062405.asp
    """

    highest = high.rolling(periods).max()
    lowest = low.rolling(periods).min()

    return (close - lowest) / (highest - lowest) * 100


def stoch(high: pd.Series, low: pd.Series, close: pd.Series,
          periods=14, slow=3) -> pd.Series:
    """Calculates the Slow Stochastic Oscillator given an input of the highes, lows, and closes. The slow stochastic has a 3 period moving average applied to the stochastic indicator.

    Args:
        high (pd.Series): Series of highs
        low (pd.Series): Series of lows
        close (pd.Series): Series of closes
        periods (int, optional): Number of periods to look back. Defaults to 14.
        slow (int, optional): The smoothing window on the stochastic indicator (1 -> fast stochastic). Defaults to 3.

    Returns:
        pd.Series: Slow Stochasic Oscillator

    Source: https://www.investopedia.com/ask/answers/05/062405.asp
    """

    return stochf(high, low, close, periods).rolling(slow).mean()


def stoch_rsi(price: pd.Series, periods=14, rsi_periods=14) -> pd.Series:
    """Calculates the Stochastic RSI given an input of a price series.

    Args:
        price (pd.Series): Series of prices
        periods (int, optional): Number of periods to look back for the stochastic RSI indicator. Defaults to 14.
        rsi_periods (int, optional): Number of periods to look back for the underlying RSI indicator. Defaults to 14.

    Returns:
        pd.Series: Stochastic RSI indicator

    Source: https://www.investopedia.com/terms/s/stochrsi.asp
    """

    rsi_indicator = rsi(price, rsi_periods)
    rsi_high = rsi_indicator.rolling(periods).max()
    rsi_low = rsi_indicator.rolling(periods).min()

    return (rsi_indicator - rsi_low) / (rsi_high - rsi_low)


def trix(price: pd.Series, periods=15) -> pd.Series:
    """Calculates the Triple Exponential Average give a price series.

    Args:
        price (pd.Series): Series of prices
        periods (int, optional): N Periods to use for the triple ema. Defaults to 15.

    Returns:
        pd.Series: Trix series

    Source: https://www.investopedia.com/terms/t/trix.asp#:~:text=The%20triple%20exponential%20average%20(TRIX)%20indicator%20is%20an%20oscillator%20used,are%20considered%20insignificant%20or%20unimportant
    """

    triple = price.ewm(span=periods, adjust=False).mean() \
        .ewm(span=periods, adjust=False).mean() \
        .ewm(span=periods, adjust=False).mean()

    diff = triple.diff()

    return (triple - triple.shift()) / triple.shift()


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

    Source: https://www.investopedia.com/terms/u/ultimateoscillator.asp
    """

    minLowPC = np.r_[[low.values], [close.shift(1).values]].T.min(axis=1)
    maxHighPC = np.r_[[high.values], [close.shift(1).values]].T.max(axis=1)

    tr = pd.Series(maxHighPC - minLowPC)

    buying_pressure = close - minLowPC

    avg1 = buying_pressure.rolling(periods1).sum() / \
        tr.rolling(periods1).sum()
    avg2 = buying_pressure.rolling(periods2).sum() / \
        tr.rolling(periods2).sum()
    avg3 = buying_pressure.rolling(periods3).sum() / \
        tr.rolling(periods3).sum()

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

    Source: https://www.investopedia.com/terms/w/williamsr.asp
    """

    highestH = high.rolling(periods).max()
    lowestL = low.rolling(periods).min()

    return (highestH - close) / (highestH - lowestL)
