import enum
import numpy as np
import pandas as pd
from src.talin.indicators import momentum
# note testing handled by test_volatility
from src.talin.indicators import volatility
# note testing handled by test_simple_stats
from src.talin.stats import simple_stats

size = 100
openPrice = pd.Series(np.random.rand(size) + 1)
close = pd.Series(np.random.rand(size) + 1)
low = pd.Series(close - np.random.rand())
high = pd.Series(close + np.random.rand())
vol = pd.Series(np.random.rand(size) + 100)


def test_plus_dm():
    """+Direction Movement

    +DM = Current High - Previous High

    source: https://www.investopedia.com/terms/a/adx.asp
    """

    pDM = high - high.shift(1)
    assert all(pDM.dropna() == momentum.plus_dm(high).dropna())


def test_minus_dm():
    """-Direction Movement

    -DM = Previous Low - Current Low

    source: https://www.investopedia.com/terms/a/adx.asp
    """
    nDM = low.shift(1) - low
    assert all(nDM.dropna() == momentum.minus_dm(low).dropna())


def test_di_minus():
    """-Direction Indicator

             (Smoothed -DM)
    -DI =  ------------------  * 100
          (Average True Range)

    Smoothing window same between Average True Range (ATR) and Smoothed -DM,
    typically 14 days

    source: https://www.investopedia.com/terms/a/adx.asp
    """

    nDM = momentum.minus_dm(low)

    for i in range(1, 21):
        atr = volatility.atr(high, low, close, periods=i)
        nDI = nDM.rolling(i).mean() / atr * 100
        assert all(nDI.dropna() == momentum.di(nDM, atr, periods=i).dropna())


def test_di_plus():
    """+Direction Indicator

             (Smoothed +DM)
    +DI =  ------------------  * 100
          (Average True Range)

    Smoothing window same between Average True Range (ATR) and Smoothed +DM,
    typically 14 days

    source: https://www.investopedia.com/terms/a/adx.asp
    """

    pDM = momentum.plus_dm(high)

    for i in range(1, 21):
        atr = volatility.atr(high, low, close, periods=i)
        pDI = pDM.rolling(i).mean() / atr * 100
        assert all(pDI.dropna() == momentum.di(pDM, atr, periods=i).dropna())


def test_dx():
    """Directional Index

          (|+DI - -DI|)
    DX =   -----------  * 100
          (|+DI + -DI|)

    source: https://www.investopedia.com/terms/a/adx.asp
    """

    pDM = momentum.plus_dm(high)
    nDM = momentum.minus_dm(low)
    atr = volatility.atr(high, low, close, periods=14)

    pDI = momentum.di(pDM, atr, periods=14)
    nDI = momentum.di(nDM, atr, periods=14)

    dx = ((pDI - nDI).abs() / (pDI + nDI).abs()) * 100
    assert all(dx.dropna() == momentum.dx(pDI, nDI).dropna())


def test_adx():
    """Average Directional Movement Index

    ADX = N-period Simple Moving Average of DX

    source: https://www.investopedia.com/terms/a/adx.asp
    """

    pDM = momentum.plus_dm(high)
    nDM = momentum.minus_dm(low)
    atr = volatility.atr(high, low, close, periods=14)

    pDI = momentum.di(pDM, atr, periods=14)
    nDI = momentum.di(nDM, atr, periods=14)

    dx = momentum.dx(pDI, nDI)

    for i in range(1, 21):
        adx = dx.rolling(i).mean()
        assert all(adx.dropna() == momentum.adx(
            high, low, close, periods=i).dropna())


def test_adxr():
    """Average Directional Movement Rating

    ADXR = (ADX - ADX N Periods Ago) / 2

    Typically 14 period lookback

    Source: https://www.marketvolume.com/technicalanalysis/adxr.asp
    """

    adx = momentum.adx(high, low, close, periods=14)

    for i in range(1, 21):
        adxr = (adx - adx.shift(i)) / 2
        assert all(adxr.dropna() == momentum.adxr(adx, periods=i).dropna())


def test_apo():
    """Absolute Price Oscillator

    APO = Shorter Period EMA - Longer Period EMA

    Source: https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/apo
    """

    for i in range(1, 21):
        apo = close.ewm(span=i, adjust=False).mean() - \
            close.ewm(span=i+5, adjust=False).mean()

        assert all(apo.dropna() == momentum.apo(
            close, short_periods=i, long_periods=i+5).dropna())


def test_aroon():
    """Aroon Indicator

                (N Periods - Periods Since N Period period High)
    Aroon Up =   ----------------------------------------------  * 100
                                  N Periods

                  (N Periods - Periods Since N Period period Low)
    Aroon Down =   ---------------------------------------------  * 100
                                    N Periods

    Typically 25

    #:~:text=The%20Aroon%20indicator%20is%20a,lows%20over%20a%20time%20period.
    Source: https://www.investopedia.com/terms/a/aroon.asp

    https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/aroon-indicator

    Note: test implementation is designed for simplicity not efficiency
    """

    n_periods = [3, 15, 25]

    for n_period in n_periods:
        up_count = []
        down_count = []

        for i in range(size):
            if i < n_period - 1:
                up_count.append(np.NaN)
                down_count.append(np.NaN)
            else:
                high_slice = high.value[i-n_period:i]
                low_slice = low.value[i-n_period:i]

                up_count.append(np.argmax(high_slice) + i)
                down_count.append(np.argmin(low_slice) + i)

        aroon_up = (n_period - pd.Series(up_count)) / n_period * 100
        aroon_down = (n_period - pd.Series(down_count)) / n_period * 100

        implement_a_up, implemented_a_down = momentum.aroon(
            high, low, periods=n_period)

        assert all(aroon_down.dropna() == implemented_a_down.dropna())
        assert all(aroon_up.dropna() == implement_a_up.dropna())


def test_aroon_osc():
    """Aroon Oscillator

    AROON OSC = AROON Up - AROON Down

    source: https://school.stockcharts.com/doku.php?id=technical_indicators:aroon_oscillator
    """

    aroon_up, aroon_down = momentum.aroon(high, low, periods=25)

    aroon_osc = aroon_up - aroon_down

    assert all(aroon_osc.dropna() == momentum.aroon_osc(
        high, low, periods=25).dropna())


def test_bop():
    """Balance of Power

    BOP = (Close - Open) / (High - Low)

    Source: https://school.stockcharts.com/doku.php?id=technical_indicators:balance_of_power
    """

    bop = (close - openPrice) / (high - low)

    assert all(bop == momentum.bop(high, low, openPrice, close))


def test_cci():
    """Commodity Channel Index

           Typical Price - Simple Moving Average of Typical Price
    CCI =  -------------------------------------
                 0.015 * Mean Deviation

    Mean Deviation = SMA(|Typical Price - SMA Typical Price|)

    Typically Lookback periods = 20

    source: https://www.investopedia.com/terms/c/commoditychannelindex.asp
    """

    typical = simple_stats.typical_price(high, low, close)

    for i in range(1, 21):
        sma = typical.rolling(i).mean()
        mean_dev = (typical - sma).rolling(i).mean()

        cci = (typical - sma) / (0.015 * mean_dev)

        assert all(cci.dropna() == momentum.cci(
            high, low, close, periods=i).dropna())


def test_cmo():
    """Chande Momentum Oscillator

          sH - sL
    CMO = ------- * 100
          sH + sL

    where

    sH = sum of higher closes (gains delta) over N periods
    sL = sum of lower closes (losses delta) over N Periods

    source: https://www.investopedia.com/terms/c/chandemomentumoscillator.asp

    https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/cmo#:~:text=The%20CMO%20indicator%20is%20created,%2D100%20to%20%2B100%20range.
    """

    diff = close.diff()

    gains = diff.mask(close < close.shift(1), 0)
    loss = diff.mask(close > close.shift(1), 0).abs()

    for i in range(1, 21):
        sH = gains.rolling(i).sum()
        sL = loss.rolling(i).sum()

        cmo = (sH - sL) / (sH + sL) / 100

        assert all(cmo.dropna() == momentum.cmo(close, periods=i).dropna())


def test_macd():
    """MACD

    MACD = 12-Period (Short) EMA - 26 Period (Long) EMA

    source: https://www.investopedia.com/terms/m/macd.asp
    """

    shorts = [6, 12, 18]
    longs = [13, 26, 52]

    for short, long in zip(shorts, longs):
        macd = close.ewm(span=short, adjust=False).mean() - \
            close.ewm(span=long, adjust=False).mean()

        assert all(macd.dropna() == momentum.macd(
            close, short_periods=short, long_periods=long).dropna())


def test_mfi():
    """Money Flow Index

                        100
    MFI = 100 - --------------------
                1 + Money Flow Ratio

          N Period sum of Positive Money Flow
    MFR = -----------------------------------
          N Period sum of Negative Money Flow

    Raw Money Flow = Typical Price * Volume

    Money Flow is positive if typical price > previous typical price and vice versa

    source: https://www.investopedia.com/terms/m/mfi.asp
    """

    typical = simple_stats.typical_price(high, low, close)
    raw_money_flow = typical * vol

    pos_mf = raw_money_flow.mask(typical < typical.shift(1), 0)
    neg_mf = raw_money_flow.mask(typical > typical.shift(1), 0)

    # first value is negative since cannot compare previous of first typ price
    pos_mf.loc[pos_mf.index[0]] = np.NaN
    neg_mf.loc[pos_mf.index[0]] = np.NaN

    for i in range(1, 21):
        mfr = pos_mf.rolling(i).sum() / neg_mf.rolling(i).sum()
        mfi = 100 - (100 / (1 + mfr))

        assert all(mfi.dropna() - momentum.mfi(high,
                   low, close, vol, periods=i).dropna())


def test_mom():
    pass


def test_ppo():
    pass


def test_roc():
    pass


def test_rocp():
    pass


def test_rocr():
    pass


"""
TODO Implement

"macd", "mfi", "mom", "ppo", "roc", "rocp", "rocr",
"rocr100", "rsi", "stochf", "stoch", "stoch_rsi",
"trix", "ultosc", "willr",
"""
