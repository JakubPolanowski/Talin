import pandas as pd
import numpy as np

__all__ = [
    "ebsw"
]

"""

TODO IMPLEMENT

HT_DCPERIOD          Hilbert Transform - Dominant Cycle Period
HT_DCPHASE           Hilbert Transform - Dominant Cycle Phase
HT_PHASOR            Hilbert Transform - Phasor Components
HT_SINE              Hilbert Transform - SineWave
HT_TRENDMODE         Hilbert Transform - Trend vs Cycle Mode
"""


def ebsw(close: pd.Series, duration: int = 40, ssfLength: int = 10) -> pd.Series:
    """Ehler's Even Better Sinewave. 

    This is based on John Ehlers' Cycle Analytics For Traders pgs 161-162

    Args:
        close (pd.Series): A series of closes
        duration (int, optional): The approximate length of a trade position in a continuing trend. Defaults to 40.
        ssfLength (int, optional): The critical period in the super smoother filter. Defaults to 10.

    Returns:
        pd.Series: The series of EBSW indicator
    """

    # vars
    alpha1 = (1 - np.sin(2 * np.pi / duration)) / \
        np.cos(2 * np.pi / duration)

    # Smooth with a Super Smoother Filter from equation 3-3
    a1 = np.exp(-1.414 * np.pi / ssfLength)
    b1 = 2 * a1 * np.cos(1.414 * np.pi / ssfLength)

    c2 = b1
    c3 = -a1 * a1
    c1 = 1 - c2 - c3

    lastClose = hp = lastHP = 0
    filterHist = [0, 0]

    wave_series = []

    for i in range(duration, close.size):

        hp = 0.5 * (1 + alpha1) * \
            (close.values[i] - lastClose) + alpha1 * lastHP

        filt = c1 * (hp + lastHP) / 2 + c2 * \
            filterHist[-1] + c3 * filterHist[-2]

        # 3 Bar average of Wave amplitude and power
        wave = (filt + filterHist[-1] + filterHist[-2]) / 3
        pwr = (filt**2 + filt[-1]**2 + filt[-2]**2) / 3

        # Normalize the Average Wave to Square Root of the Average Power
        wave_series.append(wave / np.sqrt(pwr))

        lastHP = hp
        filterHist.append(filt)
        lastClose = close.values[i]

    return pd.Series(wave_series)


def ht_indicator(close: pd.Series, lp_period: int = 20) -> pd.Series:
    """Calculates the Hilbert Transformation Indicator

    Args:
        close (pd.Series): Series of closes
        lp_period (int, optional): Low Pass Period (TODO confirm if description is correct). Defaults to 20.

    Returns:
        pd.Series: The Hilbert Transformation Indicator Series
    """

    # This is based on John Ehlers' Cycle Analytics For Traders pgs 184-185

    alpha1 = (np.cos(.707 * np.pi*2 / 48) + np.sin(.707 *
              np.pi*2 / 48) - 1) / np.cos(.707 * np.pi*2 / 48)

    a1 = np.exp(-1.414*np.pi / lp_period)
    b1 = 2 * a1 * np.cos(1.414 * np.pi / lp_period)
    c2 = b1
    c3 = -a1 * a1
    c1 = 1 - c2 - c3

    a1H = np.exp(-1.414*np.pi / (lp_period / 2))
    b1H = 2 * a1H * np.cos(1.414 * np.pi / (lp_period / 2))
    c2H = b1H
    c3H = -a1H * a1H
    c1H = 1 - c2H - c3H

    filtHist = [0, 0]
    hpHist = [0, 0]
    lastIPeak = lastReal = lastQPeak = lastQuadrature = 0

    result = [0] * 2

    for i in range(2, close.size+1):

        hp = (1 - alpha1 / 2) * (1 - alpha1 / 2) * \
             (close.values[i] - 2*close.values[i-1] + close.values[i-2]) + \
            2 * (1 - alpha1) * hpHist[-1] - (1 - alpha1) * \
            (1 - alpha1) * hpHist[-2]

        filt = c1 * (hp + hpHist[-1]) / 2 + c2 * \
            filtHist[-1] + c3 * filtHist[-2]

        iPeak = .991 * lastIPeak
        if abs(filt) > iPeak:
            iPeak = abs(filt)

        real = filt / iPeak
        quadrature = (real - lastReal)
        qPeak = .991 * lastQPeak

        if abs(quadrature) > qPeak:
            qPeak = abs(quadrature)

        quadrature /= qPeak

        result.append(
            c1H * (quadrature + lastQuadrature) / 2 +
            c2H * result[-1] + c3H * result[-2]
        )

        filtHist.append(filt)
        hpHist.append(hp)
        lastIPeak = iPeak
        lastReal = real
        lastQPeak = qPeak
        lastQuadrature = quadrature

    return result
