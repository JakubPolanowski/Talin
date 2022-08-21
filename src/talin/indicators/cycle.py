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

Indicator: Even Better SineWave (EBSW)

"""


def ebsw(close: pd.Series, duration: int = 40, ssfLength: int = 10) -> pd.Series:
    """Ehler's Even Better Sinewave. 

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
