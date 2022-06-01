from src.talin.indicators import volume
import numpy as np
import pandas as pd

# TODO test ad and adosc


def test_obv():
    """OBV is a simple piece-wise cumulative summation function

    source: https://www.investopedia.com/terms/o/onbalancevolume.asp

                     { volume,  if close > close_prev
    OBV = OBV_prev + { 0,       if close == close_prev
                     { -volume, if close < close_prev

    with the first OBV_prev value set to be 0, which should subsequently be set to NaN
    since first close will not have a previous close.
    """
    # Note that below is not written efficently, written for max simplicity
    # to test if the actual implementation matches specification

    size = 10
    vol = np.random.rand(size)
    close = np.random.rand(size)

    obv = [0]
    for i in range(1, size):

        if close[i] > close[i-1]:
            obv.append(obv[i-1] + vol[i])
        elif close[i] < close[i-1]:
            obv.append(obv[i-1] - vol[i])
        else:
            obv.append(obv[i-1] + 0)

    obv[0] = np.NaN

    implemented_obv = volume.obv(pd.Series(close), pd.Series(vol))

    assert np.isnan(implemented_obv[0])  # NaN != NaN but NaN is NaN
    assert all(obv[1:] == implemented_obv[1:])
