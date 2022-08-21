import pandas as pd
import numpy as np

from src.talin.indicators import cycle

size = 200

close = pd.Series(np.random.rand(size) + 1)
low = pd.Series(close - np.random.rand())
high = pd.Series(close + np.random.rand())
vol = pd.Series(np.random.rand(size) + 100)


def test_ebsw():
    """Test implementation based on https://www.tradingview.com/script/a1AeExr4-Ehlers-Even-Better-Sinewave-Indicator-CC/

    This itself is based on John Ehlers' Cycle Analytics For Traders pgs 161-162

    TODO confirm test case matches implementation within Cycle Analytics for Traders
    """

    # TODO

    # for duration in range(1, 50):  # input duration default = 40
    #     for ssfLength in range(1, 16):  # super smooth filter length default = 10

    #         alpha1 = (1 - np.sin(2 * np.pi / duration)) / \
    #             np.cos(2 * np.pi / duration)

    #         a1 = np.exp(-1.414 * np.pi / ssfLength)
    #         b1 = 2 * a1 * np.cos(1.414 * np.pi / ssfLength)

    #         c2 = b1
    #         c3 = -a1 * a1
    #         c1 = 1 - c2 - c3

    #         lastClose, lastHP, lastFilter, beforeLastFilter = 0, 0, 0, 0
    #         hp = [np.NaN] * (duration - 1)

    #         for i in range(duration, close.size):

    #             hp.append(
    #                 0.5 * (1 + alpha1) *
    #                 (close.values[i] - lastClose) + alpha1 * lastHP
    #             )
    #             filt = c1 * (hp[i] + lastHP) / 2 + c2 * \
    #                 lastFilter + c3*beforeLastFilter
