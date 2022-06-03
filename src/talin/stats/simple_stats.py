import pandas as pd

__all__ = ["average", "median", "var",
           "std", "typical_price", "weighted_close"]

# * Note the inclusion of average, median, var, and stdev is effectively redudant however they are included for the purpose of giving a consistent API


def average(series: pd.Series) -> float:
    """Returns average of a series of numbers. Note that this function just calls  the pd.Series.mean function.

    Args:
        series (pd.Series): Series of numbers

    Returns:
        float: Average of series
    """

    return series.mean()


def median(series: pd.Series) -> float:
    """Returns median of a series. Note that this function just calls the pd.Series.median function

    Args:
        series (pd.Series): Series of numbers

    Returns:
        float: Median of List
    """

    return series.median()


def var(series: pd.Series, ddof=1) -> float:
    """Returns the variance of a series. Note that this function just calls the pd.Series.var function. By default returns the sample variance (ddof=1).

    Args:
        series (pd.Series): Series of numbers
        ddof (int, optional): Degrees of freedom. Defaults to 1.

    Returns:
        float: Variance of input list
    """

    return series.var(ddof=ddof)


def std(series: pd.Series, ddof=1) -> float:
    """Returns the standard deviation of a series. Note that this function just calls the pd.Series.std function. By default returns the sample standard deviation (ddof=1).

    Args:
        series (pd.Series): Series of numbers
        ddof (int, optional): Degrees of freedom. Defaults to 1.

    Returns:
        Number: Standard deviation of input list
    """

    return series.std(ddof=ddof)


def typical_price(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Returns the typical price based on the high, low, and close. 

    Args:
        high (pd.Series): Series of highs
        low (pd.Series): Series of lows
        close (pd.Series): Series of closes 

    Returns:
        pd.Series: Typical Price series
    """

    return (high + low + close)/3


def weighted_close(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Returns the weighted close on the high, low, and close.

    Args:
        high (pd.Series): Series of highs
        low (pd.Series): Series of lows
        close (pd.Series): Series of closes 

    Returns:
        pd.Series: Weighted Close series
    """

    return (high + low + close*2)/4
