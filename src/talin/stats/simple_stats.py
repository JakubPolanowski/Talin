import numpy as np

# * Note the inclusion of average, median, var, and stdev is effectively redudant however they are included for the purpose of giving a consistent API


def average(list):
    """Returns average of a sequence of numbers. Note that this function just calls  the numpy.average function.

    Args:
        list (sequence of numbers): List of numbers to average (for instance price)

    Returns:
        Number: Average of list
    """

    return np.average(list)


def median(list):
    """Returns median of a sequence of numbers. Note that this function just calls the numpy.median function

    Args:
        list (sequence of numbers): List of numbers to get the median of (for instance price)

    Returns:
        Number: Median of List
    """

    return np.median(list)


def var(list, ddof=1):
    """Returns the variance of a sequence of numbers. Note that this function just calls the numpy.var function. By default returns the sample variance (ddof=1).

    Args:
        list (sequence of numbers): List of numbers to calculate the standard deviation of
        ddof (int, optional): Degrees of freedom. Defaults to 1.

    Returns:
        Number: Variance of input list
    """

    return np.var(list, ddof=ddof)


def std(list, ddof=1):
    """Returns the standard deviation of a sequence of numbers. Note that this function just calls the numpy.std function. By default returns the sample standard deviation (ddof=1).

    Args:
        list (sequence of numbers): List of numbers to calculate the standard deviation of
        ddof (int, optional): Degrees of freedom. Defaults to 1.

    Returns:
        Number: Standard deviation of input list
    """

    return np.std(list, ddof=ddof)


def typical_price(high, low, close):
    """Returns the typical price based on the period's (ex. day's) high, low, and close values. Multiple values can be calculated by passing in equal shaped numpy arrays for high, low, and close.

    Args:
        high (number or numpy array of numbers): High of period
        low (number or numpy array of numbers): Low of period
        close (number or numpy array of numbers): Close of period

    Returns:
        Number or numpy array: Typical Price of period/periods (if inputs were numpy arrays or requivalent)
    """

    return (high + low + close)/3


def weighted_close(high, low, close):
    """Returns the weighted close based on the period's (ex. day's) high, low, and close values. Multiple values can be calculated by passing in equal shaped numpy arrays for high, low, and close.

    Args:
        high (number or numpy array of numbers): High of period
        low (number or numpy array of numbers): Low of period
        close (number or numpy array of numbers): Close of period

    Returns:
        Number or numpy array: Weighted Close of period/periods (if inputs were numpy arrays or equivalent)
    """

    return (high + low + close*2)/4
