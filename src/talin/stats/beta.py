import numpy as np


def beta(stock, market, pct_change=False):
    """Calculates beta for the given stock based on the market represenative (for instance $SPY). Note that the time periodicity and start date must be matching between stock and market inputs otherwise beta value will be meaningless. 

    Args:
        stock (array of numbers): 1D array of closing values for the stock. Can also be given as percent change of closing values if pct_change=True. 
        market (array of numbers): 1D array of closing values for the market reference, such as $SPY. Can also be given as percent change of closing values if pct_change=True. 
        pct_change (bool, optional): Specifies if arrays are composed of closing values (False) or percent change (True). Defaults to False.

    Raises:
        ValueError: Stock and Market arrays of mismatching size
        ValueError: Stock/Market arrays are multi dimensional (must be 1D)

    Returns:
        Number: Beta
    """

    if not isinstance(stock, np.ndarray):
        stock = np.array(stock)

    if not isinstance(market, np.ndarray):
        market = np.array(market)

    if stock.shape != market.shape:
        raise ValueError(
            f"stock series and market series must be of same shape, stock was {stock.shape}, market was {market.shape}")

    if stock.ndim != 1:
        raise ValueError(
            f"stock (and market) must be 1 dimensional, stock was {stock.ndim}")

    if not pct_change:
        stock_shifted = np.r_[np.NaN, stock[:-1]]
        stock = ((stock - stock_shifted) / stock_shifted)[1:]  # to drop nan

        market_shift = np.r_[np.NaN, market[:-1]]
        market = ((market - market_shift) / market_shift)[1:]

    cov_matrix = np.cov(stock, market)
    return cov_matrix[0][1] / cov_matrix[0][0]
