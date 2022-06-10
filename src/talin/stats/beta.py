import numpy as np
import pandas as pd

__all__ = ["beta"]


def beta(stock: pd.Series, market: pd.Series, pct_change=False) -> float:
    """Calculates beta for the given stock based on the market represenative (for instance $SPY). Note that the time periodicity and start date must be matching between stock and market inputs otherwise beta value will be meaningless. 

    Args:
        stock (pd.Series): 1D series of closing values for the stock. Can also be given as percent change of closing values if pct_change=True. 
        market (pd.Series): 1D series of closing values for the market reference, such as $SPY. Can also be given as percent change of closing values if pct_change=True. 
        pct_change (bool, optional): Specifies if BOTH series are composed of closing values (False) or percent change (True). Defaults to False.

    Raises:
        ValueError: Stock and Market arrays of mismatching size
        ValueError: Stock/Market series are multi dimensional (must be 1D)

    Returns:
        float: Beta

    Source: https://corporatefinanceinstitute.com/resources/knowledge/finance/beta-coefficient/
    """

    if stock.shape != market.shape:
        raise ValueError(
            f"stock series and market series must be of same shape, stock was {stock.shape}, market was {market.shape}")

    if stock.ndim != 1:
        raise ValueError(
            f"stock (and market) must be 1 dimensional, stock was {stock.ndim}")

    if not pct_change:
        stock = stock.pct_change().dropna()
        market = market.pct_change().dropna()

    cov_matrix = np.cov(stock, market)
    return cov_matrix[0][1] / cov_matrix[0][0]
