from src.talin.stats import beta
import numpy as np
import pytest
import pandas as pd
from sklearn.linear_model import LinearRegression

PRECISION = 6

stock = pd.Series(np.arange(1, 101) * np.random.random(100))
market = pd.Series(np.arange(1, 101) * np.random.random(100))

stock_pct = ((stock - np.r_[np.NaN, stock[:-1]]) /
             np.r_[np.NaN, stock[:-1]])[1:]
market_pct = ((market - np.r_[np.NaN, market[:-1]]
               ) / np.r_[np.NaN, market[:-1]])[1:]

cov = np.cov(stock_pct, market_pct)
beta_cov = cov[0][1] / cov[0][0]
beta_cov = round(beta_cov, PRECISION)

lin = LinearRegression().fit(stock_pct.reshape((-1, 1)), market_pct)
beta_lin = lin.coef_[0]
beta_lin = round(beta_lin, PRECISION)


def test_beta_cov_approach():
    assert round(beta.beta(stock, market), PRECISION) == beta_cov


def test_beta_lin_approach():
    assert round(beta.beta(stock, market), PRECISION) == beta_lin


def test_beta_cov_pct_approach():
    assert round(beta.beta(stock_pct, market_pct, pct_change=True),
                 PRECISION) == beta_cov


def test_beta_lin_pct_approach():
    assert round(beta.beta(stock_pct, market_pct,
                 pct_change=True), PRECISION) == beta_lin


def test_beta_shape_mismatch():
    with pytest.raises(ValueError) as excinfo:
        beta.beta(stock[:-3], market)


def test_beta_ndim_invalid():
    with pytest.raises(ValueError) as excinfo:
        beta.beta(stock.reshape((-1, 1)), market.reshape((-1, 1)))
