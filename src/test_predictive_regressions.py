"""Unit tests for the predictive regressions (calc_predictive_regressions.py).

They check timing, which is where a predictive regression silently goes
wrong: the left-hand side must be next period's growth, every regressor must
be known at the end of the current period, and the lags must be the ones in
equation (12) of the paper.
"""

import numpy as np
import pandas as pd
import pytest

import calc_predictive_regressions as cpr
from settings import N_DIVIDEND_GROWTH_LAGS


@pytest.fixture
def quarters():
    return pd.date_range("2000-03-31", periods=12, freq="QE")


@pytest.fixture
def fake_topdown(monkeypatch, quarters):
    """Replace the top-down aggregation with a known quarterly table."""
    low = pd.DataFrame(
        {
            "date": quarters,
            "div_growth": np.arange(len(quarters), dtype=float),
            "log_dp": -3.0 - 0.01 * np.arange(len(quarters)),
        }
    )
    monkeypatch.setattr(
        cpr.calc_dividend_growth,
        "topdown_low_frequency",
        lambda topdown, start, end, freq: low,
    )
    return low


def test_period_end_value_takes_the_last_available_day():
    days = pd.bdate_range("2000-01-03", "2000-06-30")
    mu = pd.Series(np.arange(len(days), dtype=float), index=days)
    mu.iloc[-3:] = np.nan  # the last days of June have no observation
    out = cpr.period_end_value(mu, "Q")
    assert out.loc["2000-03-31"] == mu.loc["2000-03-31"]
    assert out.loc["2000-06-30"] == mu.iloc[-4]


def test_dividend_regression_uses_next_period_growth_and_current_regressors(
    fake_topdown, quarters
):
    days = pd.bdate_range(quarters[0] - pd.offsets.QuarterBegin(1), quarters[-1])
    mu = pd.Series(np.linspace(0.0, 1.0, len(days)), index=days)
    y, X = cpr.dividend_regression_data(mu, None, days[0], days[-1], "Q")

    growth = fake_topdown.set_index("date")["div_growth"]
    # Left-hand side: growth of the next quarter.
    assert y.loc[quarters[4]] == growth.loc[quarters[5]]
    assert np.isnan(y.loc[quarters[-1]])
    # Lags: d_t, d_{t-1}, d_{t-2} for the regression of d_{t+1}.
    assert N_DIVIDEND_GROWTH_LAGS == 3
    assert X.loc[quarters[4], "div_growth_lag1"] == growth.loc[quarters[4]]
    assert X.loc[quarters[4], "div_growth_lag3"] == growth.loc[quarters[2]]
    # mu is the value on the last day of the current quarter, never later.
    assert X.loc[quarters[4], "mu"] == mu[mu.index <= quarters[4]].iloc[-1]


def test_macro_growth_exists_in_the_first_sample_quarter(quarters):
    """Growth uses the level one quarter before the sample, as in the paper."""
    macro = pd.DataFrame(
        {
            "date": quarters,
            **{col: np.exp(0.01 * np.arange(12)) for col in cpr.MACRO_SERIES.values()},
        }
    )
    days = pd.bdate_range(quarters[1] - pd.offsets.QuarterBegin(1), quarters[-1])
    mu = pd.Series(1.0, index=days)
    y, X = cpr.macro_regression_data(
        mu, macro, "gdp", quarters[1] - pd.offsets.QuarterBegin(1), quarters[-1], True
    )
    assert X.index[0] == quarters[1]
    assert X.loc[quarters[1], "growth_lag1"] == pytest.approx(0.01)
    assert y.loc[quarters[1]] == pytest.approx(0.01)


def test_hac_ols_recovers_a_known_slope():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(size=400), name="mu")
    y = 0.5 + 2.0 * x + rng.normal(scale=0.1, size=400)
    result = cpr.hac_ols(pd.Series(y), x.to_frame())
    assert result.params["mu"] == pytest.approx(2.0, abs=0.05)
    assert result.nobs == 400
