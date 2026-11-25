"""Predictive regressions of Table II (Panel A) and Table IV (Panel A).

Table II, Panel A (equation (12) of the paper)
----------------------------------------------
    d_{t+1} = a + b * mu_t + c * dp_t + sum_{j=1..3} r_j * d_{t+1-j} + e_{t+1}

d is CRSP (top-down) dividend growth, mu the persistent component measured at
the end of period t, dp the log dividend-price ratio. Quarterly and annual.

Table IV, Panel A
-----------------
    y_{t+1} = a + b * mu_t (+ c * y_t) + e_{t+1}

y is quarterly growth of GDP or of nondurables-plus-services consumption. The
paper does not say whether growth is real or nominal (its consumption source,
NIPA Table 2.3.5, is in current dollars), so every regression is run both
ways: specs "gdp" and "consumption" are real, "gdp_nominal" and
"consumption_nominal" are nominal. A memo spec "inflation" regresses growth of
the GDP deflator on mu: if mu predicts nominal but not real growth, it should
predict inflation.

All t-statistics are Newey-West with NEWEY_WEST_LAGS lags.

Sample conventions (they reproduce the paper's observation counts)
-------------------------------------------------------------------
- Dividend growth is built from index data inside the sample only, so the
  first year has no growth rate. With 1973-2016 this gives 169 quarterly and
  40 annual observations, as in the paper.
- Macro growth uses the level one quarter before the sample starts, so the
  first quarter has a growth rate. With 1973-2015 this gives 171
  observations, as in the paper.

Each regression is run on the paper's sample ("paper") and on the sample
extended to END_DATE ("update"). mu is the robust filter's state
(STATE_FOR_REGRESSIONS, the filtered state), with parameters estimated on the
same sample. The filtered state at t uses observations up to t only, but the
parameters behind it are estimated on the whole sample, so these are
in-sample regressions; calc_extension_regressions.py has the out-of-sample
test. The update sample includes 2020, whose quarterly GDP and consumption
growth rates are far outside anything earlier in the sample.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

import calc_dividend_growth
import calc_persistent_component
import clean_macro
from settings import (
    END_DATE,
    N_DIVIDEND_GROWTH_LAGS,
    NEWEY_WEST_LAGS,
    PAPER_END_DATE,
    PAPER_MACRO_END_DATE,
    START_DATE,
    STATE_FOR_REGRESSIONS,
    config,
)

DATA_DIR = Path(config("DATA_DIR"))
RESULTS_FILE = "regression_results.parquet"

SAMPLE_ENDS = {
    "paper": {"dividend": PAPER_END_DATE, "macro": PAPER_MACRO_END_DATE},
    "update": {"dividend": END_DATE, "macro": END_DATE},
}
RESAMPLE_RULE = {"Q": "QE", "A": "YE"}
MACRO_SERIES = {
    "gdp": "real_gdp",
    "consumption": "real_consumption",
    "gdp_nominal": "nominal_gdp",
    "consumption_nominal": "nominal_consumption",
    "inflation": "gdp_deflator",
}


def period_end_value(series, freq):
    """Last available value of a daily series in each quarter or year."""
    return series.dropna().resample(RESAMPLE_RULE[freq]).last()


def hac_ols(y, X, lags=NEWEY_WEST_LAGS):
    """OLS with a constant and Newey-West standard errors."""
    data = pd.concat([y.rename("y"), X], axis=1).dropna()
    exog = sm.add_constant(data.drop(columns="y"))
    model = sm.OLS(data["y"], exog)
    return model.fit(cov_type="HAC", cov_kwds={"maxlags": lags})


def tidy_result(result, **labels):
    """Flatten a statsmodels result into one row per regressor."""
    rows = []
    for name in result.params.index:
        rows.append(
            {
                **labels,
                "variable": name,
                "coef": float(result.params[name]),
                "tstat": float(result.tvalues[name]),
                "r2": float(result.rsquared),
                "adj_r2": float(result.rsquared_adj),
                "nobs": int(result.nobs),
            }
        )
    return rows


def dividend_regression_data(mu, topdown, start_date, end_date, freq, extra=None):
    """Assemble the Table II data set at quarterly ("Q") or annual ("A") frequency.

    Returns (y, X): next-period dividend growth, and the regressors mu, log_dp
    and the lags of dividend growth. extra is an optional DataFrame of more
    period-end regressors.
    """
    low = calc_dividend_growth.topdown_low_frequency(
        topdown, start_date, end_date, freq
    ).set_index("date")
    mu = mu[(mu.index >= start_date) & (mu.index <= end_date)]
    X = pd.DataFrame({"mu": period_end_value(mu, freq)}).reindex(low.index)
    if extra is not None:
        X = X.join(extra.reindex(low.index))
    X["log_dp"] = low["log_dp"]
    for lag in range(N_DIVIDEND_GROWTH_LAGS):
        X[f"div_growth_lag{lag + 1}"] = low["div_growth"].shift(lag)
    y = low["div_growth"].shift(-1)
    return y, X


def macro_regression_data(
    mu, macro, series, start_date, end_date, with_lag, extra=None
):
    """Assemble the Table IV data set for one macro series.

    Growth is computed on the whole series before the sample is cut, so the
    first sample quarter has a growth rate.
    """
    level = macro.set_index("date")[MACRO_SERIES[series]]
    growth = np.log(level).diff()
    growth = growth[(growth.index >= start_date) & (growth.index <= end_date)]
    mu = mu[(mu.index >= start_date) & (mu.index <= end_date)]
    X = pd.DataFrame({"mu": period_end_value(mu, "Q")}).reindex(growth.index)
    if extra is not None:
        X = X.join(extra.reindex(growth.index))
    if with_lag:
        X["growth_lag1"] = growth
    y = growth.shift(-1)
    return y, X


def regression_state(states, sample, model="robust"):
    """The persistent component that enters the regressions for a sample."""
    return calc_persistent_component.select_state(
        states, model, sample, STATE_FOR_REGRESSIONS
    )


def run_replication_regressions(states, topdown, macro):
    """Run Table II Panel A and Table IV Panel A on both samples."""
    rows = []
    for sample, ends in SAMPLE_ENDS.items():
        mu = regression_state(states, sample)
        for freq, label in [("Q", "quarterly"), ("A", "annual")]:
            y, X = dividend_regression_data(
                mu, topdown, START_DATE, ends["dividend"], freq
            )
            rows += tidy_result(
                hac_ols(y, X), table="table2a", sample=sample, spec=label
            )
        for series in MACRO_SERIES:
            for with_lag in (False, True):
                y, X = macro_regression_data(
                    mu, macro, series, START_DATE, ends["macro"], with_lag
                )
                spec = f"{series}_{'with_lag' if with_lag else 'no_lag'}"
                rows += tidy_result(
                    hac_ols(y, X), table="table4a", sample=sample, spec=spec
                )
    return pd.DataFrame(rows)


def load_results(data_dir=DATA_DIR):
    """Load the tidy replication regression results."""
    return pd.read_parquet(Path(data_dir) / RESULTS_FILE)


if __name__ == "__main__":
    results = run_replication_regressions(
        calc_persistent_component.load_states(DATA_DIR),
        calc_dividend_growth.load_topdown_daily(DATA_DIR),
        clean_macro.load_macro(DATA_DIR),
    )
    results.to_parquet(DATA_DIR / RESULTS_FILE, index=False)
    print(
        results[results["variable"] == "mu"]
        .drop(columns="variable")
        .to_string(index=False)
    )
