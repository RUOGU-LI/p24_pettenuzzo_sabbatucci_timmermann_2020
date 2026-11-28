"""Extension, part 3: do omissions add predictive power?

The question
------------
The paper's persistent component mu ignores firms that stop paying. We test
whether CRSP-inferred omissions carry information about future dividend, GDP
and consumption growth beyond mu, over the full history since 1973.

Specifications (same left-hand sides and controls as Tables II and IV)
-----------------------------------------------------------------------
baseline                     mu                      + controls
plus_omissions               mu + omission share     + controls
plus_omissions_initiations   mu + omission share + initiation share + controls
mu_ext                       mu_ext                  + controls

mu_ext is the persistent component of growth that includes omitted dividends
with a zero numerator (calc_extensive_growth.py, real-time dating). Controls
are the dividend-price ratio and three lags of dividend growth for the
dividend regression, and the lagged dependent variable for GDP and
consumption. All regressors are known at the end of quarter t.

Out-of-sample test
------------------
Everything is frozen at PAPER_END_DATE: the state-space parameters and the
regression coefficients. The frozen models then forecast every quarter from
OOS_START_DATE on. We report the root mean squared forecast error of each
specification relative to the baseline.

Success criteria, fixed before running anything (see README, Extension)
------------------------------------------------------------------------
1. In the quarterly dividend-growth regression on the full sample, the
   omission share has a negative coefficient with |t| >= SIGNIFICANCE_T.
2. Adding the omission share raises the adjusted R-squared of that regression.
3. Out of sample, the specification with the omission share has a lower RMSE
   than the baseline (ratio below one) for dividend growth.

If a criterion fails we report it as failed. A clean "omissions do not add
information once mu is known" is a finding, not a problem to tune away.

The criteria were not changed after any run. The omission rule was: its late
window was widened from 45 to 100 days after a first run on the full CRSP
sample, because validation showed firms that had only moved their board
meetings (see settings.py and the README). The out-of-sample test freezes the
regression coefficients and the state-space parameters, but the rule itself
was chosen with the whole sample in view.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

import calc_dividend_growth
import calc_extensive_growth
import calc_omissions
import calc_persistent_component
import clean_macro
from calc_predictive_regressions import (
    SAMPLE_ENDS,
    dividend_regression_data,
    hac_ols,
    macro_regression_data,
    regression_state,
    tidy_result,
)
from settings import (
    END_DATE,
    INITIATION_INDEX_COLUMN,
    OMISSION_INDEX_COLUMN,
    OOS_START_DATE,
    PAPER_END_DATE,
    REPLICATION_MACRO_GROWTH,
    SIGNIFICANCE_T,
    START_DATE,
    STATE_FOR_REGRESSIONS,
    config,
)

DATA_DIR = Path(config("DATA_DIR"))
RESULTS_FILE = "extension_regression_results.parquet"
OOS_FILE = "extension_oos_results.parquet"
SCORECARD_FILE = "extension_scorecard.parquet"

SPECS = {
    "baseline": {"state": "mu", "extra": []},
    "plus_omissions": {"state": "mu", "extra": ["omission_share"]},
    "plus_omissions_initiations": {
        "state": "mu",
        "extra": ["omission_share", "initiation_share"],
    },
    "mu_ext": {"state": "mu_ext", "extra": []},
}
DEPENDENT_VARIABLES = ["dividend", "gdp", "consumption"]
## GDP and consumption growth are measured as in the replication of Table IV.
MACRO_SPEC = {
    "gdp": "gdp" if REPLICATION_MACRO_GROWTH == "real" else "gdp_nominal",
    "consumption": "consumption"
    if REPLICATION_MACRO_GROWTH == "real"
    else "consumption_nominal",
}


def extensive_margin_regressors(index):
    """Quarter-end omission and initiation shares used as regressors."""
    idx = index.set_index("date")
    return pd.DataFrame(
        {
            "omission_share": idx[OMISSION_INDEX_COLUMN],
            "initiation_share": idx[INITIATION_INDEX_COLUMN],
        }
    )


def extensive_state(states_ext, sample, dating="realtime"):
    """Persistent component of growth including omissions."""
    sub = states_ext[
        (states_ext["dating"] == dating) & (states_ext["model"] == "robust")
    ]
    return calc_persistent_component.select_state(
        sub, "robust", sample, STATE_FOR_REGRESSIONS
    )


def build_dataset(depvar, spec, states, topdown, macro, shares, start_date, end_date):
    """Return (y, X) for one dependent variable and specification."""
    extra = shares[SPECS[spec]["extra"]] if SPECS[spec]["extra"] else None
    mu = states[SPECS[spec]["state"]]
    if depvar == "dividend":
        return dividend_regression_data(mu, topdown, start_date, end_date, "Q", extra)
    return macro_regression_data(
        mu, macro, MACRO_SPEC[depvar], start_date, end_date, with_lag=True, extra=extra
    )


def run_in_sample(states_by_sample, topdown, macro, shares):
    """Every specification, dependent variable and sample, in sample."""
    rows = []
    for sample, ends in SAMPLE_ENDS.items():
        for depvar in DEPENDENT_VARIABLES:
            end_date = ends["dividend"] if depvar == "dividend" else ends["macro"]
            for spec in SPECS:
                y, X = build_dataset(
                    depvar,
                    spec,
                    states_by_sample[sample],
                    topdown,
                    macro,
                    shares,
                    START_DATE,
                    end_date,
                )
                rows += tidy_result(
                    hac_ols(y, X), sample=sample, depvar=depvar, spec=spec
                )
    return pd.DataFrame(rows)


def run_out_of_sample(frozen_states, topdown, macro, shares):
    """Freeze coefficients at PAPER_END_DATE and forecast from OOS_START_DATE."""
    rows = []
    for depvar in DEPENDENT_VARIABLES:
        errors = {}
        for spec in SPECS:
            y, X = build_dataset(
                depvar,
                spec,
                frozen_states,
                topdown,
                macro,
                shares,
                START_DATE,
                END_DATE,
            )
            data = pd.concat([y.rename("y"), X], axis=1).dropna()
            # y is next quarter's growth: train on rows whose target is in sample.
            target_date = data.index + pd.offsets.QuarterEnd(1)
            train = data[target_date <= pd.Timestamp(PAPER_END_DATE)]
            test = data[target_date >= pd.Timestamp(OOS_START_DATE)]
            fit = sm.OLS(train["y"], sm.add_constant(train.drop(columns="y"))).fit()
            exog = sm.add_constant(test.drop(columns="y"), has_constant="add")
            errors[spec] = test["y"] - fit.predict(exog)
        base_rmse = float(np.sqrt((errors["baseline"] ** 2).mean()))
        for spec, err in errors.items():
            rmse = float(np.sqrt((err**2).mean()))
            rows.append(
                {
                    "depvar": depvar,
                    "spec": spec,
                    "n_forecasts": int(err.notna().sum()),
                    "rmse": rmse,
                    "rmse_ratio_to_baseline": rmse / base_rmse,
                }
            )
    return pd.DataFrame(rows)


def build_scorecard(in_sample, out_of_sample):
    """Evaluate the three pre-registered success criteria."""

    def pick(frame, **conditions):
        mask = np.ones(len(frame), dtype=bool)
        for key, value in conditions.items():
            mask &= (frame[key] == value).to_numpy()
        return frame[mask].iloc[0]

    omission = pick(
        in_sample,
        sample="update",
        depvar="dividend",
        spec="plus_omissions",
        variable="omission_share",
    )
    base = pick(
        in_sample, sample="update", depvar="dividend", spec="baseline", variable="mu"
    )
    oos = pick(out_of_sample, depvar="dividend", spec="plus_omissions")
    rows = [
        {
            "criterion": "Omission share: negative and significant",
            "value": omission["tstat"],
            "threshold": -SIGNIFICANCE_T,
            "passed": bool(
                omission["coef"] < 0 and omission["tstat"] <= -SIGNIFICANCE_T
            ),
        },
        {
            "criterion": "Adjusted R-squared rises when omissions are added",
            "value": omission["adj_r2"] - base["adj_r2"],
            "threshold": 0.0,
            "passed": bool(omission["adj_r2"] > base["adj_r2"]),
        },
        {
            "criterion": "Out-of-sample RMSE ratio below one",
            "value": oos["rmse_ratio_to_baseline"],
            "threshold": 1.0,
            "passed": bool(oos["rmse_ratio_to_baseline"] < 1.0),
        },
    ]
    return pd.DataFrame(rows)


def load_results(data_dir=DATA_DIR):
    """Load in-sample extension regression results."""
    return pd.read_parquet(Path(data_dir) / RESULTS_FILE)


def load_oos(data_dir=DATA_DIR):
    """Load out-of-sample extension results."""
    return pd.read_parquet(Path(data_dir) / OOS_FILE)


def load_scorecard(data_dir=DATA_DIR):
    """Load the extension scorecard."""
    return pd.read_parquet(Path(data_dir) / SCORECARD_FILE)


if __name__ == "__main__":
    states = calc_persistent_component.load_states(DATA_DIR)
    states_ext = calc_extensive_growth.load_states_extensive(DATA_DIR)
    topdown = calc_dividend_growth.load_topdown_daily(DATA_DIR)
    macro = clean_macro.load_macro(DATA_DIR)
    shares = extensive_margin_regressors(calc_omissions.load_quarterly_index(DATA_DIR))

    states_by_sample = {
        sample: {
            "mu": regression_state(states, sample),
            "mu_ext": extensive_state(states_ext, sample),
        }
        for sample in SAMPLE_ENDS
    }
    in_sample = run_in_sample(states_by_sample, topdown, macro, shares)
    # "paper" states are filtered through END_DATE with parameters frozen at
    # PAPER_END_DATE, which is exactly what the out-of-sample test needs.
    out_of_sample = run_out_of_sample(states_by_sample["paper"], topdown, macro, shares)
    scorecard = build_scorecard(in_sample, out_of_sample)

    in_sample.to_parquet(DATA_DIR / RESULTS_FILE, index=False)
    out_of_sample.to_parquet(DATA_DIR / OOS_FILE, index=False)
    scorecard.to_parquet(DATA_DIR / SCORECARD_FILE, index=False)
    print(scorecard.to_string(index=False))
    print(out_of_sample.to_string(index=False))
