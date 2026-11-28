"""Unit tests for the extension regressions (calc_extension_regressions.py).

The scorecard decides whether the extension "worked", and the out-of-sample
split decides what counts as a forecast, so both are tested on hand-built
inputs with a known answer.
"""

import numpy as np
import pandas as pd
import pytest

import calc_extension_regressions as cer
from settings import OOS_START_DATE, PAPER_END_DATE, SIGNIFICANCE_T


def _in_sample(omission_t, adj_r2_with, adj_r2_without=0.10):
    rows = [
        {
            "sample": "update",
            "depvar": "dividend",
            "spec": "baseline",
            "variable": "mu",
            "coef": 0.5,
            "tstat": 3.0,
            "adj_r2": adj_r2_without,
        },
        {
            "sample": "update",
            "depvar": "dividend",
            "spec": "plus_omissions",
            "variable": "omission_share",
            "coef": np.sign(omission_t) * 0.1,
            "tstat": omission_t,
            "adj_r2": adj_r2_with,
        },
    ]
    return pd.DataFrame(rows)


def _oos(ratio):
    return pd.DataFrame(
        [
            {
                "depvar": "dividend",
                "spec": "plus_omissions",
                "rmse_ratio_to_baseline": ratio,
            }
        ]
    )


def test_scorecard_passes_when_all_three_criteria_hold():
    card = cer.build_scorecard(_in_sample(-SIGNIFICANCE_T - 0.5, 0.12), _oos(0.95))
    assert card["passed"].tolist() == [True, True, True]


def test_scorecard_fails_each_criterion_on_its_own():
    card = cer.build_scorecard(_in_sample(-SIGNIFICANCE_T + 0.4, 0.09), _oos(1.01))
    assert card["passed"].tolist() == [False, False, False]
    # A significant but positive coefficient does not count as success.
    card = cer.build_scorecard(_in_sample(SIGNIFICANCE_T + 1.0, 0.12), _oos(0.95))
    assert card["passed"].tolist() == [False, True, True]


def test_out_of_sample_coefficients_never_see_post_sample_targets(monkeypatch):
    """Training uses only rows whose target quarter ends by PAPER_END_DATE.

    y = 2x before the cutoff and y = 2x + 10 after it. If any post-cutoff row
    leaked into training, the forecast errors would not all equal 10.
    """
    quarters = pd.date_range("2005-03-31", "2020-12-31", freq="QE")
    x = pd.Series(np.linspace(-1.0, 1.0, len(quarters)), index=quarters, name="mu")
    target = quarters + pd.offsets.QuarterEnd(1)
    y = pd.Series(2.0 * x.to_numpy(), index=quarters)
    y[target > pd.Timestamp(PAPER_END_DATE)] += 10.0

    monkeypatch.setattr(cer, "build_dataset", lambda *args: (y, x.to_frame()))
    out = cer.run_out_of_sample(None, None, None, None)

    n_test = int((target >= pd.Timestamp(OOS_START_DATE)).sum())
    assert (out["n_forecasts"] == n_test).all()
    assert out["rmse"].to_numpy() == pytest.approx(10.0)
    assert out["rmse_ratio_to_baseline"].to_numpy() == pytest.approx(1.0)
