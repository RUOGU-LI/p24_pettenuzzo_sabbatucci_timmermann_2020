"""End-to-end test on simulated data with a known truth.

The simulator (synthetic_data.py) plants suspensions and a persistent growth
component. Running the real pipeline functions on its output must recover
both. This runs without WRDS, so it is the main test in CI, and it also
checks that the simulated tables have the schema the pipeline expects.

The thresholds are deliberately loose. The simulator plants firms that shift
their declaration dates, which the rule is known to misclassify, so precision
cannot be 1.
"""

from itertools import pairwise

import pandas as pd
import pytest

import calc_dividend_growth as cdg
import calc_extensive_growth as ceg
import calc_omissions as co
import calc_persistent_component as cpc
import clean_dividends
import synthetic_data
from settings import COMMON_SHARE_CODES, MAIN_EXCHANGE_CODES, OMISSION_GRACE_DAYS

MIN_RECALL = 0.90
MIN_PRECISION = 0.80
MIN_STATE_CORRELATION = 0.80


@pytest.fixture(scope="module")
def pipeline():
    """Run cleaning, matching and omission inference on a small simulated panel."""
    raw = synthetic_data.simulate_crsp(seed=7, n_firms=900)
    events, declarations, counts = clean_dividends.clean_dividend_events(
        raw["dsedist"], raw["dsenames"]
    )
    matched = cdg.match_prior_year(events)
    daily = cdg.daily_growth_on_trading_days(matched, raw["dsi"]["date"])
    expectations = co.build_expectations(matched, declarations, raw["dsenames"])
    omissions = co.infer_omissions(expectations)
    return {
        "raw": raw,
        "events": events,
        "matched": matched,
        "daily": daily,
        "omissions": omissions,
        "counts": counts,
    }


def _near(left, right, left_date, right_date):
    """For each row of left: is there a row of right, same firm, within the grace period?"""
    merged = left.merge(right[["permno", right_date]], on="permno", how="left")
    gap = (merged[left_date] - merged[right_date]).abs()
    merged["hit"] = gap <= pd.Timedelta(days=OMISSION_GRACE_DAYS)
    return merged.groupby(["permno", left_date])["hit"].any()


def test_filters_only_remove_rows(pipeline):
    """Each cleaning step leaves no more rows than the one before."""
    rows = pipeline["counts"]["rows"].tolist()
    assert all(later <= earlier for earlier, later in pairwise(rows))


def test_planted_suspensions_are_found(pipeline):
    """Recall: most suspensions planted in listed common stocks are detected."""
    names = pipeline["raw"]["dsenames"].set_index("permno")
    in_universe = names["shrcd"].isin(COMMON_SHARE_CODES) & names["exchcd"].isin(
        MAIN_EXCHANGE_CODES
    )
    truth = pipeline["raw"]["truth_suspensions"]
    truth = truth[truth["permno"].map(in_universe)]
    found = _near(truth, pipeline["omissions"], "first_missed_date", "expected_date")
    assert found.mean() >= MIN_RECALL


def test_inferred_suspensions_are_mostly_real(pipeline):
    """Precision: most inferred suspension spells correspond to a planted one."""
    starts = pipeline["omissions"][pipeline["omissions"]["is_spell_start"]]
    truth = pipeline["raw"]["truth_suspensions"]
    real = _near(starts, truth, "expected_date", "first_missed_date")
    assert real.mean() >= MIN_PRECISION


def test_robust_filter_recovers_the_planted_component(pipeline):
    """The filtered robust state is highly correlated with the true mu."""
    states, params = cpc.estimate_persistent_component(
        pipeline["daily"], models=("robust",)
    )
    estimate = cpc.select_state(states, "robust", "update", "filtered")
    truth = pipeline["raw"]["truth_mu"].set_index("date")["mu"]
    joined = pd.concat([estimate, truth], axis=1, join="inner").dropna()
    assert joined.corr().iloc[0, 1] >= MIN_STATE_CORRELATION
    phi = params[(params["parameter"] == "phi") & (params["sample"] == "update")]
    assert phi["value"].iloc[0] > 0.98


def test_omissions_lower_growth_in_the_planted_crisis(pipeline):
    """In 2009, growth including omitted dividends is below the paper's measure."""
    adjusted = ceg.adjust_growth_for_omissions(
        pipeline["daily"], pipeline["omissions"], "detection_date"
    )
    crisis = adjusted[adjusted["date"].dt.year == 2009]
    assert crisis["log_growth_ext"].mean() < crisis["log_growth"].mean()
