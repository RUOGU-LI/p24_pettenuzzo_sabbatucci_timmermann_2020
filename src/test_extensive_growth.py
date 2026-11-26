"""Unit tests for growth that includes omitted dividends (calc_extensive_growth.py)."""

import numpy as np
import pandas as pd
import pytest

import calc_extensive_growth as ceg
import calc_persistent_component as cpc
from settings import MIN_ANNOUNCERS_ROBUST

ENOUGH = MIN_ANNOUNCERS_ROBUST
TOO_FEW = MIN_ANNOUNCERS_ROBUST - 1


def test_carry_forward_keeps_total_omitted_dollars():
    """Omitted dollars on an unusable day move to the next usable day."""
    omitted = np.array([0.0, 5.0, 0.0, 2.0, 0.0])
    usable = np.array([True, False, False, True, True])
    out = ceg.carry_forward_to_usable_days(omitted, usable)
    assert out.tolist() == [0.0, 0.0, 0.0, 7.0, 0.0]
    assert out.sum() == omitted.sum()


def _daily():
    """Five trading days; days 1 and 4 have no announcers, day 3 too few."""
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2009-03-02", periods=5),
            "dollars": [110.0, 0.0, 220.0, 55.0, 0.0],
            "dollars_prior": [100.0, 0.0, 200.0, 50.0, 0.0],
            "n_matched": [ENOUGH, 0, ENOUGH, TOO_FEW, 0],
        }
    )


def test_growth_with_omissions_is_never_above_the_paper_measure():
    """Adding non-negative dollars to the denominator can only lower growth."""
    daily = _daily()
    days = daily["date"]
    omissions = pd.DataFrame(
        {
            "permno": [9, 8],
            "detection_date": [days[1], days[0]],
            "dollars_prior": [50.0, 25.0],
        }
    )
    adjusted = ceg.adjust_growth_for_omissions(daily, omissions, "detection_date")
    base = np.log(daily["dollars"] / daily["dollars_prior"])
    has_growth = adjusted["log_growth_ext"].notna()
    assert (
        adjusted.loc[has_growth, "log_growth_ext"] <= base[has_growth] + 1e-12
    ).all()
    assert adjusted.loc[0, "gross_growth_ext"] == pytest.approx(110.0 / 125.0)
    # The firm detected on a day without announcers enters on the next usable day.
    assert adjusted.loc[2, "gross_growth_ext"] == pytest.approx(220.0 / 250.0)


def test_omitted_dollars_only_land_on_days_the_robust_filter_uses():
    """An omission detected just before a thin day is not lost to the screen.

    Day 3 has announcers but fewer than MIN_ANNOUNCERS_ROBUST matched firms, so
    the robust filter treats it as missing. Omitted dollars detected on day 3
    must wait for a usable day instead; here there is none, so they are dropped
    and no day the filter uses is distorted.
    """
    daily = _daily()
    omissions = pd.DataFrame(
        {"permno": [7], "detection_date": [daily["date"][3]], "dollars_prior": [40.0]}
    )
    adjusted = ceg.adjust_growth_for_omissions(daily, omissions, "detection_date")
    y = cpc.prepare_observations(adjusted, "robust", value_col="log_growth_ext")
    used = ~np.isnan(y)
    assert adjusted.loc[~used, "dollars_omitted"].sum() == 0.0
    assert adjusted.loc[3, "gross_growth_ext"] == pytest.approx(55.0 / 50.0)
