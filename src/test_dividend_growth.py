"""Unit tests for year-over-year matching and daily aggregation."""

import numpy as np
import pandas as pd
import pytest

import calc_dividend_growth as cdg


def _events(rows):
    """Build a minimal cleaned announcement table from (permno, date, dollars)."""
    df = pd.DataFrame(rows, columns=["permno", "dclrdt", "dollar_div"])
    df["dclrdt"] = pd.to_datetime(df["dclrdt"])
    return df


def test_paper_example_is_matched():
    """The paper matches May 17, 2014 to May 9, 2013 (8 days off the anniversary)."""
    matched = cdg.match_prior_year(
        _events([(1, "2013-05-09", 100.0), (1, "2014-05-17", 110.0)])
    )
    row = matched[matched["dclrdt"] == "2014-05-17"].iloc[0]
    assert row["matched"]
    assert row["prior_dclrdt"] == pd.Timestamp("2013-05-09")
    assert row["prior_dollar_div"] == 100.0


def test_announcement_outside_window_is_not_matched():
    """A prior announcement 100 days off the anniversary is a different quarter."""
    matched = cdg.match_prior_year(
        _events([(1, "2013-02-01", 100.0), (1, "2014-05-17", 110.0)])
    )
    assert not matched[matched["dclrdt"] == "2014-05-17"]["matched"].iloc[0]


def test_match_is_within_the_same_firm():
    """Another firm's announcement a year ago must not be used."""
    matched = cdg.match_prior_year(
        _events([(2, "2013-05-17", 100.0), (1, "2014-05-17", 110.0)])
    )
    assert not matched["matched"].any()


def test_quarterly_payer_picks_same_quarter():
    """With four announcements last year, the one closest to the anniversary wins."""
    rows = [
        (1, d, v)
        for d, v in [
            ("2013-02-10", 1.0),
            ("2013-05-10", 2.0),
            ("2013-08-10", 3.0),
            ("2013-11-10", 4.0),
        ]
    ]
    matched = cdg.match_prior_year(_events(rows + [(1, "2014-08-12", 9.0)]))
    assert matched[matched["dclrdt"] == "2014-08-12"]["prior_dollar_div"].iloc[0] == 3.0


def test_costco_checkpoint_growth():
    """Figure 1: $155.6m against $135.3m a year earlier is gross growth 1.15."""
    matched = cdg.match_prior_year(
        _events([(1, "2013-06-20", 135.3e6), (1, "2014-06-22", 155.6e6)])
    )
    daily = cdg.aggregate_growth(matched).set_index("date")
    assert daily.loc["2014-06-22", "gross_growth"] == pytest.approx(1.15, abs=0.005)


def test_growth_is_dollar_weighted_over_matched_firms_only():
    """Unmatched announcers count as announcers but not in the growth rate."""
    rows = [
        (1, "2013-05-15", 100.0),
        (2, "2013-05-15", 300.0),
        (1, "2014-05-15", 110.0),
        (2, "2014-05-15", 330.0),
        (3, "2014-05-15", 999.0),
    ]
    daily = cdg.aggregate_growth(cdg.match_prior_year(_events(rows))).set_index("date")
    day = daily.loc["2014-05-15"]
    assert day["n_announcers"] == 3
    assert day["n_matched"] == 2
    assert day["gross_growth"] == pytest.approx(440.0 / 400.0)
    assert day["dollars_all"] == pytest.approx(110.0 + 330.0 + 999.0)


def test_weekend_announcement_moves_to_next_trading_day():
    """Sunday, June 22, 2014 belongs to Monday, June 23 on the trading-day grid."""
    trading_days = pd.bdate_range("2014-06-16", "2014-06-27")
    mapped = cdg.next_trading_day(
        pd.to_datetime(["2014-06-22", "2014-06-24"]), trading_days
    )
    assert mapped[0] == pd.Timestamp("2014-06-23")
    assert mapped[1] == pd.Timestamp("2014-06-24")


def test_trading_day_series_has_missing_growth_on_quiet_days():
    """Days without announcers stay in the grid with a missing growth rate."""
    rows = [(1, "2013-05-15", 100.0), (1, "2014-05-15", 110.0)]
    trading_days = pd.bdate_range("2014-05-12", "2014-05-16")
    daily = cdg.daily_growth_on_trading_days(
        cdg.match_prior_year(_events(rows)), trading_days
    )
    assert len(daily) == 5
    assert daily["log_growth"].notna().sum() == 1
    assert daily["n_announcers"].sum() == 1


def test_topdown_dividends_from_index_returns():
    """D_t = (vwretd - vwretx) x previous ex-dividend index level."""
    dsi = pd.DataFrame(
        {
            "date": pd.bdate_range("2020-01-01", periods=3),
            "vwretd": [0.010, 0.020, 0.000],
            "vwretx": [0.010, 0.015, 0.000],
            "totval": [1e6] * 3,
            "usdval": [1e6] * 3,
        }
    )
    top = cdg.topdown_daily(dsi)
    assert top.loc[0, "div_index"] == pytest.approx(0.0)
    assert top.loc[1, "div_index"] == pytest.approx(0.005 * 1.01)
    assert top.loc[1, "div_dollars"] == pytest.approx(0.005 * 1e6 * 1000)


def test_low_frequency_growth_needs_a_full_prior_year():
    """Quarterly growth is year over year, so the first four quarters are missing."""
    dates = pd.bdate_range("2000-01-01", "2002-12-31")
    top = pd.DataFrame(
        {
            "date": dates,
            "index_level": 1.0,
            "div_index": np.linspace(1.0, 2.0, len(dates)),
        }
    )
    low = cdg.topdown_low_frequency(
        top, pd.Timestamp("2000-01-01"), pd.Timestamp("2002-12-31"), "Q"
    )
    assert low["div_growth"].isna().sum() == 4
    assert (low["div_growth"].dropna() > 0).all()
