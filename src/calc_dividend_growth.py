"""Build the daily bottom-up dividend growth series and the top-down benchmark.

Bottom-up (equation (1) of the paper)
-------------------------------------
For every announcement by firm i on day t, find the same firm's announcement
closest to one year earlier (t-tilde). Aggregate growth on day t is

    G_t = sum_i D_{i,t} / sum_i D_{i,t-tilde}

where the sums run over firms that announced on both dates and D is the
dollar dividend. Firms without a match one year earlier (initiations) are
left out, and firms that stop paying never show up. The extension revisits
exactly that omission (see calc_omissions.py).

Two calendars
-------------
Announcements also happen on weekends and holidays (the paper's Figure 1
shows Sunday, June 22, 2014). We keep

- a calendar-day series, used for Figure 1, and
- a trading-day series, used by the state-space model, in which an
  announcement made on a non-trading day counts on the next trading day.

Top-down
--------
The conventional measure backs dividends out of the CRSP value-weighted
index: D_t = (vwretd_t - vwretx_t) x P_{t-1}, where P is the ex-dividend index
level. It records dividends when they go ex, not when they are announced.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import clean_dividends
import pull_crsp
from settings import DAYS_IN_YEAR, MATCH_WINDOW_DAYS, config

DATA_DIR = Path(config("DATA_DIR"))
MATCHED_FILE = "matched_dividend_events.parquet"
DAILY_CALENDAR_FILE = "daily_dividend_growth_calendar.parquet"
DAILY_TRADING_FILE = "daily_dividend_growth.parquet"
TOPDOWN_DAILY_FILE = "crsp_topdown_daily.parquet"


def match_prior_year(events, window_days=MATCH_WINDOW_DAYS):
    """Match each announcement to the same firm's announcement a year earlier.

    The match is the announcement whose declaration date is closest to
    (dclrdt - 365 days), if one exists within +/- window_days.

    Adds the columns prior_dclrdt, prior_dollar_div and matched.
    """
    left = events.copy()
    left["target"] = left["dclrdt"] - pd.Timedelta(days=DAYS_IN_YEAR)
    right = events[["permno", "dclrdt", "dollar_div"]].rename(
        columns={"dclrdt": "prior_dclrdt", "dollar_div": "prior_dollar_div"}
    )
    matched = pd.merge_asof(
        left.sort_values("target"),
        right.sort_values("prior_dclrdt"),
        left_on="target",
        right_on="prior_dclrdt",
        by="permno",
        direction="nearest",
        tolerance=pd.Timedelta(days=window_days),
    )
    matched["matched"] = matched["prior_dclrdt"].notna()
    return (
        matched.drop(columns="target")
        .sort_values(["permno", "dclrdt"])
        .reset_index(drop=True)
    )


def next_trading_day(dates, trading_days, max_gap_days=10):
    """Map each date to itself if it is a trading day, else the next one.

    Dates with no trading day within max_gap_days after them (that is, dates
    outside the range of the trading calendar) map to NaT.
    """
    trading_days = pd.DatetimeIndex(trading_days)
    dates = pd.DatetimeIndex(dates)
    pos = trading_days.searchsorted(dates, side="left")
    out = np.full(len(pos), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    ok = pos < len(trading_days)
    out[ok] = trading_days.to_numpy()[pos[ok]]
    out = pd.DatetimeIndex(out)
    too_far = (out - dates) > pd.Timedelta(days=max_gap_days)
    return out.where(~too_far)


def aggregate_growth(matched, date_col="dclrdt"):
    """Aggregate firm announcements into the daily growth series.

    Returns one row per date with at least one announcement:
    n_announcers (all announcers), n_matched (those with a prior-year match),
    dollars and dollars_prior (matched firms only), dollars_all,
    gross_growth = dollars / dollars_prior and log_growth.
    """
    df = matched.assign(
        _matched_dollars=matched["dollar_div"].where(matched["matched"], 0.0),
        _prior_dollars=matched["prior_dollar_div"].fillna(0.0),
    )
    daily = df.groupby(date_col).agg(
        n_announcers=("permno", "size"),
        n_matched=("matched", "sum"),
        dollars=("_matched_dollars", "sum"),
        dollars_prior=("_prior_dollars", "sum"),
        dollars_all=("dollar_div", "sum"),
    )
    daily.index.name = "date"
    daily["n_matched"] = daily["n_matched"].astype(int)
    has_growth = daily["dollars_prior"] > 0
    daily["gross_growth"] = (daily["dollars"] / daily["dollars_prior"]).where(
        has_growth
    )
    daily["log_growth"] = np.log(daily["gross_growth"].where(daily["gross_growth"] > 0))
    return daily.reset_index()


def daily_growth_on_trading_days(matched, trading_days):
    """Daily growth on the trading-day grid (missing when nobody announced)."""
    model_date = next_trading_day(matched["dclrdt"], trading_days)
    df = matched.assign(model_date=model_date).dropna(subset=["model_date"])
    daily = aggregate_growth(df, date_col="model_date").set_index("date")
    daily = daily.reindex(pd.DatetimeIndex(trading_days, name="date"))
    counts = ["n_announcers", "n_matched"]
    daily[counts] = daily[counts].fillna(0).astype(int)
    dollars = ["dollars", "dollars_prior", "dollars_all"]
    daily[dollars] = daily[dollars].fillna(0.0)
    return daily.reset_index()


def topdown_daily(dsi):
    """Daily top-down dividends implied by the CRSP value-weighted index.

    Columns: index_level (ex-dividend, starts at 1), div_index (dividends in
    index units), div_dollars (dividends in dollars, from the market value of
    the index constituents) and log_growth_yoy (log of div_index over its
    value on the closest trading day one year earlier).
    """
    df = dsi.sort_values("date").reset_index(drop=True).copy()
    level = (1.0 + df["vwretx"]).cumprod()
    lagged_level = level.shift(1).fillna(1.0)
    df["index_level"] = level
    df["div_index"] = (df["vwretd"] - df["vwretx"]) * lagged_level
    # usdval is the market value (in $000) of the securities used in the
    # index return on that day, measured at the previous close.
    df["div_dollars"] = (df["vwretd"] - df["vwretx"]) * df["usdval"] * 1000.0

    prior = df[["date", "div_index"]].rename(
        columns={"date": "prior_date", "div_index": "div_index_prior"}
    )
    df["target"] = df["date"] - pd.Timedelta(days=DAYS_IN_YEAR)
    df = pd.merge_asof(
        df,
        prior,
        left_on="target",
        right_on="prior_date",
        direction="nearest",
        tolerance=pd.Timedelta(days=7),
    )
    ok = (df["div_index"] > 0) & (df["div_index_prior"] > 0)
    df["log_growth_yoy"] = np.log((df["div_index"] / df["div_index_prior"]).where(ok))
    cols = ["date", "index_level", "div_index", "div_dollars", "log_growth_yoy"]
    return df[cols]


def topdown_low_frequency(topdown, start_date, end_date, freq):
    """Top-down dividend growth and dividend-price ratio at low frequency.

    Only data inside [start_date, end_date] are used, as in the paper, so the
    first year has no growth rate.

    freq = "Q": growth is the log change in quarterly dividends over the same
    quarter one year earlier; the dividend-price ratio uses the last four
    quarters of dividends.
    freq = "A": growth is the log change in annual dividends.
    """
    df = topdown[(topdown["date"] >= start_date) & (topdown["date"] <= end_date)]
    df = df.set_index("date")
    rule = {"Q": "QE", "A": "YE"}[freq]
    periods_per_year = {"Q": 4, "A": 1}[freq]
    low = pd.DataFrame(
        {
            "dividends": df["div_index"].resample(rule).sum(),
            "index_level": df["index_level"].resample(rule).last(),
        }
    )
    low["div_growth"] = np.log(
        low["dividends"] / low["dividends"].shift(periods_per_year)
    )
    trailing = low["dividends"].rolling(periods_per_year).sum()
    low["log_dp"] = np.log(trailing / low["index_level"])
    low.index.name = "date"
    return low.reset_index()


def load_matched_events(data_dir=DATA_DIR):
    """Load announcements with their prior-year match."""
    return pd.read_parquet(Path(data_dir) / MATCHED_FILE)


def load_daily_growth(data_dir=DATA_DIR):
    """Load the trading-day bottom-up growth series."""
    return pd.read_parquet(Path(data_dir) / DAILY_TRADING_FILE)


def load_daily_growth_calendar(data_dir=DATA_DIR):
    """Load the calendar-day bottom-up growth series."""
    return pd.read_parquet(Path(data_dir) / DAILY_CALENDAR_FILE)


def load_topdown_daily(data_dir=DATA_DIR):
    """Load the daily top-down dividend series."""
    return pd.read_parquet(Path(data_dir) / TOPDOWN_DAILY_FILE)


if __name__ == "__main__":
    events = clean_dividends.load_dividend_events(DATA_DIR)
    dsi = pull_crsp.load_dsi(DATA_DIR)

    matched = match_prior_year(events)
    matched.to_parquet(DATA_DIR / MATCHED_FILE, index=False)
    aggregate_growth(matched).to_parquet(DATA_DIR / DAILY_CALENDAR_FILE, index=False)
    daily = daily_growth_on_trading_days(matched, dsi["date"].sort_values())
    daily.to_parquet(DATA_DIR / DAILY_TRADING_FILE, index=False)
    topdown_daily(dsi).to_parquet(DATA_DIR / TOPDOWN_DAILY_FILE, index=False)

    share = matched["matched"].mean()
    print(f"Announcements: {len(matched):,}; matched to prior year: {share:.1%}")
    print(f"Trading days with a growth rate: {daily['log_growth'].notna().mean():.1%}")
