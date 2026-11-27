"""Extension, part 2: dividend growth that includes firms that stopped paying.

The paper's growth rate is

    G_t = sum of dollars announced today / sum of the same firms' dollars a year ago

Following Pettenuzzo, Sabbatucci and Timmermann (2023, Section 2.2), a firm
that omits its dividend enters with zero in the numerator and its prior-year
dividend in the denominator. We do that with the CRSP-inferred omissions:

    G_ext_t = dollars_t / (dollars_prior_t + omitted_dollars_t)

Two datings of the omission are built:

"realtime"  the omission enters on its detection date (expected date plus the
            window). Usable in real time; this is the version the
            predictive regressions use.
"expost"    the omission enters on the expected announcement date. Not
            available in real time; it shows how much the detection lag
            costs.

The robust filter of calc_persistent_component.py is then run on G_ext. It
treats days with fewer than MIN_ANNOUNCERS_ROBUST matched announcers as
missing. Omitted dollars that fall on any day the filter does not use (no
announcers, or too few) are therefore carried to the next day it does use, so
every omission reaches the filter and the log growth rate is always defined.
Omitted dollars after the last usable day have nowhere to go and are dropped.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import calc_dividend_growth
import calc_omissions
import calc_persistent_component
from settings import MIN_ANNOUNCERS_ROBUST, config

DATA_DIR = Path(config("DATA_DIR"))
DAILY_EXT_FILE = "daily_dividend_growth_extensive.parquet"
STATE_EXT_FILE = "persistent_component_extensive.parquet"
PARAMS_EXT_FILE = "persistent_component_extensive_params.parquet"

DATINGS = {"realtime": "detection_date", "expost": "expected_date"}


def carry_forward_to_usable_days(omitted, usable):
    """Move omitted dollars on unusable days to the next usable day.

    Parameters are aligned arrays over trading days: omitted dollars, and a
    boolean array that is True on days the robust filter uses. Returns the
    adjusted omitted-dollar array; its total is unchanged except for amounts
    after the last usable day, which have nowhere to go and are dropped.
    """
    out = np.zeros(len(omitted))
    pending = 0.0
    for t in range(len(omitted)):
        pending += omitted[t]
        if usable[t]:
            out[t] = pending
            pending = 0.0
    return out


def usable_days(daily, min_matched=MIN_ANNOUNCERS_ROBUST):
    """Days the robust filter uses: positive dollars and enough matched firms."""
    return ((daily["dollars"] > 0) & (daily["n_matched"] >= min_matched)).to_numpy()


def adjust_growth_for_omissions(daily, omissions, date_col):
    """Add omitted prior-year dollars to the denominator of daily growth.

    daily is the trading-day growth table; omissions the inferred omission
    events; date_col the omission date to use ("detection_date" or
    "expected_date"). Omitted dollars only land on days the robust filter
    uses (see usable_days).

    n_omitted counts omissions on their own trading day, before carry-forward;
    dollars_omitted is after carry-forward. On a day the filter skips,
    n_omitted can be positive while dollars_omitted is zero.
    """
    daily = daily.sort_values("date").reset_index(drop=True).copy()
    days = pd.DatetimeIndex(daily["date"])
    model_date = calc_dividend_growth.next_trading_day(omissions[date_col], days)
    by_day = (
        omissions.assign(model_date=model_date)
        .dropna(subset=["model_date"])
        .groupby("model_date")
        .agg(omitted=("dollars_prior", "sum"), n_omitted=("permno", "size"))
        .reindex(days, fill_value=0)
    )
    omitted = carry_forward_to_usable_days(
        by_day["omitted"].to_numpy(dtype=float), usable_days(daily)
    )
    daily["n_omitted"] = by_day["n_omitted"].to_numpy()
    daily["dollars_omitted"] = omitted
    denominator = daily["dollars_prior"] + omitted
    ok = (daily["dollars"] > 0) & (denominator > 0)
    daily["gross_growth_ext"] = (daily["dollars"] / denominator).where(ok)
    daily["log_growth_ext"] = np.log(daily["gross_growth_ext"])
    return daily


def load_daily_extensive(data_dir=DATA_DIR):
    """Load daily growth including omissions (long by dating)."""
    return pd.read_parquet(Path(data_dir) / DAILY_EXT_FILE)


def load_states_extensive(data_dir=DATA_DIR):
    """Load the persistent component of growth including omissions."""
    return pd.read_parquet(Path(data_dir) / STATE_EXT_FILE)


if __name__ == "__main__":
    daily = calc_dividend_growth.load_daily_growth(DATA_DIR)
    omissions = calc_omissions.load_omissions(DATA_DIR)

    daily_parts, state_parts, param_parts = [], [], []
    for dating, date_col in DATINGS.items():
        adjusted = adjust_growth_for_omissions(daily, omissions, date_col)
        states, params = calc_persistent_component.estimate_persistent_component(
            adjusted, value_col="log_growth_ext", models=("robust",)
        )
        daily_parts.append(adjusted.assign(dating=dating))
        state_parts.append(states.assign(dating=dating))
        param_parts.append(params.assign(dating=dating))

    pd.concat(daily_parts, ignore_index=True).to_parquet(
        DATA_DIR / DAILY_EXT_FILE, index=False
    )
    pd.concat(state_parts, ignore_index=True).to_parquet(
        DATA_DIR / STATE_EXT_FILE, index=False
    )
    pd.concat(param_parts, ignore_index=True).to_parquet(
        DATA_DIR / PARAMS_EXT_FILE, index=False
    )
    print(f"Wrote {DAILY_EXT_FILE}, {STATE_EXT_FILE}, {PARAMS_EXT_FILE}")
