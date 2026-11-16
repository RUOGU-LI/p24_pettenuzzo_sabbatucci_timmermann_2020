"""Simulate CRSP-shaped and FRED-shaped raw data with a known truth.

Why this file exists
--------------------
The real pipeline needs a WRDS account. Continuous integration does not have
one, and neither does someone who just cloned the repository. Setting
USE_SYNTHETIC_DATA=1 makes the pull scripts return the tables simulated here,
with the same column names and units as the real pulls, so every later step
runs unchanged.

Because we generate the data, we also know the truth: the persistent growth
component, which firms suspended their dividend and when. Unit tests use that
truth to check that the matching rule, the filter and the omission-inference
rule recover what was put in.

The simulated data are not calibrated to match the paper. Nothing produced
from them is a replication result.

What is simulated
-----------------
- A persistent daily growth component mu_t (AR(1)) with two crisis dips
  (2008-09 and 2020).
- A panel of firms with listing and delisting dates, share codes and exchange
  codes. Most pay quarterly; some pay monthly, semi-annually or annually.
- Dividend growth per firm equal to mu_t plus noise, so the dollar-weighted
  year-over-year growth of announcers is a noisy signal of mu_t.
- Suspensions (more likely in the crises), resumptions, initiations, special
  dividends, same-day duplicates, weekend declarations, missing prices,
  timing shifts.
- A daily value-weighted index whose dividend yield comes from the simulated
  payments, and quarterly macro series whose growth loads on mu_t.
"""

from functools import lru_cache

import numpy as np
import pandas as pd

from settings import END_DATE, PULL_START_DATE

SEED = 20201
N_FIRMS = 2500
MU_MEAN = 0.07
MU_PHI = 0.998
MU_SIGMA = 0.002
CRISES = {
    # name: (start, end, depth of the dip in mu, suspension hazard per event)
    "gfc": ("2008-07-01", "2010-03-31", 0.14, 0.035),
    "covid": ("2020-03-01", "2021-03-31", 0.12, 0.06),
}
BASE_SUSPENSION_HAZARD = 0.001
FREQUENCIES = {
    # payments per year: (third digit of distcd, probability)
    4: (3, 0.85),
    2: (4, 0.07),
    1: (5, 0.05),
    12: (2, 0.03),
}


def _simulate_mu(bdays, rng):
    """Simulate the persistent growth component on business days."""
    n = len(bdays)
    x = np.zeros(n)
    eps = rng.normal(0.0, MU_SIGMA, n)
    for t in range(1, n):
        x[t] = MU_PHI * x[t - 1] + eps[t]
    mu = MU_MEAN + x
    for start, end, depth, _ in CRISES.values():
        mask = (bdays >= start) & (bdays <= end)
        k = int(mask.sum())
        if k > 0:
            mu[mask] -= depth * np.hanning(k)
    return pd.Series(mu, index=bdays, name="mu")


def _suspension_hazard(date):
    """Probability that a scheduled payment starts a suspension."""
    for start, end, _, hazard in CRISES.values():
        # Suspensions cluster in the first half of each crisis window.
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        if start <= date <= start + (end - start) / 2:
            return hazard
    return BASE_SUSPENSION_HAZARD


def _simulate_firm(permno, bdays, mu, rng, sample_start, sample_end):
    """Simulate one firm. Returns (events, name_row, delist_row, truth)."""
    listed_from_start = rng.random() < 0.6
    if listed_from_start:
        list_start = sample_start
    else:
        span = (pd.Timestamp("2015-12-31") - sample_start).days
        list_start = sample_start + pd.Timedelta(days=int(rng.integers(0, span)))
    lifetime_days = min(int(rng.exponential(35 * 365)), 100 * 365)
    list_end = list_start + pd.Timedelta(days=max(lifetime_days, 3 * 365))
    delisted = list_end < sample_end
    if not delisted:
        list_end = sample_end

    u = rng.random()
    shrcd = 11 if u < 0.88 else (10 if u < 0.92 else 31)
    exchcd = int(rng.choice([1, 2, 3, 4], p=[0.45, 0.10, 0.42, 0.03]))
    name_row = {
        "permno": permno,
        "namedt": list_start,
        "nameendt": list_end,
        "shrcd": shrcd,
        "exchcd": exchcd,
        "siccd": int(rng.integers(1000, 9999)),
        "ticker": f"S{permno % 10000:04d}",
        "comnam": f"SYNTHETIC CORP {permno}",
    }
    delist_row = None
    if delisted:
        delist_row = {
            "permno": permno,
            "dlstdt": list_end,
            "dlstcd": int(rng.choice([231, 233, 241, 520, 574])),
        }

    truth = {"suspensions": [], "initiations": []}
    events = []
    if rng.random() > 0.8:  # one firm in five never pays
        return events, name_row, delist_row, truth

    per_year = int(
        rng.choice(list(FREQUENCIES), p=[p for _, p in FREQUENCIES.values()])
    )
    freq_digit = FREQUENCIES[per_year][0]
    distcd_regular = 1200 + 10 * freq_digit + 2
    pay_start = list_start + pd.Timedelta(days=int(rng.integers(0, 6 * 365)))
    if (pay_start - list_start).days > 600:
        truth["initiations"].append({"permno": permno, "dclrdt": None})

    shrout = float(np.exp(rng.normal(np.log(30_000), 1.5)))  # thousands
    price = float(np.exp(rng.normal(np.log(30), 0.7)))
    dps = np.full(per_year, float(np.exp(rng.normal(np.log(0.25), 0.6))))
    spacing = 365.0 / per_year
    shift_once = rng.random() < 0.01  # a few firms move their schedule
    shift_year = int(rng.integers(5, 30))

    suspended_until = None
    stopped = False
    first_event_recorded = False
    n_years = int((list_end - pay_start).days // 365) + 1
    for year in range(n_years):
        for j in range(per_year):
            jitter = int(rng.normal(0, 3))
            offset = 365 * year + spacing * j + jitter
            if shift_once and year >= shift_year:
                offset += 60
            date = pay_start + pd.Timedelta(days=int(offset))
            if date > list_end or date > sample_end or stopped:
                continue
            if suspended_until is not None:
                if date < suspended_until:
                    continue
                suspended_until = None
                if rng.random() < 0.4:
                    stopped = True
                    continue
                dps = dps * 0.5  # resume at half the old dividend
            elif year >= 2 and rng.random() < _suspension_hazard(date):
                suspended_until = date + pd.Timedelta(
                    days=int(rng.uniform(1.0, 3.0) * 365)
                )
                truth["suspensions"].append(
                    {"permno": permno, "first_missed_date": date}
                )
                continue

            mu_t = float(mu.iloc[mu.index.searchsorted(date, side="right") - 1])
            if year > 0:
                dps[j] = dps[j] * np.exp(mu_t + rng.normal(0, 0.06))
            # Boards meet on working days: roll weekend dates to Monday,
            # then move one declaration in a hundred onto a Saturday.
            dclrdt = date
            if dclrdt.weekday() >= 5:
                dclrdt = dclrdt + pd.Timedelta(days=7 - dclrdt.weekday())
            if rng.random() < 0.01:
                dclrdt = dclrdt + pd.Timedelta(days=(5 - dclrdt.weekday()) % 7)
            if not first_event_recorded and truth["initiations"]:
                truth["initiations"][0]["dclrdt"] = dclrdt
            first_event_recorded = True

            pos = bdays.searchsorted(dclrdt, side="right") - 1
            price_date = bdays[pos]
            prc = price if rng.random() > 0.02 else -price
            missing_price = rng.random() < 0.005

            rows = [(distcd_regular, float(dps[j]))]
            if rng.random() < 0.003:  # same-day duplicate
                rows = [
                    (distcd_regular, float(dps[j]) * 0.6),
                    (distcd_regular, float(dps[j]) * 0.4),
                ]
            if rng.random() < 0.005:  # special dividend on the same day
                rows.append((1272, float(dps[j]) * 3.0))
            for distcd, divamt in rows:
                base = {
                    "permno": permno,
                    "distcd": distcd,
                    "divamt": divamt,
                    "dclrdt": dclrdt,
                    "exdt": dclrdt + pd.Timedelta(days=14),
                    "rcrddt": dclrdt + pd.Timedelta(days=16),
                    "paydt": dclrdt + pd.Timedelta(days=42),
                    "price_date": pd.NaT if missing_price else price_date,
                    "prc": np.nan if missing_price else prc,
                    "shrout": np.nan if missing_price else shrout,
                }
                events.append(base)
                # The WRDS join returns one row per trading day in the
                # look-back window. Reproduce that for some events.
                if not missing_price and pos > 0 and rng.random() < 0.2:
                    events.append({**base, "price_date": bdays[pos - 1]})
    truth["initiations"] = [
        row for row in truth["initiations"] if row["dclrdt"] is not None
    ]
    return events, name_row, delist_row, truth


def _simulate_index(dsedist, dsenames, bdays, rng):
    """Build a daily index file whose dividend yield matches the payments."""
    events = dsedist.drop_duplicates(["permno", "dclrdt", "distcd", "divamt"]).dropna(
        subset=["shrout"]
    )
    paid = events.assign(dollars=events["divamt"] * events["shrout"] * 1000.0)
    pos = np.minimum(bdays.searchsorted(paid["paydt"]), len(bdays) - 1)
    paid_by_day = paid.groupby(bdays[pos])["dollars"].sum()
    paid_by_day = paid_by_day.reindex(bdays, fill_value=0.0)

    # Aggregate market value in thousands of dollars, with a mild trend.
    cap = events.groupby("permno").agg(shrout=("shrout", "first"), prc=("prc", "first"))
    total_cap_thousands = float((cap["shrout"] * cap["prc"].abs()).sum()) * 4.0
    trend = np.exp(0.0002 * np.arange(len(bdays)))
    usdval = total_cap_thousands * trend / trend[-1]
    vwretx = rng.normal(0.0003, 0.01, len(bdays))
    vwretd = vwretx + paid_by_day.to_numpy() / (usdval * 1000.0)
    return pd.DataFrame(
        {
            "date": bdays,
            "vwretd": vwretd,
            "vwretx": vwretx,
            "totval": usdval,
            "usdval": usdval,
        }
    )


def _simulate_fred(mu, rng):
    """Simulate quarterly macro series whose growth loads on lagged mu."""
    quarters = pd.date_range(mu.index[0], mu.index[-1], freq="QS")
    mu_q = mu.resample("QS").last().reindex(quarters).ffill()
    n = len(quarters)
    g_gdp = np.zeros(n)
    g_con = np.zeros(n)
    for t in range(1, n):
        g_gdp[t] = (
            0.001 + 0.08 * mu_q.iloc[t - 1] + 0.3 * g_gdp[t - 1] + rng.normal(0, 0.006)
        )
        g_con[t] = (
            0.001 + 0.05 * mu_q.iloc[t - 1] + 0.5 * g_con[t - 1] + rng.normal(0, 0.003)
        )
    price = 100 * np.exp(np.cumsum(np.full(n, 0.008)))
    real_con = 1000 * np.exp(np.cumsum(g_con))
    fred = pd.DataFrame(
        {
            "GDPC1": 5000 * np.exp(np.cumsum(g_gdp)),
            "PCND": 0.3 * real_con * price / 100,
            "PCESV": 0.7 * real_con * price / 100,
            "DNDGRG3Q086SBEA": price,
            "DSERRG3Q086SBEA": price,
        },
        index=quarters,
    )
    fred.index.name = "date"
    return fred


@lru_cache(maxsize=1)
def simulate_crsp(seed=SEED, n_firms=N_FIRMS):
    """Simulate every raw table and the truth behind them.

    Returns a dict with the keys dsedist, dsenames, dsedelist, dsi, fred
    (same schemas as the real pulls) and truth_mu, truth_suspensions,
    truth_initiations.
    """
    rng = np.random.default_rng(seed)
    sample_start = pd.Timestamp(PULL_START_DATE)
    sample_end = pd.Timestamp(END_DATE)
    bdays = pd.bdate_range(sample_start, sample_end)
    mu = _simulate_mu(bdays, rng)

    events, names, delists, suspensions, initiations = [], [], [], [], []
    for i in range(n_firms):
        firm_events, name_row, delist_row, truth = _simulate_firm(
            10_000 + i, bdays, mu, rng, sample_start, sample_end
        )
        events.extend(firm_events)
        names.append(name_row)
        if delist_row is not None:
            delists.append(delist_row)
        suspensions.extend(truth["suspensions"])
        initiations.extend(truth["initiations"])

    dsedist = pd.DataFrame(events)
    for col in ["dclrdt", "exdt", "rcrddt", "paydt", "price_date"]:
        dsedist[col] = pd.to_datetime(dsedist[col])
    dsenames = pd.DataFrame(names)
    dsedelist = pd.DataFrame(delists, columns=["permno", "dlstdt", "dlstcd"])
    dsedelist["dlstdt"] = pd.to_datetime(dsedelist["dlstdt"])
    return {
        "dsedist": dsedist,
        "dsenames": dsenames,
        "dsedelist": dsedelist,
        "dsi": _simulate_index(dsedist, dsenames, bdays, rng),
        "fred": _simulate_fred(mu, rng),
        "truth_mu": mu.rename_axis("date").reset_index(),
        "truth_suspensions": pd.DataFrame(
            suspensions, columns=["permno", "first_missed_date"]
        ),
        "truth_initiations": pd.DataFrame(initiations, columns=["permno", "dclrdt"]),
    }
