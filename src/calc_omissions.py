"""Extension, part 1: infer dividend omissions and initiations from CRSP.

The gap in the paper
--------------------
Equation (1) of the paper only uses firms that announce a dividend both today
and one year earlier. A firm that stops paying drops out of the numerator and
the denominator at once, so the growth measure cannot see it. The authors
later added suspensions by hand-collecting announcements from Capital IQ,
8-K filings and press releases, from 2005 on (Pettenuzzo, Sabbatucci and
Timmermann, 2023). CRSP itself never records a dividend that did not happen.

What we do instead
------------------
We infer an omission from the absence of an expected announcement, using
CRSP only, for the whole sample:

1. Expectation. A regular dividend announced on date s by an established
   payer creates an expectation of an announcement on s + 365 days. "Regular"
   excludes special and non-recurring dividends. "Established" means the
   announcement on s was itself matched to one a year earlier.
2. Fulfilment. The expectation is met if the firm declares a regular
   dividend from OMISSION_GRACE_DAYS before to OMISSION_LATE_DAYS after the
   expected date. The window is longer on the late side so that a firm that
   only moved its board meeting by a month or two is not flagged. Every
   declaration in CRSP counts here, including those the growth series drops
   for lack of a valid price; otherwise a data gap would look like an
   omission.
3. Omission. If it is not met, and the firm is still a listed common stock
   when the window closes, we record an omission. It becomes known on the
   detection date (expected date + OMISSION_LATE_DAYS), which is the date a
   real-time user could act on.

Because the rule works year over year it does not need to know whether a firm
pays monthly, quarterly, semi-annually or annually.

A firm that suspends a quarterly dividend produces up to four omissions in
the following year. Omissions by the same firm that are close together form
one suspension spell; the first one marks a "suspending firm", which is the
unit the hand-collected counts in the literature use.

Initiations are the mirror image: a regular dividend from a firm that has not
announced any dividend for INITIATION_LOOKBACK_DAYS and has been listed at
least that long.

Known limits (see the report): the omission is only detected when the window
closes; a firm that moves its declaration date by more than the window
is flagged by mistake; delistings and share-class changes are handled through
the CRSP name history only.

The measure catches suspensions, not every missed payment. The late side of
the window (OMISSION_LATE_DAYS = 100) is longer than a quarter (91 days), and
one declaration may fulfil more than one expectation. A quarterly payer that
skips a single quarter therefore has that expectation met by its next
quarterly declaration, and a firm that cuts from quarterly to semi-annual
payments has every expectation met by one of the remaining declarations.
Neither case produces an omission.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import calc_dividend_growth
import clean_dividends
import pull_crsp
from settings import (
    COMMON_SHARE_CODES,
    DAYS_IN_YEAR,
    END_DATE,
    INITIATION_LOOKBACK_DAYS,
    MAIN_EXCHANGE_CODES,
    OMISSION_GRACE_DAYS,
    OMISSION_LATE_DAYS,
    OMISSION_REQUIRE_ESTABLISHED,
    PULL_START_DATE,
    SUSPENSION_SPELL_GAP_DAYS,
    config,
)

DATA_DIR = Path(config("DATA_DIR"))
EXPECTATIONS_FILE = "dividend_expectations.parquet"
OMISSIONS_FILE = "omission_events.parquet"
INITIATIONS_FILE = "initiation_events.parquet"
INDEX_FILE = "extensive_margin_index_quarterly.parquet"
ANNUAL_FILE = "suspension_counts_annual.parquet"


def in_universe_on(permnos, dates, dsenames):
    """True where the security is a listed common stock on the given date."""
    query = pd.DataFrame({"permno": np.asarray(permnos), "date": np.asarray(dates)})
    query["row"] = np.arange(len(query))
    names = dsenames[["permno", "namedt", "nameendt", "shrcd", "exchcd"]]
    merged = query.merge(names, on="permno", how="left")
    ok = (
        (merged["namedt"] <= merged["date"])
        & (merged["date"] <= merged["nameendt"])
        & merged["shrcd"].isin(COMMON_SHARE_CODES)
        & merged["exchcd"].isin(MAIN_EXCHANGE_CODES)
    )
    hit = merged.loc[ok, "row"].unique()
    result = np.zeros(len(query), dtype=bool)
    result[hit] = True
    return result


def build_expectations(
    matched,
    declarations,
    dsenames,
    data_end=END_DATE,
    grace_days=OMISSION_GRACE_DAYS,
    late_days=OMISSION_LATE_DAYS,
    require_established=OMISSION_REQUIRE_ESTABLISHED,
):
    """Create one expected announcement per regular dividend and resolve it.

    matched is the cleaned announcement table with prior-year matches;
    declarations is every ordinary declaration in CRSP (used for fulfilment).

    Returns one row per expectation whose window closed on or before
    data_end and whose firm was still in the universe at that point, with
    the columns permno, source_dclrdt, expected_date, dollars_prior, comnam,
    detection_date, fulfilled_by (date of the declaration that met the
    expectation, missing if none did) and fulfilled.
    """
    regular = matched[matched["has_regular"]]
    source = regular[regular["matched"]] if require_established else regular
    grace = pd.Timedelta(days=grace_days)
    late = pd.Timedelta(days=late_days)
    exp = pd.DataFrame(
        {
            "permno": source["permno"].to_numpy(),
            "source_dclrdt": source["dclrdt"].to_numpy(),
            "expected_date": (
                source["dclrdt"] + pd.Timedelta(days=DAYS_IN_YEAR)
            ).to_numpy(),
            "dollars_prior": source["dollar_div_regular"].to_numpy(),
            "comnam": source["comnam"].to_numpy(),
        }
    )
    exp["detection_date"] = exp["expected_date"] + late
    exp["window_start"] = exp["expected_date"] - grace
    exp = exp[exp["detection_date"] <= pd.Timestamp(data_end)]

    declared = declarations[declarations["has_regular"]]
    later = declared[["permno", "dclrdt"]].rename(columns={"dclrdt": "fulfilled_by"})
    # First regular declaration on or after the window opens; the expectation
    # is met if it falls before the window closes.
    exp = pd.merge_asof(
        exp.sort_values("window_start"),
        later.sort_values("fulfilled_by"),
        left_on="window_start",
        right_on="fulfilled_by",
        by="permno",
        direction="forward",
        tolerance=grace + late,
    ).drop(columns="window_start")
    exp["fulfilled"] = exp["fulfilled_by"].notna()
    listed = in_universe_on(exp["permno"], exp["detection_date"], dsenames)
    exp = exp[exp["fulfilled"] | listed]
    return exp.sort_values(["permno", "expected_date"]).reset_index(drop=True)


def infer_omissions(expectations, spell_gap_days=SUSPENSION_SPELL_GAP_DAYS):
    """Return the unmet expectations, with suspension spells marked."""
    om = expectations[~expectations["fulfilled"]].drop(
        columns=["fulfilled", "fulfilled_by"]
    )
    om = om.sort_values(["permno", "expected_date"]).reset_index(drop=True)
    gap = om.groupby("permno")["expected_date"].diff()
    om["is_spell_start"] = gap.isna() | (gap > pd.Timedelta(days=spell_gap_days))
    return om


def infer_initiations(
    matched,
    declarations,
    dsenames,
    data_start=PULL_START_DATE,
    lookback_days=INITIATION_LOOKBACK_DAYS,
):
    """Return regular dividends announced after a long spell without any."""
    all_dates = declarations.sort_values(["permno", "dclrdt"])
    previous_map = all_dates.assign(
        previous=all_dates.groupby("permno")["dclrdt"].shift(1)
    ).set_index(["permno", "dclrdt"])["previous"]
    df = matched.sort_values(["permno", "dclrdt"]).copy()
    previous = pd.Series(
        previous_map.reindex(
            pd.MultiIndex.from_frame(df[["permno", "dclrdt"]])
        ).to_numpy(),
        index=df.index,
    )
    lookback = pd.Timedelta(days=lookback_days)
    no_recent_dividend = previous.isna() | (df["dclrdt"] - previous > lookback)
    first_listed = dsenames.groupby("permno")["namedt"].min()
    listed_long_enough = (df["dclrdt"] - df["permno"].map(first_listed)) >= lookback
    observed_long_enough = df["dclrdt"] >= pd.Timestamp(data_start) + lookback
    keep = (
        df["has_regular"]
        & no_recent_dividend
        & listed_long_enough
        & observed_long_enough
    )
    cols = ["permno", "dclrdt", "dollar_div", "comnam"]
    out = df.loc[keep, cols].copy()
    out["is_resumption"] = previous[keep].notna().to_numpy()
    return out.reset_index(drop=True)


def quarterly_index(expectations, omissions, initiations, matched):
    """Quarterly omission and initiation shares.

    Every quantity is dated by when it becomes known, so the value for
    quarter q uses information available at the end of q only.

    omission_share         unmet / all expectations whose window closed
                           in the quarter (by count)
    omission_share_dollar  the same, weighted by the prior-year dollar dividend
    initiation_share       initiations / announcements in the quarter
    initiation_share_dollar  the same, weighted by dollars announced
    n_suspension_starts    firms starting a suspension spell in the quarter
    """
    exp_q = expectations.set_index("detection_date").resample("QE")
    om_q = omissions.set_index("detection_date").resample("QE")
    ann_q = matched.set_index("dclrdt").resample("QE")
    init_q = initiations.set_index("dclrdt").resample("QE")
    idx = pd.DataFrame(
        {
            "n_expectations": exp_q["permno"].size(),
            "dollars_expected": exp_q["dollars_prior"].sum(),
            "n_omissions": om_q["permno"].size(),
            "dollars_omitted": om_q["dollars_prior"].sum(),
            "n_suspension_starts": om_q["is_spell_start"].sum(),
            "n_announcements": ann_q["permno"].size(),
            "dollars_announced": ann_q["dollar_div"].sum(),
            "n_initiations": init_q["permno"].size(),
            "dollars_initiated": init_q["dollar_div"].sum(),
        }
    ).fillna(0.0)
    idx["omission_share"] = idx["n_omissions"] / idx["n_expectations"]
    idx["omission_share_dollar"] = idx["dollars_omitted"] / idx["dollars_expected"]
    idx["initiation_share"] = idx["n_initiations"] / idx["n_announcements"]
    idx["initiation_share_dollar"] = idx["dollars_initiated"] / idx["dollars_announced"]
    idx.index.name = "date"
    return idx.reset_index()


def annual_suspension_counts(omissions):
    """Suspending firms and missed payments per calendar year.

    Dated by the expected announcement date (when the payment was missed),
    which is how hand-collected suspension counts are dated.
    """
    year = omissions["expected_date"].dt.year
    counts = pd.DataFrame(
        {
            "n_missed_payments": omissions.groupby(year)["permno"].size(),
            "n_suspending_firms": omissions[omissions["is_spell_start"]]
            .groupby(year)["permno"]
            .nunique(),
        }
    ).fillna(0)
    counts.index.name = "year"
    return counts.astype(int).reset_index()


def load_expectations(data_dir=DATA_DIR):
    """Load resolved dividend expectations."""
    return pd.read_parquet(Path(data_dir) / EXPECTATIONS_FILE)


def load_omissions(data_dir=DATA_DIR):
    """Load inferred omission events."""
    return pd.read_parquet(Path(data_dir) / OMISSIONS_FILE)


def load_initiations(data_dir=DATA_DIR):
    """Load inferred initiation events."""
    return pd.read_parquet(Path(data_dir) / INITIATIONS_FILE)


def load_quarterly_index(data_dir=DATA_DIR):
    """Load the quarterly omission and initiation index."""
    return pd.read_parquet(Path(data_dir) / INDEX_FILE)


def load_annual_counts(data_dir=DATA_DIR):
    """Load annual suspension counts."""
    return pd.read_parquet(Path(data_dir) / ANNUAL_FILE)


if __name__ == "__main__":
    matched = calc_dividend_growth.load_matched_events(DATA_DIR)
    declarations = clean_dividends.load_declarations(DATA_DIR)
    dsenames = pull_crsp.load_dsenames(DATA_DIR)

    expectations = build_expectations(matched, declarations, dsenames)
    omissions = infer_omissions(expectations)
    initiations = infer_initiations(matched, declarations, dsenames)
    index = quarterly_index(expectations, omissions, initiations, matched)
    annual = annual_suspension_counts(omissions)

    expectations.to_parquet(DATA_DIR / EXPECTATIONS_FILE, index=False)
    omissions.to_parquet(DATA_DIR / OMISSIONS_FILE, index=False)
    initiations.to_parquet(DATA_DIR / INITIATIONS_FILE, index=False)
    index.to_parquet(DATA_DIR / INDEX_FILE, index=False)
    annual.to_parquet(DATA_DIR / ANNUAL_FILE, index=False)
    print(
        f"Expectations: {len(expectations):,}; omissions: {len(omissions):,}; "
        f"suspending firms: {int(omissions['is_spell_start'].sum()):,}; "
        f"initiations: {len(initiations):,}"
    )
