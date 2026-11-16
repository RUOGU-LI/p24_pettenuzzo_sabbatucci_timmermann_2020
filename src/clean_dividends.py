"""Turn raw CRSP distributions into a tidy table of dividend announcements.

Output: one row per firm (permno) and declaration date, with the dollar
dividend announced that day. This file only cleans; the growth series and
all analysis are built elsewhere.

Filters, in order (Section I.A of the paper and the course project notes):

1. One row per distribution. The WRDS pull returns a row for every trading
   day in a short window before the declaration date; keep the latest.
2. Ordinary dividends only (distribution codes 1000-1999) with a declaration
   date.
3. A valid price and positive shares outstanding at the announcement.
4. Common stock (share codes 10, 11).
5. Listed on NYSE, AMEX or NASDAQ (exchange codes 1, 2, 3).
6. One dividend per firm per day: several ordinary dividends declared on the
   same day are summed.

Dollar dividend = dividend per share x shares outstanding on the last trading
day on or before the declaration date. CRSP reports shares in thousands.

The number of rows left after every step is saved so the report can show
where observations are lost.

A second, wider table keeps every ordinary dividend declaration (after step
2, before the price and listing filters). The extension uses it to decide
whether an expected dividend arrived: a declared dividend is a declared
dividend even when CRSP has no price for that day, and treating it as missing
would create a false omission.
"""

from pathlib import Path

import pandas as pd

import pull_crsp
from settings import (
    COMMON_SHARE_CODES,
    IRREGULAR_FREQUENCY_DIGITS,
    MAIN_EXCHANGE_CODES,
    ORDINARY_DISTCD_MAX,
    ORDINARY_DISTCD_MIN,
    config,
)

DATA_DIR = Path(config("DATA_DIR"))
EVENTS_FILE = "clean_dividend_events.parquet"
DECLARATIONS_FILE = "clean_dividend_declarations.parquet"
FILTER_COUNTS_FILE = "dividend_filter_counts.parquet"

DISTRIBUTION_KEY = ["permno", "dclrdt", "distcd", "divamt", "exdt", "paydt"]


def frequency_digit(distcd):
    """Return the third digit of a distribution code (payment frequency)."""
    return (distcd.astype(int) // 10) % 10


def attach_security_info(df, dsenames):
    """Add the share code, exchange code and name in force on the declaration date."""
    names = dsenames[["permno", "namedt", "nameendt", "shrcd", "exchcd", "comnam"]]
    merged = df.merge(names, on="permno", how="left")
    in_force = (merged["namedt"] <= merged["dclrdt"]) & (
        merged["dclrdt"] <= merged["nameendt"]
    )
    merged = merged[in_force].drop(columns=["namedt", "nameendt"])
    # A security has one name record at any date; guard against overlaps.
    return merged.drop_duplicates(DISTRIBUTION_KEY)


def clean_dividend_events(dsedist, dsenames):
    """Apply the filters above.

    Returns
    -------
    events : DataFrame
        One row per (permno, dclrdt) that passes every filter.
    declarations : DataFrame
        One row per (permno, dclrdt) for every ordinary dividend declared,
        with has_regular marking days that include a regular dividend.
    counts : DataFrame
        Rows remaining after each filter.
    """
    counts = []

    def record(step, frame):
        counts.append({"step": step, "rows": len(frame)})

    df = dsedist.copy()
    record("Rows pulled (one per look-back trading day)", df)

    df = df.sort_values("price_date", na_position="first").drop_duplicates(
        DISTRIBUTION_KEY, keep="last"
    )
    record("One row per distribution", df)

    ordinary = df["distcd"].between(ORDINARY_DISTCD_MIN, ORDINARY_DISTCD_MAX)
    df = df[ordinary & df["dclrdt"].notna() & df["divamt"].gt(0)]
    record("Ordinary dividends with a declaration date", df)

    regular = ~frequency_digit(df["distcd"]).isin(IRREGULAR_FREQUENCY_DIGITS)
    declarations = (
        df.assign(is_regular=regular)
        .groupby(["permno", "dclrdt"], as_index=False)
        .agg(has_regular=("is_regular", "any"))
    )

    valid = df["prc"].notna() & df["prc"].ne(0) & df["shrout"].gt(0)
    df = df[valid]
    record("Valid price and shares outstanding", df)

    df = attach_security_info(df, dsenames)
    df = df[df["shrcd"].isin(COMMON_SHARE_CODES)]
    record("Common stock (share codes 10, 11)", df)

    df = df[df["exchcd"].isin(MAIN_EXCHANGE_CODES)]
    record("NYSE, AMEX or NASDAQ", df)

    df = df.assign(
        dollar_div=df["divamt"] * df["shrout"] * 1000.0,
        freq_digit=frequency_digit(df["distcd"]),
    )
    df["is_regular"] = ~df["freq_digit"].isin(IRREGULAR_FREQUENCY_DIGITS)
    df["dollar_div_regular"] = df["dollar_div"].where(df["is_regular"], 0.0)
    # The frequency of the firm-day is that of its largest component.
    df = df.sort_values("dollar_div", ascending=False)
    events = (
        df.groupby(["permno", "dclrdt"], as_index=False)
        .agg(
            dollar_div=("dollar_div", "sum"),
            dollar_div_regular=("dollar_div_regular", "sum"),
            divamt=("divamt", "sum"),
            n_distributions=("distcd", "size"),
            freq_digit=("freq_digit", "first"),
            shrout=("shrout", "first"),
            prc=("prc", "first"),
            shrcd=("shrcd", "first"),
            exchcd=("exchcd", "first"),
            comnam=("comnam", "first"),
        )
        .sort_values(["permno", "dclrdt"])
        .reset_index(drop=True)
    )
    events["has_regular"] = events["dollar_div_regular"] > 0
    record("One dividend per firm per day", events)
    return events, declarations, pd.DataFrame(counts)


def load_dividend_events(data_dir=DATA_DIR):
    """Load the tidy announcement table."""
    return pd.read_parquet(Path(data_dir) / EVENTS_FILE)


def load_declarations(data_dir=DATA_DIR):
    """Load every ordinary dividend declaration (before price filters)."""
    return pd.read_parquet(Path(data_dir) / DECLARATIONS_FILE)


def load_filter_counts(data_dir=DATA_DIR):
    """Load the row counts after each cleaning step."""
    return pd.read_parquet(Path(data_dir) / FILTER_COUNTS_FILE)


if __name__ == "__main__":
    events, declarations, counts = clean_dividend_events(
        pull_crsp.load_dsedist(DATA_DIR), pull_crsp.load_dsenames(DATA_DIR)
    )
    events.to_parquet(DATA_DIR / EVENTS_FILE, index=False)
    declarations.to_parquet(DATA_DIR / DECLARATIONS_FILE, index=False)
    counts.to_parquet(DATA_DIR / FILTER_COUNTS_FILE, index=False)
    print(counts.to_string(index=False))
