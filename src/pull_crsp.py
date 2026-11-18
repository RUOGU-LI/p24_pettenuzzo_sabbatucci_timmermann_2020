"""Pull the CRSP tables the project needs from WRDS.

Four raw files are written to DATA_DIR. They keep CRSP's own column names and
units so that anyone who knows CRSP can read them:

- crsp_dsedist.parquet   ordinary dividend distributions (crsp.dsedist) with
                         the price and shares outstanding on the last trading
                         day on or before the declaration date (crsp.dsf);
                         one row per distribution
- crsp_dsenames.parquet  share code, exchange code and name history
- crsp_dsedelist.parquet delisting dates and codes
- crsp_dsi.parquet       daily value-weighted index returns with and without
                         dividends

When USE_SYNTHETIC_DATA=1 the same four files are simulated instead (see
synthetic_data.py), so the rest of the pipeline can run without WRDS.

The legacy (SIZ) CRSP tables used here end in December 2024. Extending the
sample past that date requires the CIZ tables (crsp.stkdistributions,
crsp.dsf_v2), whose column names differ.
"""

from pathlib import Path

import pandas as pd

from settings import (
    END_DATE,
    ORDINARY_DISTCD_MAX,
    ORDINARY_DISTCD_MIN,
    PRICE_LOOKBACK_DAYS,
    PULL_START_DATE,
    config,
    use_synthetic_data,
)

DATA_DIR = Path(config("DATA_DIR"))

SOURCE_MARKER_FILE = "data_source.txt"

RAW_FILES = {
    "dsedist": "crsp_dsedist.parquet",
    "dsenames": "crsp_dsenames.parquet",
    "dsedelist": "crsp_dsedelist.parquet",
    "dsi": "crsp_dsi.parquet",
}


def standardize_dtypes(df):
    """Convert pandas nullable columns (Int64, Float64, string) to plain numpy dtypes.

    The wrds package returns nullable dtypes. Mixing them with ordinary numpy
    columns breaks merges ("incompatible merge keys int64 and Int64") and
    turns missing values into masked entries instead of NaN. Every raw table
    goes through this function when it is pulled and again when it is loaded.

    Integer columns without missing values become int64; integer columns with
    missing values and all float columns become float64; strings and booleans
    become object. Dates are left alone.
    """
    out = df.copy()
    for col in out.columns:
        dtype = out[col].dtype
        if not pd.api.types.is_extension_array_dtype(dtype):
            continue
        if pd.api.types.is_datetime64_any_dtype(dtype):
            continue
        if pd.api.types.is_integer_dtype(dtype) and not out[col].isna().any():
            out[col] = out[col].astype("int64")
        elif pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(
            dtype
        ):
            out[col] = out[col].astype("float64")
        else:
            out[col] = out[col].astype(object).where(out[col].notna(), None)
    return out


def _log(message):
    """Print progress right away (doit buffers output otherwise)."""
    print(message, flush=True)


def _connect():
    """Open a WRDS connection using WRDS_USERNAME from the environment."""
    import wrds

    _log("Connecting to WRDS (approve the Duo request on your phone if asked) ...")
    return wrds.Connection(wrds_username=config("WRDS_USERNAME"))


def distributions_query(start_date=PULL_START_DATE, end_date=END_DATE):
    """SQL for ordinary dividend distributions (one row per distribution)."""
    return f"""
        SELECT permno, distcd, divamt, dclrdt, exdt, rcrddt, paydt
        FROM crsp.dsedist
        WHERE distcd BETWEEN {int(ORDINARY_DISTCD_MIN)} AND {int(ORDINARY_DISTCD_MAX)}
          AND dclrdt BETWEEN '{start_date:%Y-%m-%d}' AND '{end_date:%Y-%m-%d}'
    """


def announcement_prices_query(start_date=PULL_START_DATE, end_date=END_DATE):
    """SQL for the price and shares on the last trading day up to each declaration.

    Returns one row per (permno, dclrdt). The join to crsp.dsf is written as
    an equality on (permno, date), tried for each of the PRICE_LOOKBACK_DAYS
    calendar days before the declaration. An equality join lets the database
    use its (permno, date) index or a hash join. Writing the same thing as
    `date BETWEEN dclrdt - 7 AND dclrdt` is a range join, which made the
    database compare every dividend of a firm with every trading day of that
    firm and did not finish in half an hour.
    """
    return f"""
        SELECT DISTINCT ON (d.permno, d.dclrdt)
               d.permno, d.dclrdt, s.date AS price_date, s.prc, s.shrout
        FROM (
            SELECT DISTINCT permno, dclrdt
            FROM crsp.dsedist
            WHERE distcd BETWEEN {int(ORDINARY_DISTCD_MIN)} AND {int(ORDINARY_DISTCD_MAX)}
              AND dclrdt BETWEEN '{start_date:%Y-%m-%d}' AND '{end_date:%Y-%m-%d}'
        ) AS d
        CROSS JOIN generate_series(0, {int(PRICE_LOOKBACK_DAYS)}) AS g(days_back)
        INNER JOIN crsp.dsf AS s
          ON s.permno = d.permno
         AND s.date = d.dclrdt - CAST(g.days_back AS INTEGER)
        ORDER BY d.permno, d.dclrdt, s.date DESC
    """


def pull_dsedist(db, start_date=PULL_START_DATE, end_date=END_DATE):
    """Pull ordinary dividend distributions with announcement-day share data.

    Two queries: the distributions themselves, and the price and shares
    outstanding on the last trading day on or before each declaration date.
    They are merged here. A distribution with no trading day in the look-back
    window keeps missing price and shares, and is dropped later by the
    cleaning step (but still counts as a declared dividend).
    """
    date_cols = ["dclrdt", "exdt", "rcrddt", "paydt"]
    _log("  distributions (crsp.dsedist) ...")
    dist = db.raw_sql(distributions_query(start_date, end_date), date_cols=date_cols)
    _log(f"  {len(dist):,} distributions. Prices at announcement (crsp.dsf) ...")
    prices = db.raw_sql(
        announcement_prices_query(start_date, end_date),
        date_cols=["dclrdt", "price_date"],
    )
    _log(f"  {len(prices):,} announcement days with a price.")
    return dist.merge(prices, on=["permno", "dclrdt"], how="left")


def pull_dsenames(db):
    """Pull the full name history (share code and exchange code by date)."""
    query = """
        SELECT permno, namedt, nameendt, shrcd, exchcd, siccd, ticker, comnam
        FROM crsp.dsenames
    """
    return db.raw_sql(query, date_cols=["namedt", "nameendt"])


def pull_dsedelist(db):
    """Pull delisting dates and delisting codes."""
    query = "SELECT permno, dlstdt, dlstcd FROM crsp.dsedelist"
    return db.raw_sql(query, date_cols=["dlstdt"])


def pull_dsi(db, start_date=PULL_START_DATE, end_date=END_DATE):
    """Pull the daily CRSP value-weighted index file."""
    query = f"""
        SELECT date, vwretd, vwretx, totval, usdval
        FROM crsp.dsi
        WHERE date BETWEEN '{start_date:%Y-%m-%d}' AND '{end_date:%Y-%m-%d}'
    """
    return db.raw_sql(query, date_cols=["date"])


def load_dsedist(data_dir=DATA_DIR):
    """Load the raw distribution events."""
    return standardize_dtypes(pd.read_parquet(Path(data_dir) / RAW_FILES["dsedist"]))


def load_dsenames(data_dir=DATA_DIR):
    """Load the raw name history."""
    return standardize_dtypes(pd.read_parquet(Path(data_dir) / RAW_FILES["dsenames"]))


def load_dsedelist(data_dir=DATA_DIR):
    """Load the raw delisting events."""
    return standardize_dtypes(pd.read_parquet(Path(data_dir) / RAW_FILES["dsedelist"]))


def load_dsi(data_dir=DATA_DIR):
    """Load the raw daily index file."""
    return standardize_dtypes(pd.read_parquet(Path(data_dir) / RAW_FILES["dsi"]))


def data_source(data_dir=DATA_DIR):
    """Return "wrds" or "synthetic": where the files in data_dir came from.

    Tests that compare with the published paper only run on WRDS data.
    Returns None if nothing has been pulled yet.
    """
    marker = Path(data_dir) / SOURCE_MARKER_FILE
    return marker.read_text().strip() if marker.exists() else None


def pull_all():
    """Yield (name, table) for the four raw tables, from WRDS or the simulator.

    Tables are yielded one at a time so each can be saved as soon as it has
    been downloaded.
    """
    if use_synthetic_data():
        import synthetic_data

        tables = synthetic_data.simulate_crsp()
        for key in RAW_FILES:
            yield key, tables[key]
        return
    db = _connect()
    try:
        pulls = {
            "dsenames": pull_dsenames,
            "dsedelist": pull_dsedelist,
            "dsi": pull_dsi,
            "dsedist": pull_dsedist,
        }
        for key, pull in pulls.items():
            _log(f"Pulling {key} ...")
            yield key, standardize_dtypes(pull(db))
    finally:
        db.close()


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for key, table in pull_all():
        table.to_parquet(DATA_DIR / RAW_FILES[key], index=False)
        _log(f"Wrote {RAW_FILES[key]}: {len(table):,} rows")
    source = "synthetic" if use_synthetic_data() else "wrds"
    (DATA_DIR / SOURCE_MARKER_FILE).write_text(source + "\n")
