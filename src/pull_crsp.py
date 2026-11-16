"""Pull the CRSP tables the project needs from WRDS.

Four raw files are written to DATA_DIR. They keep CRSP's own column names and
units so that anyone who knows CRSP can read them:

- crsp_dsedist.parquet   ordinary dividend distributions (crsp.dsedist) with
                         the price and shares outstanding on the last trading
                         day on or before the declaration date (crsp.dsf)
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


def _connect():
    """Open a WRDS connection using WRDS_USERNAME from the environment."""
    import wrds

    return wrds.Connection(wrds_username=config("WRDS_USERNAME"))


def pull_dsedist(db, start_date=PULL_START_DATE, end_date=END_DATE):
    """Pull ordinary dividend distributions with announcement-day share data.

    Each distribution is joined to every crsp.dsf row of the same security in
    the PRICE_LOOKBACK_DAYS calendar days ending on the declaration date. The
    cleaning step keeps the latest of those rows. Pulling the join instead of
    the whole daily stock file keeps the download to a few million rows.
    """
    query = f"""
        SELECT d.permno, d.distcd, d.divamt, d.dclrdt, d.exdt, d.rcrddt,
               d.paydt, s.date AS price_date, s.prc, s.shrout
        FROM crsp.dsedist AS d
        LEFT JOIN crsp.dsf AS s
          ON s.permno = d.permno
         AND s.date BETWEEN d.dclrdt - {int(PRICE_LOOKBACK_DAYS)} AND d.dclrdt
        WHERE d.distcd BETWEEN {int(ORDINARY_DISTCD_MIN)} AND {int(ORDINARY_DISTCD_MAX)}
          AND d.dclrdt BETWEEN '{start_date:%Y-%m-%d}' AND '{end_date:%Y-%m-%d}'
    """
    df = db.raw_sql(
        query, date_cols=["dclrdt", "exdt", "rcrddt", "paydt", "price_date"]
    )
    return df


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
    return pd.read_parquet(Path(data_dir) / RAW_FILES["dsedist"])


def load_dsenames(data_dir=DATA_DIR):
    """Load the raw name history."""
    return pd.read_parquet(Path(data_dir) / RAW_FILES["dsenames"])


def load_dsedelist(data_dir=DATA_DIR):
    """Load the raw delisting events."""
    return pd.read_parquet(Path(data_dir) / RAW_FILES["dsedelist"])


def load_dsi(data_dir=DATA_DIR):
    """Load the raw daily index file."""
    return pd.read_parquet(Path(data_dir) / RAW_FILES["dsi"])


def data_source(data_dir=DATA_DIR):
    """Return "wrds" or "synthetic": where the files in data_dir came from.

    Tests that compare with the published paper only run on WRDS data.
    Returns None if nothing has been pulled yet.
    """
    marker = Path(data_dir) / SOURCE_MARKER_FILE
    return marker.read_text().strip() if marker.exists() else None


def pull_all():
    """Return the four raw tables, from WRDS or from the simulator."""
    if use_synthetic_data():
        import synthetic_data

        tables = synthetic_data.simulate_crsp()
        return {key: tables[key] for key in RAW_FILES}
    db = _connect()
    try:
        tables = {
            "dsedist": pull_dsedist(db),
            "dsenames": pull_dsenames(db),
            "dsedelist": pull_dsedelist(db),
            "dsi": pull_dsi(db),
        }
    finally:
        db.close()
    return tables


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for key, table in pull_all().items():
        table.to_parquet(DATA_DIR / RAW_FILES[key], index=False)
        print(f"Wrote {RAW_FILES[key]}: {len(table):,} rows")
    source = "synthetic" if use_synthetic_data() else "wrds"
    (DATA_DIR / SOURCE_MARKER_FILE).write_text(source + "\n")
