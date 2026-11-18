"""Pull quarterly macro series from FRED.

Series
------
- GDPC1            real GDP (chained dollars, seasonally adjusted annual rate)
- GDP              nominal GDP (current dollars, seasonally adjusted annual rate)
- PCND, PCESV      nominal personal consumption expenditures on nondurable
                   goods and on services. These are the lines of BEA NIPA
                   Table 2.3.5 that the paper uses; FRED mirrors them.
- DNDGRG3Q086SBEA  chain-type price index, nondurable goods
- DSERRG3Q086SBEA  chain-type price index, services

The price indexes are needed to turn nominal consumption into real
consumption in the cleaning step. Both nominal and real series are kept: the
paper does not say which it uses, and NIPA Table 2.3.5 is in current dollars.
No API key is required.

When USE_SYNTHETIC_DATA=1 the series are simulated (see synthetic_data.py).
"""

from pathlib import Path

import pandas as pd

from settings import END_DATE, PULL_START_DATE, config, use_synthetic_data

DATA_DIR = Path(config("DATA_DIR"))
FRED_FILE = "fred_macro.parquet"

SERIES = {
    "GDPC1": "Real Gross Domestic Product",
    "GDP": "Gross Domestic Product (nominal)",
    "PCND": "Personal Consumption Expenditures: Nondurable Goods",
    "PCESV": "Personal Consumption Expenditures: Services",
    "DNDGRG3Q086SBEA": "PCE: Nondurable goods (chain-type price index)",
    "DSERRG3Q086SBEA": "PCE: Services (chain-type price index)",
}


def pull_fred(start_date=PULL_START_DATE, end_date=END_DATE):
    """Download the series in SERIES and return one quarterly DataFrame."""
    if use_synthetic_data():
        import synthetic_data

        return synthetic_data.simulate_crsp()["fred"]
    import pandas_datareader.data as web

    df = web.DataReader(list(SERIES), "fred", start_date, end_date)
    df.index.name = "date"
    return df


def load_fred(data_dir=DATA_DIR):
    """Load the raw FRED series (index: quarter start date)."""
    return pd.read_parquet(Path(data_dir) / FRED_FILE)


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = pull_fred()
    df.to_parquet(DATA_DIR / FRED_FILE)
    print(f"Wrote {FRED_FILE}: {len(df):,} rows")
