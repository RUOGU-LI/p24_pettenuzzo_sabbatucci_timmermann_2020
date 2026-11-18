"""Turn raw FRED series into tidy quarterly GDP and consumption, real and nominal.

Consumption follows the paper: nondurable goods plus services (NIPA Table
2.3.5).

- Nominal consumption is the sum of the two current-dollar series, exactly as
  they appear in Table 2.3.5.
- Real consumption deflates each series by its own chain-type price index and
  adds them. Adding chained components is an approximation, but growth rates
  are what we use and they are barely affected.

Both versions are kept because the paper does not say whether its growth
rates are real or nominal, and its dividend growth measure is nominal. The
implicit GDP deflator (nominal over real GDP) is kept too: its growth is
inflation, which separates the real from the nominal part of the result.

Output: one row per quarter, dated at the quarter end, with levels only.
Growth rates are computed in the analysis step so that the sample start is
chosen in one place.
"""

from pathlib import Path

import pandas as pd

import pull_fred
from settings import config

DATA_DIR = Path(config("DATA_DIR"))
MACRO_FILE = "clean_macro_quarterly.parquet"


def clean_macro(fred):
    """Return quarterly GDP and nondurables-plus-services consumption, real and nominal."""
    df = fred.dropna(how="all").copy()
    real_nondurables = 100.0 * df["PCND"] / df["DNDGRG3Q086SBEA"]
    real_services = 100.0 * df["PCESV"] / df["DSERRG3Q086SBEA"]
    out = pd.DataFrame(
        {
            "real_gdp": df["GDPC1"],
            "real_consumption": real_nondurables + real_services,
            "nominal_gdp": df["GDP"],
            "nominal_consumption": df["PCND"] + df["PCESV"],
            "gdp_deflator": 100.0 * df["GDP"] / df["GDPC1"],
        }
    ).dropna()
    # FRED dates quarterly data at the first day of the quarter.
    out.index = pd.PeriodIndex(out.index, freq="Q").to_timestamp(how="end").normalize()
    out.index.name = "date"
    return out.reset_index()


def load_macro(data_dir=DATA_DIR):
    """Load the tidy quarterly macro table."""
    return pd.read_parquet(Path(data_dir) / MACRO_FILE)


if __name__ == "__main__":
    macro = clean_macro(pull_fred.load_fred(DATA_DIR))
    macro.to_parquet(DATA_DIR / MACRO_FILE, index=False)
    print(f"Wrote {MACRO_FILE}: {len(macro):,} quarters")
