"""Unit tests for the dividend cleaning step (clean_dividends.py)."""

import pandas as pd

import clean_dividends


def _distribution(
    permno=1,
    distcd=1232,
    divamt=0.5,
    dclrdt="2014-05-15",
    price_date="2014-05-15",
    prc=100.0,
    shrout=1000.0,
):
    """One raw distribution row in the layout of the WRDS pull."""
    return {
        "permno": permno,
        "distcd": distcd,
        "divamt": divamt,
        "dclrdt": pd.Timestamp(dclrdt),
        "exdt": pd.Timestamp(dclrdt),
        "rcrddt": pd.Timestamp(dclrdt),
        "paydt": pd.Timestamp(dclrdt),
        "price_date": pd.Timestamp(price_date) if price_date else pd.NaT,
        "prc": prc,
        "shrout": shrout,
    }


def _names(permno=1, shrcd=11, exchcd=1):
    """A name-history row that covers every date used in the tests."""
    return {
        "permno": permno,
        "namedt": pd.Timestamp("1960-01-01"),
        "nameendt": pd.Timestamp("2030-01-01"),
        "shrcd": shrcd,
        "exchcd": exchcd,
        "comnam": f"FIRM {permno}",
    }


def test_dollar_dividend_uses_shares_in_thousands():
    """$0.50 per share on 1,000 thousand shares is $500,000."""
    raw = pd.DataFrame([_distribution(divamt=0.5, shrout=1000.0)])
    events, _, _ = clean_dividends.clean_dividend_events(raw, pd.DataFrame([_names()]))
    assert events.loc[0, "dollar_div"] == 500_000.0


def test_keeps_latest_price_row_per_distribution():
    """The pull returns one row per look-back day; the latest one must win."""
    raw = pd.DataFrame(
        [
            _distribution(price_date="2014-05-14", shrout=900.0),
            _distribution(price_date="2014-05-15", shrout=1000.0),
        ]
    )
    events, _, counts = clean_dividends.clean_dividend_events(
        raw, pd.DataFrame([_names()])
    )
    assert len(events) == 1
    assert events.loc[0, "shrout"] == 1000.0
    assert counts.set_index("step").loc["One row per distribution", "rows"] == 1


def test_same_day_dividends_are_summed():
    """The paper's PPL example: 39 and 21 cents on one day become one dividend."""
    raw = pd.DataFrame(
        [
            _distribution(divamt=0.39, dclrdt="1983-11-23", price_date="1983-11-23"),
            _distribution(divamt=0.21, dclrdt="1983-11-23", price_date="1983-11-23"),
        ]
    )
    events, _, _ = clean_dividends.clean_dividend_events(raw, pd.DataFrame([_names()]))
    assert len(events) == 1
    assert events.loc[0, "n_distributions"] == 2
    assert abs(events.loc[0, "divamt"] - 0.60) < 1e-12


def test_non_common_and_other_exchanges_are_dropped():
    """ADRs (share code 31) and other exchanges do not enter the sample."""
    raw = pd.DataFrame([_distribution(permno=p) for p in (1, 2, 3)])
    names = pd.DataFrame([_names(1), _names(2, shrcd=31), _names(3, exchcd=4)])
    events, _, _ = clean_dividends.clean_dividend_events(raw, names)
    assert events["permno"].tolist() == [1]


def test_missing_price_is_dropped_from_events_but_kept_as_declaration():
    """No price means no dollar amount, but the dividend was still declared."""
    raw = pd.DataFrame(
        [_distribution(price_date=None, prc=float("nan"), shrout=float("nan"))]
    )
    events, declarations, _ = clean_dividends.clean_dividend_events(
        raw, pd.DataFrame([_names()])
    )
    assert len(events) == 0
    assert len(declarations) == 1


def test_special_dividend_is_not_regular():
    """Code 1272 (extra or special) is flagged as irregular."""
    raw = pd.DataFrame([_distribution(distcd=1272)])
    events, declarations, _ = clean_dividends.clean_dividend_events(
        raw, pd.DataFrame([_names()])
    )
    assert not events.loc[0, "has_regular"]
    assert not declarations.loc[0, "has_regular"]
