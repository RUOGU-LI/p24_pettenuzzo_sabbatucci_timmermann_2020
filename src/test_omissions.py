"""Unit tests for the extension: omission and initiation inference."""

import pandas as pd

import calc_dividend_growth as cdg
import calc_omissions as co

DATA_END = pd.Timestamp("2012-12-31")


def _names(permno=1, end="2030-01-01", shrcd=11, start="1990-01-01"):
    """Name-history row; `end` is the last date the firm is listed."""
    return {
        "permno": permno,
        "namedt": pd.Timestamp(start),
        "nameendt": pd.Timestamp(end),
        "shrcd": shrcd,
        "exchcd": 1,
        "comnam": f"FIRM {permno}",
    }


def _panel(dates_by_firm, dollars=100.0, regular=True):
    """Cleaned, matched announcements for {permno: [dates]}."""
    rows = [(p, pd.Timestamp(d)) for p, dates in dates_by_firm.items() for d in dates]
    events = pd.DataFrame(rows, columns=["permno", "dclrdt"])
    events["dollar_div"] = dollars
    events["dollar_div_regular"] = dollars if regular else 0.0
    events["has_regular"] = regular
    events["comnam"] = "FIRM " + events["permno"].astype(str)
    matched = cdg.match_prior_year(events)
    declarations = events[["permno", "dclrdt", "has_regular"]]
    return matched, declarations


QUARTERS = ["{}-02-15", "{}-05-15", "{}-08-15", "{}-11-15"]


def _years(*years):
    """Quarterly announcement dates for the given years."""
    return [q.format(y) for y in years for q in QUARTERS]


def test_steady_payer_has_no_omissions():
    """A firm that pays every quarter never triggers an omission."""
    matched, declarations = _panel({1: _years(2007, 2008, 2009, 2010)})
    exp = co.build_expectations(
        matched,
        declarations,
        pd.DataFrame([_names()]),
        data_end=pd.Timestamp("2010-12-31"),
    )
    assert len(exp) > 0
    assert exp["fulfilled"].all()
    assert len(co.infer_omissions(exp)) == 0


def test_suspension_creates_four_omissions_and_one_spell():
    """Paying through 2008 and stopping gives four missed payments in 2009."""
    matched, declarations = _panel({1: _years(2007, 2008)})
    exp = co.build_expectations(
        matched, declarations, pd.DataFrame([_names()]), DATA_END
    )
    omissions = co.infer_omissions(exp)
    assert len(omissions) == 4
    assert omissions["expected_date"].dt.year.eq(2009).all()
    assert omissions["is_spell_start"].sum() == 1
    assert omissions["dollars_prior"].eq(100.0).all()


def test_omission_is_dated_when_the_window_closes():
    """It becomes known OMISSION_LATE_DAYS after the expected date, not before."""
    matched, declarations = _panel({1: _years(2007, 2008)})
    exp = co.build_expectations(
        matched, declarations, pd.DataFrame([_names()]), DATA_END
    )
    omissions = co.infer_omissions(exp)
    lag = (omissions["detection_date"] - omissions["expected_date"]).dt.days
    assert lag.eq(co.OMISSION_LATE_DAYS).all()


def test_dividend_moved_by_two_months_is_not_an_omission():
    """Exelon-type case: from 2008 on, the board meets two months later.

    With a symmetric 45-day window the February 2008 expectation would be
    missed; with the late side extended it is met by the April declaration.
    """
    dates = _years(2006, 2007) + [
        "2008-04-15",
        "2008-07-15",
        "2008-10-15",
        "2009-01-15",
    ]
    matched, declarations = _panel({1: dates})
    exp = co.build_expectations(
        matched,
        declarations,
        pd.DataFrame([_names()]),
        data_end=pd.Timestamp("2009-03-31"),
    )
    assert len(exp) == 4
    assert len(co.infer_omissions(exp)) == 0


def test_dividend_more_than_a_quarter_late_is_an_omission():
    """Nothing from 45 days before to 100 days after the due date: an omission."""
    dates = _years(2006, 2007) + ["2008-02-15", "2008-09-15"]
    matched, declarations = _panel({1: dates})
    exp = co.build_expectations(
        matched,
        declarations,
        pd.DataFrame([_names()]),
        data_end=pd.Timestamp("2008-12-31"),
    )
    omissions = co.infer_omissions(exp)
    assert omissions["expected_date"].tolist() == [pd.Timestamp("2008-05-14")]


def test_delisted_firm_is_not_an_omission():
    """A firm acquired in early 2009 stops announcing, but it did not omit."""
    matched, declarations = _panel({1: _years(2007, 2008)})
    names = pd.DataFrame([_names(end="2009-01-31")])
    exp = co.build_expectations(matched, declarations, names, DATA_END)
    assert len(co.infer_omissions(exp)) == 0


def test_one_off_payer_creates_no_expectation():
    """Without a second year of payments the firm is not an established payer."""
    matched, declarations = _panel({1: _years(2008)})
    exp = co.build_expectations(
        matched, declarations, pd.DataFrame([_names()]), DATA_END
    )
    assert len(exp) == 0


def test_special_dividend_creates_no_expectation():
    """Nobody expects last year's special dividend to recur."""
    matched, declarations = _panel({1: ["2007-12-01", "2008-12-01"]}, regular=False)
    exp = co.build_expectations(
        matched, declarations, pd.DataFrame([_names()]), DATA_END
    )
    assert len(exp) == 0


def test_declaration_without_price_still_fulfils():
    """A dividend dropped from the growth series for lack of a price was still paid."""
    matched, declarations = _panel({1: _years(2007, 2008)})
    extra = pd.DataFrame(
        {"permno": 1, "dclrdt": pd.to_datetime(_years(2009)), "has_regular": True}
    )
    declarations = pd.concat([declarations, extra], ignore_index=True)
    exp = co.build_expectations(
        matched, declarations, pd.DataFrame([_names()]), DATA_END
    )
    assert exp["fulfilled"].all()


def test_unresolved_expectations_are_left_out():
    """Expectations whose window closes after the data end are not judged."""
    matched, declarations = _panel({1: _years(2007, 2008)})
    exp = co.build_expectations(
        matched,
        declarations,
        pd.DataFrame([_names()]),
        data_end=pd.Timestamp("2009-03-01"),
    )
    assert (exp["detection_date"] <= pd.Timestamp("2009-03-01")).all()
    assert len(co.infer_omissions(exp)) == 0


def test_initiation_requires_a_long_silence_and_a_long_listing():
    """A first dividend years after listing is an initiation; a recent IPO's is not."""
    matched, declarations = _panel({1: _years(2005), 2: _years(2005)})
    names = pd.DataFrame([_names(1, start="1990-01-01"), _names(2, start="2004-12-01")])
    init = co.infer_initiations(
        matched, declarations, names, data_start=pd.Timestamp("1990-01-01")
    )
    assert init["permno"].tolist() == [1]
    assert init["dclrdt"].iloc[0] == pd.Timestamp("2005-02-15")
    assert not init["is_resumption"].iloc[0]
