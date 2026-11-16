"""Numbers transcribed by hand from the published papers.

This file has no pipeline imports. It is the fixed target that replication
and validation tests compare against, so nothing computed by our code can
leak into it.

Sources
-------
- PST2020: Pettenuzzo, Sabbatucci, and Timmermann (2020), "Cash Flow News and
  Stock Price Dynamics", Journal of Finance 75(4), 2221-2270.
- PST2023: Pettenuzzo, Sabbatucci, and Timmermann (2023), "Dividend
  suspensions and cash flows during the Covid-19 pandemic: A dynamic
  econometric model", Journal of Econometrics 235(2), 1522-1541, and its
  working-paper version (CEPR DP14921, June 2020).
"""

## ---------------------------------------------------------------------
## PST2020, Section I: sample size (footnote 10)
## ---------------------------------------------------------------------
N_DECLARED_DIVIDENDS_POST_1964 = 402_115

## ---------------------------------------------------------------------
## PST2020, Figure 1 (2014Q2) checkpoints
## ---------------------------------------------------------------------
FIGURE1 = {
    # Largest dollar amount announced on one day in 2014Q2 (footnote 14).
    "2014-04-24": {"dollars": 7.12e9, "n_announcers": 68},
    # A single announcer: Costco, $155.6m against $135.3m a year earlier.
    "2014-06-22": {
        "dollars": 155.6e6,
        "dollars_prior": 135.3e6,
        "gross_growth": 1.15,
        "n_announcers": 1,
    },
    # Smallest amount in the quarter.
    "2014-06-30": {"dollars": 3.6e6},
}

## ---------------------------------------------------------------------
## PST2020, Table I: posterior means of the full model, 1973-2016.
## Our simplified filter is not expected to match these; they are sanity
## targets for the order of magnitude of the persistent component.
## ---------------------------------------------------------------------
TABLE1_1973_2016 = {
    "mu_d": 0.080,
    "phi_mu": 0.998,
    "sigma_mu": 0.002,
    "mu_h": -5.337,
    "phi_h": 0.833,
    "sigma_h": 0.752,
    "sigma_xi": 2.761,
    "lambda_1": -1.354,
    "lambda_2": -0.024,
}

## ---------------------------------------------------------------------
## PST2020, Table II, Panel A: CRSP dividend growth on the lagged persistent
## component, the dividend-price ratio and three lags of dividend growth.
## Newey-West t-statistics with three lags.
## ---------------------------------------------------------------------
TABLE2_PANEL_A = {
    ("quarterly", "1973-2016"): {
        "beta_mu": 0.36,
        "t_mu": 4.58,
        "beta_dp": -0.01,
        "r2": 0.2650,
        "n_obs": 169,
    },
    ("annual", "1973-2016"): {
        "beta_mu": 2.50,
        "t_mu": 3.95,
        "beta_dp": 0.03,
        "r2": 0.2745,
        "n_obs": 40,
    },
}

## ---------------------------------------------------------------------
## PST2020, Table IV, Panel A: quarterly GDP and consumption growth on the
## lagged persistent component, 1973-2015, 171 observations.
## ---------------------------------------------------------------------
TABLE4_PANEL_A = {
    ("gdp", "no_lag"): {"beta_mu": 0.15, "t_mu": 5.68, "r2": 0.2352, "n_obs": 171},
    ("gdp", "with_lag"): {
        "beta_mu": 0.09,
        "t_mu": 3.30,
        "beta_lag": 0.37,
        "r2": 0.3356,
        "n_obs": 171,
    },
    ("consumption", "no_lag"): {
        "beta_mu": 0.15,
        "t_mu": 6.58,
        "r2": 0.3108,
        "n_obs": 171,
    },
    ("consumption", "with_lag"): {
        "beta_mu": 0.04,
        "t_mu": 2.84,
        "beta_lag": 0.67,
        "r2": 0.5997,
        "n_obs": 171,
    },
}

## ---------------------------------------------------------------------
## PST2020, Figure 3 (bottom panel), qualitative benchmark from the course
## project description: a smooth series roughly between 0 and 0.15 that is
## negative only in 2008-2009.
## ---------------------------------------------------------------------
FIGURE3_NEGATIVE_YEARS = (2008, 2009)

## ---------------------------------------------------------------------
## PST2023: dividend suspensions identified from Capital IQ, 8-K filings and
## press releases. Used only to validate our CRSP-inferred omissions. The
## definitions differ (announced suspensions versus inferred missed
## payments), so we compare orders of magnitude and timing, not equality.
## ---------------------------------------------------------------------
PST2023_SUSPENDERS = {
    # Published version: counts since January 2005, data through Sept 2020.
    "2005_to_2020Q3_total": 472,
    "2020_jan_to_sep": 198,
    # Working-paper version (data through April 2020).
    "2008": 78,
    "2009": 76,
}
