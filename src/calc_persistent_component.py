"""Extract the persistent component of daily dividend growth.

Model (equations (4) and (11) of the paper, the "no-jump" benchmark)
---------------------------------------------------------------------
    observation:  d_t      = mu_t + e_t,                      e_t ~ N(0, sigma_d^2)
    state:        mu_{t+1} = mu_bar + phi (mu_t - mu_bar) + sigma_mu * eta_{t+1}

d_t is the log of daily bottom-up dividend growth. On days with no
announcement d_t is missing and the Kalman filter simply skips the update.
The four parameters are estimated by maximum likelihood.

Two versions
------------
"nojump"  The linear-Gaussian model above, estimated on every day with a
          growth rate. Large outliers on days with few announcers pull the
          state around: this is the spiky top panel of Figure 3.

"robust"  Our stand-in for the paper's full model with jumps and stochastic
          volatility (the course project allows any documented
          outlier-robust filter). Two changes, both fixed in settings.py:
          1. days with fewer than MIN_ANNOUNCERS_ROBUST matched announcers
             are treated as missing;
          2. measurement errors are Student-t with STUDENT_T_DOF degrees of
             freedom. In the filter this down-weights an observation by
             w_t = (nu + 1) / (nu + v_t^2 / F_t), where v_t is the forecast
             error and F_t the working forecast scale squared, so a huge surprise moves the state
             very little. This is the bottom panel of Figure 3.

Filtered versus smoothed
------------------------
Conditional on fixed parameter estimates, the filtered state uses
observations only up to day t, whereas the smoothed state uses the whole
estimation sample. Figures show the smoothed state; predictive regressions
use the filtered state to avoid look-ahead from state smoothing. However,
parameters estimated on the full update sample contain future information
relative to earlier dates. Only the post-2016 paper-sample extension uses
parameters frozen before the evaluation period.

Estimation samples
------------------
Parameters are estimated twice: on the paper's sample (to PAPER_END_DATE) and
on the full sample (to END_DATE). With the paper-sample parameters the filter
is also run forward to END_DATE; because the filter is causal, those values
after 2016 are a true out-of-sample extension with frozen parameters.
"""

import warnings
from math import lgamma, log, pi
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import calc_dividend_growth
from settings import (
    END_DATE,
    MIN_ANNOUNCERS_ROBUST,
    PAPER_END_DATE,
    START_DATE,
    STUDENT_T_DOF,
    config,
)

DATA_DIR = Path(config("DATA_DIR"))
STATE_FILE = "persistent_component.parquet"
PARAMS_FILE = "persistent_component_params.parquet"

PARAM_NAMES = ["mu_bar", "phi", "sigma_mu", "sigma_d"]
MODELS = ("nojump", "robust")
SAMPLES = {"paper": PAPER_END_DATE, "update": END_DATE}


def kalman_filter(y, mu_bar, phi, sigma_mu, sigma_d, nu=None):
    """Run the scalar Kalman filter for the AR(1)-plus-noise model.

    Parameters
    ----------
    y : array of float, NaN where no observation is available.
    nu : None for Gaussian measurement errors, or the Student-t degrees of
        freedom for the robust filter.
    sigma_d : measurement-error standard deviation in the Gaussian model.
        In the robust model it is a Student-t scale parameter, not a
        standard deviation. The approximate predictive likelihood treats
        p + r as the squared Student-t scale. For nu > 2, a Student-t
        standard deviation equals its scale times sqrt(nu / (nu - 2)),
        which is sqrt(2) times the scale when nu = 4.

    Returns a dict with the log-likelihood and the predicted and filtered
    state means and variances (all in levels of mu, not deviations).
    """
    n = len(y)
    q = sigma_mu**2
    r = sigma_d**2
    a_pred = np.empty(n)
    p_pred = np.empty(n)
    a_filt = np.empty(n)
    p_filt = np.empty(n)
    a = 0.0  # deviation of the state from mu_bar
    p = q / (1.0 - phi**2)
    loglik = 0.0
    if nu is not None:
        t_const = lgamma((nu + 1.0) / 2.0) - lgamma(nu / 2.0) - 0.5 * log(nu * pi)
    for t in range(n):
        a_pred[t] = a
        p_pred[t] = p
        yt = y[t]
        if not np.isnan(yt):
            v = yt - mu_bar - a
            f = p + r
            if nu is None:
                loglik += -0.5 * (log(2.0 * pi * f) + v * v / f)
                gain = p / f
            else:
                scaled = v * v / f
                loglik += (
                    t_const - 0.5 * log(f) - 0.5 * (nu + 1.0) * log(1.0 + scaled / nu)
                )
                weight = (nu + 1.0) / (nu + scaled)
                gain = p / (p + r / weight)
            a = a + gain * v
            p = p * (1.0 - gain)
        a_filt[t] = a
        p_filt[t] = p
        a = phi * a
        p = phi * phi * p + q
    return {
        "loglik": loglik,
        "pred_mean": a_pred + mu_bar,
        "pred_var": p_pred,
        "filt_mean": a_filt + mu_bar,
        "filt_var": p_filt,
    }


def kalman_smoother(filter_output, mu_bar, phi):
    """Rauch-Tung-Striebel smoother for the output of kalman_filter."""
    a_pred = filter_output["pred_mean"] - mu_bar
    a_filt = filter_output["filt_mean"] - mu_bar
    p_pred = filter_output["pred_var"]
    p_filt = filter_output["filt_var"]
    n = len(a_filt)
    a_smooth = a_filt.copy()
    for t in range(n - 2, -1, -1):
        j = p_filt[t] * phi / p_pred[t + 1]
        a_smooth[t] = a_filt[t] + j * (a_smooth[t + 1] - a_pred[t + 1])
    return a_smooth + mu_bar


def _unpack(theta):
    """Map the unconstrained optimiser vector to model parameters."""
    mu_bar = theta[0]
    phi = 1.0 - np.exp(theta[1])
    sigma_mu = np.exp(theta[2])
    sigma_d = np.exp(theta[3])
    return mu_bar, phi, sigma_mu, sigma_d


def fit_state_space(y, nu=None):
    """Estimate (mu_bar, phi, sigma_mu, sigma_d) by maximum likelihood.

    Returns parameter estimates, loglik, n_obs and a converged flag.
    """
    y = np.asarray(y, dtype=float)
    obs = y[~np.isnan(y)]
    if len(obs) < 50:
        raise ValueError("Too few observations to estimate the state-space model.")
    center = float(np.median(obs))
    spread = float(1.4826 * np.median(np.abs(obs - center))) or float(obs.std())
    theta0 = np.array([center, np.log(0.01), np.log(0.005), np.log(spread)])
    bounds = [
        (center - 1.0, center + 1.0),
        (np.log(1e-5), np.log(0.5)),  # phi between 0.5 and 0.99999
        (np.log(1e-5), np.log(1.0)),
        (np.log(1e-4), np.log(10.0)),
    ]

    def negative_loglik(theta):
        return -kalman_filter(y, *_unpack(theta), nu=nu)["loglik"]

    result = minimize(negative_loglik, theta0, method="L-BFGS-B", bounds=bounds)
    estimates = dict(zip(PARAM_NAMES, (float(x) for x in _unpack(result.x))))
    estimates["loglik"] = float(-result.fun)
    estimates["n_obs"] = len(obs)
    estimates["converged"] = bool(result.success)

    if not result.success:
        warnings.warn(
            f"State-space optimization did not converge: {result.message}",
            RuntimeWarning,
            stacklevel=2,
        )

    phi_upper = 1.0 - np.exp(bounds[1][0])
    if np.isclose(estimates["phi"], phi_upper, rtol=0.0, atol=1e-8):
        warnings.warn(
            f"Estimated phi reached its upper bound ({phi_upper:.5f}).",
            RuntimeWarning,
            stacklevel=2,
        )

    return estimates


def prepare_observations(daily, model, value_col="log_growth", count_col="n_matched"):
    """Return the observation series for a model ("nojump" or "robust")."""
    y = daily[value_col].to_numpy(dtype=float).copy()
    if model == "robust":
        y[daily[count_col].to_numpy() < MIN_ANNOUNCERS_ROBUST] = np.nan
    elif model != "nojump":
        raise ValueError(f"Unknown model: {model}")
    return y


def estimate_persistent_component(
    daily, value_col="log_growth", count_col="n_matched", models=MODELS
):
    """Estimate every model on every estimation sample.

    Parameters
    ----------
    daily : DataFrame with date, value_col and count_col on trading days.

    Returns
    -------
    states : long DataFrame with columns date, model, sample, mu_filtered,
        mu_smoothed. For sample="paper", mu_filtered runs to END_DATE with
        parameters frozen at PAPER_END_DATE and mu_smoothed stops at
        PAPER_END_DATE.
    params : DataFrame with one row per (model, sample, parameter)
        and a boolean converged column for the corresponding fit.
    """
    daily = daily[(daily["date"] >= START_DATE) & (daily["date"] <= END_DATE)]
    daily = daily.sort_values("date").reset_index(drop=True)
    states, params = [], []
    for model in models:
        nu = STUDENT_T_DOF if model == "robust" else None
        y = prepare_observations(daily, model, value_col, count_col)
        for sample, sample_end in SAMPLES.items():
            in_sample = (daily["date"] <= sample_end).to_numpy()
            if in_sample.sum() == 0:
                continue
            est = fit_state_space(y[in_sample], nu=nu)
            converged = est.pop("converged")
            theta = [est[name] for name in PARAM_NAMES]
            filtered = kalman_filter(y, *theta, nu=nu)["filt_mean"]
            smoothed = np.full(len(y), np.nan)
            inside = kalman_filter(y[in_sample], *theta, nu=nu)
            smoothed[in_sample] = kalman_smoother(inside, est["mu_bar"], est["phi"])
            states.append(
                pd.DataFrame(
                    {
                        "date": daily["date"],
                        "model": model,
                        "sample": sample,
                        "mu_filtered": filtered,
                        "mu_smoothed": smoothed,
                    }
                )
            )
            for name, value in est.items():
                params.append(
                    {
                        "model": model,
                        "sample": sample,
                        "parameter": name,
                        "value": value,
                        "converged": converged,
                    }
                )
    return pd.concat(states, ignore_index=True), pd.DataFrame(params)


def select_state(states, model, sample, kind="filtered"):
    """Return one state series (indexed by date) from the long table."""
    sub = states[(states["model"] == model) & (states["sample"] == sample)]
    return sub.set_index("date")[f"mu_{kind}"]


def load_states(data_dir=DATA_DIR):
    """Load the long table of state estimates."""
    return pd.read_parquet(Path(data_dir) / STATE_FILE)


def load_params(data_dir=DATA_DIR):
    """Load the parameter estimates."""
    return pd.read_parquet(Path(data_dir) / PARAMS_FILE)


if __name__ == "__main__":
    daily = calc_dividend_growth.load_daily_growth(DATA_DIR)
    states, params = estimate_persistent_component(daily)
    states.to_parquet(DATA_DIR / STATE_FILE, index=False)
    params.to_parquet(DATA_DIR / PARAMS_FILE, index=False)
    print(params.pivot(index=["model", "sample"], columns="parameter", values="value"))
