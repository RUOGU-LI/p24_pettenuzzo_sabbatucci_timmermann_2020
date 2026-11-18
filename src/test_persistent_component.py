"""Unit tests for the state-space filter (calc_persistent_component.py)."""

import numpy as np
import pytest

import calc_persistent_component as cpc


def _simulate(n=4000, mu_bar=0.08, phi=0.98, sigma_mu=0.01, sigma_d=0.05, seed=0):
    """Simulate the AR(1)-plus-noise model. Returns (observations, state)."""
    rng = np.random.default_rng(seed)
    state = np.empty(n)
    state[0] = mu_bar
    for t in range(1, n):
        state[t] = mu_bar + phi * (state[t - 1] - mu_bar) + sigma_mu * rng.normal()
    return state + sigma_d * rng.normal(size=n), state


def test_maximum_likelihood_recovers_parameters():
    """On data simulated from the model, the estimates are close to the truth."""
    y, _ = _simulate()
    est = cpc.fit_state_space(y)
    assert est["mu_bar"] == pytest.approx(0.08, abs=0.03)
    assert est["phi"] == pytest.approx(0.98, abs=0.01)
    assert est["sigma_mu"] == pytest.approx(0.01, rel=0.25)
    assert est["sigma_d"] == pytest.approx(0.05, rel=0.10)


def test_filter_tracks_the_true_state():
    """The filtered state is far closer to the truth than the raw observations."""
    y, state = _simulate()
    out = cpc.kalman_filter(y, 0.08, 0.98, 0.01, 0.05)
    error_filtered = np.sqrt(np.mean((out["filt_mean"] - state) ** 2))
    error_raw = np.sqrt(np.mean((y - state) ** 2))
    assert error_filtered < 0.6 * error_raw


def test_missing_observations_are_skipped():
    """With no data the state decays to its mean and its variance grows."""
    y = np.full(200, np.nan)
    y[0] = 0.5
    out = cpc.kalman_filter(y, 0.0, 0.9, 0.01, 0.05)
    assert abs(out["filt_mean"][-1]) < 1e-6
    assert out["filt_var"][-1] > out["filt_var"][0]
    assert np.isfinite(out["loglik"])


def test_smoother_is_closer_to_truth_than_filter():
    """Using the whole sample cannot be worse than using the past only."""
    y, state = _simulate(seed=1)
    out = cpc.kalman_filter(y, 0.08, 0.98, 0.01, 0.05)
    smooth = cpc.kalman_smoother(out, 0.08, 0.98)
    assert np.mean((smooth - state) ** 2) < np.mean((out["filt_mean"] - state) ** 2)


def test_student_t_filter_resists_an_outlier():
    """One absurd observation moves the Gaussian filter much more than the robust one.

    This is the property that turns the spiky top panel of Figure 3 into the
    smooth bottom panel.
    """
    y, _ = _simulate(n=500, seed=2)
    y_outlier = y.copy()
    y_outlier[250] += 5.0
    params = (0.08, 0.98, 0.01, 0.05)
    jump_gauss = (
        cpc.kalman_filter(y_outlier, *params)["filt_mean"][250]
        - cpc.kalman_filter(y, *params)["filt_mean"][250]
    )
    jump_robust = (
        cpc.kalman_filter(y_outlier, *params, nu=4.0)["filt_mean"][250]
        - cpc.kalman_filter(y, *params, nu=4.0)["filt_mean"][250]
    )
    assert abs(jump_robust) < 0.05 * abs(jump_gauss)


def test_filtered_state_does_not_use_the_future():
    """Changing later observations must not change earlier filtered values."""
    y, _ = _simulate(n=300, seed=3)
    y_changed = y.copy()
    y_changed[200:] += 1.0
    params = (0.08, 0.98, 0.01, 0.05)
    first = cpc.kalman_filter(y, *params, nu=4.0)["filt_mean"][:200]
    second = cpc.kalman_filter(y_changed, *params, nu=4.0)["filt_mean"][:200]
    assert np.array_equal(first, second)
