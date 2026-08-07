"""Tests for the periodic RBRsolo³ spike removal."""

import numpy as np
import pytest
import xarray as xr

import rbrmoored as rbr

SPIKE_PERIOD_S = 60.0


def make_record(
    dt,
    minutes=240,
    spike_amplitude=0.0,
    spike_phase_s=30.0,
    noise=5e-4,
    model="RBRsolo³",
    seed=1,
):
    """Synthetic solo record with an optional periodic single-sample spike.

    Parameters
    ----------
    dt : float
        Sampling period in seconds.
    minutes : int, optional
        Record length in minutes.
    spike_amplitude : float, optional
        Amplitude of the periodic spike in K. No spike if zero (default).
    spike_phase_s : float, optional
        Position of the spike within the 60 s cycle.
    noise : float, optional
        Standard deviation of the white noise in K.
    model : str, optional
        Instrument model written to the attributes.
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    solo : xarray.DataArray
        Synthetic record.
    spike_index : numpy.ndarray
        Indices of the samples the spike was added to.
    """
    n = int(minutes * 60 / dt)
    time = np.datetime64("2025-06-01T00:00:00") + (np.arange(n) * int(dt * 1e9)).astype(
        "timedelta64[ns]"
    )
    rng = np.random.default_rng(seed)
    # smooth background plus white noise
    values = (
        4.0
        + 0.02 * np.sin(2 * np.pi * np.arange(n) * dt / 3600)
        + rng.normal(0.0, noise, n)
    )
    spike_index = np.arange(round(spike_phase_s / dt), n, round(SPIKE_PERIOD_S / dt))
    if spike_amplitude:
        values[spike_index] += spike_amplitude

    solo = xr.DataArray(values, coords={"time": time}, dims=["time"], name="t")
    solo.attrs["units"] = "°C"
    solo.attrs["long_name"] = "temperature"
    solo.attrs["SN"] = 999999
    solo.attrs["model"] = model
    solo.attrs["sampling period in s"] = dt
    solo.attrs["time offset applied"] = 0
    return solo, spike_index


def changed_index(before, after):
    """Indices where spike removal modified the data."""
    return np.flatnonzero(~np.isclose(before.data, after.data, rtol=0, atol=1e-12))


@pytest.mark.parametrize("dt", [0.5, 1.0, 2.0])
def test_spike_removal_removes_one_sample_per_cycle(dt):
    solo, spike_index = make_record(dt, spike_amplitude=1.3e-3)
    clean = rbr.solo.spike_removal(solo)
    assert changed_index(solo, clean).size == spike_index.size


@pytest.mark.parametrize("dt", [0.5, 1.0, 2.0])
def test_spike_removal_targets_the_spiked_samples(dt):
    solo, spike_index = make_record(dt, spike_amplitude=1.3e-3)
    clean = rbr.solo.spike_removal(solo)
    np.testing.assert_array_equal(changed_index(solo, clean), spike_index)


def test_spike_removal_finds_the_spike_phase():
    solo, _ = make_record(0.5, spike_amplitude=1.3e-3, spike_phase_s=17.5)
    clean = rbr.solo.spike_removal(solo)
    assert clean.attrs["spike phase in s"] == pytest.approx(17.5)


def test_spike_removal_reports_the_spike_amplitude():
    solo, _ = make_record(0.5, spike_amplitude=1.3e-3)
    clean = rbr.solo.spike_removal(solo)
    assert clean.attrs["spike amplitude in K"] == pytest.approx(1.3e-3, rel=0.2)


def test_spike_removal_leaves_record_without_spike_untouched():
    solo, _ = make_record(0.5, spike_amplitude=0.0)
    clean = rbr.solo.spike_removal(solo)
    np.testing.assert_array_equal(clean.data, solo.data)


def test_spike_removal_flags_record_without_spike_as_not_despiked():
    solo, _ = make_record(0.5, spike_amplitude=0.0)
    clean = rbr.solo.spike_removal(solo)
    assert clean.attrs["despiking applied"] == 0


def line_to_background(da, harmonic=1):
    """Power at the spike frequency relative to the surrounding background."""
    dt = da.attrs["sampling period in s"]
    values = da.data - da.data.mean()
    power = np.abs(np.fft.rfft(values)) ** 2
    freq = np.fft.rfftfreq(values.size, d=dt)
    k = int(np.argmin(np.abs(freq - harmonic / SPIKE_PERIOD_S)))
    background = np.median(power[max(1, k - 200) : k + 200])
    return power[k] / background


def test_spike_removal_reduces_variance_at_the_spike_frequency():
    solo, _ = make_record(0.5, spike_amplitude=1.3e-3)
    clean = rbr.solo.spike_removal(solo)
    for harmonic in [1, 2, 3]:
        before = line_to_background(solo, harmonic)
        after = line_to_background(clean, harmonic)
        assert before > 10
        assert after < before / 4


def test_spike_removal_ignores_older_solo_model():
    solo, _ = make_record(0.5, spike_amplitude=1.3e-3, model="RBRsolo")
    clean = rbr.solo.spike_removal(solo)
    np.testing.assert_array_equal(clean.data, solo.data)


def test_spike_removal_is_not_applied_twice():
    solo, _ = make_record(0.5, spike_amplitude=1.3e-3)
    once = rbr.solo.spike_removal(solo)
    twice = rbr.solo.spike_removal(once)
    np.testing.assert_array_equal(twice.data, once.data)


def test_spike_removal_preserves_coordinates_and_attributes():
    solo, _ = make_record(0.5, spike_amplitude=1.3e-3)
    clean = rbr.solo.spike_removal(solo)
    np.testing.assert_array_equal(clean.time.data, solo.time.data)
    assert clean.attrs["SN"] == solo.attrs["SN"]
    assert clean.attrs["units"] == solo.attrs["units"]


def test_spike_removal_requires_sampling_period():
    solo, _ = make_record(0.5, spike_amplitude=1.3e-3)
    del solo.attrs["sampling period in s"]
    with pytest.raises(KeyError):
        rbr.solo.spike_removal(solo)
