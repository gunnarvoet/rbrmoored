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


def test_subtract_restores_the_spiked_samples():
    spiked, index = make_record(2.0, spike_amplitude=1.25e-3, noise=1e-4)
    plain, _ = make_record(2.0, spike_amplitude=0.0, noise=1e-4)
    clean = rbr.solo.spike_removal(spiked, method="subtract", n_tail=0, verbose=False)
    assert clean.attrs["despiking applied"] == 1
    assert clean.attrs["spike removal method"] == "subtract"
    inner = index[(index > 1) & (index < spiked.size - 2)]
    assert np.abs(clean.data[inner] - plain.data[inner]).max() < 5e-5
    untouched = np.setdiff1d(np.arange(spiked.size), index)
    np.testing.assert_array_equal(clean.data[untouched], spiked.data[untouched])


def test_subtract_removes_the_tail_sample():
    spiked, index = make_record(2.0, spike_amplitude=1.25e-3, noise=1e-4)
    plain, _ = make_record(2.0, spike_amplitude=0.0, noise=1e-4)
    tail = index[index + 1 < spiked.size] + 1
    spiked.data[tail] += 2.5e-5
    clean = rbr.solo.spike_removal(spiked, method="subtract", n_tail=1, verbose=False)
    inner = tail[(tail > 2) & (tail < spiked.size - 3)]
    assert np.abs(clean.data[inner] - plain.data[inner]).max() < 5e-5
    assert np.allclose(clean.attrs["spike template in K"], [1.25e-3, 2.5e-5], atol=4e-5)


def test_subtract_follows_an_amplitude_that_changes_between_blocks():
    spiked, index = make_record(2.0, minutes=480, spike_amplitude=1.0e-3, noise=1e-4)
    plain, _ = make_record(2.0, minutes=480, spike_amplitude=0.0, noise=1e-4)
    late = index[index >= spiked.size // 2]
    spiked.data[late] += 0.2e-3
    clean = rbr.solo.spike_removal(
        spiked, method="subtract", n_tail=0, block_s=4 * 3600.0, verbose=False
    )
    inner = index[(index > 1) & (index < spiked.size - 2)]
    assert np.abs(clean.data[inner] - plain.data[inner]).max() < 6e-5


def test_subtract_short_block_falls_back_to_the_record_template():
    # 250 min record, 4 h blocks: the last block holds 10 cycles, below min_cycles.
    # Its spikes are 0.3 mK larger, so the record-wide template leaves that excess behind.
    spiked, index = make_record(2.0, minutes=250, spike_amplitude=1.25e-3, noise=1e-4)
    plain, _ = make_record(2.0, minutes=250, spike_amplitude=0.0, noise=1e-4)
    last = index[(index >= 240 * 30) & (index < spiked.size - 2)]
    assert last.size == 10
    spiked.data[last] += 3e-4
    clean = rbr.solo.spike_removal(
        spiked, method="subtract", n_tail=0, block_s=4 * 3600.0, verbose=False
    )
    assert np.abs(clean.data[last] - plain.data[last] - 3e-4).max() < 5e-5


def test_subtract_leaves_a_spike_on_the_first_sample_alone():
    spiked, index = make_record(
        2.0, spike_amplitude=1.25e-3, noise=1e-4, spike_phase_s=0.0
    )
    assert index[0] == 0
    clean = rbr.solo.spike_removal(spiked, method="subtract", n_tail=0, verbose=False)
    assert clean.data[0] == spiked.data[0]
    assert clean.attrs["spikes not removed"] == 1


def test_subtract_leaves_a_spike_next_to_a_gap_alone():
    spiked, index = make_record(2.0, spike_amplitude=1.25e-3, noise=1e-4)
    # drop the sample after the tenth spike, so that spike has no right-hand support
    keep = np.ones(spiked.size, bool)
    keep[index[10] + 1] = False
    gappy = spiked.isel(time=keep)
    clean = rbr.solo.spike_removal(gappy, method="subtract", n_tail=0, verbose=False)
    at_gap = np.flatnonzero(gappy.time.data == spiked.time.data[index[10]])[0]
    assert clean.data[at_gap] == gappy.data[at_gap]
    assert clean.attrs["spikes not removed"] >= 1


def test_subtract_leaves_record_without_spike_untouched():
    plain, _ = make_record(2.0, spike_amplitude=0.0)
    clean = rbr.solo.spike_removal(plain, method="subtract", verbose=False)
    np.testing.assert_array_equal(clean.data, plain.data)
    assert clean.attrs["despiking applied"] == 0


def test_models_none_applies_to_an_older_solo_with_another_period():
    n = 32 * 400
    time = np.datetime64("2025-06-01") + (np.arange(n) * 2_000_000_000).astype(
        "timedelta64[ns]"
    )
    rng = np.random.default_rng(3)
    values = 1.4 + rng.normal(0.0, 1e-4, n)
    plain = values.copy()
    index = np.arange(12, n, 32)
    values[index] -= 1.7e-4
    solo = xr.DataArray(values, coords={"time": time}, dims=["time"], name="t")
    solo.attrs.update({"model": "RBRsolo", "sampling period in s": 2.0})
    assert rbr.solo.spike_removal(solo, period_s=64.0, verbose=False) is solo
    clean = rbr.solo.spike_removal(
        solo, period_s=64.0, method="subtract", n_tail=0, models=None, verbose=False
    )
    inner = index[(index > 1) & (index < n - 2)]
    assert np.abs(clean.data[inner] - plain[inner]).max() < 5e-5


def test_unknown_method_raises():
    spiked, _ = make_record(2.0, spike_amplitude=1.25e-3)
    with pytest.raises(ValueError, match="method"):
        rbr.solo.spike_removal(spiked, method="median", verbose=False)
