"""Tests for the time series plot, mostly its clock calibration input."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import xarray as xr  # noqa: E402

import rbrmoored as rbr  # noqa: E402

START = np.datetime64("2025-06-01T00:00:00")


@pytest.fixture
def solo():
    """Short synthetic solo record at 1 Hz."""
    n = 3600
    time = START + (np.arange(n) * int(1e9)).astype("timedelta64[ns]")
    data = xr.DataArray(
        4.0 + 0.01 * np.sin(2 * np.pi * np.arange(n) / n),
        coords={"time": time},
        dims=["time"],
        name="t",
    )
    data.attrs["units"] = "°C"
    data.attrs["long_name"] = "temperature"
    data.attrs["SN"] = 999999
    data.attrs["sampling period in s"] = 1.0
    data.attrs["time drift in ms"] = 0
    data.attrs["time offset applied"] = 0
    return data


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_plot_without_cal_time(solo):
    rbr.solo.plot(solo)
    assert len(plt.gcf().axes) == 1


def test_plot_with_one_cal_time(solo):
    rbr.solo.plot(solo, cal_time=START + np.timedelta64(600, "s"))
    assert len(plt.gcf().axes) == 2


def test_plot_with_two_cal_times(solo):
    cal = (START + np.timedelta64(600, "s"), START + np.timedelta64(1200, "s"))
    rbr.solo.plot(solo, cal_time=cal)
    assert len(plt.gcf().axes) == 3


def test_plot_does_not_clobber_cal_time_argument(solo):
    """The plotting loop must not rebind the cal_time argument."""
    cal = (START + np.timedelta64(600, "s"), START + np.timedelta64(1200, "s"))
    rbr.solo.plot(solo, cal_time=cal)
    assert cal == (START + np.timedelta64(600, "s"), START + np.timedelta64(1200, "s"))


@pytest.mark.parametrize(
    "cal_time",
    [
        "2025-06-01T00:10:00",
        ("2025-06-01T00:10:00", "2025-06-01T00:20:00"),
        (START + np.timedelta64(600, "s"),),
        (START, START, START),
        [START, START],
        42,
    ],
)
def test_plot_rejects_bad_cal_time(solo, cal_time):
    with pytest.raises(TypeError):
        rbr.solo.plot(solo, cal_time=cal_time)


def test_plot_returns_figure(solo):
    fig = rbr.solo.plot(solo)
    assert fig is plt.gcf()


@pytest.mark.parametrize(
    "drift_ms, offset_decimals, expected",
    [
        (-47998.0, 3, "time offset of -47.998 seconds applied"),
        (-48000.0, 3, "time offset of -48.000 seconds applied"),
        (-47998.0, 0, "time offset of -48 seconds applied"),
        (1234.0, 1, "time offset of 1.2 seconds applied"),
    ],
)
def test_plot_offset_label(solo, drift_ms, offset_decimals, expected):
    solo.attrs["time drift in ms"] = drift_ms
    solo.attrs["time offset applied"] = 1
    fig = rbr.solo.plot(solo, offset_decimals=offset_decimals)
    labels = [t.get_text() for t in fig.axes[0].texts]
    assert expected in labels
