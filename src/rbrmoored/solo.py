#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Read and process RBR Solo thermistor data.
- Read data in .rsk format into an `xarray.DataArray` with `read`.
- `proc` combines processing steps.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.ndimage import median_filter

import pyrsktools

import rbrmoored as rbr


def proc(
    solofile,
    data_out=None,
    apply_time_offset=True,
    apply_spike_removal=True,
    figure_out=None,
    show_plot=True,
    cal_time=None,
    max_time_drift=4.0,
    offset_to_utc=0,
    offset_time_drift=0,
):
    """Combine RBR Solo processing steps.

    The following processing steps are combined here:
    - Read data in .rsk format
    - Apply time offset
    - Save to netcdf
    - Plot

    Parameters
    ----------
    solofile : path object
        Path to solo file
    data_out : path object, optional
        Path to data output directory. Processed data will only be saved to
        netcdf format if `data_out` is provided. Default None.
    apply_time_offset : bool, optional
        Apply time offset. Default True.
    apply_spike_removal : bool, optional
        Apply spike removal (only for v3 sensors). Default True.
    figure_out : path object, optional
        Path to figure output directory. Default None.
    show_plot : bool, optional
        Plot and save time series. Default True.
    cal_time : np.datetime64 object, optional
        Time of post-deployment clock calibration. Used for plotting. Default
        None.
    max_time_drift : float, optional
        Maximum time drift to apply [hours]. Default 4.
    offset_to_utc : float
        Apply an additional offset (in hours) to the time vector. Good for
        correcting clocks accidentally set to local time. Default no offset.
    offset_time_drift : float
        Correct the time offset between sensor and download computer (in
        hours). This is helpful when the download computer has not been set to
        UTC time. Default no offset.

    Returns
    -------
    solo : xarray.DataArray
        DataArray with thermistor data
    """

    # only read raw file if we haven't written the netcdf file yet
    filename = "{:s}.nc".format(solofile.stem)
    if data_out:
        savepath = data_out.joinpath(filename)
        if savepath.exists():
            print("already processed\nreading netcdf file from\n{}".format(savepath))
            solo = xr.open_dataarray(savepath)
            # Update time file stamp. This way, make still recognizes that
            # the file has been worked on.
            # savepath.touch()
            savenc = False
            # For backwards compatibility, change name of sampling period attr
            if "sampling period" in solo.attrs:
                solo.attrs["sampling period in s"] = solo.attrs["sampling period"]
            else:
                pass
        else:
            print("reading raw rsk file")
            solo = read(
                solofile,
                offset_to_utc=offset_to_utc,
                offset_time_drift=offset_time_drift,
            )
            savenc = True
    else:
        print("reading raw rsk file")
        solo = read(solofile)
        savenc = False

    # spike removal
    if apply_spike_removal:
        solo = spike_removal(solo)

    # apply time drift
    if apply_time_offset:
        if solo.attrs["time drift in ms"] == 0:
            print("no time offset applied!")
        elif np.absolute(solo.attrs["time drift in ms"]) > max_time_drift * 3.6e6:
            print(f"time offset more than {max_time_drift} hours, not applied")
        else:
            solo = time_offset(solo)
    else:
        print("no time offset applied!")
    # make sure the file name in the meta data matches the raw file name
    # if not, derive from raw file name
    if Path(solo.attrs["file"]).stem != Path(filename).stem:
        solo.attrs["file"] = "{:s}.rsk".format(Path(filename).stem)
    # save to netcdf
    if savenc:
        save_nc(solo, data_out)
    # plot
    if show_plot:
        plot(solo, figure_out, cal_time)

    return solo


def read(solofile, offset_to_utc=0, offset_time_drift=0):
    """Read RBR Solo data in .rsk format and output as netcdf

    Parameters
    ----------
    solofile : path object
        Path to solo file
    offset_to_utc : float
        Apply an additional offset (in hours) to the time vector. Good for
        correcting clocks accidentally set to local time. Default no offset.
    offset_time_drift : float
        Correct the time offset between sensor and download computer (in
        hours). This is helpful when the download computer has not been set to
        UTC time. Default no offset.

    Returns
    -------
    solo : xarray.DataArray
        DataArray with thermistor data
    """
    # Access .rsk file
    rsk = pyrsktools.RSK(solofile)
    rsk.open()

    # Read all data
    print("reading SN {:d}".format(rsk.instrument.serialID))
    rsk.readdata()

    # Make sure we have temperature as a channel
    assert "temperature" in rsk.channelNames
    # Find temperature channel index
    temperature_channel_index = rsk.channelNames.index("temperature")
    solo_t = rsk.data["temperature"]
    time = rsk.data["timestamp"]
    # convert time to nanosecond precision, not because it is necessary but because
    # xarray currently complains about this
    time = time.astype("datetime64[ns]")
    # apply possible offset to UTC
    time += np.timedelta64(offset_to_utc, "h")

    # generate DataArray
    solo = xr.DataArray(solo_t, coords={"time": time}, dims=["time"], name="t")
    sampling_period = rsk.scheduleInfo.samplingperiod()
    # calculate sampling period in s
    sampling_period_calc = np.round(
        solo.time[:100]
        .diff(dim="time")
        .median()
        .data.astype("timedelta64[ns]")
        .astype(int)
        / 1e9,
        decimals=1,
    )
    # compare calculated sampling period to meta data
    assert sampling_period == sampling_period_calc

    # write meta data to attributes
    solo.attrs["units"] = rsk.channels[temperature_channel_index].units
    solo.attrs["long_name"] = "temperature"
    solo.attrs["SN"] = rsk.instrument.serialID
    solo.attrs["model"] = rsk.instrument.model
    solo.attrs["firmware version"] = rsk.instrument.firmwareVersion
    file = Path(rsk.deployment.name)
    solo.attrs["pyrsktools version"] = pyrsktools.__version__
    solo.attrs["rbrmoored version"] = rbr.__version__
    solo.attrs["file"] = file.name
    solo.attrs["time drift in ms"] = rsk.deployment.loggerTimeDrift + (
        offset_time_drift * 3600 * 1e3
    )
    download_time = rsk.deployment.timeOfDownload
    if download_time > solo.time[-1]:
        solo.attrs["download time"] = f"{download_time}"
    else:
        solo.attrs["download time"] = "N/A"
    sample_size = rsk.deployment.sampleSize
    solo.attrs["sample size"] = sample_size if sample_size is not None else "N/A"
    solo.attrs["sampling period in s"] = sampling_period
    # solo.attrs["path"] = os.path.dirname(solofile)
    solo.attrs["time offset applied"] = 0

    rsk.close()

    return solo


def time_offset(solo, time_drift=None):
    """Apply time offset to time series.

    Reads the time drift parameter from the dataset and applies it to the time
    vector. Adds the attribute 'time offset applied' to the dataset and sets it
    to 1. Will not re-apply the time offset if 'time offset applied' is 1.

    Parameters
    ----------
    solo : xarray.DataArray
        DataArray with thermistor data
    time_drift : float, optional
        Time drift in milliseconds. Supplying the time drift is usually not
        necessary but may be helpful if the drift determined by Ruskin fails.
        If supplied, will overwrite the Ruskin-based time drift. A general use
        case would be a time drift determined from the post-deployment clock
        calibration (warm water dip).

    Returns
    -------
    solo : xarray.DataArray
        DataArray with thermistor data
    """

    if solo.attrs["time offset applied"]:
        print("time offset has already been applied")
        return solo

    if time_drift is None:
        time_drift = solo.attrs["time drift in ms"]
    else:
        print(
            f"Replacing time drift of {solo.attrs['time drift in ms']}ms in attributes with {time_drift}ms"
        )
        solo.attrs["time drift in ms"] = time_drift
    print(f"Applying time offset of {time_drift}ms")

    # # generate linear time drift vector
    # old_time = solo.time.copy()
    # time_offset_linspace = np.linspace(
    #     0, time_drift, solo.attrs["sample size"]
    # )
    # # convert to numpy timedelta64
    # # this format can't handle non-integers, so we switch to nanoseconds
    # time_offset = [
    #     np.timedelta64(int(np.round(ti * 1e6)), "ns")
    #     for ti in time_offset_linspace
    # ]
    # new_time = solo.time - time_offset
    # solo["time"] = new_time
    # solo.attrs["time offset applied"] = 1

    # 3. VECTORIZED CALCULATION
    # Calculate drift in nanoseconds directly as a float array
    # time_drift (ms) * 1e6 = nanoseconds
    drift_ns = np.linspace(0, time_drift * 1e6, solo.sizes["time"])

    # Cast the entire array to integer and then to timedelta64[ns]
    # This replaces the slow list comprehension
    time_delta_vector = drift_ns.round().astype("timedelta64[ns]")

    # 4. Apply to coordinates
    # Using .assign_coords is often cleaner/faster in xarray than direct assignment
    solo = solo.assign_coords(time=solo.time - time_delta_vector)

    # 5. Update attributes
    solo.attrs["time offset applied"] = 1

    return solo


def time_offset_slow(solo, time_drift=None):
    """Apply time offset to time series.

    Reads the time drift parameter from the dataset and applies it to the time
    vector. Adds the attribute 'time offset applied' to the dataset and sets it
    to 1. Will not re-apply the time offset if 'time offset applied' is 1.

    Parameters
    ----------
    solo : xarray.DataArray
        DataArray with thermistor data
    time_drift : float, optional
        Time drift in milliseconds. Supplying the time drift is usually not
        necessary but may be helpful if the drift determined by Ruskin fails.
        If supplied, will overwrite the Ruskin-based time drift. A general use
        case would be a time drift determined from the post-deployment clock
        calibration (warm water dip).

    Returns
    -------
    solo : xarray.DataArray
        DataArray with thermistor data
    """

    if solo.attrs["time offset applied"]:
        print("time offset has already been applied")
    else:
        if time_drift is None:
            time_drift = solo.attrs["time drift in ms"]
        else:
            print(
                f"Replacing time drift of {solo.attrs['time drift in ms']}ms in attributes with {time_drift}ms"
            )
            solo.attrs["time drift in ms"] = time_drift
        print("applying time offset of {}ms".format(solo.attrs["time drift in ms"]))
        # generate linear time drift vector
        time_offset_linspace = np.linspace(0, time_drift, solo.attrs["sample size"])
        # convert to numpy timedelta64
        # this format can't handle non-integers, so we switch to nanoseconds
        time_offset = [
            np.timedelta64(int(np.round(ti * 1e6)), "ns") for ti in time_offset_linspace
        ]
        new_time = solo.time - time_offset
        solo["time"] = new_time
        solo.attrs["time offset applied"] = 1

    return solo


def save_nc(solo, data_out):
    """Save RBR Solo data in netcdf format.

    Parameters
    ----------
    solo : xarray.DataArray
        DataArray with thermistor data
    data_out : path object, optional
        Path to data output directory.
    """
    filename = "{:s}.nc".format(solo.attrs["file"][:-4])
    savepath = data_out.joinpath(filename)
    print("Saving to {}".format(savepath))
    solo.to_netcdf(savepath)


def plot(solo, figure_out=None, cal_time=None):
    """Plot RBR Solo time series.

    Plots additional panels with clock calibration if calibration time(s) are
    provided.

    Parameters
    ----------
    solo : xarray.DataArray
        DataArray with thermistor data
    figure_out : path object, optional
        Path to figure output directory. Default None.
    cal_time : np.datetime64 or tuple, optional
        Clock calibration(s) from warm water dip, either a single np.datetime64
        time or a tuple of two time stamps with pre- and post-deployment clock
        calibration. Default None.

    Raises
    ------
    TypeError
        If `cal_time` is neither None, a single np.datetime64, nor a tuple of
        two np.datetime64.
    """

    # Generate a list of one or two time stamps and check format of time
    # stamp(s) provided.
    def time_ok(time):
        return isinstance(time, np.datetime64)

    cal = []
    if cal_time is not None:
        if time_ok(cal_time):
            cal = [cal_time]
        elif (
            isinstance(cal_time, tuple)
            and len(cal_time) == 2
            and all(time_ok(ti) for ti in cal_time)
        ):
            cal = list(cal_time)
        else:
            raise TypeError(
                "Provide cal time(s) as a single np.datetime64 or a tuple of two"
            )
    ncal = len(cal)
    show_cal = ncal > 0

    # Check if calibration time(s) are outside the time series.
    for time in cal:
        if solo.time[-1] < time:
            print(f"clock cal time {time} is past end of time series")
        if solo.time[0] > time:
            print(f"clock cal time {time} is before start of time series")

    # Set up figure.
    if ncal == 1:
        fig, [ax0, axcal_single] = plt.subplots(
            nrows=2, ncols=1, figsize=(10, 7), constrained_layout=True
        )
        axcal = [axcal_single]
    elif ncal == 2:
        fig = plt.figure(
            constrained_layout=True,
            figsize=(10, 7),
        )
        gs = fig.add_gridspec(2, 2)
        ax0 = fig.add_subplot(gs[0, :])
        ax1 = fig.add_subplot(gs[1, 0])
        ax2 = fig.add_subplot(gs[1, 1])
        axcal = [ax1, ax2]
    else:
        fig, ax0 = plt.subplots(
            nrows=1, ncols=1, figsize=(10, 4), constrained_layout=True
        )
        axcal = []

    # Plot time series. Coarsen if it is too long to slow things down.
    if len(solo) > 1e5:
        coarsen_by = int(np.floor(60 / solo.attrs["sampling period in s"]))
        solo.coarsen(time=coarsen_by, boundary="trim").mean().plot(ax=ax0)
    else:
        solo.plot(ax=ax0)

    # Plot a warning if time offset not applied.
    if solo.attrs["time offset applied"] == 1:
        ax0.text(
            0.05,
            0.9,
            "time offset of {} seconds applied".format(
                solo.attrs["time drift in ms"] / 1000
            ),
            transform=ax0.transAxes,
            backgroundcolor="w",
        )
    else:
        if solo.attrs["time drift in ms"] == 0:
            ax0.text(
                0.05,
                0.9,
                "WARNING: time offset unknown",
                transform=ax0.transAxes,
                color="red",
                backgroundcolor="w",
            )
        elif np.absolute(solo.attrs["time drift in ms"]) > 3.6e6:
            ax0.text(
                0.05,
                0.9,
                "WARNING: time offset more than one hour, not applied",
                transform=ax0.transAxes,
                color="red",
                backgroundcolor="w",
            )
        else:
            ax0.text(
                0.05,
                0.9,
                "time offset not yet applied",
                transform=ax0.transAxes,
                backgroundcolor="w",
            )

    ax0.grid()
    ax0.set(title="RBR Solo SN {}".format(solo.attrs["SN"]))
    ax0.set(xlabel="")
    rbr.utils.concise_date(ax0)

    # Plot calibration.
    if show_cal:
        for cali, axi in zip(cal, axcal):
            tmp = solo.sel(
                time=slice(
                    cali - np.timedelta64(60, "s"),
                    cali + np.timedelta64(60, "s"),
                )
            )
            if len(tmp) > 0:
                tmp.plot(ax=axi, marker=".")
                ylims = np.array([np.floor(tmp.min().data), np.ceil(tmp.max().data)])
            else:
                ylims = np.array([1, 9])
            axi.plot(
                np.tile(cali, 2),
                ylims + np.array([1, -1]),
                linestyle="-",
                color="darkorchid",
                linewidth=1.5,
            )
            axi.annotate(
                "time calibration",
                (cali, ylims[0] + 0.5),
                xytext=(16, 8),
                textcoords="offset points",
                color="darkorchid",
                ha="left",
                backgroundcolor="w",
            )
            axi.set(
                xlim=[
                    cali - np.timedelta64(20, "s"),
                    cali + np.timedelta64(20, "s"),
                ],
                ylim=ylims,
                xlabel="",
            )
            axi.grid()
            rbr.utils.concise_date(axi)
        if ncal == 2:
            axcal[-1].set(ylabel="")

    if figure_out is not None:
        figurename = "{:s}.png".format(solo.attrs["file"][:-4])
        plt.savefig(figure_out.joinpath(figurename), dpi=300)


def _phase(index, dt, period_s, n_bins):
    """Bin each sample by its position within one artifact cycle.

    Parameters
    ----------
    index : numpy.ndarray
        Sample index within the record.
    dt : float
        Sampling period in seconds.
    period_s : float
        Cycle length in seconds. Need not be a multiple of `dt`.
    n_bins : int
        Number of phase bins.

    Returns
    -------
    phase : numpy.ndarray
        Phase bin of each sample, in `[0, n_bins)`.
    """
    cycle_samples = period_s / dt
    phase = ((index % cycle_samples) / cycle_samples * n_bins).astype(np.int16)
    return np.clip(phase, 0, n_bins - 1, out=phase)


def _fold(deviation, index, dt, period_s, n_bins, trim_fraction):
    """Fold a record onto one artifact cycle.

    A trimmed mean over all cycles is formed for each phase bin, rather than a
    plain mean, so that the result does not depend on whether the large
    excursions from deck and lab time are part of the record. A plain median is
    no good here because the deviation from a running median has a point mass at
    zero - a sample is its own running median every so often. The composite is
    centered on its median so that the peak is measured against the background
    level rather than against zero.

    Parameters
    ----------
    deviation : numpy.ndarray
        Deviation of the record from its running median.
    index : numpy.ndarray
        Sample index within the record.
    dt : float
        Sampling period in seconds.
    period_s : float
        Cycle length in seconds.
    n_bins : int
        Number of phase bins.
    trim_fraction : float
        Fraction of cycles trimmed off either end.

    Returns
    -------
    composite : numpy.ndarray
        Trimmed mean deviation per phase bin.
    """
    phase = _phase(index, dt, period_s, n_bins)
    counts = np.bincount(phase, minlength=n_bins)
    edges = np.concatenate([[0], np.cumsum(counts)])
    deviation_by_bin = deviation[np.argsort(phase, kind="stable")]
    composite = np.zeros(n_bins)
    for k in range(n_bins):
        cycles = np.sort(deviation_by_bin[edges[k] : edges[k + 1]])
        if cycles.size == 0:
            continue
        trim = int(trim_fraction * cycles.size)
        composite[k] = cycles[trim : cycles.size - trim].mean()
    return composite - np.median(composite)


def spike_removal(
    solo,
    period_s=60.0,
    window_s=30.0,
    min_snr=6.0,
    max_spike_samples=3,
    min_cycles=30,
    trim_fraction=0.1,
    verbose=True,
):
    """Remove the periodic spike in RBRsolo³ time series.

    RBRsolo³ loggers show a small positive spike in temperature, of order 1 mK
    and typically confined to a single sample, once every 60 s. It is caused by
    heating of the A/D converter during its calibration cycle. In a temperature
    variance spectrum the artifact shows up as a sharp line at the 60 s period
    together with its harmonics.

    The artifact is deterministic: it repeats at a fixed period and at a fixed
    phase within that period. This is exploited here. The record is folded onto
    the spike period and a trimmed mean deviation from a running median is
    formed for each phase bin. A spike shows up as a single bin standing well
    above the rest. Whether it does is decided by folding the same record at
    slightly detuned periods, which is what the fold looks like when there is
    nothing periodic to find. Only samples falling into the spike bin (and, if
    the spike is wider than one sample, into the adjacent bins) are replaced, by
    linear interpolation from the neighbouring samples.

    If no significant spike is found the record is returned unchanged and the
    attribute `despiking applied` is set to 0. This is the expected outcome for
    loggers that do not show the artifact - detecting spikes relative to a local
    noise estimate alone would always find *something* and remove good data.

    Records that are not RBRsolo³, that have already been despiked, or that are
    shorter than `min_cycles` spike periods are returned unchanged.

    Note that the spike is locked to the logger clock, so this should be run
    before the clock drift correction in `time_offset`. `proc` does that.

    Parameters
    ----------
    solo : xarray.DataArray
        DataArray with thermistor data. Requires the attributes `model` and
        `sampling period in s` as written by `read`.
    period_s : float, optional
        Period of the spike in seconds. Default 60.
    window_s : float, optional
        Length of the running median window in seconds. Used to remove the
        background signal before folding. Default 30.
    min_snr : float, optional
        Minimum ratio of the largest phase bin of the folded record over the
        largest phase bin found when folding at detuned periods for a spike to
        be accepted. Default 6.
    max_spike_samples : int, optional
        Maximum number of samples per cycle that may be replaced. Default 3.
    min_cycles : int, optional
        Minimum number of spike periods a record must cover. Default 30.
    trim_fraction : float, optional
        Fraction of cycles trimmed off either end when folding the record.
        Keeps deck and lab time from dominating the fold. Default 0.1.
    verbose : bool, optional
        Print what was found and done. Default True.

    Returns
    -------
    solo : xarray.DataArray
        DataArray with thermistor data. The following attributes document the
        result: `despiking applied`, `spike period in s`, `spike phase in s`
        (position of the spike within the cycle, counted from the whole minute),
        `spike amplitude in K`, `spike detection snr` and `spikes removed`.
    """

    def info(message):
        if verbose:
            print(message)

    # Only apply to the RBRsolo³ version, not older units.
    if solo.attrs.get("model", "") != "RBRsolo³":
        return solo

    if bool(solo.attrs.get("despiking applied", 0)):
        info("despiking has already been applied")
        return solo

    dt = float(solo.attrs["sampling period in s"])
    n_bins = round(period_s / dt)
    if not np.isclose(period_s / dt, n_bins):
        raise ValueError(
            f"spike period of {period_s}s is not a multiple of the sampling "
            f"period of {dt}s"
        )

    values = solo.data.astype(float)
    if values.size < min_cycles * n_bins:
        info(
            f"record covers less than {min_cycles} spike periods, "
            "not looking for spikes"
        )
        return solo

    # Sample index within the record. Derived from the time vector via the
    # median sample spacing so that gaps are accounted for and so that this
    # still works on a record that has had the clock drift applied.
    time_s = solo.time.data.astype("datetime64[ns]").astype("int64") / 1e9
    spacing = np.median(np.diff(time_s))
    index = np.round((time_s - time_s[0]) / spacing).astype(np.int64)

    # Deviation from a running median. The median tracks the background signal
    # while ignoring the spikes.
    window_size = round(window_s / dt)
    window_size += 1 - window_size % 2  # odd window, centered
    deviation = values - median_filter(values, size=window_size)

    # Fold onto the spike period.
    composite = _fold(deviation, index, dt, period_s, n_bins, trim_fraction)
    peak_bin = int(np.argmax(np.abs(composite)))

    # How large a peak does folding this record produce when there is nothing
    # periodic to find? Fold at slightly detuned periods to find out. A real
    # spike smears out over all bins there, so these folds sample the null
    # hypothesis. Comparing against the scatter between neighbouring bins of
    # the on-period fold instead would be far too generous: that scatter is
    # correlated over the length of the running median window and any record
    # would end up with a "significant" bin.
    null_level = max(
        np.max(np.abs(_fold(deviation, index, dt, null_period, n_bins, trim_fraction)))
        for null_period in [
            period_s * (1 + offset / n_bins) for offset in (-2, -1, 1, 2)
        ]
    )
    snr = np.abs(composite[peak_bin]) / null_level if null_level > 0 else np.inf

    solo = solo.copy()
    solo.attrs["spike period in s"] = period_s
    solo.attrs["spike phase in s"] = float((time_s[0] + peak_bin * dt) % period_s)
    solo.attrs["spike amplitude in K"] = float(composite[peak_bin])
    solo.attrs["spike detection snr"] = float(snr)

    if snr < min_snr:
        info(
            f"no periodic spike detected (snr {snr:.1f} < {min_snr:.1f}), "
            "no samples removed"
        )
        solo.attrs["despiking applied"] = 0
        solo.attrs["spikes removed"] = 0
        return solo

    # Scatter between neighbouring bins of the on-period fold. Now that the
    # spike is established this is the right scale for delineating its width.
    # Bins within half a median window of the peak are excluded: the spike sits
    # in their running median and biases them away from zero by a fraction of a
    # percent of its amplitude.
    excluded = min(window_size // 2, n_bins // 4)
    background = np.delete(
        composite, [(peak_bin + i) % n_bins for i in range(-excluded, excluded + 1)]
    )
    bin_noise = 1.4826 * np.median(np.abs(background - np.median(background)))

    # Grow the spike into the adjacent bins as long as they stand out as well.
    # A single sample is the norm, but the sensor may not have recovered fully
    # by the next sample. Candidates must deviate in the same direction as the
    # peak: the bins around the spike are pulled the other way by the running
    # median and are not part of the artifact.
    def is_spike_bin(bin_index):
        return (
            np.sign(composite[bin_index]) == np.sign(composite[peak_bin])
            and np.abs(composite[bin_index]) >= min_snr * bin_noise
        )

    spike_bins = [peak_bin]
    while len(spike_bins) < max_spike_samples:
        after = (spike_bins[-1] + 1) % n_bins
        if not is_spike_bin(after):
            break
        spike_bins.append(after)
    while len(spike_bins) < max_spike_samples:
        before = (spike_bins[0] - 1) % n_bins
        if not is_spike_bin(before):
            break
        spike_bins.insert(0, before)

    spike = np.isin(_phase(index, dt, period_s, n_bins), spike_bins)
    # Linear interpolation from the neighbouring good samples. For an isolated
    # sample this is the mean of the two samples either side.
    clean = values.copy()
    clean[spike] = np.interp(index[spike], index[~spike], values[~spike])

    info(
        f"removed {spike.sum()} spikes of {1e3 * composite[peak_bin]:.2f}mK "
        f"at {period_s:.0f}s period (snr {snr:.0f}, "
        f"{len(spike_bins)} sample(s) per cycle)"
    )

    solo.data = clean
    solo.attrs["despiking applied"] = 1
    solo.attrs["spikes removed"] = int(spike.sum())

    return solo
