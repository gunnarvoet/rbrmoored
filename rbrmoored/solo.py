#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Module rbrmoored.solo"""


import os
from pathlib import Path
import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import xarray as xr

import pyrsktools

import gvpy as gv


def proc(
    solofile,
    data_out=None,
    apply_time_offset=True,
    figure_out=None,
    show_plot=True,
    cal_time=None,
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
    figure_out : path object, optional
        Path to figure output directory. Default None.
    show_plot : bool, optional
        Plot and save time series. Default True.
    cal_time : np.datetime64 object, optional
        Time of post-deployment clock calibration. Used for plotting. Default
        None.

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
            print(
                "already processed\nreading netcdf file from\n{}".format(
                    savepath
                )
            )
            solo = xr.open_dataarray(savepath)
            # Update time file stamp. This way, make still recognizes that
            # the file has been worked on.
            # savepath.touch()
            savenc = False
            # For backwards compatibility, change name of sampling period attr
            if "sampling period" in solo.attrs:
                solo.attrs["sampling period in s"] = solo.attrs[
                    "sampling period"
                ]
            else:
                pass
        else:
            print("reading raw rsk file")
            solo = read(solofile)
            savenc = True
    else:
        print("reading raw rsk file")
        solo = read(solofile)
        savenc = False
    # apply time drift
    if apply_time_offset:
        if solo.attrs["time drift in ms"] == 0:
            print("no time offset applied!")
        elif np.absolute(solo.attrs["time drift in ms"]) > 3.6e6:
            print("time offset more than one hour, not applied")
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


def read_rsk(solofile):
    rsk = pyrsktools.open(solofile)
    print("reading SN {:d}".format(rsk.instrument.serial))
    # read all data
    data = rsk.npsamples()
    return rsk, data


def read(solofile):
    """Read RBR Solo data in .rsk format and output as netcdf

    Parameters
    ----------
    solofile : path object
        Path to solo file

    Returns
    -------
    solo : xarray.DataArray
        DataArray with thermistor data
    """

    # read data
    rsk, data = read_rsk(solofile)
    solo_t = data["temperature_00"]
    solo_timez = data["timestamp"]
    # strip time zone info or else numpy complains
    solo_time = [tz.replace(tzinfo=None) for tz in solo_timez]
    # generate time vector in numpy time format
    tt = np.array([np.datetime64(ti, "ns") for ti in solo_time])
    # generate DataArray
    solo = xr.DataArray(solo_t, coords={"time": tt}, dims=["time"], name="t")
    # calculate sampling period in s
    sampling_period = np.round(
        solo.time[:100]
        .diff(dim="time")
        .median()
        .data.astype("timedelta64[ns]")
        .astype(int)
        / 1e9,
        decimals=1,
    )
    # write meta data to attributes
    solo.attrs["units"] = rsk.channels["temperature_00"].units
    solo.attrs["long_name"] = "temperature"
    solo.attrs["SN"] = rsk.instrument.serial
    solo.attrs["model"] = rsk.instrument.model
    solo.attrs["firmware version"] = rsk.instrument.firmware_version
    file = Path(rsk.deployment.name)
    solo.attrs["file"] = file.name
    solo.attrs["time drift in ms"] = rsk.deployment.logger_time_drift
    download_time = np.datetime64(
        rsk.deployment.download_time.replace(tzinfo=None), "ns"
    )
    if download_time > solo.time[-1]:
        solo.attrs["download time"] = "{}".format(rsk.deployment.download_time)
    else:
        solo.attrs["download time"] = "N/A"
    sample_size = rsk.deployment.sample_size
    solo.attrs["sample size"] = (
        sample_size if sample_size is not None else "N/A"
    )
    solo.attrs["sampling period in s"] = sampling_period
    # solo.attrs["path"] = os.path.dirname(solofile)
    solo.attrs["time offset applied"] = 0

    rsk.close()

    return solo


def time_offset(solo):
    """Apply time offset to time series.

    Reads the time drift parameter from the dataset and applies it to the time
    vector. Adds the attribute 'time offset applied' to the dataset and sets it
    to 1. Will not re-apply the time offset if 'time offset applied' is 1.

    Parameters
    ----------
    solo : xarray.DataArray
        DataArray with thermistor data

    Returns
    -------
    solo : xarray.DataArray
        DataArray with thermistor data
    """

    if solo.attrs["time offset applied"]:
        print("time offset has already been applied")
    else:
        print(
            "applying time offset of {}ms".format(
                solo.attrs["time drift in ms"]
            )
        )
        # generate linear time drift vector
        old_time = solo.time.copy()
        time_offset_linspace = np.linspace(
            0, solo.attrs["time drift in ms"], solo.attrs["sample size"]
        )
        # convert to numpy timedelta64
        # this format can't handle non-integers, so we switch to nanoseconds
        time_offset = [
            np.timedelta64(int(np.round(ti * 1e6)), "ns")
            for ti in time_offset_linspace
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
    """Save RBR Solo data in netcdf format.

    Parameters
    ----------
    solo : xarray.DataArray
        DataArray with thermistor data
    figure_out : path object, optional
        Path to figure output directory. Default None.
    cal_time : np.datetime64 or tuple
        Clock calibration(s) from warm water dip, either a single np.datetime64
        time or a tuple of two time stamps with pre- and post-deployment clock
        calibration.
    """
    # Generate a list of one or two time stamps and check format of time
    # stamp(s) provided.
    def time_ok(time):
        return True if type(time) == np.datetime64 else False

    if cal_time is not None:
        if time_ok(cal_time):
            cal = [cal_time]
        elif type(cal_time) == tuple:
            if time_ok(cal_time[0]) and time_ok(cal_time[1]):
                cal = [cal_time[0], cal_time[1]]
        else:
            raise TypeError(
                "Provide cal time(s) as single np.datetime64 or tuple thereof"
            )

    # Check if calibration time(s) are outside the time series.
    if cal_time is not None:
        for time in cal:
            if solo.time[-1] < time:
                show_cal = False
                print(
                    f"clock cal time {time} is past end of time series, not plotting"
                )
            if solo.time[0] > time:
                show_cal = False
                print(
                    f"clock cal time {time} is before start of time series, not plotting"
                )
            else:
                show_cal = True

    # Register how many cals to show
    if cal_time is None:
        ncal = 0
        show_cal = False
    else:
        ncal = len(cal)
        show_cal = True

    # set up figure
    if show_cal:
        if ncal == 1:
            fig, [ax0, [axcal]] = plt.subplots(
                nrows=2, ncols=1, figsize=(10, 7), constrained_layout=True
            )
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

    # plot time series. coarsen if it is too long to slow things down
    if len(solo) > 1e5:
        coarsen_by = int(np.floor(60 / solo.attrs["sampling period in s"]))
        solo.coarsen(time=coarsen_by, boundary="trim").mean().plot(ax=ax0)
    else:
        solo.plot(ax=ax0)

    # plot a warning if time offset not applied
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
    gv.plot.concise_date(ax0)

    # plot calibration
    if show_cal:
        for cal_time, axi in zip(cal, axcal):
            tmp = solo.sel(
                time=slice(
                    cal_time - np.timedelta64(60, "s"),
                    cal_time + np.timedelta64(60, "s"),
                )
            )
            if len(tmp) > 0:
                tmp.plot(ax=axi, marker=".")
                ylims = np.array(
                    [np.floor(tmp.min().data), np.ceil(tmp.max().data)]
                )
            else:
                ylims = np.array([1, 9])
            axi.plot(
                np.tile(cal_time, 2),
                ylims + np.array([1, -1]),
                linestyle="-",
                color="darkorchid",
                linewidth=1.5,
            )
            axi.annotate(
                "time calibration",
                (cal_time, ylims[0] + 0.5),
                xytext=(16, 8),
                textcoords="offset points",
                color="darkorchid",
                ha="left",
                backgroundcolor="w",
            )
            axi.set(
                xlim=[
                    cal_time - np.timedelta64(20, "s"),
                    cal_time + np.timedelta64(20, "s"),
                ],
                ylim=ylims,
                xlabel="",
            )
            axi.grid()
            gv.plot.concise_date(axi)
        if ncal == 2:
            axcal[-1].set(ylabel='')
            # ax1.get_shared_y_axes().join(ax1, ax2)

    if figure_out is not None or False:
        figurename = "{:s}.png".format(solo.attrs["file"][:-4])
        plt.savefig(figure_out.joinpath(figurename), dpi=300)
