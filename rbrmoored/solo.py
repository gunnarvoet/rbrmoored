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
    figure_out=None,
    cal_time=None,
    show_plot=True,
    apply_time_offset=True,
):
    """
    Combining RBR Solo processing steps.

    Parameters
    ----------
    solofile : path object
        Path to solo file
    data_out : path object, optional
        Path to data output directory.
    figure_out : path object, optional
        Path to figure output directory.
    cal_time : np.datetime64 object, optional
        Time of post-deployment clock calibration.
    show_plot : bool, optional
        Plot and save time series. Default True.
    apply_time_offset : bool, optional
        Apply time offset if True.
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
            savenc = False
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
    # read data
    data = rsk.npsamples()
    return rsk, data


def read(solofile):
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
        / 1e9
    )
    # write meta data to attributes
    solo.attrs["units"] = rsk.channels["temperature_00"].units
    solo.attrs["long_name"] = "temperature"
    solo.attrs["SN"] = rsk.instrument.serial
    solo.attrs["model"] = rsk.instrument.model
    solo.attrs["firmware version"] = rsk.instrument.firmware_version
    solo.attrs["file"] = rsk.deployment.name
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
    solo.attrs["sampling period"] = sampling_period
    # solo.attrs["path"] = os.path.dirname(solofile)
    solo.attrs["time offset applied"] = 0

    rsk.close()

    return solo


def time_offset(solo):
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
    # save dataset
    filename = "{:s}.nc".format(solo.attrs["file"][:-4])
    savepath = data_out.joinpath(filename)
    print("Saving to {}".format(savepath))
    solo.to_netcdf(savepath)


def plot(solo, figure_out=None, cal_time=None):

    # check if cal_time is past end of time series
    if cal_time is not None:
        if solo.time[-1] < cal_time:
            show_cal = False
            print("clock cal time is past end of time series, not plotting")
        else:
            show_cal = True
    else:
        show_cal = False

    # set up figure
    if show_cal:
        fig, [ax0, ax1] = plt.subplots(
            nrows=2, ncols=1, figsize=(10, 7), constrained_layout=True
        )
    else:
        fig, ax0 = plt.subplots(
            nrows=1, ncols=1, figsize=(10, 4), constrained_layout=True
        )

    # plot time series. coarsen if it is too long to slow things down
    if len(solo) > 1e5:
        coarsen_by = int(np.floor(60 / solo.attrs["sampling period"]))
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
        tmp = solo.sel(
            time=slice(
                cal_time - np.timedelta64(60, "s"),
                cal_time + np.timedelta64(60, "s"),
            )
        )
        if len(tmp) > 0:
            tmp.plot(ax=ax1, marker=".")
            ylims = np.array(
                [np.floor(tmp.min().data), np.ceil(tmp.max().data)]
            )
        else:
            ylims = np.array([1, 9])
        ax1.plot(
            np.tile(cal_time, 2),
            ylims + np.array([1, -1]),
            linestyle="-",
            color="darkorchid",
            linewidth=1.5,
        )
        ax1.annotate(
            "time calibration",
            (cal_time, ylims[0] + 0.5),
            xytext=(8, 8),
            textcoords="offset points",
            color="darkorchid",
            ha="left",
            backgroundcolor="w",
        )
        ax1.set(
            xlim=[
                cal_time - np.timedelta64(60, "s"),
                cal_time + np.timedelta64(60, "s"),
            ],
            ylim=ylims,
            xlabel="",
        )
        ax1.grid()
        gv.plot.concise_date(ax1)

    if figure_out is not None or False:
        figurename = "{:s}.png".format(solo.attrs["file"][:-4])
        plt.savefig(figure_out.joinpath(figurename), dpi=300)
