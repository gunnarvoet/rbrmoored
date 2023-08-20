#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utilities.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates

def concise_date(ax=None, minticks=3, maxticks=10, show_offset=True, **kwargs):
    """
    Better date ticks using matplotlib's ConciseDateFormatter.

    Parameters
    ----------
    ax : axis handle
        Handle to axis (optional).
    minticks : int
        Minimum number of ticks (optional, default 6).
    maxticks : int
        Maximum number of ticks (optional, default 10).
    show_offset : bool, optional
        Show offset string to the right (default True).

    Note
    ----
    Currently only works for x-axis

    See Also
    --------
    matplotlib.mdates.ConciseDateFormatter : For formatting options that
      can be used here.
    """
    if ax is None:
        ax = plt.gca()
    locator = mdates.AutoDateLocator(minticks=minticks, maxticks=maxticks)
    formatter = mdates.ConciseDateFormatter(
        locator, show_offset=show_offset, **kwargs
    )
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    # remove axis label "time" if present
    if ax.get_xlabel() == "time":
        _ = ax.set_xlabel("")
