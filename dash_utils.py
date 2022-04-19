"""Utility functions for plotting with dash."""
import matplotlib as mpl
from matplotlib import cm, colors
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pyart

from . import plot_utils


def cmap_to_RGB(cmap, norm, num_colors=255, values=None):
    if isinstance(cmap, str):
        cmap = cm.get_cmap(cmap)

    if values is None:
        values = np.linspace(norm.vmin, norm.vmax, num_colors)

    rgb = []
    for value in values:
        k = colors.colorConverter.to_rgb(cmap(norm(value)))
        rgb.append(f"rgb({int(k[0]*255)},{int(k[1]*255)},{int(k[2]*255)})")

    return rgb


def plot_onepanel_ppi(radar, qty, name, max_dist=75, range_ring_res=50, figsize=(7, 7)):
    cbar_ax_kws = {
        "width": "5%",  # width = 5% of parent_bbox width
        "height": "100%",
        "loc": "lower left",
        "bbox_to_anchor": (1.01, 0.0, 1, 1),
        "borderpad": 0,
    }

    fig, ax = plt.subplots(
        nrows=1, ncols=1, figsize=figsize, sharex="col", sharey="row"
    )
    display = pyart.graph.RadarDisplay(radar)
    fmt = mpl.ticker.StrMethodFormatter("{x:.0f}")

    cax = inset_axes(ax, bbox_transform=ax.transAxes, **cbar_ax_kws)

    cmap, norm = plot_utils.get_colormap(qty)
    if norm is None:
        norm = mpl.colors.Normalize(
            vmin=plot_utils.QTY_RANGES[qty][0], vmax=plot_utils.QTY_RANGES[qty][1]
        )

    display.plot(
        name,
        0,
        title="",
        ax=ax,
        axislabels_flag=False,
        colorbar_flag=False,
        cmap=cmap,
        norm=norm,
        zorder=10,
    )

    cbar = plt.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        format=mpl.ticker.StrMethodFormatter(plot_utils.QTY_FORMATS[qty]),
        orientation="vertical",
        cax=cax,
        ax=None,
    )
    cbar.set_label(label=plot_utils.COLORBAR_TITLES[qty], weight="bold")
    cbar.ax.tick_params(labelsize=12)

    display.plot_range_rings(
        np.arange(0, max_dist + range_ring_res, range_ring_res), ax=ax, lw=0.5, col="k"
    )
    display.plot_grid_lines(ax=ax, col="grey", ls=":")

    # x-axis
    ax.set_xlabel("Distance from radar [km]")
    ax.set_title(ax.get_title(), y=-0.22)
    ax.xaxis.set_major_formatter(fmt)

    # y-axis
    ax.set_ylabel("Distance from radar [km]")
    ax.yaxis.set_major_formatter(fmt)

    ax.set_xlim([-max_dist, max_dist])
    ax.set_ylim([-max_dist, max_dist])
    ax.set_aspect(1)
    # ax.grid(zorder=20, linestyle="-", linewidth=0.4)

    return fig
