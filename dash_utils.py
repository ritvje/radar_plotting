"""Utility functions for plotting with dash."""
import matplotlib as mpl
from matplotlib import cm, colors
import numpy as np
import pyart


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
