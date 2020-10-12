from attrdict import AttrDict


def shear_line(color):
    """Return config with color."""
    return {
        "color": color,
        "linestyle": "-",
        "linewidth": 1.5,
        "alpha": 1,
        "markersize": 2,
        "zorder": 3500,
    }


def nowcast_line(color, linestyle="-."):
    """Return config with color."""
    return {
        "color": color,
        "linestyle": linestyle,
        "linewidth": 1.5,
        "alpha": 1,
        "markersize": 2,
        "zorder": 3500,
    }


# Colormap definitions
cmaps = AttrDict({
    "vrad": {
        "cmap": "PRGn_r",
        "vmin": -25,
        "vmax": 25,
        "rasterized": True,
        "alpha": 1,
        "zorder": 3000,
        "label": "Radial velocity [m/s]"
    },

    "vrad2": {
        "cmap": "BrBG_r",
        "vmin": -25,
        "vmax": 25,
        "rasterized": True,
        "alpha": 1,
        "zorder": 3500,
        "label": "Radial velocity [m/s]"
    },

    "vrad_pyart": {
        "cmap": "pyart_BuDRd18",
        "vmin": -30,
        "vmax": 30,
        "rasterized": True,
        "alpha": 1,
        "zorder": 3500,
        "label": "Radial velocity [m/s]"
    },

    "shear": {
        "cmap": "cividis_r",
        "vmin": 0,
        "vmax": 20,
        "rasterized": True,
        "alpha": 1,
        "zorder": 3000,
    },

    "shear_twosided": {
        "cmap": "bwr",
        "vmin": -15,
        "vmax": 15,
        "rasterized": True,
        "alpha": 1,
        "zorder": 3000,
    },

    "reflectivity": {
        "cmap": "pyart_NWSRef",
        "vmin": -30,
        "vmax": 75,
        "rasterized": True,
        "alpha": 1,
        "zorder": 2000,
        "edgecolor": "none",
        "label": "Radar reflectivity [dBZ]"
    },

    "angle": {
        "cmap": "twilight",
        "vmin": 0,
        "vmax": 360,
        "rasterized": True,
        "alpha": 1,
        "zorder": 3000,
    },

    "twosided_error": {
        "cmap": "coolwarm",
        "vmin": -10,
        "vmax": 10,
        "rasterized": True,
        "alpha": 1,
        "zorder": 3000,
    },

    "correlation": {
        "cmap": "coolwarm",
        "vmin": -1,
        "vmax": 1,
        "rasterized": True,
        "alpha": 1,
        "zorder": 3000,
    },

    "onesided_error": {
        "cmap": "coolwarm",
        "vmin": 0,
        "vmax": 10,
        "rasterized": True,
        "alpha": 1,
        "zorder": 3000,
    }
})

# Font sizes
fontsizes = AttrDict({
    "title": 12,
    "suptitle": 14,
    "colorbar": {
        "label": 10,
        "ticks": 8,
    },
})

# Grid style definitions
grids = AttrDict({
    "major": {
        "which": "major",
        "alpha": 0.5,
        "zorder": 0,
    },

    "minor": {
        "which": "minor",
        "alpha": 0.2,
        "zorder": 0,
    },
})

contours = AttrDict({
    "dashed_red": {
        "color": "tab:red",
        "linestyle": "-.",
        "linewidth": 0.5,
        "alpha": 0.6,
        "markersize": 1.5,
        "zorder": 3500,
    },
    "dashed_blue": {
        "color": "tab:blue",
        "linestyle": "-.",
        "linewidth": 0.5,
        "alpha": 0.6,
        "markersize": 1.5,
        "zorder": 3500,
    },
    "solid_red": {
        "color": "tab:red",
        "linestyle": "-",
        "linewidth": 1.5,
        "alpha": 1,
        "markersize": 2,
        "zorder": 3500,
    },
    "solid_orange": {
        "color": "tab:orange",
        "linestyle": "-",
        "linewidth": 1.5,
        "alpha": 1,
        "markersize": 2,
        "zorder": 3500,
    },
    "solid_blue": {
        "color": "tab:blue",
        "linestyle": "-",
        "linewidth": 1.5,
        "alpha": 1,
        "markersize": 2,
        "zorder": 3500,
    },
    "dotted_yellow": {
        "color": "y",
        "linestyle": "dotted",
        "linewidth": 1.5,
        "alpha": 1,
        "markersize": 3,
        "zorder": 3500,
    },
    "starred": {
        "color": "tab:cyan",
        "marker": "*",
        "linestyle": "dotted",
        "linewidth": 1.5,
        "alpha": 1,
        "markersize": 5,
        "zorder": 3500,
    },
    "shear": {
        "linestyle": '-',
        "linewidth": 1.5,
        "alpha": 1,
        "markersize": 1,
        "zorder": 3500,
    },
    "shear_with_danger_zone": {
        "linestyle": '-',
        "linewidth": 25,
        "alpha": 0.5,
        "markersize": 1,
        "zorder": 3500,
        "solid_capstyle": 'round',
    }
})

scatter = AttrDict({
    "red_plus": {
        "color": "tab:red",
        "marker": "+",
        "zorder": 3500,
    },
    "orange_plus": {
        "color": "tab:orange",
        "marker": "+",
        "zorder": 3500,
    },
    "blue_plus": {
        "color": "tab:blue",
        "marker": "+",
        "zorder": 3500,
    },
})
