"""Utility functions for plotting."""
import os
import types
import requests
import PIL
import numpy as np
import matplotlib as mlt
import matplotlib.pyplot as pl
from datetime import datetime, timedelta
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from distutils.spawn import find_executable
from functools import wraps
import socket

# Some decorators to make life easy
def draft(*args, **fig_kwargs):
    """Create a figure with "DRAFT" watermark."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            fig = pl.figure(**fig_kwargs)
            fig.text(0.5, 0.5, "DRAFT", fontsize=65,
                     color="gray", ha="center", va="center", alpha=0.4,
                     rotation=25, rotation_mode="anchor", transform=fig.transFigure)
            return func(fig, *args, **kwargs)
        return wrapper
    return decorator


def final(*args, **fig_kwargs):
    """Create a figure with no watermark."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            fig = pl.figure(**fig_kwargs)
            return func(fig, *args, **kwargs)
        return wrapper
    return decorator


def use_tex(use_tex=True):
    """Whether to use tex or not."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if use_tex:
                mlt.rc("text", usetex=True)
                mlt.rc(
                    'text.latex', preamble=(
                        "\\usepackage{mathptmx}"
                        "\\usepackage{amsfonts}"
                        "\\usepackage{amsmath}"
                        "\\usepackage{times}"
                        "\\usepackage{bbding}"
                        "\\usepackage{fontawesome5}"
                    ))
            else:
                mlt.rc("text", usetex=False)

            res = func(*args, **kwargs)
            # Turn off afterwards
            mlt.rc("text", usetex=False)

            return res
        return wrapper
    return decorator


class CachedTiler(object):
    """Class for caching cartopy map tiles.

    This class allows caching map tiles so they don't have to be
    fetched every time.
    Taken from https://github.com/SciTools/cartopy/issues/732
    Note that even though this improves the speed slightly, high
    resolution maps are still slow to plot

    """

    def __init__(self, tiler):
        """Init function.

        Parameters
        ----------
        tiler : cartopy.io.img_tiles tiler object
            The tiler that is cached.

        """
        self.tiler = tiler

    def __getattr__(self, name):
        """Mimic the tiler interface.

        For methods, ensure that the "self" that is passed through
        continues to be CachedTiler, and not the contained tiler
        instance.
        """
        attr = getattr(self.tiler, name, None)
        if isinstance(attr, types.MethodType):
            attr = types.MethodType(attr.__func__, self)
        return attr

    def get_image(self, tile):
        """Get the image, either from local file or download."""
        tileset_name = '{}_{}'.format(self.tiler.__class__.__name__.lower(),
                                      self.tiler.style.lower())
        cache_dir = os.path.expanduser(os.path.join('~/', 'image_tiles',
                                                    tileset_name))

        # If cache doesn't exist, create it
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        tile_fname = os.path.join(
            cache_dir, '_'.join(str(v) for v in tile) + '.png')
        # Check if image already exists; if not, download
        if not os.path.exists(tile_fname):
            response = requests.get(self._image_url(tile),
                                    stream=True)

            with open(tile_fname, "wb") as fh:
                for chunk in response:
                    fh.write(chunk)
        with open(tile_fname, 'rb') as fh:
            img = PIL.Image.open(fh)
            img = img.convert(self.desired_tile_form)
        return img, self.tileextent(tile), 'lower'


def axes_with_background_map(
        centerpoint, extent_km, zoom, fig=None, no_map=False, figsize=(10, 10),
        map='terrain', nrows=1, ncols=1, index=1, regrid_shape=1000,
        sharex=None, sharey=None):
    """Create axis with a terrain map background.

    Parameters
    ----------
    centerpoint : array-like
        Lon, lat point for the center of the map.
    extent_km : float
        Extent of the map from the centerpoint in each direction, in
        kilometers.
    zoom : int
        Zoom level of the map.
    fig : matplotlib.pyplot.Figure
        The figure the axis is created in. if not given, created.
    no_map : bool
        If True, don't plot map on background (much faster).
    figsize : tuple
        Figure size, (width, height) in inches.
    map : str
        The type of map.
    nrows : int
        The number of rows in axis.
    ncols : int
        The number of columns in axis.
    index : int
        The index of the current axis in the subplot.
    regrid_shape : int
        Parameter controlling the accuracy of the plotted map.
    sharex : cartopy.mpl.geoaxes.GeoAxesSubplot / matplotlib.axes.Axes.axis
        The axis whose x-axis is shared.
    sharey : cartopy.mpl.geoaxes.GeoAxesSubplot / matplotlib.axes.Axes.axis
        The axis whose y-axis is shared.

    Returns
    -------
    ax : cartopy.mpl.geoaxes.GeoAxesSubplot object
        The axis.

    """
    # Coordinate systems for transformations
    crs_lon_lat = ccrs.PlateCarree()
    aeqd = ccrs.AzimuthalEquidistant(central_longitude=centerpoint[0],
                                     central_latitude=centerpoint[1])

    # Calculate the extent for the map based on the given extent
    # from center in km
    xmin = ymin = - extent_km * 1e3
    xmax = ymax = extent_km * 1e3

    maxcoord = crs_lon_lat.transform_points(aeqd, np.array(xmax),
                                            np.array(ymax))
    mincoord = crs_lon_lat.transform_points(aeqd, np.array(xmin),
                                            np.array(ymin))
    lonmax = maxcoord[0][0]
    latmax = maxcoord[0][1]
    lonmin = mincoord[0][0]
    latmin = mincoord[0][1]
    ext = (lonmin, lonmax, latmin, latmax)

    # Create axis
    if fig is None:
        fig = pl.figure(figsize=figsize)

    ax = fig.add_subplot(nrows, ncols, index, projection=aeqd,
                         sharex=sharex, sharey=sharey)
    ax.set_extent(ext)

    if not no_map:
        # Get map background
        actual_tiler = cimgt.Stamen(map)
        tiler = CachedTiler(actual_tiler)
        # Add the Stamen data at zoom level
        ax.add_image(tiler, zoom)
        # , interpolation='spline36',
        # regrid_shape=regrid_shape)
    return ax, fig, aeqd, ext


def set_ticks_lon_lat(ax, extent, lon_nticks, lat_nticks, format=r"%.1f",
                      label_fontsize=8):
    """Set ticks as lon, lat to axis.

    Parameters
    ----------
    ax : cartopy.mpl.geoaxes.GeoAxesSubplot object
        The axis.
    extent : array-like
        The extent of the axis in coordinates.
    lon_nticks : int
        Number of ticks in x axis.
    lat_nticks : int
        Number of ticks in y axis.

    """
    crs_lon_lat = ccrs.PlateCarree()

    if lat_nticks > 0:
        ticks_lat = np.linspace(extent[2], extent[3], lat_nticks)
        ticks_y = ax.projection.transform_points(
            crs_lon_lat, extent[0] * np.ones(ticks_lat.shape), ticks_lat)[:, 1]
        ax.set_yticks(ticks_y, minor=False)
        fmt_E = format + "$^\circ$E"
        ax.set_yticklabels([fmt_E % c for c in ticks_lat], fontsize=8)
        ax.text(-0.20, 0.5, 'Latitude', va='bottom', ha='center',
                rotation='vertical', rotation_mode='anchor',
                transform=ax.transAxes, fontsize=label_fontsize)

    if lon_nticks > 0:
        ticks_lon = np.linspace(extent[0], extent[1], lon_nticks)
        ticks_x = ax.projection.transform_points(
            crs_lon_lat, ticks_lon, extent[2] * np.ones(ticks_lon.shape))[:, 0]
        ax.set_xticks(ticks_x, minor=False)
        fmt_N = format + "$^\circ$N"
        ax.set_xticklabels([fmt_N % c for c in ticks_lon], fontsize=8)
        ax.text(0.5, -0.15, 'Longitude', va='bottom', ha='center',
                rotation='horizontal', rotation_mode='anchor',
                transform=ax.transAxes, fontsize=label_fontsize)


def set_ticks_km(ax, extent, x_nticks, y_nticks):
    """Set ticks as lon, lat to axis.

    Parameters
    ----------
    ax : cartopy.mpl.geoaxes.GeoAxesSubplot object
        The axis.
    extent : array-like
        The extent of the axis in coordinates.
    x_nticks : int
        Number of ticks in x axis.
    y_nticks : int
        Number of ticks in y axis.

    """
    ticks_x = np.linspace(extent[0], extent[1], x_nticks)
    ticks_y = np.linspace(extent[2], extent[3], y_nticks)

    ax.set_xticks(ticks_x, minor=False)
    ax.set_yticks(ticks_y, minor=False)
    ax.set_xticklabels([r"%.0f" % c for c in ticks_x], fontsize=8)
    ax.set_yticklabels([r"%.0f" % c for c in ticks_y], fontsize=8)

    # ax.text(-0.14, 0.55, 'Latitude', va='bottom', ha='center',
    #         rotation='vertical', rotation_mode='anchor',
    #         transform=ax.transAxes, fontsize=8)
    # ax.text(0.5, -0.1, 'Longitude', va='bottom', ha='center',
    #         rotation='horizontal', rotation_mode='anchor',
    #         transform=ax.transAxes, fontsize=8)


def plot_pdf(fig, outfn, dpi=200, bbox_inches="tight"):
    """Plot the images as pdf.

    Parameters
    ----------
    fig : matplotlib Figure object
        The figure to be plotted.
    outfn : str
        The full path to the output file.
    dpi : int
        Dots per inch number.

    """
    fig.savefig(outfn, bbox_inches=bbox_inches, dpi=dpi)


def get_data_stamps(grid, ind):
    """Get time and altitude for the grid to put into plots.

    Parameters
    ----------
    grid : pyart.core.grid.Grid object
        The grid.
    ind : int
        The index of the z-level that is used.

    Returns
    -------
    altstamp : str
        String for altitude
    timestamp : str
        String for date, rounded to 5 minutes.

    """
    # Altitude
    altstamp = "%d m level" % (grid.z['data'][ind])

    # Date rounded to 5 minutes
    dt = datetime.strptime(grid.time['units'],
                           'seconds since %Y-%m-%dT%H:%M:%SZ')
    new_minute = (dt.minute // 5) * 5
    date = dt + timedelta(minutes=new_minute - dt.minute)
    timestamp = datetime.strftime(date, "%Y%m%d%H%M UTC")
    return altstamp, timestamp


def plot_ppi(ax, data, az, r, offset=None, **kwargs):
    """Plot a PPI image.

    Parameters
    ----------
    ax : cartopy.mpl.geoaxes.GeoAxesSubplot / matplotlib.axes.Axes.axis
        The axis.
    R : (N,M) numpy array
        The data that is plotted.
    az : (N,) numpy array
        The azimuth angles of the data, for edges of bins.
    r : (M,) numpy array
        The ranges of the bins of the data, for edges of bins.
    offset: array-like, 2x1
        The offset of the coordinates from the origin.
    **kwargs : dict
        Passed to matplotlib.pyplot.pcolormesh.

    Returns
    -------
    pm : matplotlib.collections.QuadMesh
        The colormesh plot object handle.

    """
    if az.shape[0] == data.shape[0]:
        # Pad to add endpoints
        az = np.pad(az, ((0, 1)), mode='wrap')
    if r.shape[0] == data.shape[1]:
        r = np.pad(r, ((0, 1)), mode='constant',
                   constant_values=(r[-1] + (r[1] - r[0])))
    R, AZ = np.meshgrid(r, az)
    biny = R * np.cos(np.radians(AZ))
    binx = R * np.sin(np.radians(AZ))
    if offset is not None:
        binx -= offset[0]
        biny -= offset[1]
    pm = ax.pcolormesh(binx, biny, data, **kwargs)
    return pm


def plot_scan_area(ax, rmin, rmax, azlims=None, offset=(0, 0), **kwargs):
    """Plot a scan area.

    Parameters
    ----------
    ax : cartopy.mpl.geoaxes.GeoAxesSubplot
        The axis.
    rmin : float
        The minimum range, in meters.
    rmax : float
        The maximum range, in meters.
    azlims : (2,) array-like
        The minimum and maximum azimuth degrees. Should be given
        clock-wise, second value can exceed 360 degrees.
        If not given, full PPI scan assumed.
    offset : (2,) array-like
        The offset of the coordinates from the origin.
    **kwargs : dict
        Passed to pyplot.plot

    """
    def bin_coords(R, AZ):
        binx = R * np.sin(np.radians(AZ)) - offset[0]
        biny = R * np.cos(np.radians(AZ)) - offset[1]
        return binx, biny

    if azlims is None:
        azlims = (0, 360)
    az = np.linspace(azlims[0], azlims[1], 100)

    # Plot inner circle if rmin != 0
    if rmin > 0:
        R, AZ = np.meshgrid(rmin, az)
        binx, biny = bin_coords(R, AZ)
        ax.plot(binx, biny, **kwargs)

    # Plot outer circle
    R, AZ = np.meshgrid(rmax, az)
    binx, biny = bin_coords(R, AZ)
    ax.plot(binx, biny, **kwargs)

    # Plot sidelines if not full circle
    if azlims != (0, 360):
        xs, ys = bin_coords(rmin, azlims[0])
        xe, ye = bin_coords(rmax, azlims[0])
        ax.plot([xs, xe], [ys, ye], **kwargs)

        xs, ys = bin_coords(rmin, azlims[1])
        xe, ye = bin_coords(rmax, azlims[1])
        ax.plot([xs, xe], [ys, ye], **kwargs)

    return


def plot_radar_locations(coords, geoaxis, **kwargs):
    """Plot radar locations given in lon, lat in coords.

    Parameters
    ----------
    coords : nx2 numpy array
        The coordinates of the radar locations.
    geoaxis : cartopy.mpl.geoaxes.GeoAxesSubplot object
        The axis.
    **kwargs : dict
        Passed to matplotlib.pyplot.scatter.

    """
    map_coords = geoaxis.projection.transform_points(
        ccrs.PlateCarree(), coords[:, 0], coords[:, 1])
    geoaxis.scatter(map_coords[:, 0], map_coords[:, 1], **kwargs)


def plot_contours(contours, X, Y, ax, closed=True, **kwargs):
    """Plot contours given by index values onto a rectangular grid.

    Parameters:
    -----------
    contours : ndarray or list of ndarrays
        The contours as (..., 2) shaped arrays.
    X : ndarray
        The x-coordinates for the grid.
    Y : ndarray
        The y-coordinates for the grid.
    ax : cartopy.mpl.geoaxes.GeoAxesSubplot / matplotlib.axes.Axes.axis
        The axis.
    closed : bool
        Whether the contour is plotted as a closed line. Default True.
    **kwargs : dict
        Passed to matplotlib.pyplot.plot.

    """
    if not isinstance(contours, list):
        contours = [contours]

    xlim = X.shape[0]
    ylim = Y.shape[0]

    for n, cont in enumerate(contours):
        if closed:
            # Make sure contour is plotted as closed
            if np.any(cont[-1, ...] != cont[0, ...]):
                cont = np.concatenate((cont, cont[np.newaxis, 0, ...]), axis=0)
        xx = np.squeeze((cont[..., 0] / xlim) * xlim).astype(int)
        yy = np.squeeze((cont[..., 1] / ylim) * ylim).astype(int)
        ax.scatter(X[xx], Y[yy], **kwargs)
    return


def plot_contours_geo(contours, ax, closed=True, **kwargs):
    """Plot contours given by geographical coordinates (meters).

    Parameters:
    -----------
    contours : ndarray or list of ndarrays
        The contours as (..., 2) shaped arrays.
    ax : cartopy.mpl.geoaxes.GeoAxesSubplot / matplotlib.axes.Axes.axis
        The axis.
    closed : bool
        Whether the contour is plotted as a closed line. Default True.
    **kwargs : dict
        Passed to matplotlib.pyplot.plot.

    """
    if not isinstance(contours, list):
        contours = [contours]

    for n, cont in enumerate(contours):
        if closed:
            # Make sure contour is plotted as closed
            if np.any(cont[-1, ...] != cont[0, ...]):
                cont = np.concatenate((cont, cont[np.newaxis, 0, ...]), axis=0)
        xx = np.squeeze(cont[..., 0])
        yy = np.squeeze(cont[..., 1])
        ax.plot(xx, yy, **kwargs)
    return


def plot_contours_polar(contours, AZ, R, ax, closed=True, **kwargs):
    """Plot contours given by index values onto a polar grid.

    Parameters:
    -----------
    contours : ndarray or list of ndarrays
        The contours as (..., 2) shaped arrays.
    AZ : ndarray
        The azimuth values for the grid points.
    R : ndarray
        The range values for the grid points.
    ax : cartopy.mpl.geoaxes.GeoAxesSubplot / matplotlib.axes.Axes.axis
        The axis.
    closed : bool
        Whether the contour is plotted as a closed line. Default True.
    **kwargs : dict
        Passed to matplotlib.pyplot.plot.

    """
    if not isinstance(contours, list):
        contours = [contours]
    biny = R * np.cos(np.radians(AZ))
    binx = R * np.sin(np.radians(AZ))

    for n, cont in enumerate(contours):
        if closed:
            # Make sure contour is plotted as closed
            if np.any(cont[-1, ...] != cont[0, ...]):
                cont = np.concatenate((cont, cont[np.newaxis, 0, ...]), axis=0)
        xx = np.squeeze(cont[..., 0])
        yy = np.squeeze(cont[..., 1])
        ax.scatter(binx[yy, xx], biny[yy, xx], **kwargs)
    return


class MidpointNormalize(mlt.colors.Normalize):
    """Normalise the colorbar around midpoint.

    Normalise the colorbar around so that diverging bars work there way
    either side from a prescribed midpoint value).
    Taken from http://chris35wills.github.io/matplotlib_diverging_colorbar/.

    Example
    ```
    im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    ```
    """

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        """Initialize norm."""
        self.midpoint = midpoint
        mlt.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        """Call function."""
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))
