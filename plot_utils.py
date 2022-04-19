"""Utility functions."""
from matplotlib import cm, colors
import numpy as np
import pyart


PYART_FIELDS = {
    "DBZH": "reflectivity",
    "HCLASS": "radar_echo_classification",
    "KDP": "specific_differential_phase",
    "PHIDP": "differential_phase",
    "RHOHV": "cross_correlation_ratio",
    "SQI": "normalized_coherent_power",
    "TH": "total_power",
    "VRAD": "velocity",
    "WRAD": "spectrum_width",
    "ZDR": "differential_reflectivity",
    "SNR": "signal_to_noise_ratio",
    "LOG": "log_signal_to_noise_ratio",
}

PYART_FIELDS_ODIM = {
    "DBZH": "reflectivity_horizontal",
    "HCLASS": "radar_echo_classification",
    "KDP": "specific_differential_phase",
    "PHIDP": "differential_phase",
    "RHOHV": "cross_correlation_ratio",
    "SQI": "normalized_coherent_power",
    "TH": "total_power_horizontal",
    "VRAD": "velocity",
    "VRADH": "velocity_horizontal",
    "WRAD": "spectrum_width",
    "ZDR": "differential_reflectivity",
    "SNR": "signal_to_noise_ratio",
    "LOG": "log_signal_to_noise_ratio",
}


QTY_FORMATS = {
    "DBZH": "{x:.0f}",
    "VRAD": "{x:.0f}",
    "SNR": "{x:.0f}",
    "ZDR": "{x:.1f}",
    "RHOHV": "{x:.2f}",
    "KDP": "{x:.1f}",
    "HCLASS": "{x:1.0f}",
    "PHIDP": "{x:.2f}",
    "SQI": "{x:.1f}",
    "TH": "{x:.0f}",
    "WRAD": "{x:.1f}",
    "LOG": "{x:.0f}",
}


QTY_RANGES = {
    "DBZH": (-15.0, 60.0),
    "HCLASS": (1.0, 6.0),
    "KDP": (-4.0, 8.0),
    "PHIDP": (0, 360.0),
    "RHOHV": (0.8, 1.0),
    "SQI": (0.0, 1.0),
    "TH": (-15.0, 60.0),
    "VRAD": (-30.0, 30.0),
    "WRAD": (0.0, 5.0),
    "ZDR": (-4.0, 4.0),
    "SNR": (-30.0, 50.0),
    "LOG": (0.0, 50.0),
}

COLORBAR_TITLES = {
    "DBZH": "Equivalent reflectivity factor (dBZ)",
    "HCLASS": "HydroClass",
    "KDP": "Specific differential phase (degrees/km)",
    "PHIDP": "Differential phase (degrees)",
    "RHOHV": "Copolar correlation coefficient",
    "SQI": "Normalized coherent power",
    "TH": "Total reflectivity factor (dBZ)",
    "VRAD": "Radial velocity (m/s)",
    "WRAD": "Doppler spectrum width (m/s)",
    "ZDR": "Differential reflectivity (dB)",
    "SNR": "Signal-to-noise ratio (dB)",
    "LOG": "LOG signal-to-noise ratio (dB)",
}


def get_colormap(quantity):
    if quantity == "HCLASS":
        cmap = colors.ListedColormap(["r", "b", "g", "y", "k", "c"])
        norm = colors.BoundaryNorm(np.arange(0.5, 7.5), cmap.N)
    elif "VRAD" in quantity:
        cmap = "pyart_BuDRd18"
        norm = None
    elif "DBZH" in quantity:
        cmap = "pyart_NWSRef"
        norm = None
    elif quantity == "TH":
        cmap = "pyart_NWSRef"
        norm = None
    elif "SNR" in quantity or "LOG" in quantity:
        cmap = "pyart_Carbone17"
        norm = None
    elif quantity == "KDP":
        cmap = "pyart_Theodore16"
        norm = None
    elif quantity == "PHIDP":
        cmap = "pyart_Wild25"
        norm = None
    elif quantity == "RHOHV":
        bounds = [0.1, 0.2, 0.35, 0.5, 0.6, 0.7, 0.8, 0.9, 0.94, 0.96, 0.98, 0.99, 1.05]
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=len(bounds))
        cmap = cm.get_cmap("pyart_RefDiff", len(bounds))
    elif "WRAD" in quantity:
        cmap = "pyart_NWS_SPW"
        norm = None
    elif quantity == "ZDR":
        cmap = "pyart_RefDiff"
        norm = None
    else:
        cmap = cm.jet
        norm = None

    return cmap, norm
