"""Utility functions."""
from matplotlib import cm, colors
import numpy as np
import pyart
from cmcrameri import cm as cm_crameri

alias_names = {
    "VRAD": [
        "velocity",
        "radial_wind_speed",
    ],
    "DBZH": [
        "reflectivity",
    ],
    "wind_shear": [
        "radial_shear",
        "azimuthal_shear",
    ],
    "total_shear": ["total_shear_thr", "tdwr_gfda_shear"],
}


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
    # "reflectivity": "{x:.0f}",
    "DBZH": "{x:.0f}",
    "VRAD": "{x:.0f}",
    "VRADH": "{x:.0f}",
    # "radial_wind_speed": "{x:.0f}",
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
    "cnr": "{x:.0f}",
    "wind_shear": "{x:.0f}",
    # "radial_shear": "{x:.0f}",
    # "azimuthal_shear": "{x:.0f}",
    "total_shear": "{x:.0f}",
    # "total_shear_thr": "{x:.0f}",
    "tdwr_gfda": "{x:.0f}",
}


QTY_RANGES = {
    "DBZH": (-15.0, 50.0),
    "HCLASS": (1.0, 6.0),
    "KDP": (-4.0, 8.0),
    "PHIDP": (0, 360.0),
    "RHOHV": (0.8, 1.0),
    "SQI": (0.0, 1.0),
    "TH": (-15.0, 60.0),
    "VRAD": (-30.0, 30.0),
    "VRADH": (-30.0, 30.0),
    # "radial_wind_speed": (-30.0, 30.0),
    "WRAD": (0.0, 5.0),
    "ZDR": (-8.0, 8.0),
    "SNR": (-30.0, 50.0),
    "LOG": (0.0, 50.0),
    "cnr": (-30, 0),
    "wind_shear": (-20, 20),
    # "azimuthal_shear": (-20, 20),
    "total_shear": (0, 15),
    # "total_shear_thr": (0, 20),
    "tdwr_gfda": (-1, 1),
}

COLORBAR_TITLES = {
    # "reflectivity": "Equivalent reflectivity factor [dBZ]",
    "DBZH": "Equivalent reflectivity factor [dBZ]",
    "HCLASS": "HydroClass",
    "KDP": "Specific differential phase [°/km]",
    "PHIDP": "Differential phase [°]",
    "RHOHV": "Copolar correlation coefficient",
    "SQI": "Normalized coherent power",
    "TH": "Total reflectivity factor [dBZ]",
    "VRAD": "Doppler velocity [m s$^{-1}$]",
    # "velocity": "Doppler velocity [m s$^{-1}$]",
    "VRADH": "Doppler velocity [m s$^{-1}$]",
    "WRAD": "Doppler spectrum width [m s$^{-1}$",
    "ZDR": "Differential reflectivity [dB]",
    "SNR": "Signal-to-noise ratio [dB]",
    "LOG": "LOG signal-to-noise ratio [dB]",
    # "radial_wind_speed": "Doppler velocity [m s$^{-1}$]",
    "cnr": "Carrier-to-noise ratio [dB]",
    "wind_shear": "Wind shear [m s$^{-1}$ km$^{-1}$]",
    # "azimuthal_shear": "Azimuthal wind shear [m s$^{-1}$ km$^{-1}$]",
    "total_shear": "Total wind shear [m s$^{-1}$ km$^{-1}$]",
    # "total_shear_thr": "",
    "tdwr_gfda": "TWDR gust front segments",
}

for name, alias_list in alias_names.items():
    for a in alias_list:
        COLORBAR_TITLES[a] = COLORBAR_TITLES[name]
        QTY_RANGES[a] = QTY_RANGES[name]
        QTY_FORMATS[a] = QTY_FORMATS[name]


def get_colormap(quantity):
    if quantity == "HCLASS":
        cmap = colors.ListedColormap(["r", "b", "g", "y", "k", "c"])
        norm = colors.BoundaryNorm(np.arange(0.5, 7.5), cmap.N)
    elif "VRAD" in quantity or quantity in alias_names["VRAD"]:
        # cmap = "pyart_BuDRd18"
        # cmap = cm_crameri.roma
        cmap = "cmc.roma_r"
        norm = None
    elif "DBZH" in quantity or quantity in alias_names["DBZH"]:
        bounds = np.arange(QTY_RANGES[quantity][0], QTY_RANGES[quantity][1] + 1, 2.5)
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=len(bounds))
        cmap = cm.get_cmap("pyart_HomeyerRainbow", len(bounds))
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
    elif quantity == "cnr":
        bounds = np.arange(QTY_RANGES[quantity][0], QTY_RANGES[quantity][1] + 0.1, 1.0)
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=len(bounds))
        cmap = cm.get_cmap("viridis", len(bounds))
    elif "wind_shear" in quantity or quantity in alias_names["wind_shear"]:
        cmap = "cmc.roma_r"
        norm = None
    elif "total_shear" == quantity or quantity in alias_names["total_shear"]:
        cmap = "cmc.hawaii_r"
        norm = None
    elif "total_shear_thr" in quantity:
        cmap = "tab20"
        norm = None
    elif "tdwr_gfda" == quantity:
        cmap = "tab20"
        norm = None
    else:
        cmap = cm.viridis
        norm = None

    return cmap, norm
