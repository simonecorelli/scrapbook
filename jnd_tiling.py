#!/usr/bin/env python3
"""
CIE 1931 Chromaticity Diagram with Hexagonal Color Tiling.

Generates PDF outputs with Display P3 and Fogra39 gamut boundaries.
Colors defined in xyY, rendered correctly inside gamut,
approximated to nearest in-gamut color outside.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib.path import Path
from matplotlib.collections import PatchCollection
import colour

# ---------------------------------------------------------------------------
# 1. Named colors defined in xyY (CIE 1931 x, y, Y)
#    Y = luminance factor [0..1]
# ---------------------------------------------------------------------------
NAMED_COLORS = {
    "Rosso":            {"xy": (0.640, 0.330), "Y": 0.21},
    "Arancione":        {"xy": (0.550, 0.400), "Y": 0.45},
    "Marrone":          {"xy": (0.500, 0.380), "Y": 0.08},
    "Giallo":           {"xy": (0.419, 0.505), "Y": 0.93},
    "Giallo-Verde":     {"xy": (0.354, 0.566), "Y": 0.75},
    "Verde":            {"xy": (0.270, 0.630), "Y": 0.50},
    "Verde acqua":      {"xy": (0.226, 0.470), "Y": 0.55},
    "Ciano":            {"xy": (0.200, 0.330), "Y": 0.50},
    "Azzurro":          {"xy": (0.175, 0.220), "Y": 0.25},
    "Blu":              {"xy": (0.150, 0.060), "Y": 0.07},
    "Viola":            {"xy": (0.220, 0.100), "Y": 0.05},
    "Porpora":          {"xy": (0.320, 0.150), "Y": 0.10},
    "Magenta":          {"xy": (0.380, 0.180), "Y": 0.20},
    "Rosa":             {"xy": (0.400, 0.250), "Y": 0.40},
    "Bianco":           {"xy": (0.333, 0.333), "Y": 1.00},
    "Rosa salmone":     {"xy": (0.440, 0.310), "Y": 0.50},
}

# ---------------------------------------------------------------------------
# 2. Gamut primaries in CIE xy
# ---------------------------------------------------------------------------
# Display P3 (DCI-P3 D65)
P3_PRIMARIES = np.array([
    [0.680, 0.320],  # R
    [0.265, 0.690],  # G
    [0.150, 0.060],  # B
    [0.680, 0.320],  # close polygon
])

# Fogra39 (ISO 12647-2, approximate CMYK gamut in xy at mid-luminance)
# Vertices: Cyan, Green(C+Y), Yellow, Red(M+Y), Magenta(C+M), Blue-ish
FOGRA39_GAMUT = np.array([
    [0.170, 0.300],  # Cyan
    [0.310, 0.560],  # Green (C+Y)
    [0.430, 0.500],  # Yellow
    [0.520, 0.380],  # Red (M+Y)
    [0.370, 0.180],  # Magenta
    [0.170, 0.100],  # Blue (C+M)
    [0.170, 0.300],  # close polygon
])

# sRGB for reference
SRGB_PRIMARIES = np.array([
    [0.640, 0.330],  # R
    [0.300, 0.600],  # G
    [0.150, 0.060],  # B
    [0.640, 0.330],  # close polygon
])


def get_spectral_locus():
    """Return CIE 1931 spectral locus xy coordinates."""
    cmfs = colour.colorimetry.MSDS_CMFS_STANDARD_OBSERVER[
        "CIE 1931 2 Degree Standard Observer"
    ]
    wl = cmfs.wavelengths
    XYZ = cmfs.values
    denom = XYZ.sum(axis=1, keepdims=True)
    denom[denom == 0] = 1
    xy = (XYZ / denom)[:, :2]
    return wl, xy


def spectral_locus_path():
    """Return a matplotlib Path for the visible gamut (spectral locus + purple line)."""
    _, xy = get_spectral_locus()
    # close with purple line
    verts = np.vstack([xy, xy[0]])
    codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 2) + [Path.CLOSEPOLY]
    return Path(verts, codes)


def xyY_to_XYZ(x, y, Y):
    """Convert CIE xyY to XYZ."""
    if y == 0:
        return np.array([0.0, 0.0, 0.0])
    X = (x / y) * Y
    Z = ((1 - x - y) / y) * Y
    return np.array([X, Y, Z])


def XYZ_to_linear_sRGB(XYZ):
    """XYZ (D65) to linear sRGB."""
    M = np.array([
        [ 3.2406255, -1.5372080, -0.4986286],
        [-0.9689307,  1.8757561,  0.0415175],
        [ 0.0557101, -0.2040211,  1.0569959],
    ])
    return M @ XYZ


def XYZ_to_linear_P3(XYZ):
    """XYZ (D65) to linear Display P3."""
    M = np.array([
        [ 2.4934969, -0.9313836, -0.4027108],
        [-0.8294890,  1.7626641,  0.0236247],
        [ 0.0358458, -0.0761724,  0.9568845],
    ])
    return M @ XYZ


def linear_to_sRGB_gamma(c):
    """Apply sRGB gamma (companding)."""
    c = np.clip(c, 0, None)
    out = np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1.0/2.4) - 0.055)
    return np.clip(out, 0, 1)


def gamut_clip_nearest(linear_rgb):
    """Clip out-of-gamut linear RGB to nearest in-gamut by desaturating toward white."""
    rgb = linear_rgb.copy()
    if np.all((rgb >= 0) & (rgb <= 1)):
        return rgb
    # Desaturate: blend toward D65 white (equal-energy in linear)
    white = np.array([1.0, 1.0, 1.0]) * np.clip(rgb.max(), 0.01, 1.0)
    for alpha in np.linspace(0, 1, 200):
        blended = (1 - alpha) * rgb + alpha * white
        clipped = np.clip(blended, 0, 1)
        if np.allclose(blended, clipped, atol=0.01):
            return clipped
    return np.clip(rgb, 0, 1)


def point_in_polygon(point, polygon):
    """Ray-casting test for point in polygon."""
    x, y = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def get_color_for_gamut(x, y, Y, gamut_name, gamut_polygon):
    """
    Convert xyY to display RGB for a given gamut.
    If inside gamut polygon -> correct color.
    If outside -> desaturated approximation.
    """
    XYZ = xyY_to_XYZ(x, y, Y)
    in_gamut = point_in_polygon((x, y), gamut_polygon[:-1])  # exclude closing vertex

    if gamut_name == "P3":
        linear = XYZ_to_linear_P3(XYZ)
    else:
        # For Fogra39 and sRGB, use sRGB matrix as display approximation
        linear = XYZ_to_linear_sRGB(XYZ)

    if in_gamut:
        rgb = gamut_clip_nearest(linear)
    else:
        # Desaturate more aggressively for out-of-gamut
        rgb = gamut_clip_nearest(linear)
        # Dim slightly to signal out-of-gamut
        rgb = rgb * 0.7 + 0.3 * np.array([0.8, 0.8, 0.8])

    return linear_to_sRGB_gamma(rgb)


def draw_diagram(ax, gamut_name, gamut_polygon, gamut_color, hex_radius=0.038):
    """Draw the CIE diagram with hexagonal color tiles for a specific gamut."""
    wl, locus_xy = get_spectral_locus()
    locus_path = spectral_locus_path()

    # Draw spectral locus
    ax.plot(locus_xy[:, 0], locus_xy[:, 1], 'k-', linewidth=1.2)
    # Purple line
    ax.plot([locus_xy[0, 0], locus_xy[-1, 0]],
            [locus_xy[0, 1], locus_xy[-1, 1]], 'k-', linewidth=1.2)

    # Wavelength labels on locus
    label_wl = [460, 480, 500, 520, 540, 560, 580, 600, 620, 650]
    for wl_val in label_wl:
        idx = np.argmin(np.abs(wl - wl_val))
        ax.annotate(f"{int(wl_val)}",
                     xy=(locus_xy[idx, 0], locus_xy[idx, 1]),
                     fontsize=5, color='gray',
                     xytext=(5, 5), textcoords='offset points')

    # Draw gamut boundary
    ax.plot(gamut_polygon[:, 0], gamut_polygon[:, 1],
            color=gamut_color, linewidth=2.0, linestyle='--',
            label=f'{gamut_name} gamut', zorder=5)

    # Draw named color hexagons
    for name, cdata in NAMED_COLORS.items():
        cx, cy = cdata["xy"]
        Y = cdata["Y"]

        # Check if point is inside spectral locus
        if not locus_path.contains_point((cx, cy)):
            continue

        rgb = get_color_for_gamut(cx, cy, Y, gamut_name, gamut_polygon)
        in_gamut = point_in_polygon((cx, cy), gamut_polygon[:-1])

        hex_patch = RegularPolygon(
            (cx, cy), numVertices=6, radius=hex_radius,
            orientation=0,
            facecolor=rgb, edgecolor='black',
            linewidth=0.8 if in_gamut else 0.4,
            linestyle='-' if in_gamut else ':',
            zorder=3,
        )
        ax.add_patch(hex_patch)

        # Determine text color (white on dark, black on light)
        lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        tc = 'white' if lum < 0.45 else 'black'

        # Color name
        ax.text(cx, cy + 0.006, name,
                ha='center', va='center', fontsize=4.5,
                fontweight='bold', color=tc, zorder=4)
        # xy coordinates below name
        ax.text(cx, cy - 0.010, f"({cx:.3f}, {cy:.3f})",
                ha='center', va='center', fontsize=3.2,
                color=tc, zorder=4)

    # Axes formatting
    ax.set_xlim(-0.02, 0.80)
    ax.set_ylim(-0.02, 0.90)
    ax.set_xlabel("x", fontsize=10)
    ax.set_ylabel("y", fontsize=10)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, linewidth=0.3, alpha=0.5)


def main():
    # --- PDF 1: Display P3 ---
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("CIE 1931 — Display P3 Gamut", fontsize=13)
    draw_diagram(ax, "P3", P3_PRIMARIES, "#E07000")
    fig.tight_layout()
    fig.savefig("cie1931_P3.pdf", format='pdf', dpi=300)
    plt.close(fig)
    print("Saved: cie1931_P3.pdf")

    # --- PDF 2: Fogra39 ---
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("CIE 1931 — Fogra39 (CMYK) Gamut", fontsize=13)
    draw_diagram(ax, "Fogra39", FOGRA39_GAMUT, "#0060A0")
    fig.tight_layout()
    fig.savefig("cie1931_Fogra39.pdf", format='pdf', dpi=300)
    plt.close(fig)
    print("Saved: cie1931_Fogra39.pdf")

    # --- PDF 3: Both gamuts overlaid ---
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("CIE 1931 — P3 + Fogra39 Gamuts", fontsize=13)
    draw_diagram(ax, "P3", P3_PRIMARIES, "#E07000")
    # Overlay Fogra39 boundary
    ax.plot(FOGRA39_GAMUT[:, 0], FOGRA39_GAMUT[:, 1],
            color='#0060A0', linewidth=2.0, linestyle='-.',
            label='Fogra39 gamut', zorder=5)
    ax.legend(loc='upper right', fontsize=7)
    fig.tight_layout()
    fig.savefig("cie1931_P3_Fogra39.pdf", format='pdf', dpi=300)
    plt.close(fig)
    print("Saved: cie1931_P3_Fogra39.pdf")


if __name__ == "__main__":
    main()
