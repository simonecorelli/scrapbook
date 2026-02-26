#!/usr/bin/env python3
"""
CIE 1931 Chromaticity Diagram – hexagonal color tiling.

Colori definiti in CIE Lab (D65), convertiti automaticamente in xy.
Rendering sempre via XYZ→sRGB (corretto).
ProPhoto, P3 e Fogra39 come spazi di riferimento gamut.
Uscita: 3 PDF.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib.path import Path
import colour

# ── Illuminante di riferimento (D65) ─────────────────────────────────────────
D65 = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']

# ── Colori definiti in CIE Lab D65 + Y per il rendering ─────────────────────
#   Fonte: centroidi Munsell/ISCC-NBS convertiti in Lab; Y da standard sRGB/P3
#   (L*, a*, b*, Y_rendering)
COLORS = {
    "Rosso":       (50,  65,  45, 0.21),
    "Arancione":   (65,  40,  58, 0.42),
    "Marrone":     (30,  14,  20, 0.07),   # bassa luminanza → aspetto marrone
    "Giallo":      (97,  -6,  94, 0.93),
    "Giallo-verde":(78, -46,  68, 0.70),
    "Verde":       (54, -56,  38, 0.40),
    "Verde acqua": (70, -30, -10, 0.50),
    "Ciano":       (88, -26,  -6, 0.55),
    "Azzurro":     (58, -12, -44, 0.22),
    "Blu":         (30,  12, -62, 0.07),
    "Viola":       (26,  30, -50, 0.04),
    "Magenta":     (52,  67, -28, 0.19),
    "Rosa":        (73,  32,   6, 0.38),
    "Bianco":      (100,  0,   0, 1.00),
}

# ── Parametro globale esagono ─────────────────────────────────────────────────
HEX_R = 0.040   # raggio in unità CIE xy
# Con orientamento=np.pi/6 (punta in alto): altezza=2r, larghezza=r√3

# ── Gamut: vertici in CIE xy ──────────────────────────────────────────────────
# Display P3 (D65)
P3 = np.array([[0.680,0.320],[0.265,0.690],[0.150,0.060],[0.680,0.320]])

# ProPhoto / ROMM RGB (D50 – vertici "immaginari" fuori dal luogo spettrale)
PROPHOTO = np.array([[0.7347,0.2653],[0.1596,0.8404],[0.0366,0.0001],[0.7347,0.2653]])

# Fogra39 (gamut approssimato offset coated, da dati ICC/XYZ medi)
FOGRA39 = np.array([
    [0.170,0.300], [0.218,0.478], [0.310,0.560],
    [0.430,0.500], [0.520,0.378], [0.370,0.178],
    [0.170,0.098], [0.170,0.300],
])


# ═══════════════════════════════════════════════════════════════════════════════
# Conversioni colore
# ═══════════════════════════════════════════════════════════════════════════════

def lab_to_xy(L, a, b):
    """CIE Lab D65 → CIE xy (solo cromaticità)."""
    XYZ = colour.Lab_to_XYZ(np.array([L, a, b]), illuminant=D65)
    s = XYZ.sum()
    if s < 1e-9:
        return 0.3127, 0.3290   # D65 white
    return float(XYZ[0]/s), float(XYZ[1]/s)


def xyY_to_XYZ(x, y, Y):
    if y < 1e-9:
        return np.zeros(3)
    return np.array([(x/y)*Y, Y, ((1-x-y)/y)*Y])


# Matrici XYZ D65 → RGB lineare
_M_sRGB = np.array([
    [ 3.2406255, -1.5372080, -0.4986286],
    [-0.9689307,  1.8757561,  0.0415175],
    [ 0.0557101, -0.2040211,  1.0569959],
])
_M_P3 = np.array([
    [ 2.4934969, -0.9313836, -0.4027108],
    [-0.8294890,  1.7626641,  0.0236247],
    [ 0.0358458, -0.0761724,  0.9568845],
])
# ProPhoto usa D50: prima adattamento Bradford D65→D50, poi matrice ProPhoto
_M_bradford = np.array([
    [ 0.9555766, -0.0230393,  0.0631636],
    [-0.0282895,  1.0099416,  0.0210077],
    [ 0.0122982, -0.0204830,  1.3299098],
])
_M_ProPhoto_D50 = np.array([
    [ 1.3459433, -0.2556075, -0.0511118],
    [-0.5445989,  1.5081673,  0.0205351],
    [ 0.0000000,  0.0000000,  1.2118128],
])


def srgb_gamma(c):
    c = np.clip(c, 0, None)
    return np.where(c <= 0.0031308, 12.92*c, 1.055*c**(1/2.4) - 0.055)


def prophoto_gamma(c):
    c = np.clip(c, 0, None)
    return np.where(c < 0.001953125, 16*c, c**(1/1.8))


def desaturate_to_gamut(linear_rgb):
    """Desatura verso bianco finché tutti i canali sono in [0,1]."""
    if np.all((linear_rgb >= -1e-4) & (linear_rgb <= 1.0001)):
        return np.clip(linear_rgb, 0, 1)
    white = np.ones(3) * min(max(linear_rgb.max(), 0.05), 1.0)
    for alpha in np.linspace(0, 1, 300):
        blended = (1-alpha)*linear_rgb + alpha*white
        if np.all((blended >= -1e-4) & (blended <= 1.0001)):
            return np.clip(blended, 0, 1)
    return np.clip(linear_rgb, 0, 1)


def point_in_polygon(pt, poly):
    """Ray-casting point-in-polygon."""
    x, y = pt
    inside = False
    j = len(poly)-1
    for i in range(len(poly)):
        xi, yi = poly[i]; xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and x < (xj-xi)*(y-yi)/(yj-yi)+xi:
            inside = not inside
        j = i
    return inside


def xy_to_display_rgb(x, y, Y, gamut_poly):
    """
    Converti xyY → RGB sRGB per visualizzazione.
    Dentro il gamut: colore corretto.
    Fuori: desaturato + leggermente grigiato per indicarlo.
    """
    XYZ = xyY_to_XYZ(x, y, Y)
    in_gamut = point_in_polygon((x, y), gamut_poly[:-1])

    # Rendering SEMPRE via sRGB (fix bug versione precedente)
    lin = _M_sRGB @ XYZ

    if in_gamut:
        rgb = desaturate_to_gamut(lin)
    else:
        rgb = desaturate_to_gamut(lin)
        # Grigio leggero per colori fuori gamut
        rgb = rgb * 0.62 + 0.38 * np.array([0.80, 0.80, 0.80])

    return np.clip(srgb_gamma(rgb), 0, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# Luogo spettrale
# ═══════════════════════════════════════════════════════════════════════════════

def get_locus():
    cmfs = colour.colorimetry.MSDS_CMFS_STANDARD_OBSERVER[
        'CIE 1931 2 Degree Standard Observer']
    XYZ = cmfs.values
    s = XYZ.sum(axis=1, keepdims=True)
    s[s == 0] = 1
    return cmfs.wavelengths, (XYZ / s)[:, :2]


def locus_path():
    _, xy = get_locus()
    verts = np.vstack([xy, xy[0]])
    codes = [Path.MOVETO] + [Path.LINETO]*(len(verts)-2) + [Path.CLOSEPOLY]
    return Path(verts, codes)


# ═══════════════════════════════════════════════════════════════════════════════
# Testo auto-scalato nell'esagono
# ═══════════════════════════════════════════════════════════════════════════════

def add_hex_labels(ax, cx, cy, name, xy_label, rgb_fill, hex_r):
    """
    Due righe di testo ruotate 90° dentro l'esagono (punta in alto).
    Font auto-calcolato per riempire l'asse lungo (verticale = 2r).
    """
    # Dimensioni esagono in unità dati → pollici
    fig = ax.figure
    ax_w_in  = fig.get_figwidth()  * ax.get_position().width
    ax_h_in  = fig.get_figheight() * ax.get_position().height
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    scale_x = ax_w_in  / (x1 - x0)   # pollici per unità dati x
    scale_y = ax_h_in  / (y1 - y0)

    # Con orientamento=pi/6 (punta in alto):
    #   altezza (V-V) = 2*r, larghezza (F-F) = r*sqrt(3)
    hex_tall_in = 2 * hex_r * scale_y          # verticale disponibile
    hex_wide_in = hex_r * np.sqrt(3) * scale_x  # orizzontale disponibile

    # Rapporto larghezza/altezza carattere tipico
    W_RATIO = 0.52

    def best_font(text, avail_long, avail_short, fraction=1.0):
        """
        avail_long  = spazio disponibile lungo la direzione del testo (verticale)
        avail_short = spazio disponibile perpendicolare (orizzontale → altezza font)
        """
        n = max(len(text), 1)
        f_from_len  = avail_long  * fraction * 72 / (n * W_RATIO)
        f_from_height = avail_short * 72
        return float(np.clip(min(f_from_len, f_from_height), 3.0, 16.0))

    # Linea 1: nome colore (usa ~55% dell'altezza hex per il testo, 42% per font-height)
    fs1 = best_font(name,    hex_tall_in * 0.55, hex_wide_in * 0.40)
    # Linea 2: coordinate xy (usa ~45% dell'altezza, font più piccolo)
    fs2 = best_font(xy_label, hex_tall_in * 0.50, hex_wide_in * 0.32)

    lum = 0.299*rgb_fill[0] + 0.587*rgb_fill[1] + 0.114*rgb_fill[2]
    tc = 'white' if lum < 0.46 else 'black'

    # Offset per separare le due righe (in unità dati)
    off = hex_r * 0.22
    ax.text(cx, cy + off, name,
            ha='center', va='center',
            rotation=90, fontsize=fs1,
            fontweight='bold', color=tc, zorder=4,
            clip_on=False)
    ax.text(cx, cy - off, xy_label,
            ha='center', va='center',
            rotation=90, fontsize=fs2,
            color=tc, zorder=4,
            clip_on=False)


# ═══════════════════════════════════════════════════════════════════════════════
# Disegno diagramma
# ═══════════════════════════════════════════════════════════════════════════════

def draw(ax, title, gamut_poly, gamut_label, gamut_color):
    wl, locus_xy = get_locus()
    lpath = locus_path()

    # Luogo spettrale
    ax.plot(locus_xy[:,0], locus_xy[:,1], 'k-', lw=1.2, zorder=2)
    ax.plot([locus_xy[0,0], locus_xy[-1,0]],
            [locus_xy[0,1], locus_xy[-1,1]], 'k-', lw=1.2, zorder=2)

    # Etichette lunghezze d'onda
    for nm in [460, 480, 500, 520, 540, 560, 580, 600, 620, 650]:
        idx = np.argmin(np.abs(wl - nm))
        ax.annotate(str(nm), xy=(locus_xy[idx,0], locus_xy[idx,1]),
                    fontsize=5.5, color='#555555',
                    xytext=(4,4), textcoords='offset points')

    # Gamut boundary
    ax.plot(gamut_poly[:,0], gamut_poly[:,1],
            color=gamut_color, lw=2.2, ls='--',
            label=gamut_label, zorder=5)
    # ProPhoto per contesto (linea sottile grigia)
    ax.plot(PROPHOTO[:,0], PROPHOTO[:,1],
            color='#999999', lw=1.0, ls=':',
            label='ProPhoto', zorder=5)

    # Punti bianco D65
    ax.plot(0.3127, 0.3290, 'k+', ms=6, zorder=6)
    ax.annotate('D65', xy=(0.3127,0.3290), fontsize=5,
                xytext=(4,4), textcoords='offset points')

    # Esagoni colori
    for name, (L, a, b, Y) in COLORS.items():
        cx, cy = lab_to_xy(L, a, b)
        if not lpath.contains_point((cx, cy)):
            continue                       # fuori dal luogo spettrale: salta

        rgb = xy_to_display_rgb(cx, cy, Y, gamut_poly)
        in_g = point_in_polygon((cx, cy), gamut_poly[:-1])

        hex_patch = RegularPolygon(
            (cx, cy), numVertices=6, radius=HEX_R,
            orientation=np.pi/6,           # punta in alto
            facecolor=rgb,
            edgecolor='black' if in_g else '#888888',
            linewidth=0.8 if in_g else 0.4,
            linestyle='-' if in_g else ':',
            zorder=3,
        )
        ax.add_patch(hex_patch)

        xy_lbl = f"({cx:.3f}, {cy:.3f})"
        add_hex_labels(ax, cx, cy, name, xy_lbl, rgb, HEX_R)

    ax.set_xlim(-0.05, 0.82)
    ax.set_ylim(-0.05, 0.92)
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('y', fontsize=11)
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, lw=0.3, alpha=0.4)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def save_pdf(fname, title, gamut_poly, gamut_label, gamut_color):
    fig, ax = plt.subplots(figsize=(11, 11))
    draw(ax, title, gamut_poly, gamut_label, gamut_color)
    fig.tight_layout()
    fig.savefig(fname, format='pdf', dpi=300)
    plt.close(fig)
    print(f"Salvato: {fname}")


if __name__ == '__main__':
    save_pdf('cie1931_P3.pdf',
             'CIE 1931 — Display P3 gamut',
             P3, 'Display P3', '#D06000')

    save_pdf('cie1931_Fogra39.pdf',
             'CIE 1931 — Fogra39 (CMYK) gamut',
             FOGRA39, 'Fogra39', '#0060A0')

    save_pdf('cie1931_P3_Fogra39.pdf',
             'CIE 1931 — P3 + Fogra39',
             P3, 'Display P3', '#D06000')
    # Aggiungi Fogra39 overlay sull'ultimo
    fig2, ax2 = plt.subplots(figsize=(11, 11))
    draw(ax2, 'CIE 1931 — P3 + Fogra39', P3, 'Display P3', '#D06000')
    ax2.plot(FOGRA39[:,0], FOGRA39[:,1],
             color='#0060A0', lw=2.0, ls='-.', label='Fogra39', zorder=5)
    ax2.legend(loc='upper right', fontsize=7)
    fig2.tight_layout()
    fig2.savefig('cie1931_P3_Fogra39.pdf', format='pdf', dpi=300)
    plt.close(fig2)
    print("Salvato: cie1931_P3_Fogra39.pdf (overlay)")
