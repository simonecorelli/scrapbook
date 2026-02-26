#!/usr/bin/env python3
"""
CIE 1931 — diagramma di cromaticità con esagoni colorati.

Ogni esagono è posizionato in xyY; il NOME del colore viene derivato
automaticamente dalla posizione tramite CIE Lab → angolo di tinta LCh.

Rendering in sRGB solo per visualizzazione in PDF (matplotlib non supporta
profili ICC embedded). Il colore VERO è quello in xyY; sRGB è la miglior
approssimazione stampabile/visualizzabile da matplotlib.

Gamut P3 e ProPhoto: dai dati interni di colour-science (nessun file ICC).
Gamut Fogra39: poligono approssimato (profilo ICC ~2MB, non embeddabile).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib.path import Path
import colour

# ── Illuminante D65 ───────────────────────────────────────────────────────────
D65 = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']

# ── Gamut da colour-science (primari in xy) ───────────────────────────────────
def _poly(cs_name):
    """Poligono chiuso (N+1,2) dalle primarie di uno spazio RGB."""
    p = colour.RGB_COLOURSPACES[cs_name].primaries   # shape (3,2) — xy
    return np.vstack([p, p[0]])

P3_POLY      = _poly('Display P3')
PROPHOTO_POLY = _poly('ProPhoto RGB')

# Fogra39 — poligono approssimato (ISO 12647-2, stampa offset su patinata)
FOGRA39_POLY = np.array([
    [0.170, 0.300], [0.218, 0.478], [0.310, 0.562],
    [0.430, 0.500], [0.520, 0.378], [0.370, 0.178],
    [0.170, 0.098], [0.170, 0.300],
])

# ── Punti campione in CIE xyY ─────────────────────────────────────────────────
#  Il nome verrà calcolato automaticamente dalla posizione.
#  Y (luminanza relativa, 0-1) influenza sia il nome sia il colore reso.
# Posizioni scelte in modo che nessun esagono si sovrapponga:
# distanza minima tra centri > 2·HEX_R·cos(30°) ≈ 0.087
# Il nome viene calcolato automaticamente da xyY → Lab → LCh.
# I colori FUORI dal gamut P3/Fogra39 vengono nominati ugualmente —
# il gamut influenza solo il colore reso (grigiatura), non il nome.
SAMPLES = [     # x,      y,      Y
    (0.618, 0.316, 0.21),  # rosso saturo
    (0.562, 0.408, 0.42),  # arancione
    (0.478, 0.358, 0.08),  # marrone  ← Y bassa = marrone, non arancione!
    (0.430, 0.490, 0.93),  # giallo   (h≈92° in LCh)
    (0.348, 0.555, 0.70),  # giallo-verde
    (0.210, 0.720, 0.50),  # verde ~520nm
    (0.155, 0.430, 0.42),  # verde acqua
    (0.083, 0.338, 0.50),  # ciano ~490nm
    (0.175, 0.210, 0.22),  # azzurro
    (0.148, 0.055, 0.07),  # blu ~460nm
    (0.230, 0.083, 0.04),  # viola
    (0.305, 0.140, 0.08),  # porpora
    (0.380, 0.185, 0.18),  # magenta
    (0.418, 0.280, 0.38),  # rosa
    (0.313, 0.329, 1.00),  # bianco D65
]

HEX_R = 0.050   # raggio esagono — più grande per testo leggibile


# ═══════════════════════════════════════════════════════════════════════════════
# Naming automatico: xyY → Lab → LCh → nome italiano
# ═══════════════════════════════════════════════════════════════════════════════

# Soglie hue angle (°) → nome colore (CIE LCh D65)
# Calibrate sui colori sRGB primari:
#   Rosso puro ≈ 40°, Giallo ≈ 103°, Verde ≈ 136°,
#   Ciano ≈ 196°, Blu puro ≈ 306°, Magenta ≈ 328°
_HUE_NAMES = [
    (22,  "Rosso-Viola"),   #   0– 22°
    (48,  "Rosso"),          #  22– 48°
    (72,  "Arancione"),      #  48– 72°
    (98,  "Giallo"),         #  72– 98°  (include giallo-arancio)
    (135, "Verde-Giallo"),   #  98–135°
    (190, "Verde"),          # 135–190°
    (225, "Ciano"),          # 190–225°
    (270, "Azzurro"),        # 225–270°  (azzurro = celeste/cielo)
    (308, "Blu"),            # 270–308°  (blu puro sRGB ≈ 306°)
    (325, "Viola"),          # 308–325°
    (337, "Porpora"),        # 325–337°
    (360, "Magenta"),        # 337–360°
]

def name_from_lab(L, a, b):
    C = float(np.sqrt(a**2 + b**2))
    h = float(np.degrees(np.arctan2(b, a)) % 360)

    if C < 8:
        if L > 93:  return "Bianco"
        if L < 12:  return "Nero"
        if L > 65:  return "Grigio chiaro"
        if L < 40:  return "Grigio scuro"
        return "Grigio"

    # Marrone: hue arancio/rosso-arancio + bassa luminanza + cromaticità sufficiente
    # Soglia bassa (h>15) per catturare anche rosso-arancio scuro
    if 15 < h < 76 and L < 46 and C > 10:
        return "Marrone"

    for thresh, name in _HUE_NAMES:
        if h < thresh:
            return name
    return "Rosso-Viola"


def name_from_xyY(x, y, Y):
    XYZ = xyY_to_XYZ(x, y, Y)
    Lab = colour.XYZ_to_Lab(XYZ, illuminant=D65)
    return name_from_lab(*Lab)


# ═══════════════════════════════════════════════════════════════════════════════
# Conversioni colore
# ═══════════════════════════════════════════════════════════════════════════════

def xyY_to_XYZ(x, y, Y):
    if y < 1e-9: return np.zeros(3)
    return np.array([(x/y)*Y, float(Y), ((1-x-y)/y)*Y])

_M_sRGB = np.array([
    [ 3.2406255, -1.5372080, -0.4986286],
    [-0.9689307,  1.8757561,  0.0415175],
    [ 0.0557101, -0.2040211,  1.0569959],
])

def srgb_gamma(c):
    c = np.clip(c, 0, None)
    return np.where(c <= 0.0031308, 12.92*c, 1.055*c**(1/2.4)-0.055)

def desaturate(lin_rgb):
    """Desatura verso bianco finché tutti i canali rientrano in [0,1]."""
    if np.all(lin_rgb >= -1e-4) and np.all(lin_rgb <= 1.0001):
        return np.clip(lin_rgb, 0, 1)
    white = np.ones(3) * np.clip(lin_rgb.max(), 0.05, 1.0)
    for a in np.linspace(0, 1, 400):
        b = (1-a)*lin_rgb + a*white
        if np.all(b >= -1e-4) and np.all(b <= 1.0001):
            return np.clip(b, 0, 1)
    return np.clip(lin_rgb, 0, 1)

def point_in_poly(pt, poly):
    x, y = pt; inside = False; j = len(poly)-1
    for i in range(len(poly)):
        xi,yi = poly[i]; xj,yj = poly[j]
        if (yi>y) != (yj>y) and x < (xj-xi)*(y-yi)/(yj-yi)+xi:
            inside = not inside
        j = i
    return inside

def xyY_to_display(x, y, Y, gamut_poly):
    """
    Converti in sRGB per visualizzazione.
    Dentro il gamut: colore corretto (o best-effort sRGB).
    Fuori gamut: desaturato + grigiato per indicarlo visivamente.
    Nota: sRGB è l'unica uscita possibile da matplotlib/PDF senza ICC workflow.
    """
    XYZ = xyY_to_XYZ(x, y, Y)
    lin = _M_sRGB @ XYZ
    in_g = point_in_poly((x, y), gamut_poly[:-1])
    rgb = desaturate(lin)
    if not in_g:
        rgb = rgb * 0.60 + 0.40 * np.array([0.80, 0.80, 0.80])
    return np.clip(srgb_gamma(rgb), 0, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# Luogo spettrale
# ═══════════════════════════════════════════════════════════════════════════════

def get_locus():
    cmfs = colour.colorimetry.MSDS_CMFS_STANDARD_OBSERVER[
        'CIE 1931 2 Degree Standard Observer']
    XYZ = cmfs.values
    s = XYZ.sum(axis=1, keepdims=True); s[s==0]=1
    return cmfs.wavelengths, (XYZ/s)[:,:2]

def locus_path():
    _, xy = get_locus()
    v = np.vstack([xy, xy[0]])
    c = [Path.MOVETO]+[Path.LINETO]*(len(v)-2)+[Path.CLOSEPOLY]
    return Path(v, c)


# ═══════════════════════════════════════════════════════════════════════════════
# Testo auto-scalato nell'esagono (punta in alto, testo ruotato 90°)
# ═══════════════════════════════════════════════════════════════════════════════

def add_labels(ax, cx, cy, name, fill_rgb):
    """
    Due righe nell'esagono:
    - Nome: ruotato 90°, metà superiore (usa l'asse lungo).
    - Coordinate xy: ORIZZONTALI, metà inferiore (usa la larghezza piena).
    Approccio separato perché testo ruotato a font piccolo risulta invisibile in PDF.
    """
    fig = ax.figure
    aw = fig.get_figwidth()  * ax.get_position().width
    ah = fig.get_figheight() * ax.get_position().height
    x0,x1 = ax.get_xlim(); y0,y1 = ax.get_ylim()
    sx = aw/(x1-x0); sy = ah/(y1-y0)          # pollici per unità dati

    # Dimensioni esagono punta-in-alto in pollici
    h_tall = 2*HEX_R*sy           # asse lungo (verticale)
    h_wide = HEX_R*np.sqrt(3)*sx  # asse corto (orizzontale = larghezza massima)

    W = 0.52   # rapporto larghezza/altezza carattere

    # — Nome colore: ruotato 90°, parte superiore —
    n1 = max(len(name), 1)
    fs1 = float(np.clip(min(
        h_tall * 0.50 * 72 / (n1 * W),   # la stringa deve stare nell'asse lungo
        h_wide * 0.36 * 72               # il font non supera la larghezza
    ), 6.0, 14.0))

    # — Coordinate xy: ORIZZONTALI, parte inferiore —
    # Senza spazio dopo virgola per ridurre i caratteri ("0.48,0.36" = 9 car.)
    xy_lbl = f"{cx:.2f},{cy:.2f}"
    n2 = max(len(xy_lbl), 1)
    fs2 = float(np.clip(min(
        h_wide * 0.80 * 72 / (n2 * W),   # stringa orizzontale entro la larghezza
        h_tall * 0.22 * 72               # font non supera un quinto dell'altezza
    ), 5.5, 10.0))

    lum = 0.299*fill_rgb[0] + 0.587*fill_rgb[1] + 0.114*fill_rgb[2]
    tc = 'white' if lum < 0.46 else 'black'

    # Nome: centro spostato verso l'alto
    ax.text(cx, cy + HEX_R * 0.20, name,
            fontsize=fs1, fontweight='bold', color=tc,
            ha='center', va='center', rotation=90, zorder=4, clip_on=False)

    # Coordinate: orizzontali, spostate verso il basso
    ax.text(cx, cy - HEX_R * 0.50, xy_lbl,
            fontsize=fs2, color=tc,
            ha='center', va='center', rotation=0, zorder=4, clip_on=False)


# ═══════════════════════════════════════════════════════════════════════════════
# Disegno principale
# ═══════════════════════════════════════════════════════════════════════════════

def draw(ax, title, gamut_poly, gamut_label, gamut_color):
    wl, lxy = get_locus()
    lpath = locus_path()

    # Luogo spettrale + linea delle purpuree
    ax.plot(lxy[:,0], lxy[:,1], 'k-', lw=1.2, zorder=2)
    ax.plot([lxy[0,0],lxy[-1,0]], [lxy[0,1],lxy[-1,1]], 'k-', lw=1.2, zorder=2)

    # Etichette nm sul luogo
    for nm in [460,480,500,520,540,560,580,600,620,650]:
        idx = np.argmin(np.abs(wl-nm))
        ax.annotate(str(nm), xy=(lxy[idx,0],lxy[idx,1]),
                    fontsize=5, color='#666', xytext=(3,3),
                    textcoords='offset points')

    # Gamut di riferimento
    ax.plot(PROPHOTO_POLY[:,0], PROPHOTO_POLY[:,1],
            color='#AAAAAA', lw=0.9, ls=':', label='ProPhoto', zorder=5)
    ax.plot(gamut_poly[:,0], gamut_poly[:,1],
            color=gamut_color, lw=2.2, ls='--', label=gamut_label, zorder=5)

    # Punto bianco D65
    ax.plot(0.3127, 0.3290, 'k+', ms=7, zorder=6)
    ax.annotate('D65', xy=(0.3127,0.3290), fontsize=5.5,
                xytext=(4,4), textcoords='offset points')

    # Esagoni
    for (x, y, Y) in SAMPLES:
        if not lpath.contains_point((x, y)):
            continue
        name   = name_from_xyY(x, y, Y)
        rgb    = xyY_to_display(x, y, Y, gamut_poly)
        in_g   = point_in_poly((x, y), gamut_poly[:-1])

        patch = RegularPolygon(
            (x, y), numVertices=6, radius=HEX_R,
            orientation=np.pi/6,          # punta in alto
            facecolor=rgb,
            edgecolor='#333' if in_g else '#999',
            linewidth=0.8  if in_g else 0.4,
            linestyle='-'  if in_g else ':',
            zorder=3,
        )
        ax.add_patch(patch)
        add_labels(ax, x, y, name, rgb)

    ax.set_xlim(-0.06, 0.84)
    ax.set_ylim(-0.06, 0.94)
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('y', fontsize=11)
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=7, framealpha=0.8)
    ax.grid(True, lw=0.25, alpha=0.35)


# ═══════════════════════════════════════════════════════════════════════════════
# Genera i 3 PDF
# ═══════════════════════════════════════════════════════════════════════════════

def save(fname, title, gamut_poly, gamut_label, gamut_color,
         extra_poly=None, extra_label=None, extra_color=None):
    fig, ax = plt.subplots(figsize=(11, 11))
    draw(ax, title, gamut_poly, gamut_label, gamut_color)
    if extra_poly is not None:
        ax.plot(extra_poly[:,0], extra_poly[:,1],
                color=extra_color, lw=2.0, ls='-.', label=extra_label, zorder=5)
        ax.legend(loc='upper right', fontsize=7, framealpha=0.8)
    fig.tight_layout()
    fig.savefig(fname, format='pdf', dpi=300)
    plt.close(fig)
    print(f"✓ {fname}")


if __name__ == '__main__':
    save('cie1931_P3.pdf',
         'CIE 1931 — Display P3 gamut',
         P3_POLY, 'Display P3', '#CC5500')

    save('cie1931_Fogra39.pdf',
         'CIE 1931 — Fogra39 (CMYK offset patinata)',
         FOGRA39_POLY, 'Fogra39 ≈', '#005599')

    save('cie1931_P3_Fogra39.pdf',
         'CIE 1931 — P3 + Fogra39',
         P3_POLY, 'Display P3', '#CC5500',
         extra_poly=FOGRA39_POLY, extra_label='Fogra39 ≈', extra_color='#005599')
