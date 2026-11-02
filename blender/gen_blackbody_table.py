# blackbody -> linear rec.709 lookup table inlined in kerr_bh.osl (bb_rgb).
# planck spectrum x cie 1931 2-degree cmf (multi-lobe gaussian fit, wyman, sloan & shirley 2013),
# xyz -> linear rec.709 (d65), desaturated toward grey at constant Y where the locus leaves the gamut.
# prints log Y(T) relative to 6500 K and the chromaticity rgb / Y on a log-spaced T grid.
#   python gen_blackbody_table.py
import numpy as np

h, c, k = 6.62607015e-34, 2.99792458e8, 1.380649e-23

def g(x, mu, s1, s2):
    s = np.where(x < mu, s1, s2)
    return np.exp(-0.5 * ((x - mu) / s) ** 2)

def cmf(lam):  # lam in nm
    x = 1.056*g(lam, 599.8, 37.9, 31.0) + 0.362*g(lam, 442.0, 16.0, 26.7) - 0.065*g(lam, 501.1, 20.4, 26.2)
    y = 0.821*g(lam, 568.8, 46.9, 40.5) + 0.286*g(lam, 530.9, 16.3, 31.1)
    z = 1.217*g(lam, 437.0, 11.8, 36.0) + 0.681*g(lam, 459.0, 26.0, 13.8)
    return x, y, z

M = np.array([[3.2404542, -1.5371385, -0.4985314],
              [-0.9692660, 1.8760108, 0.0415560],
              [0.0556434, -0.2040259, 1.0572252]])

lam = np.linspace(360, 830, 941)
X_, Y_, Z_ = cmf(lam)
def xyz(T):
    L = lam * 1e-9
    B = 2*h*c**2 / L**5 / np.expm1(h*c/(L*k*T))
    return np.array([np.trapezoid(B*X_, lam), np.trapezoid(B*Y_, lam), np.trapezoid(B*Z_, lam)])

TMIN, TMAX, N = 1000.0, 100000.0, 48
Ts = np.exp(np.linspace(np.log(TMIN), np.log(TMAX), N))
Yref = xyz(6500.0)[1]
logY, rr, gg, bb = [], [], [], []
for T in Ts:
    XYZ = xyz(T) / Yref
    rgb = M @ XYZ
    Y = XYZ[1]
    chrom = rgb / Y
    if chrom.min() < 0:  # desaturate toward grey at the same Y until in gamut
        t = -chrom.min() / (1.0 - chrom.min())
        chrom = chrom * (1 - t) + t
    logY.append(np.log(Y)); rr.append(chrom[0]); gg.append(chrom[1]); bb.append(chrom[2])

def arr(name, v):
    return f"    float {name}[{N}] = {{" + ", ".join(f"{x:.6g}" for x in v) + "};"

if __name__ == "__main__":
    print(f"    // blackbody table: T = exp(linspace(log {TMIN:g}, log {TMAX:g}, {N})), Y rel. to 6500 K")
    print(arr("BB_LOGY", logY)); print(arr("BB_R", rr)); print(arr("BB_G", gg)); print(arr("BB_B", bb))
