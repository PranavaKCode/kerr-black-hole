# figure "bent glass" vs "bent spacetime". left: 64 nested glass shells with an ior rising toward the
# centre, rays refracted by snell's law. right: equatorial kerr null geodesics (a = 0.99) with the
# shader's hamiltonian / mino-time rk4 scheme in float64, drawn in kerr-schild cartesian coordinates
# (x + iy = (r + i a) e^{i phi_KS}) so rays cross the horizon smoothly.
#   python geodesic_diagram.py out.png
import sys, math
import numpy as np
import numba as nb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import font_manager

A = 0.99


@nb.njit(cache=True)
def rhs_eq(a, L, r, pr):
    # equatorial (theta = pi/2, p_theta = 0) photon, E = 1, mino time; returns dr, dphi_BL, dphi_KS, dpr
    E = 1.0
    r2 = r * r; a2 = a * a
    D = r2 - 2.0 * r + a2
    Dp = 2.0 * r - 2.0
    P = (r2 + a2) * E - a * L
    dr = D * pr
    dph = a * P / D + L - a * E
    dphks = dph + a * pr               # d(phi_KS) = d(phi_BL) + (a / Delta) dr
    dpr = 0.5 * (-Dp * pr * pr + 4.0 * r * E * P / D - P * P * Dp / (D * D))
    return dr, dph, dphks, dpr


@nb.njit(cache=True)
def trace_eq(a, b, x0, rmax, nmax):
    # photon from x = -inf travelling +x at height y = b; returns the kerr-schild x, y path and its fate
    rh = 1.0 + math.sqrt(1.0 - a * a)
    r = math.sqrt(x0 * x0 + b * b)
    ph = math.atan2(b, x0)
    L = -b                                   # L = x p_y - y p_x with p = (1, 0)
    E = 1.0
    D = r * r - 2.0 * r + a * a
    R = (E * (r * r + a * a) - a * L) ** 2 - D * (L - a * E) ** 2
    pr = -math.sqrt(max(R, 0.0)) / D
    phks = ph
    xs = np.empty(nmax); ys = np.empty(nmax)
    n = 0
    fate = 0
    for i in range(nmax):
        xs[n] = r * math.cos(phks) - a * math.sin(phks)
        ys[n] = r * math.sin(phks) + a * math.cos(phks)
        n += 1
        if r < rh * 1.0005:
            fate = 1
            break
        dr, dph, dphks, dpr = rhs_eq(a, L, r, pr)
        if r > rmax and dr > 0:
            fate = 2
            break
        ds = 0.01 * max(r - rh, 0.003)
        v = math.sqrt(dr * dr + r * r * dphks * dphks) + 1e-12
        h = min(ds / v, 0.02 / (abs(dphks) + 1e-12))
        k1 = (dr, dphks, dpr)
        d2 = rhs_eq(a, L, r + 0.5 * h * k1[0], pr + 0.5 * h * k1[2]); k2 = (d2[0], d2[2], d2[3])
        d3 = rhs_eq(a, L, r + 0.5 * h * k2[0], pr + 0.5 * h * k2[2]); k3 = (d3[0], d3[2], d3[3])
        d4 = rhs_eq(a, L, r + h * k3[0], pr + h * k3[2]); k4 = (d4[0], d4[2], d4[3])
        r += h / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        phks += h / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        pr += h / 6.0 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        D = r * r - 2.0 * r + a * a
        if n >= nmax:
            break
    return xs[:n], ys[:n], fate


def critical_b(a, sign):
    lo, hi = (0.5, 12.0)
    # sign = +1: retrograde side (y > 0), sign = -1: prograde side (y < 0)
    for _ in range(60):
        b = 0.5 * (lo + hi)
        _, _, f = trace_eq(a, sign * b, -60.0, 70.0, 400000)
        if f == 1:
            lo = b
        else:
            hi = b
    return 0.5 * (lo + hi)


def glass_shells(n_shell=64, R0=10.0, rcore=1.6, n_in=1.9, gamma=1.5, nrays=41):
    # 2-d rays through concentric shells of constant ior rising toward the centre, with snell's law
    # and total internal reflection at every boundary
    radii = R0 * np.arange(1, n_shell + 1) / n_shell
    radii = radii[radii > rcore]

    def ior_region(r):
        if r > R0:
            return 1.0
        k = np.searchsorted(radii, r)          # region bounded by radii[k-1] < r <= radii[k]
        Rk = radii[min(k, len(radii) - 1)]
        return 1.0 + (n_in - 1.0) * (1.0 - Rk / R0) ** gamma
    paths = []
    for b in np.linspace(-11, 11, nrays):
        p = np.array([-30.0, b]); d = np.array([1.0, 0.0])
        pts = [p.copy()]
        for _ in range(4000):
            best_t, best_R = 1e9, None
            for R in list(radii) + [rcore]:
                bb = np.dot(p, d); cc = np.dot(p, p) - R * R
                disc = bb * bb - cc
                if disc < 0:
                    continue
                sq = math.sqrt(disc)
                for t in (-bb - sq, -bb + sq):
                    if 1e-6 < t < best_t:
                        best_t, best_R = t, R
            if best_R is None:
                pts.append(p + 60 * d); break
            p = p + best_t * d
            pts.append(p.copy())
            if best_R == rcore:
                break
            nrm = p / np.linalg.norm(p)
            inward = np.dot(d, nrm) < 0
            n_out = ior_region(best_R * 1.0005); n_inn = ior_region(best_R * 0.9995)
            n1, n2 = (n_out, n_inn) if inward else (n_inn, n_out)
            nn = nrm if inward else -nrm
            ci = -np.dot(d, nn)
            eta = n1 / n2
            k = 1 - eta * eta * (1 - ci * ci)
            if k < 0:
                d = d + 2 * ci * nn                       # total internal reflection
            else:
                d = eta * d + (eta * ci - math.sqrt(k)) * nn
            d /= np.linalg.norm(d)
            p = p + 1e-5 * d
        paths.append(np.array(pts))
    return paths, radii, rcore


def ray_colour(b, bmax):
    # retrograde side (b > 0, top) warm, prograde side (b < 0, bottom) cool
    t = float(np.clip(b / bmax, -1, 1))
    warm = np.array([1.0, 0.72, 0.36]); cool = np.array([0.45, 0.78, 1.0]); mid = np.array([0.92, 0.9, 0.86])
    return tuple(mid + (warm - mid) * t) if t > 0 else tuple(mid + (cool - mid) * (-t))


def main(out):
    try:
        fpath = r"C:\Windows\Fonts\bahnschrift.ttf"
        font_manager.fontManager.addfont(fpath)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=fpath).get_name()
    except Exception:
        pass
    bg = "#05060a"; ink = "#e8e2d6"; dim = "#8a92a2"
    fig = plt.figure(figsize=(18, 9), dpi=100, facecolor=bg)
    ax1 = fig.add_axes([0.03, 0.10, 0.36, 0.72], facecolor=bg)
    ax2 = fig.add_axes([0.42, 0.03, 0.56, 0.84], facecolor=bg)
    for ax in (ax1, ax2):
        ax.set_aspect("equal"); ax.axis("off")

    # left: glass shells
    paths, radii, rcore = glass_shells()
    for R in radii:
        u = 1 - R / 10.0
        ax1.add_patch(plt.Circle((0, 0), R, fill=False, lw=0.6, color=(0.45, 0.65, 0.95, 0.10 + 0.30 * u)))
    for p in paths:
        ax1.plot(p[:, 0], p[:, 1], color=ray_colour(p[0, 1], 11), lw=0.9, alpha=0.85, solid_capstyle="round")
    ax1.add_patch(plt.Circle((0, 0), rcore, color="black", zorder=3))
    ax1.add_patch(plt.Circle((0, 0), rcore, fill=False, lw=0.8, color=(0.6, 0.7, 0.8, 0.6), zorder=4))
    ax1.set_xlim(-14, 14); ax1.set_ylim(-14, 14)
    fig.text(0.03, 0.915, "Bent glass", color=ink, fontsize=22)
    fig.text(0.03, 0.88, "the tutorial: 64 nested glass shells, IOR rising toward the centre (Snell's law)",
             color=dim, fontsize=12)
    fig.text(0.03, 0.075, "Rays are focused like a lens. No photon orbits, no frame dragging,\n"
                          "no D-shaped shadow, no higher-order images, no capture threshold.",
             color=dim, fontsize=12, va="top")

    # right: kerr geodesics
    a = A
    rh = 1 + math.sqrt(1 - a * a)
    bpro = critical_b(a, -1); bret = critical_b(a, +1)
    print(f"critical impact parameters: prograde {bpro:.5f}, retrograde {bret:.5f}")

    def ksr(r):
        return math.sqrt(r * r + a * a)
    rr = 2 * (1 + math.cos(2 / 3 * math.acos(a)))
    rp = 2 * (1 + math.cos(2 / 3 * math.acos(-a)))
    ax2.add_patch(plt.Circle((0, 0), ksr(2.0), fill=False, ls=(0, (3, 3)), lw=0.9, color="#5b6b86", zorder=2))
    ax2.add_patch(plt.Circle((0, 0), ksr(rr), fill=False, ls=(0, (1, 2.5)), lw=1.1, color="#e0a050", zorder=2))
    W = 15.5
    segs, cols = [], []

    def add_faded(x, y, c, a0, fade_len, lw_list=None):
        # full opacity until periapsis, then fade with arc length (keeps scattered rays from cluttering)
        r = np.hypot(x, y); i0 = int(np.argmin(r))
        step = max(1, len(x) // 1500)
        idx = np.unique(np.r_[np.arange(0, len(x), step), len(x) - 1])
        x, y = x[idx], y[idx]
        i0 = int(np.searchsorted(idx, i0))
        ds = np.hypot(np.diff(x), np.diff(y))
        sarc = np.r_[0, np.cumsum(ds)]
        s_after = np.maximum(sarc[:-1] - sarc[i0], 0)
        al = a0 * np.exp(-s_after / fade_len)
        pts = np.stack([x, y], 1)
        for k in range(len(x) - 1):
            if al[k] < 0.01:
                break
            segs.append(pts[k:k + 2]); cols.append((*c, al[k]))

    for b in np.linspace(-11.5, 11.5, 47):
        x, y, f = trace_eq(a, b, -40.0, 40.0, 400000)
        add_faded(x, y, ray_colour(b, 11.5), 0.5 if f == 1 else 0.85, 7.0)
    ax2.add_collection(LineCollection(segs, colors=cols, linewidths=0.85, zorder=3, capstyle="round"))
    for bc, sgn in ((bpro, -1), (bret, +1)):
        for eps in (1e-2, 3e-4, 1e-5, 3e-7):
            for s2 in (+1, -1):
                b = sgn * bc * (1 + s2 * eps)
                x, y, f = trace_eq(a, b, -40.0, 40.0, 3000000)
                r = np.hypot(x, y); i0 = np.argmin(r)
                keep = np.ones(len(x), bool)
                outside = np.where((np.arange(len(x)) > i0) & (r > 9.0))[0]
                if len(outside):
                    keep[outside[0]:] = False
                c = (1.0, 0.82, 0.48) if sgn > 0 else (0.62, 0.86, 1.0)
                ax2.plot(x[keep], y[keep], color=c, lw=0.75, alpha=0.95, zorder=4)
    ax2.add_patch(plt.Circle((0, 0), ksr(rh), color="black", zorder=5))
    ax2.add_patch(plt.Circle((0, 0), ksr(rh), fill=False, lw=1.0, color="#c9d3e6", zorder=6))
    ax2.set_xlim(-W, W); ax2.set_ylim(-W * 0.92, W * 0.92)
    fig.text(0.42, 0.915, "Bent spacetime", color=ink, fontsize=22)
    fig.text(0.42, 0.88, f"this piece: Kerr null geodesics, a = {a}, equatorial plane (same RK4 integrator as the shader)",
             color=dim, fontsize=12)
    kw = dict(fontsize=11.5, zorder=7, bbox=dict(boxstyle="round,pad=0.25", fc=bg, ec="none", alpha=0.75))
    ax2.annotate("event horizon  r+ = %.2f M" % rh, xy=(0.9, -1.2), xytext=(4.0, -6.8), color="#c9d3e6",
                 arrowprops=dict(arrowstyle="-", color="#c9d3e6", lw=0.6), **kw)
    ax2.annotate("ergosphere  r = 2 M", xy=(ksr(2) * 0.71, ksr(2) * 0.71), xytext=(5.5, 6.6), color="#8fa0bf",
                 arrowprops=dict(arrowstyle="-", color="#8fa0bf", lw=0.6), **kw)
    ax2.annotate("retrograde photon orbit  r = %.2f M" % rr, xy=(-ksr(rr) * 0.7, ksr(rr) * 0.72),
                 xytext=(-14.8, 8.6), color="#e0a050", arrowprops=dict(arrowstyle="-", color="#e0a050", lw=0.6), **kw)
    ax2.text(-14.8, 11.6, f"counter-rotating light: captured if b < {bret:.2f} M", color="#ffc27a", **kw)
    ax2.text(-14.8, -12.6, f"co-rotating light skims to r = {rp:.2f} M: captured only if b < {bpro:.2f} M",
             color="#9fd8ff", **kw)
    ax2.text(6.2, -11.0, "spin (counter-clockwise):\nframe dragging swirls\ninfalling light", color=dim, **kw)
    ax2.text(-14.8, -0.9, "light from a distant source", color=dim, **kw)
    fig.savefig(out, facecolor=bg, dpi=100)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "geodesics.png")
