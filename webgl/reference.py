# cpu reference for the webgl tracer (shaders/trace.frag), same equations: bl hamiltonian with E = 1,
#   2 Sigma H = Delta p_r^2 + p_th^2 + (L/s - a s)^2 - P^2/Delta,  P = r^2 + a^2 - a L,
# zamo camera, backwards rk4 in the pole-free vector form. checks carter constant conservation and
# the shadow edge against the analytic bardeen curve mapped through the same camera.
#   python reference.py [a] [incl_deg] [r_obs] [N] [overlay.png]      (N = 160: ~25k rays, a few seconds)
import sys, math
import numpy as np


def horizon(a):
    return 1.0 + math.sqrt(max(0.0, 1.0 - a * a))


def isco(a):
    z1 = 1 + (1 - a * a) ** (1 / 3) * ((1 + a) ** (1 / 3) + (1 - a) ** (1 / 3))
    z2 = math.sqrt(3 * a * a + z1 * z1)
    return 3 + z2 - math.sqrt((3 - z1) * (3 + z1 + 2 * z2))


def photon_orbits(a):
    rp = 2 * (1 + math.cos(2 / 3 * math.acos(-a)))  # prograde
    rm = 2 * (1 + math.cos(2 / 3 * math.acos(a)))   # retrograde
    return rp, rm


def zamo(r, th, a):
    s, c = math.sin(th), math.cos(th)
    Sig = r * r + a * a * c * c
    Del = r * r - 2 * r + a * a
    A = (r * r + a * a) ** 2 - a * a * Del * s * s
    alpha = math.sqrt(Sig * Del / A)
    omega = 2 * a * r / A
    gphph = A * s * s / Sig
    return Sig, Del, A, alpha, omega, gphph


def camera_rays(a, r_o, th_o, tan_half, N):
    # square N x N pixel grid -> initial state[5, M], L[M], screen x, y
    Sig, Del, A, alpha, omega, gphph = zamo(r_o, th_o, a)
    xs = (np.arange(N) + 0.5) / N * 2 - 1
    X, Y = np.meshgrid(xs * tan_half, -xs * tan_half)  # Y up
    X = X.ravel(); Y = Y.ravel()
    nrm = np.sqrt(1 + X * X + Y * Y)
    # look direction n: fwd = -e_r, right = +e_phi, up = -e_th ; photon travels m = -n
    m_r = 1 / nrm
    m_th = Y / nrm
    m_ph = -X / nrm
    p_r = math.sqrt(Sig / Del) * m_r
    p_th = math.sqrt(Sig) * m_th
    L = math.sqrt(gphph) * m_ph
    E = alpha + omega * L
    p_r /= E; p_th /= E; L = L / E
    M = X.size
    st = np.zeros((5, M))
    st[0] = r_o; st[1] = th_o; st[2] = 0.0; st[3] = p_r; st[4] = p_th
    return st, L, X, Y


def hamiltonian(st, L, a):
    r, th, ph, pr, pth = st
    s = np.sin(th); c = np.cos(th)
    Sig = r * r + a * a * c * c
    Del = r * r - 2 * r + a * a
    P = r * r + a * a - a * L
    return (Del * pr * pr + pth * pth + (L / s - a * s) ** 2 - P * P / Del) / (2 * Sig)


def carter(st, L, a):
    th, pth = st[1], st[4]
    s = np.sin(th); c = np.cos(th)
    return pth * pth + c * c * (L * L / (s * s) - a * a)


# pole-free form used by the shader, in mino time: angular position as a unit vector n and angular
# momentum J = n x p_ang (J_z = L, Q = |J|^2 - L^2 - a^2 cos^2 th)
#   dr/ds  = Delta p_r
#   dpr/ds = -(r-1) p_r^2 + 2 r P/Delta - (r-1) P^2/Delta^2
#   dn/ds  = J x n + W z x n
#   dJ/ds  = a^2 mu (n x z) + W z x J          W = a (P/Delta - 1),  mu = n_z
def to_vec(st, L):
    r, th, ph, pr, pth = st
    s, c = np.sin(th), np.cos(th)
    n = np.stack([s * np.cos(ph), s * np.sin(ph), c])
    eth = np.stack([c * np.cos(ph), c * np.sin(ph), -s])
    eph = np.stack([-np.sin(ph), np.cos(ph), 0 * ph])
    J = pth * eph - (L / s) * eth
    return np.concatenate([r[None], pr[None], n, J])  # 8 x M


def rhs_vec(y, a):
    r, pr = y[0], y[1]
    n = y[2:5]; J = y[5:8]
    L = J[2]; mu = n[2]
    Del = r * r - 2 * r + a * a
    P = r * r + a * a - a * L
    W = a * (P / Del - 1)
    out = np.empty_like(y)
    out[0] = Del * pr
    out[1] = -(r - 1) * pr * pr + 2 * r * P / Del - (r - 1) * P * P / (Del * Del)
    Jxn = np.cross(J.T, n.T).T
    zxn = np.stack([-n[1], n[0], 0 * mu])
    zxJ = np.stack([-J[1], J[0], 0 * mu])
    nxz = np.stack([n[1], -n[0], 0 * mu])
    out[2:5] = Jxn + W * zxn
    out[5:8] = a * a * mu * nxz + W * zxJ
    return out


def trace_vec(y, a, r_far, steps=4000, k=0.08):
    rh = horizon(a)
    M = y.shape[1]
    alive = np.ones(M, bool); captured = np.zeros(M, bool)
    for _ in range(steps):
        if not alive.any():
            break
        idx = np.where(alive)[0]
        z = y[:, idx]
        r = z[0]
        k1 = rhs_vec(z, a)
        # adaptive mino-time step: limit relative change of (r - r+) and angular change per step
        rdot = np.abs(k1[0]); ndot = np.linalg.norm(k1[2:5], axis=0)
        h = -k * np.minimum((r - rh) / (rdot + 1e-9), 1.0 / (ndot + 1e-9))
        k2 = rhs_vec(z + 0.5 * h * k1, a)
        k3 = rhs_vec(z + 0.5 * h * k2, a)
        k4 = rhs_vec(z + h * k3, a)
        z = z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        z[2:5] /= np.linalg.norm(z[2:5], axis=0)
        y[:, idx] = z
        cap = z[0] < rh * 1.01
        esc = (z[0] > r_far) & (z[1] < 0)
        captured[idx[cap]] = True
        alive[idx[cap | esc]] = False
    return captured, alive


def carter_vec(y, a):
    J = y[5:8]; mu = y[4]
    return (J * J).sum(0) - J[2] ** 2 - a * a * mu * mu


def ham_vec(y, a):
    r, pr = y[0], y[1]; n = y[2:5]; J = y[5:8]; L = J[2]; mu = n[2]
    Del = r * r - 2 * r + a * a; P = r * r + a * a - a * L
    s2 = 1 - mu * mu
    # p_th^2 + (L/s - a s)^2 = |J|^2 - 2 a L + a^2 s^2
    return Del * pr * pr - P * P / Del + (J * J).sum(0) - 2 * a * L + a * a * s2


def bardeen_screen(a, r_o, th_o, n=4000):
    # exact screen (tan) coordinates of the critical curve for the zamo camera at (r_o, th_o)
    rp, rm = photon_orbits(a)
    Sig, Del, A, alpha, omega, gphph = zamo(r_o, th_o, a)
    pts = []
    if a < 1e-4:
        # schwarzschild: b = sqrt(27) for every direction
        b = math.sqrt(27.0)
        for t in np.linspace(0, 2 * math.pi, n):
            xi = b * math.cos(t) * math.sin(th_o)
            eta_plus = b * b - xi * xi  # Q for a=0: Q = b^2 - L^2
            pts.append((xi, eta_plus, 1 if math.sin(t) > 0 else -1))
    else:
        for r in np.linspace(rp, rm, n):
            xi = (r * r * (3 - r) - a * a * (r + 1)) / (a * (r - 1))
            eta = r ** 3 * (4 * a * a - r * (r - 3) ** 2) / (a * a * (r - 1) ** 2)
            pts.append((xi, eta, 1)); pts.append((xi, eta, -1))
    out = []
    c, s = math.cos(th_o), math.sin(th_o)
    for xi, eta, sg in pts:
        Th = eta + a * a * c * c - xi * xi * c * c / (s * s)
        if Th < 0:
            continue
        pth = sg * math.sqrt(Th)
        R = (r_o * r_o + a * a - a * xi) ** 2 - Del * (eta + (xi - a) ** 2)
        if R < 0:
            continue
        pr = math.sqrt(R) / Del
        Eloc = (1 - omega * xi) / alpha
        m_r = math.sqrt(Del / Sig) * pr / Eloc
        m_th = pth / (math.sqrt(Sig) * Eloc)
        m_ph = xi / (math.sqrt(gphph) * Eloc)
        out.append((-m_ph / m_r, m_th / m_r))
    return np.array(out)


def main():
    a = float(sys.argv[1]) if len(sys.argv) > 1 else 0.99
    inc = float(sys.argv[2]) if len(sys.argv) > 2 else 90.0
    r_o = float(sys.argv[3]) if len(sys.argv) > 3 else 60.0
    N = int(sys.argv[4]) if len(sys.argv) > 4 else 160
    th_o = math.radians(inc)
    tan_half = 8.0 / r_o  # +-8 M field
    st, L, X, Y = camera_rays(a, r_o, th_o, tan_half, N)
    Q0 = carter(st, L, a); H0 = hamiltonian(st, L, a)
    y = to_vec(st, L)
    Qv0 = carter_vec(y, a)
    print('Q(vec) - Q(BL) at camera:', np.abs(Qv0 - Q0).max(), ' 2SigH(vec):', np.abs(ham_vec(y, a)).max())
    captured, alive = trace_vec(y, a, r_far=r_o * 2 + 50)
    esc = ~captured
    dQ = np.abs(carter_vec(y, a) - Qv0)[esc]
    dH = np.abs(ham_vec(y, a))[esc]
    print(f"a={a} incl={inc} r_o={r_o}  rh={horizon(a):.4f} isco={isco(a):.4f} rph={photon_orbits(a)}")
    print(f"|H0| max {np.abs(H0).max():.2e}; escaped subset: max|dQ| {dQ.max():.2e} (Q~{np.abs(Qv0).mean():.1f}), max|H| {dH.max():.2e}")
    print(f"unfinished rays: {alive.sum()}")
    mask = captured.reshape(N, N)
    # compare edge: for each analytic curve point, distance to nearest mask boundary pixel
    curve = bardeen_screen(a, r_o, th_o)
    px = (curve[:, 0] / tan_half + 1) / 2 * N - 0.5
    py = (-curve[:, 1] / tan_half + 1) / 2 * N - 0.5
    edge = np.zeros_like(mask)
    edge[1:-1, 1:-1] = mask[1:-1, 1:-1] & ~(mask[:-2, 1:-1] & mask[2:, 1:-1] & mask[1:-1, :-2] & mask[1:-1, 2:])
    ey, ex = np.nonzero(edge)
    d = np.sqrt((px[:, None] - ex[None, :]) ** 2 + (py[:, None] - ey[None, :]) ** 2).min(axis=1)
    print(f"shadow edge vs Bardeen: mean {d.mean():.3f} px, max {d.max():.3f} px  (pixel = {2*tan_half*r_o/N:.3f} M at r_o)")
    print(f"shadow extent x: mask [{ex.min()},{ex.max()}] curve [{px.min():.1f},{px.max():.1f}]; y: mask [{ey.min()},{ey.max()}] curve [{py.min():.1f},{py.max():.1f}]")
    if len(sys.argv) > 5:
        from PIL import Image, ImageDraw
        im = Image.fromarray((~mask * 255).astype(np.uint8)).convert('RGB').resize((N * 3, N * 3), Image.NEAREST)
        dr = ImageDraw.Draw(im)
        for x, y in zip(px, py):
            dr.point(((x + 0.5) * 3, (y + 0.5) * 3), fill=(255, 60, 30))
        im.save(sys.argv[5])


if __name__ == '__main__':
    main()
