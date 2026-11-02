# float64 numpy version of the kerr_bh.osl integrator (bl hamiltonian in mino time, rk4, same step
# heuristic). run as a script it bisects the critical image-plane offset on both sides of the shadow.
#   python kerr_ref.py [a]
import numpy as np

def rhs(a, E, L, y):
    r, th, ph, t, pr, pth = y
    s = np.sin(th); c = np.cos(th)
    s = np.where(np.abs(s) < 1e-6, np.where(s < 0, -1e-6, 1e-6), s)
    s2 = s*s; r2 = r*r; a2 = a*a
    D = r2 - 2*r + a2; Dp = 2*r - 2
    P = (r2 + a2)*E - a*L
    return np.array([D*pr, pth, a*P/D + L/s2 - a*E, (r2 + a2)*P/D + a*(L - a*E*s2),
                     0.5*(-Dp*pr*pr + 4*r*E*P/D - P*P*Dp/(D*D)), c*(L*L/(s2*s) - a2*E*E*s)])

def init_ray(a, r, th, ph, nr, nt, np_):
    st, ct = np.sin(th), np.cos(th)
    Sg = r*r + a*a*ct*ct; Dl = r*r - 2*r + a*a
    Ak = (r*r + a*a)**2 - a*a*Dl*st*st
    alpha = np.sqrt(Sg*Dl/Ak); omega = 2*a*r/Ak; varpi = np.sqrt(Ak/Sg)*st
    Eloc = 1/(alpha - omega*varpi*np_)
    E = -1.0; L = varpi*np_*Eloc
    pr = np.sqrt(Sg/Dl)*nr*Eloc; pth = np.sqrt(Sg)*nt*Eloc
    return E, L, np.array([r, th, ph, 0.0, pr, pth], dtype=float)

def trace(a, E, L, y, acc=0.04, max_steps=20000, r_esc=60.0):
    # returns (fate, state, steps), fate 0 = budget, 1 = captured, 2 = escaped
    rh = 1 + np.sqrt(1 - a*a)
    rphp = 2*(1 + np.cos(2/3*np.arccos(-a)))
    rterm = rh + 0.25*(rphp - rh)
    for i in range(max_steps):
        k1 = rhs(a, E, L, y)
        r, th = y[0], y[1]
        if r < rh*1.0005 or (r < rterm and k1[0] < 0):
            return 1, y, i
        if r > r_esc and k1[0] > 0:
            return 2, y, i
        s = np.sin(th)
        kpf = L/max(s*s, 1e-12) - a*E
        v = np.sqrt(k1[0]**2 + r*r*(k1[1]**2 + s*s*kpf**2)) + 1e-12
        ds = acc*max(r - rh, 0.002)
        h = ds/v
        h = min(h, 0.06/(abs(k1[1]) + 1e-12), 0.06/(abs(kpf) + 1e-12))
        k2 = rhs(a, E, L, y + 0.5*h*k1); k3 = rhs(a, E, L, y + 0.5*h*k2); k4 = rhs(a, E, L, y + h*k3)
        y = y + h/6*(k1 + 2*k2 + 2*k3 + k4)
    return 0, y, max_steps

if __name__ == "__main__":
    import sys
    a = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    r0, th0 = 1000.0, np.radians(85)
    # bisection on the image-plane offset along +x (n_phi) and -x for the critical boundary
    for sgn in (+1, -1):
        lo, hi = 3.0, 9.0
        for it in range(40):
            b = 0.5*(lo + hi)
            # direction toward the hole with a transverse offset angle b / r0 along phi
            ang = b/r0
            E, L, y = init_ray(a, r0, th0, -np.pi/2, -np.cos(ang), 0.0, sgn*np.sin(ang))
            fate, yf, n = trace(a, E, L, y, acc=0.02)
            if fate == 1: lo = b
            else: hi = b
        print(f"a={a} side {sgn:+d}: critical offset (r0*angle) = {0.5*(lo+hi):.5f}")
