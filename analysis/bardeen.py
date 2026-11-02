# analytic kerr critical curve (bardeen 1973) and its exact image in a zamo camera at finite r.
#   critical_curve(a, inc) -> (alpha, beta) distant-observer screen coords in M
#   camera_pixels(a, r_obs, inc, lens_mm, sensor_mm, W, H) -> (x, y) raster coords (y down) in the
#   blender camera of kerr_bh.osl (image right = +phi_hat, image up = -theta_hat)
import numpy as np

def r_ph(a, sgn):
    return 2.0 * (1.0 + np.cos(2.0 / 3.0 * np.arccos(sgn * a)))

def xi_eta(a, r):
    xi = (r**2 * (3 - r) - a**2 * (r + 1)) / (a * (r - 1))
    eta = r**3 * (4 * a**2 - r * (r - 3)**2) / (a**2 * (r - 1)**2)
    return xi, eta

def critical_constants(a, inc, n=200000):
    # (xi, eta, Theta) of the photon region visible at inclination inc (rad); cosine spacing in r
    # crowds samples toward both ends, where beta = sqrt(Theta) is steep
    if a < 1e-6:
        a = 1e-6
    r1, r2 = r_ph(a, -1), r_ph(a, +1)          # prograde (smaller), retrograde
    u = np.linspace(0.0, 1.0, n)
    r = r1 + (r2 - r1) * 0.5 * (1.0 - np.cos(np.pi * u))
    xi, eta = xi_eta(a, r)
    th = eta + a**2 * np.cos(inc)**2 - xi**2 / np.tan(inc)**2
    ok = th >= 0
    return xi[ok], eta[ok], th[ok]

def critical_curve(a, inc, n=200000):
    xi, eta, th = critical_constants(a, inc, n)
    alpha = -xi / np.sin(inc)
    beta = np.sqrt(th)
    return np.concatenate([alpha, alpha[::-1]]), np.concatenate([beta, -beta[::-1]])

def camera_pixels(a, r_obs, inc, lens_mm, sensor_mm, W, H, n=20000):
    xi, eta, th = critical_constants(a, inc, n)
    L = xi                                     # E = 1
    r, t = r_obs, inc
    st, ct = np.sin(t), np.cos(t)
    Sg = r*r + a*a*ct*ct; Dl = r*r - 2*r + a*a
    Ak = (r*r + a*a)**2 - a*a*Dl*st*st
    alpha = np.sqrt(Sg*Dl/Ak); omega = 2*a*r/Ak; varpi = np.sqrt(Ak/Sg)*st
    Eloc = (1 - omega*L) / alpha
    # Theta(theta_obs) = eta + a^2 cos^2 - xi^2 cot^2  (E=1)  -> |p_theta|
    pth = np.sqrt(np.maximum(eta + a*a*ct*ct - L*L*ct*ct/(st*st), 0))
    nphi = -L / (varpi * Eloc)
    nth = pth / (np.sqrt(Sg) * Eloc)
    fwd = np.sqrt(np.maximum(1 - nphi**2 - nth**2, 0))
    scale = lens_mm / (sensor_mm / W)          # pixels per unit tan
    x = W / 2 + nphi / fwd * scale
    yu = nth / fwd * scale
    xs = np.concatenate([x, x[::-1], x[:1]])                 # closed curve
    ys = np.concatenate([H / 2 - yu, (H / 2 + yu)[::-1], (H / 2 - yu)[:1]])
    return xs, ys

if __name__ == "__main__":
    for a in (0.0, 0.5, 0.99):
        al, be = critical_curve(max(a, 1e-4), np.radians(90))
        print(a, "alpha range", al.min(), al.max(), "beta max", be.max())
    # schwarzschild check: radius sqrt(27)
    al, be = critical_curve(1e-5, np.radians(60))
    print("a~0 radius:", np.sqrt(al**2 + be**2).mean(), np.sqrt(27))
