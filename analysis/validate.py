# distance in pixels between the 50 % capture contour of a DebugMode 3 render (1 = escaped,
# 0 = captured) and the analytic bardeen curve.
#   python validate.py mask.exr a r_obs incl_deg lens_mm [out.png]
import os, sys
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2, numpy as np
from bardeen import camera_pixels

def load_mask(path):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return im[..., 1].astype(np.float64)    # green channel (budget-exhausted pixels are red = 0 in G)

def contour_points(m):
    # sub-pixel 0.5 iso-contour via marching squares (skimage-free): use cv2 on upsampled mask
    up = 8
    mu = cv2.resize(m, None, fx=up, fy=up, interpolation=cv2.INTER_LINEAR)
    b = (mu < 0.5).astype(np.uint8)
    cs, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=len)[:, 0, :].astype(np.float64)
    return (c + 0.5) / up                          # raster coords (pixel centres at i+0.5)

def compare(path, a, r_obs, inc_deg, lens, sensor=36.0):
    m = load_mask(path)
    H, W = m.shape
    xs, ys = camera_pixels(a, r_obs, np.radians(inc_deg), lens, sensor, W, H)
    c = contour_points(m)
    from scipy.spatial import cKDTree
    t = cKDTree(np.c_[xs, ys])
    d, _ = t.query(c)
    return m, (xs, ys), c, d

if __name__ == "__main__":
    path, a, r_obs, inc, lens = sys.argv[1], *map(float, sys.argv[2:6])
    m, (xs, ys), c, d = compare(path, a, r_obs, inc, lens)
    print(f"contour pts {len(c)}  mean |d| = {d.mean():.3f} px  median {np.median(d):.3f}  95% {np.percentile(d,95):.3f}  max {d.max():.3f}")
    sz = (np.ptp(xs), np.ptp(ys))
    print(f"analytic shadow size {sz[0]:.1f} x {sz[1]:.1f} px")
    if len(sys.argv) > 6:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 6), dpi=120); plt.imshow(m, cmap="gray"); plt.plot(xs - 0.5, ys - 0.5, "c-", lw=0.6)
        plt.savefig(sys.argv[6]); print("wrote", sys.argv[6])
