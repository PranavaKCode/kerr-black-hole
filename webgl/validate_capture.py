# validate the webgl tracer: compares the 50 % contour of a "Shadow" view capture (view=1, ui hidden,
# 16:9 canvas, no lens shift) with the bardeen curve mapped through the same zamo camera.
#   python validate_capture.py <png> <a> <incl_deg> <r_obs> <fov_deg> [overlay_out.png]
import sys, math
import numpy as np
from PIL import Image, ImageDraw
import reference as ref


def main():
    png, a, inc, ro, fov = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
    im = np.asarray(Image.open(png).convert('L'), dtype=np.float64) / 255.0
    H, W = im.shape
    tan_half = math.tan(math.radians(fov) / 2)          # 16:9 canvas -> no aspect widening
    curve = ref.bardeen_screen(a, ro, math.radians(inc), n=6000)
    # tan coords -> pixel centres (x right, y up in tan space; image y down)
    px = (curve[:, 0] / (tan_half * W / H) * 0.5 + 0.5) * W - 0.5
    py = (0.5 - curve[:, 1] / tan_half * 0.5) * H - 0.5
    # sub-pixel boundary of the rendered mask: along the ray from the curve centroid through each
    # analytic point, find where the mask crosses 0.5
    cx, cy = px.mean(), py.mean()
    errs = []
    for x, y in zip(px[::7], py[::7]):
        dx, dy = x - cx, y - cy
        L = math.hypot(dx, dy)
        ux, uy = dx / L, dy / L
        ts = np.linspace(-8, 8, 321)
        sx, sy = x + ts * ux, y + ts * uy
        # bilinear sample
        x0 = np.clip(np.floor(sx).astype(int), 0, W - 2); y0 = np.clip(np.floor(sy).astype(int), 0, H - 2)
        fx, fy = sx - x0, sy - y0
        v = (im[y0, x0] * (1 - fx) * (1 - fy) + im[y0, x0 + 1] * fx * (1 - fy) +
             im[y0 + 1, x0] * (1 - fx) * fy + im[y0 + 1, x0 + 1] * fx * fy)
        k = np.where((v[:-1] < 0.5) & (v[1:] >= 0.5))[0]
        if len(k) == 0:
            continue
        k = k[np.argmin(np.abs(ts[k]))]
        t0 = ts[k] + (0.5 - v[k]) / (v[k + 1] - v[k] + 1e-12) * (ts[k + 1] - ts[k])
        errs.append(t0)
    errs = np.array(errs)
    shadow_w = px.max() - px.min()
    print(f'{png}: a={a} i={inc} r_o={ro} fov={fov}  shadow width {shadow_w:.1f} px; '
          f'edge offset mean {errs.mean():+.3f} px, |max| {np.abs(errs).max():.3f} px, rms {np.sqrt((errs**2).mean()):.3f} px '
          f'({len(errs)} samples; {100*np.abs(errs).max()/shadow_w:.3f} % of the shadow width)')
    if len(sys.argv) > 6:
        rgb = Image.open(png).convert('RGB')
        d = ImageDraw.Draw(rgb)
        ang = np.arctan2(py - cy, px - cx)
        o = np.argsort(ang)
        pts = list(zip(px[o] + 0.5, py[o] + 0.5))
        for i in range(len(pts)):
            if (i // 40) % 2 == 0:
                d.line([pts[i], pts[(i + 1) % len(pts)]], fill=(255, 90, 40), width=1)
        rgb.save(sys.argv[6])


if __name__ == '__main__':
    main()
