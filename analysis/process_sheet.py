# the hero camera in four cumulative stages, with the bardeen curve dashed over the lensed sky.
#   python process_sheet.py <pass_dir> <hero.png> <out.jpg> <out.png>
# pass_dir holds 1920x1080 renders of the hero shot:
#   sky_bright.png       DebugMode 6 (lensed sky only), hero grade at exposure +2
#   mask.exr             DebugMode 3 capture mask (only used to report the curve fit)
#   dopp0grav0_hero.png  Doppler 0, GravRedshift 0, hero grade
#   dopp0_hero.png       Doppler 0, GravRedshift 1, hero grade
# hero.png is the full render (Doppler 1, GravRedshift 1).
import os, sys
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import numpy as np
from PIL import Image, ImageDraw, ImageFont
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_figures import load8
from bardeen import camera_pixels

FONT = r"C:\Windows\Fonts\bahnschrift.ttf"
BG = (9, 10, 14)
INK = (206, 200, 189)
DIM = (120, 126, 138)
CURVE = (176, 214, 255)

# hero camera (kerr_panel.HERO_CAM + shots.SHOTS["hero"])
A, R_OBS, INCL, LENS, SHIFT_X = 0.99, 300.0, 86.5, 220.0, -0.08


def font(size, weight="SemiLight"):
    f = ImageFont.truetype(FONT, size)
    try:
        f.set_variation_by_name(weight)
    except Exception:
        pass
    return f


def dashed_curve(panel, xs, ys, dash=9.0, gap=7.0, width=1.7, ss=4):
    # antialiased dashed polyline (supersampled, alpha-composited)
    W, H = panel.size
    lay = Image.new("L", (W * ss, H * ss), 0)
    d = ImageDraw.Draw(lay)
    p = np.c_[xs, ys] * ss - 0.5                     # PIL draws with pixel centres at integers
    seg = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    period = (dash + gap) * ss
    on = (seg % period) < dash * ss
    run = []
    for i in range(len(p)):
        if on[i]:
            run.append(tuple(p[i]))
        elif run:
            if len(run) > 1:
                d.line(run, fill=255, width=int(round(width * ss)), joint="curve")
            run = []
    if len(run) > 1:
        d.line(run, fill=255, width=int(round(width * ss)), joint="curve")
    a = lay.resize((W, H), Image.LANCZOS)
    col = Image.new("RGB", (W, H), CURVE)
    return Image.composite(col, panel, a.point(lambda v: int(v * 0.92)))


def main(pdir, hero_png, out_jpg, out_png, Wout=3000):
    M, G = 76, 30                     # outer margin, gutter between panels
    pw = (Wout - 2 * M - G) // 2
    ph = int(round(pw * 9 / 16))
    LH, RG = 62, 20                   # label strip under each panel, extra gap between rows
    Hout = M + 2 * (ph + LH) + RG + M - 14
    sheet = Image.new("RGB", (Wout, Hout), BG)
    d = ImageDraw.Draw(sheet)
    J = lambda n: os.path.join(pdir, n)

    # panel 1: lensed sky + analytic critical curve
    p1 = load8(J("sky_bright.png"))
    W0, H0 = p1.size
    p1 = p1.resize((pw, ph), Image.LANCZOS)
    xs, ys = camera_pixels(A, R_OBS, np.radians(INCL), LENS, 36.0, W0, H0)
    xs = xs - SHIFT_X * W0
    s = pw / W0
    p1 = dashed_curve(p1, xs * s, ys * s)            # raster coordinates (pixel centres at i + 0.5)
    try:                               # report how well the render's capture edge matches the curve
        from validate import load_mask, contour_points
        from scipy.spatial import cKDTree
        m = load_mask(J("mask.exr"))
        dd, _ = cKDTree(np.c_[xs, ys]).query(contour_points(m))
        print("capture edge vs Bardeen curve: mean %.3f px, 95%% %.3f px (1920 px frame)" % (dd.mean(), np.percentile(dd, 95)))
    except Exception as e:
        print("mask check skipped:", e)

    panels = [
        (p1, "1", "lensed sky, disk off"),
        (load8(J("dopp0grav0_hero.png")).resize((pw, ph), Image.LANCZOS), "2", "+ disk, no redshift"),
        (load8(J("dopp0_hero.png")).resize((pw, ph), Image.LANCZOS), "3", "+ gravitational redshift"),
        (load8(hero_png).resize((pw, ph), Image.LANCZOS), "4", "+ Doppler shift and beaming"),
    ]
    fl = font(31, "SemiLight")
    fn = font(31, "Regular")
    fs = font(26, "SemiLight")
    for i, (im, num, lab) in enumerate(panels):
        r, c = divmod(i, 2)
        x = M + c * (pw + G)
        y = M + r * (ph + LH + RG)
        sheet.paste(im, (x, y))
        ty = y + ph + 17
        d.text((x, ty), num, font=fn, fill=INK)
        d.text((x + 34, ty), lab, font=fl, fill=INK)
        if i == 0:                    # key for the dashed line, after the panel-1 label
            kx = x + 34 + d.textlength(lab, font=fl) + 64
            ly = ty + 19
            for k in range(3):
                x0 = kx + k * 19
                d.line([(x0, ly), (x0 + 10, ly)], fill=CURVE, width=2)
            d.text((kx + 3 * 19 + 14, ty + 4), "Bardeen critical curve", font=fs, fill=DIM)
    sheet.save(out_png, optimize=True)
    for q in (90, 88, 86, 84):
        sheet.save(out_jpg, quality=q, optimize=True, progressive=True, subsampling=0)
        if os.path.getsize(out_jpg) < 2.5 * 1024 * 1024:
            break
    print("wrote", out_jpg, sheet.size, os.path.getsize(out_jpg) // 1024, "KB q", q)


if __name__ == "__main__":
    main(*sys.argv[1:5])
