# labelled figure sheets and the validation overlay.
#   python make_figures.py grid  out.jpg  a0_off.png a0_on.png a99_off.png a99_on.png
#   python make_figures.py valid out.jpg  sky_a099.png mask_a099.exr mask_a0.exr  r_obs incl lens
#   python make_figures.py pair  out.jpg  left.png right.png "left title" "right title" "caption"
import os, sys
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BG = (7, 8, 12)
INK = (232, 226, 214)
DIM = (140, 147, 162)
FONT = r"C:\Windows\Fonts\bahnschrift.ttf"


def font(sz):
    try:
        return ImageFont.truetype(FONT, sz)
    except Exception:
        return ImageFont.load_default()


def wrap(draw, text, fnt, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=fnt) > width and cur:
            lines.append(cur); cur = w
        else:
            cur = t
    if cur:
        lines.append(cur)
    return lines


def grid(out, files, cols, rows, col_labels, row_labels, title, caption, W=1800):
    ims = [load8(f) for f in files]
    g = 10; lw = 150; th = 64; ch = 22
    pw = (W - lw - g * (cols + 1)) // cols
    ph = int(round(pw * ims[0].height / ims[0].width))
    d0 = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    cap_lines = wrap(d0, caption, font(19), W - 2 * g - 10)
    H = th + ch + rows * (ph + g) + g + 26 * len(cap_lines) + 16
    sheet = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(sheet)
    d.text((g + 4, 14), title, font=font(30), fill=INK)
    for c, lab in enumerate(col_labels):
        x = lw + g + c * (pw + g)
        d.text((x + 4, th), lab, font=font(20), fill=DIM)
    for r in range(rows):
        y = th + ch + g + r * (ph + g)
        for i, line in enumerate(row_labels[r].split("\n")):
            d.text((g + 4, y + ph // 2 - 22 + 26 * i), line, font=font(20 if i == 0 else 17), fill=INK if i == 0 else DIM)
        for c in range(cols):
            im = ims[r * cols + c].resize((pw, ph), Image.LANCZOS)
            sheet.paste(im, (lw + g + c * (pw + g), y))
    y = th + ch + rows * (ph + g) + g + 6
    for line in cap_lines:
        d.text((g + 4, y), line, font=font(19), fill=DIM); y += 26
    save_jpg(sheet, out)


def validation(out, sky_png, mask_a099, mask_a0, r_obs, incl, lens, crop_frac=0.62):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from validate import compare
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    try:
        font_manager.fontManager.addfont(FONT)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
    except Exception:
        pass
    m99, (x99, y99), c99, d99 = compare(mask_a099, 0.99, r_obs, incl, lens)
    m0, (x0, y0), c0, d0 = compare(mask_a0, 1e-4, r_obs, incl, lens)
    sky = np.asarray(Image.open(sky_png).convert("RGB"))
    H, W = m99.shape
    bg = "#07080c"
    fig = plt.figure(figsize=(18, 9.4), dpi=100, facecolor=bg)
    # main panel: the actual render (lensed sky, a = 0.99) with the analytic curve
    ax = fig.add_axes([0.01, 0.025, 0.51, 0.86], facecolor=bg)
    cx, cy = np.mean(x99), H / 2
    half = W * crop_frac / 2
    ax.imshow(sky, extent=(0, W, H, 0), interpolation="lanczos")
    ax.plot(x99, y99, color="#36e0ff", lw=0.9, alpha=0.9)
    ax.plot(x0, y0, color="#ffb347", lw=0.8, alpha=0.8, ls=(0, (4, 3)))
    ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy + half, cy - half); ax.axis("off")
    # zoom insets
    def inset(rect, xc, yc, hw, mask, xs, ys, col, title):
        a = fig.add_axes(rect, facecolor=bg)
        a.imshow(mask, cmap="gray", vmin=0, vmax=1, extent=(0, W, H, 0), interpolation="nearest")
        a.plot(xs, ys, color=col, lw=2.2)
        a.set_xlim(xc - hw, xc + hw); a.set_ylim(yc + hw, yc - hw)
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_color("#3a4152")
        a.set_title(title, color="#a7afbf", fontsize=12, loc="left")
    i_flat = np.argmin(x99)
    inset([0.56, 0.47, 0.20, 0.36], x99[i_flat], y99[i_flat], 9, m99, x99, y99, "#36e0ff",
          "a = 0.99: flat (prograde) edge of the D, pixels")
    i_top = np.argmin(y99)
    inset([0.78, 0.47, 0.20, 0.36], x99[i_top], y99[i_top], 9, m99, x99, y99, "#36e0ff",
          "a = 0.99: top of the shadow, pixels")
    i0 = np.argmin(x0)
    inset([0.56, 0.06, 0.20, 0.36], x0[i0], y0[i0], 9, m0, x0, y0, "#ffb347", "a = 0: left edge, pixels")
    tx = fig.add_axes([0.78, 0.06, 0.20, 0.36], facecolor=bg); tx.axis("off")
    rpx = np.ptp(x0) / 2
    lines = [
        ("Render vs. analytic critical curve", "#e8e2d6", 15),
        ("(Bardeen 1973, mapped exactly into the", "#8a92a2", 12),
        (f" ZAMO camera at r = {r_obs:.0f} M, i = {incl:.0f}\u00b0)", "#8a92a2", 12),
        ("", "#000", 8),
        (f"a = 0.99   mean |\u0394| = {d99.mean():.2f} px", "#36e0ff", 14),
        (f"              95 % < {np.percentile(d99, 95):.2f} px", "#36e0ff", 12),
        (f"a = 0      mean |\u0394| = {d0.mean():.2f} px", "#ffb347", 14),
        (f"              radius {rpx:.1f} px = sqrt(27) M", "#ffb347", 12),
        ("", "#000", 8),
        ("capture mask: 50 % contour of a", "#8a92a2", 12),
        ("DebugMode 3 render (escaped = white)", "#8a92a2", 12),
    ]
    y = 0.98
    for t, c, s in lines:
        tx.text(0.0, y, t, color=c, fontsize=s, va="top", transform=tx.transAxes); y -= 0.085 if s > 9 else 0.04
    fig.text(0.012, 0.955, "Validation: the shadow is exactly where general relativity puts it", color="#e8e2d6", fontsize=21)
    fig.text(0.012, 0.915, "Disk off. Lensed star field rendered by the OSL shader in Cycles; cyan: analytic Kerr critical "
                           "curve a = 0.99, dashed amber: a = 0 (circle of radius sqrt(27) M)", color="#8a92a2", fontsize=12.5)
    tmp = out + ".png"
    fig.savefig(tmp, facecolor=bg, dpi=100)
    im = Image.open(tmp).convert("RGB"); os.remove(tmp); save_jpg(im, out)
    print("wrote", out, "mean a0.99 %.3f px, a0 %.3f px" % (d99.mean(), d0.mean()))


def load8(path):
    # 16-bit png -> 8-bit with triangular dither (no banding in the tone-mapped shoulder)
    import cv2
    a = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if a is None:
        return Image.open(path).convert("RGB")
    if a.ndim == 3 and a.shape[2] == 4:
        a = a[..., :3]
    a = a[..., ::-1].astype(np.float32)
    if a.max() > 255.5:
        a = a / 257.0
    a = a + (np.random.rand(*a.shape) - np.random.rand(*a.shape))
    return Image.fromarray(np.clip(a + 0.5, 0, 255).astype(np.uint8))


def save_jpg(im, out, maxkb=790):
    for q in (90, 88, 86, 84, 82, 80, 78, 75):
        im.save(out, quality=q, optimize=True, progressive=True)
        if os.path.getsize(out) < maxkb * 1024:
            break
    print("wrote", out, im.size, os.path.getsize(out) // 1024, "KB q", q)


def panels(out, files, cols, labels, title, caption, W=1800):
    ims = [load8(f) for f in files]
    g = 10; th = 64 if title else 10; lh = 30
    pw = (W - g * (cols + 1)) // cols
    ph = int(round(pw * ims[0].height / ims[0].width))
    rows = (len(ims) + cols - 1) // cols
    d0 = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    cap_lines = wrap(d0, caption, font(19), W - 2 * g - 10) if caption else []
    H = th + rows * (ph + lh + g) + 26 * len(cap_lines) + 18
    sheet = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(sheet)
    if title:
        d.text((g + 4, 14), title, font=font(30), fill=INK)
    for i, im in enumerate(ims):
        r, c = divmod(i, cols)
        x = g + c * (pw + g); y = th + r * (ph + lh + g)
        d.text((x + 2, y + 2), labels[i], font=font(19), fill=DIM)
        sheet.paste(im.resize((pw, ph), Image.LANCZOS), (x, y + lh))
    y = th + rows * (ph + lh + g)
    for line in cap_lines:
        d.text((g + 4, y + 4), line, font=font(19), fill=DIM); y += 26
    save_jpg(sheet, out)


def pair(out, left, right, lt, rt, caption, W=1800):
    panels(out, [left, right], 2, [lt, rt], "", caption, W)


if __name__ == "__main__":
    mode, out = sys.argv[1], sys.argv[2]
    if mode == "grid":
        f = sys.argv[3:7]
        grid(out, f, 2, 2, ["Doppler / beaming OFF", "Doppler / beaming ON  (physical)"],
             ["a = 0\nSchwarzschild", "a = 0.99\nnear-extremal Kerr"],
             "The two physical controls: spin and Doppler",
             "Same camera (r = 300 M, 4.5\u00b0 above the disk plane), same disk. Spin pulls the inner edge of the "
             "disk (ISCO 6 M to 1.45 M) and the photon ring inward and flattens the approaching side of the shadow into "
             "a D. Doppler boosting and relativistic beaming make the approaching (left) side blaze blue-white and the "
             "receding side dim to orange; with the control at 0 only gravitational redshift remains, as in the film.")
    elif mode == "valid":
        validation(out, sys.argv[3], sys.argv[4], sys.argv[5], float(sys.argv[6]), float(sys.argv[7]), float(sys.argv[8]))
    elif mode == "pair":
        pair(out, sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7])
