# build the scene, apply a shot and render <out>.exr (scene linear) plus an <out>.png preview.
#   blender -b --factory-startup --python render_shot.py -- --shot hero --res 1280 720 --spp 32 \
#       --out /abs/path/name [--set Spin=0.5 Doppler=0] [--cam r_obs=200 incl=80] [--border x0 x1 y0 y1]
#       [--device OPTIX|CPU] [--save-blend path]
import bpy, sys, os, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_scene as B

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--shot", default="hero")
ap.add_argument("--res", nargs=2, type=int, default=[1280, 720])
ap.add_argument("--spp", type=int, default=32)
ap.add_argument("--out", required=True)
ap.add_argument("--set", nargs="*", default=[])
ap.add_argument("--cam", nargs="*", default=[])
ap.add_argument("--border", nargs=4, type=float, default=None)
ap.add_argument("--device", default="OPTIX")
ap.add_argument("--save-blend", default=None)
ap.add_argument("--frame", type=int, default=1)
ap.add_argument("--exposure", type=float, default=0.0)
a = ap.parse_args(argv)

def parse_kv(lst):
    d = {}
    for kv in lst:
        k, v = kv.split("=", 1)
        if "," in v:
            d[k] = tuple(float(x) for x in v.split(","))
        else:
            try:
                d[k] = int(v)
            except ValueError:
                d[k] = float(v)
    return d

t0 = time.time()
shot = B.build(a.shot, a.device, tuple(a.res), a.spp)
cam_over = parse_kv(a.cam)
par_over = parse_kv(a.set)
if cam_over or par_over:
    s = dict(shot); s.update(cam_over); s["params"] = dict(shot.get("params", {})); s["params"].update(par_over)
    B.apply_shot(s)
scn = bpy.context.scene
scn.frame_set(a.frame)
scn.view_settings.exposure = a.exposure
if a.border:
    scn.render.use_border = True; scn.render.use_crop_to_border = True
    scn.render.border_min_x, scn.render.border_max_x, scn.render.border_min_y, scn.render.border_max_y = a.border
node = B.kerr_node()
B.log("params:", {s.name: (tuple(s.default_value) if hasattr(s.default_value, '__len__') else round(s.default_value, 6))
                  for s in node.inputs if hasattr(s, 'default_value')})
B.log("camera:", tuple(round(x, 3) for x in scn.camera.location), "lens", scn.camera.data.lens)
if a.save_blend:
    bpy.ops.wm.save_as_mainfile(filepath=a.save_blend, compress=True)
out = a.out
os.makedirs(os.path.dirname(out), exist_ok=True)
scn.render.image_settings.file_format = 'OPEN_EXR'
scn.render.image_settings.color_depth = '32'
scn.render.image_settings.exr_codec = 'ZIP'
scn.render.filepath = out + ".exr"
B.log("setup %.1fs" % (time.time() - t0))
t1 = time.time()
bpy.ops.render.render(write_still=True)
B.log("RENDER_SECONDS %.1f" % (time.time() - t1))
img = bpy.data.images.load(out + ".exr")
img.colorspace_settings.name = 'Linear Rec.709' if 'Linear Rec.709' in [i.identifier for i in img.colorspace_settings.bl_rna.properties['name'].enum_items] else img.colorspace_settings.name
scn.render.image_settings.file_format = 'PNG'
scn.render.image_settings.color_depth = '8'
img.save_render(out + ".png", scene=scn)
B.log("wrote", out + ".exr", out + ".png", "total %.1fs" % (time.time() - t0))
