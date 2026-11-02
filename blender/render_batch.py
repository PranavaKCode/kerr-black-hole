# render a list of tasks in one blender process, so the osl kernel compiles once.
# each task writes <out>.exr (scene linear), <out>.png and optional graded comps.
#   blender -b --factory-startup --python render_batch.py -- tasks.json
#   tasks.json = [{"shot": "hero", "res": [1280, 720], "spp": 32, "out": "/abs/out/name",
#                  "set": {"Spin": 0.5}, "cam": {"incl": 80}, "border": [x0, x1, y0, y1],
#                  "exposure": 0.0, "frame": 1, "device": "OPTIX", "comps": {"hero": {...}}}, ...]
import bpy, sys, os, time, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_scene as B

argv = sys.argv[sys.argv.index("--") + 1:]
tasks = json.load(open(argv[0]))
t0 = time.time()
first = tasks[0]
B.build(first.get("shot", "hero"), first.get("device", "OPTIX"), tuple(first.get("res", (640, 360))), first.get("spp", 16))
scn = bpy.context.scene
B.log("build %.1fs" % (time.time() - t0))
for t in tasks:
    shot = B.SHOTS.get(t.get("shot", "hero"))
    shot.update(t.get("cam", {}))
    shot.setdefault("params", {}).update(t.get("set", {}))
    B.apply_shot(shot)
    scn.render.resolution_x, scn.render.resolution_y = t.get("res", (640, 360))
    scn.render.resolution_percentage = 100
    scn.cycles.samples = t.get("spp", 16)
    scn.cycles.adaptive_min_samples = min(32, scn.cycles.samples)
    scn.cycles.use_adaptive_sampling = t.get("adaptive", True)
    scn.cycles.adaptive_threshold = t.get("adaptive_threshold", 0.004)
    scn.cycles.seed = t.get("seed", 0)
    scn.view_settings.exposure = t.get("exposure", 0.0)
    scn.frame_set(t.get("frame", 1))
    if t.get("border"):
        scn.render.use_border = True; scn.render.use_crop_to_border = True
        (scn.render.border_min_x, scn.render.border_max_x,
         scn.render.border_min_y, scn.render.border_max_y) = t["border"]
    else:
        scn.render.use_border = False
    out = t["out"]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    node = B.kerr_node()
    B.log("TASK", os.path.basename(out), {s.name: (tuple(round(x, 4) for x in s.default_value) if hasattr(s.default_value, '__len__') else round(s.default_value, 7))
                                          for s in node.inputs if hasattr(s, 'default_value')})
    scn.render.image_settings.file_format = 'OPEN_EXR'
    scn.render.image_settings.color_depth = '32'
    scn.render.image_settings.exr_codec = 'ZIP'
    scn.render.filepath = out + ".exr"
    t1 = time.time()
    bpy.ops.render.render(write_still=True)
    B.log("RENDER_SECONDS", os.path.basename(out), "%.1f" % (time.time() - t1))
    img = bpy.data.images.load(out + ".exr", check_existing=False)
    scn.render.image_settings.file_format = 'PNG'
    scn.render.image_settings.color_depth = '8'
    img.save_render(out + ".png", scene=scn)
    bpy.data.images.remove(img)
    comps = dict(t.get("comps", {}))
    if t.get("comp") is not None:
        comps["comp"] = t["comp"]
    for cname, cset in comps.items():
        import comp as C
        t2 = time.time()
        C.comp(out + ".exr", out + "_" + cname + ".png", cset)
        B.log("COMP_SECONDS", os.path.basename(out), cname, "%.1f" % (time.time() - t2))
B.log("ALL DONE %.1fs" % (time.time() - t0))
