# compositor-only passes over rendered exrs (no tracing).
#   blender -b --factory-startup --python comp_batch.py -- comps.json
#   comps.json = [{"exr": "out/x.exr", "out": "out/x_grade.png", "settings": {...}}, ...]
import sys, os, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import comp as C
for t in json.load(open(sys.argv[sys.argv.index("--") + 1])):
    t0 = time.time()
    C.comp(t["exr"], t["out"], t.get("settings", {}), t.get("fmt", "PNG"))
    print("[comp_batch]", os.path.basename(t["out"]), "%.1fs" % (time.time() - t0), flush=True)
