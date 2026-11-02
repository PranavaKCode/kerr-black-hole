# builds the kerr black hole scene in blender 5.2 from scratch: world node group around the osl
# shader, orbit camera, drivers, cycles settings, compositor grade and the sidebar panel.
#   blender -b --factory-startup --python build_scene.py -- [--shot hero] [--save out.blend] [--device OPTIX|CUDA|CPU]
import bpy, math, os, sys
from mathutils import Vector, Quaternion

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import shots as SHOTS

# (osl name, ui label, kind, default, min, max)
PARAMS = [
    ("Spin",             "Spin a",                  "F", 0.99, 0.0, 0.999),
    ("DiskInner",        "Disk Inner Edge (x ISCO)", "F", 1.0, 1.0, 8.0),
    ("DiskOuter",        "Disk Outer Radius (M)",   "F", 34.0, 4.0, 80.0),
    ("DiskThickness",    "Disk Thickness H/R",      "F", 0.02, 0.002, 0.3),
    ("Turbulence",       "Turbulence",              "F", 1.0, 0.0, 1.0),
    ("Swirl",            "Shear Time (M)",          "F", 90.0, 0.0, 2000.0),
    ("Time",             "Time (M)",                "F", 0.0, -1e6, 1e6),
    ("PeakTemp",         "Peak Temperature (K)",    "F", 7500.0, 1500.0, 40000.0),
    ("DiskBrightness",   "Disk Brightness",         "F", 1.0, 0.0, 100.0),
    ("DiskOpacity",      "Disk Opacity",            "F", 2.0, 0.0, 50.0),
    ("PlungeBrightness", "Plunging Flow",           "F", 0.35, 0.0, 2.0),
    ("DustLanes",        "Dust Lanes",              "F", 1.0, 0.0, 1.0),
    ("IrradTemp",        "Irradiation T at 10M (K)", "F", 5000.0, 0.0, 20000.0),
    ("TempProfile",      "Temperature Exponent",    "F", 0.75, 0.3, 1.2),
    ("Doppler",          "Doppler / Beaming",       "F", 1.0, 0.0, 1.0),
    ("GravRedshift",     "Gravitational Redshift",  "F", 1.0, 0.0, 1.0),
    ("StarBrightness",   "Star Brightness",         "F", 1.6, 0.0, 100.0),
    ("StarSize",         "Star Size (microrad)",    "F", 20.0, 0.1, 10000.0),
    ("StarDensity",      "Star Density",            "F", 600.0, 10.0, 20000.0),
    ("GalaxyBrightness", "Galaxy Brightness",       "F", 0.02, 0.0, 10.0),
    ("GalaxyNormal",     "Galaxy Plane Normal",     "V", (0.42, 0.0, -0.91), None, None),
    ("GalaxyWidth",      "Galaxy Width (rad)",      "F", 0.06, 0.002, 1.0),
    ("Accuracy",         "Step Size",               "F", 0.04, 0.005, 0.2),
    ("VolumeStep",       "Volume Step (x H)",       "F", 0.2, 0.05, 2.0),
    ("MaxSteps",         "Step Budget",             "I", 3000, 100, 20000),
    ("DebugMode",        "Debug Mode",              "I", 0, 0, 6),
]
LABEL = {p[0]: p[1] for p in PARAMS}
GROUP_NAME = "Kerr Black Hole"


def log(*a):
    print("[build]", *a, flush=True)


def load_osl_text():
    src = open(os.path.join(HERE, "kerr_bh.osl"), encoding="utf-8").read()
    txt = bpy.data.texts.get("kerr_bh.osl") or bpy.data.texts.new("kerr_bh.osl")
    txt.clear(); txt.write(src)
    return txt


def make_group():
    import cycles.osl as cosl
    ng = bpy.data.node_groups.get(GROUP_NAME)
    if ng:
        bpy.data.node_groups.remove(ng)
    ng = bpy.data.node_groups.new(GROUP_NAME, 'ShaderNodeTree')
    ng.description = "Physically based Kerr black hole: GR null-geodesic ray tracer + volumetric accretion disk (OSL)"
    gi = ng.nodes.new('NodeGroupInput'); gi.location = (-500, 0)
    go = ng.nodes.new('NodeGroupOutput'); go.location = (400, 0)
    sn = ng.nodes.new('ShaderNodeScript'); sn.location = (0, 0); sn.name = "KerrOSL"
    sn.mode = 'INTERNAL'; sn.script = load_osl_text()
    ok = cosl.update_script_node(sn, lambda t, m: log("OSL", t, m))
    if not ok:
        raise RuntimeError("OSL compile failed")
    panels = {}
    def panel(name):
        if name not in panels:
            panels[name] = ng.interface.new_panel(name)
        return panels[name]
    grouping = {
        "Black Hole": ["Spin"],
        "Accretion Disk": ["DiskInner", "DiskOuter", "DiskThickness", "Turbulence", "Swirl", "Time",
                           "PeakTemp", "TempProfile", "IrradTemp", "DiskBrightness", "DiskOpacity",
                           "PlungeBrightness", "DustLanes"],
        "Relativity": ["Doppler", "GravRedshift"],
        "Sky": ["StarBrightness", "StarSize", "StarDensity", "GalaxyBrightness", "GalaxyNormal", "GalaxyWidth"],
        "Quality": ["Accuracy", "VolumeStep", "MaxSteps", "DebugMode"],
    }
    where = {k: g for g, ks in grouping.items() for k in ks}
    ng.interface.new_socket("Color", in_out='OUTPUT', socket_type='NodeSocketColor')
    for name, label, kind, dflt, lo, hi in PARAMS:
        stype = {"F": 'NodeSocketFloat', "I": 'NodeSocketInt', "V": 'NodeSocketVector'}[kind]
        s = ng.interface.new_socket(label, in_out='INPUT', socket_type=stype, parent=panel(where[name]))
        if kind in "FI":
            s.min_value = lo; s.max_value = hi
        s.default_value = dflt
        if name in ("Doppler", "GravRedshift", "Turbulence", "DustLanes"):
            s.subtype = 'FACTOR'
        ng.links.new(gi.outputs[label], sn.inputs[name])
        sn.inputs[name].default_value = dflt
    ng.links.new(sn.outputs["Color"], go.inputs["Color"])
    return ng


def make_world(ng):
    w = bpy.data.worlds.get("Kerr World") or bpy.data.worlds.new("Kerr World")
    scn = bpy.context.scene
    scn.world = w
    nt = w.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    g = nt.nodes.new('ShaderNodeGroup'); g.node_tree = ng; g.name = GROUP_NAME; g.label = GROUP_NAME
    g.location = (-300, 0); g.width = 300
    bg = nt.nodes.new('ShaderNodeBackground'); bg.location = (100, 0)
    wo = nt.nodes.new('ShaderNodeOutputWorld'); wo.location = (320, 0)
    nt.links.new(g.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs[0], wo.inputs[0])
    w.cycles.sampling_method = 'NONE'        # never bake an importance map of a gr shader
    try:
        w.cycles.volume_sampling = 'DISTANCE'
    except Exception:
        pass
    return w, g


def kerr_node():
    return bpy.context.scene.world.node_tree.nodes[GROUP_NAME]


def set_param(name, value):
    node = kerr_node()
    sock = node.inputs[LABEL[name]]
    if isinstance(sock.default_value, (float, int)):
        sock.default_value = value
    else:
        sock.default_value = tuple(value)


def place_camera(cam, r_obs, incl_deg, azim_deg, roll_deg=0.0, lens=200.0, shift_x=0.0, shift_y=0.0,
                 look_at=(0.0, 0.0, 0.0)):
    th = math.radians(incl_deg); ph = math.radians(azim_deg)
    cam.location = Vector((r_obs * math.sin(th) * math.cos(ph), r_obs * math.sin(th) * math.sin(ph),
                           r_obs * math.cos(th)))
    fwd = Vector(look_at) - cam.location
    q = fwd.to_track_quat('-Z', 'Y')
    cam.rotation_mode = 'QUATERNION'
    cam.rotation_quaternion = q @ Quaternion((0, 0, 1), math.radians(roll_deg))
    cam.data.lens = lens
    cam.data.sensor_fit = 'HORIZONTAL'
    cam.data.sensor_width = 36.0
    cam.data.shift_x = shift_x; cam.data.shift_y = shift_y
    cam.data.clip_start = 0.1; cam.data.clip_end = 1e6
    cam["kerr_r_obs"] = r_obs; cam["kerr_incl"] = incl_deg; cam["kerr_azim"] = azim_deg; cam["kerr_roll"] = roll_deg


def make_camera():
    scn = bpy.context.scene
    cam = bpy.data.objects.get("Kerr Camera")
    if cam is None:
        cd = bpy.data.cameras.new("Kerr Camera")
        cam = bpy.data.objects.new("Kerr Camera", cd)
        scn.collection.objects.link(cam)
    scn.camera = cam
    return cam


def add_drivers(cam):
    node = kerr_node()
    # star size ~ 0.75 px (angular) so unlensed stars stay crisp points at any lens / resolution
    sock = node.inputs[LABEL["StarSize"]]
    try:
        sock.driver_remove("default_value")
    except Exception:
        pass
    fc = sock.driver_add("default_value")
    drv = fc.driver; drv.type = 'SCRIPTED'
    for nm, idtype, idb, path in (("lens", 'CAMERA', cam.data, "lens"),
                                  ("sw", 'CAMERA', cam.data, "sensor_width"),
                                  ("rx", 'SCENE', bpy.context.scene, "render.resolution_x"),
                                  ("pct", 'SCENE', bpy.context.scene, "render.resolution_percentage")):
        v = drv.variables.new(); v.name = nm; v.type = 'SINGLE_PROP'
        v.targets[0].id_type = idtype; v.targets[0].id = idb; v.targets[0].data_path = path
    drv.expression = "0.75e6 * sw / (lens * rx * pct / 100)"   # microradians (drivers round tiny values)
    # time follows the frame, 1 frame = 2 M of coordinate time
    sock = node.inputs[LABEL["Time"]]
    try:
        sock.driver_remove("default_value")
    except Exception:
        pass
    fc = sock.driver_add("default_value"); drv = fc.driver; drv.type = 'SCRIPTED'
    drv.expression = "2.0 * frame"


def setup_render(res=(1920, 1080), spp=64, device="OPTIX", seed=0):
    scn = bpy.context.scene
    scn.render.engine = 'CYCLES'
    c = scn.cycles
    c.shading_system = True
    if device in ("OPTIX", "CUDA"):
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type = device
        prefs.get_devices()
        n = 0
        for d in prefs.devices:
            d.use = (d.type == device); n += d.use
        log("GPU devices:", [(d.name, d.type, d.use) for d in prefs.devices])
        c.device = 'GPU' if n else 'CPU'
    else:
        c.device = 'CPU'
    c.samples = spp
    c.use_adaptive_sampling = True
    c.adaptive_threshold = 0.004
    c.adaptive_min_samples = min(32, spp)
    c.use_denoising = False
    c.max_bounces = 0; c.diffuse_bounces = 0; c.glossy_bounces = 0
    c.transmission_bounces = 0; c.volume_bounces = 0; c.transparent_max_bounces = 0
    c.sample_clamp_direct = 0.0; c.sample_clamp_indirect = 0.0
    c.seed = seed
    c.film_exposure = 1.0
    scn.render.film_transparent = False
    scn.render.filter_size = 1.2
    scn.render.resolution_x, scn.render.resolution_y = res
    scn.render.resolution_percentage = 100
    scn.render.use_compositing = False
    scn.render.use_sequencer = False
    vs = scn.view_settings
    try:
        vs.view_transform = 'AgX'
        vs.look = 'AgX - Base Contrast'
    except Exception as e:
        log("view transform:", e)
    vs.exposure = 0.0; vs.gamma = 1.0


def build_compositor():
    # in-scene compositor with the hero grade (comp.HERO)
    import comp as C
    scn = bpy.context.scene
    st = dict(C.DEFAULT); st.update(C.HERO)
    cg = C.new_tree("Kerr Comp")
    scn.compositing_node_group = cg
    rl = cg.nodes.new('CompositorNodeRLayers'); rl.location = (-850, 0)
    C.make_nodes(cg, rl.outputs['Image'], st)
    C.apply_view(scn, st)
    return cg


def install_panel_text():
    p = os.path.join(HERE, "kerr_panel.py")
    if not os.path.exists(p):
        return None
    t = bpy.data.texts.get("kerr_panel.py") or bpy.data.texts.new("kerr_panel.py")
    t.clear(); t.write(open(p, encoding="utf-8").read()); t.use_module = True
    return t


def apply_shot(name_or_dict, overrides=None):
    shot = SHOTS.get(name_or_dict) if isinstance(name_or_dict, str) else dict(name_or_dict)
    if overrides:
        shot.update(overrides)
    cam = make_camera()
    for name, label, kind, dflt, lo, hi in PARAMS:      # reset to defaults first
        if name not in ("StarSize", "Time"):
            set_param(name, dflt)
    place_camera(cam, shot["r_obs"], shot["incl"], shot["azim"], shot.get("roll", 0.0), shot["lens"],
                 shot.get("shift_x", 0.0), shot.get("shift_y", 0.0), shot.get("look_at", (0, 0, 0)))
    if "gal_tilt" in shot:
        set_param("GalaxyNormal", galaxy_normal(cam, shot["gal_tilt"], shot.get("gal_offset", 0.0)))
    for k, v in shot.get("params", {}).items():
        set_param(k, v)
    return shot


def galaxy_normal(cam, tilt_deg, offset_deg):
    # pole of a galactic band crossing the view behind the hole at tilt_deg to the image
    # horizontal, offset by offset_deg perpendicular to itself
    from mathutils import Matrix
    v = (-cam.location).normalized()
    m = cam.rotation_quaternion.to_matrix()
    right = (m @ Vector((1, 0, 0))).normalized(); up = (m @ Vector((0, 1, 0))).normalized()
    t = math.cos(math.radians(tilt_deg)) * right + math.sin(math.radians(tilt_deg)) * up
    n = v.cross(t).normalized()
    n = Matrix.Rotation(math.radians(offset_deg), 3, t) @ n
    return tuple(n)


def build(shot="hero", device="OPTIX", res=(1920, 1080), spp=64):
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    scn = bpy.context.scene
    scn.name = "Kerr"
    setup_render(res, spp, device)
    ng = make_group()
    make_world(ng)
    cam = make_camera()
    shot = apply_shot(shot)
    add_drivers(cam)
    build_compositor()
    install_panel_text()
    scn.frame_start = 1; scn.frame_end = 240; scn.render.fps = 24
    return shot


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", default="hero")
    ap.add_argument("--save", default=None)
    ap.add_argument("--device", default="OPTIX")
    a = ap.parse_args(argv)
    build(a.shot, a.device)
    if a.save:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(a.save), compress=True)
        log("saved", a.save)
