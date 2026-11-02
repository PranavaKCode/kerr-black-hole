# compositor grade: exposure -> bloom -> [streaks] -> linear power -> hue/sat, plus view transform and
# white balance. used inside the scene (build_scene.build_compositor) and as a separate pass that
# regrades a rendered exr without re-tracing (comp(), comp_batch.py).
import bpy, math

DEFAULT = dict(exposure=0.0, bloom=0.08, bloom_thresh=0.8, bloom_size=1.0, streak=0.0,
               streak_thresh=6.0, streak_fade=0.92, streaks=4, streak_angle=45.0, sat=1.0, hue=0.5,
               look="AgX - Base Contrast", gamma=1.0)
# hero grade: khronos pbr neutral keeps the blue-white of the beamed side, linear power 0.78
# compresses the ~7-stop doppler contrast, 7000 K white balance, a soft wide bloom
HERO = dict(exposure=0.4, gamma=0.78, bloom=0.06, bloom_size=0.6, bloom_thresh=1.8, sat=1.45, hue=0.494,
            wb=7000.0, view='Khronos PBR Neutral')


def apply_view(scn, st):
    vs = scn.view_settings
    vs.view_transform = st.get("view", 'AgX')
    try:
        vs.look = st["look"] if vs.view_transform == 'AgX' else 'None'
    except Exception as e:
        print("[comp] look", e)
    vs.exposure = 0.0; vs.gamma = 1.0
    vs.use_white_balance = "wb" in st
    if "wb" in st:      # camera white balance (K): > 6500 warms the image, < 6500 cools it
        vs.white_balance_temperature = st["wb"]
        vs.white_balance_tint = st.get("tint", 10.0)
    print("[comp] view", vs.view_transform, "look", vs.look, "wb",
          vs.use_white_balance and vs.white_balance_temperature)


def make_nodes(cg, src_socket, st):
    # grade nodes after src_socket, returns the final socket
    N = cg.nodes; Lk = cg.links
    x = -600
    def place(n):
        nonlocal x
        n.location = (x, 0); x += 220
        return n
    ex = place(N.new('CompositorNodeExposure')); ex.inputs['Exposure'].default_value = st["exposure"]
    Lk.new(src_socket, ex.inputs['Image'])
    cur = ex.outputs['Image']
    if st["bloom"] > 0:
        gb = place(N.new('CompositorNodeGlare')); gb.name = gb.label = "Bloom"
        gb.inputs['Type'].default_value = 'Bloom'
        gb.inputs['Quality'].default_value = 'High'
        gb.inputs['Threshold'].default_value = st["bloom_thresh"]
        gb.inputs['Smoothness'].default_value = 1.0
        gb.inputs['Strength'].default_value = st["bloom"]
        gb.inputs['Size'].default_value = st["bloom_size"]
        Lk.new(cur, gb.inputs['Image']); cur = gb.outputs['Image']
    if st["streak"] > 0:
        gs = place(N.new('CompositorNodeGlare')); gs.name = gs.label = "Streaks"
        gs.inputs['Type'].default_value = 'Streaks'
        gs.inputs['Quality'].default_value = 'High'
        gs.inputs['Threshold'].default_value = st["streak_thresh"]
        gs.inputs['Smoothness'].default_value = 1.0
        gs.inputs['Strength'].default_value = st["streak"]
        gs.inputs['Streaks'].default_value = st["streaks"]
        gs.inputs['Streaks Angle'].default_value = math.radians(st["streak_angle"])
        gs.inputs['Fade'].default_value = st["streak_fade"]
        Lk.new(cur, gs.inputs['Image']); cur = gs.outputs['Image']
    if abs(st["gamma"] - 1.0) > 1e-3:
        # linear power curve before the view transform
        gm = place(N.new('ShaderNodeGamma')); gm.label = "Linear power"
        gm.inputs['Gamma'].default_value = st["gamma"]
        Lk.new(cur, gm.inputs['Color']); cur = gm.outputs['Color']
    if abs(st["sat"] - 1.0) > 1e-3 or abs(st.get("hue", 0.5) - 0.5) > 1e-4:
        hs = place(N.new('CompositorNodeHueSat'))
        hs.inputs['Saturation'].default_value = st["sat"]
        hs.inputs['Hue'].default_value = st.get("hue", 0.5)
        Lk.new(cur, hs.inputs['Image']); cur = hs.outputs['Image']
    go = place(N.new('NodeGroupOutput'))
    Lk.new(cur, go.inputs['Image'])
    return cur


def new_tree(name):
    old = bpy.data.node_groups.get(name)
    if old:
        bpy.data.node_groups.remove(old)
    cg = bpy.data.node_groups.new(name, 'CompositorNodeTree')
    cg.interface.new_socket("Image", in_out='OUTPUT', socket_type='NodeSocketColor')
    return cg


def comp(exr_path, out_path, settings=None, fmt='PNG', depth='16'):
    st = dict(DEFAULT); st.update(settings or {})
    img = bpy.data.images.load(exr_path, check_existing=False)
    W, H = img.size
    scn = bpy.data.scenes.get("KerrComp") or bpy.data.scenes.new("KerrComp")
    scn.render.engine = 'CYCLES'
    scn.cycles.device = 'CPU'; scn.cycles.samples = 1; scn.cycles.use_denoising = False
    scn.render.resolution_x, scn.render.resolution_y = W, H
    scn.render.resolution_percentage = 100
    scn.render.use_compositing = True
    scn.render.use_border = False
    scn.world = None
    if scn.camera is None:
        cd = bpy.data.cameras.new("CompCam"); co = bpy.data.objects.new("CompCam", cd)
        scn.collection.objects.link(co); scn.camera = co
    apply_view(scn, st)
    cg = new_tree("Kerr Comp Pass")
    scn.compositing_node_group = cg
    src = cg.nodes.new('CompositorNodeImage'); src.image = img; src.location = (-850, 0)
    make_nodes(cg, src.outputs['Image'], st)
    scn.render.image_settings.file_format = fmt
    if fmt == 'PNG':
        scn.render.image_settings.color_depth = depth
        scn.render.image_settings.compression = 15
    elif fmt == 'JPEG':
        scn.render.image_settings.quality = 92
    scn.render.filepath = out_path
    bpy.ops.render.render(write_still=True, scene=scn.name)
    bpy.data.images.remove(img)
    return out_path
