# sidebar panel (3d viewport > n-panel > kerr) for the "Kerr Black Hole" world node group: grouped
# sliders, four presets and an orbit camera at (r, inclination, azimuth) looking at the hole.
# build_scene.py embeds it as a text block registered on load; it also installs as an add-on.
bl_info = {
    "name": "Kerr Black Hole",
    "version": (1, 0, 0),
    "blender": (5, 2, 0),
    "location": "3D Viewport > Sidebar > Kerr",
    "description": "Controls for the physically based Kerr black hole OSL world shader",
    "category": "Render",
}

import bpy
import math
from bpy.props import FloatProperty, EnumProperty
from mathutils import Vector, Quaternion

GROUP = "Kerr Black Hole"

# shared look (disk, sky, quality); presets only change the physics and camera
BASE = dict(PeakTemp=7500.0, IrradTemp=5000.0, TempProfile=0.75, DiskInner=1.0, DiskOuter=34.0,
            DiskOpacity=2.0, DiskThickness=0.02, Swirl=90.0, Turbulence=1.0, DustLanes=1.0,
            PlungeBrightness=0.35, DiskBrightness=1.0, StarBrightness=1.6, StarDensity=600.0,
            GalaxyBrightness=0.02, GalaxyWidth=0.06, Accuracy=0.04, VolumeStep=0.2, MaxSteps=3000, DebugMode=0)
HERO_CAM = dict(r_obs=300.0, incl=86.5, azim=-90.0, lens=220.0)


def _p(**kw):
    d = dict(BASE); d.update(kw); return d


# preset -> (label, tooltip, shader params, camera)
PRESETS = {
    "GARGANTUA": ("Gargantua (film)",
                  "a = 0.6 as used for the film's renders, Doppler shift and beaming removed (as Nolan chose), "
                  "no gravitational redshift, warmer disk",
                  _p(Spin=0.6, Doppler=0.0, GravRedshift=0.0, PeakTemp=5200.0, IrradTemp=4200.0),
                  dict(HERO_CAM)),
    "PHYSICAL": ("Physical Kerr a=0.99",
                 "Near-extremal spin with full Doppler + beaming + gravitational redshift: "
                 "D-shaped shadow, blue-white approaching side, dim orange receding side",
                 _p(Spin=0.99, Doppler=1.0, GravRedshift=1.0),
                 dict(HERO_CAM)),
    "SCHWARZSCHILD": ("Schwarzschild a=0",
                      "Non-rotating hole: circular shadow of radius sqrt(27) M, ISCO at 6 M",
                      _p(Spin=0.0, Doppler=1.0, GravRedshift=1.0),
                      dict(HERO_CAM)),
    "FACEON": ("Face-on",
               "Looking down toward the spin axis (30 deg): the whole disk as a lensed whirlpool",
               _p(Spin=0.99, Doppler=1.0, GravRedshift=1.0, DiskOpacity=8.0),
               dict(r_obs=300.0, incl=30.0, azim=-90.0, lens=245.0)),
}

# socket labels, must match build_scene.PARAMS
LABELS = {
    "Spin": "Spin a", "DiskInner": "Disk Inner Edge (x ISCO)", "DiskOuter": "Disk Outer Radius (M)",
    "DiskThickness": "Disk Thickness H/R", "Turbulence": "Turbulence", "Swirl": "Shear Time (M)",
    "Time": "Time (M)", "PeakTemp": "Peak Temperature (K)", "TempProfile": "Temperature Exponent",
    "IrradTemp": "Irradiation T at 10M (K)", "DiskBrightness": "Disk Brightness",
    "DiskOpacity": "Disk Opacity", "PlungeBrightness": "Plunging Flow", "DustLanes": "Dust Lanes",
    "Doppler": "Doppler / Beaming", "GravRedshift": "Gravitational Redshift",
    "StarBrightness": "Star Brightness", "StarSize": "Star Size (microrad)", "StarDensity": "Star Density",
    "GalaxyBrightness": "Galaxy Brightness", "GalaxyNormal": "Galaxy Plane Normal",
    "GalaxyWidth": "Galaxy Width (rad)",
    "Accuracy": "Step Size", "VolumeStep": "Volume Step (x H)", "MaxSteps": "Step Budget",
    "DebugMode": "Debug Mode",
}

SECTIONS = [
    ("Black Hole", "WORLD", ["Spin"]),
    ("Relativity", "LIGHT_SUN", ["Doppler", "GravRedshift"]),
    ("Accretion Disk", "MESH_CIRCLE", ["DiskInner", "DiskOuter", "DiskThickness", "Turbulence", "Swirl",
                                       "PeakTemp", "TempProfile", "IrradTemp", "DiskBrightness",
                                       "DiskOpacity", "PlungeBrightness", "DustLanes"]),
    ("Sky", "STICKY_UVS_DISABLE", ["StarBrightness", "StarDensity", "GalaxyBrightness", "GalaxyWidth", "GalaxyNormal"]),
    ("Quality", "PREFERENCES", ["Accuracy", "VolumeStep", "MaxSteps", "DebugMode"]),
]


def kerr_node(context):
    w = context.scene.world
    if not (w and w.node_tree):
        return None
    for n in w.node_tree.nodes:
        if n.type == 'GROUP' and n.node_tree and n.node_tree.name.startswith(GROUP):
            return n
    return None


def kerr_radii(a):
    a = min(max(a, 0.0), 0.9999)
    rh = 1.0 + math.sqrt(1.0 - a * a)
    z1 = 1.0 + (1.0 - a * a) ** (1 / 3) * ((1.0 + a) ** (1 / 3) + (1.0 - a) ** (1 / 3))
    z2 = math.sqrt(3.0 * a * a + z1 * z1)
    risco = 3.0 + z2 - math.sqrt(max((3.0 - z1) * (3.0 + z1 + 2.0 * z2), 0.0))
    rp = 2.0 * (1.0 + math.cos(2.0 / 3.0 * math.acos(-a)))
    rr = 2.0 * (1.0 + math.cos(2.0 / 3.0 * math.acos(a)))
    return rh, risco, rp, rr


def place_camera(cam, r_obs, incl_deg, azim_deg, roll_deg=0.0):
    th = math.radians(incl_deg)
    ph = math.radians(azim_deg)
    cam.location = Vector((r_obs * math.sin(th) * math.cos(ph), r_obs * math.sin(th) * math.sin(ph),
                           r_obs * math.cos(th)))
    q = (-cam.location).to_track_quat('-Z', 'Y')
    cam.rotation_mode = 'QUATERNION'
    cam.rotation_quaternion = q @ Quaternion((0, 0, 1), math.radians(roll_deg))


def _cam_update(self, context):
    cam = context.scene.camera
    if cam is None:
        return
    s = context.scene
    place_camera(cam, s.kerr_r_obs, s.kerr_incl, s.kerr_azim, s.kerr_roll)


def apply_preset(context, key):
    label, desc, params, cam = PRESETS[key]
    node = kerr_node(context)
    if node is None:
        return False
    for k, v in params.items():
        node.inputs[LABELS[k]].default_value = v
    s = context.scene
    s.kerr_r_obs, s.kerr_incl, s.kerr_azim = cam["r_obs"], cam["incl"], cam["azim"]
    if s.camera:
        s.camera.data.lens = cam["lens"]
    _cam_update(s, context)
    return True


class KERR_OT_preset(bpy.types.Operator):
    bl_idname = "kerr.preset"
    bl_label = "Kerr Preset"
    bl_description = "Apply a black-hole preset"
    bl_options = {'REGISTER', 'UNDO'}
    preset: EnumProperty(items=[(k, v[0], v[1]) for k, v in PRESETS.items()])

    @classmethod
    def description(cls, context, props):
        return PRESETS[props.preset][1]

    def execute(self, context):
        if not apply_preset(context, self.preset):
            self.report({'ERROR'}, "No 'Kerr Black Hole' node group in the world")
            return {'CANCELLED'}
        return {'FINISHED'}


class KERR_PT_panel(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Kerr"
    bl_label = "Kerr Black Hole"

    def draw(self, context):
        lay = self.layout
        node = kerr_node(context)
        if node is None:
            lay.label(text="World has no 'Kerr Black Hole' group", icon='ERROR')
            return
        col = lay.column(align=True)
        col.label(text="Presets")
        g = col.grid_flow(columns=2, align=True)
        for k, v in PRESETS.items():
            g.operator("kerr.preset", text=v[0]).preset = k

        a = node.inputs[LABELS["Spin"]].default_value
        rh, risco, rp, rr = kerr_radii(a)
        box = lay.box()
        col2 = box.column(align=True)
        col2.label(text=f"r+ {rh:.3f} M   ISCO {risco:.3f} M", icon='INFO')
        col2.label(text=f"photon orbits {rp:.2f} / {rr:.2f} M", icon='BLANK1')

        for title, icon, keys in SECTIONS:
            b = lay.box()
            b.label(text=title, icon=icon)
            c = b.column(align=True)
            for k in keys:
                sock = node.inputs.get(LABELS[k])
                if sock is not None:
                    c.prop(sock, "default_value", text=LABELS[k])

        b = lay.box()
        b.label(text="Observer (camera)", icon='CAMERA_DATA')
        c = b.column(align=True)
        s = context.scene
        c.prop(s, "kerr_r_obs")
        c.prop(s, "kerr_incl")
        c.prop(s, "kerr_azim")
        c.prop(s, "kerr_roll")
        if s.camera:
            c.prop(s.camera.data, "lens", text="Focal Length")
        c.prop(s.view_settings, "exposure", text="Exposure")


classes = (KERR_OT_preset, KERR_PT_panel)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    S = bpy.types.Scene
    S.kerr_r_obs = FloatProperty(name="Distance r (M)", default=300.0, min=4.0, max=1e5, update=_cam_update)
    S.kerr_incl = FloatProperty(name="Inclination (deg)", default=84.0, min=0.5, max=179.5, update=_cam_update,
                                description="Angle between the line of sight and the spin axis (90 = edge-on)")
    S.kerr_azim = FloatProperty(name="Azimuth (deg)", default=-90.0, min=-360.0, max=360.0, update=_cam_update)
    S.kerr_roll = FloatProperty(name="Roll (deg)", default=0.0, min=-180.0, max=180.0, update=_cam_update)


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
    for p in ("kerr_r_obs", "kerr_incl", "kerr_azim", "kerr_roll"):
        if hasattr(bpy.types.Scene, p):
            delattr(bpy.types.Scene, p)


if __name__ == "__main__":
    register()
