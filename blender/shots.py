# named camera + parameter presets. distances in M, angles in degrees,
# incl = angle from the spin axis (90 = edge-on), azim = camera azimuth.
import copy

from kerr_panel import PRESETS      # the presets are defined once, in the panel add-on


def _preset_shot(key, **cam):
    label, desc, params, c = PRESETS[key]
    d = dict(c); d.update(cam); d["params"] = dict(params)
    return d


SHOTS = {
    # the four panel presets, framed like the hero
    "gargantua": _preset_shot("GARGANTUA", shift_x=-0.08, shift_y=0.0, gal_tilt=15.0, gal_offset=-3.5),
    "physical": _preset_shot("PHYSICAL", shift_x=-0.08, shift_y=0.0, gal_tilt=15.0, gal_offset=-3.5),
    "schwarzschild": _preset_shot("SCHWARZSCHILD", shift_x=-0.08, shift_y=0.0, gal_tilt=15.0, gal_offset=-3.5),
    "faceon": _preset_shot("FACEON", gal_tilt=15.0, gal_offset=-3.5),
    "hero": _preset_shot("PHYSICAL", shift_x=-0.08, shift_y=0.0, gal_tilt=15.0, gal_offset=-3.5),
    # validation: capture mask (DebugMode 3) seen by a distant observer
    "validate": dict(r_obs=1000.0, incl=85.0, azim=-90.0, lens=1500.0, gal_tilt=28.0, gal_offset=0.25,
                     params=dict(Spin=0.99, DebugMode=3, Accuracy=0.02, MaxSteps=8000)),
}


def get(name):
    return copy.deepcopy(SHOTS[name])
