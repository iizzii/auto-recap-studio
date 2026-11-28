#!/usr/bin/env python3
"""Headless Blender 3D Cycles Pixar-style renderer for Kiko (25 RGBA sprites + 4 genre booths).

Run via:
    blender --background --python scripts/render_kiko_blender_3d.py -- \
        --sprites-dir assets/host/sprites --booths-dir assets/host/booths
"""

from __future__ import annotations

import argparse
import contextlib
import math
import sys
from pathlib import Path
from typing import Any

import bpy  # type: ignore[import-not-found]
from mathutils import Euler, Vector  # type: ignore[import-not-found]

EMOTIONS: tuple[str, ...] = (
    "neutral",
    "laughing",
    "shocked",
    "horrified",
    "smug",
    "hyped",
    "sad",
    "facepalm",
)
MOUTHS: tuple[str, ...] = ("closed", "half", "open")
GENRES: tuple[str, ...] = ("comedy", "horror_comedy", "action_scifi", "emotional_drama")


def _parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--sprites-dir", default="assets/host/sprites")
    parser.add_argument("--booths-dir", default="assets/host/booths")
    parser.add_argument("--samples", type=int, default=28)
    return parser.parse_args(argv)


def _clear_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _set_input(node: Any, names: tuple[str, ...], value: Any) -> None:
    for name in names:
        if name in node.inputs:
            node.inputs[name].default_value = value
            return


def _smooth_and_subsurf(obj: Any, levels: int = 2) -> None:
    for poly in obj.data.polygons:
        poly.use_smooth = True
    if levels > 0:
        mod = obj.modifiers.new(name="Subsurf", type="SUBSURF")
        mod.levels = levels
        mod.render_levels = levels


def _add_sphere(
    name: str,
    loc: tuple[float, float, float],
    scale: tuple[float, float, float],
    rot: tuple[float, float, float] = (0.0, 0.0, 0.0),
    mat: Any = None,
    subsurf: int = 2,
    segments: int = 32,
    rings: int = 16,
) -> Any:
    bpy.ops.mesh.primitive_uv_sphere_add(segments=segments, ring_count=rings, radius=1.0, location=loc)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = scale
    obj.rotation_euler = Euler(rot, "XYZ")
    _smooth_and_subsurf(obj, subsurf)
    if mat is not None:
        obj.data.materials.append(mat)
    return obj


def _add_cylinder(
    name: str,
    loc: tuple[float, float, float],
    scale: tuple[float, float, float],
    rot: tuple[float, float, float] = (0.0, 0.0, 0.0),
    mat: Any = None,
    vertices: int = 32,
    subsurf: int = 1,
) -> Any:
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=1.0, depth=2.0, location=loc)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = scale
    obj.rotation_euler = Euler(rot, "XYZ")
    _smooth_and_subsurf(obj, subsurf)
    if mat is not None:
        obj.data.materials.append(mat)
    return obj


def _add_cone(
    name: str,
    loc: tuple[float, float, float],
    scale: tuple[float, float, float],
    rot: tuple[float, float, float] = (0.0, 0.0, 0.0),
    mat: Any = None,
    r1: float = 1.0,
    r2: float = 0.14,
    subsurf: int = 2,
) -> Any:
    bpy.ops.mesh.primitive_cone_add(vertices=28, radius1=r1, radius2=r2, depth=2.0, location=loc)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = scale
    obj.rotation_euler = Euler(rot, "XYZ")
    _smooth_and_subsurf(obj, subsurf)
    if mat is not None:
        obj.data.materials.append(mat)
    return obj


def _add_torus(
    name: str,
    loc: tuple[float, float, float],
    rot: tuple[float, float, float],
    major_r: float,
    minor_r: float,
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    mat: Any = None,
) -> Any:
    bpy.ops.mesh.primitive_torus_add(
        major_segments=48,
        minor_segments=20,
        major_radius=major_r,
        minor_radius=minor_r,
        location=loc,
        rotation=rot,
    )
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = scale
    _smooth_and_subsurf(obj, 1)
    if mat is not None:
        obj.data.materials.append(mat)
    return obj


def _add_cube(
    name: str,
    loc: tuple[float, float, float],
    scale: tuple[float, float, float],
    rot: tuple[float, float, float] = (0.0, 0.0, 0.0),
    mat: Any = None,
    bevel: float = 0.04,
) -> Any:
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=loc)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = scale
    obj.rotation_euler = Euler(rot, "XYZ")
    if bevel > 0.0:
        b = obj.modifiers.new(name="Bevel", type="BEVEL")
        b.width = bevel
        b.segments = 3
    for poly in obj.data.polygons:
        poly.use_smooth = True
    if mat is not None:
        obj.data.materials.append(mat)
    return obj


def _add_tube_curve(
    name: str,
    points: list[tuple[float, float, float]],
    bevel_depth: float,
    mat: Any,
    taper_end: bool = False,
) -> Any:
    curve_data = bpy.data.curves.new(name=f"{name}_curve", type="CURVE")
    curve_data.dimensions = "3D"
    curve_data.resolution_u = 16
    curve_data.bevel_depth = bevel_depth
    curve_data.bevel_resolution = 6
    curve_data.use_fill_caps = True
    spline = curve_data.splines.new("NURBS")
    spline.points.add(len(points) - 1)
    for idx, (x, y, z) in enumerate(points):
        spline.points[idx].co = (x, y, z, 1.0)
        if taper_end:
            frac = idx / max(1, len(points) - 1)
            spline.points[idx].radius = max(0.18, 1.0 - 0.78 * frac)
    spline.use_endpoint_u = True
    obj = bpy.data.objects.new(name, curve_data)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(mat)
    return obj


# ── Procedural Pixar-Style Materials ────────────────────────────────────────


def _make_fur_material(
    name: str,
    base_rgb: tuple[float, float, float],
    stripe_rgb: tuple[float, float, float] | None = None,
    roughness: float = 0.62,
    bump_strength: float = 0.22,
) -> Any:
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    _set_input(bsdf, ("Roughness",), roughness)
    _set_input(bsdf, ("Sheen Weight", "Sheen"), 0.72)
    _set_input(bsdf, ("Sheen Roughness",), 0.45)
    _set_input(bsdf, ("Subsurface Weight", "Subsurface"), 0.16)
    _set_input(bsdf, ("Subsurface Scale",), 0.03)
    _set_input(bsdf, ("Specular IOR Level", "Specular"), 0.35)

    tex_coord = nt.nodes.new("ShaderNodeTexCoord")

    # Fine velvety micro-fur bump texture
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 95.0
    noise.inputs["Detail"].default_value = 8.0
    noise.inputs["Roughness"].default_value = 0.65
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = bump_strength
    bump.inputs["Distance"].default_value = 0.012
    nt.links.new(tex_coord.outputs["Object"], noise.inputs["Vector"])
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    if stripe_rgb is not None:
        # Subtle tabby fur wave stripes + fur grain variation
        wave = nt.nodes.new("ShaderNodeTexWave")
        wave.wave_type = "BANDS"
        wave.bands_direction = "Z"
        wave.inputs["Scale"].default_value = 4.2
        wave.inputs["Distortion"].default_value = 4.5
        wave.inputs["Detail"].default_value = 3.0
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].position = 0.56
        ramp.color_ramp.elements[0].color = (*base_rgb, 1.0)
        ramp.color_ramp.elements[1].position = 0.86
        ramp.color_ramp.elements[1].color = (*stripe_rgb, 1.0)
        nt.links.new(tex_coord.outputs["Object"], wave.inputs["Vector"])
        nt.links.new(wave.outputs["Fac"], ramp.inputs["Fac"])
        nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        bsdf.inputs["Base Color"].default_value = (*base_rgb, 1.0)

    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def _make_simple_material(
    name: str,
    rgb: tuple[float, float, float],
    *,
    roughness: float = 0.35,
    metallic: float = 0.0,
    coat: float = 0.0,
    emission: float = 0.0,
    subsurface: float = 0.0,
) -> Any:
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    _set_input(bsdf, ("Roughness",), roughness)
    _set_input(bsdf, ("Metallic",), metallic)
    _set_input(bsdf, ("Coat Weight", "Clearcoat"), coat)
    _set_input(bsdf, ("Subsurface Weight", "Subsurface"), subsurface)
    if emission > 0.0:
        _set_input(bsdf, ("Emission Color", "Emission"), (*rgb, 1.0))
        _set_input(bsdf, ("Emission Strength",), emission)
    return mat


def _make_emerald_iris_material() -> Any:
    """Rich glossy Pixar radial emerald-green cat iris with luminous inner glow."""
    mat = bpy.data.materials.new(name="Kiko_Emerald_Iris")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    _set_input(bsdf, ("Roughness",), 0.05)
    _set_input(bsdf, ("Coat Weight", "Clearcoat"), 1.0)
    _set_input(bsdf, ("Coat Roughness", "Clearcoat Roughness"), 0.02)
    _set_input(bsdf, ("Emission Strength",), 0.28)

    tc = nt.nodes.new("ShaderNodeTexCoord")
    grad = nt.nodes.new("ShaderNodeTexGradient")
    grad.gradient_type = "SPHERICAL"
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.15
    ramp.color_ramp.elements[0].color = (0.015, 0.24, 0.09, 1.0)
    e_mid = ramp.color_ramp.elements.new(0.55)
    e_mid.color = (0.06, 0.72, 0.26, 1.0)
    ramp.color_ramp.elements[2].position = 0.90
    ramp.color_ramp.elements[2].color = (0.28, 0.95, 0.42, 1.0)

    nt.links.new(tc.outputs["Object"], grad.inputs["Vector"])
    nt.links.new(grad.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    if "Emission Color" in bsdf.inputs:
        nt.links.new(ramp.outputs["Color"], bsdf.inputs["Emission Color"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def _setup_cycles(width: int, height: int, samples: int, transparent: bool) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.025
    scene.cycles.max_bounces = 6
    scene.cycles.diffuse_bounces = 3
    scene.cycles.glossy_bounces = 3
    scene.cycles.use_denoising = True
    with contextlib.suppress(Exception):
        scene.cycles.denoiser = "OPENIMAGEDENOISE"
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA" if transparent else "RGB"
    scene.render.film_transparent = transparent
    scene.view_settings.view_transform = "AgX" if "AgX" in [
        i.identifier for i in scene.view_settings.bl_rna.properties["view_transform"].enum_items
    ] else "Filmic"
    scene.view_settings.exposure = 0.15


def _add_area_light(
    name: str,
    loc: tuple[float, float, float],
    target: tuple[float, float, float],
    energy: float,
    color: tuple[float, float, float],
    size: float = 2.0,
) -> Any:
    ldata = bpy.data.lights.new(name=name, type="AREA")
    ldata.energy = energy
    ldata.color = color
    ldata.size = size
    lobj = bpy.data.objects.new(name=name, object_data=ldata)
    bpy.context.collection.objects.link(lobj)
    lobj.location = Vector(loc)
    direction = Vector(target) - Vector(loc)
    lobj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    return lobj


def _add_camera(
    loc: tuple[float, float, float],
    target: tuple[float, float, float],
    lens: float = 52.0,
    use_dof: bool = False,
    fstop: float = 2.2,
    focus_dist: float | None = None,
) -> Any:
    cdata = bpy.data.cameras.new(name="Camera")
    cdata.lens = lens
    if use_dof:
        cdata.dof.use_dof = True
        cdata.dof.focus_distance = (
            focus_dist if focus_dist is not None else (Vector(target) - Vector(loc)).length
        )
        cdata.dof.aperture_fstop = fstop
    cobj = bpy.data.objects.new(name="Camera", object_data=cdata)
    bpy.context.collection.objects.link(cobj)
    cobj.location = Vector(loc)
    direction = Vector(target) - Vector(loc)
    cobj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = cobj
    return cobj


# ── Sculpted 3D Pixar-Style Kiko Character Builder ──────────────────────────


def _build_kiko_scene(emotion: str, mouth: str, blink: bool, samples: int) -> None:
    _clear_scene()
    _setup_cycles(640, 640, samples=samples, transparent=True)

    # Subtle ambient studio world
    world = bpy.data.worlds.new("StudioWorld")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.08, 0.09, 0.13, 1.0)
        bg.inputs["Strength"].default_value = 0.55

    # Materials
    m_ginger = _make_fur_material("Kiko_Ginger", (0.86, 0.34, 0.065), (0.44, 0.13, 0.02), roughness=0.60)
    m_ginger_dark = _make_fur_material("Kiko_GingerDark", (0.48, 0.15, 0.025), None, roughness=0.62)
    m_cream = _make_fur_material("Kiko_Cream", (0.97, 0.87, 0.70), None, roughness=0.64, bump_strength=0.18)
    m_ear_pink = _make_simple_material("Kiko_EarPink", (0.94, 0.44, 0.45), roughness=0.50, subsurface=0.25)
    m_nose = _make_simple_material("Kiko_Nose", (0.96, 0.36, 0.44), roughness=0.18, coat=0.85, subsurface=0.2)
    m_sclera = _make_simple_material("Kiko_Sclera", (0.98, 0.98, 0.99), roughness=0.05, coat=1.0)
    m_iris = _make_emerald_iris_material()
    m_pupil = _make_simple_material("Kiko_Pupil", (0.01, 0.012, 0.018), roughness=0.02, coat=1.0)
    m_lash = _make_simple_material("Kiko_Lash", (0.09, 0.04, 0.02), roughness=0.45)
    m_mouth_in = _make_simple_material("Kiko_MouthIn", (0.28, 0.04, 0.06), roughness=0.35)
    m_tongue = _make_simple_material("Kiko_Tongue", (0.92, 0.34, 0.42), roughness=0.22, coat=0.6, subsurface=0.2)
    m_tooth = _make_simple_material("Kiko_Tooth", (0.98, 0.97, 0.94), roughness=0.15, coat=0.5)
    m_whisker = _make_simple_material("Kiko_Whisker", (0.96, 0.93, 0.88), roughness=0.28)
    m_hp_body = _make_simple_material("HP_Body", (0.06, 0.07, 0.095), roughness=0.26, metallic=0.35, coat=0.5)
    m_hp_pad = _make_simple_material("HP_Pad", (0.035, 0.04, 0.055), roughness=0.55)
    m_hp_led = _make_simple_material("HP_LED", (0.04, 0.88, 0.94), roughness=0.10, emission=6.5)
    m_mic_foam = _make_simple_material("Mic_Foam", (0.045, 0.05, 0.065), roughness=0.85)
    m_tear = _make_simple_material("Tear", (0.55, 0.85, 1.0), roughness=0.04, coat=1.0, emission=0.25)
    m_paw_pad = _make_simple_material("Paw_Pad", (0.94, 0.48, 0.52), roughness=0.32, subsurface=0.25)

    # Emotion-driven posture tweaks
    head_z = 0.22
    if emotion == "horrified":
        head_z = 0.16
    elif emotion == "hyped":
        head_z = 0.26

    # 1. Torso & Shoulders (waist-up broadcast silhouette)
    _add_sphere("Torso", (0.0, 0.04, -1.06), (0.86, 0.62, 0.98), mat=m_ginger, subsurf=2)
    _add_sphere("Chest_Cream", (0.0, -0.34, -0.96), (0.48, 0.36, 0.66), mat=m_cream, subsurf=2)
    _add_sphere("Neck_Fluff", (0.0, -0.22, -0.46), (0.44, 0.36, 0.26), mat=m_cream, subsurf=2)
    # Cozy shoulders
    _add_sphere("Shoulder_L", (-0.66, 0.05, -0.92), (0.38, 0.40, 0.44), (0.0, 0.0, 0.25), mat=m_ginger)
    _add_sphere("Shoulder_R", (0.66, 0.05, -0.92), (0.38, 0.40, 0.44), (0.0, 0.0, -0.25), mat=m_ginger)

    # 2. Sculpted Pixar Cat Head + Cheek Fluffs + Muzzle
    _add_sphere("Head_Main", (0.0, 0.0, head_z), (0.88, 0.78, 0.76), mat=m_ginger, subsurf=3)
    # Soft cream lower face / cheeks / chin
    _add_sphere("Face_LowerCream", (0.0, -0.36, head_z - 0.20), (0.64, 0.50, 0.44), mat=m_cream, subsurf=2)
    # Puffed cheek cushions
    for side in (-1.0, 1.0):
        _add_sphere(
            f"Cheek_{side}", (side * 0.56, -0.18, head_z - 0.14), (0.40, 0.36, 0.30), mat=m_ginger, subsurf=2
        )
        # Sculpted fluffy side cheek tufts (signature Pixar stylized fur!)
        _add_cone(
            f"TuftA_{side}",
            (side * 0.86, -0.02, head_z - 0.12),
            (0.14, 0.12, 0.22),
            (0.0, side * 1.38, 0.0),
            mat=m_ginger,
        )
        _add_cone(
            f"TuftB_{side}",
            (side * 0.80, -0.04, head_z - 0.26),
            (0.11, 0.10, 0.18),
            (0.0, side * 1.62, 0.0),
            mat=m_cream,
        )
        # Cute puffed muzzle whisker pads (:3)
        _add_sphere(
            f"MuzzlePad_{side}",
            (side * 0.17, -0.71, head_z - 0.22),
            (0.21, 0.18, 0.16),
            mat=m_cream,
            subsurf=2,
        )
    # Tabby forehead M-stripes (subtle sculpted fur crests)
    for ox, ry in ((-0.16, -0.22), (0.0, 0.0), (0.16, 0.22)):
        _add_sphere(
            f"Stripe_{ox}",
            (ox, -0.58, head_z + 0.44),
            (0.055, 0.12, 0.16),
            (0.45, ry, 0.0),
            mat=m_ginger_dark,
            subsurf=1,
        )

    # 3. Expressive 3D Cat Ears
    ear_tilt = 0.0
    if emotion in ("sad", "horrified"):
        ear_tilt = 0.30
    elif emotion == "laughing":
        ear_tilt = 0.14
    elif emotion in ("shocked", "hyped"):
        ear_tilt = -0.10

    for side in (-1.0, 1.0):
        ex = side * 0.54
        ez = head_z + 0.72 - ear_tilt * 0.28
        _add_cone(
            f"EarOuter_{side}",
            (ex, 0.04, ez),
            (0.28, 0.20, 0.36),
            (0.12, side * (0.34 + ear_tilt), side * 0.15),
            mat=m_ginger,
            r1=1.0,
            r2=0.16,
            subsurf=2,
        )
        _add_cone(
            f"EarInner_{side}",
            (ex * 0.97, -0.06, ez - 0.03),
            (0.18, 0.11, 0.26),
            (0.12, side * (0.34 + ear_tilt), side * 0.15),
            mat=m_ear_pink,
            r1=0.95,
            r2=0.14,
            subsurf=2,
        )

    # 4. Glossy Ray-Traced Pixar Emerald Eyes + Dynamic Eyelids & Brows
    eye_y = -0.63
    eye_z = head_z + 0.06
    closed_eyes = blink or emotion == "laughing"

    for side in (-1.0, 1.0):
        ex = side * 0.31
        if closed_eyes:
            # Joyful curved closed eye arc (laughing) or soft natural blink lid
            arc_h = 0.09 if emotion == "laughing" else -0.04
            pts = [
                (ex - 0.17, eye_y - 0.10, eye_z - 0.01),
                (ex - 0.07, eye_y - 0.15, eye_z + arc_h),
                (ex + 0.07, eye_y - 0.15, eye_z + arc_h),
                (ex + 0.17, eye_y - 0.10, eye_z - 0.01),
            ]
            _add_tube_curve(f"ClosedEye_{side}", pts, 0.028, m_lash)
            if emotion == "laughing":
                # Raised happy cheek puff beneath laughing eye
                _add_sphere(
                    f"LaughCheek_{side}", (ex, eye_y - 0.02, eye_z - 0.13), (0.19, 0.13, 0.10), mat=m_cream
                )
        else:
            eye_scale = 1.14 if emotion in ("shocked", "horrified") else (1.06 if emotion == "hyped" else 1.0)
            # Eyeball sclera
            _add_sphere(
                f"EyeWhite_{side}",
                (ex, eye_y, eye_z),
                (0.20 * eye_scale, 0.16 * eye_scale, 0.22 * eye_scale),
                mat=m_sclera,
                subsurf=2,
            )
            # Vibrant emerald iris
            _add_sphere(
                f"EyeIris_{side}",
                (ex * 0.98, eye_y - 0.105 * eye_scale, eye_z),
                (0.135 * eye_scale, 0.075, 0.155 * eye_scale),
                mat=m_iris,
                subsurf=2,
            )
            # Dynamic pupil size (tiny startled pupils on shocked/horrified, big glossy pupils on hyped/sad)
            p_sc = 0.56 if emotion in ("shocked", "horrified") else (1.12 if emotion in ("hyped", "sad") else 0.90)
            _add_sphere(
                f"EyePupil_{side}",
                (ex * 0.98, eye_y - 0.155 * eye_scale, eye_z),
                (0.068 * p_sc, 0.04, 0.102 * p_sc),
                mat=m_pupil,
                subsurf=2,
            )
            # Crisp dual Pixar specular catchlight orbs
            _add_sphere(
                f"CatchlightBig_{side}",
                (ex - 0.048, eye_y - 0.178 * eye_scale, eye_z + 0.055),
                (0.032, 0.016, 0.032),
                mat=_make_simple_material(f"CL1_{side}", (1.0, 1.0, 1.0), emission=3.5),
                subsurf=1,
            )
            _add_sphere(
                f"CatchlightSmall_{side}",
                (ex + 0.042, eye_y - 0.176 * eye_scale, eye_z - 0.048),
                (0.016, 0.012, 0.016),
                mat=_make_simple_material(f"CL2_{side}", (0.95, 1.0, 1.0), emission=2.5),
                subsurf=1,
            )
            # Upper lash liner
            _add_tube_curve(
                f"Lash_{side}",
                [
                    (ex - 0.19, eye_y - 0.07, eye_z + 0.10),
                    (ex, eye_y - 0.13, eye_z + 0.21 * eye_scale),
                    (ex + 0.19, eye_y - 0.07, eye_z + 0.10),
                ],
                0.018,
                m_lash,
            )
            # Half-lidded smug or facepalm squint upper eyelid
            if emotion == "smug":
                lid_z = eye_z + (0.14 if side < 0 else 0.10)
                _add_sphere(
                    f"SmugLid_{side}", (ex, eye_y - 0.04, lid_z), (0.21, 0.18, 0.11), mat=m_ginger
                )
            elif emotion == "facepalm":
                _add_sphere(
                    f"SquintLid_{side}", (ex, eye_y - 0.04, eye_z + 0.10), (0.21, 0.18, 0.12), mat=m_ginger
                )
            elif emotion == "sad":
                # Glistening 3D teardrops
                _add_sphere(
                    f"Tear_{side}",
                    (ex + side * 0.10, eye_y - 0.13, eye_z - 0.17),
                    (0.042, 0.032, 0.068),
                    mat=m_tear,
                )

        # Expressive 3D eyebrows
        bx = side * 0.31
        bz = eye_z + 0.28
        tilt = 0.0
        if emotion == "smug":
            tilt = 0.26 if side < 0 else -0.14
            bz += 0.05 if side < 0 else -0.02
        elif emotion in ("sad", "horrified"):
            tilt = side * 0.32
            bz += 0.02
        elif emotion == "shocked":
            bz += 0.11
        elif emotion == "hyped":
            tilt = -side * 0.18
            bz += 0.06
        elif emotion == "facepalm":
            tilt = -side * 0.22
            bz -= 0.02

        _add_sphere(
            f"Brow_{side}",
            (bx, eye_y - 0.08, bz),
            (0.16, 0.055, 0.042),
            (0.0, tilt, 0.0),
            mat=m_ginger_dark,
            subsurf=2,
        )

    # 5. Cute 3D Pink Nose & Tapered Whiskers
    _add_sphere("Nose", (0.0, -0.82, head_z - 0.12), (0.11, 0.085, 0.075), mat=m_nose, subsurf=2)
    for side in (-1.0, 1.0):
        for w_idx, dz in enumerate((0.03, -0.03, -0.09)):
            _add_tube_curve(
                f"Whisker_{side}_{w_idx}",
                [
                    (side * 0.34, -0.66, head_z - 0.20 + dz),
                    (side * 0.66, -0.58, head_z - 0.18 + dz * 1.4),
                    (side * 0.96, -0.44, head_z - 0.20 + dz * 2.0),
                ],
                0.011,
                m_whisker,
                taper_end=True,
            )

    # 6. Animated 3D Mouth (closed / half / open)
    mz = head_z - 0.34
    if mouth == "closed":
        smile_up = 0.06 if emotion in ("neutral", "smug", "laughing", "hyped") else -0.04
        _add_tube_curve(
            "MouthClosed",
            [
                (-0.18, -0.73, mz + smile_up),
                (-0.07, -0.78, mz - 0.01),
                (0.0, -0.80, mz + 0.02),
                (0.07, -0.78, mz - 0.01),
                (0.18, -0.73, mz + smile_up),
            ],
            0.019,
            m_lash,
        )
    elif mouth == "half":
        _add_sphere("MouthHalf", (0.0, -0.75, mz - 0.02), (0.13, 0.08, 0.10), mat=m_mouth_in, subsurf=2)
        _add_sphere("TongueHalf", (0.0, -0.78, mz - 0.07), (0.085, 0.05, 0.045), mat=m_tongue, subsurf=2)
    else:
        open_w = 0.22 if emotion in ("laughing", "shocked", "horrified", "hyped") else 0.18
        open_h = 0.20 if emotion in ("shocked", "horrified") else 0.17
        _add_sphere("MouthOpen", (0.0, -0.74, mz - 0.05), (open_w, 0.10, open_h), mat=m_mouth_in, subsurf=2)
        _add_sphere("TongueOpen", (0.0, -0.78, mz - 0.13), (open_w * 0.68, 0.065, 0.075), mat=m_tongue, subsurf=2)
        # Cute tiny upper cat fangs on wide open mouth
        for side in (-1.0, 1.0):
            _add_cone(
                f"Fang_{side}",
                (side * 0.09, -0.80, mz + 0.06),
                (0.026, 0.022, 0.042),
                (math.pi, 0.0, 0.0),
                mat=m_tooth,
                subsurf=1,
            )

    # 7. Sleek 3D Charcoal Studio Headphones + Glowing Teal LED Rings + Boom Mic
    # Curved top headband arching between ears
    hb_pts = [
        (-0.88, -0.02, head_z + 0.08),
        (-0.82, -0.01, head_z + 0.56),
        (-0.46, 0.0, head_z + 0.88),
        (0.0, 0.0, head_z + 0.96),
        (0.46, 0.0, head_z + 0.88),
        (0.82, -0.01, head_z + 0.56),
        (0.88, -0.02, head_z + 0.08),
    ]
    _add_tube_curve("Headband", hb_pts, 0.078, m_hp_body)
    for side in (-1.0, 1.0):
        # Plush inner earcup pad
        _add_cylinder(
            f"HPPad_{side}",
            (side * 0.86, -0.02, head_z - 0.02),
            (0.28, 0.24, 0.09),
            (0.0, math.pi / 2.0, 0.0),
            mat=m_hp_pad,
        )
        # Matte-charcoal sculpted outer earcup dome
        _add_sphere(
            f"HPCup_{side}",
            (side * 0.97, -0.02, head_z - 0.02),
            (0.16, 0.25, 0.30),
            mat=m_hp_body,
            subsurf=2,
        )
        # Glowing cyan-teal LED accent ring on earcup
        _add_torus(
            f"HPLed_{side}",
            (side * 1.07, -0.02, head_z - 0.02),
            (0.0, math.pi / 2.0, 0.0),
            major_r=0.17,
            minor_r=0.022,
            scale=(1.12, 0.92, 1.0),
            mat=m_hp_led,
        )

    # Articulated broadcast boom microphone from left earcup to cheek
    mic_pts = [
        (-0.98, -0.08, head_z - 0.16),
        (-0.92, -0.45, head_z - 0.32),
        (-0.62, -0.78, head_z - 0.39),
        (-0.36, -0.88, head_z - 0.37),
    ]
    _add_tube_curve("BoomArm", mic_pts, 0.024, m_hp_body)
    _add_sphere("MicWindscreen", (-0.33, -0.90, head_z - 0.36), (0.095, 0.075, 0.085), mat=m_mic_foam, subsurf=2)
    _add_torus(
        "MicLedRing",
        (-0.38, -0.88, head_z - 0.37),
        (0.0, 1.35, 0.35),
        major_r=0.048,
        minor_r=0.011,
        mat=m_hp_led,
    )

    # 8. Real 3D Fluffy Orange Cat Paw on Facepalm (furry cat paw + cute pink toe beans, NEVER a human hand!)
    if emotion == "facepalm":
        # Fluffy ginger forearm rising from right shoulder up to forehead
        arm_pts = [
            (0.68, -0.25, -0.86),
            (0.60, -0.66, -0.32),
            (0.30, -0.88, head_z + 0.05),
            (0.10, -0.92, head_z + 0.24),
        ]
        _add_tube_curve("FacepalmArm", arm_pts, 0.165, m_ginger)
        # Cute rounded furry cream-tipped cat paw resting on forehead/brow
        _add_sphere("CatPawPalm", (0.04, -0.94, head_z + 0.24), (0.26, 0.14, 0.24), (0.2, 0.0, -0.25), mat=m_cream)
        # 4 rounded furry feline toes curving gently at top of paw
        for t_idx, (tx, tz) in enumerate(((-0.15, 0.18), (-0.05, 0.22), (0.07, 0.21), (0.17, 0.15))):
            _add_sphere(
                f"CatToe_{t_idx}",
                (0.04 + tx, -0.95, head_z + 0.24 + tz),
                (0.075, 0.075, 0.085),
                mat=m_ginger,
                subsurf=2,
            )
            # Subtle pink toe bean peek on inner edge
            _add_sphere(
                f"ToeBean_{t_idx}",
                (0.04 + tx, -0.91, head_z + 0.23 + tz),
                (0.038, 0.04, 0.042),
                mat=m_paw_pad,
                subsurf=1,
            )

    # 9. Cinematic 3-Point Pixar Lighting + Camera
    _add_area_light("KeyLight", (-2.6, -3.4, 2.5), (0.0, 0.0, 0.0), energy=360.0, color=(1.0, 0.93, 0.84), size=2.4)
    _add_area_light("FillLight", (3.0, -2.8, 1.0), (0.0, 0.0, 0.0), energy=150.0, color=(0.76, 0.88, 1.0), size=3.0)
    _add_area_light("RimLight", (0.8, 2.6, 2.8), (0.0, 0.0, 0.2), energy=320.0, color=(1.0, 0.82, 0.62), size=2.2)
    _add_area_light("EyeCatch", (-0.5, -3.6, 0.9), (0.0, -0.6, 0.25), energy=55.0, color=(1.0, 1.0, 1.0), size=0.8)

    _add_camera((0.0, -4.65, -0.02), (0.0, 0.0, -0.04), lens=55.0, use_dof=False)


# ── 4 Lush 3D Pixar-Style Broadcast Booth Environments (1080x900) ───────────


def _build_booth_scene(genre: str, samples: int) -> None:
    _clear_scene()
    _setup_cycles(1080, 900, samples=samples, transparent=False)

    palettes: dict[str, dict[str, Any]] = {
        "comedy": {
            "wall": (0.14, 0.075, 0.042),
            "panel": (0.20, 0.11, 0.06),
            "accent": (1.0, 0.72, 0.16),
            "accent2": (1.0, 0.35, 0.18),
            "desk": (0.18, 0.10, 0.055),
            "world": (0.06, 0.035, 0.018),
        },
        "horror_comedy": {
            "wall": (0.065, 0.04, 0.13),
            "panel": (0.10, 0.06, 0.19),
            "accent": (0.06, 0.92, 0.76),
            "accent2": (0.62, 0.22, 1.0),
            "desk": (0.08, 0.05, 0.14),
            "world": (0.03, 0.018, 0.06),
        },
        "action_scifi": {
            "wall": (0.038, 0.058, 0.105),
            "panel": (0.06, 0.09, 0.16),
            "accent": (0.06, 0.86, 1.0),
            "accent2": (0.16, 0.44, 1.0),
            "desk": (0.05, 0.075, 0.12),
            "world": (0.015, 0.028, 0.055),
        },
        "emotional_drama": {
            "wall": (0.045, 0.068, 0.14),
            "panel": (0.075, 0.10, 0.20),
            "accent": (1.0, 0.66, 0.24),
            "accent2": (0.38, 0.56, 1.0),
            "desk": (0.07, 0.085, 0.15),
            "world": (0.022, 0.032, 0.07),
        },
    }
    pal = palettes.get(genre, palettes["comedy"])

    world = bpy.data.worlds.new("BoothWorld")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (*pal["world"], 1.0)
        bg.inputs["Strength"].default_value = 0.8

    m_wall = _make_simple_material("Wall", pal["wall"], roughness=0.72)
    m_panel = _make_simple_material("Panel", pal["panel"], roughness=0.58, metallic=0.12)
    m_accent = _make_simple_material("Accent", pal["accent"], roughness=0.15, emission=7.5)
    m_accent2 = _make_simple_material("Accent2", pal["accent2"], roughness=0.15, emission=5.5)
    m_desk = _make_simple_material("Desk", pal["desk"], roughness=0.35, coat=0.45)

    # Back studio wall + angled side walls
    _add_cube("BackWall", (0.0, 3.6, 0.4), (5.2, 0.15, 3.2), mat=m_wall, bevel=0.0)
    _add_cube("LeftWall", (-4.2, 1.2, 0.4), (0.15, 3.5, 3.2), (0.0, 0.0, -0.25), mat=m_wall, bevel=0.0)
    _add_cube("RightWall", (4.2, 1.2, 0.4), (0.15, 3.5, 3.2), (0.0, 0.0, 0.25), mat=m_wall, bevel=0.0)

    # 3D Beveled acoustic studio tiles across back wall
    for r in range(4):
        for c in range(7):
            tx = -2.8 + c * 0.94
            tz = -0.6 + r * 0.86
            _add_cube(f"Tile_{r}_{c}", (tx, 3.42, tz), (0.41, 0.08, 0.37), mat=m_panel, bevel=0.04)

    # Glowing recessed neon strips framing the acoustic studio wall
    _add_cube("NeonTop", (0.0, 3.34, 1.72), (3.6, 0.03, 0.045), mat=m_accent, bevel=0.01)
    _add_cube("NeonLeft", (-3.15, 3.30, 0.35), (0.045, 0.03, 1.6), mat=m_accent2, bevel=0.01)
    _add_cube("NeonRight", (3.15, 3.30, 0.35), (0.045, 0.03, 1.6), mat=m_accent, bevel=0.01)

    # Background studio shelf on the right side (leaves left-center x=0.39 open for Kiko!)
    _add_cube("RightShelf", (1.85, 2.55, -0.45), (1.35, 0.36, 0.07), mat=m_desk, bevel=0.02)
    # Foreground broadcast desk surface (bottom of booth)
    _add_cube("FrontDesk", (0.0, 0.2, -1.65), (4.6, 1.1, 0.22), mat=m_desk, bevel=0.04)

    # Genre-specific 3D Pixar props & bokeh lighting
    if genre == "comedy":
        # Cozy draped golden fairy-light bokeh orbs across upper studio
        for idx in range(11):
            fx = -2.8 + idx * 0.56
            fz = 1.48 - 0.28 * math.sin(idx * 0.31)
            _add_sphere(f"FairyBulb_{idx}", (fx, 3.10, fz), (0.075, 0.075, 0.075), mat=m_accent, subsurf=1)
        # Iconic 3D red-and-cream cinema popcorn bucket on right shelf
        m_red = _make_simple_material("PopRed", (0.85, 0.12, 0.14), roughness=0.32, coat=0.4)
        m_pop = _make_simple_material("PopKernel", (1.0, 0.88, 0.45), roughness=0.55)
        _add_cone("PopBucket", (1.65, 2.48, -0.04), (0.28, 0.28, 0.34), (math.pi, 0.0, 0.0), mat=m_red, r1=1.0, r2=0.72)
        for k_i, (kx, ky, kz) in enumerate(((-0.12, -0.05, 0.34), (0.0, -0.08, 0.39), (0.12, -0.04, 0.35), (-0.05, 0.06, 0.36), (0.08, 0.05, 0.33))):
            _add_sphere(f"PopK_{k_i}", (1.65 + kx, 2.48 + ky, -0.04 + kz), (0.11, 0.11, 0.10), mat=m_pop)
        # Warm glowing tabletop studio lamp
        _add_sphere("WarmLamp", (2.45, 2.50, 0.12), (0.24, 0.24, 0.24), mat=m_accent)

    elif genre == "horror_comedy":
        # Glowing whimsical bioluminescent forest mushroom lamp on right shelf!
        m_stem = _make_simple_material("MushStem", (0.82, 0.90, 0.86), roughness=0.45, subsurface=0.25)
        _add_cylinder("MushStem", (1.85, 2.48, -0.10), (0.10, 0.10, 0.28), mat=m_stem)
        _add_sphere("MushCap", (1.85, 2.48, 0.22), (0.36, 0.36, 0.20), mat=m_accent)
        _add_cylinder("MushStem2", (2.32, 2.55, -0.22), (0.065, 0.065, 0.16), mat=m_stem)
        _add_sphere("MushCap2", (2.32, 2.55, -0.02), (0.22, 0.22, 0.13), mat=m_accent2)
        # Floating glowing spore bokeh orbs
        for idx, (ox, oz) in enumerate(((-2.2, 1.1), (-1.4, 1.5), (0.9, 1.35), (2.5, 1.25), (1.3, 0.85))):
            _add_sphere(f"Spore_{idx}", (ox, 2.95, oz), (0.085, 0.085, 0.085), mat=m_accent if idx % 2 == 0 else m_accent2)

    elif genre == "action_scifi":
        # Glowing futuristic 3D LED audio visualizer equalizer bars on right shelf
        heights = (0.35, 0.72, 0.50, 0.95, 0.65, 0.84, 0.42, 0.58)
        for idx, bh in enumerate(heights):
            bx = 1.05 + idx * 0.22
            _add_cube(
                f"EQBar_{idx}",
                (bx, 2.55, -0.36 + bh * 0.5),
                (0.065, 0.065, bh * 0.5),
                mat=m_accent if idx % 2 == 0 else m_accent2,
                bevel=0.01,
            )
        # Sleek holographic glowing halo ring on back wall
        _add_torus("SciFiRing", (1.85, 3.18, 0.85), (math.pi / 2.0, 0.0, 0.0), major_r=0.72, minor_r=0.032, mat=m_accent)

    else:  # emotional_drama
        # Warm golden hanging Edison bokeh globes + soft twilight drapes
        for idx, (ex, ez) in enumerate(((-2.4, 1.35), (-1.3, 1.55), (1.2, 1.45), (2.0, 1.15), (2.6, 1.40))):
            _add_cylinder(f"Cord_{idx}", (ex, 2.95, ez + 0.8), (0.012, 0.012, 0.75), mat=m_desk)
            _add_sphere(f"Edison_{idx}", (ex, 2.95, ez), (0.12, 0.12, 0.15), mat=m_accent)
        # Soft vertical velvet acoustic curtain flutes on right
        for c_i in range(5):
            _add_cylinder(f"Velvet_{c_i}", (2.4 + c_i * 0.28, 3.20, 0.4), (0.16, 0.16, 2.4), mat=m_panel)

    # Studio lighting + shallow depth-of-field camera for authentic Pixar bokeh!
    _add_area_light("BoothKey", (-2.2, -1.2, 2.2), (0.5, 2.8, 0.2), energy=260.0, color=pal["accent"], size=2.8)
    _add_area_light("BoothFill", (2.6, -1.0, 1.6), (-0.5, 3.0, 0.2), energy=220.0, color=pal["accent2"], size=3.0)
    _add_area_light("BoothBack", (0.0, 2.4, 2.4), (0.0, 3.4, 0.0), energy=180.0, color=(0.95, 0.92, 1.0), size=3.5)

    # Focus in foreground where Kiko sits so background wall & props melt into creamy Pixar bokeh
    _add_camera((0.0, -2.55, 0.05), (0.0, 2.5, 0.08), lens=36.0, use_dof=True, fstop=1.6, focus_dist=2.5)


def main() -> None:
    args = _parse_args()
    sprites_dir = Path(args.sprites_dir)
    booths_dir = Path(args.booths_dir)
    sprites_dir.mkdir(parents=True, exist_ok=True)
    booths_dir.mkdir(parents=True, exist_ok=True)

    # 1. Render all 24 emotion x mouth sprites + blink.png (25 total, 640x640 transparent RGBA)
    for emotion in EMOTIONS:
        for mouth in MOUTHS:
            out_path = sprites_dir / f"{emotion}_{mouth}.png"
            print(f"[blender-3d] Rendering sprite {out_path.name}...", flush=True)
            _build_kiko_scene(emotion, mouth, blink=False, samples=args.samples)
            bpy.context.scene.render.filepath = str(out_path.resolve())
            bpy.ops.render.render(write_still=True)

    blink_path = sprites_dir / "blink.png"
    print(f"[blender-3d] Rendering sprite {blink_path.name}...", flush=True)
    _build_kiko_scene("neutral", "closed", blink=True, samples=args.samples)
    bpy.context.scene.render.filepath = str(blink_path.resolve())
    bpy.ops.render.render(write_still=True)

    # 2. Render 4 lush 3D Pixar-style genre broadcast booths (1080x900 RGB)
    for genre in GENRES:
        b_path = booths_dir / f"{genre}.png"
        print(f"[blender-3d] Rendering booth {b_path.name}...", flush=True)
        _build_booth_scene(genre, samples=max(24, args.samples))
        bpy.context.scene.render.filepath = str(b_path.resolve())
        bpy.ops.render.render(write_still=True)

    print("[blender-3d] Completed 25 Kiko 3D sprites + 4 3D genre booths!", flush=True)


if __name__ == "__main__":
    main()
