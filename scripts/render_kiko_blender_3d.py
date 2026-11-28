#!/usr/bin/env python3
"""Headless Blender 3D Cycles Pixar-style renderer for Kiko 2.0 (32 RGBA sprites + 4 dark 1080x700 booths).

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
    "disgusted",
    "suspicious",
)
MOUTHS: tuple[str, ...] = ("closed", "half", "open")
GENRES: tuple[str, ...] = ("comedy", "horror_comedy", "action_scifi", "emotional_drama")


def _parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--sprites-dir", default="assets/host/sprites")
    parser.add_argument("--booths-dir", default="assets/host/booths")
    parser.add_argument("--samples", type=int, default=26)
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


def _attach_parent(obj: Any, parent: Any | None) -> Any:
    if parent is not None:
        obj.parent = parent
    return obj


def _add_sphere(
    name: str,
    loc: tuple[float, float, float],
    scale: tuple[float, float, float],
    rot: tuple[float, float, float] = (0.0, 0.0, 0.0),
    mat: Any = None,
    subsurf: int = 2,
    segments: int = 32,
    rings: int = 16,
    parent: Any | None = None,
) -> Any:
    bpy.ops.mesh.primitive_uv_sphere_add(segments=segments, ring_count=rings, radius=1.0, location=loc)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = scale
    obj.rotation_euler = Euler(rot, "XYZ")
    _smooth_and_subsurf(obj, subsurf)
    if mat is not None:
        obj.data.materials.append(mat)
    return _attach_parent(obj, parent)


def _add_cylinder(
    name: str,
    loc: tuple[float, float, float],
    scale: tuple[float, float, float],
    rot: tuple[float, float, float] = (0.0, 0.0, 0.0),
    mat: Any = None,
    vertices: int = 32,
    subsurf: int = 1,
    parent: Any | None = None,
) -> Any:
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=1.0, depth=2.0, location=loc)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = scale
    obj.rotation_euler = Euler(rot, "XYZ")
    _smooth_and_subsurf(obj, subsurf)
    if mat is not None:
        obj.data.materials.append(mat)
    return _attach_parent(obj, parent)


def _add_cone(
    name: str,
    loc: tuple[float, float, float],
    scale: tuple[float, float, float],
    rot: tuple[float, float, float] = (0.0, 0.0, 0.0),
    mat: Any = None,
    r1: float = 1.0,
    r2: float = 0.14,
    subsurf: int = 2,
    parent: Any | None = None,
) -> Any:
    bpy.ops.mesh.primitive_cone_add(vertices=28, radius1=r1, radius2=r2, depth=2.0, location=loc)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = scale
    obj.rotation_euler = Euler(rot, "XYZ")
    _smooth_and_subsurf(obj, subsurf)
    if mat is not None:
        obj.data.materials.append(mat)
    return _attach_parent(obj, parent)


def _add_torus(
    name: str,
    loc: tuple[float, float, float],
    rot: tuple[float, float, float],
    major_r: float,
    minor_r: float,
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    mat: Any = None,
    parent: Any | None = None,
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
    return _attach_parent(obj, parent)


def _add_cube(
    name: str,
    loc: tuple[float, float, float],
    scale: tuple[float, float, float],
    rot: tuple[float, float, float] = (0.0, 0.0, 0.0),
    mat: Any = None,
    bevel: float = 0.04,
    parent: Any | None = None,
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
    return _attach_parent(obj, parent)


def _add_tube_curve(
    name: str,
    points: list[tuple[float, float, float]],
    bevel_depth: float,
    mat: Any,
    taper_end: bool = False,
    parent: Any | None = None,
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
    return _attach_parent(obj, parent)


# ── Procedural Pixar-Style Materials ────────────────────────────────────────


def _make_fur_material(
    name: str,
    base_rgb: tuple[float, float, float],
    stripe_rgb: tuple[float, float, float] | None = None,
    roughness: float = 0.58,
    bump_strength: float = 0.24,
) -> Any:
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    _set_input(bsdf, ("Roughness",), roughness)
    _set_input(bsdf, ("Sheen Weight", "Sheen"), 0.78)
    _set_input(bsdf, ("Sheen Roughness",), 0.42)
    _set_input(bsdf, ("Subsurface Weight", "Subsurface"), 0.16)
    _set_input(bsdf, ("Subsurface Scale",), 0.03)
    _set_input(bsdf, ("Specular IOR Level", "Specular"), 0.36)

    tex_coord = nt.nodes.new("ShaderNodeTexCoord")

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
        wave = nt.nodes.new("ShaderNodeTexWave")
        wave.wave_type = "BANDS"
        wave.bands_direction = "Z"
        wave.inputs["Scale"].default_value = 4.4
        wave.inputs["Distortion"].default_value = 4.6
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


def _make_fabric_material(
    name: str,
    base_rgb: tuple[float, float, float],
    roughness: float = 0.82,
    bump_strength: float = 0.16,
) -> Any:
    """Matte charcoal streetwear cotton-fleece hoodie shader with soft micro-weave sheen."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (*base_rgb, 1.0)
    _set_input(bsdf, ("Roughness",), roughness)
    _set_input(bsdf, ("Sheen Weight", "Sheen"), 0.55)
    _set_input(bsdf, ("Sheen Roughness",), 0.65)
    _set_input(bsdf, ("Specular IOR Level", "Specular"), 0.18)

    tc = nt.nodes.new("ShaderNodeTexCoord")
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 140.0
    noise.inputs["Detail"].default_value = 6.0
    noise.inputs["Roughness"].default_value = 0.72
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = bump_strength
    bump.inputs["Distance"].default_value = 0.008
    nt.links.new(tc.outputs["Object"], noise.inputs["Vector"])
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
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
    _set_input(bsdf, ("Emission Strength",), 0.30)

    tc = nt.nodes.new("ShaderNodeTexCoord")
    grad = nt.nodes.new("ShaderNodeTexGradient")
    grad.gradient_type = "SPHERICAL"
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.15
    ramp.color_ramp.elements[0].color = (0.015, 0.24, 0.09, 1.0)
    e_mid = ramp.color_ramp.elements.new(0.55)
    e_mid.color = (0.06, 0.74, 0.28, 1.0)
    ramp.color_ramp.elements[2].position = 0.90
    ramp.color_ramp.elements[2].color = (0.30, 0.96, 0.44, 1.0)

    nt.links.new(tc.outputs["Object"], grad.inputs["Vector"])
    nt.links.new(grad.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    if "Emission Color" in bsdf.inputs:
        nt.links.new(ramp.outputs["Color"], bsdf.inputs["Emission Color"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def _setup_cycles(
    width: int,
    height: int,
    samples: int,
    transparent: bool,
    *,
    exposure: float = 0.15,
) -> None:
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
    scene.view_settings.view_transform = (
        "AgX"
        if "AgX" in [i.identifier for i in scene.view_settings.bl_rna.properties["view_transform"].enum_items]
        else "Filmic"
    )
    scene.view_settings.exposure = exposure


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


# ── Expressive Feline Paw & Hoodie Arm Rig Helper ───────────────────────────


def _add_cat_arm_and_paw(
    prefix: str,
    sleeve_pts: list[tuple[float, float, float]],
    paw_pos: tuple[float, float, float],
    *,
    paw_rot: tuple[float, float, float] = (0.0, 0.0, 0.0),
    paw_scale: tuple[float, float, float] = (0.21, 0.14, 0.21),
    show_beans_front: bool = False,
    fist_mode: bool = False,
    point_mode: bool = False,
    m_hoodie: Any,
    m_hoodie_rib: Any,
    m_ginger: Any,
    m_cream: Any,
    m_paw_pad: Any,
) -> None:
    """Build a dark hoodie sleeve ending in a fluffy ginger-and-cream feline cat paw."""
    # Upper hoodie sleeve along first 82% of arm path
    _add_tube_curve(f"{prefix}_Sleeve", sleeve_pts, 0.165, m_hoodie)
    wrist = sleeve_pts[-1]
    # Dark ribbed hoodie cuff near wrist
    _add_sphere(f"{prefix}_Cuff", wrist, (0.175, 0.155, 0.155), mat=m_hoodie_rib, subsurf=1)
    # Fluffy ginger cat wrist bridge into paw
    mid = (
        0.45 * wrist[0] + 0.55 * paw_pos[0],
        0.45 * wrist[1] + 0.55 * paw_pos[1],
        0.45 * wrist[2] + 0.55 * paw_pos[2],
    )
    _add_sphere(f"{prefix}_WristFur", mid, (0.145, 0.135, 0.145), mat=m_ginger, subsurf=2)

    # Paw root empty for clean oriented toes
    paw_rig = bpy.data.objects.new(f"{prefix}_PawRig", None)
    bpy.context.collection.objects.link(paw_rig)
    paw_rig.location = Vector(paw_pos)
    paw_rig.rotation_euler = Euler(paw_rot, "XYZ")

    # Rounded furry cream cat paw mitt
    _add_sphere(f"{prefix}_Palm", (0.0, 0.0, 0.0), paw_scale, mat=m_cream, subsurf=2, parent=paw_rig)

    if fist_mode:
        # Compact curled excited cat fist toes
        for idx, (tx, tz) in enumerate(((-0.11, 0.12), (-0.04, 0.14), (0.04, 0.14), (0.11, 0.11))):
            _add_sphere(
                f"{prefix}_FistToe_{idx}",
                (tx, -0.07, tz),
                (0.068, 0.072, 0.068),
                mat=m_ginger,
                subsurf=2,
                parent=paw_rig,
            )
    elif point_mode:
        # Cute plush curled paw with one soft rounded ginger feline toe extended slyly
        for idx, (tx, tz) in enumerate(((-0.10, 0.08), (0.08, 0.09), (0.12, 0.02))):
            _add_sphere(
                f"{prefix}_CurlToe_{idx}",
                (tx, -0.05, tz),
                (0.062, 0.065, 0.065),
                mat=m_ginger,
                subsurf=2,
                parent=paw_rig,
            )
        _add_sphere(
            f"{prefix}_PointToe",
            (-0.02, -0.14, 0.21),
            (0.068, 0.075, 0.125),
            (0.28, 0.0, 0.0),
            mat=m_cream,
            subsurf=2,
            parent=paw_rig,
        )
    else:
        # 4 soft rounded feline cat toes around top of paw
        for idx, (tx, tz) in enumerate(((-0.13, 0.15), (-0.045, 0.19), (0.045, 0.19), (0.13, 0.15))):
            _add_sphere(
                f"{prefix}_Toe_{idx}",
                (tx, -0.02, tz),
                (0.068, 0.068, 0.076),
                mat=m_ginger,
                subsurf=2,
                parent=paw_rig,
            )
            if show_beans_front:
                _add_sphere(
                    f"{prefix}_Bean_{idx}",
                    (tx, -0.082, tz - 0.01),
                    (0.036, 0.032, 0.042),
                    mat=m_paw_pad,
                    subsurf=1,
                    parent=paw_rig,
                )
        if show_beans_front:
            # Cute central pink metacarpal paw pad
            _add_sphere(
                f"{prefix}_MainPad",
                (0.0, -0.118, -0.02),
                (0.095, 0.042, 0.080),
                mat=m_paw_pad,
                subsurf=2,
                parent=paw_rig,
            )


# ── Sculpted 3D Pixar-Style Kiko 2.0 Character Builder ──────────────────────


def _build_kiko_scene(emotion: str, mouth: str, blink: bool, samples: int) -> None:
    _clear_scene()
    _setup_cycles(640, 640, samples=samples, transparent=True, exposure=0.18)

    world = bpy.data.worlds.new("StudioWorld")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.07, 0.085, 0.13, 1.0)
        bg.inputs["Strength"].default_value = 0.58

    # Kiko 2.0 Pixar Materials
    m_ginger = _make_fur_material("Kiko_Ginger", (0.89, 0.35, 0.06), (0.46, 0.13, 0.018), roughness=0.58)
    m_ginger_dark = _make_fur_material("Kiko_GingerDark", (0.48, 0.14, 0.022), None, roughness=0.60)
    m_cream = _make_fur_material("Kiko_Cream", (0.98, 0.88, 0.72), None, roughness=0.62, bump_strength=0.18)
    m_hoodie = _make_fabric_material("Kiko_Hoodie", (0.065, 0.072, 0.092), roughness=0.82)
    m_hoodie_rib = _make_fabric_material("Kiko_HoodieRib", (0.045, 0.050, 0.065), roughness=0.76)
    m_cord = _make_simple_material("Kiko_Cord", (0.86, 0.90, 0.96), roughness=0.55)
    m_aglet_cyan = _make_simple_material("Kiko_AgletCyan", (0.05, 0.90, 0.96), roughness=0.16, metallic=0.6, emission=2.6)
    m_aglet_gold = _make_simple_material("Kiko_AgletGold", (1.0, 0.78, 0.18), roughness=0.18, metallic=0.85, emission=1.2)
    m_ear_pink = _make_simple_material("Kiko_EarPink", (0.95, 0.44, 0.46), roughness=0.48, subsurface=0.25)
    m_nose = _make_simple_material("Kiko_Nose", (0.96, 0.36, 0.44), roughness=0.16, coat=0.85, subsurface=0.2)
    m_sclera = _make_simple_material("Kiko_Sclera", (0.98, 0.98, 0.99), roughness=0.05, coat=1.0)
    m_iris = _make_emerald_iris_material()
    m_pupil = _make_simple_material("Kiko_Pupil", (0.01, 0.012, 0.018), roughness=0.02, coat=1.0)
    m_lash = _make_simple_material("Kiko_Lash", (0.08, 0.035, 0.018), roughness=0.42)
    m_mouth_in = _make_simple_material("Kiko_MouthIn", (0.28, 0.04, 0.06), roughness=0.35)
    m_tongue = _make_simple_material("Kiko_Tongue", (0.94, 0.34, 0.44), roughness=0.20, coat=0.65, subsurface=0.22)
    m_tooth = _make_simple_material("Kiko_Tooth", (0.98, 0.97, 0.94), roughness=0.15, coat=0.5)
    m_whisker = _make_simple_material("Kiko_Whisker", (0.97, 0.94, 0.89), roughness=0.26)
    m_hp_body = _make_simple_material("HP_Body", (0.048, 0.055, 0.078), roughness=0.24, metallic=0.38, coat=0.55)
    m_hp_pad = _make_simple_material("HP_Pad", (0.030, 0.035, 0.048), roughness=0.55)
    m_hp_led = _make_simple_material("HP_LED", (0.04, 0.90, 0.96), roughness=0.08, emission=7.0)
    m_mic_foam = _make_simple_material("Mic_Foam", (0.042, 0.046, 0.060), roughness=0.85)
    m_tear = _make_simple_material("Tear", (0.55, 0.88, 1.0), roughness=0.03, coat=1.0, emission=0.45)
    m_paw_pad = _make_simple_material("Paw_Pad", (0.95, 0.46, 0.52), roughness=0.30, subsurface=0.28)

    # Emotion-driven head rig position & tilt (chest/shoulders stay strictly anchored!)
    head_z = 0.14
    head_x = 0.0
    head_y = 0.0
    head_rot = (0.02, 0.03, 0.0)
    if emotion == "laughing":
        head_z = 0.16
        head_rot = (-0.12, -0.14, 0.06)
    elif emotion == "shocked":
        head_z = 0.20
        head_y = 0.06
        head_rot = (-0.08, 0.0, 0.0)
    elif emotion == "horrified":
        head_z = 0.08
        head_y = 0.08
        head_rot = (-0.04, 0.07, -0.04)
    elif emotion == "smug":
        head_z = 0.15
        head_rot = (0.04, 0.14, -0.10)
    elif emotion == "hyped":
        head_z = 0.20
        head_y = -0.06
        head_rot = (0.08, -0.05, 0.0)
    elif emotion == "sad":
        head_z = 0.09
        head_rot = (0.14, -0.10, 0.0)
    elif emotion == "facepalm":
        head_z = 0.10
        head_rot = (0.11, -0.08, 0.04)
    elif emotion == "disgusted":
        head_z = 0.13
        head_x = -0.08
        head_y = 0.08
        head_rot = (-0.07, 0.13, 0.14)
    elif emotion == "suspicious":
        head_z = 0.14
        head_y = -0.06
        head_rot = (0.10, -0.15, 0.09)

    # 1. Fixed Chest/Shoulder Torso in Sleek Dark Charcoal Streetwear Hoodie
    _add_sphere("Hoodie_Torso", (0.0, 0.04, -1.24), (0.94, 0.66, 1.06), mat=m_hoodie, subsurf=2)
    _add_sphere("Hoodie_ChestFront", (0.0, -0.30, -1.26), (0.72, 0.42, 0.78), mat=m_hoodie, subsurf=2)
    # Sculpted cozy streetwear hood cowl resting around neck/back shoulders
    _add_torus(
        "Hoodie_Cowl",
        (0.0, 0.02, -0.62),
        (0.22, 0.0, 0.0),
        major_r=0.58,
        minor_r=0.17,
        scale=(1.12, 0.96, 0.88),
        mat=m_hoodie_rib,
    )
    # Subtle cream chest fur peeking at hoodie neckline V
    _add_sphere("Neck_CreamFur", (0.0, -0.26, -0.56), (0.36, 0.28, 0.22), mat=m_cream, subsurf=2)
    # Hoodie shoulders (anchored identically across all 10 poses)
    shoulder_z = -0.96 if emotion != "horrified" else -0.84
    _add_sphere("Shoulder_L", (-0.72, 0.04, shoulder_z), (0.40, 0.42, 0.44), (0.0, 0.0, 0.22), mat=m_hoodie)
    _add_sphere("Shoulder_R", (0.72, 0.04, shoulder_z), (0.40, 0.42, 0.44), (0.0, 0.0, -0.22), mat=m_hoodie)
    # Stylish cyan & gold-tipped hoodie drawcords hanging neatly on chest
    _add_tube_curve(
        "Cord_L",
        [(-0.20, -0.56, -0.64), (-0.23, -0.68, -0.96), (-0.21, -0.71, -1.28)],
        0.024,
        m_cord,
    )
    _add_tube_curve(
        "Cord_R",
        [(0.20, -0.56, -0.64), (0.23, -0.68, -0.96), (0.21, -0.71, -1.28)],
        0.024,
        m_cord,
    )
    _add_cylinder("Aglet_L", (-0.21, -0.71, -1.34), (0.032, 0.032, 0.068), mat=m_aglet_cyan)
    _add_cylinder("Aglet_R", (0.21, -0.71, -1.34), (0.032, 0.032, 0.068), mat=m_aglet_gold)

    # 2. Parented HeadRig Empty (all head features built in local coords around 0,0,0)
    head_rig = bpy.data.objects.new("HeadRig", None)
    bpy.context.collection.objects.link(head_rig)
    head_rig.location = Vector((head_x, head_y, head_z))
    head_rig.rotation_euler = Euler(head_rot, "XYZ")

    _add_sphere("Head_Main", (0.0, 0.0, 0.0), (0.86, 0.76, 0.75), mat=m_ginger, subsurf=3, parent=head_rig)
    _add_sphere("Face_LowerCream", (0.0, -0.35, -0.20), (0.62, 0.48, 0.43), mat=m_cream, subsurf=2, parent=head_rig)

    for side in (-1.0, 1.0):
        _add_sphere(
            f"Cheek_{side}", (side * 0.55, -0.18, -0.14), (0.39, 0.35, 0.29), mat=m_ginger, subsurf=2, parent=head_rig
        )
        _add_cone(
            f"TuftA_{side}",
            (side * 0.84, -0.02, -0.11),
            (0.14, 0.12, 0.22),
            (0.0, side * 1.38, 0.0),
            mat=m_ginger,
            parent=head_rig,
        )
        _add_cone(
            f"TuftB_{side}",
            (side * 0.78, -0.04, -0.25),
            (0.11, 0.10, 0.18),
            (0.0, side * 1.62, 0.0),
            mat=m_cream,
            parent=head_rig,
        )
        _add_sphere(
            f"MuzzlePad_{side}",
            (side * 0.165, -0.70, -0.21),
            (0.20, 0.175, 0.155),
            mat=m_cream,
            subsurf=2,
            parent=head_rig,
        )

    # Signature Kiko 2.0 messy ginger hair tuft sprouting between large cat ears
    _add_cone(
        "HairTuft_Center",
        (0.0, -0.20, 0.80),
        (0.14, 0.12, 0.24),
        (0.32, 0.16, 0.0),
        mat=m_ginger,
        r1=1.0,
        r2=0.10,
        parent=head_rig,
    )
    _add_cone(
        "HairTuft_Left",
        (-0.14, -0.18, 0.76),
        (0.11, 0.10, 0.20),
        (0.28, -0.38, 0.0),
        mat=m_ginger_dark,
        r1=1.0,
        r2=0.10,
        parent=head_rig,
    )
    _add_cone(
        "HairTuft_Right",
        (0.14, -0.16, 0.75),
        (0.10, 0.09, 0.19),
        (0.24, 0.44, 0.0),
        mat=m_ginger,
        r1=1.0,
        r2=0.10,
        parent=head_rig,
    )

    # Tabby forehead M-stripes
    for ox, ry in ((-0.16, -0.22), (0.0, 0.0), (0.16, 0.22)):
        _add_sphere(
            f"Stripe_{ox}",
            (ox, -0.57, 0.42),
            (0.052, 0.11, 0.15),
            (0.45, ry, 0.0),
            mat=m_ginger_dark,
            subsurf=1,
            parent=head_rig,
        )

    # 3. Expressive 3D Cat Ears
    ear_tilt = 0.0
    if emotion in ("sad", "horrified"):
        ear_tilt = 0.34
    elif emotion == "disgusted":
        ear_tilt = 0.24
    elif emotion == "laughing":
        ear_tilt = 0.16
    elif emotion == "suspicious":
        ear_tilt = 0.10
    elif emotion in ("shocked", "hyped"):
        ear_tilt = -0.14

    for side in (-1.0, 1.0):
        ex = side * 0.54
        ez = 0.72 - ear_tilt * 0.28
        _add_cone(
            f"EarOuter_{side}",
            (ex, 0.04, ez),
            (0.29, 0.20, 0.38),
            (0.12, side * (0.34 + ear_tilt), side * 0.15),
            mat=m_ginger,
            r1=1.0,
            r2=0.15,
            subsurf=2,
            parent=head_rig,
        )
        _add_cone(
            f"EarInner_{side}",
            (ex * 0.97, -0.06, ez - 0.03),
            (0.18, 0.11, 0.27),
            (0.12, side * (0.34 + ear_tilt), side * 0.15),
            mat=m_ear_pink,
            r1=0.95,
            r2=0.13,
            subsurf=2,
            parent=head_rig,
        )

    # 4. Glossy Pixar Emerald Eyes + Dynamic Eyelids, Pupils, Side-Eye & Brows
    eye_y = -0.62
    eye_z = 0.06
    closed_eyes = blink or emotion == "laughing"
    gaze_shift_x = 0.045 if emotion in ("smug", "suspicious") else (-0.03 if emotion == "disgusted" else 0.0)

    for side in (-1.0, 1.0):
        ex = side * 0.31
        if closed_eyes:
            arc_h = 0.095 if emotion == "laughing" else -0.04
            pts = [
                (ex - 0.17, eye_y - 0.10, eye_z - 0.01),
                (ex - 0.07, eye_y - 0.15, eye_z + arc_h),
                (ex + 0.07, eye_y - 0.15, eye_z + arc_h),
                (ex + 0.17, eye_y - 0.10, eye_z - 0.01),
            ]
            _add_tube_curve(f"ClosedEye_{side}", pts, 0.030, m_lash, parent=head_rig)
            if emotion == "laughing":
                _add_sphere(
                    f"LaughCheek_{side}", (ex, eye_y - 0.02, eye_z - 0.13), (0.19, 0.13, 0.10), mat=m_cream, parent=head_rig
                )
                # Sparkling happy laugh tear droplet at outer eye corner
                _add_sphere(
                    f"HappyTear_{side}",
                    (ex + side * 0.18, eye_y - 0.11, eye_z - 0.04),
                    (0.045, 0.035, 0.060),
                    (0.0, side * 0.45, 0.0),
                    mat=m_tear,
                    parent=head_rig,
                )
        else:
            eye_scale = 1.15 if emotion in ("shocked", "horrified") else (1.07 if emotion == "hyped" else 1.0)
            _add_sphere(
                f"EyeWhite_{side}",
                (ex, eye_y, eye_z),
                (0.20 * eye_scale, 0.16 * eye_scale, 0.22 * eye_scale),
                mat=m_sclera,
                subsurf=2,
                parent=head_rig,
            )
            _add_sphere(
                f"EyeIris_{side}",
                (ex * 0.98 + gaze_shift_x, eye_y - 0.105 * eye_scale, eye_z),
                (0.135 * eye_scale, 0.075, 0.155 * eye_scale),
                mat=m_iris,
                subsurf=2,
                parent=head_rig,
            )
            p_sc = (
                0.54
                if emotion in ("shocked", "horrified")
                else (0.72 if emotion == "suspicious" else (1.12 if emotion in ("hyped", "sad") else 0.90))
            )
            _add_sphere(
                f"EyePupil_{side}",
                (ex * 0.98 + gaze_shift_x, eye_y - 0.155 * eye_scale, eye_z),
                (0.068 * p_sc, 0.04, 0.102 * p_sc),
                mat=m_pupil,
                subsurf=2,
                parent=head_rig,
            )
            _add_sphere(
                f"CatchlightBig_{side}",
                (ex - 0.048 + gaze_shift_x * 0.5, eye_y - 0.178 * eye_scale, eye_z + 0.055),
                (0.034, 0.016, 0.034),
                mat=_make_simple_material(f"CL1_{side}", (1.0, 1.0, 1.0), emission=3.8),
                subsurf=1,
                parent=head_rig,
            )
            _add_sphere(
                f"CatchlightSmall_{side}",
                (ex + 0.042 + gaze_shift_x * 0.5, eye_y - 0.176 * eye_scale, eye_z - 0.048),
                (0.017, 0.012, 0.017),
                mat=_make_simple_material(f"CL2_{side}", (0.95, 1.0, 1.0), emission=2.6),
                subsurf=1,
                parent=head_rig,
            )
            _add_tube_curve(
                f"Lash_{side}",
                [
                    (ex - 0.19, eye_y - 0.07, eye_z + 0.10),
                    (ex, eye_y - 0.13, eye_z + 0.21 * eye_scale),
                    (ex + 0.19, eye_y - 0.07, eye_z + 0.10),
                ],
                0.018,
                m_lash,
                parent=head_rig,
            )

            # Emotion-specific eyelids & tears
            if emotion == "smug":
                lid_z = eye_z + (0.14 if side < 0 else 0.09)
                _add_sphere(
                    f"SmugLid_{side}", (ex, eye_y - 0.04, lid_z), (0.21, 0.18, 0.115), mat=m_ginger, parent=head_rig
                )
            elif emotion == "suspicious":
                # Narrowed detective eye slits (both top AND bottom lids pinched!)
                _add_sphere(
                    f"SuspTopLid_{side}",
                    (ex, eye_y - 0.05, eye_z + 0.115),
                    (0.22, 0.18, 0.105),
                    (0.0, -0.10 * side, 0.0),
                    mat=m_ginger,
                    parent=head_rig,
                )
                _add_sphere(
                    f"SuspBotLid_{side}",
                    (ex, eye_y - 0.05, eye_z - 0.125),
                    (0.21, 0.17, 0.090),
                    mat=m_cream,
                    parent=head_rig,
                )
            elif emotion == "disgusted":
                # Asymmetric grossed-out squint + raised lower cringe lid
                lid_drop = 0.085 if side > 0 else 0.125
                _add_sphere(
                    f"DisgTopLid_{side}",
                    (ex, eye_y - 0.05, eye_z + lid_drop),
                    (0.22, 0.18, 0.115),
                    (0.0, -0.22 * side, 0.0),
                    mat=m_ginger,
                    parent=head_rig,
                )
                _add_sphere(
                    f"DisgBotLid_{side}",
                    (ex, eye_y - 0.05, eye_z - 0.115),
                    (0.21, 0.17, 0.095),
                    mat=m_cream,
                    parent=head_rig,
                )
            elif emotion == "facepalm":
                _add_sphere(
                    f"SquintLid_{side}", (ex, eye_y - 0.04, eye_z + 0.095), (0.21, 0.18, 0.12), mat=m_ginger, parent=head_rig
                )
            elif emotion == "sad":
                _add_sphere(
                    f"Tear_{side}",
                    (ex + side * 0.10, eye_y - 0.13, eye_z - 0.17),
                    (0.044, 0.034, 0.072),
                    mat=m_tear,
                    parent=head_rig,
                )

        # Expressive 3D eyebrows
        bx = side * 0.31
        bz = eye_z + 0.28
        tilt = 0.0
        if emotion == "smug":
            tilt = 0.28 if side < 0 else -0.16
            bz += 0.06 if side < 0 else -0.02
        elif emotion == "suspicious":
            tilt = -0.30 if side < 0 else 0.18
            bz += -0.05 if side < 0 else 0.04
        elif emotion == "disgusted":
            tilt = -side * 0.34
            bz -= 0.04
        elif emotion in ("sad", "horrified"):
            tilt = side * 0.34
            bz += 0.02
        elif emotion == "shocked":
            bz += 0.12
        elif emotion == "hyped":
            tilt = -side * 0.20
            bz += 0.07
        elif emotion == "facepalm":
            tilt = -side * 0.24
            bz -= 0.02

        _add_sphere(
            f"Brow_{side}",
            (bx, eye_y - 0.08, bz),
            (0.165, 0.056, 0.044),
            (0.0, tilt, 0.0),
            mat=m_ginger_dark,
            subsurf=2,
            parent=head_rig,
        )

    # 5. Cute 3D Pink Nose (scrunched higher on disgusted!) & Tapered Whiskers
    nose_z = -0.085 if emotion == "disgusted" else -0.12
    _add_sphere("Nose", (0.0, -0.81, nose_z), (0.11, 0.085, 0.075), mat=m_nose, subsurf=2, parent=head_rig)
    if emotion == "disgusted":
        # Subtle scrunched nose-bridge wrinkle ridges
        _add_sphere("NoseWrinkle", (0.0, -0.77, -0.02), (0.11, 0.05, 0.032), mat=m_ginger_dark, subsurf=1, parent=head_rig)
    for side in (-1.0, 1.0):
        for w_idx, dz in enumerate((0.03, -0.03, -0.09)):
            _add_tube_curve(
                f"Whisker_{side}_{w_idx}",
                [
                    (side * 0.33, -0.65, -0.20 + dz),
                    (side * 0.65, -0.57, -0.18 + dz * 1.4),
                    (side * 0.94, -0.43, -0.20 + dz * 2.0),
                ],
                0.011,
                m_whisker,
                taper_end=True,
                parent=head_rig,
            )

    # 6. Animated 3D Mouth (closed / half / open)
    mz = -0.34
    if mouth == "closed":
        if emotion == "disgusted":
            # Wavy "ew" disgusted grimace curve
            _add_tube_curve(
                "MouthClosed",
                [
                    (-0.20, -0.72, mz - 0.04),
                    (-0.08, -0.78, mz + 0.03),
                    (0.02, -0.79, mz - 0.02),
                    (0.11, -0.77, mz + 0.02),
                    (0.20, -0.72, mz - 0.06),
                ],
                0.021,
                m_lash,
                parent=head_rig,
            )
        elif emotion == "suspicious":
            # Tight skeptical flat sideways line
            _add_tube_curve(
                "MouthClosed",
                [
                    (-0.16, -0.74, mz - 0.01),
                    (0.0, -0.79, mz - 0.01),
                    (0.17, -0.73, mz + 0.02),
                ],
                0.020,
                m_lash,
                parent=head_rig,
            )
        elif emotion == "smug":
            # Asymmetric lopsided smirk
            _add_tube_curve(
                "MouthClosed",
                [
                    (-0.14, -0.75, mz - 0.01),
                    (0.0, -0.79, mz + 0.01),
                    (0.13, -0.76, mz + 0.05),
                    (0.22, -0.70, mz + 0.11),
                ],
                0.020,
                m_lash,
                parent=head_rig,
            )
        else:
            smile_up = 0.065 if emotion in ("neutral", "laughing", "hyped") else -0.045
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
                parent=head_rig,
            )
    elif mouth == "half":
        _add_sphere("MouthHalf", (0.0, -0.75, mz - 0.02), (0.13, 0.08, 0.10), mat=m_mouth_in, subsurf=2, parent=head_rig)
        _add_sphere("TongueHalf", (0.0, -0.78, mz - 0.07), (0.085, 0.05, 0.045), mat=m_tongue, subsurf=2, parent=head_rig)
    else:
        open_w = 0.23 if emotion in ("laughing", "shocked", "horrified", "hyped") else 0.18
        open_h = 0.22 if emotion in ("shocked", "horrified") else 0.17
        _add_sphere("MouthOpen", (0.0, -0.74, mz - 0.05), (open_w, 0.10, open_h), mat=m_mouth_in, subsurf=2, parent=head_rig)
        _add_sphere(
            "TongueOpen",
            (0.0, -0.78, mz - 0.13),
            (open_w * 0.68, 0.065, 0.075),
            mat=m_tongue,
            subsurf=2,
            parent=head_rig,
        )
        for side in (-1.0, 1.0):
            _add_cone(
                f"Fang_{side}",
                (side * 0.09, -0.80, mz + 0.06),
                (0.026, 0.022, 0.042),
                (math.pi, 0.0, 0.0),
                mat=m_tooth,
                subsurf=1,
                parent=head_rig,
            )

    # 7. Matte-Black Over-Ear Studio Headphones + Glowing Cyan LED Rings + Boom Mic (parented to HeadRig)
    hb_pts = [
        (-0.88, -0.02, 0.08),
        (-0.82, -0.01, 0.56),
        (-0.46, 0.0, 0.88),
        (0.0, 0.0, 0.96),
        (0.46, 0.0, 0.88),
        (0.82, -0.01, 0.56),
        (0.88, -0.02, 0.08),
    ]
    _add_tube_curve("Headband", hb_pts, 0.078, m_hp_body, parent=head_rig)
    for side in (-1.0, 1.0):
        _add_cylinder(
            f"HPPad_{side}",
            (side * 0.86, -0.02, -0.02),
            (0.28, 0.24, 0.09),
            (0.0, math.pi / 2.0, 0.0),
            mat=m_hp_pad,
            parent=head_rig,
        )
        _add_sphere(
            f"HPCup_{side}",
            (side * 0.97, -0.02, -0.02),
            (0.16, 0.25, 0.30),
            mat=m_hp_body,
            subsurf=2,
            parent=head_rig,
        )
        _add_torus(
            f"HPLed_{side}",
            (side * 1.07, -0.02, -0.02),
            (0.0, math.pi / 2.0, 0.0),
            major_r=0.17,
            minor_r=0.022,
            scale=(1.12, 0.92, 1.0),
            mat=m_hp_led,
            parent=head_rig,
        )

    mic_pts = [
        (-0.98, -0.08, -0.16),
        (-0.92, -0.45, -0.32),
        (-0.62, -0.78, -0.39),
        (-0.36, -0.88, -0.37),
    ]
    _add_tube_curve("BoomArm", mic_pts, 0.024, m_hp_body, parent=head_rig)
    _add_sphere("MicWindscreen", (-0.33, -0.90, -0.36), (0.095, 0.075, 0.085), mat=m_mic_foam, subsurf=2, parent=head_rig)
    _add_torus(
        "MicLedRing",
        (-0.38, -0.88, -0.37),
        (0.0, 1.35, 0.35),
        major_r=0.048,
        minor_r=0.011,
        mat=m_hp_led,
        parent=head_rig,
    )

    # 8. Expressive Two-Paw Feline Acting for ALL 10 Silhouette-Breaking Poses
    arm_mats = {
        "m_hoodie": m_hoodie,
        "m_hoodie_rib": m_hoodie_rib,
        "m_ginger": m_ginger,
        "m_cream": m_cream,
        "m_paw_pad": m_paw_pad,
    }
    if emotion == "neutral":
        _add_cat_arm_and_paw(
            "ArmL",
            [(-0.74, 0.02, -0.98), (-0.86, -0.22, -1.40), (-0.72, -0.48, -1.56)],
            (-0.66, -0.58, -1.52),
            paw_rot=(0.3, 0.2, 0.1),
            **arm_mats,
        )
        _add_cat_arm_and_paw(
            "ArmR",
            [(0.74, 0.02, -0.98), (0.82, -0.30, -1.32), (0.50, -0.64, -1.24)],
            (0.40, -0.74, -1.16),
            paw_rot=(0.15, -0.35, -0.2),
            **arm_mats,
        )
    elif emotion == "laughing":
        # Left paw clutching belly/hoodie chest, right paw waving joyfully out to the side
        _add_cat_arm_and_paw(
            "ArmL",
            [(-0.74, 0.0, -0.98), (-0.72, -0.42, -1.26), (-0.28, -0.78, -1.16)],
            (-0.14, -0.84, -1.12),
            paw_rot=(0.2, 0.55, 0.1),
            **arm_mats,
        )
        _add_cat_arm_and_paw(
            "ArmR",
            [(0.74, 0.0, -0.98), (1.06, -0.32, -0.78), (1.16, -0.56, -0.22)],
            (1.18, -0.62, -0.06),
            paw_rot=(0.0, -0.32, -0.18),
            show_beans_front=True,
            **arm_mats,
        )
    elif emotion == "shocked":
        # BOTH paws flung up beside cheeks in jaw-dropping Home Alone shock!
        _add_cat_arm_and_paw(
            "ArmL",
            [(-0.76, -0.04, -0.96), (-0.98, -0.46, -0.56), (-0.80, -0.78, -0.06)],
            (-0.72, -0.86, 0.06),
            paw_rot=(0.1, 0.25, 0.25),
            show_beans_front=True,
            **arm_mats,
        )
        _add_cat_arm_and_paw(
            "ArmR",
            [(0.76, -0.04, -0.96), (0.98, -0.46, -0.56), (0.80, -0.78, -0.06)],
            (0.72, -0.86, 0.06),
            paw_rot=(0.1, -0.25, -0.25),
            show_beans_front=True,
            **arm_mats,
        )
    elif emotion == "horrified":
        # BOTH paws gripping sides of the studio headphones in terror!
        _add_cat_arm_and_paw(
            "ArmL",
            [(-0.76, -0.04, -0.86), (-1.12, -0.32, -0.46), (-1.12, -0.26, 0.02)],
            (-1.10, -0.20, 0.14),
            paw_rot=(0.0, 0.15, 0.72),
            **arm_mats,
        )
        _add_cat_arm_and_paw(
            "ArmR",
            [(0.76, -0.04, -0.86), (1.12, -0.32, -0.46), (1.12, -0.26, 0.02)],
            (1.10, -0.20, 0.14),
            paw_rot=(0.0, -0.15, -0.72),
            **arm_mats,
        )
    elif emotion == "smug":
        # Left arm casually crossed on chest, right paw raised pointing slyly!
        _add_cat_arm_and_paw(
            "ArmL",
            [(-0.74, 0.0, -0.98), (-0.70, -0.42, -1.24), (-0.22, -0.76, -1.12)],
            (-0.10, -0.82, -1.10),
            paw_rot=(0.15, 0.65, 0.0),
            **arm_mats,
        )
        _add_cat_arm_and_paw(
            "ArmR",
            [(0.74, -0.04, -0.96), (0.98, -0.46, -0.72), (0.86, -0.82, -0.36)],
            (0.80, -0.92, -0.24),
            paw_rot=(0.15, -0.25, -0.22),
            point_mode=True,
            **arm_mats,
        )
    elif emotion == "hyped":
        # BOTH fluffy paws pumped HIGH as excited fists above/beside ear height (true 3D pop-out!)
        _add_cat_arm_and_paw(
            "ArmL",
            [(-0.76, -0.04, -0.94), (-1.16, -0.26, -0.38), (-1.18, -0.38, 0.66)],
            (-1.14, -0.44, 0.88),
            paw_rot=(0.0, 0.18, 0.12),
            fist_mode=True,
            **arm_mats,
        )
        _add_cat_arm_and_paw(
            "ArmR",
            [(0.76, -0.04, -0.94), (1.16, -0.26, -0.38), (1.18, -0.38, 0.66)],
            (1.14, -0.44, 0.88),
            paw_rot=(0.0, -0.18, -0.12),
            fist_mode=True,
            **arm_mats,
        )
    elif emotion == "sad":
        # BOTH paws clasped sadly together in front of chest
        _add_cat_arm_and_paw(
            "ArmL",
            [(-0.74, 0.0, -0.98), (-0.66, -0.42, -1.24), (-0.22, -0.78, -1.06)],
            (-0.12, -0.84, -1.02),
            paw_rot=(0.25, 0.40, 0.15),
            **arm_mats,
        )
        _add_cat_arm_and_paw(
            "ArmR",
            [(0.74, 0.0, -0.98), (0.66, -0.42, -1.24), (0.22, -0.78, -1.06)],
            (0.12, -0.84, -1.02),
            paw_rot=(0.25, -0.40, -0.15),
            **arm_mats,
        )
    elif emotion == "facepalm":
        # Right fluffy cat paw slapped onto forehead/brow, left paw out exasperated
        _add_cat_arm_and_paw(
            "ArmR",
            [(0.74, -0.08, -0.94), (0.72, -0.62, -0.38), (0.30, -0.92, 0.14)],
            (0.10, -0.98, 0.30),
            paw_rot=(0.22, 0.18, -0.22),
            paw_scale=(0.24, 0.13, 0.23),
            **arm_mats,
        )
        _add_cat_arm_and_paw(
            "ArmL",
            [(-0.74, -0.02, -0.98), (-1.04, -0.34, -1.04), (-1.12, -0.62, -0.76)],
            (-1.12, -0.70, -0.64),
            paw_rot=(0.2, 0.40, 0.22),
            show_beans_front=True,
            **arm_mats,
        )
    elif emotion == "disgusted":
        # Right paw raised defensively palm-out ("ew, stay back!"), left arm pulled back
        _add_cat_arm_and_paw(
            "ArmR",
            [(0.74, -0.04, -0.96), (0.92, -0.54, -0.70), (0.76, -0.96, -0.20)],
            (0.70, -1.06, -0.04),
            paw_rot=(-0.15, -0.20, -0.22),
            paw_scale=(0.23, 0.13, 0.23),
            show_beans_front=True,
            **arm_mats,
        )
        _add_cat_arm_and_paw(
            "ArmL",
            [(-0.74, 0.04, -0.98), (-0.84, -0.20, -1.36), (-0.64, -0.48, -1.48)],
            (-0.56, -0.58, -1.44),
            paw_rot=(0.2, 0.2, 0.1),
            **arm_mats,
        )
    elif emotion == "suspicious":
        # Right paw raised stroking chin thoughtfully, left arm crossed cleanly
        _add_cat_arm_and_paw(
            "ArmL",
            [(-0.74, 0.0, -0.98), (-0.68, -0.42, -1.24), (-0.22, -0.76, -1.16)],
            (-0.10, -0.82, -1.14),
            paw_rot=(0.15, 0.55, 0.0),
            **arm_mats,
        )
        _add_cat_arm_and_paw(
            "ArmR",
            [(0.74, -0.04, -0.96), (0.76, -0.54, -0.78), (0.34, -0.88, -0.46)],
            (0.18, -0.94, -0.34),
            paw_rot=(0.22, -0.42, -0.18),
            **arm_mats,
        )

    # 9. Cinematic 3-Point Warm Key + Cyan Rim Pixar Lighting + Fixed Anchor Camera
    _add_area_light("KeyLight", (-2.6, -3.5, 2.5), (0.0, 0.0, 0.0), energy=390.0, color=(1.0, 0.92, 0.82), size=2.4)
    _add_area_light("FillLight", (3.0, -2.8, 1.0), (0.0, 0.0, 0.0), energy=155.0, color=(0.74, 0.88, 1.0), size=3.0)
    _add_area_light("RimLight", (0.8, 2.6, 2.8), (0.0, 0.0, 0.1), energy=340.0, color=(0.35, 0.88, 1.0), size=2.2)
    _add_area_light("WarmRimL", (-1.8, 2.4, 2.2), (0.0, 0.0, 0.1), energy=210.0, color=(1.0, 0.78, 0.52), size=2.0)
    _add_area_light("EyeCatch", (-0.5, -3.8, 0.9), (0.0, -0.6, 0.18), energy=62.0, color=(1.0, 1.0, 1.0), size=0.8)

    # Fixed camera at y=-5.15 gives generous headroom for raised hyped fists while anchoring torso identically
    _add_camera((0.0, -5.15, -0.12), (0.0, 0.0, -0.12), lens=52.0, use_dof=False)


# ── 4 Dark Compact Atmospheric Cinema Booths (1080x700 RGB) ─────────────────


def _build_booth_scene(genre: str, samples: int) -> None:
    _clear_scene()
    # Low-key cinema exposure (-0.55) keeps booths dramatically darker than the movie above!
    _setup_cycles(1080, 700, samples=samples, transparent=False, exposure=-0.55)

    palettes: dict[str, dict[str, Any]] = {
        "comedy": {
            "wall": (0.022, 0.017, 0.014),
            "slat": (0.048, 0.032, 0.020),
            "accent": (1.0, 0.62, 0.14),
            "accent2": (0.85, 0.28, 0.12),
            "world": (0.012, 0.008, 0.006),
        },
        "horror_comedy": {
            "wall": (0.014, 0.015, 0.026),
            "slat": (0.026, 0.028, 0.048),
            "accent": (0.04, 0.86, 0.74),
            "accent2": (0.28, 0.18, 0.68),
            "world": (0.006, 0.008, 0.016),
        },
        "action_scifi": {
            "wall": (0.012, 0.018, 0.028),
            "slat": (0.024, 0.034, 0.052),
            "accent": (0.04, 0.84, 1.0),
            "accent2": (0.12, 0.34, 0.88),
            "world": (0.006, 0.010, 0.018),
        },
        "emotional_drama": {
            "wall": (0.014, 0.018, 0.030),
            "slat": (0.028, 0.034, 0.052),
            "accent": (1.0, 0.58, 0.18),
            "accent2": (0.20, 0.34, 0.68),
            "world": (0.007, 0.010, 0.018),
        },
    }
    pal = palettes.get(genre, palettes["comedy"])

    world = bpy.data.worlds.new("BoothWorld")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (*pal["world"], 1.0)
        bg.inputs["Strength"].default_value = 0.35

    m_wall = _make_simple_material("DarkWall", pal["wall"], roughness=0.88)
    m_slat = _make_simple_material("AcousticSlat", pal["slat"], roughness=0.62, metallic=0.08)
    m_trim = _make_simple_material("DarkTrim", (0.022, 0.024, 0.032), roughness=0.45, metallic=0.25)
    m_accent = _make_simple_material("Accent", pal["accent"], roughness=0.12, emission=4.2)
    m_accent2 = _make_simple_material("Accent2", pal["accent2"], roughness=0.14, emission=2.6)

    # Deep dark charcoal soundproof studio walls (NO bright grids, NO floating circles!)
    _add_cube("BackWall", (0.0, 4.0, 0.2), (6.0, 0.15, 3.4), mat=m_wall, bevel=0.0)
    _add_cube("LeftWall", (-4.4, 1.5, 0.2), (0.15, 3.8, 3.4), (0.0, 0.0, -0.22), mat=m_wall, bevel=0.0)
    _add_cube("RightWall", (4.4, 1.5, 0.2), (0.15, 3.8, 3.4), (0.0, 0.0, 0.22), mat=m_wall, bevel=0.0)

    # Architectural vertical matte acoustic slats (sleek modern podcast studio depth, zero checkerboard grid)
    for idx in range(25):
        sx = -3.8 + idx * 0.315
        _add_cube(f"Slat_{idx}", (sx, 3.80, 0.25), (0.085, 0.09, 2.65), mat=m_slat, bevel=0.012)

    # Subtle recessed architectural vertical accent light strip on far-right & far-left edges
    _add_cube("EdgeLightR", (3.25, 3.68, 0.25), (0.032, 0.025, 1.85), mat=m_accent, bevel=0.008)
    _add_cube("EdgeLightL", (-3.55, 3.68, 0.25), (0.024, 0.025, 1.40), mat=m_accent2, bevel=0.008)

    # Tasteful dark floating monitor/console shelf on FAR RIGHT only (keeps left-center x=0.36 clean for Kiko!)
    _add_cube("RightShelf", (2.15, 2.90, -0.32), (1.10, 0.32, 0.055), mat=m_trim, bevel=0.015)

    # Genre-specific restrained dark cinema background practicals (far right, softly blurred via shallow DOF)
    if genre == "comedy":
        # Dim warm tabletop practical cinema lamp on far-right shelf + subtle acoustic bass trap
        _add_cylinder("LampBase", (2.45, 2.86, -0.14), (0.12, 0.12, 0.14), mat=m_trim)
        _add_cylinder("LampShade", (2.45, 2.86, 0.12), (0.19, 0.19, 0.16), mat=m_accent)
        _add_cube("FilmSlate", (1.68, 2.90, -0.02), (0.24, 0.04, 0.20), (0.0, 0.0, -0.22), mat=m_slat, bevel=0.01)
    elif genre == "horror_comedy":
        # Subtle dim teal vertical studio tube light + moody dark acoustic column on far right
        _add_cylinder("TubeBase", (2.35, 2.85, -0.20), (0.14, 0.14, 0.08), mat=m_trim)
        _add_cylinder("TealTube", (2.35, 2.85, 0.36), (0.042, 0.042, 0.52), mat=m_accent)
        _add_cube("DarkMon", (1.62, 2.92, 0.02), (0.30, 0.04, 0.22), (0.0, 0.0, -0.18), mat=m_trim, bevel=0.015)
        _add_cube("DarkMonGlow", (1.61, 2.87, 0.02), (0.26, 0.01, 0.02), (0.0, 0.0, -0.18), mat=m_accent2, bevel=0.004)
    elif genre == "action_scifi":
        # Subtle recessed low-key equalizer bars on far right console
        for idx, bh in enumerate((0.18, 0.38, 0.26, 0.50, 0.34, 0.42)):
            bx = 1.50 + idx * 0.22
            _add_cube(
                f"EQ_{idx}",
                (bx, 2.86, -0.24 + bh * 0.5),
                (0.045, 0.04, bh * 0.5),
                mat=m_accent if idx % 2 == 0 else m_accent2,
                bevel=0.008,
            )
    else:  # emotional_drama
        # Subtle dark draped acoustic velvet columns + one warm tungsten practical filament bulb far right
        for c_i in range(4):
            _add_cylinder(f"Velvet_{c_i}", (2.55 + c_i * 0.26, 3.45, 0.25), (0.15, 0.15, 2.4), mat=m_slat)
        _add_cylinder("PendantCord", (2.15, 2.85, 1.35), (0.010, 0.010, 0.72), mat=m_trim)
        _add_sphere("WarmBulb", (2.15, 2.85, 0.56), (0.095, 0.095, 0.125), mat=m_accent, subsurf=1)

    # Low-key cinema studio lighting (subtle rim wash ONLY, keeps mean luma ~30-55!)
    _add_area_light("BoothRimR", (2.8, 1.4, 1.8), (0.6, 3.6, 0.1), energy=75.0, color=pal["accent"], size=2.4)
    _add_area_light("BoothRimL", (-2.8, 1.6, 1.6), (-0.8, 3.6, 0.1), energy=48.0, color=pal["accent2"], size=2.6)
    _add_area_light("BoothSoftTop", (0.0, 0.8, 2.6), (0.0, 3.2, -0.2), energy=36.0, color=(0.75, 0.82, 0.95), size=3.4)

    _add_camera((0.0, -2.45, 0.06), (0.0, 2.8, 0.06), lens=34.0, use_dof=True, fstop=1.8, focus_dist=2.4)


def main() -> None:
    args = _parse_args()
    sprites_dir = Path(args.sprites_dir)
    booths_dir = Path(args.booths_dir)
    sprites_dir.mkdir(parents=True, exist_ok=True)
    booths_dir.mkdir(parents=True, exist_ok=True)

    # 1. Render all 30 Kiko 2.0 emotion x mouth sprites + neutral_blink.png + blink.png (32 total, 640x640 RGBA)
    for emotion in EMOTIONS:
        for mouth in MOUTHS:
            out_path = sprites_dir / f"{emotion}_{mouth}.png"
            print(f"[blender-3d] Rendering Kiko 2.0 sprite {out_path.name}...", flush=True)
            _build_kiko_scene(emotion, mouth, blink=False, samples=args.samples)
            bpy.context.scene.render.filepath = str(out_path.resolve())
            bpy.ops.render.render(write_still=True)

    neutral_blink_path = sprites_dir / "neutral_blink.png"
    print(f"[blender-3d] Rendering Kiko 2.0 sprite {neutral_blink_path.name}...", flush=True)
    _build_kiko_scene("neutral", "closed", blink=True, samples=args.samples)
    bpy.context.scene.render.filepath = str(neutral_blink_path.resolve())
    bpy.ops.render.render(write_still=True)
    # Also save backward-compatible blink.png copy
    (sprites_dir / "blink.png").write_bytes(neutral_blink_path.read_bytes())

    # 2. Render 4 dark compact atmospheric cinema booths (1080x700 RGB)
    for genre in GENRES:
        b_path = booths_dir / f"{genre}.png"
        print(f"[blender-3d] Rendering dark 1080x700 booth {b_path.name}...", flush=True)
        _build_booth_scene(genre, samples=max(22, args.samples))
        bpy.context.scene.render.filepath = str(b_path.resolve())
        bpy.ops.render.render(write_still=True)

    print("[blender-3d] Completed 32 Kiko 2.0 3D sprites + 4 dark 1080x700 booths!", flush=True)


if __name__ == "__main__":
    main()
