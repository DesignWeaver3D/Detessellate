# -*- coding: utf-8 -*-
# SPDX-License-Identifier: LGPL-2.1-or-later
# SPDX-FileCopyrightText: 2026 DesignWeaver3D
# SPDX-FileNotice: Part of the Detessellate addon.
"""
CrossSectionToAlignedSplines
-----------------------------
Converts a Mesh Workbench "Cross Sections" (CS) object into one Sketch per
profile, each wrapped with a periodic B-spline suitable for a multi-profile
loft (Part or PartDesign).

Problem this solves: profiles from mesh cross-sections are irregular
("lumpy") point loops, not clean geometric shapes. A loft needs every
profile to have the same pole count, matching winding direction, and a
start point that corresponds to the same physical location on every
profile. Without that, lofts twist or fail. This macro builds that
correspondence directly rather than relying on the source wire's native
point order, which is not reliable for any of those three properties.

Scope: builds aligned, loft-ready sketches only. Assembling the loft itself
is left to the user (Part::Loft or PartDesign::AdditiveLoft), since profile
grouping, ruled/smooth, and workbench choice are all use-case-specific and
can't be inferred here.

Known limitation: designed for manifold cross-sections. A profile broken
into multiple wire fragments (e.g. from a hole in the source mesh) is not
detected or repaired; each wire is processed independently. Inspect
generated sketches against the source mesh before lofting.

Usage: select the CS object, then run. Prompts once for spline knot count.
"""

import FreeCAD as App
import FreeCADGui as Gui
import Part
import Draft
from PySide import QtWidgets


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N_KNOTS_DEFAULT = 16                           # prepopulated in the prompt
GLOBAL_REFERENCE_VECTOR = App.Vector(1, 0, 0)  # shared "start direction" for
                                                # all profiles; see find_start_index


# ---------------------------------------------------------------------------
# Stage 1 â CS object -> list of Sketch objects
# ---------------------------------------------------------------------------

def cross_section_to_sketches(cs_obj):
    """
    Build one Sketch per profile from cs_obj.Shape.Wires (one wire per
    profile; a hole in the source mesh at a given slice produces its own
    separate wire rather than a nested loop within one wire).

    Each wire is staged as a temporary document object (Part.show) so
    Draft.make_sketch can consume it, then the temp object is discarded.
    """
    sketches = []

    wires = cs_obj.Shape.Wires
    if not wires:
        raise RuntimeError(
            "cs_obj.Shape.Wires is empty. If a different CS-generating tool "
            "exposes profiles differently (e.g. via SubShapes or a Group "
            "property), update this function to match."
        )

    for i, wire in enumerate(wires):
        temp_wire_obj = Part.show(wire, f"TempWire_{i:03d}")
        temp_wire_name = temp_wire_obj.Name
        App.ActiveDocument.recompute()

        sketch = Draft.make_sketch(temp_wire_obj, autoconstraints=False, delete=True)
        sketch.Label = f"CS_Profile_{i:03d}"
        sketches.append(sketch)

        # delete=True above should already remove the temp object; this is
        # just a safety net in case that behavior changes.
        leftover = App.ActiveDocument.getObject(temp_wire_name)
        if leftover is not None:
            App.ActiveDocument.removeObject(temp_wire_name)

    App.ActiveDocument.recompute()
    return sketches


# ---------------------------------------------------------------------------
# Per-sketch geometry helpers
# ---------------------------------------------------------------------------

def get_ordered_points(sketch):
    """
    Return the sketch's line-geometry vertices in geom index order. Assumes
    geom index order already matches the physical walk order around the
    loop, which holds for wires produced by cross_section_to_sketches().
    """
    points = []
    for geo in sketch.Geometry:
        if geo.TypeId == "Part::GeomLineSegment":
            points.append(App.Vector(geo.StartPoint))
    return points


def enforce_consistent_winding(sketch, points):
    """
    Reverse the point list if needed so every profile winds CW as viewed
    along global +Z. Source wires do not have a reliable native winding
    direction (it varies per profile depending on how the mesh-cutting
    algorithm walked that slice), so this must be checked explicitly rather
    than assumed. Direction is computed in global coordinates (via the
    sketch's own Placement) since that's the frame the loft actually uses;
    checking in local sketch coordinates would be wrong for any sketch with
    a rotated Placement.

    CW is an arbitrary but fixed convention, chosen to match Sketcher's own
    arc-drawing direction. Either direction works for the loft as long as
    it's applied consistently.
    """
    global_pts = [sketch.Placement.multVec(p) for p in points]

    n = len(global_pts)
    signed_area = 0.0
    for i in range(n):
        p0, p1 = global_pts[i], global_pts[(i + 1) % n]
        signed_area += (p0.x * p1.y - p1.x * p0.y)
    signed_area /= 2.0

    if signed_area > 0:
        return list(reversed(points))
    return points


def polygon_centroid_2d(points):
    """
    Area-weighted centroid via the shoelace method, treating the point list
    as closed regardless of whether the source wire actually is (open wires
    from small mesh-slicing artifacts are common and are not treated as an
    error elsewhere in this pipeline; see find_start_index).
    """
    area_sum = 0.0
    cx = 0.0
    cy = 0.0
    n = len(points)

    for i in range(n):
        p0 = points[i]
        p1 = points[(i + 1) % n]
        cross = p0.x * p1.y - p1.x * p0.y
        area_sum += cross
        cx += (p0.x + p1.x) * cross
        cy += (p0.y + p1.y) * cross

    area = area_sum / 2.0
    if abs(area) < 1e-12:
        avg = sum(points, App.Vector(0, 0, 0)) * (1.0 / n)
        return avg, area

    cx /= (6.0 * area)
    cy /= (6.0 * area)
    return App.Vector(cx, cy, points[0].z), area


def find_start_index(sketch, points, centroid, global_reference):
    """
    Pick the point closest to global_reference (as seen from centroid) as
    the start knot for this profile. Every sketch's Placement.Rotation may
    differ (source wires do not share a consistent local axis orientation,
    even when their normals all agree), so global_reference is transformed
    into each sketch's own local space before comparing. This is what makes
    the chosen start point correspond to the same physical location across
    all profiles regardless of each sketch's individual rotation.
    """
    local_reference = sketch.Placement.Rotation.inverted().multVec(global_reference)
    local_reference.z = 0.0
    local_reference.normalize()

    best_index = 0
    best_dot = -1e9

    for i, pt in enumerate(points):
        direction = pt - centroid
        direction.z = 0.0
        if direction.Length < 1e-9:
            continue
        direction.normalize()
        dot = direction.dot(local_reference)
        if dot > best_dot:
            best_dot = dot
            best_index = i

    return best_index


def sample_knots(points, start_index, n_knots):
    """
    Sample n_knots points by walking the ordered list at a fixed index
    step from start_index, wrapping via modulo. Index-based (not
    arc-length-based) sampling: acceptable because mesh-scan segment
    lengths vary randomly rather than regionally, so length differences
    average out over a knot-spacing window. The final wraparound segment
    (last knot back to the first) absorbs whatever remainder n_geoms is not
    evenly divisible by n_knots, and so is somewhat longer than the others;
    this is a structural property of index-stepping, not a bug.
    """
    n = len(points)
    step = max(1, n // n_knots)

    knot_points = []
    for i in range(n_knots):
        idx = (start_index + i * step) % n
        knot_points.append(points[idx])

    return knot_points


def set_all_construction(sketch):
    """Convert all line geometry to construction, leaving only the spline visible."""
    for i, geo in enumerate(sketch.Geometry):
        if geo.TypeId == "Part::GeomLineSegment":
            sketch.setConstruction(i, True)


def add_periodic_spline(sketch, knot_points):
    """
    Fit a periodic B-spline through knot_points and add it to the sketch.
    Accuracy near the start knot may be reduced if that point lands in a
    high-curvature region of the profile, since the periodic fit has less
    local freedom there due to the wraparound continuity constraint.
    Increasing n_knots is the mitigation; start point can't be chosen for
    curvature (it must stay tied to global_reference for cross-profile
    alignment) and the closure math offers no direction-related fix.
    """
    bspline = Part.BSplineCurve()
    bspline.interpolate(knot_points, PeriodicFlag=True)
    sketch.addGeometry(bspline, False)
    return bspline


# ---------------------------------------------------------------------------
# Per-sketch driver
# ---------------------------------------------------------------------------

def process_sketch(sketch, n_knots, global_reference):
    """Run centroid -> winding -> start point -> knot sampling -> spline on one sketch."""
    points = get_ordered_points(sketch)
    if len(points) < n_knots:
        App.Console.PrintWarning(
            f"{sketch.Name}: only {len(points)} geoms, fewer than requested "
            f"{n_knots} knots. Choose a lower knot count or check this profile.\n"
        )
        return None

    points = enforce_consistent_winding(sketch, points)
    centroid, _area = polygon_centroid_2d(points)
    start_index = find_start_index(sketch, points, centroid, global_reference)
    knot_points = sample_knots(points, start_index, n_knots)

    bspline = add_periodic_spline(sketch, knot_points)
    set_all_construction(sketch)

    return bspline


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def process_cross_section(global_reference=GLOBAL_REFERENCE_VECTOR):
    """
    Full pipeline: CS object (must be pre-selected) -> sketches -> aligned
    periodic splines. Wrapped in a single undo transaction so the entire
    run reverts as one action; aborts (no partial document changes) if any
    step raises.
    """
    doc = App.ActiveDocument

    sel = Gui.Selection.getSelection()
    if not sel:
        raise RuntimeError(
            "Nothing selected. Select the mesh_crossSections object in the "
            "3D view or model tree before running this macro."
        )
    cs_obj = sel[0]

    n_knots, ok = QtWidgets.QInputDialog.getInt(
        None,
        "Detessellate: Cross Section to Aligned Splines",
        "Number of spline knots per profile:",
        N_KNOTS_DEFAULT,
        4,     # min: a periodic spline needs at least a handful of points
        200,   # max: comfortably above any tested profile's geom count
        1
    )
    if not ok:
        App.Console.PrintMessage("Cancelled, no sketches created.\n")
        return None

    doc.openTransaction("Cross Section to Aligned Splines")
    try:
        sketches = cross_section_to_sketches(cs_obj)
        for sk in sketches:
            process_sketch(sk, n_knots, global_reference)
        doc.recompute()
    except Exception:
        doc.abortTransaction()
        raise
    else:
        doc.commitTransaction()

    return sketches


def run():
    process_cross_section()
