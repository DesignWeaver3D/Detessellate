# -*- coding: utf-8 -*-
# SPDX-License-Identifier: LGPL-2.1-or-later
# SPDX-FileCopyrightText: 2026 DesignWeaver3D
# SPDX-FileNotice: Part of the Detessellate addon.
"""
MeshCSToSplines
---------------
Converts a Mesh Workbench "Cross Sections" (CS) object into one Sketch per
profile, each wrapped with a periodic B-spline suitable for a multi-profile
loft (Part or PartDesign).

Problem this solves: profiles from mesh cross-sections are irregular
("lumpy") point loops, not clean geometric shapes. A loft needs every
profile to have the same pole count and a start point that corresponds to
the same physical location on every profile. Without that, lofts twist or
fail. This macro builds that correspondence directly rather than relying
on the source wire's native point order, which is not reliable for either
property.

Note: winding direction (CW/CCW) was tested empirically and found to have
no effect on loft output for this pipeline, in both smooth and ruled loft
modes, so it's not enforced. Each profile's native point order from the
source wire is used as-is.

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
import traceback
from PySide import QtWidgets


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N_KNOTS_DEFAULT = 16                           # prepopulated in the prompt
GLOBAL_REFERENCE_VECTOR = App.Vector(1, 0, 0)  # shared "start direction" for
                                                # all profiles; see find_start_index


# ---------------------------------------------------------------------------
# Stage 1 - CS object -> list of Sketch objects
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

    # cs_obj is guaranteed valid here; validate_input_shape() in
    # process_cross_section() is the sole gate for this check.
    wires = cs_obj.Shape.Wires

    # Determine this run's number for this source object by checking for
    # existing sketches with this object's run-prefix already in the
    # document (e.g. a prior, non-undone run on the same cs_obj). This is
    # a single filtered pass over doc.Objects, done once per cs_obj rather
    # than per wire, so it stays cheap even for large profile counts.
    # Without this, a rerun would still be made unique by FreeCAD's own
    # addObject suffixing, but that suffix is a bare incrementing number
    # tacked onto the end with no visual grouping; a shared "R##" tag lets
    # every sketch from one run be identified and grouped at a glance.
    run_prefix = f"{cs_obj.Name}_R"
    existing_runs = []
    for o in App.ActiveDocument.Objects:
        if o.Name.startswith(run_prefix):
            digits = o.Name[len(run_prefix):].split("_", 1)[0]
            if digits.isdigit():
                existing_runs.append(int(digits))
    run_number = max(existing_runs, default=0) + 1

    for i, wire in enumerate(wires):
        temp_wire_obj = Part.show(wire, f"TempWire_{i:03d}")
        temp_wire_name = temp_wire_obj.Name

        # No recompute() here: Part.show assigns Shape directly (this is
        # not a parametric feature deriving its shape from inputs), and
        # Draft.make_sketch reads temp_wire_obj.Shape straight from the
        # object. A single recompute already happens once per source
        # object at the end of processing in process_cross_section.

        # Request a name prefixed with the source cs_obj's internal Name
        # (stable, guaranteed-unique at creation time) rather than the
        # Label (user-renameable), plus this run's number, so sketches
        # from two source objects, or two runs on the same source object,
        # are both distinguishable and groupable in the tree. If this
        # exact name is still somehow already taken, addObject silently
        # appends its own numeric suffix to keep Name unique; reading
        # that final Name back for the Label means collisions still can't
        # happen even if run_number's own detection is ever wrong.
        requested_name = f"{cs_obj.Name}_R{run_number:02d}_Spline_{i:03d}"
        sketch = Draft.make_sketch(
            temp_wire_obj, autoconstraints=False, delete=True, name=requested_name
        )
        sketch.Label = sketch.Name
        sketches.append(sketch)

        # delete=True above should already remove the temp object; this is
        # just a safety net in case that behavior changes.
        leftover = App.ActiveDocument.getObject(temp_wire_name)
        if leftover is not None:
            App.ActiveDocument.removeObject(temp_wire_name)

    # deferred: batched into the final recompute in process_cross_section
    return sketches


def resolve_global_reference(sketches, requested_reference):
    """
    All profiles in one CS object share the same plane orientation, so this
    only needs to be checked once. If requested_reference is (nearly)
    parallel to that shared plane normal, its projection onto every
    profile's plane would collapse to a null vector (see find_start_index).
    Substitute a fallback axis in that case so every sketch in this run
    still gets a consistent, valid reference direction.
    """
    probe = sketches[0]
    local = probe.Placement.Rotation.inverted().multVec(requested_reference)
    local.z = 0.0
    if local.Length > 1e-6:
        return requested_reference

    for candidate in (App.Vector(1, 0, 0), App.Vector(0, 1, 0), App.Vector(0, 0, 1)):
        local = probe.Placement.Rotation.inverted().multVec(candidate)
        local.z = 0.0
        if local.Length > 1e-6:
            return candidate

    raise RuntimeError("Could not find a valid reference direction for these profiles.")


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
    """
    Run centroid -> start point -> knot sampling -> spline on one sketch.

    No winding-direction enforcement: tested empirically (an alternating
    CW/CCW/CW interior triplet, plus a manual two-profile reversed-winding
    test) against both smooth and ruled lofts, with no effect on output in
    either mode. Only start-point (vertex) alignment across profiles
    matters for this loft engine; the mesh-cutting algorithm's native
    per-profile point order can be used as-is.
    """
    points = get_ordered_points(sketch)
    if len(points) < n_knots:
        App.Console.PrintWarning(
            f"{sketch.Name}: only {len(points)} geoms, fewer than requested "
            f"{n_knots} knots. Choose a lower knot count or check this profile.\n"
        )
        return None

    centroid, _area = polygon_centroid_2d(points)
    start_index = find_start_index(sketch, points, centroid, global_reference)
    knot_points = sample_knots(points, start_index, n_knots)

    bspline = add_periodic_spline(sketch, knot_points)
    set_all_construction(sketch)

    return bspline


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def validate_input_shape(obj):
    """
    Confirm obj.Shape looks like mesh-derived cross-section wires rather
    than a solid (e.g. a Part Cube), a faced shape, or a raw Mesh object.
    Returns (True, None) if valid, or (False, reason) with a short,
    user-facing explanation if not.

    This is a pre-flight gate: it's meant to run before any processing
    starts, so a rejection here should never surface as a Python traceback
    from deeper in the pipeline (e.g. a knot-count mismatch or an empty
    Shape.Wires list).
    """
    expected = (
        "Select the object created by Mesh Workbench's Cross-sections command "
        "(Mesh menu > Cross-sections), typically named <MeshName>_cs, where "
        "MeshName is the source mesh object's name. It should contain only "
        "wires, no faces or solid geometry."
    )

    if not hasattr(obj, "Shape"):
        return False, (
            f"'{obj.Name}' is a raw Mesh object with no Shape. {expected}"
        )

    shape = obj.Shape

    if shape.Solids or shape.Volume > 1e-7:
        return False, (
            f"'{obj.Name}' contains solid geometry (Volume={shape.Volume:.4f}). "
            f"{expected}"
        )

    if shape.Faces:
        return False, (
            f"'{obj.Name}' has {len(shape.Faces)} face(s). {expected}"
        )

    if not shape.Wires:
        return False, (
            f"'{obj.Name}' has no wires. {expected}"
        )

    return True, None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def process_cross_section(global_reference=GLOBAL_REFERENCE_VECTOR):
    """
    Full pipeline: one or more pre-selected CS objects -> sketches -> aligned
    periodic splines, run once per valid selected object.

    Selection handling:
    - No selection: clean exit, no processing.
    - Mixed valid/invalid selection: invalid objects are skipped (reported
      together in the Report View, not processed), valid objects proceed.
    - All invalid: clean exit, no processing.

    Each valid object gets its own undo transaction, so if one object fails
    partway through (an unexpected error, not a pre-flight validation
    rejection), only that object's partial work is rolled back; sketches
    already completed and committed for other objects in the same run are
    kept. Such a failure is a logic error, not a user input mistake, so it
    is still logged with a full traceback rather than a quiet warning; it
    just doesn't halt the rest of the batch.
    """
    doc = App.ActiveDocument

    sel = Gui.Selection.getSelection()
    if not sel:
        msg = "MeshCSToSplines: select the mesh cross-sections object before running this tool."
        Gui.getMainWindow().statusBar().showMessage(msg, 5000)
        App.Console.PrintWarning(msg + "\n")
        return None

    valid_objs = []
    invalid_entries = []
    for obj in sel:
        ok, reason = validate_input_shape(obj)
        if ok:
            valid_objs.append(obj)
        else:
            invalid_entries.append((obj.Name, reason))

    if invalid_entries:
        lines = [f"  - {name}: {reason}" for name, reason in invalid_entries]
        report_msg = "MeshCSToSplines: skipped {} of {} selected object(s):\n{}".format(
            len(invalid_entries), len(sel), "\n".join(lines)
        )
        App.Console.PrintWarning(report_msg + "\n")
        Gui.getMainWindow().statusBar().showMessage(
            f"MeshCSToSplines: skipped {len(invalid_entries)} of {len(sel)} "
            "selected object(s), see Report View.",
            5000
        )

    if not valid_objs:
        return None

    n_knots, ok = QtWidgets.QInputDialog.getInt(
        None,
        "Detessellate: MeshCSToSplines",
        "Number of spline knots per profile:",
        N_KNOTS_DEFAULT,
        4,     # min: a periodic spline needs at least a handful of points
        200,   # max: comfortably above any tested profile's geom count
        1
    )
    if not ok:
        App.Console.PrintMessage("Cancelled, no sketches created.\n")
        return None

    all_sketches = []
    failures = []

    for cs_obj in valid_objs:
        doc.openTransaction(f"MeshCSToSplines: {cs_obj.Name}")
        try:
            sketches = cross_section_to_sketches(cs_obj)
            obj_reference = resolve_global_reference(sketches, global_reference)
            for sk in sketches:
                process_sketch(sk, n_knots, obj_reference)
            doc.recompute()
        except Exception:
            doc.abortTransaction()
            failures.append(cs_obj.Name)
            App.Console.PrintError(
                f"MeshCSToSplines: '{cs_obj.Name}' failed and was rolled back "
                f"(other objects in this run are unaffected):\n{traceback.format_exc()}"
            )
            continue
        else:
            doc.commitTransaction()
            all_sketches.extend(sketches)

    if failures:
        summary = (
            f"MeshCSToSplines: {len(failures)} of {len(valid_objs)} object(s) "
            f"failed and were rolled back, see Report View. "
            f"{len(valid_objs) - len(failures)} completed successfully."
        )
        Gui.getMainWindow().statusBar().showMessage(summary, 5000)
        App.Console.PrintWarning(summary + "\n")

    return all_sketches


def run():
    process_cross_section()


if __name__ == "__main__":
    run()