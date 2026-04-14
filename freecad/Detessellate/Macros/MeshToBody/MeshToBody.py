# -*- coding: utf-8 -*-
"""
MeshToBody Macro for FreeCAD
----------------------------
Version: 2.0.1
Date:    2025-11-18
Author:  NSUBB (DesignWeaver3D)
License: GPL-3.0-or-later
Repository: https://github.com/NSUBB/MeshToBody
Forum Thread: https://forum.freecad.org/viewtopic.php?t=101189

Description:
    FreeCAD Macro to convert one or many Meshes to Refined Solids as BaseFeature of PartDesign Bodies.
"""

import FreeCAD
import FreeCADGui
import Mesh
import Part
import PartDesign
from PySide import QtGui, QtCore
import time
import sys

# --- Cleanup helper ---
def cleanup_interims(doc, names, label=None, verbose=False):
    """Remove interim objects by name if they still exist in the document."""
    for name in names:
        obj = doc.getObject(name)
        if obj:
            try:
                doc.removeObject(name)
                if verbose and label:
                    FreeCAD.Console.PrintMessage(f"🧹 Cleaned up {label}: {name}\n")
            except Exception as e:
                if verbose:
                    FreeCAD.Console.PrintError(f"⚠️ Could not remove {name}: {e}\n")
    doc.recompute()

# --- Conservative repair ---
def attempt_mesh_repair(mesh_obj):
    try:
        mesh = mesh_obj.Mesh
        if not mesh.isSolid():
            FreeCAD.Console.PrintMessage("🔧 Skipping repair: mesh is not solid (repairs can be destructive on overlap).\n")
            return False
        FreeCAD.Console.PrintMessage(f"🔧 Conservative repair for '{mesh_obj.Name}'...\n")
        mutable_mesh = Mesh.Mesh(mesh.Topology)
        if hasattr(mutable_mesh, "harmonizeNormals"):
            mutable_mesh.harmonizeNormals()
        for method in ["removeDuplicatedPoints", "removeDuplicatedFacets", "removeInvalidPoints"]:
            if hasattr(mutable_mesh, method):
                getattr(mutable_mesh, method)()
        if hasattr(mutable_mesh, "fillupHoles"):
            mutable_mesh.fillupHoles(100)
        mesh_obj.Mesh = mutable_mesh
        FreeCAD.ActiveDocument.recompute()
        return mutable_mesh.isSolid() and not mutable_mesh.hasNonManifolds() and not mutable_mesh.hasSelfIntersections()
    except Exception as e:
        FreeCAD.Console.PrintError(f"Repair failed: {e}\n")
        return False

# --- Convert mesh to solid ---
def convert_mesh_to_solid(mesh_obj, base_name, doc):
    try:
        # Mesh → Shape
        shape_obj = doc.addObject('Part::Feature', base_name + "_shape")
        shape = Part.Shape()
        shape.makeShapeFromMesh(mesh_obj.Mesh.Topology, 0.1, False)
        shape_obj.Shape = shape

        if not shape_obj.Shape.Faces:
            raise ValueError("No faces to build solid")

        # Shape → Solid
        solid_obj = doc.addObject("Part::Feature", base_name + "_solid")
        solid_obj.Shape = Part.Solid(Part.Shell(shape_obj.Shape.Faces))

        # Solid → Refine
        refined_obj = doc.addObject("Part::Refine", base_name + "_solid_refined")
        refined_obj.Source = solid_obj
        solid_obj.Visibility = False
        doc.recompute()

        # --- Pragmatic validity check ---
        target_shape = None
        if refined_obj.Shape and refined_obj.Shape.Solids:
            # Refined solid exists and has solids → keep it
            target_shape = refined_obj.Shape
        elif solid_obj.Shape and solid_obj.Shape.Solids:
            # Refine failed, but unrefined solid is usable → fallback
            FreeCAD.Console.PrintMessage(
                f"⚠️ Refine invalid for {base_name}, keeping unrefined solid\n"
            )
            target_shape = solid_obj.Shape
        else:
            raise ValueError("No usable solid produced")

        # Final keeper: simple copy
        simple_copy_obj = doc.addObject("Part::Feature", base_name + "_solid_simple")
        simple_copy_obj.Shape = target_shape
        simple_copy_obj.ViewObject.Visibility = True

        # Delete interims
        for o in (shape_obj, solid_obj, refined_obj):
            if doc.getObject(o.Name):
                doc.removeObject(o.Name)
        doc.recompute()

        return True, simple_copy_obj, []

    except Exception as e:
        for name in [base_name + "_shape", base_name + "_solid", base_name + "_solid_refined"]:
            if doc.getObject(name):
                doc.removeObject(name)
        FreeCAD.Console.PrintError(f"❌ Solid conversion failed for '{mesh_obj.Name}': {e}\n")
        return False, None, []

# --- Compound solids ---
def fusion_solids(solids, base_name, doc):
    if len(solids) == 0:
        return False, None, []
    if len(solids) == 1:
        return True, solids[0], []

    FreeCAD.Console.PrintMessage(f"🔄 Creating fusion for {len(solids)} solids...\n")
    fusion = doc.addObject("Part::MultiFuse", base_name + "_fusion")
    fusion.Shapes = solids
    for s in solids:
        s.Visibility = False
    doc.recompute()

    return True, fusion, []  # keep solids, they are part of the fusion

# --- Split components ---
def split_components_safe(mesh_obj, base_name, doc):
    try:
        FreeCAD.Console.PrintMessage(f"🔄 Splitting components for '{mesh_obj.Name}'...\n")
        comps = mesh_obj.Mesh.getSeparateComponents()
        comp_objs = []
        for i, comp in enumerate(comps):
            comp_name = f"{base_name}_comp_{i+1:02d}"
            comp_obj = doc.addObject("Mesh::Feature", comp_name)
            comp_obj.Mesh = comp
            comp_objs.append(comp_obj)
        doc.recompute()
        FreeCAD.Console.PrintMessage(f"✅ Split produced {len(comp_objs)} components\n")
        return comp_objs, (len(comp_objs) > 1)
    except Exception as e:
        FreeCAD.Console.PrintError(f"SplitComponents failed for '{mesh_obj.Name}': {e}\n")
        return [], False

# --- Evaluation order ---
def evaluate_mesh(mesh_obj):
    mesh = mesh_obj.Mesh
    comps = mesh.countComponents()
    if comps > 1:
        return "fusion"
    if mesh.isSolid() and not mesh.hasNonManifolds() and not mesh.hasSelfIntersections():
        return "proceed"
    if not mesh.isSolid() and mesh.hasSelfIntersections():
        return "try_split"
    if mesh.isSolid():
        return "try_repair"
    return "repair"

# --- Convert single mesh ---
def convert_single_mesh(mesh_obj, doc):
    base_name = mesh_obj.Name
    decision = evaluate_mesh(mesh_obj)

    interims = []  # names to delete on success/failure
    try:
        if decision == "proceed":
            success, final_solid, _ = convert_mesh_to_solid(mesh_obj, base_name, doc)

        elif decision == "fusion":
            comp_objs, _ = split_components_safe(mesh_obj, base_name, doc)
            interims.extend([c.Name for c in comp_objs])  # mesh features are interim

            solids = []
            for i, comp_obj in enumerate(comp_objs):
                comp_name = comp_obj.Name
                success_i, solid_i, _ = convert_mesh_to_solid(comp_obj, f"{base_name}_comp_{i+1:02d}", doc)
                if doc.getObject(comp_name):
                    doc.removeObject(comp_name)  # remove mesh feature
                if success_i and solid_i:
                    solids.append(solid_i)
                else:
                    raise RuntimeError(f"Component {comp_name} failed conversion")

            success, final_solid, _ = fusion_solids(solids, base_name, doc)

        elif decision == "try_split":
            comp_objs, is_true_comp = split_components_safe(mesh_obj, base_name, doc)
            interims.extend([c.Name for c in comp_objs])

            if is_true_comp:
                solids = []
                for i, comp_obj in enumerate(comp_objs):
                    comp_name = comp_obj.Name
                    success_i, solid_i, _ = convert_mesh_to_solid(comp_obj, f"{base_name}_comp_{i+1:02d}", doc)
                    if doc.getObject(comp_name):
                        doc.removeObject(comp_name)
                    if success_i and solid_i:
                        solids.append(solid_i)
                    else:
                        raise RuntimeError(f"Component {comp_name} failed conversion")
                success, final_solid, _ = fusion_solids(solids, base_name, doc)
            else:
                FreeCAD.Console.PrintMessage("🧩 Split produced only one component, treating as single mesh.\n")
                success, final_solid, _ = convert_mesh_to_solid(mesh_obj, base_name, doc)

        elif decision == "try_repair":
            FreeCAD.Console.PrintMessage(f"🔧 Attempting repair for '{mesh_obj.Name}'...\n")
            if attempt_mesh_repair(mesh_obj):
                success, final_solid, _ = convert_mesh_to_solid(mesh_obj, base_name, doc)
            else:
                FreeCAD.Console.PrintMessage(f"⏭️ Skipping '{mesh_obj.Name}' (repair not successful)\n")
                return False, None

        else:
            FreeCAD.Console.PrintMessage(f"⏭️ Skipping '{mesh_obj.Name}' (not solid and no split path)\n")
            return False, None

        if success and final_solid:
            mesh_name = mesh_obj.Name  # capture before removal

            body = doc.addObject("PartDesign::Body", base_name + "_Body")
            body.BaseFeature = final_solid
            final_solid.Visibility = False

            if doc.getObject(mesh_obj.Name):
                doc.removeObject(mesh_obj.Name)

            # SUCCESS CLEANUP: remove interim mesh features
            cleanup_interims(doc, interims, label="mesh component")

            doc.recompute()
            FreeCAD.Console.PrintMessage(f"✅ Created Body: {body.Name}\n")
            return True, (mesh_name, body.Name)

        raise RuntimeError("Conversion failed")

    except Exception as e:
        FreeCAD.Console.PrintMessage(f"❌ Conversion failed for '{mesh_obj.Name}': {e}\n")
        # FAILURE CLEANUP
        cleanup_interims(doc, interims, label="failure cleanup")
        for suffix in ["_Body", "_fusion"]:
            name = f"{base_name}{suffix}"
            if doc.getObject(name):
                doc.removeObject(name)
        doc.recompute()
        return False, None

# --- Component count guardrail ---
def should_skip_for_component_count(mesh_obj, total_meshes):
    comps = mesh_obj.Mesh.countComponents()
    if comps > 50 and total_meshes > 1:
        return True, comps
    return False, comps

# --- Unified entry point with pre-collection, sorted schedule, and per-mesh transactions ---
def run_unified_macro(auto_mode=True):
    doc = FreeCAD.ActiveDocument

    # Initial banner
    FreeCAD.Console.PrintMessage("\n🚀 Mesh-to-Body macro started...\n")
    QtGui.QApplication.processEvents()
    sys.stdout.flush()

    # Collect selection or all meshes
    selection = FreeCADGui.Selection.getSelection()
    if selection:
        mesh_objects = [obj for obj in selection if hasattr(obj, 'Mesh')]
        FreeCAD.Console.PrintMessage(f"Found {len(mesh_objects)} selected meshes...\n")
    else:
        mesh_objects = [obj for obj in doc.Objects if hasattr(obj, 'Mesh')]
        FreeCAD.Console.PrintMessage(f"No selection found. Found {len(mesh_objects)} meshes in document...\n")

    QtGui.QApplication.processEvents()
    sys.stdout.flush()

    # --- Pre-collect metadata ---
    metadata = []
    for obj in mesh_objects:
        metadata.append({
            "obj": obj,
            "name": obj.Name,
            "facets": obj.Mesh.CountFacets,
            "components": obj.Mesh.countComponents()
        })

    # --- Sort: first by components, then by facets ---
    metadata.sort(key=lambda m: (m["components"], m["facets"]))

    # --- Print schedule only ---
    FreeCAD.Console.PrintMessage("\n📋 Conversion schedule:\n")
    for idx, m in enumerate(metadata, 1):
        FreeCAD.Console.PrintMessage(
            f"{idx}. {m['name']} — {m['components']} components, {m['facets']} facets\n"
        )
    QtGui.QApplication.processEvents()
    sys.stdout.flush()

    # --- Conversion loop ---
    results = {"converted": 0, "skipped": 0}
    skip_reasons = []
    total_start = time.perf_counter()

    for idx, m in enumerate(metadata, 1):
        obj = m["obj"]
        comps = m["components"]
        facets = m["facets"]

        # Skip guardrail
        if comps > 50 and len(metadata) > 1:
            FreeCAD.Console.PrintMessage(
                f"\n=== [{idx}/{len(metadata)}] {m['name']} ===\n"
                f"📐 {m['name']} has {facets} facets, {comps} components\n"
                f"⚠️ Skipping {m['name']} ({comps} components > 50)\n"
            )
            results["skipped"] += 1
            skip_reasons.append((m["name"], comps))
            continue

        # Header + counts + warnings + start marker together
        FreeCAD.Console.PrintMessage(
            f"\n=== [{idx}/{len(metadata)}] {m['name']} ===\n"
            f"📐 {m['name']} has {facets} facets, {comps} components\n"
        )
        if facets > 10000:
            FreeCAD.Console.PrintMessage(
                f"⏳ {m['name']} exceeds 10k facets. Expect long processing time...\n"
            )
        FreeCAD.Console.PrintMessage(
            f"▶️ Starting conversion for {m['name']}, please wait...\n"
        )
        QtGui.QApplication.processEvents()
        sys.stdout.flush()

        # Conversion
        doc.openTransaction(f"Convert {obj.Name}")
        start = time.perf_counter()
        try:
            success, names = convert_single_mesh(obj, doc)
            elapsed = time.perf_counter() - start
            if success:
                mesh_name, body_name = names
                results["converted"] += 1
                FreeCAD.Console.PrintMessage(
                    f"✅ {mesh_name} converted to {body_name} in {elapsed:.2f}s\n"
                )
            else:
                results["skipped"] += 1
                FreeCAD.Console.PrintMessage(
                    f"⏭️ {obj.Name} skipped after {elapsed:.2f}s\n"
                )
        except Exception as e:
            results["skipped"] += 1
            FreeCAD.Console.PrintError(f"❌ Error converting {obj.Name}: {e}\n")
        finally:
            doc.commitTransaction()
            QtGui.QApplication.processEvents()

    doc.recompute()
    total_elapsed = time.perf_counter() - total_start

    # --- Final summary ---
    FreeCAD.Console.PrintMessage(
        f"\n=== Conversion complete: {results['converted']} converted, "
        f"{results['skipped']} skipped ===\n"
    )
    FreeCAD.Console.PrintMessage(f"⏱️ Total elapsed time: {total_elapsed:.2f}s\n")

    if skip_reasons:
        FreeCAD.Console.PrintMessage("\n=== Skipped meshes due to component count ===\n")
        for name, comps in skip_reasons:
            FreeCAD.Console.PrintMessage(f"⏭️ {name}: {comps} components. Rerun separately.\n")

    return results

# --- Main execution ---
if __name__ == "__main__":
    run_unified_macro(auto_mode=True)
