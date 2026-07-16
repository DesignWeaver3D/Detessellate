# MeshCSToSplines

<img width="128" height="128" alt="MeshCSToSplines" src="PLACEHOLDER" />

The `MeshCSToSplines` FreeCAD macro converts a Mesh Workbench "Cross Sections" (CS)
object into one Sketch per profile, each wrapped with a periodic B-spline suitable
for a multi-profile loft (Part or PartDesign). Profiles from mesh cross-sections are
irregular polygons of straight edges that are often not ready for 3D feature operations. Lofts need every
profile to have the same vertex or pole count and a start point
that corresponds to the same physical location on every profile — without that,
lofts twist or fail. This macro builds that correspondence directly rather than
relying on the source wire's native point order, which is not reliable for either
property.

Forum Post: PLACEHOLDER

## Features
* **Batch processing**: Select one or more Cross Sections objects at once; each is processed and committed independently, so a failure on one doesn't affect the others.
* **Pre-flight validation**: Selected objects are checked before any processing starts — raw Mesh objects, solids, and faced shapes are rejected with a clear, specific reason instead of failing deep in the geometry pipeline. Invalid objects in a mixed selection are skipped and reported; valid ones still proceed.
* **One sketch per cross-section profile**: Each wire in the CS object's `Shape.Wires` becomes its own Sketch using Draft MakeSketch.
* **Run-grouped naming**: Sketches are named `<CS object name>_R##_Spline_###`, so sketches from the same run on the same source object (or from separate reruns) stay identifiable and groupable in the tree.
* **Winding direction left as-is**: Profile point order from the source wire is used unchanged. Winding direction (CW/CCW) was tested empirically and found to have no effect on loft output for this pipeline in either smooth or ruled loft modes, so no correction is applied.
* **Cross-profile aligned start point**: Picks the start knot on each profile as the point closest to a shared global reference direction, transformed into each sketch's own local space — so the same physical location is used as the start point across all profiles regardless of individual sketch rotation.
* **Periodic B-spline fitting**: Samples a user-specified number of knots per profile by index-stepping around the ordered point loop, then fits a periodic B-spline through them, leaving the original polyline as construction geometry.
* **Per-object undo transactions**: Each source object's sketch creation + spline fitting is wrapped in its own document transaction, so a failure rolls back only that object's work — sketches already completed for other objects in the same run are kept.

## Running the Macro
* Select one or more Mesh Workbench Cross Sections objects in the 3D view or model tree.
* Run the macro via the Detessellate toolbar/menu, or via `Macro → Macros... → MeshCSToSplines.py → Execute` if installed standalone.
* If any selected object isn't a valid Cross Sections object, it's skipped with the reason printed to the Report View; valid objects still proceed. If nothing in the selection is valid, no dialog appears.
* When prompted, enter the number of spline knots to fit per profile.
* One sketch per profile is created for each valid object, each containing the original polyline (as construction geometry) and a periodic B-spline fitted through the sampled knots.

**Post-Creation**:
* Inspect each generated sketch against the source mesh, especially for profiles that were broken into multiple wire fragments (e.g. from a hole in the source mesh) — these are not detected or repaired automatically.
* Assemble the sketches into a loft (`Part::Loft` or `PartDesign::AdditiveLoft`) yourself; profile grouping, ruled/smooth, and workbench choice are use-case-specific and are left to you.

## Alternative Installation

This macro is bundled with the Detessellate Workbench, but can also be installed as a standalone macro.

1. Download `MeshCSToSplines.py` from this repository.
2. Place it in your FreeCAD macros directory:
   - `Macro → Macros...` → **User macros location**, or
   - `Edit → Preferences → Python → Macro → Macro path`
3. Restart FreeCAD, or click **Refresh** in the Macro dialog.
4. (Optional) Copy the matching icon from `Resources/Icons/MeshCSToSplines.svg` for use as a custom toolbar button.

## 📜 Changelog
- **v1.1.0** - Added multi-object batch processing with per-object transaction isolation, pre-flight input validation with descriptive rejection messages, and run-grouped sketch naming. Removed winding-direction enforcement after empirical testing found it has no effect on loft output.
- **v1.0.0** - Initial version integrated into Detessellate.
