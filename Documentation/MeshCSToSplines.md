# MeshCSToSplines FreeCAD Macro

<img width="128" height="128" alt="MeshCSToSplines" src="PLACEHOLDER" />

The `MeshCSToSplines` FreeCAD macro converts a Mesh Workbench "Cross Sections" (CS)
object into one Sketch per profile, each wrapped with a periodic B-spline suitable
for a multi-profile loft (Part or PartDesign). Profiles from mesh cross-sections are
irregular ("lumpy") point loops, not clean geometric shapes, and a loft needs every
profile to have the same pole count, matching winding direction, and a start point
that corresponds to the same physical location on every profile — without that,
lofts twist or fail. This macro builds that correspondence directly rather than
relying on the source wire's native point order, which is not reliable for any of
those three properties.

Forum Post: PLACEHOLDER

## Features
* **One sketch per cross-section profile**: Each wire in the CS object's `Shape.Wires` becomes its own Sketch, staged via a temporary Draft object and discarded once converted.
* **Consistent winding direction**: Every profile is checked and, if necessary, reversed so all profiles wind the same way (CW as viewed along global +Z), computed in global coordinates via each sketch's own Placement.
* **Cross-profile aligned start point**: Picks the start knot on each profile as the point closest to a shared global reference direction, transformed into each sketch's own local space — so the same physical location is used as the start point across all profiles regardless of individual sketch rotation.
* **Periodic B-spline fitting**: Samples a user-specified number of knots per profile by index-stepping around the ordered point loop, then fits a periodic B-spline through them, leaving the original polyline as construction geometry.
* **Single undo transaction**: The whole run (sketch creation + spline fitting for every profile) is wrapped in one document transaction, so it can be undone or aborts cleanly as one action if any step fails.

## Alternative Installation
This macro is bundled with the Detessellate Workbench, but can also be manually installed separately.

1.  **Installation**:
    * Save the `MeshCSToSplines.py` file into your FreeCAD Macros directory. You can find this directory by going to `Macros -> Macros...` in FreeCAD and checking the "User macros location" path.
    * (Optional but Recommended) Restart FreeCAD.
    * (Optional) Copy the icon file for use as a custom toolbar icon.

2.  **Running the Macro**:
    * Select the Mesh Workbench Cross Sections object in the 3D view or model tree.
    * Go to `Macros -> Macros...`.
    * Select `MeshCSToSplines.py` from the list and click "Execute".
    * When prompted, enter the number of spline knots to fit per profile.
    * One sketch per profile is created, each containing the original polyline (as construction geometry) and a periodic B-spline fitted through the sampled knots.

3.  **Post-Creation**:
    * Inspect each generated sketch against the source mesh, especially for profiles that were broken into multiple wire fragments (e.g. from a hole in the source mesh) — these are not detected or repaired automatically.
    * Assemble the sketches into a loft (`Part::Loft` or `PartDesign::AdditiveLoft`) yourself; profile grouping, ruled/smooth, and workbench choice are use-case-specific and are left to you.

## Compatibility
This macro has been developed and tested with the following FreeCAD environment:
* **FreeCAD Version**: PLACEHOLDER
* **Python Version**: PLACEHOLDER
* **PySide Version**: PLACEHOLDER
* **Operating System**: PLACEHOLDER

While it may work on other versions or operating systems, compatibility is ensured for the listed environment.

## Contribution & Feedback
Feel free to open issues on this repository if you encounter any bugs or have suggestions for improvements.

## 📜 Changelog
- **v1.0.0** - Initial version integrated into Detessellate.
