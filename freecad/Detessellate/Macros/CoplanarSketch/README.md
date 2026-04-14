# CoplanarSketch FreeCAD Macro

<img width="128" height="128" alt="CoplanarSketch" src="https://github.com/user-attachments/assets/a941d04a-1707-400b-bd9c-d0751c8ea021" />

The `CoplanarSketch` FreeCAD macro is a powerful tool designed to streamline the creation of sketches from existing 3D geometry, specifically focusing on **coplanar edges** found on tessellated solid bodies such as those converted from Mesh objects imported from STL files. It automates the process of identifying and selecting edges that lie on the same plane, then generates a new sketch containing these edges as construction geometry, correctly oriented in space.

Forum Post: https://forum.freecad.org/viewtopic.php?p=830918#p830918

![image](https://github.com/user-attachments/assets/88df8cf1-5ee3-4aa6-868f-9386a0d87e94)

## Features
* **Intelligent Coplanar Edge Selection**: Selects edges that share a common plane from a selected 3D object based on the actively selected face or by two coplanar edges.
* **Flexible Sketch Placement**: Offers multiple options for where the new sketch is created:
    * As a **standalone sketch** in the document's root (Part Workbench style).
    * Within a **newly created PartDesign Body**, with the sketch automatically attached to the body's XY plane.
    * Attached to the XY plane of an **existing PartDesign Body**, correctly nesting the sketch within the body in the model tree.
* **Construction Geometry**: All derived edges are added to the new sketch as construction geometry, providing a precise basis for further design without interfering with solid operations.
* **Robust Constraints**: Automatically applies block constraints to maintain the relative positions of the construction lines and adds coincident constraints where vertices are shared, ensuring stability.
* **User-Friendly Interface**: Provides a dockable GUI panel within FreeCAD with a guided user experience for easy access to its functionalities.
* **Degenerate Edge Detection**: Detects degenerate edges in non-solid shape objects and provides a means to create a clean shape by copying non-degenerate faces and omitting degenerate faces and edges.

## Alternative Installation
This macro is bundled with the Detessellate Workbench, but can also be manually installed separately.

1.  **Installation**:
    * Save the `CoplanarSketch.py` file into your FreeCAD Macros directory. You can find this directory by going to `Macros -> Macros...` in FreeCAD and checking the "User macros location" path.
    * (Optional but Recommended) Restart FreeCAD.
    * (Optional) Copy the icon file for use as custom toolbar icon.

2.  **Running the Macro**:
    * Open a FreeCAD document containing a tessellated shape or solid body.
    * Select the shape (or specific edges/faces within it if desired).
    * Go to `Macros -> Macros...`.
    * Select `CoplanarSketch.py` from the list and click "Execute".
    * A dockable "Coplanar Sketch" docker panel will appear.
    * Click "Collect Edge Data" (required for an initial scan of all edge data for the selected object).
       * If degenerate edges are detected, Click "Clean Degenerate Edges" to create a cleaned copy of the shape. 
    * With a single face or two edges selected, Click "Select Coplanar Edges" to have the macro identify and select coplanar edges based the plane calculated from the selection.
    * Click "Create Sketch from Selection". A dialog will appear asking you to choose the sketch destination:
        * `<Standalone (Part Workbench)>`: Creates the sketch directly in the document root using placement properties.
        * `<Create New Body (PartDesign)>`: Creates a new PartDesign Body and positions the sketch inside using attachment offset from the body's XY plane.
        * `[Existing Body Name]`: Places the sketch inside a selected existing PartDesign Body and positions the sketch inside using attachment offset from the body's XY plane.
    * Choose your desired option and click "OK". The created sketch will have the coplanar edges drawn as block-constrained construction geometry with coincident edge vertices constrained. Sketch origin will be at the center of mass of the selected edeges and the sketch geometry aligned with the real edges' 3D position.

3.  **Post-Creation**:
    * The new sketch will be created and automatically selected in the tree view.
    * You can then enter the sketch to convert the construction to regular or add further geometry, dimensions, and constraints.

## Compatibility
This macro has been developed and tested with the following FreeCAD environment:
* **FreeCAD Version**: v1.0.1
* **Python Version**: 3.11.12
* **PySide Version**: 5.15.15 (Qt 5.15.15)
* **Operating System**: Windows 11 64-bit

While it may work on other versions or operating systems, compatibility is ensured for the listed environment.

## Contribution & Feedback
Feel free to open issues on this repository if you encounter any bugs or have suggestions for improvements.

## 📜 Changelog
- **v3.1.0** (2026-01-18) - Massive optimization for 10-71x faster sketch generation.
- **3.0** Guided user experience. Improved performance. New feature to detect degenerate edges and rebuild shape from faces without degenerate edges.
- **2.06** Combined features of previous 2.02 and 250608. Fixed regressions introduced in 2.02.
- **2.02** Improved geometry to sketch translation fixing bug when selected edges were coplanar with global XY plane. Added sketch normal direction logic based on selected edges comparison to center mass. Edge selections away from the model center of mass now get their sketches created with proper sketch normal facing away from the center of mass. Improved messaging inside the docker.
- **250608** Initial version.
