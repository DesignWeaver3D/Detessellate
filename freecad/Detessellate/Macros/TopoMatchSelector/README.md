# TopoMatchSelector

**TopoMatchSelector** is a FreeCAD macro designed to help users select more stable geometric references across the feature history of a PartDesign body. By dynamically detecting exact and similar matches for faces, edges, and vertices, the macro empowers users to anchor their dependencies to earlier features—minimizing risk of topological naming issues (TNP).

FreeCAD Forum discussion: https://forum.freecad.org/viewtopic.php?t=98205

Reddit discussion: https://www.reddit.com/r/FreeCAD/comments/1luer9h/a_macro_for_partdesign_to_help_mitigate_tnp/

![image](https://github.com/user-attachments/assets/edef74e0-6a6e-4310-b09c-c42be3a09554)

## 🧠 Purpose

In PartDesign workflows, features are built on prior geometry, yet selecting from the tip feature often exposes models to TNP instability. TopoMatchSelector enables users to:

- Select a face, edge, or vertex from the visible tip
- View **exact matches** from earlier features in the body
- Discover **similar geometry** (e.g. coplanar faces, collinear edges)
- Dynamically reselect a stable match—such as from the BaseFeature—for sketch attachment or dependency creation

## 🛠️ Features

- 🧬 **Exact and Similar Matching** for faces, edges, and vertices
- 🧲 **Live Selection Tracking** from the 3D view
- 📋 **Match Lists Ordered by Feature History** (early to late)
- 🪁 **Selectable Matches** from the UI to change active selection
- 🧱 **Docker Interface** that integrates natively into FreeCAD's UI
- 🔍 **Geometry Matching Engine** with refined tolerance logic

## 📦 Alternative Installation
This macro is bundled with the Detessellate Workbench, but can also be manually installed separately.

1. Place `TopoMatchSelector1.0.py` into your FreeCAD macro directory.
2. Launch FreeCAD, switch to PartDesign Workbench.
3. Execute the macro from the Macro menu.
4. The **TopoMatchSelector** docker will appear on the right side of the UI.

## 🎯 Usage

1. Select a face/edge/vertex from the visible tip feature.
2. The docker displays:
   - **Exact Matches** (stable geometry in earlier features)
   - **Similar Matches** (coplanar or overlapping geometry)
3. Click any listed match to reselect it in the 3D view.
4. Proceed with sketch attachment or feature creation using the stable reference.

## 📐 Matching Logic

| Type    | Exact Match                          | Similar Match                            |
|---------|--------------------------------------|-------------------------------------------|
| Face    | Same surface type, area, center, axis | Coplanar, overlapping, similar area       |
| Edge    | Same curve type, length, endpoints    | Collinear or circular with overlap        |
| Vertex  | Same coordinates                      | *(Only exact matching supported)*         |

## 🧪 Tip

To mitigate TNP risks, always base dependent sketches and features on geometry **from earlier features** in the body when possible. TopoMatchSelector helps you do just that—visually and interactively.

## 💬 Feedback & Contributions

Feel free to open issues or pull requests to suggest improvements, feature requests, or bug fixes. This macro is continuously evolving to support precise, traceable workflows in FreeCAD.

## 🔗 Compatibility

This macro was tested using the Windows FreeCAD v1.0.1.
