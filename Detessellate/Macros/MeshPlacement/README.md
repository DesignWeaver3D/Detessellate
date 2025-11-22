# MeshPlacement

<img width="64" height="64" alt="MeshPlacement" src="https://github.com/user-attachments/assets/57233128-99af-42cd-b23f-17bc44b23b97" />

A FreeCAD macro to position one or more selected meshes at the global origin based on their bounding box.  
This tool provides a dock widget with intuitive buttons to **center** or **align** meshes along X, Y, Z axes (or combinations like XY, XYZ). It works on single or multiple selected meshes, with undo safety and a clean UI layout.

<img width="1228" height="800" alt="image" src="https://github.com/user-attachments/assets/a15c7244-2c3f-4190-8c75-4178c0d279d4" />

---

## ✨ Features
- Center meshes at the origin:
  - Center XYZ (bounding box center)
  - Center XY (retains current Z-position)
  - Center X
  - Center Y
  - Center Z
- Align meshes to bounding box planes:
  - Align Top / Bottom
  - Align Left / Right
  - Align Front / Back
- Works on **one or many selected meshes** at once.
- Undo/redo safety: each action is wrapped in a FreeCAD transaction.
- Dock widget UI with grouped controls for clarity.

---

## Installation

1. Download `MeshPlacement.py`  
2. Place it in your FreeCAD macros directory which can be determined in these locations:
   - `Edit > Preferences > Python > Macro > Macro path`
   - `Macro > Macros... > User macros location:`
3. Restart FreeCAD if already running

---

## 🛠 Usage
1. Select one or more mesh objects from the Model Tree.
2. Run the macro via `Macro > Macros... > MeshPlacement.py > Execute`
3. Use the dock widget buttons to center or align them.
4. Each action is undoable via **Edit → Undo**.

---

## 🔒 License
This macro is licensed under the **GNU GPL v3.0**.  
You are free to use, modify, and distribute it under the terms of the license.

---

## 📌 Compatibility
- FreeCAD **v1.0.2 or later**
- Tested with FreeCAD 1.0.2 stable

---

## 📜 Changelog

- v1.0.0 – 2025.11.21  Initial release of **MeshPlacement** macro.
