# ConstrainAllPointOnPoint

**ConstrainAllPointOnPoint** is a FreeCAD macro that automatically adds coincident constraints to all overlapping vertices in a sketch.

<img width="128" height="128" alt="ConstrainAllPointOnPoint" src="https://github.com/user-attachments/assets/9d359aa9-5027-4e1f-b349-1d1f51a7f6f4" />

### 🔍 Features

- **One-Click Operation**: No dialogs, no manual selection, transaction-safe with rollback
- **Tolerance-Based Grouping**: Configurable precision (default 100µm manufacturing tolerance)
- **Smart B-spline Handling**: Excludes B-spline endpoints (managed via construction circles)
  - **Control Point Filtering**: Only constrains construction circles at actual curve endpoints
  - **InternalAlignment Awareness**: Respects FreeCAD's internal geometric relationships
  - **Transitive Detection**: Avoids redundant constraints through graph-based connectivity checking

### ⚙️ Requirements

- FreeCAD **v1.0.2 stable** or later

### 📦 Alternative Installation
This macro is bundled with the Detessellate Workbench, but can also be manually installed separately.

1. Place `ConstrainAllPointOnPoint.py` in your FreeCAD macros directory
2. Restart FreeCAD or refresh the macro list

### 🚀 Usage

1. Open a sketch in edit mode (double-click the sketch)
2. Run the macro
3. All coincident vertices within tolerance are automatically constrained

### 🔧 How It Works

1. **Vertex Collection**: Gathers all vertices from sketch geometry (construction and normal)
2. **Proximity Grouping**: Groups vertices within tolerance (default 100µm)
3. **Intelligent Filtering**: 
   - Excludes B-spline endpoints (Pos1, Pos2) - managed via construction circles
   - Excludes control point circles not at curve endpoints
   - Keeps only construction circles at actual B-spline endpoints
4. **Anchor Selection**: Chooses vertices with existing constraints as anchors
5. **Transitive Checking**: Builds constraint graph to avoid redundant connections
6. **Batch Application**: Adds all constraints in a single transaction

### 📜 Changelog

- **v1.0.0** (2026-01-31) - Initial release with B-spline endpoint exclusion, control point filtering, and redundancy elimination
