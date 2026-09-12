# SketcherWireDoctor

FreeCAD macro for detecting and fixing sketch problems that prevent 3D operations that lead to the dreaded Wire Not Closed error in PartDesign workbench.

<img width="64" height="64" alt="SketcherWireDoctor_Icon" src="https://github.com/user-attachments/assets/21fd3989-5f19-4127-a680-0e17d17534ec" />


## What It Does

Finds problems automatically and assists the user in fixing them through a 4-tab docker interface:
- **Zero-Length Lines** - Zero length lines are discovered and optionally deleted.
- **Duplicate Geometry** - Overlapping geometry are discovered and optionally deleted.
- **Non-Coincident Vertices** - Finds missing coincident constraints and can apply all or one at a time.
- **Wire Topology Issues** - Finds floating, intersecting, T-junctions, bridges, & subdivisions, and facilitates conversion to construction geometry. 

## Features

- Visual highlighting of candidate edges and vertices with 8 highlight color choices
- One-click automated fixes
- Safe operations with rollback (transaction-enabled undo enries)
- Works with B-splines and complex sketches

## Alternative Installation

This macro is bundled with the Detessellate Workbench, but can also be installed as a standalone macro. Since it's split across multiple files, all 5 must be installed together.

1. Download all 5 files (`SketcherWireDoctor_Main.py`, `SketcherWireDoctor_Tab1.py`, `SketcherWireDoctor_Tab2.py`, `SketcherWireDoctor_Tab3.py`, `SketcherWireDoctor_Tab4.py`) from this repository.
2. Place them in your FreeCAD macros directory:
   - `Macro → Macros...` → **User macros location**, or
   - `Edit → Preferences → Python → Macro → Macro path`
3. Restart FreeCAD, or click **Refresh** in the Macro dialog.
4. Run `SketcherWireDoctor_Main.py` from the Macro dialog.
5. (Optional) Copy the matching icon from `Resources/Icons/SketcherWireDoctor.svg` for use as a custom toolbar button.

## Usage

1. Open a sketch for editing
2. Run the macro (appears as dockable panel)
3. Click "Re-Analyze Sketch"
4. Use tabs to view and fix different issue types

## Requirements

- FreeCAD 1.0.1+ (tested on Windows 11)
- Active sketch in edit mode

## 📜 Changelog
- **v1.0.0** (2026-04-01) - Added support for external geometry for duplicate detection & repair for FreeCAD v1.1. 
- v250728-1915: Initial commit.
