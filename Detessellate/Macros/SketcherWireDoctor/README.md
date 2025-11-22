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

## Installation

1. Download all 5 files to your FreeCAD macro directory.
2. Run `SketcherWireDoctor_Main.py`.
3. Optionally, download the icon for use in custom toolbar in Sketcher workbench.

## Usage

1. Open a sketch for editing
2. Run the macro (appears as dockable panel)
3. Click "Re-Analyze Sketch"
4. Use tabs to view and fix different issue types

## Requirements

- FreeCAD 1.0.1+ (tested on Windows 11)
- Active sketch in edit mode

## History

- v250728-1915: Initial commit.
