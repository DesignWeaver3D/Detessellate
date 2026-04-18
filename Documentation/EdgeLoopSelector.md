# FreeCAD Edge Loop Selector

<img width="64" height="64" alt="EdgeLoopSelector" src="https://github.com/user-attachments/assets/e2de31f8-bf12-46f7-b5eb-5a7271ec598b" />

A FreeCAD macro that automatically selects all edges in connected loops from your edge selection. Works with both sketches and solid shapes.

## Why Use This Macro?
This macro bridges a workflow gap in FreeCAD. While you can select entire faces, sometimes you need specific edge loops from complex geometry:
- **Sketches**: Easily separate multiple profiles in a single sketch for different Pad/Pocket operations
- **3D Models**: Select specific holes or boundaries on a face without selecting unwanted loops
- **Efficiency**: Faster than manually clicking dozens of edges in complex geometry

## Features
- **Smart Loop Detection**: Select one or more edges, get entire connected loop(s) or wire(s)
- **Multi-Loop Selection**: Select edges from multiple loops to select all those loops at once
- **Multi-Edge Support**: Select multiple edges from the same loop for confirmation
- **Sketch Support**: Works with sketches in both standalone and Body contexts
- **Wire Support**: Works with SubShapeBinder wires created from binding sketches
- **Solid Support**: Works with edges on faces of solid models with coplanarity validation
- **Automatic Detection**: Intelligently determines if you're working with a sketch or solid

## Usage

### For Sketches

1. Make your sketch visible in the 3D view (if it's hidden by features)
2. Select one or more edges from desired connected loops
3. Run the macro by clicking its tool button or Detessellate > EdgeLoopSelector
4. All edges in the connected loop will be selected

### For Wires

1. Select one or more edges from a wire or multiple wires
3. Run the macro by clicking its tool button or Detessellate > EdgeLoopSelector
4. All edges in all wires will be selected. If any edges or wires in the new selection are not coplanar, a warning will print in the Report View.

### For Solid Models

1. Select two or more coplanar edges
2. Run the macro by clicking its tool button or Detessellate > EdgeLoopSelector
3. All coplanar loops for selected edges will be selected

## How It Works

### Sketch Mode
The macro builds a connectivity graph of all edges in the sketch by analyzing shared vertices. It then uses depth-first search to find connected components (loops). For each selected edge, it identifies which loop contains that edge. All unique loops are collected and their edges are selected, allowing you to select multiple disconnected loops within the same sketch by selecting one or more edges from each desired loop.

### Solid Mode
For 3D objects, at least two edges must be selected to define a plane. The macro first validates that all selected edges are coplanar by collecting their vertex points and computing a reference plane. It then filters each selected edge's parent faces to only those that are coplanar with the validated plane. For each selected edge, the macro identifies which wire (closed loop) on the coplanar faces contains that edge. All unique loops are collected and their edges are selected, allowing you to select multiple coplanar loops (such as an outer boundary and specific holes on a face) by selecting edges from each desired loop.

## Alternative Installation
This macro is bundled with the Detessellate Workbench, but can also be manually installed separately.

1. Download `SelectEdgeLoop.py`
2. In FreeCAD, go to `Macro → Macros...`
3. Click `User macros location` to open your macro folder
4. Copy the downloaded file to this folder
5. Restart FreeCAD or click `Refresh` in the Macro dialog

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Known Issues

- **Visual Highlighting Bug**: When selecting edges from sketches that are part of a PartDesign Body feature, the edges are selected internally and will work for operations (Pad, Pocket, etc.), but the visual highlighting may not appear in the 3D view. The Report Viewer will confirm the edges were selected.
  - **Workaround**: The selection is still valid - proceed with your operation normally.

## Requirements

- FreeCAD 1.0.2 or later (tested on 1.0.2 and 1.1 dev builds)

## Changelog
- **2.1.0** (2025-12-11) Updated to handle SubShapeBinder and wire edges.
- **2.0.0** (2025-12-07) Updated to allow multiple loop selection based on all selected edges from sketch or solid object
- **1.0.0** (2025-11-12) Initial release supports single loop selection from selected edge or edges in sketch or solid object
