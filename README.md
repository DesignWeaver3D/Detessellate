# FreeCAD Edge Loop Selector

<img width="64" height="64" alt="EdgeLoopSelector" src="https://github.com/user-attachments/assets/e2de31f8-bf12-46f7-b5eb-5a7271ec598b" />

A FreeCAD macro that automatically selects all edges in a connected loop from a single or dual edge selection. Works with both sketches and solid shapes.

## Features

- **Smart Loop Detection**: Select one edge, get the entire connected loop
- **Multi-Edge Support**: Select multiple edges from the same loop to confirm selection
- **Sketch Support**: Works with sketches in both standalone and Body contexts
- **Solid Support**: Works with edges on faces of solid models
- **Automatic Detection**: Intelligently determines if you're working with a sketch or solid

## Installation

### Manual Installation

1. Download `SelectEdgeLoop.py`
2. In FreeCAD, go to `Macro → Macros...`
3. Click `User macros location` to open your macro folder
4. Copy the downloaded file to this folder
5. Restart FreeCAD or click `Refresh` in the Macro dialog


## Usage

### For Sketches

1. Make your sketch visible in the 3D view (if it's hidden by features)
2. Select one or more edges from a connected loop
3. Run the macro: `Macro → Macros... → SelectEdgeLoop → Execute`
4. All edges in the connected loop will be selected

### For Solid Models

1. Select one or more edges from a face
2. Run the macro
3. All edges in that wire/loop will be selected

## Examples

### Example 1: Rectangle in Sketch
- Select one side of a rectangle → All 4 edges selected

### Example 2: Complex Sketch Profile
- Select any edge from an outer profile → Entire outer loop selected
- Select edge from inner cutout → Entire inner loop selected

### Example 3: Face on Solid
- Select one edge of a rectangular face → All edges of that face selected
- Select two edges to ensure the desired face edge loop is selected

## Known Issues

- **Visual Highlighting Bug**: When selecting edges from sketches that are part of a PartDesign Body feature, the edges ARE selected internally and will work for operations (Pad, Pocket, etc.), but the visual highlighting may not appear in the 3D view. The Report Viewer will confirm the edges were selected.
  - **Workaround**: The selection is still valid - proceed with your operation normally.

## Requirements

- FreeCAD 1.0.2 or later (tested on 1.0.2 and 1.1 dev builds)

## How It Works

### Sketch Mode
The macro builds a connectivity graph of all edges in the sketch by analyzing shared vertices. It then uses depth-first search to find connected components (loops) and selects the loop containing your initially selected edge(s).

### Solid Mode
The macro finds the parent face(s) of the selected edge(s), then identifies which wire (closed loop) contains all selected edges, and selects all edges in that wire.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Changelog

### Version 1.0.0 (2025-11-12)
- Initial release
- Support for sketch edge loops
- Support for solid face edge loops
- Automatic sketch vs solid detection
