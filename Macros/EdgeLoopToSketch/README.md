# FreeCAD EdgeLoopToSketch

<img width="128" height="128" alt="EdgeLoopToSketch" src="https://github.com/user-attachments/assets/2ed2553e-633c-4be4-aacc-32f5d4691915" />

A FreeCAD macro that converts selected 3D edges into parametric sketches while preserving curve types (lines, arcs, circles, B-splines).

## Why Use This Tool?

When working with imported STEP files or existing 3D geometry, you often need to extract edge loops as parametric sketches for:
- Re-parametrizing imported CAD models
- Creating profiles for new features based on existing geometry
- Converting 3D edge loops to editable 2D sketches
- Extracting reference geometry for design modifications

Unlike tools that convert everything to line segments, EdgeLoopToSketch **preserves the original curve types**, giving you fully parametric sketches with circles, arcs, and B-splines intact.

## Features

- **Curve Type Preservation**: Maintains lines, arcs, circles, and B-splines from source geometry
- **Coplanarity Validation**: Automatically verifies selected edges are coplanar before conversion
- **Smart Single-Edge Handling**: Can work with single circles, arcs, or planar B-splines
- **Flexible Placement**: Choose between standalone sketch (Part) or PartDesign Body attachment
- **Automatic Constraints**: Adds coincident constraints at shared vertices
- **Full Undo Support**: Wrapped in FreeCAD transaction for easy undo

## Usage

### Basic Workflow

1. **Select edges** from a 3D object (typically using [EdgeLoopSelector](../EdgeLoopSelector) to select complete loops)
2. **Run the macro**: `Macro → Macros... → EdgeLoopToSketch → Execute`
3. **Choose destination**: Standalone sketch, new Body, or existing Body
4. **Result**: Parametric sketch with preserved curve types

### Selection Requirements

**For multiple edges:**
- Select 2 or more coplanar edges
- All edges must be from the same object
- Macro validates coplanarity automatically

**For single edge:**
- Circle or arc (defines its own plane via axis)
- Planar B-spline (all control points coplanar)
- Single lines require a second edge to define the plane

### Typical Use Cases

**Re-parametrizing STEP Files:**
1. Import STEP file
2. Use EdgeLoopSelector to select desired loops
3. Run EdgeLoopToSketch
4. Use resulting sketch for Pad, Pocket, or other features

**Extracting Face Boundaries:**
1. Select edges forming a face boundary
2. Run EdgeLoopToSketch
3. Get editable sketch of that boundary

**Converting Circular Features:**
1. Select a circular edge (hole, boss, etc.)
2. Run EdgeLoopToSketch
3. Get parametric circle sketch you can dimension/constrain

## How It Works

### Curve Type Detection

The macro analyzes each edge's geometry type and creates appropriate sketch elements:

- **Line edges** → Sketch line segments
- **Full circles** → Sketch circles (center + radius)
- **Circular arcs** → Sketch arcs (3-point definition)
- **B-spline curves** → Sketch B-splines (preserves poles, knots, degree)

### Coplanarity Validation

**For multiple edges:**
1. Collects all vertex points from selected edges
2. Finds 3 non-collinear points to define reference plane
3. Verifies all vertices lie on that plane (within 1e-6 tolerance)
4. Proceeds if validation passes

**For single edges:**
- **Circles/Arcs**: Uses the curve's axis as plane normal
- **Planar B-splines**: Checks if all control points are coplanar
- **Other types**: Requires additional edge(s) to define plane

### Sketch Placement

The macro creates a sketch with proper 3D placement matching the original edge geometry:
1. Calculates plane normal and center point
2. Creates placement transformation
3. Transforms all geometry to sketch local coordinates
4. Maintains spatial relationship to original geometry

## Installation

This macro is bundled with the [Detessellate Workbench](https://github.com/yourusername/Detessellate), but can also be installed separately.

### Manual Installation

1. Download `EdgeLoopToSketch.py`
2. In FreeCAD, go to `Macro → Macros...`
3. Click `User macros location` to open your macro folder
4. Copy the downloaded file to this folder
5. Restart FreeCAD or click `Refresh` in the Macro dialog

## Workflow Integration

EdgeLoopToSketch works perfectly with [EdgeLoopSelector](../EdgeLoopSelector):

1. **EdgeLoopSelector** → Select complete edge loops (handles coplanar multi-loop selection)
2. **EdgeLoopToSketch** → Convert selected edges to parametric sketch

This two-step workflow makes it easy to extract and re-parametrize complex geometry from imported CAD files.

## Limitations

- **Coplanar edges only**: All selected edges must lie on the same plane
- **Single object**: All edges must be from the same FreeCAD object
- **Supported curve types**: Lines, circles, arcs, B-splines (other types converted to lines)
- **3D B-splines**: Non-planar B-splines cannot be converted (error shown)

## Requirements

- FreeCAD 1.0.2 or later (tested on 1.0.2 and 1.1 dev builds)
- Part and Sketcher workbenches (built-in)

## Comparison with CoplanarSketch

| Feature | EdgeLoopToSketch | CoplanarSketch |
|---------|------------------|----------------|
| **Primary Use** | STEP/parametric geometry | Tessellated meshes (STL/OBJ) |
| **Curve Preservation** | Yes (arcs, circles, splines) | No (lines only) |
| **Workflow** | Direct conversion | Creates construction lines for manual tracing |
| **Best For** | Re-parametrizing CAD imports | Detessellating mesh geometry |

## Troubleshooting

**"Selected edges are not coplanar"**
- Verify edges actually lie on same plane
- Use EdgeLoopSelector to ensure loop selection from same face
- Check tolerance if edges are nearly but not exactly coplanar

**"Single edge of type 'Line' cannot define a plane"**
- Lines need a second edge to define the plane
- Select at least 2 edges

**"B-spline is non-planar (3D curve)"**
- The B-spline curves through 3D space
- Select additional coplanar edges to define the working plane
- The 3D B-spline will still be projected/converted

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Changelog

- **Version 1.0.0** (2025-12-07): Initial release
  - Preserves curve types (lines, arcs, circles, B-splines)
  - Coplanarity validation
  - Single circle/arc/planar B-spline support
  - Standalone and Body sketch placement options
  - Automatic coincident constraints


## Related Tools

- **[EdgeLoopSelector](../EdgeLoopSelector)** - Select complete edge loops for conversion
- **[CoplanarSketch](../CoplanarSketch)** - Create construction line sketches from tessellated geometry
- **[SketchReProfile](../SketchReProfile)** - Trace parametric geometry over construction lines
