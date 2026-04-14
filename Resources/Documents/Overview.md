# Detessellate

A FreeCAD workbench of algorithm-assisted tools for reverse engineering imported geometry including mesh models (STL, OBJ, 3MF), point cloud data, and non-parametric STEP solids.

## Use Cases
- Convert STL/OBJ/3MF meshes to parametric CAD models
- Generate sketches from scanned point cloud data
- Rebuild imported STEP files with sketch-based features
- Clean up and constrain sketch geometry

## Tools

### General Tools
- **MeshPlacement** — Recenter and align meshes to origin
- **MeshToBody** — Convert meshes into solids and bodies
- **CoplanarSketch** — Generate construction sketches from coplanar edges on tessellated solids
- **PointPlaneSketch** — Generate sketches from selected points of a PointsObject
- **EdgeLoopSelector** — Select connected edge loops from sketches, faces, or solids
- **EdgeLoopToSketch** — Generate sketches from selected coplanar 3D edges, preserving edge type
- **ReconstructSolid** — Rebuild a non-parametric solid from the underlying faces
- **VarSetUpdate** — Update VarSet variable properties

### Sketcher Tools
- **SketchReProfile** — Rebuild normal geometry profiles from construction sketches
- **ConstrainAllPointOnPoint** — Automatically add coincident constraints to all overlapping vertices
- **SketcherWireDoctor** — Detect and repair sketch wire issues

### PartDesign Tools
- **TopoMatchSelector** — Match and select topology from earlier body features

## Resources
- [GitHub Repository](https://github.com/DesignWeaver3D/Detessellate)
- [FreeCAD Forum Thread](https://forum.freecad.org/viewtopic.php?t=101467)
- [Demo Video](https://www.youtube.com/watch?v=QLw4me9nutA)
