__version__ = "3.0.0"
__date__ = "2026-01-23"
__author__ = "DesignWeaver3D"

# EdgeLoopToSketch - Enhanced with DraftGeomUtils.orientEdge integration
#
# This version uses DraftGeomUtils.orientEdge() for universal edge type conversion,
# which provides automatic support for:
#   - Lines
#   - Circles and Arcs
#   - Ellipses and Arcs of Ellipses
#   - B-Splines
#   - And any other edge types that FreeCAD's Draft workbench supports
#
# Benefits over manual conversion:
#   - Simpler code (single conversion path)
#   - Automatic support for new edge types
#   - Uses FreeCAD's proven conversion logic
#
# Features:
#   - Automatic plane calculation from selected geometry
#   - Camera-based sketch orientation
#   - Support for face and edge selection
#   - Performance timing (excludes user dialog time)
#

import FreeCAD
import FreeCADGui
import Part
import Sketcher
import time
from PySide.QtWidgets import QInputDialog, QMessageBox
import DraftGeomUtils
from collections import Counter

def edge_loop_to_sketch():
    """
    Creates a parametric sketch from selected edges or faces, preserving curve types.

    Supports:
    - Edge selection: Calculates plane from edge geometry
    - Face selection: Extracts all edges and calculates plane from vertices
    - Mixed selection: Combines edges from faces and individual edges

    Sketch orientation is determined by camera position relative to calculated plane.
    Reports performance timing for all processing phases.
    """
    doc = FreeCAD.ActiveDocument
    if not doc:
        FreeCAD.Console.PrintError("Error: No active document.\n")
        return

    selection = FreeCADGui.Selection.getSelectionEx()
    if not selection:
        FreeCAD.Console.PrintError("Error: No selection.\n")
        return

    # Collect selected edges and faces
    obj = selection[0].Object
    selected_edges = []
    selected_faces = []

    for sel_ex in selection:
        if sel_ex.Object.Name != obj.Name:
            FreeCAD.Console.PrintError("Error: All selections must be from the same object.\n")
            return
        for sub_name in sel_ex.SubElementNames:
            if sub_name.startswith("Edge"):
                edge_idx = int(sub_name[4:]) - 1
                selected_edges.append(obj.Shape.Edges[edge_idx])
            elif sub_name.startswith("Face"):
                face_idx = int(sub_name[4:]) - 1
                selected_faces.append(obj.Shape.Faces[face_idx])

    # Extract edges from selected faces
    if selected_faces:
        for face in selected_faces:
            selected_edges.extend(face.Edges)
        # Remove duplicates while preserving order
        seen = set()
        unique_edges = []
        for edge in selected_edges:
            edge_id = id(edge)
            if edge_id not in seen:
                seen.add(edge_id)
                unique_edges.append(edge)
        selected_edges = unique_edges

    if not selected_edges:
        FreeCAD.Console.PrintError("Error: No valid edges or faces selected.\n")
        return

    FreeCAD.Console.PrintMessage(f"Processing {len(selected_edges)} edges...\n")

    doc.openTransaction("Create Sketch from Edge Loop")

    try:
        # Start performance timing (excludes user dialogs)
        perf_start = time.perf_counter()

        tolerance = 1e-6
        plane_point = None
        plane_normal = None

        # Get camera position at macro start
        phase_start = time.perf_counter()
        camera_position = FreeCADGui.ActiveDocument.ActiveView.getCameraNode().position.getValue()
        camera_position = FreeCAD.Vector(camera_position[0], camera_position[1], camera_position[2])

        # Calculate plane from geometry
        # Special case: single edge that defines its own plane
        if len(selected_edges) == 1:
            edge = selected_edges[0]
            curve_type = type(edge.Curve).__name__

            if curve_type == 'Circle':
                # Circle defines its own plane via its axis
                circle = edge.Curve
                plane_normal = circle.Axis.normalize()
                plane_point = circle.Center
                FreeCAD.Console.PrintMessage("Using plane from circle geometry.\n")
            elif curve_type == 'BSplineCurve':
                # Check if B-spline is planar by examining control points
                bspline = edge.Curve
                poles = bspline.getPoles()

                if len(poles) < 3:
                    raise Exception("B-spline does not have enough control points to define a plane.")

                # Find 3 non-collinear poles to define plane
                plane_point = poles[0]

                for i in range(1, len(poles)):
                    v1 = poles[i] - plane_point
                    for j in range(i + 1, len(poles)):
                        v2 = poles[j] - plane_point
                        cross = v1.cross(v2)
                        if cross.Length > tolerance:
                            plane_normal = cross.normalize()
                            break
                    if plane_normal:
                        break

                if not plane_normal:
                    raise Exception("B-spline control points are collinear and cannot define a plane.")

                # Verify all poles are coplanar
                for pole in poles:
                    distance = abs((pole - plane_point).dot(plane_normal))
                    if distance > tolerance:
                        raise Exception("B-spline is non-planar (3D curve). Select additional edges to define the plane.")

                FreeCAD.Console.PrintMessage("Using plane from planar B-spline geometry.\n")
            else:
                raise Exception(f"Single edge of type '{curve_type}' cannot define a plane. Select at least 2 edges.")

        # Multiple edges: validate coplanarity
        if len(selected_edges) >= 2:
            unique_points = []
            all_points = []

            for edge in selected_edges:
                for vertex in edge.Vertexes:
                    pt = vertex.Point
                    all_points.append(pt)
                    if not any(pt.isEqual(existing, tolerance) for existing in unique_points):
                        unique_points.append(pt)

            if len(unique_points) < 3:
                raise Exception("Selected edges do not provide enough unique points to define a plane.")

            # Find 3 non-collinear points to define plane
            plane_point = unique_points[0]
            point_A = None
            point_B = None
            point_C = None

            for i in range(1, len(unique_points)):
                v1 = unique_points[i] - plane_point
                for j in range(i + 1, len(unique_points)):
                    v2 = unique_points[j] - plane_point
                    cross = v1.cross(v2)
                    if cross.Length > tolerance:
                        # Found three non-collinear points
                        point_A = plane_point
                        point_B = unique_points[i]
                        point_C = unique_points[j]

                        # Canonicalize the winding by sorting vertices
                        # This ensures consistent triangle orientation regardless of iteration order
                        sorted_pts = sorted([point_A, point_B, point_C], key=lambda v: (v.x, v.y, v.z))
                        point_A, point_B, point_C = sorted_pts[0], sorted_pts[1], sorted_pts[2]

                        # Recalculate plane normal from canonicalized winding
                        v1_canonical = point_B - point_A
                        v2_canonical = point_C - point_A
                        plane_normal = v1_canonical.cross(v2_canonical).normalize()
                        break
                if plane_normal:
                    break

            if not plane_normal:
                raise Exception("Selected edges are collinear and cannot define a plane.")

            # Verify all points are coplanar
            for pt in all_points:
                distance = abs((pt - plane_point).dot(plane_normal))
                if distance > tolerance:
                    raise Exception("Selected edges are not coplanar.")

        # Calculate placement
        if len(selected_edges) == 1:
            # For single circle, use its center
            center = plane_point
        else:
            # For multiple edges, use centroid of all points
            center = sum(all_points, FreeCAD.Vector()).multiply(1.0 / len(all_points))

        plane_time = time.perf_counter() - phase_start

        # Orient plane normal toward camera using determinant test
        # This is more robust than dot product as it uses orientation rather than projection
        if plane_normal:
            if len(selected_edges) >= 2:
                # Use determinant/tetrahedron volume test for multiple edges
                # Calculate signed volume of tetrahedron formed by A, B, C (plane points) and P (camera)
                AB = point_B - point_A
                AC = point_C - point_A
                AP = camera_position - point_A

                # Determinant = AB · (AC × AP)
                determinant = AB.dot(AC.cross(AP))

                if determinant < 0:
                    plane_normal = -plane_normal
                    FreeCAD.Console.PrintMessage("Sketch normal flipped using determinant test.\n")
                else:
                    FreeCAD.Console.PrintMessage("Sketch normal oriented using determinant test.\n")
            else:
                # Single edge case - use simple dot product with plane point
                plane_to_camera = camera_position - plane_point
                if plane_normal.dot(plane_to_camera) < 0:
                    plane_normal = -plane_normal
                FreeCAD.Console.PrintMessage("Sketch normal oriented toward camera.\n")

        placement = create_sketch_placement(plane_normal, center)

        # Calculate pre-dialog timing
        pre_dialog_time = time.perf_counter() - perf_start
        FreeCAD.Console.PrintMessage(f"Pre-dialog processing: {pre_dialog_time:.3f}s\n")

        # Show destination dialog (not timed - user interaction)
        choice = show_destination_dialog()
        if not choice:
            doc.abortTransaction()
            FreeCAD.Console.PrintMessage("Sketch creation cancelled.\n")
            return

        # Restart timer after dialog for geometry phase
        phase_start = time.perf_counter()

        # Create sketch based on destination choice
        if choice["type"] == "standalone":
            sketch, geometry_summary, constraint_summary = create_standalone_sketch(doc, placement, selected_edges)
        elif choice["type"] == "new_body":
            body = doc.addObject("PartDesign::Body", "Body")
            sketch, geometry_summary, constraint_summary = create_body_sketch(doc, body, placement, selected_edges)
        elif choice["type"] == "existing_body":
            body = doc.getObject(choice["body_name"])
            if not body:
                raise Exception(f"Body {choice['body_name']} not found.")
            sketch, geometry_summary, constraint_summary = create_body_sketch(doc, body, placement, selected_edges)

        geometry_time = time.perf_counter() - phase_start

        doc.recompute()

        # Select the new sketch and fit view
        FreeCADGui.Selection.clearSelection()
        FreeCADGui.Selection.addSelection(sketch)
        FreeCADGui.activeDocument().activeView().viewAxonometric()
        FreeCADGui.activeDocument().activeView().fitAll()

        # Calculate total time (sum of all phases, excluding dialog)
        total_time = plane_time + geometry_time

        # Performance report
        doc.commitTransaction()
        FreeCAD.Console.PrintMessage(f"\nSketch created successfully with {len(selected_edges)} edges.\n")
        FreeCAD.Console.PrintMessage(f"\n{'='*60}\n")
        FreeCAD.Console.PrintMessage(f"GEOMETRY SUMMARY\n")
        FreeCAD.Console.PrintMessage(f"{'='*60}\n")
        for geom_type, count in sorted(geometry_summary.items()):
            FreeCAD.Console.PrintMessage(f"  {geom_type:20s} {count:>3d}\n")
        FreeCAD.Console.PrintMessage(f"\n{'='*60}\n")
        FreeCAD.Console.PrintMessage(f"CONSTRAINT SUMMARY\n")
        FreeCAD.Console.PrintMessage(f"{'='*60}\n")
        for constraint_type, count in sorted(constraint_summary.items()):
            if count > 0:  # Only show non-zero counts
                FreeCAD.Console.PrintMessage(f"  {constraint_type:20s} {count:>3d}\n")
        FreeCAD.Console.PrintMessage(f"\n{'='*60}\n")
        FreeCAD.Console.PrintMessage(f"PERFORMANCE REPORT\n")
        FreeCAD.Console.PrintMessage(f"{'='*60}\n")
        FreeCAD.Console.PrintMessage(f"  Plane calculation:      {plane_time:>8.3f}s\n")
        FreeCAD.Console.PrintMessage(f"  Geometry & constraints: {geometry_time:>8.3f}s\n")
        FreeCAD.Console.PrintMessage(f"  {'-'*58}\n")
        FreeCAD.Console.PrintMessage(f"  Total processing:       {total_time:>8.3f}s\n")
        FreeCAD.Console.PrintMessage(f"  (Excludes user dialog time)\n")
        FreeCAD.Console.PrintMessage(f"{'='*60}\n")

    except Exception as e:
        doc.abortTransaction()
        FreeCAD.Console.PrintError(f"Sketch creation failed: {e}\n")
        QMessageBox.critical(None, "Error", f"Sketch creation failed:\n{str(e)}")


def create_sketch_placement(normal, center):
    """Create placement for sketch from normal and center point."""
    normal = normal.normalize() if normal.Length > 1e-6 else FreeCAD.Vector(0, 0, 1)
    z_axis = FreeCAD.Vector(0, 0, 1)

    if abs(normal.dot(z_axis)) > 0.999:
        rotation = FreeCAD.Rotation() if normal.z > 0 else FreeCAD.Rotation(FreeCAD.Vector(1, 0, 0), 180)
    else:
        rotation = FreeCAD.Rotation(z_axis, normal)

    return FreeCAD.Placement(center, rotation)


def show_destination_dialog():
    """Show dialog for sketch destination choice."""
    doc = FreeCAD.ActiveDocument
    body_names = [o.Name for o in doc.Objects if o.isDerivedFrom("PartDesign::Body")]
    options = ["<Standalone (Part Workbench)>", "<Create New Body (PartDesign)>"] + body_names

    item, ok = QInputDialog.getItem(
        FreeCADGui.getMainWindow(),
        "Sketch Destination",
        "Choose where to create the sketch:",
        options, 0, False
    )

    if not ok or not item:
        return None

    if item == "<Standalone (Part Workbench)>":
        return {"type": "standalone"}
    elif item == "<Create New Body (PartDesign)>":
        return {"type": "new_body"}
    else:
        return {"type": "existing_body", "body_name": item}


def create_standalone_sketch(doc, placement, edges):
    """Create standalone sketch in Part workbench. Returns (sketch, geometry_summary, constraint_summary)."""
    sketch = doc.addObject("Sketcher::SketchObject", "Sketch")
    sketch.Placement = placement

    inverse_placement = sketch.getGlobalPlacement().inverse()
    geometry_summary, constraint_summary = add_geometry_to_sketch(sketch, edges, inverse_placement)

    return sketch, geometry_summary, constraint_summary


def create_body_sketch(doc, body, placement, edges):
    """Create sketch attached to PartDesign body. Returns (sketch, geometry_summary, constraint_summary)."""
    sketch = doc.addObject("Sketcher::SketchObject", "Sketch")

    # Add sketch to body
    body.ViewObject.dropObject(sketch, None, '', [])

    # Set up attachment to body origin
    sketch.AttachmentSupport = [(body.Origin.OriginFeatures[0], '')]
    sketch.MapMode = 'ObjectXY'
    sketch.AttachmentOffset.Base = placement.Base
    sketch.AttachmentOffset.Rotation = placement.Rotation
    sketch.Placement = FreeCAD.Placement()

    doc.recompute()  # Resolve attachment before adding geometry

    inverse_placement = sketch.getGlobalPlacement().inverse()
    geometry_summary, constraint_summary = add_geometry_to_sketch(sketch, edges, inverse_placement)

    return sketch, geometry_summary, constraint_summary


def add_geometry_to_sketch(sketch, edges, inverse_placement):
    """
    Add geometry to sketch, preserving curve types.
    Uses DraftGeomUtils.orientEdge for universal edge type support.
    Returns summary statistics of geometry added.
    """

    geo_indices = []  # Track created geometry indices with original edges
    geometry_summary = Counter()  # Track geometry types added

    for edge in edges:
        try:
            curve_type = type(edge.Curve).__name__

            # Transform edge to sketch local coordinate system
            edge_copy = edge.copy()
            edge_copy.transformShape(inverse_placement.toMatrix())

            # Use DraftGeomUtils.orientEdge to convert any edge type to sketch geometry
            # This handles: Line, Circle, Arc, Ellipse, ArcOfEllipse, BSpline, and more
            sketch_geom = DraftGeomUtils.orientEdge(edge_copy, FreeCAD.Vector(0, 0, 1), make_arc=True)

            # Add to sketch
            geo_index = sketch.addGeometry(sketch_geom, False)

            if geo_index is not None:
                geo_indices.append((geo_index, edge))
                geometry_summary[type(sketch_geom).__name__] += 1

        except Exception as e:
            FreeCAD.Console.PrintWarning(f"Failed to add edge ({curve_type}): {e}\n")

    # Phase 2: Build constraint data from actual created geometry and apply constraints
    constraint_summary = build_constraint_data(sketch, geo_indices)

    return geometry_summary, constraint_summary

def build_constraint_data(sketch, geo_indices):
    """
    Query actual geometry from sketch and build accurate constraint maps.
    This approach is geometry-agnostic and handles any future types.
    Returns summary of constraints added by type.
    """
    constraint_summary = Counter()

    # Step 1: Add coincident constraints FIRST (while geometry can still move)
    coincident_count = add_vertex_coincident_constraints(sketch, geo_indices)
    constraint_summary['Coincident'] = coincident_count

    # Step 2: Add construction lines from arc/circle centers to origin
    construction_count, block_count = add_center_construction_lines(sketch, geo_indices)
    constraint_summary['Construction Lines'] = construction_count
    constraint_summary['Block'] = block_count

    # Step 3: Add radius constraints to all arcs and circles
    radius_count = add_radius_constraints(sketch, geo_indices)
    constraint_summary['Radius'] = radius_count

    # Step 4: Add reference distance constraints to all lines
    distance_count = add_line_distance_constraints(sketch, geo_indices)
    constraint_summary['Distance (ref)'] = distance_count

    return constraint_summary


def add_center_construction_lines(sketch, geo_indices):
    """
    Add construction lines from arc/circle/ellipse centers to sketch origin.
    These lines are coincident-constrained at both ends and prevent geometry from flipping.
    Returns (construction_line_count, block_constraint_count).
    """
    origin = FreeCAD.Vector(0, 0, 0)
    construction_line_count = 0
    construction_line_indices = []
    tolerance = 1e-6

    for geo_idx, original_edge in geo_indices:
        geo = sketch.Geometry[geo_idx]

        # Check if this geometry has a center point (arc, circle, or ellipse)
        if isinstance(geo, (Part.ArcOfCircle, Part.Circle, Part.Ellipse, Part.ArcOfEllipse)):
            center = geo.Center
            distance_to_origin = center.Length

            # Only create construction line if center is not at origin
            if distance_to_origin > tolerance:
                # Add construction line from center to origin
                line = Part.LineSegment(center, origin)
                line_idx = sketch.addGeometry(line, True)  # True = construction mode

                # Coincident constraint: line start to geometry center (point 3)
                sketch.addConstraint(Sketcher.Constraint('Coincident', line_idx, 1, geo_idx, 3))

                # Coincident constraint: line end to origin
                sketch.addConstraint(Sketcher.Constraint('Coincident', line_idx, 2, -1, 1))

                construction_line_indices.append(line_idx)
                construction_line_count += 1
            else:
                # Center is already at origin - just constrain it directly
                sketch.addConstraint(Sketcher.Constraint('Coincident', geo_idx, 3, -1, 1))

    # Block constrain all construction lines to prevent rotation/movement
    block_count = 0
    for line_idx in construction_line_indices:
        sketch.addConstraint(Sketcher.Constraint('Block', line_idx))
        block_count += 1

    return construction_line_count, block_count


def add_radius_constraints(sketch, geo_indices):
    """
    Add radius constraints to circles and arcs to lock their size.
    Note: Ellipses are skipped because they require complex construction line-based
    constraints for major/minor axes that need manual setup.
    Returns count of radius constraints added.
    """
    radius_constraint_count = 0

    for geo_idx, original_edge in geo_indices:
        geo = sketch.Geometry[geo_idx]

        # Check if this is an arc or circle
        if isinstance(geo, Part.ArcOfCircle):
            radius = geo.Radius
            sketch.addConstraint(Sketcher.Constraint('Radius', geo_idx, radius))
            radius_constraint_count += 1

        elif isinstance(geo, Part.Circle):
            radius = geo.Radius
            sketch.addConstraint(Sketcher.Constraint('Radius', geo_idx, radius))
            radius_constraint_count += 1

        # Skip ellipses - they require construction lines for major/minor axes
        # Users can manually add these constraints if needed
        elif isinstance(geo, (Part.Ellipse, Part.ArcOfEllipse)):
            pass  # Silently skip

    return radius_constraint_count


def add_vertex_coincident_constraints(sketch, geo_indices):
    """
    Add coincident constraints at shared vertices by matching actual vertex coordinates.
    Uses sketch.getPoint() to get the actual vertex positions, not StartPoint/EndPoint.
    Returns count of coincident constraints added.
    """
    tolerance = 1e-6
    vertex_map = {}  # Maps rounded coordinates to list of (geo_idx, vertex_id)

    for geo_idx, original_edge in geo_indices:
        geo = sketch.Geometry[geo_idx]

        # Skip full circles (no vertices to constrain)
        if isinstance(geo, Part.Circle):
            continue

        # Get vertices from geometry that has them
        if hasattr(geo, 'StartPoint') and hasattr(geo, 'EndPoint'):
            # Use sketch.getPoint() to get actual vertex coordinates
            # StartPoint/EndPoint are NOT reliable for vertex numbering!
            try:
                vertex1 = sketch.getPoint(geo_idx, 1)
                vertex2 = sketch.getPoint(geo_idx, 2)

                # Round coordinates for matching
                v1_key = (round(vertex1.x, 6), round(vertex1.y, 6), round(vertex1.z, 6))
                v2_key = (round(vertex2.x, 6), round(vertex2.y, 6), round(vertex2.z, 6))

                # Store in vertex map
                vertex_map.setdefault(v1_key, []).append((geo_idx, 1))
                vertex_map.setdefault(v2_key, []).append((geo_idx, 2))

            except Exception as e:
                FreeCAD.Console.PrintWarning(f"Could not get points for geometry[{geo_idx}]: {e}\n")

    # Find and create coincident constraints for matching vertices
    constraint_count = 0

    for coord_key, vertex_list in vertex_map.items():
        if len(vertex_list) > 1:
            # Create constraints between all pairs (for now just first to second)
            base = vertex_list[0]
            other = vertex_list[1]

            try:
                sketch.addConstraint(Sketcher.Constraint('Coincident', base[0], base[1], other[0], other[1]))
                constraint_count += 1
            except Exception as e:
                FreeCAD.Console.PrintWarning(f"Failed to constrain geo[{base[0]}] vertex {base[1]} to geo[{other[0]}] vertex {other[1]}: {e}\n")

    return constraint_count


def add_line_distance_constraints(sketch, geo_indices):
    """
    Add reference distance constraints to all straight lines.
    Reference constraints are non-driving (informational only).
    Returns count of distance constraints added.
    """
    distance_constraint_count = 0

    for geo_idx, original_edge in geo_indices:
        geo = sketch.Geometry[geo_idx]

        # Check if this is a line segment
        if isinstance(geo, Part.LineSegment):
            # Calculate distance
            p1 = sketch.getPoint(geo_idx, 1)
            p2 = sketch.getPoint(geo_idx, 2)
            distance = p1.distanceToPoint(p2)

            # Add reference distance constraint (driving=False makes it reference-only)
            constraint = Sketcher.Constraint('Distance', geo_idx, distance)
            constraint_idx = sketch.addConstraint(constraint)

            # Set constraint to reference mode (non-driving)
            sketch.setDriving(constraint_idx, False)
            distance_constraint_count += 1

    return distance_constraint_count


# Run the macro
edge_loop_to_sketch()
