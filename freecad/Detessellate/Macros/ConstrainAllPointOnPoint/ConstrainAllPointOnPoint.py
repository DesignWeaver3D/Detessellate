# ============================================================================
# Constrain All Point-On-Point - Robust 1-Click Solution
# ============================================================================
# Comprehensive vertex constraint addition using intelligent anchor selection.
# Based on SketcherWireDoctor's proven constraint detection logic.
#
# Key Features:
# - Includes ALL geometry (construction and normal)
# - Excludes B-spline Pos1/Pos2 endpoints (managed via construction circles)
# - Smart B-spline filtering: only keeps construction circles at actual endpoints
# - Excludes control point circles (InternalAlignment but not at endpoints)
# - Smart anchor selection prioritizing existing constraint networks
# - Tolerance-based grouping with comprehensive constraint checking
# - Transaction-safe with rollback protection
# - NO REDUNDANT CONSTRAINTS: Full transitive closure checking
#
# Usage:
#   1. Open a sketch in edit mode (double-click the sketch)
#   2. Run this macro - it executes immediately with no dialogs
# ============================================================================

import FreeCAD as App
import FreeCADGui as Gui
import Sketcher
import math
import time
from collections import defaultdict, deque

# ============================================================================
# CONFIGURATION - Adjust this value for your precision needs
# ============================================================================
# Tolerance for grouping vertices:
#   Ultra-precision (aerospace/medical): 5e-3 to 13e-3   (5-13 µm)
#   High-precision/Tight tolerance:     25e-3            (25 µm)
#   Standard CNC:                       50e-3 to 130e-3  (50-130 µm)
#   3D printing (FDM):                 200e-3 to 500e-3  (200-500 µm)

DEFAULT_TOLERANCE = 50e-3  # 50 micrometers (Standard CNC)

# ============================================================================

def format_distance(distance_m):
    """Format distance for human-readable display."""
    if distance_m < 1e-6:
        return f"{distance_m*1e9:.1f}nm"
    elif distance_m < 1e-3:
        return f"{distance_m*1e6:.2f}µm"
    elif distance_m < 1:
        return f"{distance_m*1e3:.2f}mm"
    else:
        return f"{distance_m:.3f}m"


def get_sketch():
    """Get the currently active sketch in edit mode."""
    sketch = None

    # Method 1: Try getting from edit mode (most reliable)
    try:
        if Gui.ActiveDocument:
            edit_obj = Gui.ActiveDocument.getInEdit()
            if edit_obj and hasattr(edit_obj, 'Object'):
                obj = edit_obj.Object
                if hasattr(obj, 'TypeId') and 'Sketcher::SketchObject' in obj.TypeId:
                    sketch = obj
    except:
        pass

    # Method 2: Try getActiveObject (fallback)
    if not sketch:
        try:
            if Gui.ActiveDocument:
                active_view = Gui.ActiveDocument.ActiveView
                if hasattr(active_view, 'getActiveObject'):
                    sketch = active_view.getActiveObject('sketch')
        except:
            pass

    if not sketch:
        from PySide import QtGui
        QtGui.QMessageBox.warning(None, "No Active Sketch",
            "Please open a sketch in edit mode first.\n\n"
            "Double-click a sketch to enter edit mode, then run this macro.")
        return None

    return sketch


def collect_all_vertices(sketch):
    """
    Collect comprehensive data about every vertex in the sketch.
    Includes ALL geometry - both construction and normal.
    """
    vertices_data = []
    geometry = sketch.Geometry

    for geo_idx, geo in enumerate(geometry):
        if not hasattr(geo, 'TypeId'):
            continue

        type_id = geo.TypeId
        is_construction = sketch.getConstruction(geo_idx)

        # Determine which positions this geometry has
        positions_to_check = []
        if type_id in ['Part::GeomLineSegment', 'Part::GeomArcOfCircle', 'Part::GeomArcOfEllipse',
                      'Part::GeomArcOfHyperbola', 'Part::GeomArcOfParabola', 'Part::GeomBSplineCurve']:
            positions_to_check = [1, 2]  # Start, End
        elif type_id == 'Part::GeomCircle':
            # Check if it's a full circle or arc
            try:
                start_point = sketch.getPoint(geo_idx, 1)
                end_point = sketch.getPoint(geo_idx, 2)
                if abs(start_point.x - end_point.x) < 1e-10 and abs(start_point.y - end_point.y) < 1e-10:
                    positions_to_check = [3]  # Center only for full circles
                else:
                    positions_to_check = [1, 2, 3]  # Start, End, Center for arcs
            except:
                positions_to_check = [3]  # Default to center
        elif type_id == 'Part::GeomPoint':
            positions_to_check = [1]  # Point geometry

        # Collect data for each position
        for pos in positions_to_check:
            try:
                point = sketch.getPoint(geo_idx, pos)
                coordinate = (point.x, point.y)

                vertex_data = {
                    'vertex': (geo_idx, pos),
                    'coordinate': coordinate,
                    'geometry_type': type_id,
                    'is_construction': is_construction,
                    'is_bspline': type_id == 'Part::GeomBSplineCurve',
                    'is_bspline_endpoint': False,  # Will be updated below
                    'existing_constraints': [],
                    'constrained_to': []
                }

                vertices_data.append(vertex_data)

            except:
                pass  # Skip positions that can't be accessed

    return vertices_data


def analyze_existing_constraints(sketch, vertices_data):
    """Analyze existing constraints and update vertex data."""
    # Create lookup map for vertices
    vertex_lookup = {vdata['vertex']: vdata for vdata in vertices_data}

    # OPTIMIZATION: Cache constraints list to avoid repeated API calls
    constraints = sketch.Constraints

    # Track InternalAlignment constraints to identify B-spline endpoints managed via circles
    for constraint in constraints:
        if constraint.Type == "Coincident":
            v1 = (constraint.First, constraint.FirstPos)
            v2 = (constraint.Second, constraint.SecondPos)

            # Update both vertices with constraint info
            if v1 in vertex_lookup:
                vertex_lookup[v1]['existing_constraints'].append({
                    'type': 'Coincident',
                    'to_vertex': v2
                })
                if v2 in vertex_lookup:
                    vertex_lookup[v1]['constrained_to'].append(v2)

            if v2 in vertex_lookup:
                vertex_lookup[v2]['existing_constraints'].append({
                    'type': 'Coincident',
                    'to_vertex': v1
                })
                if v1 in vertex_lookup:
                    vertex_lookup[v2]['constrained_to'].append(v1)

        elif constraint.Type == "InternalAlignment":
            # InternalAlignment connects construction circle centers to B-spline control points
            # For B-splines: typically Circle center (FirstPos=3) to B-spline (SecondPos=0)
            v1 = (constraint.First, constraint.FirstPos)
            v2 = (constraint.Second, constraint.SecondPos)

            # Update both vertices with InternalAlignment constraint info
            if v1 in vertex_lookup:
                vertex_lookup[v1]['existing_constraints'].append({
                    'type': 'InternalAlignment',
                    'to_vertex': v2
                })
                if v2 in vertex_lookup:
                    vertex_lookup[v1]['constrained_to'].append(v2)

            if v2 in vertex_lookup:
                vertex_lookup[v2]['existing_constraints'].append({
                    'type': 'InternalAlignment',
                    'to_vertex': v1
                })
                if v1 in vertex_lookup:
                    vertex_lookup[v2]['constrained_to'].append(v1)


def find_vertex_groups(vertices_data, tolerance):
    """
    Group vertices within tolerance.
    Returns list of groups (ordered by first vertex appearance, deterministic).
    Matches SketcherWireDoctor's approach for deterministic ordering.
    """
    groups = []
    processed_indices = set()

    for i, vertex_data in enumerate(vertices_data):
        if i in processed_indices:
            continue

        coord = vertex_data['coordinate']
        group = [vertex_data]
        processed_indices.add(i)

        # Find other vertices within tolerance
        for j, other_vertex_data in enumerate(vertices_data):
            if j in processed_indices:
                continue

            other_coord = other_vertex_data['coordinate']
            distance = ((coord[0] - other_coord[0])**2 + (coord[1] - other_coord[1])**2)**0.5

            if distance <= tolerance:
                group.append(other_vertex_data)
                processed_indices.add(j)

        if len(group) > 0:
            groups.append({
                'coordinate': coord,
                'vertices': group
            })

    return groups


def filter_eligible_vertices(sketch, vertex_group, tolerance):
    """
    Filter vertices to only include those eligible for constraining.

    Exclude vertices with InternalAlignment EXCEPT construction circle centers
    that are actually at B-spline/curve endpoint positions.

    For curves (B-spline, ellipse, parabola, hyperbola), Pos0 represents the entire edge,
    not a specific point. Multiple construction circles may have InternalAlignment to Pos0
    for control points. We only keep circles whose centers match the actual curve endpoints
    (Pos1 or Pos2).

    CRITICAL: B-spline Pos1 and Pos2 vertices are NEVER constrained directly. They are
    managed through construction circle centers via InternalAlignment to Pos0. Including
    them creates redundant constraints.

    This prevents redundancy by:
    - Excluding B-spline Pos1/Pos2 vertices (managed via construction circles)
    - Excluding curve control point circles (InternalAlignment but not at endpoints)
    - Excluding curve Pos0 vertices (entire edge, not a specific point)
    - Including construction circle centers at actual curve endpoints
    - Including all regular geometry without InternalAlignment
    """
    eligible = []

    for vdata in vertex_group:
        vertex = vdata['vertex']
        geo_idx, pos = vertex

        # CRITICAL FILTER: Exclude B-spline Pos1 and Pos2 (start/end points)
        # These are managed through construction circles via InternalAlignment
        if vdata['geometry_type'] == 'Part::GeomBSplineCurve' and pos in [1, 2]:
            continue

        # Check if this vertex has InternalAlignment
        has_internal_alignment = any(
            c['type'] == 'InternalAlignment'
            for c in vdata['existing_constraints']
        )

        if has_internal_alignment:
            # Exception: Keep construction circle centers that are at curve endpoints
            is_circle_center = (
                vdata['geometry_type'] == 'Part::GeomCircle' and
                vdata['is_construction']
            )

            if not is_circle_center:
                # Not a circle center - skip all other InternalAlignment vertices
                continue

            # Circle center with InternalAlignment - check if it's at a curve endpoint
            # Find what this circle is InternalAlignment-connected to
            is_at_endpoint = False

            for constraint_info in vdata['existing_constraints']:
                if constraint_info['type'] != 'InternalAlignment':
                    continue

                target_vertex = constraint_info['to_vertex']
                target_geo_idx, target_pos = target_vertex

                # Check if target is a curve with Pos0 (entire edge)
                try:
                    target_geo = sketch.Geometry[target_geo_idx]
                    target_type = target_geo.TypeId if hasattr(target_geo, 'TypeId') else None

                    # Curves that use Pos0 for the entire edge
                    if target_type in ['Part::GeomBSplineCurve', 'Part::GeomArcOfEllipse',
                                      'Part::GeomArcOfParabola', 'Part::GeomArcOfHyperbola']:
                        if target_pos == 0:
                            # Get curve endpoints (Pos1 = start, Pos2 = end)
                            try:
                                start_point = sketch.getPoint(target_geo_idx, 1)
                                end_point = sketch.getPoint(target_geo_idx, 2)

                                # Get this circle's center coordinate
                                circle_coord = vdata['coordinate']

                                # Check if circle center matches either endpoint
                                dist_to_start = ((circle_coord[0] - start_point.x)**2 +
                                               (circle_coord[1] - start_point.y)**2)**0.5
                                dist_to_end = ((circle_coord[0] - end_point.x)**2 +
                                             (circle_coord[1] - end_point.y)**2)**0.5

                                if dist_to_start <= tolerance or dist_to_end <= tolerance:
                                    is_at_endpoint = True
                                    break
                            except:
                                pass
                except:
                    pass

            if not is_at_endpoint:
                # Circle is InternalAlignment but not at a curve endpoint - skip it
                continue

        eligible.append(vdata)

    return eligible


def find_best_anchor(sketch, vertex_group):
    """
    Find the best anchor vertex in a group.
    Prioritizes vertices with existing constraints.
    """
    if not vertex_group:
        return None

    # Sort by number of existing constraints (descending)
    sorted_vertices = sorted(vertex_group,
                            key=lambda v: len(v['existing_constraints']),
                            reverse=True)

    return sorted_vertices[0]['vertex']


def build_constraint_graph(sketch):
    """
    Build a graph of all constraint connections.
    Includes Coincident, Tangent, and InternalAlignment.

    Returns: defaultdict mapping vertex -> set of connected vertices
    """
    connection_map = defaultdict(set)
    for c in sketch.Constraints:
        if c.Type in ["Coincident", "Tangent", "InternalAlignment"]:
            a = (c.First, c.FirstPos)
            b = (c.Second, c.SecondPos)
            connection_map[a].add(b)
            connection_map[b].add(a)

    return connection_map


def find_connected_component(vertex, connection_map):
    """
    Find all vertices transitively connected to the given vertex.

    Returns: set of vertices in the same connected component
    """
    if not connection_map:
        return {vertex}

    visited = set()
    queue = deque([vertex])

    while queue:
        current = queue.popleft()
        if current in visited:
            continue
        visited.add(current)
        for neighbor in connection_map.get(current, set()):
            if neighbor not in visited:
                queue.append(neighbor)

    return visited


def main():
    """Main function to run the macro."""
    start_time = time.time()

    # Get active sketch in edit mode
    sketch = get_sketch()
    if not sketch:
        return

    print(f"\n=== Constrain All Point-On-Point ===")
    print(f"Sketch: {sketch.Label}")
    print(f"Tolerance: {format_distance(DEFAULT_TOLERANCE)}")

    # Step 1: Collect all vertices
    vertices_data = collect_all_vertices(sketch)
    print(f"Found {len(vertices_data)} total vertices")

    # Step 2: Analyze existing constraints
    analyze_existing_constraints(sketch, vertices_data)

    # Step 3: Group vertices by proximity
    vertex_groups = find_vertex_groups(vertices_data, DEFAULT_TOLERANCE)
    print(f"Found {len(vertex_groups)} vertex groups within tolerance")

    # Step 4: Build initial constraint graph
    connection_map = build_constraint_graph(sketch)

    # Step 5: Filter groups and add constraints
    sketch.Document.openTransaction("Constrain All Point-On-Point")

    constraints_added = 0
    groups_processed = 0
    apply_start = time.time()

    try:
        for group_data in vertex_groups:
            coord = group_data['coordinate']
            group = group_data['vertices']
            eligible = filter_eligible_vertices(sketch, group, DEFAULT_TOLERANCE)

            # Need at least 2 vertices to constrain
            if len(eligible) < 2:
                continue

            # Find best anchor
            anchor_vertex = find_best_anchor(sketch, eligible)

            # Get all vertices that need constraining (excluding anchor)
            vertices_to_constrain = [v['vertex'] for v in eligible if v['vertex'] != anchor_vertex]

            if not vertices_to_constrain:
                continue

            # Find connected component containing anchor
            anchor_component = find_connected_component(anchor_vertex, connection_map)

            # Only constrain vertices NOT already in anchor's component
            for vertex in vertices_to_constrain:
                if vertex in anchor_component:
                    continue  # Already transitively connected to anchor

                # Add the constraint
                try:
                    geo1, pos1 = vertex
                    geo2, pos2 = anchor_vertex
                    constraint_index = sketch.addConstraint(Sketcher.Constraint('Coincident', geo1, pos1, geo2, pos2))

                    if constraint_index >= 0:
                        constraints_added += 1

                        # Update connection map to reflect new constraint
                        connection_map[vertex].add(anchor_vertex)
                        connection_map[anchor_vertex].add(vertex)

                        # Merge this vertex's component into anchor's component
                        # (for subsequent vertices in this same group)
                        anchor_component.add(vertex)

                except Exception as e:
                    pass  # Silently skip failures

            groups_processed += 1

        sketch.Document.commitTransaction()

        apply_time = time.time() - apply_start

        # Solve and recompute
        sketch.solve()
        sketch.Document.recompute()

        total_time = time.time() - start_time

        print(f"Processed {groups_processed} groups")
        print(f"Application time: {apply_time:.3f}s")
        print(f"Added {constraints_added} coincident constraints")
        print(f"Total time: {total_time:.3f}s")
        if apply_time > 0:
            print(f"Rate: {constraints_added/apply_time:.0f} constraints/second")
        print("=" * 50)

    except Exception as e:
        sketch.Document.abortTransaction()
        print(f"ERROR: Failed to add constraints: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 50)


# Run the macro
if __name__ == "__main__":
    main()
