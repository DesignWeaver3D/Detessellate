__version__ = "2.0.0"
__date__ = "2025-01-18"
__author__ = "DesignWeaver3D"

"""
Optimized graph-based approach that:
1. Detects circular arc runs and creates arcs/circles
2. Detects colinear line runs and simplifies
3. Creates b-splines for remaining curves
4. Adds endpoint constraints

PREREQUISITES:
- Sketch must be open in edit mode
- Geometry should be construction mode (recommended workflow)
"""

import FreeCAD as App
import FreeCADGui as Gui
import Part, Sketcher
import time
import math
from collections import defaultdict

def get_open_sketch():
    """Get the currently open sketch for editing."""
    if Gui.ActiveDocument:
        edit_obj = Gui.ActiveDocument.getInEdit()
        if edit_obj and hasattr(edit_obj, 'Object'):
            obj = edit_obj.Object
            if hasattr(obj, 'TypeId') and 'Sketch' in obj.TypeId:
                return obj
    return None

def extract_edge_data(sketch):
    """
    Extract edge data from sketch geometry with geo_idx tracked.
    Each edge includes a 'used_by' field to track which processing phase used it.

    OPTIMIZATION: Cache property accesses to minimize FreeCAD API calls.
    """
    data = []

    # Single API call to get geometry list
    geometry = sketch.Geometry

    for idx, geo in enumerate(geometry):
        if hasattr(geo, 'TypeId'):
            # Cache TypeId property access
            type_id = geo.TypeId

            if type_id == 'Part::GeomLineSegment':
                # Cache all property accesses
                start_pt = geo.StartPoint
                end_pt = geo.EndPoint

                data.append({
                    'geo_idx': idx,
                    'type': 'line',
                    'start': (start_pt.x, start_pt.y),
                    'end': (end_pt.x, end_pt.y),
                    'length': math.hypot(
                        end_pt.x - start_pt.x,
                        end_pt.y - start_pt.y
                    ),
                    'used_by': None  # Track usage: 'arc', 'colinear', 'spline', etc.
                })
            elif type_id == 'Part::GeomArcOfCircle':
                # Cache all property accesses
                start_pt = geo.StartPoint
                end_pt = geo.EndPoint
                center_pt = geo.Center

                data.append({
                    'geo_idx': idx,
                    'type': 'arc',
                    'start': (start_pt.x, start_pt.y),
                    'end': (end_pt.x, end_pt.y),
                    'center': (center_pt.x, center_pt.y),
                    'radius': geo.Radius,
                    'length': geo.toShape().Length,
                    'used_by': None
                })
    return data

def find_connected_vertices(edges, tol=1e-6):
    """Group vertices within tolerance."""
    groups = []
    for e in edges:
        for pt in [e['start'], e['end']]:
            for g in groups:
                if any(abs(pt[0]-p[0])<tol and abs(pt[1]-p[1])<tol for p in g):
                    g.append(pt)
                    break
            else:
                groups.append([pt])
    mapping = {pt: g[0] for g in groups for pt in g}
    for e in edges:
        e['start'], e['end'] = mapping[e['start']], mapping[e['end']]
    return mapping

def build_graph(edges):
    """Build adjacency graph preserving geo_idx."""
    graph, edge_lookup = {}, {}
    for e in edges:
        s, t = e['start'], e['end']
        if s != t:
            graph.setdefault(s, []).append(t)
            graph.setdefault(t, []).append(s)
            edge_lookup[(s, t)] = e
            edge_lookup[(t, s)] = e
    return graph, edge_lookup

def walk_graph_find_sequences(graph, edge_lookup):
    """Walk graph once to find all connected sequences with traversal direction."""
    visited_edges = set()
    sequences = []

    for start_vertex in graph:
        for first_neighbor in graph[start_vertex]:
            edge_key = (start_vertex, first_neighbor)

            if edge_key in visited_edges or (first_neighbor, start_vertex) in visited_edges:
                continue

            sequence = []
            prev_vertex = start_vertex
            curr_vertex = first_neighbor

            while True:
                edge_key = (prev_vertex, curr_vertex)
                visited_edges.add(edge_key)
                visited_edges.add((curr_vertex, prev_vertex))

                edge = edge_lookup[edge_key]

                # Store edge with traversal direction
                sequence.append({
                    'edge': edge,
                    'from': prev_vertex,
                    'to': curr_vertex
                })

                neighbors = [n for n in graph[curr_vertex] if n != prev_vertex]

                if not neighbors:
                    break

                if neighbors[0] == start_vertex and len(sequence) >= 3:
                    break

                prev_vertex = curr_vertex
                curr_vertex = neighbors[0]

            if sequence:
                sequences.append(sequence)

    return sequences

def find_equal_length_runs(sequence, length_tolerance=1e-4):
    """Find consecutive equal-length edge runs in a sequence."""
    if not sequence:
        return []

    runs = []
    current_run = {
        'start_idx': 0,
        'length': sequence[0]['edge']['length'],
        'items': [sequence[0]]  # Store the full item (edge + direction)
    }

    for i in range(1, len(sequence)):
        item = sequence[i]
        edge = item['edge']
        length_diff = abs(edge['length'] - current_run['length'])

        if length_diff <= length_tolerance:
            current_run['items'].append(item)
        else:
            if len(current_run['items']) >= 3:
                current_run['end_idx'] = i - 1
                current_run['count'] = len(current_run['items'])
                runs.append(current_run)

            current_run = {
                'start_idx': i,
                'length': edge['length'],
                'items': [item]
            }

    if len(current_run['items']) >= 3:
        current_run['end_idx'] = len(sequence) - 1
        current_run['count'] = len(current_run['items'])
        runs.append(current_run)

    return runs

def analyze_run_geometry(run):
    """Analyze if a run represents circular geometry."""
    items = run['items']

    # Build ordered vertex path using traversal direction
    vertices = [items[0]['from']]
    for item in items:
        vertices.append(item['to'])

    first_pt = vertices[0]
    last_pt = vertices[-1]
    is_closed = math.hypot(first_pt[0] - last_pt[0], first_pt[1] - last_pt[1]) < 1e-6

    if len(vertices) < 3:
        return {'is_circular': False}

    p1 = vertices[0]
    p2 = vertices[len(vertices)//2]
    p3 = vertices[-1]

    try:
        ax, ay = p1[0], p1[1]
        bx, by = p2[0], p2[1]
        cx, cy = p3[0], p3[1]

        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(d) < 1e-9:
            return {'is_circular': False}

        ux = ((ax*ax + ay*ay) * (by - cy) + (bx*bx + by*by) * (cy - ay) + (cx*cx + cy*cy) * (ay - by)) / d
        uy = ((ax*ax + ay*ay) * (cx - bx) + (bx*bx + by*by) * (ax - cx) + (cx*cx + cy*cy) * (bx - ax)) / d

        center = (ux, uy)

        radii = [math.hypot(v[0] - ux, v[1] - uy) for v in vertices]
        avg_radius = sum(radii) / len(radii)
        max_deviation = max(abs(r - avg_radius) for r in radii)

        if max_deviation > avg_radius * 0.05:
            return {'is_circular': False}

        start_angle = math.atan2(vertices[0][1] - uy, vertices[0][0] - ux)
        end_angle = math.atan2(vertices[-1][1] - uy, vertices[-1][0] - ux)
        arc_angle = math.degrees((end_angle - start_angle) % (2 * math.pi))

        return {
            'is_circular': True,
            'center': center,
            'radius': avg_radius,
            'arc_angle': arc_angle,
            'is_full_circle': is_closed,
            'max_deviation': max_deviation,
            'start_point': vertices[0],
            'end_point': vertices[-1]
        }

    except Exception as e:
        return {'is_circular': False}

def create_arc_from_run(sketch, run, geom_info):
    """
    Create arc geometry from a circular run.
    Marks edges as used by 'arc' phase.
    Returns tuple: (new geometry index, set of replaced geo_idx, arc data for constraints)
    """
    radius = geom_info['radius']

    # Create arc using traversal-ordered vertices
    items = run['items']

    # Build ordered vertex path
    vertices = [items[0]['from']]
    for item in items:
        vertices.append(item['to'])

    # Use vertices along the traversal path
    start_pt = App.Vector(vertices[0][0], vertices[0][1], 0)
    mid_pt = App.Vector(vertices[1][0], vertices[1][1], 0)  # Second vertex
    end_pt = App.Vector(vertices[-1][0], vertices[-1][1], 0)

    # Create arc geometry only (no constraints or center lines yet)
    arc = Part.Arc(start_pt, mid_pt, end_pt)
    idx_arc = sketch.addGeometry(arc, False)

    # Mark edges as used by arc phase
    for item in run['items']:
        item['edge']['used_by'] = 'arc'

    # Collect geometry indices that were replaced
    replaced_indices = set(item['edge']['geo_idx'] for item in run['items'])

    # Store arc data for later - DON'T read center yet
    arc_data = {
        'geo_idx': idx_arc,
        'radius': radius,
        'is_circle': False
    }

    return idx_arc, replaced_indices, arc_data

def detect_and_create_circles(sketch, sequences, graph, edge_lookup):
    """
    Detect and create circles from closed loops BEFORE arc detection.
    Returns tuple: (set of edges used, list of circle data for later constraints)
    """
    phase_start = time.time()

    circles_created = 0
    used_edges = set()
    circle_data = []  # Store (geo_idx, center_x, center_y, radius) for later

    for sequence in sequences:
        if len(sequence) < 3:
            continue

        # Check if this sequence forms a closed loop
        first_vertex = sequence[0]['from']
        last_vertex = sequence[-1]['to']

        # Check if there's an edge connecting last back to first (closing edge)
        closing_edge_key = (last_vertex, first_vertex)
        closing_edge = edge_lookup.get(closing_edge_key) or edge_lookup.get((first_vertex, last_vertex))

        if not closing_edge or id(closing_edge) in used_edges:
            continue

        # Check if all edges in sequence are equal length
        runs = find_equal_length_runs(sequence, length_tolerance=1e-4)

        for run in runs:
            # Only consider runs that use the entire sequence
            if len(run['items']) != len(sequence):
                continue

            # Analyze geometry
            items = run['items']
            vertices = [items[0]['from']]
            for item in items:
                vertices.append(item['to'])

            if len(vertices) < 3:
                continue

            # Check circularity
            center_x = sum(v[0] for v in vertices) / len(vertices)
            center_y = sum(v[1] for v in vertices) / len(vertices)

            radii = [math.hypot(v[0] - center_x, v[1] - center_y) for v in vertices]
            avg_radius = sum(radii) / len(radii)
            max_deviation = max(abs(r - avg_radius) for r in radii)

            # If deviation is small enough, it's a circle
            if max_deviation <= avg_radius * 0.05:
                # Create circle geometry only (no constraints or center lines yet)
                circle = Part.Circle()
                circle.Center = App.Vector(center_x, center_y, 0)
                circle.Radius = avg_radius
                idx_circle = sketch.addGeometry(circle, False)

                circles_created += 1

                # Store data for later - DON'T read center yet
                circle_data.append({
                    'geo_idx': idx_circle,
                    'radius': avg_radius,
                    'is_circle': True
                })

                # Mark all edges as used (including closing edge)
                for item in run['items']:
                    item['edge']['used_by'] = 'circle'
                    used_edges.add(id(item['edge']))

                closing_edge['used_by'] = 'circle'
                used_edges.add(id(closing_edge))

    elapsed = time.time() - phase_start
    App.Console.PrintMessage(f"Circles: {circles_created} created, {len(used_edges)} edges ({elapsed:.4f}s)\n")

    return used_edges, circle_data

def process_circular_runs(sketch, sequences):
    """
    Process all sequences to find and create arcs from circular runs.
    Only processes edges not already used by circles.
    Returns tuple: (set of used geometry indices, list of arc data for constraints)
    """
    phase_start = time.time()

    used_geo_indices = set()
    processed_edges = set()
    arcs_created = 0
    arc_data_list = []  # Collect arc data for later constraints

    for seq_idx, sequence in enumerate(sequences):
        # Filter out already-processed edges AND edges used by circles
        unprocessed_sequence = [item for item in sequence
                               if id(item['edge']) not in processed_edges
                               and item['edge']['used_by'] is None]

        if not unprocessed_sequence:
            continue

        runs = find_equal_length_runs(unprocessed_sequence)

        for run in runs:
            geom_info = analyze_run_geometry(run)

            if geom_info['is_circular']:
                # Should only be arcs here (circles handled separately)
                new_idx, replaced_indices, arc_data = create_arc_from_run(sketch, run, geom_info)
                used_geo_indices.update(replaced_indices)
                arc_data_list.append(arc_data)

                # Mark these edges as processed to prevent duplicates
                for item in run['items']:
                    processed_edges.add(id(item['edge']))

                arcs_created += 1

    elapsed = time.time() - phase_start
    App.Console.PrintMessage(f"Arcs: {arcs_created} created, {len(used_geo_indices)} edges ({elapsed:.4f}s)\n")

    return used_geo_indices, arc_data_list

def add_coincident_constraints_to_endpoints(sketch):
    """Add coincident constraints to endpoints (from original code)."""
    phase_start = time.time()

    try:
        # Use FreeCAD's built-in constraint detection with default tolerance
        sketch.detectMissingPointOnPointConstraints()
        initial_constraints = len(sketch.Constraints)

        sketch.makeMissingPointOnPointCoincident()

        final_constraints = len(sketch.Constraints)
        added = final_constraints - initial_constraints

        if added > 0:
            sketch.solve()

        elapsed = time.time() - phase_start
        App.Console.PrintMessage(f"Constraints: {added} coincident added ({elapsed:.4f}s)\n")

        return added

    except Exception as e:
        App.Console.PrintError(f"Error adding endpoint constraints: {e}\n")
        return 0

def apply_arc_circle_constraints(sketch, circle_data, arc_data):
    """
    Apply all constraints for circles and arcs in batch.
    1. Create center lines
    2. Point-on-point constraints
    3. Radius constraints
    4. Recompute
    5. Center point coincident constraints (in batches)
    6. Block constraints

    OPTIMIZATION: Cache sketch.Geometry access and use larger solver batches.
    """
    phase_start = time.time()

    # Combine all arc/circle data and sort by geo_idx for consistent pairing
    all_data = circle_data + arc_data
    all_data.sort(key=lambda d: d['geo_idx'])

    if not all_data:
        return

    # Step 1: Batch create all center lines
    # Read centers fresh now (after all geometry/constraint changes)
    origin = App.Vector(0, 0, 0)

    # OPTIMIZATION: Cache geometry access (single API call)
    geometry = sketch.Geometry

    # OPTIMIZATION: Build all centerlines first, then add in single batch
    centerlines = []
    for data in all_data:
        geo_idx = data['geo_idx']
        # Read center NOW from cached geometry list
        geo = geometry[geo_idx]
        endpoint = geo.Center
        line = Part.LineSegment(origin, endpoint)
        centerlines.append(line)

    # Add all centerlines in single batch call
    # Returns tuple of indices when adding multiple geometries
    centerline_indices = sketch.addGeometry(centerlines, True)

    # Convert tuple to list for consistent indexing
    centerline_indices = list(centerline_indices)

    # Step 2: Native point-on-point constraints
    sketch.detectMissingPointOnPointConstraints()
    sketch.makeMissingPointOnPointCoincident()

    # Step 3: Batch add radius constraints
    radius_constraints = []
    for data in all_data:
        radius_constraints.append(
            Sketcher.Constraint('Radius', data['geo_idx'], data['radius'])
        )

    if radius_constraints:
        sketch.addConstraint(radius_constraints)

    # Step 4: Recompute
    sketch.recompute()

    # Step 5: Add ALL center constraints at once (no batching needed)
    # PointPos 3 = center for circles/arcs, PointPos 2 = end for lines
    center_constraints = []
    for i, data in enumerate(all_data):
        geo_idx = data['geo_idx']
        line_idx = centerline_indices[i]
        center_constraints.append(
            Sketcher.Constraint('Coincident', geo_idx, 3, line_idx, 2)
        )

    if center_constraints:
        try:
            sketch.addConstraint(center_constraints)
            sketch.solve()
            center_constraints_added = len(center_constraints)
        except Exception as e:
            App.Console.PrintWarning(f"Center constraints failed: {e}\n")
            center_constraints_added = 0

    # Step 6: Batch add block constraints on center lines (AFTER center constraints)
    block_constraints = []
    for idx_line in centerline_indices:
        block_constraints.append(Sketcher.Constraint('Block', idx_line))

    if block_constraints:
        sketch.addConstraint(block_constraints)

    elapsed = time.time() - phase_start
    App.Console.PrintMessage(
        f"Arc/Circle constraints: {len(centerline_indices)} centerlines, "
        f"{len(radius_constraints)} radius, {len(block_constraints)} block"
    )
    if center_constraints_added > 0:
        App.Console.PrintMessage(f", {center_constraints_added} center")
    App.Console.PrintMessage(f" ({elapsed:.4f}s)\n")


def find_colinear_runs(sequences, angle_tol=1.0, min_run=2):
    """
    Find runs of colinear line segments in sequences.
    Only processes items where edge['used_by'] is None.
    Returns list of runs, where each run is a list of items with traversal direction.

    TOLERANCE CONCERN:
    The angle_tol parameter (default 1.0 degree) determines how strictly edges must
    be aligned to be considered colinear. For mesh-derived geometry with noise:
    - Too strict (e.g., 0.1°): Misses legitimately colinear segments with minor noise
    - Too loose (e.g., 5.0°): May incorrectly combine segments with intentional curves

    Current default of 1.0° appears conservative in testing but may need adjustment
    based on mesh quality. Consider making this user-configurable or adaptive in future.
    """
    from math import atan2, degrees

    def compute_angle_from_traversal(item):
        """Compute angle based on traversal direction (from -> to)"""
        from_pt = item['from']
        to_pt = item['to']
        dx = to_pt[0] - from_pt[0]
        dy = to_pt[1] - from_pt[1]
        return degrees(atan2(dy, dx))

    runs = []
    used_edges = set()  # Track EDGES (not items) already used in a run

    for sequence in sequences:
        # Filter to only unused line segments
        unused_items = [item for item in sequence
                       if item['edge']['used_by'] is None
                       and item['edge']['type'] == 'line'
                       and id(item['edge']) not in used_edges]

        if len(unused_items) < min_run:
            continue

        # Find colinear runs within this sequence
        i = 0
        while i < len(unused_items):
            run = [unused_items[i]]
            used_edges.add(id(unused_items[i]['edge']))
            base_angle = compute_angle_from_traversal(unused_items[i])

            # Extend run while edges are colinear AND connected
            j = i + 1
            while j < len(unused_items):
                item = unused_items[j]

                # Check connectivity: current end must match next start
                if run[-1]['to'] != item['from']:
                    break

                angle = compute_angle_from_traversal(item)

                # Normalize angle difference to [-180, 180]
                angle_diff = (angle - base_angle + 180) % 360 - 180

                if abs(angle_diff) <= angle_tol:
                    run.append(item)
                    used_edges.add(id(item['edge']))
                    j += 1
                else:
                    break

            # Keep run if it meets minimum length
            if len(run) >= min_run:
                runs.append(run)
                i = j
            else:
                # Remove from used set if run is too short
                for item in run:
                    used_edges.discard(id(item['edge']))
                i += 1

    return runs

def create_line_from_colinear_run(sketch, run):
    """
    Create a single line segment from a colinear run.
    Marks edges as used by 'colinear' phase.
    Returns the new geometry index.
    """
    # Build ordered vertex path
    vertices = [run[0]['from']]
    for item in run:
        vertices.append(item['to'])

    # Create line from first to last vertex
    start_pt = App.Vector(vertices[0][0], vertices[0][1], 0)
    end_pt = App.Vector(vertices[-1][0], vertices[-1][1], 0)

    line = Part.LineSegment(start_pt, end_pt)
    idx_line = sketch.addGeometry(line, False)

    # Mark edges as used
    for item in run:
        item['edge']['used_by'] = 'colinear'

    return idx_line

def process_colinear_runs(sketch, sequences):
    """
    Process sequences to find and simplify colinear runs.
    Returns count of lines created.
    """
    phase_start = time.time()

    runs = find_colinear_runs(sequences)

    lines_created = 0
    edges_simplified = 0

    for run in runs:
        idx = create_line_from_colinear_run(sketch, run)
        lines_created += 1
        edges_simplified += len(run)

    elapsed = time.time() - phase_start
    App.Console.PrintMessage(f"Lines: {lines_created} created, {edges_simplified} edges ({elapsed:.4f}s)\n")

    return lines_created

def find_spline_runs(graph, edge_lookup, angle_threshold=15.0, min_run=4):
    """
    Find smooth curved runs suitable for B-spline interpolation by walking the graph.
    Only follows edges where edge['used_by'] is None.

    ANGLE THRESHOLD:
    The angle_threshold (default 15.0 degrees) determines maximum angle change
    between consecutive edges to be considered a smooth curve. Sharp angles
    indicate corners that should break the spline into separate segments.
    """
    from math import atan2, degrees, sqrt

    def compute_edge_vector(item):
        """Get normalized direction vector based on traversal direction"""
        from_pt = item['from']
        to_pt = item['to']
        dx = to_pt[0] - from_pt[0]
        dy = to_pt[1] - from_pt[1]
        length = sqrt(dx*dx + dy*dy)
        if length > 0:
            return (dx/length, dy/length)
        return (0, 0)

    def angle_between_vectors(v1, v2):
        """Calculate angle between two direction vectors in degrees"""
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        angle_rad = atan2(abs(cross), dot)
        return degrees(angle_rad)

    def get_edge_item(v1, v2):
        """Get edge with traversal direction from vertex v1 to v2"""
        edge = edge_lookup.get((v1, v2)) or edge_lookup.get((v2, v1))
        if not edge:
            return None
        # Create item with correct traversal direction
        return {
            'edge': edge,
            'from': v1,
            'to': v2
        }

    def walk_smooth_run_bidirectional(start_vertex, first_neighbor, used_edges):
        """
        Walk graph in BOTH directions from the starting edge to build complete run.
        This ensures we capture smooth edges both before and after our starting point.
        """
        # First, walk forward from start_vertex -> first_neighbor
        forward_run = []
        prev_vertex = start_vertex
        curr_vertex = first_neighbor

        while True:
            # Get edge from prev to curr
            item = get_edge_item(prev_vertex, curr_vertex)
            if not item:
                break

            edge = item['edge']

            # Check if edge is unused and is a line
            if edge['used_by'] is not None or edge['type'] != 'line':
                break

            # Check if already used in this phase
            if id(edge) in used_edges:
                break

            # Check angular smoothness if we have previous edge
            if len(forward_run) > 0:
                v1 = compute_edge_vector(forward_run[-1])
                v2 = compute_edge_vector(item)
                angle_change = angle_between_vectors(v1, v2)

                if angle_change > angle_threshold:
                    break  # Sharp corner - end run

            # Add to run
            forward_run.append(item)
            used_edges.add(id(edge))

            # Find next vertex - check all neighbors for smoothest continuation
            neighbors = [n for n in graph.get(curr_vertex, []) if n != prev_vertex]
            if not neighbors:
                break

            # If multiple neighbors, find the one with smoothest angle
            best_neighbor = None
            best_angle = float('inf')

            for neighbor in neighbors:
                candidate_item = get_edge_item(curr_vertex, neighbor)
                if not candidate_item:
                    continue

                candidate_edge = candidate_item['edge']

                # Skip if not usable
                if (candidate_edge['used_by'] is not None or
                    candidate_edge['type'] != 'line' or
                    id(candidate_edge) in used_edges):
                    continue

                # Calculate angle change
                v1 = compute_edge_vector(item)
                v2 = compute_edge_vector(candidate_item)
                angle_change = angle_between_vectors(v1, v2)

                # Keep if smoothest and within threshold
                if angle_change < best_angle and angle_change <= angle_threshold:
                    best_angle = angle_change
                    best_neighbor = neighbor

            if best_neighbor is None:
                break

            prev_vertex = curr_vertex
            curr_vertex = best_neighbor

        # Now walk backward from start_vertex (in the opposite direction)
        backward_run = []
        prev_vertex = first_neighbor  # Coming from the first_neighbor direction
        curr_vertex = start_vertex

        # Find neighbors of start_vertex (excluding first_neighbor which we already walked)
        neighbors = [n for n in graph.get(start_vertex, []) if n != first_neighbor]

        if neighbors:
            # Find smoothest backward continuation
            best_neighbor = None
            best_angle = float('inf')

            # Need to check angle relative to the first forward edge (if exists)
            for neighbor in neighbors:
                candidate_item = get_edge_item(curr_vertex, neighbor)
                if not candidate_item:
                    continue

                candidate_edge = candidate_item['edge']

                # Skip if not usable
                if (candidate_edge['used_by'] is not None or
                    candidate_edge['type'] != 'line' or
                    id(candidate_edge) in used_edges):
                    continue

                # Calculate angle change relative to first forward edge
                if forward_run:
                    # Reverse the candidate vector since we're going backward
                    v1_reverse = compute_edge_vector(candidate_item)
                    v1 = (-v1_reverse[0], -v1_reverse[1])  # Reverse direction
                    v2 = compute_edge_vector(forward_run[0])
                    angle_change = angle_between_vectors(v1, v2)

                    if angle_change < best_angle and angle_change <= angle_threshold:
                        best_angle = angle_change
                        best_neighbor = neighbor
                else:
                    # No forward run, just take first valid neighbor
                    best_neighbor = neighbor
                    break

            # Walk backward if we found a valid start
            if best_neighbor is not None:
                curr_vertex = best_neighbor
                prev_vertex = start_vertex

                while True:
                    # Get edge from prev to curr (going backward from start_vertex)
                    item = get_edge_item(prev_vertex, curr_vertex)
                    if not item:
                        break

                    edge = item['edge']

                    if edge['used_by'] is not None or edge['type'] != 'line':
                        break

                    if id(edge) in used_edges:
                        break

                    # Check smoothness relative to previous backward edge
                    if len(backward_run) > 0:
                        v1 = compute_edge_vector(backward_run[-1])
                        v2 = compute_edge_vector(item)
                        angle_change = angle_between_vectors(v1, v2)

                        if angle_change > angle_threshold:
                            break

                    backward_run.append(item)
                    used_edges.add(id(edge))

                    # Find next vertex in backward direction
                    neighbors = [n for n in graph.get(curr_vertex, []) if n != prev_vertex]
                    if not neighbors:
                        break

                    best_neighbor = None
                    best_angle = float('inf')

                    for neighbor in neighbors:
                        candidate_item = get_edge_item(curr_vertex, neighbor)
                        if not candidate_item:
                            continue

                        candidate_edge = candidate_item['edge']

                        if (candidate_edge['used_by'] is not None or
                            candidate_edge['type'] != 'line' or
                            id(candidate_edge) in used_edges):
                            continue

                        v1 = compute_edge_vector(item)
                        v2 = compute_edge_vector(candidate_item)
                        angle_change = angle_between_vectors(v1, v2)

                        if angle_change < best_angle and angle_change <= angle_threshold:
                            best_angle = angle_change
                            best_neighbor = neighbor

                    if best_neighbor is None:
                        break

                    prev_vertex = curr_vertex
                    curr_vertex = best_neighbor

        # Combine: backward_run (reversed with swapped from/to) + forward_run
        # When reversing backward_run, we need to swap from/to in each item
        for item in backward_run:
            item['from'], item['to'] = item['to'], item['from']

        backward_run.reverse()
        complete_run = backward_run + forward_run

        return complete_run

    spline_runs = []
    used_edges = set()  # Track edges already used in splines

    # Walk graph to find smooth runs
    for start_vertex in graph:
        for first_neighbor in graph.get(start_vertex, []):
            # Get edge between start and first neighbor
            item = get_edge_item(start_vertex, first_neighbor)
            if not item:
                continue

            edge = item['edge']

            # Skip if already used, not a line, or not unused
            if (id(edge) in used_edges or
                edge['used_by'] is not None or
                edge['type'] != 'line'):
                continue

            # Walk smooth run from this starting edge (bidirectional)
            run = walk_smooth_run_bidirectional(start_vertex, first_neighbor, used_edges)

            # Keep run if it meets minimum length
            if len(run) >= min_run:
                spline_runs.append(run)
            else:
                # Remove from used set if run is too short
                for item in run:
                    used_edges.discard(id(item['edge']))

    return spline_runs

def create_spline_from_run(sketch, run):
    """
    Create B-spline from a smooth curved run.
    Marks edges as used by 'spline' phase.
    Returns the new geometry index.
    """
    # Build ordered vertex path
    vertices = [run[0]['from']]
    for item in run:
        vertices.append(item['to'])

    # Convert to FreeCAD vectors
    vectors = [App.Vector(x, y, 0) for x, y in vertices]

    # Create interpolating B-spline (degree 3, non-periodic)
    spline = Part.BSplineCurve()
    spline.interpolate(vectors, False)  # False = not periodic

    # Add to sketch
    idx_spline = sketch.addGeometry(spline, False)

    # Mark edges as used
    for item in run:
        item['edge']['used_by'] = 'spline'

    return idx_spline

def process_spline_runs(sketch, graph, edge_lookup):
    """
    Process graph to find and create B-splines from smooth curves.
    Returns count of splines created.
    """
    phase_start = time.time()

    runs = find_spline_runs(graph, edge_lookup)

    splines_created = 0
    edges_interpolated = 0

    for run in runs:
        try:
            idx = create_spline_from_run(sketch, run)
            splines_created += 1
            edges_interpolated += len(run)
        except Exception as e:
            App.Console.PrintError(f"Failed to create spline: {e}\n")

    elapsed = time.time() - phase_start
    App.Console.PrintMessage(f"Splines: {splines_created} created, {edges_interpolated} edges ({elapsed:.4f}s)\n")

    return splines_created

def toggle_unused_to_normal(sketch, edges):
    """
    Toggle remaining unused construction edges to normal geometry.
    These are edges that didn't fit any pattern (arc/line/spline) but are
    part of the final profile.
    """
    phase_start = time.time()

    toggled_count = 0

    for edge in edges:
        # Only process edges that are still unused
        if edge['used_by'] is not None:
            continue

        geo_idx = edge['geo_idx']

        # Check if it's currently construction geometry and set to normal
        try:
            if sketch.getConstruction(geo_idx):
                sketch.setConstruction(geo_idx, False)  # Faster than toggleConstruction
                toggled_count += 1
        except Exception as e:
            App.Console.PrintWarning(f"Failed to toggle geometry {geo_idx}: {e}\n")

    elapsed = time.time() - phase_start
    App.Console.PrintMessage(f"Toggled: {toggled_count} unused edges ({elapsed:.4f}s)\n")

    return toggled_count

def main():
    """Main execution."""
    start_time = time.time()

    sketch = get_open_sketch()

    if not sketch:
        App.Console.PrintError("ERROR: No sketch is currently open for editing.\n")
        return

    App.Console.PrintMessage("\n" + "="*70 + "\n")
    App.Console.PrintMessage(f"SketchReProfile v{__version__}\n")
    App.Console.PrintMessage("="*70 + "\n")
    App.Console.PrintMessage(f"Sketch: {sketch.Label} ({len(sketch.Geometry)} edges)\n")
    App.Console.PrintMessage("="*70 + "\n")

    sketch.Document.openTransaction("SketchReProfile")

    try:
        # Step 1: Extract and build graph
        edges = extract_edge_data(sketch)
        find_connected_vertices(edges)
        graph, edge_lookup = build_graph(edges)

        # Step 2: Find sequences
        sequences = walk_graph_find_sequences(graph, edge_lookup)

        # Step 3: Detect and create circles (geometry only, collect data)
        circle_edges, circle_data = detect_and_create_circles(sketch, sequences, graph, edge_lookup)

        # Step 4: Process arcs (geometry only, collect data)
        used_geo_indices, arc_data = process_circular_runs(sketch, sequences)

        # Step 5: Process colinear runs
        lines_created = process_colinear_runs(sketch, sequences)

        # Step 6: Process splines
        splines_created = process_spline_runs(sketch, graph, edge_lookup)

        # Step 7: Toggle unused edges
        toggled_count = toggle_unused_to_normal(sketch, edges)

        # Step 8: Apply all arc/circle constraints in batch
        apply_arc_circle_constraints(sketch, circle_data, arc_data)

        # Step 9: Add endpoint constraints
        add_coincident_constraints_to_endpoints(sketch)

        # Summary
        elapsed = time.time() - start_time

        App.Console.PrintMessage("="*70 + "\n")
        App.Console.PrintMessage(f"Complete! Total time: {elapsed:.4f}s\n")
        App.Console.PrintMessage("="*70 + "\n\n")

        sketch.Document.commitTransaction()

    except Exception as e:
        sketch.Document.abortTransaction()
        App.Console.PrintError(f"ERROR: Macro aborted due to error: {e}\n")
        import traceback
        App.Console.PrintError(traceback.format_exc())

if __name__ == "__main__":
    main()
