
__version__ = "1.0.0"
__status__ = "beta"
__date__ = "2025-08-21"
__author__ = "NSUBB (aka DesignWeaver3D)"

"""
SketchReProfile Macro for FreeCAD

Version: 1.0.0
Status: beta
Author: NSUBB (aka DesignWeaver3D)
License: GNU GPL v3.0
GitHub: https://github.com/NSUBB
Reddit: u/DesignWeaver3D

This macro converts mesh-derived sketches to clean reconstructed geometry:
automatically detects and creates circles, arcs, B-splines from tessellated construction lines.
"""

# SPDX-License-Identifier: GPL-3.0-or-later

# License: GNU General Public License v3.0
# This macro is free software: you can redistribute it and/or modify
# it under the terms of the GNU GPL as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# See https://www.gnu.org/licenses/gpl-3.0.html for details.


import FreeCAD as App
import FreeCADGui as Gui
import Part, Sketcher
import math
import numpy as np
import time
from collections import defaultdict

def get_open_sketch():
    edit_obj = Gui.ActiveDocument.getInEdit()
    if edit_obj and hasattr(edit_obj, 'Object'):
        obj = edit_obj.Object
        if hasattr(obj, 'TypeId') and 'Sketch' in obj.TypeId:
            return obj
    return None

def extract_edge_data(sketch):
    data = []
    for geo in sketch.Geometry:
        if hasattr(geo, 'TypeId'):
            if geo.TypeId == 'Part::GeomLineSegment':
                data.append({'type': 'line',
                             'start': (geo.StartPoint.x, geo.StartPoint.y),
                             'end': (geo.EndPoint.x, geo.EndPoint.y)})
            elif geo.TypeId == 'Part::GeomArcOfCircle':
                data.append({'type': 'arc',
                             'start': (geo.StartPoint.x, geo.StartPoint.y),
                             'end': (geo.EndPoint.x, geo.EndPoint.y),
                             'center': (geo.Center.x, geo.Center.y),
                             'radius': geo.Radius})
    return data

def find_connected_vertices(edges, tol=1e-6):
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
    graph, lookup = {}, {}
    for e in edges:
        s, t = e['start'], e['end']
        if s != t:
            graph.setdefault(s, []).append(t)
            graph.setdefault(t, []).append(s)
            lookup[(s, t)] = lookup[(t, s)] = e
    return graph, lookup

def detect_polygons(graph, edge_lookup):
    visited, polys = set(), []
    for v in graph:
        if v in visited:
            continue
        poly, curr, prev = [], v, None
        for _ in range(100):
            poly.append(curr)
            visited.add(curr)
            next_v = [n for n in graph[curr] if n != prev]
            if not next_v or (next_v[0] == v and len(poly) >= 3):
                break
            prev, curr = curr, next_v[0]
        if len(poly) >= 3:
            edges = []
            for i in range(len(poly)):
                key = (poly[i], poly[(i + 1) % len(poly)])
                if key in edge_lookup:
                    edges.append(edge_lookup[key])
                else:
                    print(f"[INFO] Skipped missing edge: {key}")
            if edges:
                polys.append({'vertices': poly, 'edges': edges})
    return polys

def calculate_edge_length(edge):
    if edge['type'] == 'line':
        dx = edge['end'][0] - edge['start'][0]
        dy = edge['end'][1] - edge['start'][1]
        return math.hypot(dx, dy)
    if edge['type'] == 'arc':
        cx, cy = edge['center']
        sx, sy = edge['start']
        ex, ey = edge['end']
        a1 = math.atan2(sy - cy, sx - cx)
        a2 = math.atan2(ey - cy, ex - cx)
        diff = (a2 - a1 + math.pi * 3) % (2 * math.pi) - math.pi
        return edge['radius'] * abs(diff)
    return 0.0

def log_initial_edge_stats(edges):
    from collections import Counter

    lengths = [calculate_edge_length(e) for e in edges]
    total = len(lengths)

    App.Console.PrintMessage(f"Total edges analyzed: {total}\n")

    if not lengths:
        App.Console.PrintMessage("No edges with measurable length.\n")
        return

    avg = sum(lengths) / total
    rounded_lengths = [round(l, 4) for l in lengths]
    count = Counter(rounded_lengths)
    mode_val, mode_freq = count.most_common(1)[0]

    greater = [l for l in lengths if l > avg]
    lesser = [l for l in lengths if l < avg]
    lesser_avg = sum(lesser) / len(lesser) if lesser else 0.0

    App.Console.PrintMessage(f"Average length: {avg:.4f}\n")
    App.Console.PrintMessage(f"Mode length: {mode_val:.4f} ({mode_freq} edges)\n")
    App.Console.PrintMessage(f"Edges > average: {len(greater)}\n")
    App.Console.PrintMessage(f"Edges < average: {len(lesser)}\n")
    App.Console.PrintMessage(f"Mean length of edges < average: {lesser_avg:.4f}\n")

def log_initial_geometry_stats(edges):
    App.Console.PrintMessage("\n=== INITIAL EDGE GEOMETRY STATISTICS ===\n")
    log_initial_edge_stats(edges)
    App.Console.PrintMessage("\n=== END OF INITIAL EDGE GEOMETRY STATISTICS ===\n")

def is_equilateral_polygon(polygon, tolerance=1e-3):
    lengths = [calculate_edge_length(e) for e in polygon['edges']]
    min_len, max_len = min(lengths), max(lengths)
    stats = {'min': min_len, 'max': max_len, 'avg': sum(lengths)/len(lengths), 'range': max_len - min_len}
    return (stats['range'] <= tolerance), lengths, stats

def analyze_equilateral(polygons):
    return [p for p in polygons if is_equilateral_polygon(p)[0]]

def best_fit_circle(verts):
    A, b = [], []
    for x, y in verts:
        A.append([2*x, 2*y, 1])
        b.append(x**2 + y**2)
    try:
        sol = np.linalg.solve(np.dot(np.transpose(A), A), np.dot(np.transpose(A), b))
        h, k, _ = sol
        center = (h, k)
        radius = sum(math.hypot(x - h, y - k) for x, y in verts) / len(verts)
        return center, radius
    except:
        return None, None

def add_circumcircle_with_constraints(sketch, center, radius):
    cx, cy = center
    origin = App.Vector(0, 0, 0)
    endpoint = App.Vector(cx, cy, 0)
    line = Part.LineSegment(origin, endpoint)
    idx_line = sketch.addGeometry(line, True)
    sketch.addConstraint(Sketcher.Constraint('Block', idx_line))
    circle = Part.Circle()
    circle.Center = endpoint
    idx_circle = sketch.addGeometry(circle, False)
    sketch.addConstraint(Sketcher.Constraint('Radius', idx_circle, radius))

def find_partial_regular_edge_runs(polygon, length_tol=1e-3, angle_tol=1.5):
    from math import degrees, acos
    runs = []
    edges = polygon['edges']
    n = len(edges)

    def unit(vec):
        dx, dy = vec
        mag = math.hypot(dx, dy)
        return (dx / mag, dy / mag) if mag > 0 else (0, 0)

    def vector_angle(u, v):
        ux, uy = unit(u)
        vx, vy = unit(v)
        dot = max(min(ux * vx + uy * vy, 1.0), -1.0)
        return degrees(acos(dot))

    i = 0
    while i < n - 2:
        run = [edges[i]]
        base_len = calculate_edge_length(edges[i])
        v1 = (edges[i]['end'][0] - edges[i]['start'][0], edges[i]['end'][1] - edges[i]['start'][1])
        j = i + 1
        angles, lengths = [], [base_len]

        while j < n:
            curr = edges[j]
            curr_len = calculate_edge_length(curr)
            if abs(curr_len - base_len) > length_tol:
                break
            v2 = (curr['end'][0] - curr['start'][0], curr['end'][1] - curr['start'][1])
            angle = vector_angle(v1, v2)
            if len(angles) == 0 or abs(angle - angles[-1]) <= angle_tol:
                angles.append(angle)
                lengths.append(curr_len)
                run.append(curr)
                v1 = v2
                j += 1
            else:
                break

        if len(run) >= 3:
            avg_len = sum(lengths) / len(lengths)
            avg_angle = sum(angles) / len(angles) if angles else 0
            runs.append({'edges': run, 'length': avg_len, 'angle': avg_angle, 'count': len(run)})
            i = j
        else:
            i += 1

    return runs

def fit_arc_by_3_points_from_chain(run):
    conn = defaultdict(list)
    for edge in run['edges']:
        s, e = edge['start'], edge['end']
        conn[s].append(e)
        conn[e].append(s)
    endpoints = [v for v, nbrs in conn.items() if len(nbrs) == 1]
    start = endpoints[0] if len(endpoints) == 2 else list(conn.keys())[0]
    path = []
    visited = set()
    current = start
    while current and current not in visited:
        path.append(current)
        visited.add(current)
        neighbors = [n for n in conn[current] if n not in visited]
        current = neighbors[0] if neighbors else None
    if len(path) < 3:
        raise ValueError("Not enough vertices to define an arc")
    return path[0], path[len(path) // 2], path[-1]

def process_equilateral_polygons(sketch, polygons):
    equi = analyze_equilateral(polygons)
    used = set()
    for poly in equi:
        center, radius = best_fit_circle(poly['vertices'])
        if center and radius:
            add_circumcircle_with_constraints(sketch, center, radius)
            App.Console.PrintMessage(f"Added circumcircle at ({center[0]:.4f}, {center[1]:.4f}), radius={radius:.4f}\n")
            used.update(id(e) for e in poly['edges'])
    sketch.recompute()
    return equi, used

def draw_partial_arcs(sketch, polygons, equi):
    used = set()
    for i, poly in enumerate(polygons):
        if poly not in equi:
            partials = find_partial_regular_edge_runs(poly)
            if partials:
                App.Console.PrintMessage(f"Polygon {i+1} contains {len(partials)} partial regular edge runs:\n")
                for j, run in enumerate(partials):
                    App.Console.PrintMessage(
                        f"  Run {j+1}: {run['count']} edges, avg. length = {run['length']:.4f}, avg. angle = {run['angle']:.2f}°\n"
                    )
                    try:
                        start_pt, mid_pt, end_pt = fit_arc_by_3_points_from_chain(run)

                        # Create vectors directly
                        start = App.Vector(*start_pt, 0)
                        mid = App.Vector(*mid_pt, 0)
                        end = App.Vector(*end_pt, 0)

                        arc = Part.Arc(start, mid, end)
                        idx_arc = sketch.addGeometry(arc, False)
                        used.update(id(e) for e in run['edges'])
                        App.Console.PrintMessage(f"  DEBUG: Flagged edge IDs as used: {[id(e) for e in run['edges']]}\n")

                        # Estimate center from 3 points
                        center, radius = best_fit_circle([start_pt, mid_pt, end_pt])
                        if center:
                            origin = App.Vector(0, 0, 0)
                            endpoint = App.Vector(*center, 0)
                            center_line = Part.LineSegment(origin, endpoint)
                            idx_line = sketch.addGeometry(center_line, True)
                            sketch.addConstraint(Sketcher.Constraint('Block', idx_line))
                            sketch.addConstraint(Sketcher.Constraint('Radius', idx_arc, radius))

                        App.Console.PrintMessage(
                            f"  ➤ Arc added: {start_pt} → {end_pt} via {mid_pt}, radius={radius:.4f}\n"
                        )

                    except Exception as e:
                        App.Console.PrintError(f"  ✖ Failed to create arc: {e}\n")
    return used

def draw_colinear_lines(sketch, runs):
    used = set()
    for i, run in enumerate(runs):
        pts = [run[0]['start']] + [e['end'] for e in run]
        line = Part.LineSegment(App.Vector(*pts[0], 0), App.Vector(*pts[-1], 0))
        sketch.addGeometry(line, False)
        used.update(id(e) for e in run)
        print(f"[INFO] ➤ Colinear run {i+1}: {len(run)} segments → line from {pts[0]} to {pts[-1]}")
    sketch.recompute()
    return used

# Updated find_colinear_edge_runs function that uses existing graph and only processes unused edges
def find_colinear_edge_runs(unused_edges, graph, edge_lookup, angle_tol=1e-2, min_run=2):
    from math import atan2, degrees

    def compute_angle(e):
        dx = e['end'][0] - e['start'][0]
        dy = e['end'][1] - e['start'][1]
        return degrees(atan2(dy, dx))

    def get_edge_between_vertices(v1, v2):
        """Get edge between two vertices using existing lookup"""
        return edge_lookup.get((v1, v2)) or edge_lookup.get((v2, v1))

    runs = []
    used_edge_ids = set()

    for start_edge in unused_edges:
        # Skip if already used or not a line
        if id(start_edge) in used_edge_ids or start_edge['type'] != 'line':
            continue

        # Start a new run with this edge
        run = [start_edge]
        used_edge_ids.add(id(start_edge))
        base_angle = compute_angle(start_edge)

        # Try to extend the run in both directions from start_edge
        for start_vertex in [start_edge['start'], start_edge['end']]:
            current_vertex = start_vertex
            current_edge = start_edge

            # Follow the chain as far as possible
            while True:
                # Find connected vertices from current vertex
                connected_vertices = graph.get(current_vertex, [])

                next_edge = None
                next_vertex = None

                # Check each connected vertex for a colinear edge
                for candidate_vertex in connected_vertices:
                    if candidate_vertex == current_vertex:
                        continue

                    # Get the edge between current and candidate vertex
                    candidate_edge = get_edge_between_vertices(current_vertex, candidate_vertex)

                    # Check if this edge qualifies
                    if (candidate_edge and
                        id(candidate_edge) not in used_edge_ids and
                        candidate_edge['type'] == 'line' and
                        abs(compute_angle(candidate_edge) - base_angle) <= angle_tol):

                        next_edge = candidate_edge
                        next_vertex = candidate_vertex
                        break

                # If we found a connecting colinear edge, add it to the run
                if next_edge:
                    run.append(next_edge)
                    used_edge_ids.add(id(next_edge))
                    current_vertex = next_vertex
                    current_edge = next_edge
                else:
                    break  # No more colinear edges in this direction

        # Only keep runs with minimum length
        if len(run) >= min_run:
            runs.append(run)
        else:
            # Remove from used set if run is too short
            for edge in run:
                used_edge_ids.discard(id(edge))

    return runs

def find_spline_runs(unused_edges, graph, edge_lookup, global_used_edge_ids, angle_threshold=15.0, min_run=4):
    """
    FIXED VERSION: Detect smooth spline-like edge runs from unused edges.
    Now takes global_used_edge_ids to prevent accessing edges used by arcs/circles.
    """
    from math import atan2, degrees, sqrt

    def compute_edge_vector(edge):
        """Get normalized direction vector for an edge"""
        dx = edge['end'][0] - edge['start'][0]
        dy = edge['end'][1] - edge['start'][1]
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

    def get_edge_between_vertices(v1, v2):
        """Get edge between two vertices using existing lookup"""
        return edge_lookup.get((v1, v2)) or edge_lookup.get((v2, v1))

    def build_connected_chain(start_edge, local_used_edges):
        """Build a connected chain starting from an edge"""
        chain = [start_edge]
        local_used_edges.add(id(start_edge))

        # Extend chain in both directions from start_edge
        for start_vertex in [start_edge['start'], start_edge['end']]:
            current_vertex = start_vertex
            current_edge = start_edge

            # Follow chain as far as possible in this direction
            while True:
                connected_vertices = graph.get(current_vertex, [])

                next_edge = None
                next_vertex = None

                # Find next unused edge connected to current vertex
                for candidate_vertex in connected_vertices:
                    if candidate_vertex == current_vertex:
                        continue

                    candidate_edge = get_edge_between_vertices(current_vertex, candidate_vertex)

                    # FIXED: Check against BOTH global used edges AND local spline used edges
                    if (candidate_edge and
                        id(candidate_edge) not in global_used_edge_ids and  # Not used by arcs/circles
                        id(candidate_edge) not in local_used_edges and     # Not used by this spline run
                        candidate_edge['type'] == 'line'):

                        next_edge = candidate_edge
                        next_vertex = candidate_vertex
                        break

                if next_edge:
                    # Add to appropriate end of chain based on direction
                    if current_vertex == start_edge['start']:
                        chain.insert(0, next_edge)  # Add to beginning
                    else:
                        chain.append(next_edge)  # Add to end

                    local_used_edges.add(id(next_edge))
                    current_vertex = next_vertex
                    current_edge = next_edge
                else:
                    break  # No more connected edges

        return chain

    def analyze_angular_progression(chain):
        """Analyze chain for smooth angular progression"""
        if len(chain) < 2:
            return False, []

        break_points = []
        vectors = []

        # Calculate direction vectors for each edge
        for edge in chain:
            vectors.append(compute_edge_vector(edge))

        # Check angular changes between consecutive edges
        for i in range(len(vectors) - 1):
            angle_change = angle_between_vectors(vectors[i], vectors[i + 1])

            if angle_change > angle_threshold:
                break_points.append(i + 1)

        return len(break_points) == 0, break_points

    def split_chain_at_breaks(chain, break_points):
        """Split a chain into sub-chains at sharp angle points"""
        if not break_points:
            return [chain]

        sub_chains = []
        start_idx = 0

        for break_idx in break_points:
            if break_idx - start_idx >= min_run:
                sub_chains.append(chain[start_idx:break_idx])
            start_idx = break_idx

        # Add final segment
        if len(chain) - start_idx >= min_run:
            sub_chains.append(chain[start_idx:])

        return sub_chains

    # Main detection logic
    spline_runs = []
    local_used_edge_ids = set()  # Track edges used by spline runs only

    for start_edge in unused_edges:
        # Skip if already processed or not a line segment
        if id(start_edge) in local_used_edge_ids or start_edge['type'] != 'line':
            continue

        # Build connected chain starting from this edge
        chain = build_connected_chain(start_edge, local_used_edge_ids)

        if len(chain) >= min_run:
            # Analyze for smooth angular progression
            is_smooth, break_points = analyze_angular_progression(chain)

            if is_smooth:
                # Entire chain is smooth
                spline_runs.append(chain)
            elif break_points:
                # Split chain at sharp angles and keep smooth segments
                sub_chains = split_chain_at_breaks(chain, break_points)
                spline_runs.extend(sub_chains)
        else:
            # Chain too short, remove edges from used set
            for edge in chain:
                local_used_edge_ids.discard(id(edge))

    return spline_runs

def extract_ordered_vertices_from_spline_run(spline_run):
    """
    Extract ordered vertices from a spline run to create interpolation points.
    Returns list of (x, y) coordinates in correct order.
    """
    if not spline_run:
        return []

    # Build connectivity for this run
    from collections import defaultdict
    connections = defaultdict(list)

    for edge in spline_run:
        start, end = edge['start'], edge['end']
        connections[start].append(end)
        connections[end].append(start)

    # Find endpoints (vertices with only one connection)
    endpoints = [v for v, neighbors in connections.items() if len(neighbors) == 1]

    if len(endpoints) != 2:
        # Fallback: use first edge's vertices and all end vertices
        vertices = [spline_run[0]['start']]
        for edge in spline_run:
            vertices.append(edge['end'])
        return vertices

    # Traverse from one endpoint to the other
    start_vertex = endpoints[0]
    vertices = [start_vertex]
    current = start_vertex
    visited = {start_vertex}

    while True:
        neighbors = [v for v in connections[current] if v not in visited]
        if not neighbors:
            break

        next_vertex = neighbors[0]
        vertices.append(next_vertex)
        visited.add(next_vertex)
        current = next_vertex

    return vertices

def draw_splines(sketch, spline_runs):
    """
    Create B-spline geometry for detected spline runs.
    Returns set of edge IDs that were used.
    """
    used = set()

    for i, run in enumerate(spline_runs):
        try:
            # Extract ordered vertices
            vertices = extract_ordered_vertices_from_spline_run(run)

            if len(vertices) < 2:
                App.Console.PrintError(f"Spline run {i+1}: insufficient vertices\n")
                continue

            # Convert to FreeCAD vectors
            vectors = [App.Vector(x, y, 0) for x, y in vertices]

            # Create interpolating B-spline (degree 3, non-periodic)
            spline = Part.BSplineCurve()
            spline.interpolate(vectors, False)  # False = not periodic

            # Add to sketch
            sketch.addGeometry(spline, False)

            # Track used edges
            used.update(id(e) for e in run)

            App.Console.PrintMessage(
                f"Spline {i+1}: {len(run)} edges → B-spline through {len(vertices)} points\n"
            )

        except Exception as e:
            App.Console.PrintError(f"Failed to create spline {i+1}: {e}\n")

    return used

def toggle_remaining_construction_to_normal(sketch, final_unused_edges):
    """
    Toggle remaining unused construction edges to normal geometry.
    These are edges that didn't fit any pattern but are part of the final profile.
    """
    if not final_unused_edges:
        App.Console.PrintMessage("No unused edges to toggle.\n")
        return 0

    # We need to find which sketch geometry indices correspond to our unused edges
    # This requires matching edge coordinates to sketch geometry

    toggled_count = 0
    tolerance = 1e-6

    for unused_edge in final_unused_edges:
        # Find the corresponding geometry index in the sketch
        for i, geo in enumerate(sketch.Geometry):
            if hasattr(geo, 'TypeId') and geo.TypeId == 'Part::GeomLineSegment':
                # Check if this geometry matches our unused edge
                geo_start = (geo.StartPoint.x, geo.StartPoint.y)
                geo_end = (geo.EndPoint.x, geo.EndPoint.y)

                edge_start = unused_edge['start']
                edge_end = unused_edge['end']

                # Check if coordinates match (either direction)
                if ((abs(geo_start[0] - edge_start[0]) < tolerance and
                     abs(geo_start[1] - edge_start[1]) < tolerance and
                     abs(geo_end[0] - edge_end[0]) < tolerance and
                     abs(geo_end[1] - edge_end[1]) < tolerance) or
                    (abs(geo_start[0] - edge_end[0]) < tolerance and
                     abs(geo_start[1] - edge_end[1]) < tolerance and
                     abs(geo_end[0] - edge_start[0]) < tolerance and
                     abs(geo_end[1] - edge_start[1]) < tolerance)):

                    # Check if it's currently construction geometry
                    if sketch.getConstruction(i):
                        # Toggle to normal geometry
                        sketch.toggleConstruction(i)
                        toggled_count += 1
                        App.Console.PrintMessage(f"Toggled Geom{i} from construction to normal\n")
                        break

    App.Console.PrintMessage(f"Toggled {toggled_count} unused construction edges to normal geometry\n")
    return toggled_count

def add_coincident_constraints_to_endpoints(sketch):
    """
    Add coincident constraints to unconstrained endpoints that are at the same location.
    Simplified version of SketcherWireDoctor constraint logic for integration.
    """
    import math
    from collections import defaultdict

    TOLERANCE = 5e-6  # 5 micrometers tolerance

    def get_all_endpoints(sketch):
        """Collect all geometry endpoints with their coordinates."""
        endpoints = []

        for geo_idx, geometry in enumerate(sketch.Geometry):
            if not hasattr(geometry, 'TypeId'):
                continue

            # Skip construction geometry - only constrain normal geometry
            if sketch.getConstruction(geo_idx):
                continue

            try:
                if geometry.TypeId == 'Part::GeomLineSegment':
                    # Add start and end points
                    start = sketch.getPoint(geo_idx, 1)
                    end = sketch.getPoint(geo_idx, 2)
                    endpoints.append({
                        'vertex': (geo_idx, 1),
                        'coordinate': (start.x, start.y),
                        'geo_type': 'Line'
                    })
                    endpoints.append({
                        'vertex': (geo_idx, 2),
                        'coordinate': (end.x, end.y),
                        'geo_type': 'Line'
                    })

                elif geometry.TypeId == 'Part::GeomArcOfCircle':
                    # Add start and end points (not center)
                    start = sketch.getPoint(geo_idx, 1)
                    end = sketch.getPoint(geo_idx, 2)
                    endpoints.append({
                        'vertex': (geo_idx, 1),
                        'coordinate': (start.x, start.y),
                        'geo_type': 'Arc'
                    })
                    endpoints.append({
                        'vertex': (geo_idx, 2),
                        'coordinate': (end.x, end.y),
                        'geo_type': 'Arc'
                    })

                elif geometry.TypeId == 'Part::GeomBSplineCurve':
                    # Add start and end points of B-splines
                    start = sketch.getPoint(geo_idx, 1)
                    end = sketch.getPoint(geo_idx, 2)
                    endpoints.append({
                        'vertex': (geo_idx, 1),
                        'coordinate': (start.x, start.y),
                        'geo_type': 'BSpline'
                    })
                    endpoints.append({
                        'vertex': (geo_idx, 2),
                        'coordinate': (end.x, end.y),
                        'geo_type': 'BSpline'
                    })

            except Exception:
                continue  # Skip if can't get points

        return endpoints

    def constraint_exists(sketch, v1, v2):
        """Check if a coincident constraint already exists between two vertices."""
        for constraint in sketch.Constraints:
            if constraint.Type == "Coincident":
                existing_v1 = (constraint.First, constraint.FirstPos)
                existing_v2 = (constraint.Second, constraint.SecondPos)
                if (existing_v1 == v1 and existing_v2 == v2) or (existing_v1 == v2 and existing_v2 == v1):
                    return True
        return False

    def group_endpoints_by_location(endpoints):
        """Group endpoints that are at the same location."""
        location_groups = []
        processed = set()

        for i, endpoint in enumerate(endpoints):
            if i in processed:
                continue

            coord = endpoint['coordinate']
            group = [endpoint]
            processed.add(i)

            # Find other endpoints at the same location
            for j, other_endpoint in enumerate(endpoints):
                if j in processed:
                    continue

                other_coord = other_endpoint['coordinate']
                distance = math.sqrt((coord[0] - other_coord[0])**2 + (coord[1] - other_coord[1])**2)

                if distance <= TOLERANCE:
                    group.append(other_endpoint)
                    processed.add(j)

            # Only keep groups with multiple endpoints
            if len(group) > 1:
                location_groups.append(group)

        return location_groups

    # Main constraint addition logic
    try:
        endpoints = get_all_endpoints(sketch)
        location_groups = group_endpoints_by_location(endpoints)

        constraints_added = 0

        for group in location_groups:
            # Use first endpoint as anchor
            anchor = group[0]
            anchor_vertex = anchor['vertex']

            # Constrain all other endpoints to the anchor
            for endpoint in group[1:]:
                vertex = endpoint['vertex']

                # Don't constrain geometry to itself
                if vertex[0] == anchor_vertex[0]:
                    continue

                # Check if constraint already exists
                if constraint_exists(sketch, vertex, anchor_vertex):
                    continue

                # Add coincident constraint
                try:
                    sketch.addConstraint(Sketcher.Constraint(
                        'Coincident',
                        vertex[0], vertex[1],
                        anchor_vertex[0], anchor_vertex[1]
                    ))
                    constraints_added += 1

                    App.Console.PrintMessage(
                        f"Added coincident constraint: {endpoint['geo_type']} endpoint → {anchor['geo_type']} endpoint\n"
                    )

                except Exception as e:
                    App.Console.PrintError(f"Failed to add constraint: {e}\n")

        if constraints_added > 0:
            App.Console.PrintMessage(f"Added {constraints_added} coincident constraints to endpoints\n")
            # Trigger solver
            sketch.solve()
        else:
            App.Console.PrintMessage("No endpoint constraints needed - all endpoints properly constrained\n")

        return constraints_added

    except Exception as e:
        App.Console.PrintError(f"Error adding endpoint constraints: {e}\n")
        return 0

def log_unused_edge_stats(unused_edges):
    App.Console.PrintMessage("\n=== RESIDUAL EDGE LENGTH STATISTICS ===\n")
    log_initial_edge_stats(unused_edges)
    App.Console.PrintMessage("\n=== END OF RESIDUAL EDGE LENGTH STATISTICS ===\n")

def log_summary(total_edges, polygons, edges_used, unused_edges, equi_count, duration):
    App.Console.PrintMessage("\n=== MACRO SUMMARY ===\n")
    App.Console.PrintMessage(f"Total edges in sketch: {total_edges}\n")
    App.Console.PrintMessage(f"Closed polygons detected: {len(polygons)}\n")
    for i, p in enumerate(polygons):
        App.Console.PrintMessage(f"  Polygon {i+1}: {len(p['edges'])} edges\n")
    App.Console.PrintMessage(f"Edges used in polygons: {edges_used}\n")
    App.Console.PrintMessage(f"Edges not in any polygon: {unused_edges}\n")
    App.Console.PrintMessage(f"Equilateral polygons found: {equi_count}\n")
    App.Console.PrintMessage(f"Total execution time: {duration:.4f} seconds\n")

# Updated main function with spline detection

def final_sketcher_main():
    sketch = get_open_sketch()
    if not sketch:
        App.Console.PrintError("No sketch open.\n")
        return

    sketch.Document.openTransaction("SketchPolygonsToCircles")
    start_time = time.time()

    try:
        # Extract and report initial edge stats
        edges = extract_edge_data(sketch)
        log_initial_geometry_stats(edges)

        # Preprocessing - build connectivity graph
        find_connected_vertices(edges)
        graph, lookup = build_graph(edges)
        polygons = detect_polygons(graph, lookup)

        # Analyze equilateral polygons and track used edges
        equi, used_from_equi = process_equilateral_polygons(sketch, polygons)

        # Analyze partial edge runs and track used edges
        used_from_arcs = draw_partial_arcs(sketch, polygons, equi)

        # Combine IDs of edges used so far
        used_edge_ids = used_from_equi.union(used_from_arcs)

        # Filter out edges that were already used
        unused_edges = [e for e in edges if id(e) not in used_edge_ids]

        # Detect and draw colinear lines from remaining unused edges
        colinear_runs = find_colinear_edge_runs(unused_edges, graph, lookup)
        used_from_colinear = draw_colinear_lines(sketch, colinear_runs)

        # Update used edges
        used_edge_ids = used_edge_ids.union(used_from_colinear)

        # Filter edges again for spline detection
        unused_edges_for_splines = [e for e in edges if id(e) not in used_edge_ids]

        # ==> DEBUG CHECK <==
        App.Console.PrintMessage(f"\n=== CRITICAL DEBUG CHECK ===\n")
        App.Console.PrintMessage(f"Total edges: {len(edges)}\n")
        App.Console.PrintMessage(f"Used edge IDs: {len(used_edge_ids)}\n")
        App.Console.PrintMessage(f"Unused edges for splines: {len(unused_edges_for_splines)}\n")

        unused_edge_ids = [id(e) for e in unused_edges_for_splines]
        overlap = set(unused_edge_ids) & used_edge_ids
        App.Console.PrintMessage(f"OVERLAP (should be empty): {len(overlap)} edges\n")

        if overlap:
            App.Console.PrintMessage(f"🚨 BUG FOUND: {len(overlap)} edges are in both used and unused sets!\n")
        else:
            App.Console.PrintMessage(f"✅ No overlap - filtering appears correct\n")

        # FIXED: Pass global used edge IDs to spline detection
        spline_runs = find_spline_runs(unused_edges_for_splines, graph, lookup, used_edge_ids)
        used_from_splines = draw_splines(sketch, spline_runs)

        # Debug: Check edge accounting
        total_spline_edges = sum(len(run) for run in spline_runs)
        App.Console.PrintMessage(f"DEBUG: Spline runs claim to use {total_spline_edges} edges\n")
        App.Console.PrintMessage(f"DEBUG: Available unused edges: {len(unused_edges_for_splines)}\n")

        # Final used edge count
        used_edge_ids = used_edge_ids.union(used_from_splines)

        # Filter out edges that were not used in any generated geometry
        final_unused_edges = [e for e in edges if id(e) not in used_edge_ids]

        # Toggle remaining unused construction edges to normal geometry
        App.Console.PrintMessage(f"\n=== TOGGLING UNUSED EDGES ===\n")
        toggled_count = toggle_remaining_construction_to_normal(sketch, final_unused_edges)

        # Update final unused count after toggling
        final_unused_count = len(final_unused_edges) - toggled_count

        # Log stats specifically for unused edges (before toggling)
        if final_unused_edges:
            log_unused_edge_stats(final_unused_edges)

        # NEW: Add coincident constraints to endpoints
        App.Console.PrintMessage(f"\n=== ADDING ENDPOINT CONSTRAINTS ===\n")
        constraints_added = add_coincident_constraints_to_endpoints(sketch)

        # Enhanced summary log
        total_edges = len(edges)
        edges_used = len(used_edge_ids)
        equi_count = len(equi)
        spline_count = len(spline_runs)
        duration = time.time() - start_time

        App.Console.PrintMessage("\n=== MACRO SUMMARY ===\n")
        App.Console.PrintMessage(f"Total edges in sketch: {total_edges}\n")
        App.Console.PrintMessage(f"Closed polygons detected: {len(polygons)}\n")
        App.Console.PrintMessage(f"Equilateral polygons found: {equi_count}\n")
        App.Console.PrintMessage(f"Spline runs detected: {spline_count}\n")
        App.Console.PrintMessage(f"Colinear runs simplified: {len(colinear_runs)}\n")
        App.Console.PrintMessage(f"Construction edges toggled to normal: {toggled_count}\n")
        App.Console.PrintMessage(f"Endpoint constraints added: {constraints_added}\n")
        App.Console.PrintMessage(f"Edges used in geometry: {edges_used}\n")
        App.Console.PrintMessage(f"Edges remaining unused: {final_unused_count}\n")
        App.Console.PrintMessage(f"Total execution time: {duration:.4f} seconds\n")

        sketch.Document.commitTransaction()

    except Exception as e:
        sketch.Document.abortTransaction()
        App.Console.PrintError(f"Macro aborted due to error: {e}\n")

if __name__ == "__main__":
    final_sketcher_main()
