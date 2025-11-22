# ============================================================================
# TAB3: Non-Coincident Vertex Analysis and Constraint Addition v73
# ============================================================================
# Comprehensive vertex analysis with intelligent B-spline filtering
# - Full vertex data collection (coordinates, names, constraints)
# - Tolerance-based grouping with clear indicators  
# - Smart anchor selection prioritizing existing constraint networks
# - INTELLIGENT B-SPLINE HANDLING:
#   * B-spline endpoints: FILTERED OUT (managed via circle centers)
#   * Construction points with InternalAlignment: FILTERED OUT (auto-managed)
#   * Construction circle centers: INCLUDED (key connection points)
#   * Regular geometry vertices: INCLUDED (lines, arcs, etc.)
# - Transaction-safe constraint addition with rollback protection
# ============================================================================

import FreeCAD as App
import Sketcher
import math
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict, deque

# Tolerance thresholds for practical FreeCAD manufacturing use cases
NEAR_COINCIDENT_THRESHOLD = 5e-6    # 5 micrometers - bell emoji (tiny discrepancy, high confidence)
LOOSE_COINCIDENT_THRESHOLD = 100e-6  # 100 micrometers - caution emoji (manufacturing tolerance range)

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

def get_geometry_name(geo_idx, geometry):
    """Get a descriptive name for a geometry element."""
    try:
        if hasattr(geometry, 'TypeId'):
            if geometry.TypeId == 'Part::GeomLineSegment':
                return f"Line{geo_idx+1}"
            elif geometry.TypeId == 'Part::GeomCircle':
                return f"Circle{geo_idx+1}"
            elif geometry.TypeId == 'Part::GeomArcOfCircle':
                return f"Arc{geo_idx+1}"
            elif geometry.TypeId == 'Part::GeomBSplineCurve':
                return f"BSpline{geo_idx+1}"
            elif geometry.TypeId == 'Part::GeomEllipse':
                return f"Ellipse{geo_idx+1}"
            elif geometry.TypeId == 'Part::GeomArcOfEllipse':
                return f"ArcOfEllipse{geo_idx+1}"
            elif geometry.TypeId == 'Part::GeomArcOfHyperbola':
                return f"ArcOfHyperbola{geo_idx+1}"
            elif geometry.TypeId == 'Part::GeomArcOfParabola':
                return f"ArcOfParabola{geo_idx+1}"
            elif geometry.TypeId == 'Part::GeomPoint':
                return f"Point{geo_idx+1}"
            else:
                return f"Geometry{geo_idx+1}"
        else:
            return f"Geometry{geo_idx+1}"
    except:
        return f"Geometry{geo_idx+1}"

def get_gui_vertex_name(sketch, geo_idx, pos):
    """Get GUI-style vertex name (1-indexed like Vertex12)."""
    try:
        # Try to find the actual vertex ID from FreeCAD
        for vid in range(1000):  # reasonable upper limit
            try:
                mapped_geo, mapped_pos = sketch.getGeoVertexIndex(vid)
                if mapped_geo == geo_idx and mapped_pos == pos:
                    return f"Vertex{vid + 1}"  # 1-indexed for GUI
            except:
                break
        # Fallback: calculate approximate vertex ID
        approx_vid = geo_idx * 3 + pos
        return f"Vertex{approx_vid}"
    except:
        return f"Vertex{geo_idx * 3 + pos}"

def get_position_name(pos):
    """Get human-readable position name."""
    pos_map = {1: "Start", 2: "End", 3: "Center"}
    return pos_map.get(pos, f"Pos{pos}")

def collect_all_vertices(analyzer):
    """Collect comprehensive data about every vertex in the sketch."""
    sketch = analyzer.sketch
    vertices_data = []
    
    for geo_idx, (_, geometry) in enumerate(analyzer.all_geometry):
        geo_name = get_geometry_name(geo_idx, geometry)
        
        # Determine which positions this geometry has
        positions_to_check = []
        if hasattr(geometry, 'TypeId'):
            if geometry.TypeId in ['Part::GeomLineSegment', 'Part::GeomArcOfCircle', 'Part::GeomArcOfEllipse', 
                                 'Part::GeomArcOfHyperbola', 'Part::GeomArcOfParabola', 'Part::GeomBSplineCurve']:
                positions_to_check = [1, 2]  # Start, End
            elif geometry.TypeId == 'Part::GeomCircle':
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
            elif geometry.TypeId == 'Part::GeomPoint':
                positions_to_check = [1]  # Point geometry
        
        # Collect data for each position
        for pos in positions_to_check:
            try:
                point = sketch.getPoint(geo_idx, pos)
                coordinate = (point.x, point.y)
                
                vertex_data = {
                    'vertex': (geo_idx, pos),
                    'coordinate': coordinate,
                    'gui_vertex_name': get_gui_vertex_name(sketch, geo_idx, pos),
                    'geometry_name': geo_name,
                    'position_name': get_position_name(pos),
                    'geometry_type': geometry.TypeId if hasattr(geometry, 'TypeId') else 'Unknown',
                    'is_construction': sketch.getConstruction(geo_idx) if hasattr(sketch, 'getConstruction') else False,
                    'is_bspline': geometry.TypeId == 'Part::GeomBSplineCurve' if hasattr(geometry, 'TypeId') else False,
                    'existing_constraints': [],
                    'constrained_to': []
                }
                
                vertices_data.append(vertex_data)
                
            except Exception as e:
                pass  # Skip positions that can't be accessed
    
    return vertices_data

def analyze_existing_constraints(sketch, vertices_data):
    """Analyze existing constraints and update vertex data."""
    
    # Create lookup map for vertices
    vertex_lookup = {}
    for vdata in vertices_data:
        vertex_lookup[vdata['vertex']] = vdata
    
    # Track geometry-level InternalAlignment constraints (pos0)
    geometry_internal_alignments = {}  # geo_idx -> [constraint_info]
    
    constraint_count = 0
    for constraint in sketch.Constraints:
        if constraint.Type == "Coincident":
            constraint_count += 1
            v1 = (constraint.First, constraint.FirstPos)
            v2 = (constraint.Second, constraint.SecondPos)
            
            # Update both vertices with constraint info
            if v1 in vertex_lookup:
                vertex_lookup[v1]['existing_constraints'].append({
                    'type': 'Coincident',
                    'to_vertex': v2,
                    'constraint_index': constraint_count
                })
                if v2 in vertex_lookup:
                    vertex_lookup[v1]['constrained_to'].append({
                        'vertex': v2,
                        'geometry_name': vertex_lookup[v2]['geometry_name'],
                        'position_name': vertex_lookup[v2]['position_name'],
                        'gui_vertex_name': vertex_lookup[v2]['gui_vertex_name']
                    })
            
            if v2 in vertex_lookup:
                vertex_lookup[v2]['existing_constraints'].append({
                    'type': 'Coincident',
                    'to_vertex': v1,
                    'constraint_index': constraint_count
                })
                if v1 in vertex_lookup:
                    vertex_lookup[v2]['constrained_to'].append({
                        'vertex': v1,
                        'geometry_name': vertex_lookup[v1]['geometry_name'],
                        'position_name': vertex_lookup[v1]['position_name'],
                        'gui_vertex_name': vertex_lookup[v1]['gui_vertex_name']
                    })
        
        elif constraint.Type == "InternalAlignment":
            constraint_count += 1
            # InternalAlignment: First geometry to Second geometry
            # For B-splines: typically Circle center (FirstPos=3) to B-spline (SecondPos=0)
            v1 = (constraint.First, constraint.FirstPos)
            v2 = (constraint.Second, constraint.SecondPos)
            
            # Track geometry-level constraints (pos0) for later propagation
            if constraint.SecondPos == 0:
                if constraint.Second not in geometry_internal_alignments:
                    geometry_internal_alignments[constraint.Second] = []
                geometry_internal_alignments[constraint.Second].append({
                    'type': 'InternalAlignment',
                    'to_vertex': v1,
                    'constraint_index': constraint_count
                })
            
            # Update both vertices with InternalAlignment constraint info
            if v1 in vertex_lookup:
                vertex_lookup[v1]['existing_constraints'].append({
                    'type': 'InternalAlignment',
                    'to_vertex': v2,
                    'constraint_index': constraint_count
                })
                if v2 in vertex_lookup:
                    vertex_lookup[v1]['constrained_to'].append({
                        'vertex': v2,
                        'geometry_name': vertex_lookup[v2]['geometry_name'],
                        'position_name': vertex_lookup[v2]['position_name'],
                        'gui_vertex_name': vertex_lookup[v2]['gui_vertex_name']
                    })
            
            if v2 in vertex_lookup:
                vertex_lookup[v2]['existing_constraints'].append({
                    'type': 'InternalAlignment',
                    'to_vertex': v1,
                    'constraint_index': constraint_count
                })
                if v1 in vertex_lookup:
                    vertex_lookup[v2]['constrained_to'].append({
                        'vertex': v1,
                        'geometry_name': vertex_lookup[v1]['geometry_name'],
                        'position_name': vertex_lookup[v1]['position_name'],
                        'gui_vertex_name': vertex_lookup[v1]['gui_vertex_name']
                    })
    
    # Propagate geometry-level InternalAlignment constraints to all vertices of those geometries
    for geo_idx, internal_alignments in geometry_internal_alignments.items():
        for vertex_data in vertices_data:
            if vertex_data['vertex'][0] == geo_idx:  # This vertex belongs to the geometry
                for alignment in internal_alignments:
                    # Add the geometry-level constraint to this vertex
                    vertex_data['existing_constraints'].append({
                        'type': 'InternalAlignment_FromGeometry',
                        'to_vertex': alignment['to_vertex'],
                        'constraint_index': alignment['constraint_index']
                    })
                    # Don't add to constrained_to as it's not a direct vertex relationship
    
    # Post-process: Detect coordinate-based InternalAlignment relationships for all geometry types
    for vertex_data in vertices_data:
        # Check any vertex without direct constraints for coordinate-based InternalAlignment
        if not vertex_data['existing_constraints']:
            vertex_coord = vertex_data['coordinate']
            
            # Find other vertices at the same coordinate with InternalAlignment constraints
            for other_vertex_data in vertices_data:
                if other_vertex_data == vertex_data:
                    continue
                    
                other_coord = other_vertex_data['coordinate']
                distance = math.sqrt((vertex_coord[0] - other_coord[0])**2 + 
                                   (vertex_coord[1] - other_coord[1])**2)
                
                # If at same coordinate (within tolerance) and other vertex has InternalAlignment
                if distance < 1e-6:  # Very tight tolerance for coordinate matching
                    has_internal_alignment = any(c['type'] in ['InternalAlignment', 'InternalAlignment_Derived', 'InternalAlignment_FromGeometry'] 
                                               for c in other_vertex_data['existing_constraints'])
                    
                    if has_internal_alignment:
                        # Mark this vertex as constrained to the other via derived InternalAlignment
                        vertex_data['existing_constraints'].append({
                            'type': 'InternalAlignment_Derived',
                            'to_vertex': other_vertex_data['vertex'],
                            'constraint_index': -1  # Special marker for derived constraint
                        })
                        vertex_data['constrained_to'].append({
                            'vertex': other_vertex_data['vertex'],
                            'geometry_name': other_vertex_data['geometry_name'],
                            'position_name': other_vertex_data['position_name'],
                            'gui_vertex_name': other_vertex_data['gui_vertex_name']
                        })
                        
                        # For B-splines connected to circles, add special display attribute
                        if (vertex_data['is_bspline'] and 
                            other_vertex_data['geometry_type'] == 'Part::GeomCircle' and
                            other_vertex_data['is_construction']):
                            vertex_data['connected_via_circle'] = other_vertex_data['geometry_name']
                        
                        break
    
    return vertices_data

def group_vertices_by_coordinates(vertices_data):
    """Group vertices by their coordinates with tolerance checking."""
    
    coordinate_groups = []
    processed_vertices = set()
    
    for i, vertex_data in enumerate(vertices_data):
        if i in processed_vertices:
            continue
            
        coord = vertex_data['coordinate']
        group = {
            'coordinate': coord,
            'vertices': [vertex_data],
            'exact_matches': [vertex_data],
            'near_coincident': [],
            'loose_coincident': []
        }
        processed_vertices.add(i)
        
        # Find other vertices within tolerance
        for j, other_vertex_data in enumerate(vertices_data):
            if j in processed_vertices:
                continue
                
            other_coord = other_vertex_data['coordinate']
            distance = math.sqrt((coord[0] - other_coord[0])**2 + (coord[1] - other_coord[1])**2)
            
            if distance <= NEAR_COINCIDENT_THRESHOLD:
                group['near_coincident'].append(other_vertex_data)
                group['vertices'].append(other_vertex_data)
                processed_vertices.add(j)
            elif distance <= LOOSE_COINCIDENT_THRESHOLD:
                group['loose_coincident'].append(other_vertex_data)
                group['vertices'].append(other_vertex_data)
                processed_vertices.add(j)
        
        # Only include groups that have potential constraint candidates
        unconstrained_count = len([v for v in group['vertices'] if not v['existing_constraints']])
        if len(group['vertices']) > 1 or unconstrained_count > 0:
            coordinate_groups.append(group)
    
    return coordinate_groups

def _constraint_exists_comprehensive(sketch, v1, v2):
    """Check if a Coincident constraint already exists between two vertices, directly or transitively."""
    # Build connection map of all coincident constraints
    connection_map = defaultdict(set)
    for c in sketch.Constraints:
        if c.Type == "Coincident":
            a = (c.First, c.FirstPos)
            b = (c.Second, c.SecondPos)
            connection_map[a].add(b)
            connection_map[b].add(a)
    
    # Early exit if no constraints exist
    if not connection_map:
        return False
    
    # BFS to find if v1 and v2 are in the same connected group
    visited = set()
    queue = deque([v1])
    
    while queue:
        current = queue.popleft()
        if current == v2:
            return True
        if current in visited:
            continue
        visited.add(current)
        for neighbor in connection_map[current]:
            if neighbor not in visited:
                queue.append(neighbor)
    
    return False

def find_best_anchor_in_group(sketch, group):
    """Find the best anchor vertex in a group, prioritizing existing constraint connections."""
    vertices = group['vertices']
    
    # First priority: Find vertices that are already part of existing constraint networks
    constraint_scores = []
    for vertex_data in vertices:
        vertex = vertex_data['vertex']
        score = 0
        
        # Count existing constraints (higher = better connected)
        constraint_count = len(vertex_data['existing_constraints'])
        score += constraint_count * 100  # High weight for existing constraints
        
        # Bonus for being connected to multiple different geometries
        connected_geometries = set()
        for connection in vertex_data['constrained_to']:
            connected_geometries.add(connection['geometry_name'])
        score += len(connected_geometries) * 50
        
        # Favor construction circle centers (they're often key connection points)
        if (vertex_data['geometry_type'] == 'Part::GeomCircle' and 
            vertex_data['is_construction'] and 
            vertex_data['position_name'] == 'Center'):
            score += 20
        
        # Favor line geometry
        if vertex_data['geometry_type'] == 'Part::GeomLineSegment':
            score += 10
        
        constraint_scores.append((vertex_data, score))
    
    # Sort by score (highest first)
    constraint_scores.sort(key=lambda x: x[1], reverse=True)
    
    return constraint_scores[0][0] if constraint_scores else None

def display_coordinate_groups(coordinate_groups):
    """Display detailed information about coordinate groups."""
    
    for i, group in enumerate(coordinate_groups, 1):
        coord = group['coordinate']
        vertices = group['vertices']
        
        # Show exact matches
        if group['exact_matches']:
            for vertex_data in group['exact_matches']:
                constraints_info = ""
                if vertex_data['existing_constraints']:
                    constraint_targets = [conn['gui_vertex_name'] for conn in vertex_data['constrained_to']]
                    constraints_info = f" → Constrained to: {', '.join(constraint_targets)}"
        
        # Show near coincident
        if group['near_coincident']:
            for vertex_data in group['near_coincident']:
                distance = math.sqrt((coord[0] - vertex_data['coordinate'][0])**2 + (coord[1] - vertex_data['coordinate'][1])**2)
                constraints_info = ""
                if vertex_data['existing_constraints']:
                    constraint_targets = [conn['gui_vertex_name'] for conn in vertex_data['constrained_to']]
                    constraints_info = f" → Constrained to: {', '.join(constraint_targets)}"
        
        # Show loose coincident  
        if group['loose_coincident']:
            for vertex_data in group['loose_coincident']:
                distance = math.sqrt((coord[0] - vertex_data['coordinate'][0])**2 + (coord[1] - vertex_data['coordinate'][1])**2)
                constraints_info = ""
                if vertex_data['existing_constraints']:
                    constraint_targets = [conn['gui_vertex_name'] for conn in vertex_data['constrained_to']]
                    constraints_info = f" → Constrained to: {', '.join(constraint_targets)}"
        
        # Show constraint relationships within group
        constrained_pairs = []
        for vertex_data in vertices:
            for connection in vertex_data['constrained_to']:
                if connection['vertex'] in [v['vertex'] for v in vertices]:
                    pair = (vertex_data['gui_vertex_name'], connection['gui_vertex_name'])
                    reverse_pair = (connection['gui_vertex_name'], vertex_data['gui_vertex_name'])
                    if pair not in constrained_pairs and reverse_pair not in constrained_pairs:
                        constrained_pairs.append(pair)

def find_non_coincident_vertices(analyzer):
    """Comprehensive analysis of non-coincident vertices with intelligent B-spline filtering."""
    
    try:
        # Step 1: Collect all vertex data
        vertices_data = collect_all_vertices(analyzer)
        
        # Step 2: Analyze existing constraints
        vertices_data = analyze_existing_constraints(analyzer.sketch, vertices_data)
        
        # Step 3: Group by coordinates
        coordinate_groups = group_vertices_by_coordinates(vertices_data)
        
        # Step 4: Display comprehensive analysis
        display_coordinate_groups(coordinate_groups)
        
        # Step 5: Convert to OLD FORMAT for backward compatibility with existing Tab3 UI
        non_coincident_groups = []
        for group in coordinate_groups:
            # INTELLIGENT B-SPLINE FILTERING:
            # Filter vertices using the new intelligent approach
            eligible_vertices = []
            filtered_vertices = []
            
            for vertex_data in group['vertices']:
                vertex = vertex_data['vertex']
                
                # FILTER OUT: B-spline endpoints (managed via circle centers)
                if vertex_data['is_bspline']:
                    filtered_vertices.append((vertex_data, "B-spline endpoint (managed via circle centers)"))
                    continue
                
                # FILTER OUT: Construction points with InternalAlignment (auto-managed)
                elif (vertex_data['geometry_type'] == 'Part::GeomPoint' and 
                      vertex_data['is_construction'] and
                      any(c['type'] in ['InternalAlignment', 'InternalAlignment_Derived', 'InternalAlignment_FromGeometry'] 
                          for c in vertex_data['existing_constraints'])):
                    filtered_vertices.append((vertex_data, "Construction point with InternalAlignment (auto-managed)"))
                    continue
                
                # INCLUDE: Construction circle centers (key connection points)
                elif (vertex_data['geometry_type'] == 'Part::GeomCircle' and 
                      vertex_data['is_construction'] and 
                      vertex_data['position_name'] == 'Center'):
                    eligible_vertices.append(vertex_data)
                    
                # INCLUDE: Regular geometry vertices (lines, arcs, etc.)
                elif not vertex_data['is_construction']:
                    eligible_vertices.append(vertex_data)
                    
                # INCLUDE: Construction points WITHOUT InternalAlignment (if any exist)
                elif (vertex_data['geometry_type'] == 'Part::GeomPoint' and 
                      vertex_data['is_construction'] and
                      not any(c['type'] in ['InternalAlignment', 'InternalAlignment_Derived', 'InternalAlignment_FromGeometry'] 
                              for c in vertex_data['existing_constraints'])):
                    eligible_vertices.append(vertex_data)
                    
                # FILTER OUT: Any other construction geometry
                else:
                    filtered_vertices.append((vertex_data, "Other construction geometry"))
            
            # Check if eligible vertices need constraints
            needs_constraints = False
            if len(eligible_vertices) >= 2:
                # Check if all eligible vertices are already constrained to each other
                unconstrained_pairs = []
                for i, v1_data in enumerate(eligible_vertices):
                    for j, v2_data in enumerate(eligible_vertices[i+1:], i+1):
                        v1 = v1_data['vertex']
                        v2 = v2_data['vertex']
                        if not _constraint_exists_comprehensive(analyzer.sketch, v1, v2):
                            unconstrained_pairs.append((v1_data, v2_data))
                            needs_constraints = True
                
            # Only include groups that actually need constraint attention
            if needs_constraints:
                # Create EXACT old format that existing Tab3 UI expects
                old_format_entry = {
                    'coordinate': group['coordinate'],
                    'vertices': [v['vertex'] for v in group['vertices']],  # Keep all for compatibility
                    'constrained_groups': [[v['vertex']] for v in group['vertices']],  # Old format expected this
                    'needs_constraint': True,
                    'vertices_data': group['vertices'],  # Full data for UI display
                    'eligible_vertices': eligible_vertices,  # New: vertices that can be auto-constrained
                    'near_coincident_info': [],
                    'loose_coincident_info': []
                }
                
                # Add near/loose coincident info in old format
                if group['near_coincident']:
                    for vertex_data in group['near_coincident']:
                        distance = math.sqrt((group['coordinate'][0] - vertex_data['coordinate'][0])**2 + 
                                           (group['coordinate'][1] - vertex_data['coordinate'][1])**2)
                        old_format_entry['near_coincident_info'].append({
                            'group': {'vertices': [vertex_data['vertex']]},  # The 'group' key that was missing!
                            'distance': distance,
                            'confidence': 'high'
                        })
                
                if group['loose_coincident']:
                    for vertex_data in group['loose_coincident']:
                        distance = math.sqrt((group['coordinate'][0] - vertex_data['coordinate'][0])**2 + 
                                           (group['coordinate'][1] - vertex_data['coordinate'][1])**2)
                        old_format_entry['loose_coincident_info'].append({
                            'group': {'vertices': [vertex_data['vertex']]},  # The 'group' key that was missing!
                            'distance': distance,
                            'confidence': 'moderate'
                        })
                
                non_coincident_groups.append(old_format_entry)
        
        return non_coincident_groups
        
    except Exception as e:
        return []

def populate_tab3_list(widget):
    """Populate the Tab3 UI list with non-coincident vertex groups."""
    try:
        widget.coincident_list.clear()
        
        # Get the analysis data
        non_coincident_groups = find_non_coincident_vertices(widget.analyzer)
        
        for i, group_data in enumerate(non_coincident_groups, 1):
            coord = group_data['coordinate']
            vertices_data = group_data['vertices_data']
            eligible_vertices = group_data.get('eligible_vertices', vertices_data)  # Use eligible if available
            
            # Create main group item
            main_text = f"Group {i}: ({coord[0]:.3f}, {coord[1]:.3f}) - {len(eligible_vertices)} eligible vertices"
            
            try:
                from PySide import QtGui, QtCore
                item = QtGui.QListWidgetItem(main_text)
                
                # Make group item bold using Qt font
                bold_font = QtGui.QFont()
                bold_font.setBold(True)
                item.setFont(bold_font)
                
                item.setData(QtCore.Qt.UserRole, {'type': 'group', 'data': group_data})
                widget.coincident_list.addItem(item)
                
                # Sort eligible vertices by constraint count and group constrained pairs
                def sort_and_group_vertices(vertices):
                    """Sort vertices by constraint count and group constrained pairs adjacently."""
                    # First, sort by constraint count (descending)
                    sorted_vertices = sorted(vertices, key=lambda v: len(v['existing_constraints']), reverse=True)
                    
                    # Then group constrained pairs adjacently
                    processed = set()
                    grouped_vertices = []
                    
                    for vertex_data in sorted_vertices:
                        if vertex_data['vertex'] in processed:
                            continue
                            
                        grouped_vertices.append(vertex_data)
                        processed.add(vertex_data['vertex'])
                        
                        # Find constrained partners in the same group and add them immediately after
                        for connection in vertex_data['constrained_to']:
                            partner_vertex = connection['vertex']
                            # Find the partner in our vertex list
                            for partner_data in sorted_vertices:
                                if (partner_data['vertex'] == partner_vertex and 
                                    partner_vertex not in processed):
                                    grouped_vertices.append(partner_data)
                                    processed.add(partner_vertex)
                                    break
                    
                    return grouped_vertices
                
                eligible_vertices = sort_and_group_vertices(eligible_vertices)
                
                # Add eligible vertex details with standard indentation
                for vertex_data in eligible_vertices:
                    # Calculate distance from vertex to group coordinate for tolerance indicator
                    vertex_coord = vertex_data['coordinate']
                    distance = math.sqrt((coord[0] - vertex_coord[0])**2 + (coord[1] - vertex_coord[1])**2)
                    
                    # Determine tolerance indicator
                    tolerance_info = ""
                    if distance < 1e-10:
                        tolerance_emoji = "✅"  # Exact match
                    elif distance <= 5e-6:  # NEAR_COINCIDENT_THRESHOLD
                        tolerance_emoji = "🔔"  # Near coincident
                        distance_um = distance * 1e6  # Convert to micrometers
                        rounded_um = math.ceil(distance_um * 100) / 100  # Round up to hundredth
                        tolerance_info = f" » Tight: {rounded_um:.2f}µm «"
                    elif distance <= 100e-6:  # LOOSE_COINCIDENT_THRESHOLD
                        tolerance_emoji = "⚠️"  # Loose coincident
                        distance_um = distance * 1e6  # Convert to micrometers
                        rounded_um = math.ceil(distance_um * 100) / 100  # Round up to hundredth
                        tolerance_info = f" » Loose: {rounded_um:.2f}µm «"
                    else:
                        tolerance_emoji = ""  # No indicator for distances beyond loose threshold
                    
                    # Build indicators that go before geometry name
                    prefix_indicators = tolerance_emoji
                    
                    # Standard indentation for eligible vertices
                    position_name = vertex_data['position_name'].lower()  # "Start" -> "start"
                    detail_text = f"  └─ {prefix_indicators}{vertex_data['geometry_name']} ({position_name}) {vertex_data['gui_vertex_name']}"
                    
                    # Add constraint info if exists
                    if vertex_data['existing_constraints']:
                        constraint_targets = [conn['gui_vertex_name'] for conn in vertex_data['constrained_to']]
                        detail_text += f" 🔗 {', '.join(constraint_targets)}"
                    
                    # Add tolerance info
                    if tolerance_info:
                        detail_text += tolerance_info
                    
                    # Add construction indicator at the end
                    if vertex_data['is_construction']:
                        detail_text += " 🔧"
                    
                    detail_item = QtGui.QListWidgetItem(detail_text)
                    
                    # Store geometry index for highlighting compatibility
                    detail_item.setData(QtCore.Qt.UserRole, {
                        'type': 'edge', 
                        'geo_idx': vertex_data['vertex'][0],  # geometry index for highlighting
                        'data': vertex_data
                    })
                    widget.coincident_list.addItem(detail_item)
                    
            except Exception as e:
                # Fallback: just add text without fancy UI
                try:
                    item = QtGui.QListWidgetItem(main_text)
                    widget.coincident_list.addItem(item)
                except:
                    pass
        
    except Exception as e:
        # Try to at least clear the list
        try:
            widget.coincident_list.clear()
        except:
            pass

def coincident_all_vertices(widget, analyzer=None):
    """Main function to add coincident constraints for all non-coincident vertices."""
    # If no analyzer provided, get it from widget
    if analyzer is None:
        analyzer = widget.analyzer
        
    # Start transaction for safe rollback
    sketch = analyzer.sketch
    sketch.Document.openTransaction("Coincident All Constraints")
    
    try:
        # Get comprehensive non-coincident analysis
        non_coincident_vertices = find_non_coincident_vertices(analyzer)
        
        constraints_added = 0
        
        for i, vertex_group in enumerate(non_coincident_vertices):
            coordinate = vertex_group['coordinate']
            eligible_vertices = vertex_group.get('eligible_vertices', [])
            
            if len(eligible_vertices) < 2:
                continue
            
            # Find the best anchor using comprehensive analysis
            group_for_anchor = {'vertices': eligible_vertices, 'coordinate': coordinate}
            anchor_data = find_best_anchor_in_group(sketch, group_for_anchor)
            anchor_vertex = anchor_data['vertex']
            
            # Add constraints from other eligible vertices to anchor
            group_constraints_added = 0
            for vertex_data in eligible_vertices:
                vertex = vertex_data['vertex']
                
                if vertex == anchor_vertex:
                    continue
                
                # Check if constraint already exists (comprehensive check)
                if _constraint_exists_comprehensive(sketch, vertex, anchor_vertex):
                    continue
                
                # Add the constraint
                try:
                    geo1, pos1 = vertex
                    geo2, pos2 = anchor_vertex
                    constraint_index = sketch.addConstraint(Sketcher.Constraint('Coincident', geo1, pos1, geo2, pos2))
                    
                    if constraint_index >= 0:
                        group_constraints_added += 1
                        
                except Exception as e:
                    pass
            
            constraints_added += group_constraints_added
        
        # Commit transaction
        sketch.Document.commitTransaction()
        
        # Force sketch recompute and solver update
        solver_result = analyzer.sketch.solve()
        
        # Trigger re-analysis instead of manual recompute
        widget.analyze_sketch()
        
    except Exception as e:
        # Rollback transaction on error
        sketch.Document.abortTransaction()
        raise

# Button callback function - this should match what Tab3 expects
def coincident_all_vertices_button_callback(widget):
    """Button callback function for the Coincident All Vertices button."""
    return coincident_all_vertices(widget, widget.analyzer)

def coincident_selected_vertices(widget):
    """Add coincident constraints for selected vertex group using anchor method."""
    current_item = widget.coincident_list.currentItem()
    if not current_item:
        return
        
    data = current_item.data(QtCore.Qt.UserRole)
    if not data:
        return
        
    # Handle both edge items and location items
    if data.get('type') == 'edge':
        vertex_data = data['data']
    elif data.get('type') == 'location':
        vertex_data = data['data']
    else:
        vertex_data = data  # Legacy format
        
    widget.sketch.Document.openTransaction("Add Selected Coincident Constraints")
    
    try:
        constrained_groups = vertex_data.get('constrained_groups', [])
        
        if len(constrained_groups) > 1:
            # Use the first group as anchor, connect all other groups to it
            anchor_group = constrained_groups[0]
            anchor_vertex = anchor_group[0]  # First vertex in first group
            
            for group in constrained_groups[1:]:
                for vertex in group:
                    if anchor_vertex[0] != vertex[0]:  # Don't constrain same geometry to itself
                        widget.sketch.addConstraint(Sketcher.Constraint(
                            'Coincident', anchor_vertex[0], anchor_vertex[1], 
                            vertex[0], vertex[1]))
                        
        widget.sketch.Document.commitTransaction()
        widget.analyze_sketch()  # Re-analyze
        
    except Exception as e:
        widget.sketch.Document.abortTransaction()

def setup_coincident_tab(widget):
    """Setup the non-coincident vertices tab."""
    from PySide import QtGui, QtCore
    
    tab = QtGui.QWidget()
    layout = QtGui.QVBoxLayout(tab)
    
    widget.coincident_list = QtGui.QListWidget()
    widget.coincident_list.itemEntered.connect(widget._on_hover)
    widget.coincident_list.itemClicked.connect(widget._on_coincident_selected)
    layout.addWidget(widget.coincident_list)
    
    # Buttons
    button_layout = QtGui.QHBoxLayout()
    
    coincident_all_btn = QtGui.QPushButton("Coincident All")
    coincident_all_btn.clicked.connect(lambda: coincident_all_vertices_button_callback(widget))
    button_layout.addWidget(coincident_all_btn)
    
    coincident_selected_btn = QtGui.QPushButton("Coincident Selected")
    coincident_selected_btn.clicked.connect(lambda: coincident_selected_vertices(widget))
    button_layout.addWidget(coincident_selected_btn)
    
    layout.addLayout(button_layout)
    widget.tab_widget.addTab(tab, "Non-Coincident Vertices")

def get_non_coincident_vertices_basic(analyzer):
    """Legacy function for backward compatibility."""
    return find_non_coincident_vertices(analyzer)

# Temporary diagnostic functions (kept for compatibility but with no output)
def diagnose_bspline_constraints(analyzer, vertices_data):
    """Temporary diagnostic function to check B-spline constraint data."""
    # Find B-spline vertices
    bspline_vertices = []
    for vertex_data in vertices_data:
        if vertex_data['is_bspline']:
            bspline_vertices.append(vertex_data)
    
    if not bspline_vertices:
        return
    
    # Analysis logic preserved but output removed
    for i, vertex_data in enumerate(bspline_vertices, 1):
        constraint_count = len(vertex_data['existing_constraints'])
        connection_count = len(vertex_data['constrained_to'])

def diagnose_circle_constraints(analyzer, vertices_data):
    """Check if circle centers show InternalAlignment to B-splines."""
    # Find circle center vertices near the B-spline coordinates
    target_coords = [(29.1882, 25.475), (23.316, -9.58546)]  # B-spline start/end
    
    for coord in target_coords:
        for vertex_data in vertices_data:
            if (vertex_data['geometry_type'] == 'Part::GeomCircle' and 
                vertex_data['position_name'] == 'Center'):
                
                # Check if this circle center is near the B-spline coordinate
                distance = ((vertex_data['coordinate'][0] - coord[0])**2 + 
                           (vertex_data['coordinate'][1] - coord[1])**2)**0.5
                
                if distance < 1e-3:  # Within 1mm
                    # Analysis logic preserved but output removed
                    constraint_count = len(vertex_data['existing_constraints'])
                    connection_count = len(vertex_data['constrained_to'])

def diagnose_bspline_connections(analyzer, vertices_data):
    """Check if B-spline connection detection is working."""
    for vertex_data in vertices_data:
        if vertex_data['is_bspline']:
            # Analysis logic preserved but output removed
            has_connected_via_circle = 'connected_via_circle' in vertex_data
            constraint_count = len(vertex_data['existing_constraints'])