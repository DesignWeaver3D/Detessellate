# SPDX-License-Identifier: LGPL-2.1-or-later
# SPDX-FileCopyrightText: 2024 DesignWeaver3D
# SPDX-FileNotice: Part of the Detessellate addon.
# ============================================================================
# TAB2: Duplicate Edge Detection with Tolerance v75
# ============================================================================
# Complete Tab2 implementation - handles all duplicate detection logic
# - Enhanced tolerance-based detection (5µm tight, 100µm loose)
# - Clean interface: analyze_duplicates(widget) called by main
# - Comprehensive duplicate geometry checking with proper UI integration
# ============================================================================

import FreeCAD as App
import Sketcher
import math
from typing import List, Tuple, Dict, Any
from PySide6 import QtCore, QtWidgets

# Tolerance thresholds - same as Tab3 for consistency
NEAR_COINCIDENT_THRESHOLD = 5e-6    # 5 micrometers - tight tolerance
LOOSE_COINCIDENT_THRESHOLD = 100e-6  # 100 micrometers - loose tolerance

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
    except Exception:
        return f"Geometry{geo_idx+1}"

def get_geometry_endpoints(geometry, geo_idx, sketch):
    """Get geometry endpoints using solver coordinates with enhanced error handling."""
    try:
        if hasattr(geometry, 'TypeId'):
            # Map external raw types to their segment/arc equivalents for endpoint lookup
            external_type_map = {
                'Part::GeomLine':      'Part::GeomLineSegment',
                'Part::GeomCircle':    'Part::GeomArcOfCircle',
                'Part::GeomEllipse':   'Part::GeomArcOfEllipse',
                'Part::GeomHyperbola': 'Part::GeomArcOfHyperbola',
                'Part::GeomParabola':  'Part::GeomArcOfParabola',
            }
            effective_type = external_type_map.get(geometry.TypeId, geometry.TypeId)

            if effective_type in ['Part::GeomLineSegment', 'Part::GeomArcOfCircle',
                                   'Part::GeomArcOfEllipse', 'Part::GeomArcOfHyperbola',
                                   'Part::GeomArcOfParabola', 'Part::GeomBSplineCurve']:
                start_point = sketch.getPoint(geo_idx, 1)
                end_point = sketch.getPoint(geo_idx, 2)
                return (start_point.x, start_point.y), (end_point.x, end_point.y)

            elif effective_type == 'Part::GeomCircle':
                # Full circle — check if start == end
                try:
                    start_point = sketch.getPoint(geo_idx, 1)
                    end_point = sketch.getPoint(geo_idx, 2)
                    start_dist = math.sqrt((start_point.x - end_point.x)**2 +
                                           (start_point.y - end_point.y)**2)
                    if start_dist < 1e-10:
                        return None, None
                    else:
                        return (start_point.x, start_point.y), (end_point.x, end_point.y)
                except Exception:
                    return None, None

            elif effective_type == 'Part::GeomPoint':
                point = sketch.getPoint(geo_idx, 1)
                return (point.x, point.y), (point.x, point.y)

    except Exception as e:
        return None, None

    return None, None

def normalize_type(geo, geo_idx, sketch):
    """Normalize TypeId — external geometry uses base curve types instead of segment/arc types."""
    type_id = geo.TypeId if hasattr(geo, 'TypeId') else type(geo).__name__

    # External geometry reports base curve types; map to internal segment/arc equivalents
    # For curve types that can be either full or partial, use getPoint to distinguish
    if type_id == 'Part::GeomLine':
        return 'Part::GeomLineSegment'

    if type_id in ('Part::GeomCircle', 'Part::GeomEllipse',
                   'Part::GeomHyperbola', 'Part::GeomParabola'):
        # Check if partial (arc) or full via getPoint distance
        arc_map = {
            'Part::GeomCircle':    'Part::GeomArcOfCircle',
            'Part::GeomEllipse':   'Part::GeomArcOfEllipse',
            'Part::GeomHyperbola': 'Part::GeomArcOfHyperbola',
            'Part::GeomParabola':  'Part::GeomArcOfParabola',
        }
        try:
            p1 = sketch.getPoint(geo_idx, 1)
            p2 = sketch.getPoint(geo_idx, 2)
            dist = math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
            if dist > 1e-10:
                return arc_map[type_id]
        except Exception:
            pass

    return type_id

def are_geometries_duplicate_with_tolerance(geo1_data, geo2_data, sketch):
    """Enhanced duplicate checking with tolerance support."""
    geo1_idx, geometry1 = geo1_data
    geo2_idx, geometry2 = geo2_data

    if geo1_idx == geo2_idx:
        return False, 0.0, "same_geometry"

    # Normalize types before comparing so external GeomCircle-arcs match GeomArcOfCircle
    type1 = normalize_type(geometry1, geo1_idx, sketch)
    type2 = normalize_type(geometry2, geo2_idx, sketch)

    if type1 != type2:
        return False, 0.0, "different_types"
    
    # Get endpoints for both geometries
    start1, end1 = get_geometry_endpoints(geometry1, geo1_idx, sketch)
    start2, end2 = get_geometry_endpoints(geometry2, geo2_idx, sketch)
    
    if start1 is None or start2 is None:
        return False, 0.0, "no_endpoints"
    
    # Check endpoint distances for both orientations
    patterns_to_check = [
        # Forward: start1->start2, end1->end2  
        (start1, start2, end1, end2, "forward"),
        # Reverse: start1->end2, end1->start2
        (start1, end2, end1, start2, "reverse")
    ]
    
    for s1, s2, e1, e2, direction in patterns_to_check:
        start_dist = math.sqrt((s1[0] - s2[0])**2 + (s1[1] - s2[1])**2)
        end_dist = math.sqrt((e1[0] - e2[0])**2 + (e1[1] - e2[1])**2)
        max_dist = max(start_dist, end_dist)
        
        # Enhanced tolerance checking
        if start_dist <= LOOSE_COINCIDENT_THRESHOLD and end_dist <= LOOSE_COINCIDENT_THRESHOLD:
            return True, max_dist, direction
    
    return False, 0.0, "endpoints_too_far"

def find_duplicate_geometry(analyzer):
    """Find duplicate or overlapping geometry with tolerance support."""
    duplicates = []
    checked = set()
    
    sketch = analyzer.sketch
    all_geometry = analyzer.all_geometry
    
    for i, (geo_idx1, geo1) in enumerate(all_geometry):
        if geo_idx1 in checked:
            continue
            
        group = [{'geo_idx': geo_idx1, 'geometry': geo1, 'constraints': count_constraints(analyzer, geo_idx1)}]
        
        for j, (geo_idx2, geo2) in enumerate(all_geometry[i+1:], i+1):
            if geo_idx2 in checked:
                continue
                
            # Use enhanced tolerance-based duplicate checking
            is_duplicate, max_distance, direction = are_geometries_duplicate_with_tolerance(
                (geo_idx1, geo1), (geo_idx2, geo2), sketch)
            
            if is_duplicate:
                group.append({'geo_idx': geo_idx2, 'geometry': geo2, 'constraints': count_constraints(analyzer, geo_idx2)})
                checked.add(geo_idx2)
                
        if len(group) > 1:
            # Sort by constraint count (ascending) to recommend least constrained for deletion
            group.sort(key=lambda x: (0 if x['geo_idx'] < 0 else 1, x['constraints']), reverse=False)

            duplicates.append(group)
            checked.add(geo_idx1)
                
    return duplicates

def count_constraints(analyzer, geo_idx):
    """Count constraints applied to a geometry element."""
    count = 0
    for constraint in analyzer.constraints:
        if constraint.First == geo_idx or constraint.Second == geo_idx:
            count += 1
    return count

def populate_duplicate_list(widget):
    """Populate the duplicate list UI with found duplicates."""
    try:
        widget.duplicate_list.clear()
        
        # Get duplicates from analysis_data
        duplicates = getattr(widget.analysis_data, 'duplicates', [])
        
        for group_idx, group in enumerate(duplicates):
            # Add group header
            header_item = QtWidgets.QListWidgetItem(f"--- Duplicate Group {group_idx + 1} ---")
            header_item.setData(QtCore.Qt.UserRole, {'type': 'header'})
            widget.duplicate_list.addItem(header_item)
            
            for item in group:
                geo_idx = item['geo_idx']
                geometry = item['geometry']
                constraints = item['constraints']
                
                geo_name = get_geometry_name(geo_idx, geometry)
                recommended = " [RECOMMENDED]" if item == group[0] else ""
                
                list_item = QtWidgets.QListWidgetItem(
                    f"  {geo_name} (constraints: {constraints}){recommended}")
                list_item.setData(QtCore.Qt.UserRole, {
                    'type': 'geometry', 
                    'data': item, 
                    'group': group
                })
                widget.duplicate_list.addItem(list_item)
        
    except Exception as e:
        import traceback

def analyze_duplicates(widget):
    """Main Tab2 function - find duplicates and populate UI."""
    try:
        # Find duplicates using enhanced tolerance-based detection
        duplicates = find_duplicate_geometry(widget.analyzer)
        
        # Store in widget for UI
        widget.analysis_data.duplicates = duplicates
        
        # Populate UI list
        populate_duplicate_list(widget)
        
    except Exception as e:
        import traceback

def delete_selected_duplicates(widget):
    """Delete selected duplicate edges from the sketch."""
    try:
        selected_items = widget.duplicate_list.selectedItems()
        if not selected_items:
            return 0
        
        sketch = widget.analyzer.sketch
        sketch.Document.openTransaction("Delete Duplicate Edges")
        
        deleted_count = 0
        
        # Get selected duplicates and sort by geometry index (highest first)
        duplicates_to_delete = []
        for item in selected_items:
            try:
                from PySide6 import QtCore
                data = item.data(QtCore.Qt.UserRole)
                if data and data.get('type') == 'geometry':
                    duplicates_to_delete.append(data['data'])
            except Exception:
                pass
        
        # Sort by highest geometry index first to avoid index shifting
        duplicates_to_delete.sort(key=lambda x: x['geo_idx'], reverse=True)
        
        for duplicate in duplicates_to_delete:
            geo_idx = duplicate['geo_idx']
            
            try:
                sketch.delGeometry(geo_idx)
                deleted_count += 1
                
            except Exception as e:
                pass
        
        sketch.Document.commitTransaction()
        
        # Trigger recompute + re-analysis + UI refresh
        widget.analyze_sketch()
        
        return deleted_count
        
    except Exception as e:
        sketch.Document.abortTransaction()
        return 0

def delete_recommended_duplicates(widget):
    """Delete all recommended duplicate geometries."""
    try:
        sketch = widget.analyzer.sketch
        if not sketch:
            print("DEBUG: no sketch")
            return

        duplicates_data = find_duplicate_geometry(widget.analyzer)

        if not duplicates_data:
            print("DEBUG: no duplicates found")
            return

        indices_to_delete = []
        for group in duplicates_data:

            for duplicate_item in group[1:]:

                indices_to_delete.append(duplicate_item['geo_idx'])


        
        sketch.Document.openTransaction("Delete All Recommended Duplicates")

        if not indices_to_delete:
            sketch.Document.abortTransaction()
            return
        
        # Sort in descending order to avoid index shifting issues
        indices_to_delete.sort(reverse=True)
        
        # Delete geometries
        deleted_count = 0
        for geo_idx in indices_to_delete:
            if geo_idx < 0:
                print(f"WARNING: attempted to delete external geometry {geo_idx}, skipping")
                continue
            try:
                sketch.delGeometry(geo_idx)
                deleted_count += 1
                
            except Exception as e:
                pass
        
        sketch.Document.commitTransaction()
        
        # Trigger recompute + re-analysis + UI refresh
        widget.analyze_sketch()
        
    except Exception as e:
        sketch.Document.abortTransaction()

def setup_duplicate_tab(widget):
    """Setup the duplicate geometry tab UI."""
    tab = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(tab)

    widget.duplicate_list = QtWidgets.QListWidget()
    widget.duplicate_list.itemEntered.connect(widget._on_hover)
    widget.duplicate_list.itemClicked.connect(widget._on_duplicate_selected)
    layout.addWidget(widget.duplicate_list)

    # Button layouts
    button_layout1 = QtWidgets.QHBoxLayout()
    delete_all_btn = QtWidgets.QPushButton("Delete All Recommended")
    delete_all_btn.clicked.connect(lambda: delete_recommended_duplicates(widget))
    button_layout1.addWidget(delete_all_btn)

    button_layout2 = QtWidgets.QHBoxLayout()
    delete_selected_btn = QtWidgets.QPushButton("Delete Selected")
    delete_selected_btn.clicked.connect(lambda: delete_selected_duplicates(widget))
    button_layout2.addWidget(delete_selected_btn)

    layout.addLayout(button_layout1)
    layout.addLayout(button_layout2)
    widget.tab_widget.addTab(tab, "Duplicate Geometry")
