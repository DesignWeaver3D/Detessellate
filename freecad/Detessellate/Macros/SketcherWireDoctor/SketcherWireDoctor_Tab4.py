#!/usr/bin/env python3
"""
SketcherWireDoctor - Tab4: Wire Topology Analysis (Phased Approach)

Implements the phased approach to sketch topology analysis:
Phase 1: Constraint & Connectivity Resolution
Phase 2: Isolation Detection
Phase 3: Bridge Detection
Phase 4: Subdivision Detection
Phase 5: T-junction Cleanup

Author: Refactored based on analysis findings
Version: 1.0 (Phased approach implementation)
"""

import FreeCAD as App
import Part
from PySide import QtCore, QtGui
from typing import Any, List, Dict, Tuple, Set, Optional
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum

# Import from main module
try:
    from SketcherWireDoctor_Main import GeometryUtils, Config, round_coord
except ImportError:
    # Fallback for when running as standalone
    pass

# Local constants and utilities to avoid circular imports
TOLERANCE = 1e-6
MAX_PATH_LENGTH = 10
ARC_TYPES = [
    "Part::GeomArcOfCircle",
    "Part::GeomArcOfEllipse",
    "Part::GeomArcOfHyperbola",
    "Part::GeomArcOfParabola"
]

GEOMETRY_TYPE_MAP = {
    'Part::GeomLineSegment': 'Line',
    'Part::GeomArcOfCircle': 'Arc',
    'Part::GeomCircle': 'Circle',
    'Part::GeomBSplineCurve': 'BSpline',
    'Part::GeomEllipse': 'Ellipse'
}

class TopologyIssueType(Enum):
    ISOLATION = "isolation"
    GEOMETRIC_VALIDITY = "geometric_validity"
    BRIDGE = "bridge"
    SUBDIVISION = "subdivision"
    T_JUNCTION = "t_junction"

@dataclass
class TopologyIssue:
    geo_idx: int
    geometry: Any
    issue_type: TopologyIssueType
    description: str
    severity: int = 1  # Higher = more critical

class PhaseResult:
    def __init__(self, phase_name: str):
        self.phase_name = phase_name
        self.issues: List[TopologyIssue] = []
        self.success = True
        self.error_message = ""

    def add_issue(self, issue: TopologyIssue):
        self.issues.append(issue)

def round_coord(point, digits: int = 7):
    """Round coordinates for precision comparison."""
    return (round(point.x, digits), round(point.y, digits))

def get_geometry_name(geo_idx: int, geometry) -> str:
    """Get the FreeCAD display name for geometry (1-based indexing)."""
    geo_type = GEOMETRY_TYPE_MAP.get(geometry.TypeId, 'Geometry')
    return f"{geo_idx + 1}-{geo_type}"

def get_geometry_endpoints(geometry):
    """Get start and end coordinates for any geometry type."""
    start_coord = None
    end_coord = None

    if geometry.TypeId == "Part::GeomLineSegment":
        start_coord = round_coord(geometry.StartPoint)
        end_coord = round_coord(geometry.EndPoint)
    elif geometry.TypeId in ARC_TYPES:
        if hasattr(geometry, 'StartPoint') and hasattr(geometry, 'EndPoint'):
            start_coord = round_coord(geometry.StartPoint)
            end_coord = round_coord(geometry.EndPoint)
    elif geometry.TypeId in ["Part::GeomBSplineCurve", "Part::GeomEllipse"]:
        if hasattr(geometry, 'StartPoint') and hasattr(geometry, 'EndPoint'):
            start_coord = round_coord(geometry.StartPoint)
            end_coord = round_coord(geometry.EndPoint)

    return start_coord, end_coord

class WireTopologyAnalyzer:
    """Phased topology analyzer following the dependency hierarchy."""

    def __init__(self, sketch):
        self.sketch = sketch
        self.normal_geometry = []
        self.construction_geometry = []
        self.constraint_graph = {}
        self.connectivity_graph = {}
        self.bspline_resolution_map = {}

        # Results from each phase
        self.phase_results = {}

    def analyze(self) -> Dict[TopologyIssueType, List[TopologyIssue]]:
        """Run the complete phased analysis."""
        App.Console.PrintMessage("\n" + "=" * 80 + "\n")
        App.Console.PrintMessage("🔍 PHASED WIRE TOPOLOGY ANALYSIS v1.0\n")
        App.Console.PrintMessage("=" * 80 + "\n")

        try:
            # Prepare geometry data
            self._prepare_geometry()

            # Phase 1: Constraint & Connectivity Resolution
            phase1_result = self._phase1_constraint_resolution()
            self.phase_results[1] = phase1_result

            if not phase1_result.success:
                App.Console.PrintError(f"Phase 1 failed: {phase1_result.error_message}\n")
                return self._collect_all_issues()

            # Phase 2: Isolation Detection
            phase2_result = self._phase2_isolation_detection()
            self.phase_results[2] = phase2_result

            # Phase 2.5: Geometric Validity Check
            phase25_result = self._phase25_geometric_validity()
            self.phase_results[2.5] = phase25_result

            # Phase 3: Bridge Detection
            phase3_result = self._phase3_bridge_detection()
            self.phase_results[3] = phase3_result

            # Phase 4: Subdivision Detection (only if bridges are handled)
            phase4_result = self._phase4_subdivision_detection()
            self.phase_results[4] = phase4_result

            # Phase 5: T-junction Cleanup
            phase5_result = self._phase5_tjunction_cleanup()
            self.phase_results[5] = phase5_result

            return self._collect_all_issues()

        except Exception as e:
            App.Console.PrintError(f"Analysis failed: {str(e)}\n")
            return {}

    def _prepare_geometry(self):
        """Separate normal and construction geometry."""
        self.normal_geometry = []
        self.construction_geometry = []

        for i, geo in enumerate(self.sketch.Geometry):
            if self.sketch.getConstruction(i):
                self.construction_geometry.append((i, geo))
            else:
                self.normal_geometry.append((i, geo))

        App.Console.PrintMessage(f"📊 Geometry: {len(self.normal_geometry)} normal, {len(self.construction_geometry)} construction\n")

    def _phase1_constraint_resolution(self) -> PhaseResult:
        """Phase 1: Build accurate constraint and connectivity maps, especially for B-splines."""
        App.Console.PrintMessage("🔧 Phase 1: Constraint & Connectivity Resolution\n")

        result = PhaseResult("Constraint Resolution")

        try:
            # Build constraint connectivity map
            self.constraint_graph = self._build_constraint_connectivity()

            # Resolve B-spline connectivity through construction circles
            self.bspline_resolution_map = self._resolve_bspline_connectivity()

            # Build complete connectivity graph
            self.connectivity_graph = self._build_complete_connectivity_graph()

            # Validate connectivity resolution
            validation_issues = self._validate_connectivity_resolution()
            for issue in validation_issues:
                result.add_issue(issue)

            App.Console.PrintMessage(f"✅ Phase 1 complete: {len(result.issues)} connectivity issues found\n")

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            App.Console.PrintError(f"❌ Phase 1 failed: {e}\n")

        return result

    def _build_geometry_connection_graph(self) -> Dict[str, any]:
        """Build normal geometry graph with constraint-informed coordinate replacement."""
        App.Console.PrintMessage("🔗 Building normal geometry graph with constraint coordinate replacement...\n")
        
        # Step 1: Collect all constraint coordinates
        constraint_coords = []
        constraint_count = 0
        
        for constraint in self.sketch.Constraints:
            try:
                if constraint.Type == "Coincident":
                    coord1 = self.sketch.getPoint(constraint.First, constraint.FirstPos)
                    coord2 = self.sketch.getPoint(constraint.Second, constraint.SecondPos)
                    
                    constraint_coords.append((coord1.x, coord1.y, coord1.z))
                    constraint_coords.append((coord2.x, coord2.y, coord2.z))
                    constraint_count += 1
                    
                elif constraint.Type == "InternalAlignment":
                    coord1 = self.sketch.getPoint(constraint.First, constraint.FirstPos)
                    coord2 = self.sketch.getPoint(constraint.Second, constraint.SecondPos)
                    
                    constraint_coords.append((coord1.x, coord1.y, coord1.z))
                    constraint_coords.append((coord2.x, coord2.y, coord2.z))
                    constraint_count += 1
                    
            except Exception as e:
                App.Console.PrintMessage(f"⚠️ Error processing constraint: {e}\n")
                continue
        
        # Remove duplicates from constraint coordinates
        unique_constraint_coords = list(set(constraint_coords))
        App.Console.PrintMessage(f"🔗 Collected {len(unique_constraint_coords)} unique constraint coordinates from {constraint_count} constraints\n")
        
        # Step 2: Build vertex-to-vertex graph from normal geometry with coordinate replacement
        def find_nearest_constraint_coord(geom_coord, tolerance=1e-3):
            """Find nearest constraint coordinate within tolerance."""
            geom_tuple = (geom_coord[0], geom_coord[1], geom_coord[2] if len(geom_coord) > 2 else 0)
            
            for constraint_coord in unique_constraint_coords:
                distance = ((geom_tuple[0] - constraint_coord[0])**2 + 
                           (geom_tuple[1] - constraint_coord[1])**2 + 
                           (geom_tuple[2] - constraint_coord[2])**2)**0.5
                
                if distance <= tolerance:
                    return constraint_coord
            
            return geom_tuple  # Use original if no constraint coord found nearby
        
        vertex_graph = defaultdict(set)
        edge_map = {}
        coordinate_replacements = 0
        
        for geo_idx, geometry in self.normal_geometry:
            # Get geometry endpoints with full precision
            start_coord, end_coord = get_geometry_endpoints(geometry)
            
            if not start_coord or not end_coord or start_coord == end_coord:
                continue
            
            # Replace with nearest constraint coordinates if available
            original_start = (start_coord[0], start_coord[1], start_coord[2] if len(start_coord) > 2 else 0)
            original_end = (end_coord[0], end_coord[1], end_coord[2] if len(end_coord) > 2 else 0)
            
            constraint_start = find_nearest_constraint_coord(original_start)
            constraint_end = find_nearest_constraint_coord(original_end)
            
            # Count replacements for debugging
            if constraint_start != original_start:
                coordinate_replacements += 1
            if constraint_end != original_end:
                coordinate_replacements += 1
            
            if constraint_start != constraint_end:  # Valid edge
                # Add bidirectional connection
                vertex_graph[constraint_start].add(constraint_end)
                vertex_graph[constraint_end].add(constraint_start)
                
                # Track which geometry creates this edge
                edge_key = tuple(sorted([constraint_start, constraint_end]))
                edge_map[edge_key] = geo_idx
        
        App.Console.PrintMessage(f"🔗 Made {coordinate_replacements} coordinate replacements using constraint system\n")
        
        # Convert to regular dict with lists
        result = {
            "vertex_graph": {vertex: list(connections) for vertex, connections in vertex_graph.items()},
            "edge_map": edge_map
        }
        
        App.Console.PrintMessage(f"🔗 Built vertex graph: {len(result['vertex_graph'])} vertices, {len(result['edge_map'])} edges\n")
        return result

    def _find_all_loops(self) -> List[List[int]]:
        """Find all closed loops using constraint-informed vertex traversal with edge uniqueness."""
        App.Console.PrintMessage(f"🔍 Constraint-informed vertex traversal loop detection...\n")
        
        # Build constraint-informed vertex graph
        graph_data = self._build_geometry_connection_graph()
        vertex_graph = graph_data["vertex_graph"]
        edge_map = graph_data["edge_map"]
        
        if not vertex_graph:
            App.Console.PrintMessage("🔍 No vertex connections found\n")
            return []
        
        App.Console.PrintMessage(f"🔍 Graph: {len(vertex_graph)} vertices, {len(edge_map)} edges\n")
        
        # Safety limits
        all_loops = []
        max_loops = 1000
        max_path_length = len(self.normal_geometry)  # No loop can use more edges than exist
        
        App.Console.PrintMessage(f"🔍 Safety limits: max_loops={max_loops}, max_path_length={max_path_length}\n")
        
        def dfs_find_loops(start_vertex, current_vertex, path_vertices, used_edges):
            """DFS to find loops starting from start_vertex, currently at current_vertex."""
            
            # Safety checks
            if len(all_loops) >= max_loops:
                return
            if len(path_vertices) >= max_path_length:
                return
            
            # Get neighbors of current vertex
            neighbors = vertex_graph.get(current_vertex, [])
            
            for next_vertex in neighbors:
                # Get the edge between current and next vertex
                edge_key = tuple(sorted([current_vertex, next_vertex]))
                edge_geo_idx = edge_map.get(edge_key)
                
                if edge_geo_idx is None:
                    continue  # No edge found (shouldn't happen)
                
                if edge_geo_idx in used_edges:
                    continue  # Skip already used edges
                
                if next_vertex == start_vertex and len(path_vertices) >= 3:
                    # Found a loop back to start with at least 3 vertices (2+ edges)
                    loop_edges = list(used_edges) + [edge_geo_idx]
                    all_loops.append(loop_edges)
                    App.Console.PrintMessage(f"🔍 Found loop {len(all_loops)}: {loop_edges}\n")
                    
                    # Safety check after finding loop
                    if len(all_loops) >= max_loops:
                        App.Console.PrintMessage(f"⚠️ Reached maximum loop limit ({max_loops}), stopping search\n")
                        return
                    continue
                
                if next_vertex not in path_vertices:  # Avoid revisiting vertices
                    # Continue DFS with this next vertex
                    new_path_vertices = path_vertices + [next_vertex]
                    new_used_edges = used_edges | {edge_geo_idx}
                    dfs_find_loops(start_vertex, next_vertex, new_path_vertices, new_used_edges)
                    
                    # Safety check after recursion
                    if len(all_loops) >= max_loops:
                        return
        
        # Start DFS from each vertex
        vertices = list(vertex_graph.keys())
        for i, start_vertex in enumerate(vertices):
            if len(all_loops) >= max_loops:
                break
                
            # Progress reporting
            if i > 0 and i % 10 == 0:
                App.Console.PrintMessage(f"🔍 Progress: checked {i}/{len(vertices)} starting vertices, found {len(all_loops)} loops\n")
            
            initial_path = [start_vertex]
            initial_used_edges = set()
            dfs_find_loops(start_vertex, start_vertex, initial_path, initial_used_edges)
        
        # Remove duplicate loops (same edges in different order)
        unique_loops = []
        seen_signatures = set()
        
        for loop in all_loops:
            # Create canonical signature by sorting edge indices
            signature = tuple(sorted(loop))
            
            if signature not in seen_signatures:
                unique_loops.append(loop)
                seen_signatures.add(signature)
        
        App.Console.PrintMessage(f"🔍 Constraint-informed algorithm: Found {len(unique_loops)} unique loops\n")
        return unique_loops 

    def _phase2_isolation_detection(self) -> PhaseResult:
        """Phase 2: Find isolated geometry not connected to main wire systems."""
        App.Console.PrintMessage("🔧 Phase 2: Isolation Detection\n")

        result = PhaseResult("Isolation Detection")

        try:
            # Find orphaned geometry (completely disconnected)
            orphaned = self._find_orphaned_geometry()
            for geo_idx in orphaned:
                geometry = next(g for i, g in self.normal_geometry if i == geo_idx)
                issue = TopologyIssue(
                    geo_idx=geo_idx,
                    geometry=geometry,
                    issue_type=TopologyIssueType.ISOLATION,
                    description="Orphaned geometry (not connected to anything)",
                    severity=2
                )
                result.add_issue(issue)

            # Find floating/unconstrained geometry
            floating = self._find_floating_geometry()
            for geo_idx in floating:
                geometry = next(g for i, g in self.normal_geometry if i == geo_idx)
                issue = TopologyIssue(
                    geo_idx=geo_idx,
                    geometry=geometry,
                    issue_type=TopologyIssueType.ISOLATION,
                    description="Floating elements (unconstrained)",
                    severity=1
                )
                result.add_issue(issue)

            App.Console.PrintMessage(f"✅ Phase 2 complete: {len(result.issues)} isolation issues found\n")

        except Exception as e:
            App.Console.PrintError(f"❌ Phase 2 error: {e}\n")

        return result

    def _find_overlapping_geometry(self) -> List[Tuple[int, int]]:
        """Find pairs of normal geometry with real intersections within edge boundaries."""
        overlapping_pairs = []
        processed_pairs = set()  # Track pairs to avoid duplicates

        App.Console.PrintMessage("🔍 Checking for real geometric intersections...\n")

        # Check each unique pair of normal geometry
        for i, (idx1, geo1) in enumerate(self.normal_geometry):
            for j, (idx2, geo2) in enumerate(self.normal_geometry[i+1:], i+1):
                # Skip if we've already processed this pair
                pair_key = tuple(sorted([idx1, idx2]))
                if pair_key in processed_pairs:
                    continue
                processed_pairs.add(pair_key)

                geo_name1 = get_geometry_name(idx1, geo1)
                geo_name2 = get_geometry_name(idx2, geo2)

                try:
                    edge1 = geo1.toShape()
                    edge2 = geo2.toShape()

                    # Use section() to get only intersections within edge boundaries
                    section = edge1.section(edge2)

                    if section.Vertexes:
                        intersection_count = len(section.Vertexes)

                        # Log all intersections for debugging
                        App.Console.PrintMessage(f"🔍 Intersection check: {geo_name1} ↔ {geo_name2} ({intersection_count} points)\n")
                        for k, vertex in enumerate(section.Vertexes):
                            pt = vertex.Point
                            App.Console.PrintMessage(f"    Point {k+1}: ({pt[0]:.3f}, {pt[1]:.3f}, {pt[2]:.3f})\n")

                        # Check if these intersections are at geometry endpoints (likely constraints)
                        is_endpoint_connection = self._are_intersections_at_endpoints(
                            section.Vertexes, geo1, geo2
                        )

                        # Flag as problematic if:
                        # 1. Multiple intersections (like line passing through circle), OR
                        # 2. Single intersection that's NOT at endpoints (mid-edge crossing)
                        is_problematic = False

                        if intersection_count >= 2:
                            # Multiple intersections = likely unconstrained crossing/overlap
                            is_problematic = True
                            App.Console.PrintMessage(f"    → FLAGGED: Multiple intersections ({intersection_count})\n")
                        elif not is_endpoint_connection:
                            # Single intersection away from endpoints = likely X-crossing
                            is_problematic = True
                            App.Console.PrintMessage(f"    → FLAGGED: Mid-edge crossing\n")
                        else:
                            App.Console.PrintMessage(f"    → SKIPPED: Endpoint connection (likely constraint)\n")

                        if is_problematic:
                            overlapping_pairs.append((idx1, idx2))

                except Exception as e:
                    App.Console.PrintMessage(f"⚠️ Error checking intersection between {geo_name1} and {geo_name2}: {e}\n")

        App.Console.PrintMessage(f"✅ Found {len(overlapping_pairs)} pairs with problematic intersections\n")
        return overlapping_pairs

    def _are_intersections_at_endpoints(self, vertices, geo1, geo2) -> bool:
        """Check if intersection points are at geometry endpoints (indicating constraint connections)."""
        tolerance = 1e-3

        try:
            # Get endpoints of both geometries
            geo1_start, geo1_end = get_geometry_endpoints(geo1)
            geo2_start, geo2_end = get_geometry_endpoints(geo2)

            if not geo1_start or not geo1_end or not geo2_start or not geo2_end:
                return False

            # Collect all endpoints
            endpoints = [geo1_start, geo1_end, geo2_start, geo2_end]
            endpoint_coords = [(ep[0], ep[1], ep[2] if len(ep) > 2 else 0) for ep in endpoints]

            # Check if all intersections are very close to endpoints
            for vertex in vertices:
                pt = vertex.Point
                pt_coord = (pt[0], pt[1], pt[2])

                is_at_endpoint = False
                for ep_coord in endpoint_coords:
                    if (abs(pt_coord[0] - ep_coord[0]) < tolerance and
                        abs(pt_coord[1] - ep_coord[1]) < tolerance and
                        abs(pt_coord[2] - ep_coord[2]) < tolerance):
                        is_at_endpoint = True
                        break

                if not is_at_endpoint:
                    return False  # At least one intersection is not at an endpoint

            return True  # All intersections are at endpoints

        except Exception as e:
            App.Console.PrintMessage(f"⚠️ Error checking endpoints: {e}\n")
            return False  # If we can't determine, assume it's problematic

    def _filter_constrained_intersections(self, vertices, idx1: int, idx2: int) -> List:
        """Filter out intersection points that are legitimate constraint connections."""
        unconstrained = []

        # Get constrained points from the constraint system
        constrained_points = self._get_constrained_intersection_points(idx1, idx2)

        for vertex in vertices:
            pt = vertex.Point
            pt_rounded = (round(pt[0], 3), round(pt[1], 3), round(pt[2], 3))

            # Check if this intersection point is involved in constraints
            is_constrained = False
            for constrained_pt in constrained_points:
                if (abs(constrained_pt[0] - pt_rounded[0]) < 1e-3 and
                    abs(constrained_pt[1] - pt_rounded[1]) < 1e-3 and
                    abs(constrained_pt[2] - pt_rounded[2]) < 1e-3):
                    is_constrained = True
                    break

            if not is_constrained:
                unconstrained.append(vertex)

        return unconstrained

    def _get_constrained_intersection_points(self, idx1: int, idx2: int) -> Set[Tuple[float, float, float]]:
        """Get intersection points that are legitimate constraint connections between two geometries."""
        constrained_points = set()

        try:
            # Get endpoints of both geometries
            geo1 = next(g for i, g in self.normal_geometry if i == idx1)
            geo2 = next(g for i, g in self.normal_geometry if i == idx2)

            geo1_start, geo1_end = get_geometry_endpoints(geo1)
            geo2_start, geo2_end = get_geometry_endpoints(geo2)

            # Check all constraint combinations between these geometries
            for constraint in self.sketch.Constraints:
                if constraint.Type in ['Coincident', 'PointOnObject']:
                    try:
                        # Check if constraint involves both geometries
                        involves_geo1 = (constraint.First == idx1 or
                                       (hasattr(constraint, 'Second') and constraint.Second == idx1))
                        involves_geo2 = (constraint.First == idx2 or
                                       (hasattr(constraint, 'Second') and constraint.Second == idx2))

                        if involves_geo1 and involves_geo2:
                            # Get the constrained point
                            if constraint.First == idx1:
                                pt1 = self.sketch.getPoint(constraint.First, constraint.FirstPos)
                            elif hasattr(constraint, 'Second') and constraint.Second == idx1:
                                pt1 = self.sketch.getPoint(constraint.Second, constraint.SecondPos)
                            else:
                                continue

                            constrained_points.add((round(pt1.x, 3), round(pt1.y, 3), round(pt1.z, 3)))

                    except Exception as e:
                        App.Console.PrintMessage(f"⚠️ Error processing constraint: {e}\n")
                        continue

        except Exception as e:
            App.Console.PrintMessage(f"⚠️ Error getting constrained points for {idx1}, {idx2}: {e}\n")

        return constrained_points

    def _phase25_geometric_validity(self) -> PhaseResult:
        """Phase 2.5: Find normal geometry that creates real intersections (unconstrained crossings/overlaps)."""
        App.Console.PrintMessage("🔧 Phase 2.5: Geometric Validity Check (Real Intersections)\n")

        result = PhaseResult("Geometric Validity Check")

        try:
            # Find pairs with real unconstrained intersections
            overlapping_pairs = self._find_overlapping_geometry()

            # Create issues for geometry involved in real intersections
            flagged_geometry = set()
            for geo_idx1, geo_idx2 in overlapping_pairs:
                if geo_idx1 not in flagged_geometry:
                    geometry1 = next(g for i, g in self.normal_geometry if i == geo_idx1)
                    geo_name1 = get_geometry_name(geo_idx1, geometry1)

                    issue = TopologyIssue(
                        geo_idx=geo_idx1,
                        geometry=geometry1,
                        issue_type=TopologyIssueType.GEOMETRIC_VALIDITY,
                        description=f"Real intersection with {get_geometry_name(geo_idx2, next(g for i, g in self.normal_geometry if i == geo_idx2))}",
                        severity=3  # High severity - breaks 3D validity
                    )
                    result.add_issue(issue)
                    flagged_geometry.add(geo_idx1)

                if geo_idx2 not in flagged_geometry:
                    geometry2 = next(g for i, g in self.normal_geometry if i == geo_idx2)
                    geo_name2 = get_geometry_name(geo_idx2, geometry2)

                    issue = TopologyIssue(
                        geo_idx=geo_idx2,
                        geometry=geometry2,
                        issue_type=TopologyIssueType.GEOMETRIC_VALIDITY,
                        description=f"Real intersection with {get_geometry_name(geo_idx1, next(g for i, g in self.normal_geometry if i == geo_idx1))}",
                        severity=3  # High severity - breaks 3D validity
                    )
                    result.add_issue(issue)
                    flagged_geometry.add(geo_idx2)

            App.Console.PrintMessage(f"✅ Phase 2.5 complete: {len(result.issues)} geometric validity issues found\n")

        except Exception as e:
            App.Console.PrintError(f"❌ Phase 2.5 error: {e}\n")

        return result


    def _phase3_bridge_detection(self) -> PhaseResult:
        """Phase 3: Find bridge edges connecting separate loop systems."""
        App.Console.PrintMessage("🔧 Phase 3: Bridge Detection\n")

        result = PhaseResult("Bridge Detection")

        try:
            # Find connected components (for info only)
            components = self._find_connected_components()
            App.Console.PrintMessage(f"Found {len(components)} connected components\n")

            # Find bridge edges using simple loop membership analysis
            bridges = self._find_bridge_edges(components)
            for geo_idx in bridges:
                geometry = next(g for i, g in self.normal_geometry if i == geo_idx)
                issue = TopologyIssue(
                    geo_idx=geo_idx,
                    geometry=geometry,
                    issue_type=TopologyIssueType.BRIDGE,
                    description="Cross-wire bridge (connects separate loop systems)",
                    severity=3  # High severity - breaks wire topology rules
                )
                result.add_issue(issue)

            App.Console.PrintMessage(f"✅ Phase 3 complete: {len(result.issues)} bridge issues found\n")

        except Exception as e:
            App.Console.PrintError(f"❌ Phase 3 error: {e}\n")

        return result

    def _phase4_subdivision_detection(self) -> PhaseResult:
        """Phase 4: Find edges that subdivide individual wires with confidence ranking."""
        App.Console.PrintMessage("🔧 Phase 4: Subdivision Detection\n")

        result = PhaseResult("Subdivision Detection")

        try:
            # Find subdivision edges using edge removal analysis
            subdivisions = self._find_subdivision_edges_clean()

            if subdivisions:
                # Find all loops for confidence ranking
                all_loops = self._find_all_loops()

                for geo_idx in subdivisions:
                    geometry = next(g for i, g in self.normal_geometry if i == geo_idx)

                    # Get confidence level and reason
                    emoji, confidence_desc = self._get_subdivision_confidence(geo_idx, all_loops)

                    issue = TopologyIssue(
                        geo_idx=geo_idx,
                        geometry=geometry,
                        issue_type=TopologyIssueType.SUBDIVISION,
                        description=f"{emoji} {confidence_desc}",
                        severity=3 if emoji == "❗" else 2  # Higher severity for strong candidates
                    )
                    result.add_issue(issue)

            App.Console.PrintMessage(f"✅ Phase 4 complete: {len(result.issues)} subdivision issues found\n")

        except Exception as e:
            App.Console.PrintError(f"❌ Phase 4 error: {e}\n")

        return result

    def _phase5_tjunction_cleanup(self) -> PhaseResult:
        """Phase 5: Final cleanup of T-junctions and dangling edges."""
        App.Console.PrintMessage("🔧 Phase 5: T-junction Cleanup\n")

        result = PhaseResult("T-junction Cleanup")

        try:
            # Find true T-junctions (dangling edges)
            tjunction_data = self._find_true_tjunctions_with_details()
            for geo_idx, description in tjunction_data:
                geometry = next(g for i, g in self.normal_geometry if i == geo_idx)
                issue = TopologyIssue(
                    geo_idx=geo_idx,
                    geometry=geometry,
                    issue_type=TopologyIssueType.T_JUNCTION,
                    description=description,
                    severity=1
                )
                result.add_issue(issue)

            App.Console.PrintMessage(f"✅ Phase 5 complete: {len(result.issues)} T-junction issues found\n")

        except Exception as e:
            App.Console.PrintError(f"❌ Phase 5 error: {e}\n")

        return result

    # Helper methods for connectivity resolution
    def _build_constraint_connectivity(self) -> Dict:
        """Build map of constraint-based vertex connections."""
        constraint_map = defaultdict(set)
        internal_alignment_map = defaultdict(set)

        App.Console.PrintMessage("🔍 Building constraint connectivity map...\n")

        try:
            for i, constraint in enumerate(self.sketch.Constraints):
                if constraint.Type == "Coincident":
                    try:
                        v1_coord = round_coord(self.sketch.getPoint(constraint.First, constraint.FirstPos))
                        v2_coord = round_coord(self.sketch.getPoint(constraint.Second, constraint.SecondPos))

                        constraint_map[v1_coord].add(v2_coord)
                        constraint_map[v2_coord].add(v1_coord)

                    except Exception as e:
                        App.Console.PrintMessage(f"⚠️ Constraint {i} processing error: {e}\n")
                        continue

                elif constraint.Type == "InternalAlignment":
                    try:
                        # Enhanced B-spline internal alignment processing
                        App.Console.PrintMessage(f"🌀 Processing InternalAlignment constraint {i}:\n")
                        App.Console.PrintMessage(f"   First: {constraint.First}, FirstPos: {constraint.FirstPos}\n")
                        App.Console.PrintMessage(f"   Second: {constraint.Second}, SecondPos: {constraint.SecondPos}\n")

                        # Get geometry types for debugging
                        first_geo = self.sketch.Geometry[constraint.First]
                        second_geo = self.sketch.Geometry[constraint.Second]
                        App.Console.PrintMessage(f"   First geo: {first_geo.TypeId}\n")
                        App.Console.PrintMessage(f"   Second geo: {second_geo.TypeId}\n")

                        # Map B-spline endpoints to construction circle centers
                        v1_coord = round_coord(self.sketch.getPoint(constraint.First, constraint.FirstPos))

                        if constraint.SecondPos in [1, 2]:  # B-spline start/end
                            v2_coord = round_coord(self.sketch.getPoint(constraint.Second, constraint.SecondPos))
                            internal_alignment_map[v2_coord].add(v1_coord)
                            App.Console.PrintMessage(f"   B-spline endpoint {v2_coord} → circle center {v1_coord}\n")

                    except Exception as e:
                        App.Console.PrintMessage(f"⚠️ InternalAlignment {i} processing error: {e}\n")
                        continue

        except Exception as e:
            App.Console.PrintError(f"Error building constraint connectivity: {e}\n")

        App.Console.PrintMessage(f"Built constraint map: {len(constraint_map)} coincident, {len(internal_alignment_map)} internal alignment\n")

        return {
            'coincident': constraint_map,
            'internal_alignment': internal_alignment_map
        }

    def _resolve_bspline_connectivity(self) -> Dict:
        """Create map of B-spline endpoint effective connectivity."""
        resolution_map = {}
        constraint_data = self.constraint_graph

        App.Console.PrintMessage("🌀 Resolving B-spline connectivity...\n")

        for geo_idx, geometry in self.normal_geometry:
            if geometry.TypeId == 'Part::GeomBSplineCurve':
                start_coord, end_coord = get_geometry_endpoints(geometry)

                if start_coord and end_coord:
                    # Resolve effective connections through construction circles
                    start_connections = self._resolve_vertex_connectivity(
                        start_coord, constraint_data
                    )
                    end_connections = self._resolve_vertex_connectivity(
                        end_coord, constraint_data
                    )

                    resolution_map[geo_idx] = {
                        'start_coord': start_coord,
                        'end_coord': end_coord,
                        'start_connections': start_connections,
                        'end_connections': end_connections
                    }

                    geo_name = get_geometry_name(geo_idx, geometry)
                    App.Console.PrintMessage(f"🌀 {geo_name}: start→{len(start_connections)}, end→{len(end_connections)} connections\n")

        return resolution_map

    def _resolve_vertex_connectivity(self, coord: tuple, constraint_data: Dict) -> Set[tuple]:
        """Resolve all effective connections for a vertex coordinate."""
        connections = set()

        # Direct coincident connections
        direct = constraint_data['coincident'].get(coord, set())
        connections.update(direct)

        # Indirect connections through B-spline internal alignment
        if coord in constraint_data['internal_alignment']:
            for circle_center in constraint_data['internal_alignment'][coord]:
                circle_connections = constraint_data['coincident'].get(circle_center, set())
                connections.update(circle_connections)
                App.Console.PrintMessage(f"   B-spline {coord} via circle {circle_center} → {len(circle_connections)} vertices\n")

        return connections

    def _build_complete_connectivity_graph(self) -> Dict:
        """Build complete connectivity graph including B-spline resolution."""
        graph = defaultdict(list)
        edge_map = {}

        App.Console.PrintMessage("🔗 Building complete connectivity graph...\n")

        for geo_idx, geometry in self.normal_geometry:
            start_coord, end_coord = get_geometry_endpoints(geometry)

            if not start_coord or not end_coord or start_coord == end_coord:
                continue

            if geometry.TypeId == 'Part::GeomBSplineCurve' and geo_idx in self.bspline_resolution_map:
                # Use resolved B-spline connectivity
                bspline_data = self.bspline_resolution_map[geo_idx]

                # IMPORTANT: Add the B-spline geometry edge itself (start → end)
                self._add_graph_edge(graph, edge_map, start_coord, end_coord, geo_idx)

                # Connect to resolved endpoints
                for connected_coord in bspline_data['start_connections']:
                    self._add_graph_edge(graph, edge_map, start_coord, connected_coord, geo_idx)

                for connected_coord in bspline_data['end_connections']:
                    self._add_graph_edge(graph, edge_map, end_coord, connected_coord, geo_idx)
            else:
                # Regular geometric connectivity
                self._add_graph_edge(graph, edge_map, start_coord, end_coord, geo_idx)

        App.Console.PrintMessage(f"Connectivity graph: {len(graph)} vertices, {len(edge_map)} edges\n")

        return {'graph': graph, 'edge_map': edge_map}

    def _add_graph_edge(self, graph: Dict, edge_map: Dict, coord1: tuple, coord2: tuple, geo_idx: int):
        """Add bidirectional edge to graph."""
        if coord2 not in graph[coord1]:
            graph[coord1].append(coord2)
        if coord1 not in graph[coord2]:
            graph[coord2].append(coord1)

        edge_map[(coord1, coord2)] = geo_idx
        edge_map[(coord2, coord1)] = geo_idx

    def _validate_connectivity_resolution(self) -> List[TopologyIssue]:
        """Validate that connectivity resolution worked correctly."""
        issues = []

        # Check for unresolved B-splines
        for geo_idx, geometry in self.normal_geometry:
            if geometry.TypeId == 'Part::GeomBSplineCurve':
                if geo_idx not in self.bspline_resolution_map:
                    issue = TopologyIssue(
                        geo_idx=geo_idx,
                        geometry=geometry,
                        issue_type=TopologyIssueType.ISOLATION,
                        description="B-spline connectivity not resolved",
                        severity=3
                    )
                    issues.append(issue)

        return issues

    def _find_orphaned_geometry(self) -> List[int]:
        """Find completely disconnected geometry."""
        graph = self.connectivity_graph['graph']
        orphaned = []

        for geo_idx, geometry in self.normal_geometry:
            start_coord, end_coord = get_geometry_endpoints(geometry)

            if start_coord and end_coord:
                start_connections = len(graph.get(start_coord, []))
                end_connections = len(graph.get(end_coord, []))

                # If both endpoints have no connections beyond this edge, it's orphaned
                if start_connections == 0 and end_connections == 0:
                    orphaned.append(geo_idx)

        return orphaned

    def _find_floating_geometry(self) -> List[int]:
        """Find geometry that appears unconstrained."""
        # This is a placeholder - would need constraint analysis to determine
        # which geometry is truly unconstrained vs just not endpoint-connected
        return []

    def _find_connected_components(self) -> List[Set[tuple]]:
        """Find separate connected components in the graph."""
        graph = self.connectivity_graph['graph']
        visited = set()
        components = []

        def dfs(vertex, component):
            if vertex in visited:
                return
            visited.add(vertex)
            component.add(vertex)

            for neighbor in graph.get(vertex, []):
                dfs(neighbor, component)

        for vertex in graph:
            if vertex not in visited:
                component = set()
                dfs(vertex, component)
                if len(component) > 1:  # Ignore isolated vertices
                    components.append(component)

        return components

    def _find_bridge_edges(self, components: List[Set[tuple]]) -> List[int]:
        """Find bridge edges using simple loop membership analysis."""
        bridges = []

        # Find all loops in the graph
        all_loops = self._find_all_loops()
        App.Console.PrintMessage(f"Found {len(all_loops)} loops for bridge analysis\n")

        # Create set of all edges that are members of at least one loop
        edges_in_loops = set()
        for loop in all_loops:
            edges_in_loops.update(loop)

        App.Console.PrintMessage(f"Edges in loops: {len(edges_in_loops)} out of {len(self.normal_geometry)} normal geometry\n")

        # Check each normal geometry edge
        for geo_idx, geometry in self.normal_geometry:
            start_coord, end_coord = get_geometry_endpoints(geometry)

            if not start_coord or not end_coord or start_coord == end_coord:
                continue

            # Skip if this edge is part of any loop
            if geo_idx in edges_in_loops:
                continue

            # Check if this edge connects vertices (both ends connected to other geometry)
            graph = self.connectivity_graph['graph']
            start_connections = len([v for v in graph.get(start_coord, []) if v != end_coord])
            end_connections = len([v for v in graph.get(end_coord, []) if v != start_coord])

            # Bridge: not in any loop AND both ends are connected
            if start_connections > 0 and end_connections > 0:
                bridges.append(geo_idx)
                geo_name = get_geometry_name(geo_idx, geometry)
                App.Console.PrintMessage(f"🌉 Bridge detected: {geo_name} (not in any loop, both ends connected)\n")

        return bridges

    def _find_loop_connector_edges(self, components: List[Set[tuple]]) -> List[int]:
        """Find edges connecting separate loops within components."""
        loop_connectors = []

        # Find all loops in the graph
        all_loops = self._find_all_loops()
        App.Console.PrintMessage(f"Found {len(all_loops)} loops for loop connector analysis\n")

        # Create set of all edges that are members of at least one loop
        edges_in_loops = set()
        for loop in all_loops:
            edges_in_loops.update(loop)

        App.Console.PrintMessage(f"Edges in loops: {len(edges_in_loops)} out of {len(self.normal_geometry)} normal geometry\n")

        # Check each normal geometry edge that's NOT in any loop
        for geo_idx, geometry in self.normal_geometry:
            start_coord, end_coord = get_geometry_endpoints(geometry)

            if not start_coord or not end_coord or start_coord == end_coord:
                continue

            # Skip if this edge is part of any loop
            if geo_idx in edges_in_loops:
                continue

            # Check if this edge connects vertices (not dangling)
            graph = self.connectivity_graph['graph']
            start_connections = len([v for v in graph.get(start_coord, []) if v != end_coord])
            end_connections = len([v for v in graph.get(end_coord, []) if v != start_coord])

            # Only consider edges where both ends are connected to other geometry
            if start_connections > 0 and end_connections > 0:
                geo_name = get_geometry_name(geo_idx, geometry)
                App.Console.PrintMessage(f"🔗 Loop connector candidate: {geo_name} (not in any loop, both ends connected)\n")

                # Check if removing this edge would separate the connectivity
                # (indicating it connects different loop systems)
                if self._test_edge_creates_separation(geo_idx, start_coord, end_coord):
                    loop_connectors.append(geo_idx)
                    App.Console.PrintMessage(f"🔗 Confirmed loop connector: {geo_name} (separates loop systems)\n")

        return loop_connectors

    def _test_edge_creates_separation(self, geo_idx: int, start_coord: tuple, end_coord: tuple) -> bool:
        """Test if removing an edge would separate the graph into disconnected components."""
        # Create temporary graph without this edge
        temp_graph = defaultdict(list)

        for vertex, connections in self.connectivity_graph['graph'].items():
            temp_graph[vertex] = connections.copy()

        # Remove the edge connections
        if end_coord in temp_graph[start_coord]:
            temp_graph[start_coord].remove(end_coord)
        if start_coord in temp_graph[end_coord]:
            temp_graph[end_coord].remove(start_coord)

        # Test if start and end vertices are still connected via other paths
        visited = set()

        def dfs(current, target):
            if current == target:
                return True
            if current in visited:
                return False
            visited.add(current)

            for neighbor in temp_graph.get(current, []):
                if dfs(neighbor, target):
                    return True
            return False

        # If start and end are no longer connected, this edge was a bridge/connector
        return not dfs(start_coord, end_coord)

    def _find_subdivision_edges_clean(self) -> List[int]:
        """Find subdivision edges using clean edge removal analysis."""
        # Find all loops first
        loops = self._find_all_loops()
        original_loop_count = len(loops)

        if original_loop_count == 0:
            return []

        # Get candidate edges (edges connecting vertices with degree > 2)
        candidate_edges = self._get_subdivision_candidates()

        subdivision_edges = []

        for geo_idx in candidate_edges:
            # Test edge removal
            loop_reduction = self._test_edge_removal(geo_idx, original_loop_count)

            if loop_reduction > 0:
                subdivision_edges.append(geo_idx)

                geometry = next(g for i, g in self.normal_geometry if i == geo_idx)
                geo_name = get_geometry_name(geo_idx, geometry)
                App.Console.PrintMessage(f"🔧 Subdivision edge: {geo_name} reduces loops by {loop_reduction}\n")

        return subdivision_edges

    def _calculate_loop_perimeter(self, loop_edges: List[int]) -> float:
        """Calculate total perimeter length of a loop."""
        total_length = 0.0
        for geo_idx in loop_edges:
            if geo_idx < len(self.sketch.Geometry):
                geometry = self.sketch.Geometry[geo_idx]
                if hasattr(geometry, 'length'):
                    try:
                        total_length += geometry.length()
                    except Exception as e:
                        App.Console.PrintMessage(f"⚠️ Error getting length for geo {geo_idx}: {e}\n")
        return total_length

    def _find_loops_containing_edge(self, geo_idx: int, all_loops: List[List[int]]) -> List[List[int]]:
        """Find all loops that contain a specific geometry edge."""
        containing_loops = []
        for loop in all_loops:
            if geo_idx in loop:
                containing_loops.append(loop)
        return containing_loops

    def _get_subdivision_confidence(self, geo_idx: int, all_loops: List[List[int]]) -> Tuple[str, str]:
        """Get confidence level based on loop group membership and within-group size comparison."""
        # Find which loops contain this edge
        containing_loops = self._find_loops_containing_edge(geo_idx, all_loops)

        if not containing_loops:
            return "❔", "WEAK (not part of any loop)"

        # Debug: Show which loops this candidate belongs to
        App.Console.PrintMessage(f"🔍 Candidate geo {geo_idx} belongs to {len(containing_loops)} loops:\n")
        for i, loop in enumerate(containing_loops):
            perimeter = self._calculate_loop_perimeter(loop)
            App.Console.PrintMessage(f"   Loop {i}: edges {loop}, perimeter: {perimeter:.1f}\n")

        # Find interconnected loop groups
        loop_groups = self._find_loop_groups(all_loops)

        # Find which group this edge belongs to
        edge_group = None
        for group_idx, group_loops in enumerate(loop_groups):
            for loop in containing_loops:
                if loop in group_loops:
                    edge_group = group_idx
                    break
            if edge_group is not None:
                break

        if edge_group is None:
            return "❔", "WEAK (cannot determine group membership)"

        # Get all loops in this edge's group
        group_loops = loop_groups[edge_group]

        # Calculate perimeters for loops in this group only
        group_perimeters = []
        for loop in group_loops:
            perimeter = self._calculate_loop_perimeter(loop)
            group_perimeters.append((loop, perimeter))

        if not group_perimeters:
            return "❔", "WEAK (no group perimeter data)"

        # Find the largest loop in this group
        largest_group_loop, largest_group_perimeter = max(group_perimeters, key=lambda x: x[1])

        # Find the largest perimeter this edge participates in within its group
        edge_max_perimeter = 0
        edge_largest_loop = None
        for loop in containing_loops:
            if loop in group_loops:
                perimeter = self._calculate_loop_perimeter(loop)
                if perimeter > edge_max_perimeter:
                    edge_max_perimeter = perimeter
                    edge_largest_loop = loop

        # Debug logging
        App.Console.PrintMessage(f"🔍 Confidence analysis for geo {geo_idx}:\n")
        App.Console.PrintMessage(f"   Group {edge_group} has {len(group_loops)} loops\n")
        App.Console.PrintMessage(f"   Group loop perimeters:\n")
        for loop, perimeter in group_perimeters:
            App.Console.PrintMessage(f"     Loop {loop}: {perimeter:.1f} units\n")
        App.Console.PrintMessage(f"   Largest group loop: {largest_group_loop} ({largest_group_perimeter:.1f} units)\n")
        App.Console.PrintMessage(f"   Edge largest loop: {edge_largest_loop} ({edge_max_perimeter:.1f} units)\n")
        App.Console.PrintMessage(f"   Edge in largest group loop: {edge_largest_loop == largest_group_loop}\n")

        # Confidence based on whether edge is part of the largest loop in its group
        if edge_largest_loop == largest_group_loop:
            # Edge is part of the main boundary loop of its group
            return "❔", f"WEAK (part of main boundary, {edge_max_perimeter:.1f} units)"
        else:
            # Edge is part of smaller internal loops within its group
            return "❗", f"STRONG (internal subdivision, {edge_max_perimeter:.1f} units)"


    def _find_loop_groups(self, all_loops: List[List[int]]) -> List[List[List[int]]]:
        """Find interconnected groups of loops by checking vertex overlap."""
        if not all_loops:
            return []

        # Build vertex to loops mapping
        vertex_to_loops = defaultdict(list)
        for loop_idx, loop in enumerate(all_loops):
            # Get all vertices for this loop by examining edge endpoints
            loop_vertices = set()
            for geo_idx in loop:
                geometry = next(g for i, g in self.normal_geometry if i == geo_idx)
                start_coord, end_coord = get_geometry_endpoints(geometry)
                if start_coord:
                    loop_vertices.add(start_coord)
                if end_coord:
                    loop_vertices.add(end_coord)

            # Map each vertex to this loop
            for vertex in loop_vertices:
                vertex_to_loops[vertex].append(loop_idx)

        # Find connected components of loops
        visited_loops = set()
        loop_groups = []

        def dfs_loops(loop_idx, current_group):
            if loop_idx in visited_loops:
                return
            visited_loops.add(loop_idx)
            current_group.append(all_loops[loop_idx])

            # Find all loops that share vertices with this loop
            current_loop = all_loops[loop_idx]
            current_vertices = set()
            for geo_idx in current_loop:
                geometry = next(g for i, g in self.normal_geometry if i == geo_idx)
                start_coord, end_coord = get_geometry_endpoints(geometry)
                if start_coord:
                    current_vertices.add(start_coord)
                if end_coord:
                    current_vertices.add(end_coord)

            # Visit connected loops
            for vertex in current_vertices:
                for connected_loop_idx in vertex_to_loops[vertex]:
                    if connected_loop_idx not in visited_loops:
                        dfs_loops(connected_loop_idx, current_group)

        # Create groups
        for loop_idx in range(len(all_loops)):
            if loop_idx not in visited_loops:
                group = []
                dfs_loops(loop_idx, group)
                if group:
                    loop_groups.append(group)

        # Debug logging
        App.Console.PrintMessage(f"🔍 Found {len(loop_groups)} interconnected loop groups:\n")
        for group_idx, group in enumerate(loop_groups):
            App.Console.PrintMessage(f"   Group {group_idx}: {len(group)} loops\n")
            for loop_idx, loop in enumerate(group):
                perimeter = self._calculate_loop_perimeter(loop)
                App.Console.PrintMessage(f"     Loop {loop_idx}: edges {loop}, perimeter: {perimeter:.1f}\n")

        return loop_groups

    def _get_loops_for_group(self, group_loops: List[List[int]], containing_loops: List[List[int]]) -> List[List[int]]:
        """Get loops from containing_loops that belong to the specified group."""
        group_loops_set = set(tuple(sorted(loop)) for loop in group_loops)
        result = []
        for loop in containing_loops:
            loop_tuple = tuple(sorted(loop))
            if loop_tuple in group_loops_set:
                result.append(loop)
        return result

    def _get_subdivision_candidates(self) -> List[int]:
        """Get candidate edges for subdivision analysis."""
        graph = self.connectivity_graph['graph']

        # Build vertex degree map
        vertex_degrees = defaultdict(int)
        for vertex, connections in graph.items():
            vertex_degrees[vertex] = len(connections)

        candidate_edges = []

        for geo_idx, geometry in self.normal_geometry:
            start_coord, end_coord = get_geometry_endpoints(geometry)

            if start_coord and end_coord and start_coord != end_coord:
                start_degree = vertex_degrees.get(start_coord, 0)
                end_degree = vertex_degrees.get(end_coord, 0)

                # True subdivision edge: BOTH endpoints must have degree > 2
                if start_degree > 2 and end_degree > 2:
                    candidate_edges.append(geo_idx)

        return candidate_edges

    def _test_edge_removal(self, geo_idx: int, original_loop_count: int) -> int:
        """Test removing an edge and count loop reduction."""
        geometry = next(g for i, g in self.normal_geometry if i == geo_idx)
        start_coord, end_coord = get_geometry_endpoints(geometry)

        if not start_coord or not end_coord:
            return 0

        # Create temporary connectivity without this edge
        temp_graph = defaultdict(list)
        temp_edge_map = {}

        for vertex, connections in self.connectivity_graph['graph'].items():
            temp_graph[vertex] = connections.copy()

        # Remove the edge connections
        if end_coord in temp_graph[start_coord]:
            temp_graph[start_coord].remove(end_coord)
        if start_coord in temp_graph[end_coord]:
            temp_graph[end_coord].remove(start_coord)

        # Rebuild edge map without this edge
        for (v1, v2), edge_idx in self.connectivity_graph['edge_map'].items():
            if edge_idx != geo_idx:
                temp_edge_map[(v1, v2)] = edge_idx

        # Count loops in modified graph
        temp_connectivity = {'graph': temp_graph, 'edge_map': temp_edge_map}

        # Use a simplified loop counting method for the temporary graph
        new_loop_count = self._count_loops_in_graph(temp_connectivity)

        return original_loop_count - new_loop_count

    def _count_loops_in_graph(self, connectivity: Dict) -> int:
        """Simple loop counting for temporary graphs."""
        graph = connectivity['graph']
        edge_map = connectivity['edge_map']

        # Quick approximation: count cycles using DFS
        visited_edges = set()
        loop_count = 0

        def dfs_cycle_count(start_vertex, current_vertex, path, path_edges):
            nonlocal loop_count

            for next_vertex in graph.get(current_vertex, []):
                edge_key = (current_vertex, next_vertex)
                reverse_edge_key = (next_vertex, current_vertex)

                if len(path) > 1 and next_vertex == path[-2]:
                    continue

                edge_idx = edge_map.get(edge_key, edge_map.get(reverse_edge_key))
                if edge_idx is None or edge_idx in path_edges:
                    continue

                if next_vertex == start_vertex and len(path_edges) >= 2:
                    edge_tuple = tuple(sorted(path_edges | {edge_idx}))
                    if edge_tuple not in visited_edges:
                        visited_edges.add(edge_tuple)
                        loop_count += 1
                    continue

                if next_vertex not in path and len(path) < 8:  # Shorter path for temp analysis
                    dfs_cycle_count(start_vertex, next_vertex, path + [next_vertex], path_edges | {edge_idx})

        for vertex in graph:
            dfs_cycle_count(vertex, vertex, [vertex], set())

        return loop_count

    def _find_true_tjunctions_with_details(self) -> List[Tuple[int, str]]:
        """Find genuine T-junction dangling edges with specific descriptions."""
        graph = self.connectivity_graph['graph']
        constraint_data = self.constraint_graph
        tjunctions = []

        for geo_idx, geometry in self.normal_geometry:
            start_coord, end_coord = get_geometry_endpoints(geometry)

            if start_coord and end_coord:
                # Count geometric connections (excluding this edge's self-connection)
                start_geo_connections = len([v for v in graph.get(start_coord, []) if v != end_coord])
                end_geo_connections = len([v for v in graph.get(end_coord, []) if v != start_coord])

                # Count coincident constraint connections (vertex-to-vertex connections)
                start_coincident_connections = len(constraint_data['coincident'].get(start_coord, set()))
                end_coincident_connections = len(constraint_data['coincident'].get(end_coord, set()))

                # Check for other constraint types (point-on-edge, etc.) by checking all constraints
                start_other_constraints = self._count_other_constraints(geo_idx, 1)  # Start = position 1
                end_other_constraints = self._count_other_constraints(geo_idx, 2)    # End = position 2

                # Total vertex connectivity (geometric + coincident)
                start_vertex_total = start_geo_connections + start_coincident_connections
                end_vertex_total = end_geo_connections + end_coincident_connections

                # T-junction: one end has no vertex connections OR both ends have no vertex connections
                is_tjunction = start_vertex_total == 0 or end_vertex_total == 0

                if is_tjunction:
                    geo_name = get_geometry_name(geo_idx, geometry)

                    # Debug logging for description logic
                    App.Console.PrintMessage(f"🔍 T-junction analysis for {geo_name}:\n")
                    App.Console.PrintMessage(f"   Start: geo={start_geo_connections}, coincident={start_coincident_connections}, other_constraints={start_other_constraints}, vertex_total={start_vertex_total}\n")
                    App.Console.PrintMessage(f"   End: geo={end_geo_connections}, coincident={end_coincident_connections}, other_constraints={end_other_constraints}, vertex_total={end_vertex_total}\n")

                    # Determine specific description based on connection pattern
                    if start_vertex_total == 0 and end_vertex_total == 0:
                        # Both ends disconnected from vertices
                        if start_other_constraints > 0 or end_other_constraints > 0:
                            description = "Anchored but not connected"
                            App.Console.PrintMessage(f"   → Description: {description} (has non-vertex constraints but no vertex connections)\n")
                        else:
                            description = "No connections"
                            App.Console.PrintMessage(f"   → Description: {description} (completely isolated)\n")
                    elif start_vertex_total == 0:
                        # Start end disconnected
                        if start_other_constraints > 0:
                            description = "Anchored but not connected"
                            App.Console.PrintMessage(f"   → Description: {description} (start has non-vertex constraints only)\n")
                        else:
                            description = "Dangling edge"
                            App.Console.PrintMessage(f"   → Description: {description} (start end unconnected, end connected)\n")
                    elif end_vertex_total == 0:
                        # End disconnected
                        if end_other_constraints > 0:
                            description = "Anchored but not connected"
                            App.Console.PrintMessage(f"   → Description: {description} (end has non-vertex constraints only)\n")
                        else:
                            description = "Dangling edge"
                            App.Console.PrintMessage(f"   → Description: {description} (end unconnected, start connected)\n")
                    else:
                        # Fallback - shouldn't reach here given our filtering
                        description = "Incomplete connection"
                        App.Console.PrintMessage(f"   → Description: {description} (unexpected case)\n")

                    tjunctions.append((geo_idx, description))
                    App.Console.PrintMessage(f"⚠️  T-junction: {geo_name} (start_vertex={start_vertex_total}, end_vertex={end_vertex_total}) - {description}\n")

        return tjunctions

    def _count_other_constraints(self, geo_idx: int, position: int) -> int:
        """Count non-coincident constraints for a geometry endpoint."""
        count = 0
        try:
            for constraint in self.sketch.Constraints:
                # Check if this constraint involves our geometry at the specified position
                if ((constraint.First == geo_idx and constraint.FirstPos == position) or
                    (hasattr(constraint, 'Second') and constraint.Second == geo_idx and constraint.SecondPos == position)):

                    # Count non-coincident constraints (like PointOnObject, Distance, etc.)
                    if constraint.Type != "Coincident":
                        count += 1

        except Exception as e:
            App.Console.PrintMessage(f"⚠️ Error counting constraints for geo {geo_idx}, pos {position}: {e}\n")

        return count

    def _collect_all_issues(self) -> Dict[TopologyIssueType, List[TopologyIssue]]:
        """Collect all issues from all phases by type using enum keys."""
        issues_by_type = defaultdict(list)

        for phase_result in self.phase_results.values():
            for issue in phase_result.issues:
                # Use enum as key for Tab4 UI compatibility
                issues_by_type[issue.issue_type].append(issue)

        return dict(issues_by_type)

def find_problematic_intersections(analyzer: Any) -> List[Dict[str, Any]]:
    """Main entry point for problematic intersections analysis - maintains compatibility."""
    App.Console.PrintMessage("\n" + "=" * 80 + "\n")
    App.Console.PrintMessage("🔍 TAB4: WIRE TOPOLOGY ANALYSIS (Phased Approach)\n")
    App.Console.PrintMessage("=" * 80 + "\n")

    # Create the phased analyzer
    phased_analyzer = WireTopologyAnalyzer(analyzer.sketch)
    issues_by_type = phased_analyzer.analyze()

    # Build flat list directly for Main compatibility
    problematic = []

    for issue_type, issues in issues_by_type.items():
        for issue in issues:
            # Map to descriptive problem types
            if issue.issue_type == TopologyIssueType.ISOLATION:
                problem_type = f"Isolation: {issue.description}"
            elif issue.issue_type == TopologyIssueType.GEOMETRIC_VALIDITY:
                problem_type = f"Geometric Validity: {issue.description}"
            elif issue.issue_type == TopologyIssueType.BRIDGE:
                problem_type = f"Bridge: {issue.description}"
            elif issue.issue_type == TopologyIssueType.SUBDIVISION:
                problem_type = f"Subdivision: {issue.description}"
            elif issue.issue_type == TopologyIssueType.T_JUNCTION:
                problem_type = f"T-junction: {issue.description}"
            else:
                problem_type = f"{issue.issue_type.value}: {issue.description}"

            problematic.append({
                'geo_idx': issue.geo_idx,
                'geometry': issue.geometry,
                'problem_type': problem_type,
                'is_bridge': issue.issue_type == TopologyIssueType.BRIDGE,
                'issue_type': issue.issue_type,
                'severity': issue.severity
            })

    # Store the analyzer and results for Tab4 UI access
    analyzer._phased_analyzer = phased_analyzer
    analyzer._issues_by_type = issues_by_type

    # Debug logging
    App.Console.PrintMessage(f"🔍 Debug: Stored {len(issues_by_type)} issue types on analyzer\n")
    for issue_type, issues in issues_by_type.items():
        App.Console.PrintMessage(f"   {issue_type.value}: {len(issues)} issues\n")

    App.Console.PrintMessage(f"✅ Tab4: Analysis complete - found {len(problematic)} total issues\n")
    App.Console.PrintMessage("=" * 80 + "\n")
    return problematic


# ============================================================================
# UI FUNCTIONS - Updated for Phased Approach
# ============================================================================

def setup_intersections_tab(widget):
    """Setup the wire topology analysis tab with grouped issue categories."""
    from PySide import QtGui, QtCore

    tab = QtGui.QWidget()
    layout = QtGui.QVBoxLayout(tab)

    # Create sections for each issue type with logical repair order
    sections = [
        ("GEOMETRIC VALIDITY ISSUES", "geometric_validity", TopologyIssueType.GEOMETRIC_VALIDITY),
        ("INCOMPLETE CONNECTIONS", "tjunction", TopologyIssueType.T_JUNCTION),
        ("BRIDGE/CONNECTION ISSUES", "bridge", TopologyIssueType.BRIDGE),
        ("SUBDIVISION ISSUES", "subdivision", TopologyIssueType.SUBDIVISION)
    ]

    widget.issue_sections = {}

    for section_title, section_key, issue_type in sections:
        # Group box for each section with initial count of 0
        group_box = QtGui.QGroupBox(f"{section_title} (0)")
        group_layout = QtGui.QVBoxLayout(group_box)

        # Issue list
        issue_list = QtGui.QListWidget()
        issue_list.setMaximumHeight(100)
        issue_list.itemEntered.connect(widget._on_hover)
        issue_list.itemClicked.connect(widget._on_intersection_selected)
        widget.issue_sections[section_key] = {
            'list': issue_list,
            'type': issue_type,
            'group_box': group_box,  # Store reference for updating title
            'base_title': section_title  # Store base title for updates
        }
        group_layout.addWidget(issue_list)

        # Action buttons
        button_layout = QtGui.QHBoxLayout()

        make_all_btn = QtGui.QPushButton("Make All Construction")
        make_all_btn.clicked.connect(
            lambda checked=False, key=section_key: make_section_construction(widget, key, True, False)
        )
        button_layout.addWidget(make_all_btn)

        # Add "Make Strong Construction" button only for subdivision section
        if section_key == "subdivision":
            make_strong_btn = QtGui.QPushButton("Make Strong Construction")
            make_strong_btn.clicked.connect(
                lambda checked=False, key=section_key: make_section_construction(widget, key, False, True)
            )
            button_layout.addWidget(make_strong_btn)

        make_selected_btn = QtGui.QPushButton("Make Selected Construction")
        make_selected_btn.clicked.connect(
            lambda checked=False, key=section_key: make_section_construction(widget, key, False, False)
        )
        button_layout.addWidget(make_selected_btn)

        group_layout.addLayout(button_layout)
        layout.addWidget(group_box)

    widget.tab_widget.addTab(tab, "Wire Topology Analysis")

def populate_intersections_list(widget):
    """Populate the wire topology analysis lists with phased results."""
    # Clear all section lists
    for section_data in widget.issue_sections.values():
        section_data['list'].clear()

    App.Console.PrintMessage("🖥️  Tab4: Populating phased topology analysis UI...\n")

    # Debug: Check what data is available
    has_phased_data = hasattr(widget.analyzer, '_issues_by_type')
    App.Console.PrintMessage(f"🔍 Debug: Has phased data: {has_phased_data}\n")

    if has_phased_data:
        issues_by_type = widget.analyzer._issues_by_type
        App.Console.PrintMessage(f"🔍 Debug: Phased data keys: {list(issues_by_type.keys())}\n")
        App.Console.PrintMessage(f"🔍 Debug: Key types: {[type(k) for k in issues_by_type.keys()]}\n")

        # Debug: Show actual data content
        for key, issues in issues_by_type.items():
            App.Console.PrintMessage(f"🔍 Debug: Key {key} has {len(issues)} issues\n")

    # Check if we have phased analysis results
    if has_phased_data and widget.analyzer._issues_by_type:
        issues_by_type = widget.analyzer._issues_by_type

        App.Console.PrintMessage("🔍 Debug: Using phased analysis results\n")

        # Populate each section and update counts
        for section_key, section_data in widget.issue_sections.items():
            issue_type = section_data['type']
            issue_list = section_data['list']
            group_box = section_data['group_box']
            base_title = section_data['base_title']

            # Find issues by comparing enum values instead of object identity
            issues = []
            for stored_key, stored_issues in issues_by_type.items():
                if hasattr(stored_key, 'value') and hasattr(issue_type, 'value'):
                    if stored_key.value == issue_type.value:
                        issues = stored_issues
                        App.Console.PrintMessage(f"🔍 Debug: Found matching enum values {stored_key.value} = {issue_type.value}\n")
                        break

            App.Console.PrintMessage(f"🔍 Debug: Section {section_key} looking for enum {issue_type}: {len(issues)} issues\n")

            # Update section title with count
            group_box.setTitle(f"{base_title} ({len(issues)})")

            for issue in issues:
                geo_name = get_geometry_name(issue.geo_idx, issue.geometry)
                item_text = f"{geo_name}: {issue.description}"

                list_item = QtGui.QListWidgetItem(item_text)
                list_item.setData(QtCore.Qt.UserRole, {
                    'geo_idx': issue.geo_idx,
                    'geometry': issue.geometry,
                    'problem_type': issue.description,
                    'is_bridge': issue.issue_type == TopologyIssueType.BRIDGE,
                    'issue_type': issue.issue_type,
                    'severity': issue.severity
                })
                issue_list.addItem(list_item)
                App.Console.PrintMessage(f"   Added: {item_text}\n")

        # Count total issues
        total_issues = sum(len(issues) for issues in issues_by_type.values())
        App.Console.PrintMessage(f"✅ Tab4: UI populated with {total_issues} issues across {len(widget.issue_sections)} categories\n")

    else:
        # Fallback to old format if phased analysis not available
        App.Console.PrintMessage("🔍 Debug: Using fallback to old format\n")
        App.Console.PrintMessage(f"🔍 Debug: Available problematic items: {len(widget.analysis_data.problematic)}\n")

        # Initialize section counts
        section_counts = {key: 0 for key in widget.issue_sections.keys()}

        for item in widget.analysis_data.problematic:
            geo_name = get_geometry_name(item['geo_idx'], item['geometry'])
            problem_type = item['problem_type']

            App.Console.PrintMessage(f"🔍 Debug: Processing item: {geo_name} - {problem_type}\n")

            # Enhanced categorization based on problem type text
            if any(keyword in problem_type.lower() for keyword in ["geometric validity", "overlapping"]):
                section_key = "geometric_validity"
            elif any(keyword in problem_type.lower() for keyword in ["bridge", "connection", "cross-wire"]):
                section_key = "bridge"
            elif any(keyword in problem_type.lower() for keyword in ["subdivision", "subdivide"]):
                section_key = "subdivision"
            elif any(keyword in problem_type.lower() for keyword in ["t-junction", "dangling"]):
                section_key = "tjunction"
            else:
                # Default categorization - put unknown types in subdivision for now
                section_key = "subdivision"
                App.Console.PrintMessage(f"🔍 Debug: Unknown problem type, defaulting to subdivision: {problem_type}\n")

            App.Console.PrintMessage(f"🔍 Debug: Categorized as: {section_key}\n")

            if section_key in widget.issue_sections:
                issue_list = widget.issue_sections[section_key]['list']
                list_item = QtGui.QListWidgetItem(f"{geo_name}: {problem_type}")
                list_item.setData(QtCore.Qt.UserRole, item)
                issue_list.addItem(list_item)
                section_counts[section_key] += 1
                App.Console.PrintMessage(f"   Added to {section_key}: {geo_name}\n")
            else:
                App.Console.PrintMessage(f"⚠️  Debug: Section {section_key} not found in widget.issue_sections\n")

        # Update section titles with counts for fallback mode
        for section_key, section_data in widget.issue_sections.items():
            group_box = section_data['group_box']
            base_title = section_data['base_title']
            count = section_counts[section_key]
            group_box.setTitle(f"{base_title} ({count})")

def make_section_construction(widget, section_key: str, all_items: bool, strong_only: bool = False):
    """Convert section items to construction geometry."""
    action_desc = "All" if all_items else ("Strong" if strong_only else "Selected")
    App.Console.PrintMessage(f"\n🔧 Button pressed: Make {action_desc} Construction for {section_key}\n")

    if section_key not in widget.issue_sections:
        App.Console.PrintMessage(f"❌ Error: Section key '{section_key}' not found in issue_sections\n")
        return

    issue_list = widget.issue_sections[section_key]['list']
    App.Console.PrintMessage(f"📊 Issue list has {issue_list.count()} total items\n")

    # Collect geometry indices
    indices = []

    if all_items:
        App.Console.PrintMessage("🔍 Collecting all items in section...\n")
        for i in range(issue_list.count()):
            item = issue_list.item(i)
            data = item.data(QtCore.Qt.UserRole)
            App.Console.PrintMessage(f"  Item {i}: {item.text()}\n")
            if data and 'geo_idx' in data:
                geo_idx = data['geo_idx']
                indices.append(geo_idx)
                App.Console.PrintMessage(f"    → Found geo_idx: {geo_idx}\n")
            else:
                App.Console.PrintMessage(f"    → No geo_idx found in data: {data}\n")
    elif strong_only:
        App.Console.PrintMessage("🔍 Collecting strong confidence items only...\n")
        for i in range(issue_list.count()):
            item = issue_list.item(i)
            data = item.data(QtCore.Qt.UserRole)
            item_text = item.text()
            App.Console.PrintMessage(f"  Item {i}: {item_text}\n")

            # Check if this is a strong candidate (contains ❗)
            if "❗" in item_text and data and 'geo_idx' in data:
                geo_idx = data['geo_idx']
                indices.append(geo_idx)
                App.Console.PrintMessage(f"    → Strong candidate geo_idx: {geo_idx}\n")
            else:
                App.Console.PrintMessage(f"    → Skipped (not strong or no data)\n")
    else:
        App.Console.PrintMessage("🔍 Collecting selected item...\n")
        current_item = issue_list.currentItem()
        if current_item:
            App.Console.PrintMessage(f"  Selected item: {current_item.text()}\n")
            data = current_item.data(QtCore.Qt.UserRole)
            if data and 'geo_idx' in data:
                geo_idx = data['geo_idx']
                indices.append(geo_idx)
                App.Console.PrintMessage(f"    → Found geo_idx: {geo_idx}\n")
            else:
                App.Console.PrintMessage(f"    → No geo_idx found in data: {data}\n")
        else:
            App.Console.PrintMessage("  → No item selected\n")

    App.Console.PrintMessage(f"📊 Collected {len(indices)} geometry indices: {indices}\n")

    if not indices:
        App.Console.PrintMessage("⚠️  No geometry indices to process - exiting\n")
        return

    # Get the user-facing display title for transaction naming
    section_data = widget.issue_sections[section_key]
    display_title = section_data['base_title']

    # Start transaction for the entire operation
    transaction_name = f"Convert {action_desc} {display_title} to Construction"
    App.Console.PrintMessage(f"🔄 Starting transaction: {transaction_name}\n")

    widget.sketch.Document.openTransaction(transaction_name)

    try:
        # Apply construction state changes
        App.Console.PrintMessage("🔧 Applying construction state changes...\n")
        for geo_idx in indices:
            App.Console.PrintMessage(f"  Setting geometry {geo_idx} to construction\n")
            widget.sketch.setConstruction(geo_idx, True)

        # Force sketch recompute
        App.Console.PrintMessage("🔄 Recomputing sketch...\n")
        widget.sketch.recompute()
        widget.sketch.Document.recompute()

        try:
            import FreeCADGui as Gui
            Gui.updateGui()
            App.Console.PrintMessage("✅ GUI updated\n")
        except Exception as e:
            App.Console.PrintMessage(f"⚠️  GUI update warning: {e}\n")

        # Commit transaction
        App.Console.PrintMessage("💾 Committing transaction...\n")
        widget.sketch.Document.commitTransaction()

        # Re-analyze sketch after successful commit
        App.Console.PrintMessage("🔍 Re-analyzing sketch...\n")
        widget.analyze_sketch()
        App.Console.PrintMessage("✅ Operation completed successfully\n")

    except Exception as e:
        # Rollback transaction on error
        App.Console.PrintError(f"❌ Error during operation: {e}\n")
        try:
            widget.sketch.Document.abortTransaction()
            App.Console.PrintMessage("🔄 Transaction rolled back\n")
        except Exception as rollback_error:
            App.Console.PrintError(f"❌ Error during rollback: {rollback_error}\n")
