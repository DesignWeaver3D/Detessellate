# -*- coding: utf-8 -*-
# FreeCAD Macro
# Name: CoplanarSketch
# Author: DesignWeaver3D
# Version: 3.1.0
# Date: 2026-01-18
# FreeCAD Version: 1.0.2
# Description: Creates sketches from coplanar edges of tessellated solid objects that result from mesh import & conversion.
# License: GPL‑3.0‑or‑later

import FreeCAD
import FreeCADGui
import Part
import Sketcher
import PartDesign
from PySide.QtWidgets import QDockWidget, QWidget, QVBoxLayout, QPushButton, QTextEdit, QLabel, QInputDialog, QLineEdit
from PySide.QtCore import Qt
import time
import random

class EdgeDataCollector(QDockWidget):
    def __init__(self):
        super().__init__("CoplanarSketch")
        self.setWidget(self.create_ui())
        self.collected_edges = []
        self.edge_dict_by_name = {}  # OPTIMIZATION: Pre-computed lookup dictionary
        self.edge_mass_center = FreeCAD.Vector(0, 0, 0)

    def create_ui(self):
        widget = QWidget()
        layout = QVBoxLayout()

        self.collect_button = QPushButton("Collect Edge Data")
        self.collect_button.clicked.connect(self.collect_data)

        self.select_coplanar_label = QLabel("Select a face or two coplanar edges before using this button.")
        self.select_coplanar_label.setVisible(False)
        self.select_coplanar_button = QPushButton("Select Coplanar Edges")
        self.select_coplanar_button.clicked.connect(self.select_coplanar_edges)
        self.select_coplanar_button.setVisible(False)

        self.tolerance_label = QLabel("Coplanar tolerance:")
        self.tolerance_input = QLineEdit("0.000001")  # default 1e-6

        self.clean_label = QLabel("Degenerate edges detected. Cleaning recommended for better performance.")
        self.clean_label.setVisible(False)
        self.clean_button = QPushButton("Clean Degenerate Edges")
        self.clean_button.clicked.connect(self.clean_degenerate_edges)
        self.clean_button.setVisible(False)

        self.create_sketch_button = QPushButton("Create Sketch from Selection")
        self.create_sketch_button.clicked.connect(self.create_sketch_from_selection)
        self.create_sketch_button.setVisible(False)

        self.info_display = QTextEdit()
        self.info_display.setReadOnly(True)

        self.clear_button = QPushButton("Clear Messages")
        self.clear_button.clicked.connect(self.clear_messages)

        layout.addWidget(self.collect_button)
        layout.addWidget(self.select_coplanar_label)

        layout.addWidget(self.tolerance_label)
        layout.addWidget(self.tolerance_input)

        layout.addWidget(self.select_coplanar_button)
        layout.addWidget(self.clean_label)
        layout.addWidget(self.clean_button)
        layout.addWidget(self.create_sketch_button)
        layout.addWidget(self.info_display)
        layout.addWidget(self.clear_button)
        widget.setLayout(layout)

        return widget

    def collect_data(self):
        start_time = time.time()

        selection = FreeCADGui.Selection.getSelectionEx()
        if not selection:
            self.info_display.append("Error: No selection made.")
            return

        obj = selection[0].Object
        edges = obj.Shape.Edges

        # OPTIMIZATION: Use enumerate instead of .index() - O(n) instead of O(n²)
        self.collected_edges = []
        invalid_count = 0
        degenerate_count = 0
        property_error_count = 0

        for edge_index, edge in enumerate(edges):  # OPTIMIZED: Direct enumeration
            # Cache vertex points to avoid repeated API calls
            try:
                vertex_points = [v.Point for v in edge.Vertexes]
                vertex_count = len(vertex_points)
            except:
                vertex_points = []
                vertex_count = 0

            edge_dict = {
                'edge': edge,
                'name': f"Edge{edge_index+1}",
                'index': edge_index,
                'valid': True,
                'vertex_points': vertex_points,
                'vertex_count': vertex_count,
                'error_reason': None
            }

            # Check for validity issues
            if vertex_count != 2:
                edge_dict['valid'] = False
                edge_dict['error_reason'] = f"Degenerate edge ({vertex_count} vertices)"
                degenerate_count += 1
                invalid_count += 1
            else:
                # Check for property access errors
                try:
                    _ = edge.Length
                except:
                    edge_dict['valid'] = False
                    edge_dict['error_reason'] = "Property access error"
                    property_error_count += 1
                    invalid_count += 1

            self.collected_edges.append(edge_dict)

        # OPTIMIZATION: Build lookup dictionary once
        self.edge_dict_by_name = {e['name']: e for e in self.collected_edges}

        # Calculate mass center from valid edges only
        all_points = [point for edge_dict in self.collected_edges
                      if edge_dict['valid'] for point in edge_dict['vertex_points']]
        if all_points:
            self.edge_mass_center = sum(all_points, FreeCAD.Vector()).multiply(1.0 / len(all_points))

        duration = time.time() - start_time
        expected_count = len(edges)
        actual_count = len(self.collected_edges)

        self.info_display.append(f"Collected {actual_count} edges from {obj.Label}.")

        if expected_count != actual_count:
            skipped_count = expected_count - actual_count
            self.info_display.append(f"Skipped {skipped_count} edges due to processing errors.")

        if invalid_count > 0:
            error_details = []
            if degenerate_count > 0:
                error_details.append(f"{degenerate_count} degenerate edges")
            if property_error_count > 0:
                error_details.append(f"{property_error_count} property errors")
            self.info_display.append(f"Invalid edges found: {', '.join(error_details)}.")

            # Show cleaning option only when degenerates are found
            self.clean_label.setVisible(True)
            self.clean_button.setVisible(True)
            self.select_coplanar_label.setVisible(False)
            self.select_coplanar_button.setVisible(False)
            self.create_sketch_button.setVisible(False)
        else:
            # Show coplanar operations when geometry is clean
            self.clean_label.setVisible(False)
            self.clean_button.setVisible(False)
            self.select_coplanar_label.setVisible(True)
            self.select_coplanar_button.setVisible(True)
            self.create_sketch_button.setVisible(False)  # Hidden until coplanar selection

        self.info_display.append(f"Elapsed time: {duration:.4f} seconds.\n")

    def select_coplanar_edges(self):
        # Check if edge data has been collected first
        if not self.collected_edges:
            self.info_display.append("Error: No edge data collected. Click 'Collect Edge Data' first.")
            return

        start_time = time.time()
        selection = FreeCADGui.Selection.getSelectionEx()
        if not selection:
            self.info_display.append("Error: Select a face or two edges first.")
            return

        obj = selection[0].Object

        # OPTIMIZATION: Use SubObjects directly instead of string parsing
        subObjects = selection[0].SubObjects
        selected_edges = [sub for sub in subObjects if sub.ShapeType == "Edge"]
        selected_faces = [sub for sub in subObjects if sub.ShapeType == "Face"]

        if selected_faces:
            # Use face directly, no string parsing needed
            plane_normal = selected_faces[0].Surface.Axis
            plane_point = selected_faces[0].CenterOfMass
            self.info_display.append(f"Using plane defined by selected face")
        elif len(selected_edges) >= 2:
            # OPTIMIZATION: Use cached vertex points instead of re-accessing edge.Vertexes
            # We need edge names to look up in our cache
            selected_edge_names = [name for name in selection[0].SubElementNames if name.startswith("Edge")]

            if len(selected_edge_names) < 2:
                self.info_display.append("Error: Select at least two edges to define a plane.")
                return

            edge1_name = selected_edge_names[0]
            edge2_name = selected_edge_names[1]

            # OPTIMIZATION: Use pre-computed dictionary lookup instead of linear search
            edge1_dict = self.edge_dict_by_name.get(edge1_name)
            edge2_dict = self.edge_dict_by_name.get(edge2_name)

            if not edge1_dict or not edge2_dict:
                self.info_display.append("Error: Selected edges not found in collected data.")
                return

            if not edge1_dict['valid'] or not edge2_dict['valid']:
                self.info_display.append("Error: One or both selected edges are invalid.")
                return

            # OPTIMIZATION: Use cached vertex_points instead of accessing edge.Vertexes
            edge1_points = edge1_dict['vertex_points']
            edge2_points = edge2_dict['vertex_points']

            # Gather unique points from both edges
            unique_points = list(edge1_points)
            tolerance = float(self.tolerance_input.text())

            for p2 in edge2_points:
                is_duplicate = False
                for p1 in unique_points:
                    if (p2 - p1).Length < tolerance:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    unique_points.append(p2)

            if len(unique_points) < 3:
                self.info_display.append("Error: Need at least 3 unique points to define a plane. Selected edges may be collinear.")
                return

            # Calculate plane from the unique points
            plane_point = unique_points[0]
            plane_normal = None

            # Try all combinations of 3 points to find best plane
            for i in range(len(unique_points)):
                for j in range(i + 1, len(unique_points)):
                    for k in range(j + 1, len(unique_points)):
                        v1 = unique_points[j] - unique_points[i]
                        v2 = unique_points[k] - unique_points[i]
                        n = v1.cross(v2)
                        if n.Length > 1e-6:
                            plane_normal = n.normalize()
                            break
                    if plane_normal:
                        break
                if plane_normal:
                    break

            if not plane_normal:
                self.info_display.append("Error: Could not determine a valid plane from selected edges.")
                return

            self.info_display.append(f"Using plane defined by edges: {edge1_name}, {edge2_name}")
        else:
            self.info_display.append("Error: Select a face or at least two edges to define a plane.")
            return

        # Find all coplanar edges using the cached vertex_points
        try:
            # Clamp user input between 1e-6 and 1.0
            tolerance = max(1e-6, min(float(self.tolerance_input.text()), 1.0))
        except:
            tolerance = 1e-6  # fallback if input is invalid

        coplanar_edge_indices = set()

        # OPTIMIZATION: Use cached vertex_points, but check BOTH vertices are in plane
        # An edge is coplanar only if BOTH endpoints lie in the plane
        for edge_dict in self.collected_edges:
            if edge_dict['valid'] and len(edge_dict['vertex_points']) == 2:
                # Use cached vertex_points
                v1, v2 = edge_dict['vertex_points'][0], edge_dict['vertex_points'][1]
                dist1 = abs(plane_normal.dot(v1 - plane_point))
                dist2 = abs(plane_normal.dot(v2 - plane_point))

                # Both vertices must be within tolerance of the plane
                if dist1 <= tolerance and dist2 <= tolerance:
                    coplanar_edge_indices.add(edge_dict['index'])

        if not coplanar_edge_indices:
            self.info_display.append("No coplanar edges found within the tolerance.")
            return

        # Validate coplanar results - warn if we selected too many edges
        total_valid_edges = len([ed for ed in self.collected_edges if ed['valid']])
        if len(coplanar_edge_indices) > total_valid_edges * 0.5:
            self.info_display.append(f"Warning: {len(coplanar_edge_indices)} coplanar edges found ({len(coplanar_edge_indices)/total_valid_edges*100:.1f}% of valid edges) - check plane definition.")

        # OPTIMIZATION: Batch selection instead of individual addSelection calls
        FreeCADGui.Selection.clearSelection()
        edge_names = [f"Edge{idx+1}" for idx in sorted(coplanar_edge_indices)]
        FreeCADGui.Selection.addSelection(obj, edge_names)  # Single API call

        self.info_display.append(f"Selected {len(coplanar_edge_indices)} coplanar edges.")

        # Show the create sketch button
        self.create_sketch_button.setVisible(True)

        duration = time.time() - start_time
        self.info_display.append(f"Elapsed time: {duration:.4f} seconds.\n")

    def calculate_robust_plane_normal_and_placement(self, vertices, source_object):
        """OPTIMIZED: Sample-based approach instead of O(n³) exhaustive search"""
        if len(vertices) < 3:
            return FreeCAD.Vector(0, 0, 1), vertices[0] if vertices else FreeCAD.Vector()

        center = sum(vertices, FreeCAD.Vector()).multiply(1.0 / len(vertices))
        vectors = [v.sub(center) for v in vertices]

        best_normal = None
        best_magnitude = 0

        if len(vertices) <= 20:
            # Small set: check all combinations (still fast)
            for i in range(len(vectors)):
                for j in range(i+1, len(vectors)):
                    for k in range(j+1, len(vectors)):
                        n = (vectors[j] - vectors[i]).cross(vectors[k] - vectors[i])
                        if n.Length > best_magnitude:
                            best_magnitude = n.Length
                            best_normal = n.normalize()
        else:
            # OPTIMIZATION: Large set - use random sampling instead of O(n³)
            max_samples = 100
            samples = 0
            attempts = 0
            max_attempts = max_samples * 10

            while samples < max_samples and attempts < max_attempts:
                attempts += 1
                i, j, k = random.sample(range(len(vectors)), 3)
                n = (vectors[j] - vectors[i]).cross(vectors[k] - vectors[i])
                if n.Length > best_magnitude:
                    best_magnitude = n.Length
                    best_normal = n.normalize()
                    samples += 1

        if not best_normal:
            best_normal = FreeCAD.Vector(0, 0, 1)

        if self.edge_mass_center:
            delta = center.sub(self.edge_mass_center)
            if delta.Length > 1e-6 and delta.normalize().dot(best_normal) < 0:
                best_normal = best_normal.multiply(-1)

        return best_normal, center

    def create_robust_placement(self, normal, center):
        normal = normal.normalize() if normal.Length > 1e-6 else FreeCAD.Vector(0, 0, 1)
        z_axis = FreeCAD.Vector(0, 0, 1)
        if abs(normal.dot(z_axis)) > 0.999:
            rotation = FreeCAD.Rotation() if normal.z > 0 else FreeCAD.Rotation(FreeCAD.Vector(1, 0, 0), 180)
        else:
            rotation = FreeCAD.Rotation(z_axis, normal)
        return FreeCAD.Placement(center, rotation)

    def _build_sketch_geometry_batch(self, edges, inverse_placement):
        """OPTIMIZED: Build geometry batch with vertex caching - avoids repeated edge.Vertexes calls"""
        geometries = []
        geo_metadata = []
        edge_signatures = set()
        duplicate_edges = 0
        tolerance = 0.001

        FreeCAD.Console.PrintMessage(f"DEBUG: Building geometry batch for {len(edges)} edges...\n")

        # PRE-CACHE all vertex data ONCE to avoid repeated API calls
        edge_vertices_cache = []
        for edge in edges:
            try:
                vertices = edge.Vertexes
                if len(vertices) >= 2:
                    v_start = vertices[0].Point
                    v_end = vertices[-1].Point
                    edge_vertices_cache.append((v_start, v_end))
                else:
                    edge_vertices_cache.append(None)
            except:
                edge_vertices_cache.append(None)

        FreeCAD.Console.PrintMessage(f"DEBUG: Cached {len(edge_vertices_cache)} vertex pairs\n")

        # Now build geometries using cached data
        for i, vertex_pair in enumerate(edge_vertices_cache):
            if i % 500 == 0 and i > 0:
                FreeCAD.Console.PrintMessage(f"DEBUG: Processed {i}/{len(edges)} edges\n")

            if vertex_pair is None:
                continue

            v_start, v_end = vertex_pair

            if (v_start - v_end).Length < tolerance:
                continue  # Skip degenerate

            # Create signature to detect duplicate edges (check both directions)
            sig1 = (round(v_start.x, 4), round(v_start.y, 4), round(v_start.z, 4),
                    round(v_end.x, 4), round(v_end.y, 4), round(v_end.z, 4))
            sig2 = (round(v_end.x, 4), round(v_end.y, 4), round(v_end.z, 4),
                    round(v_start.x, 4), round(v_start.y, 4), round(v_start.z, 4))

            if sig1 in edge_signatures or sig2 in edge_signatures:
                duplicate_edges += 1
                continue  # Skip duplicate
            edge_signatures.add(sig1)

            v_start_local = inverse_placement.multVec(v_start)
            v_end_local = inverse_placement.multVec(v_end)

            geometries.append(Part.LineSegment(v_start_local, v_end_local))
            geo_metadata.append((v_start, v_end))

        FreeCAD.Console.PrintMessage(f"DEBUG: Built {len(geometries)} geometries ({duplicate_edges} duplicates skipped)\n")

        return geometries, geo_metadata

    def _add_coincident_constraints_fast(self, sketch, tolerance=100e-6):
        """Add missing coincident constraints using FreeCAD's built-in fast method.

        Args:
            sketch: The sketch object to add constraints to
            tolerance: Distance tolerance for coincident detection (default 100µm)

        Returns:
            Number of constraints added
        """
        try:
            # Detect missing coincident constraints
            num_missing = sketch.detectMissingPointOnPointConstraints(tolerance)

            if num_missing == 0:
                return 0

            # Apply constraints using batch mode (False = don't solve after each)
            sketch.makeMissingPointOnPointCoincident(False)

            return num_missing

        except Exception as e:
            FreeCAD.Console.PrintWarning(f"Failed to add coincident constraints: {e}\n")
            return 0

    def create_standalone_sketch(self, temp_sketch, edges):
        """OPTIMIZED: Create standalone sketch using batch geometry addition - NO CONSTRAINTS"""
        doc = FreeCAD.ActiveDocument

        FreeCAD.Console.PrintMessage(f"DEBUG: Creating standalone sketch for {len(edges)} edges\n")

        final_sketch = doc.addObject("Sketcher::SketchObject", "Sketch")
        final_sketch.Placement = temp_sketch.Placement

        # Cache the inverse placement once
        inverse_placement = final_sketch.getGlobalPlacement().inverse()

        # OPTIMIZATION: Build geometry batch using cached vertices
        geometries, geo_metadata = self._build_sketch_geometry_batch(edges, inverse_placement)

        # OPTIMIZATION: Disable auto-solve during batch operations
        if hasattr(final_sketch, 'setAutomaticSolve'):
            try:
                final_sketch.setAutomaticSolve(False)
                FreeCAD.Console.PrintMessage("DEBUG: Disabled auto-solve\n")
            except:
                pass

        # OPTIMIZATION: Batch add all geometries at once - MASSIVE SPEEDUP!
        FreeCAD.Console.PrintMessage(f"DEBUG: Batch adding {len(geometries)} geometries...\n")
        add_start = time.time()
        # Pass True as third parameter to mark as construction during creation
        geo_indices = final_sketch.addGeometry(geometries, True)  # True = construction mode
        add_duration = time.time() - add_start
        FreeCAD.Console.PrintMessage(f"DEBUG: Batch add complete (with construction flag) in {add_duration:.3f}s\n")

        # OPTIMIZATION: Re-enable automatic solving
        if hasattr(final_sketch, 'setAutomaticSolve'):
            try:
                final_sketch.setAutomaticSolve(True)
            except:
                pass

        FreeCAD.Console.PrintMessage(f"DEBUG: Calling recompute\n")

        # Single recompute at the end
        recompute_start = time.time()
        doc.recompute()
        recompute_duration = time.time() - recompute_start
        FreeCAD.Console.PrintMessage(f"DEBUG: Recompute complete in {recompute_duration:.3f}s\n")

        # OPTIMIZATION: Add coincident constraints using fast built-in method
        constraint_start = time.time()
        num_constraints_added = self._add_coincident_constraints_fast(final_sketch)
        constraint_duration = time.time() - constraint_start
        if num_constraints_added > 0:
            FreeCAD.Console.PrintMessage(f"DEBUG: Added {num_constraints_added} coincident constraints in {constraint_duration:.3f}s\n")

        return final_sketch, num_constraints_added, 0  # Return constraints added count

    def create_body_sketch(self, temp_sketch, edges, target_body):
        """OPTIMIZED: Create sketch attached to PartDesign body - NO CONSTRAINTS"""
        FreeCAD.Console.PrintMessage(f"DEBUG: create_body_sketch CALLED with {len(edges)} edges\n")

        doc = FreeCAD.ActiveDocument

        final_sketch = doc.addObject("Sketcher::SketchObject", "Sketch")

        # Add sketch to body
        target_body.ViewObject.dropObject(final_sketch, None, '', [])
        FreeCAD.Console.PrintMessage(f"DEBUG: Sketch added to body\n")

        # Set up attachment to body origin
        final_sketch.AttachmentSupport = [(target_body.Origin.OriginFeatures[0], '')]
        final_sketch.MapMode = 'ObjectXY'
        final_sketch.AttachmentOffset.Base = temp_sketch.Placement.Base
        final_sketch.AttachmentOffset.Rotation = temp_sketch.Placement.Rotation
        final_sketch.Placement = FreeCAD.Placement()

        doc.recompute()  # Needed to resolve attachment before adding geometry

        # Cache the inverse placement once
        inverse_placement = final_sketch.getGlobalPlacement().inverse()

        # OPTIMIZATION: Build geometry batch using cached vertices
        geometries, geo_metadata = self._build_sketch_geometry_batch(edges, inverse_placement)

        # OPTIMIZATION: Disable auto-solve during batch operations
        if hasattr(final_sketch, 'setAutomaticSolve'):
            try:
                final_sketch.setAutomaticSolve(False)
                FreeCAD.Console.PrintMessage("DEBUG: Disabled auto-solve\n")
            except:
                pass

        # OPTIMIZATION: Batch add all geometries at once - MASSIVE SPEEDUP!
        FreeCAD.Console.PrintMessage(f"DEBUG: Batch adding {len(geometries)} geometries...\n")
        add_start = time.time()
        # Pass True as third parameter to mark as construction during creation
        geo_indices = final_sketch.addGeometry(geometries, True)  # True = construction mode
        add_duration = time.time() - add_start
        FreeCAD.Console.PrintMessage(f"DEBUG: Batch add complete (with construction flag) in {add_duration:.3f}s\n")

        # OPTIMIZATION: Re-enable automatic solving
        if hasattr(final_sketch, 'setAutomaticSolve'):
            try:
                final_sketch.setAutomaticSolve(True)
            except:
                pass

        FreeCAD.Console.PrintMessage(f"DEBUG: Calling recompute\n")

        # Single recompute at the end
        recompute_start = time.time()
        doc.recompute()
        recompute_duration = time.time() - recompute_start
        FreeCAD.Console.PrintMessage(f"DEBUG: Recompute complete in {recompute_duration:.3f}s\n")

        # OPTIMIZATION: Add coincident constraints using fast built-in method
        constraint_start = time.time()
        num_constraints_added = self._add_coincident_constraints_fast(final_sketch)
        constraint_duration = time.time() - constraint_start
        if num_constraints_added > 0:
            FreeCAD.Console.PrintMessage(f"DEBUG: Added {num_constraints_added} coincident constraints in {constraint_duration:.3f}s\n")

        return final_sketch, num_constraints_added, 0  # Return constraints added count

    def show_destination_dialog(self):
        """Show destination dialog and return choice info"""
        try:
            FreeCAD.Console.PrintMessage("DEBUG: Starting destination dialog...\n")

            doc = FreeCAD.ActiveDocument
            body_names = [o.Name for o in doc.Objects if o.isDerivedFrom("PartDesign::Body")]
            options = ["<Standalone (Part Workbench)>", "<Create New Body (PartDesign)>"] + body_names

            FreeCAD.Console.PrintMessage(f"DEBUG: Dialog options: {options}\n")

            item, ok = QInputDialog.getItem(FreeCADGui.getMainWindow(),
                                            "Sketch Placement Options",
                                            "Choose a placement option:",
                                            options, 0, False)

            FreeCAD.Console.PrintMessage(f"DEBUG: Dialog result - item: '{item}', ok: {ok}\n")

            if not ok or not item:
                FreeCAD.Console.PrintMessage("DEBUG: Dialog cancelled or no item selected\n")
                return None

            if item == "<Standalone (Part Workbench)>":
                FreeCAD.Console.PrintMessage("DEBUG: Standalone option selected\n")
                return {"type": "standalone"}
            elif item == "<Create New Body (PartDesign)>":
                FreeCAD.Console.PrintMessage("DEBUG: New body option selected\n")
                return {"type": "new_body"}
            else:
                FreeCAD.Console.PrintMessage(f"DEBUG: Existing body option selected: {item}\n")
                return {"type": "existing_body", "body_name": item}

        except Exception as e:
            FreeCAD.Console.PrintError(f"DEBUG: Dialog error: {e}\n")
            return {"type": "standalone"}  # Fallback

    def collect_unique_vertices_fast(self, edges, tolerance=1e-6):
        """OPTIMIZED: Fast vertex deduplication using spatial hashing - O(n) instead of O(n²)"""

        def hash_point(point, tol):
            # Grid-based bucketing for fast duplicate detection
            return (round(point.x / tol), round(point.y / tol), round(point.z / tol))

        unique_vertices = []
        vertex_hash_set = set()

        for edge in edges:
            for v in edge.Vertexes:
                p = v.Point
                p_hash = hash_point(p, tolerance)

                if p_hash not in vertex_hash_set:
                    vertex_hash_set.add(p_hash)
                    unique_vertices.append(p)

        return unique_vertices

    def create_sketch_from_selection(self):
        # Check if edge data has been collected first
        if not self.collected_edges:
            self.info_display.append("Error: No edge data collected. Click 'Collect Edge Data' first.")
            return

        doc = FreeCAD.ActiveDocument
        if not doc:
            self.info_display.append("Error: No active FreeCAD document.")
            return

        # Check if any edges are selected
        selection = FreeCADGui.Selection.getSelectionEx()
        if not selection or not any(name.startswith("Edge") for s in selection for name in s.SubElementNames):
            self.info_display.append("Error: No edges selected. Use 'Select Coplanar Edges' or manually select edges first.")
            return

        start_time = time.time()
        doc.openTransaction("Create Sketch from Selection")
        temp_sketch = None
        final_sketch = None

        try:
            FreeCAD.Console.PrintMessage("DEBUG: Starting sketch creation\n")
            FreeCAD.Console.PrintMessage(f"DEBUG: Processing selection\n")

            selected_edges = []
            selected_objects = FreeCADGui.Selection.getSelectionEx()
            source_object = None

            for sel in selected_objects:
                source_object = sel.Object if not source_object else source_object
                selected_edges.extend([sub for sub in sel.SubObjects if isinstance(sub, Part.Edge)])

            if not selected_edges:
                self.info_display.append("No edges selected.")
                doc.abortTransaction()
                return

            FreeCAD.Console.PrintMessage(f"DEBUG: Found {len(selected_edges)} edges\n")

            # OPTIMIZATION: Use fast spatial hashing for vertex deduplication - O(n) instead of O(n²)
            FreeCAD.Console.PrintMessage("DEBUG: Collecting unique vertices (optimized)\n")
            vertex_start = time.time()
            unique_vertices = self.collect_unique_vertices_fast(selected_edges, tolerance=1e-6)
            vertex_duration = time.time() - vertex_start
            FreeCAD.Console.PrintMessage(f"DEBUG: Found {len(unique_vertices)} unique vertices in {vertex_duration:.3f}s\n")

            # OPTIMIZATION: Use sampling-based plane calculation for large vertex sets
            plane_start = time.time()
            normal, center = self.calculate_robust_plane_normal_and_placement(unique_vertices, source_object)
            placement = self.create_robust_placement(normal, center)
            plane_duration = time.time() - plane_start
            FreeCAD.Console.PrintMessage(f"DEBUG: Calculated placement in {plane_duration:.3f}s\n")

            # Show destination dialog FIRST, before creating any sketches
            FreeCAD.Console.PrintMessage("DEBUG: About to show dialog\n")
            choice = self.show_destination_dialog()
            FreeCAD.Console.PrintMessage(f"DEBUG: Dialog returned: {choice}\n")

            if not choice:
                self.info_display.append("Sketch creation cancelled by user.")
                doc.abortTransaction()
                return

            # Now create temp sketch for placement (only after user confirms)
            FreeCAD.Console.PrintMessage("DEBUG: Creating temp sketch\n")
            temp_sketch = doc.addObject("Sketcher::SketchObject", "TempSketch")
            temp_sketch.Placement = placement
            doc.recompute()
            FreeCAD.Console.PrintMessage("DEBUG: Temp sketch created\n")

            # Create sketch based on user choice
            FreeCAD.Console.PrintMessage(f"DEBUG: Creating final sketch for {len(selected_edges)} edges\n")

            num_constraints_added = 0
            if choice["type"] == "standalone":
                final_sketch, num_constraints_added, _ = self.create_standalone_sketch(temp_sketch, selected_edges)
            elif choice["type"] == "new_body":
                # Create new body first
                FreeCAD.Console.PrintMessage("DEBUG: Creating new body\n")
                target_body = doc.addObject("PartDesign::Body", "NewBody")
                FreeCAD.Console.PrintMessage("DEBUG: Body created, now creating sketch\n")
                final_sketch, num_constraints_added, _ = self.create_body_sketch(temp_sketch, selected_edges, target_body)
            elif choice["type"] == "existing_body":
                # Get existing body
                target_body = doc.getObject(choice["body_name"])
                if not target_body:
                    self.info_display.append(f"Error: Body {choice['body_name']} not found.")
                    if temp_sketch:
                        doc.removeObject(temp_sketch.Name)
                    doc.abortTransaction()
                    return
                final_sketch, num_constraints_added, _ = self.create_body_sketch(temp_sketch, selected_edges, target_body)

            doc.recompute()
            FreeCADGui.Selection.clearSelection()
            FreeCADGui.Selection.addSelection(final_sketch)
            FreeCADGui.activeDocument().activeView().viewAxonometric()
            FreeCADGui.activeDocument().activeView().fitAll()

            # Clean up temporary sketch now that we're done
            try:
                if temp_sketch is not None:
                    doc.removeObject(temp_sketch.Name)
                    doc.recompute()
            except Exception as e:
                FreeCAD.Console.PrintWarning(f"Could not remove temporary sketch: {e}\n")

            # Inform user about constraint addition
            self.info_display.append(f"Note: Coincident constraints added using fast built-in method ({num_constraints_added} constraints).")

            duration = time.time() - start_time
            self.info_display.append("Sketch created successfully.")
            self.info_display.append(f"Elapsed time: {duration:.4f} seconds.\n")
            doc.commitTransaction()

        except Exception as e:
            doc.abortTransaction()
            self.info_display.append(f"Sketch creation failed:\n{e}")
            FreeCAD.Console.PrintError(f"Sketch error: {e}\n")
            import traceback
            FreeCAD.Console.PrintError(traceback.format_exc())
            # Try to clean up temp sketch on error too
            try:
                if temp_sketch is not None:
                    doc.removeObject(temp_sketch.Name)
            except:
                pass

    def clean_degenerate_edges(self):
        doc = FreeCAD.ActiveDocument
        if not doc:
            self.info_display.append("Error: No active FreeCAD document.")
            return

        selection = FreeCADGui.Selection.getSelectionEx()
        if not selection or not hasattr(selection[0].Object, "Shape"):
            self.info_display.append("Error: Select a valid Part object to clean.")
            return

        start_time = time.time()
        doc.openTransaction("Clean Degenerate Edges")

        try:
            source_object = selection[0].Object
            original_shape = source_object.Shape

            self.info_display.append(f"Cleaning degenerate edges from {source_object.Label}...")

            # Copy the shape to edit
            shape = original_shape.copy()
            valid_faces = []
            skipped_faces = 0

            # Check each face for degenerate edge references
            for f in shape.Faces:
                try:
                    degens = [e for e in f.Edges if len(e.Vertexes) < 2]
                    if not degens:
                        valid_faces.append(f)
                    else:
                        skipped_faces += 1
                except Exception as err:
                    self.info_display.append(f"Warning: Skipping face due to error: {err}")
                    skipped_faces += 1

            if not valid_faces:
                self.info_display.append("Error: No valid faces found after cleaning.")
                doc.abortTransaction()
                return

            # Rebuild shape from retained faces
            cleaned_shape = Part.Compound(valid_faces)

            # Create new object with cleaned shape
            cleaned_object = doc.addObject("Part::Feature", f"{source_object.Label}_Cleaned")
            cleaned_object.Shape = cleaned_shape
            cleaned_object.Label = f"{source_object.Label}_Cleaned"

            doc.recompute()

            # Select the new clean object and hide the original
            FreeCADGui.Selection.clearSelection()
            FreeCADGui.Selection.addSelection(cleaned_object)
            source_object.Visibility = False

            duration = time.time() - start_time
            self.info_display.append(f"Created cleaned object: {cleaned_object.Label}")
            self.info_display.append(f"Retained {len(valid_faces)} faces, skipped {skipped_faces} faces with degenerate edges.")
            self.info_display.append(f"Original object hidden following Part workbench convention.")
            self.info_display.append(f"Elapsed time: {duration:.4f} seconds.")
            self.info_display.append("Automatically collecting edge data from cleaned object...\n")

            doc.commitTransaction()

            # Automatically re-collect data on the cleaned object
            self.collect_data()

        except Exception as e:
            doc.abortTransaction()
            self.info_display.append(f"Cleaning failed: {e}")
            FreeCAD.Console.PrintError(f"Cleaning error: {e}\n")

    def clear_messages(self):
        self.info_display.clear()

def show_edge_data_collector_docker():
    mw = FreeCADGui.getMainWindow()
    for d in mw.findChildren(QDockWidget):
        if d.windowTitle() == "CoplanarSketch":
            d.close()
            d.deleteLater()
    mw.addDockWidget(Qt.RightDockWidgetArea, EdgeDataCollector())

show_edge_data_collector_docker()
