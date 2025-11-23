import FreeCAD
import FreeCADGui
import Part
import Sketcher
import PartDesign
from PySide.QtWidgets import QDockWidget, QWidget, QVBoxLayout, QPushButton, QTextEdit, QLabel, QInputDialog, QLineEdit
from PySide.QtCore import Qt
import time

class EdgeDataCollector(QDockWidget):
    def __init__(self):
        super().__init__("CoplanarSketch")
        self.setWidget(self.create_ui())
        self.collected_edges = []
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

        # Collect edges with validity checking
        self.collected_edges = []
        invalid_count = 0
        degenerate_count = 0
        property_error_count = 0

        for edge in edges:
            edge_index = edges.index(edge)  # Get actual FreeCAD edge index

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
        selected_edge_names = [name for s in selection for name in s.SubElementNames if name.startswith("Edge")]
        selected_face_names = [name for s in selection for name in s.SubElementNames if name.startswith("Face")]

        if selected_face_names:
            face_idx = int(selected_face_names[0][4:]) - 1
            plane_normal = obj.Shape.Faces[face_idx].Surface.Axis
            plane_point = obj.Shape.Faces[face_idx].CenterOfMass
            self.info_display.append(f"Using plane defined by face: {selected_face_names[0]}")
        elif len(selected_edge_names) >= 2:
            # Use cached vertex points from our collected data for plane calculation
            edge1_name = selected_edge_names[0]
            edge2_name = selected_edge_names[1]

            # Find edges in our collected data
            edge1_dict = next((ed for ed in self.collected_edges if ed['name'] == edge1_name), None)
            edge2_dict = next((ed for ed in self.collected_edges if ed['name'] == edge2_name), None)

            if not edge1_dict or not edge2_dict:
                self.info_display.append("Error: Selected edges not found in collected data.")
                return

            if edge1_dict['vertex_count'] != 2 or edge2_dict['vertex_count'] != 2:
                self.info_display.append("Error: Selected edges have invalid vertex counts.")
                return

            # Collect all unique vertices from both edges
            all_vertices = edge1_dict['vertex_points'] + edge2_dict['vertex_points']
            unique_vertices = []
            for v in all_vertices:
                if not any((v - existing).Length < 1e-6 for existing in unique_vertices):
                    unique_vertices.append(v)

            if len(unique_vertices) < 3:
                self.info_display.append("Error: Edges are colinear; cannot define plane.")
                return

            # Use first three unique vertices to define plane
            v1, v2, v3 = unique_vertices[:3]
            plane_normal = (v2 - v1).cross(v3 - v1)
            if plane_normal.Length > 0:  # Only check for truly zero cross product
                plane_normal = plane_normal.normalize()
            else:
                self.info_display.append("Error: Cannot define plane from selected edges.")
                return

            plane_point = v1
            self.info_display.append(f"Using plane defined by edges: {edge1_name}, {edge2_name}")
        else:
            self.info_display.append("Error: Select either a face or two edges.")
            return

        def is_coplanar(edge_dict):
            if edge_dict['vertex_count'] != 2:
                return False
            v1, v2 = edge_dict['vertex_points']
            try:
                # Clamp user input between 1e-6 and 1.0
                tol = max(1e-6, min(float(self.tolerance_input.text()), 1.0))
            except:
                tol = 1e-6  # fallback if input is invalid
            return abs((v1 - plane_point).dot(plane_normal)) < tol and \
                   abs((v2 - plane_point).dot(plane_normal)) < tol

        coplanar_edge_dicts = [edge_dict for edge_dict in self.collected_edges if is_coplanar(edge_dict)]
        coplanar_edges = [edge_dict['edge'] for edge_dict in coplanar_edge_dicts]

        # Validate coplanar results
        total_valid_edges = len([ed for ed in self.collected_edges if ed['valid']])
        if len(coplanar_edges) > total_valid_edges * 0.5:
            self.info_display.append(f"Warning: {len(coplanar_edges)} coplanar edges found ({len(coplanar_edges)/total_valid_edges*100:.1f}% of valid edges) - check plane definition.")

        FreeCADGui.Selection.clearSelection()
        for edge_dict in coplanar_edge_dicts:
            FreeCADGui.Selection.addSelection(obj, edge_dict['name'])

        duration = time.time() - start_time
        self.info_display.append(f"Selected {len(coplanar_edges)} coplanar edges.")
        self.info_display.append(f"Elapsed time: {duration:.4f} seconds.\n")

        # Show Create Sketch button after coplanar selection
        self.create_sketch_button.setVisible(True)

    def calculate_robust_plane_normal_and_placement(self, vertices, source_object=None):
        if len(vertices) < 3:
            return FreeCAD.Vector(0, 0, 1), FreeCAD.Vector(0, 0, 0)

        center = sum(vertices, FreeCAD.Vector()).multiply(1.0 / len(vertices))
        vectors = [v.sub(center) for v in vertices]

        best_normal = None
        best_magnitude = 0
        for i in range(len(vectors)):
            for j in range(i+1, len(vectors)):
                for k in range(j+1, len(vectors)):
                    n = (vectors[j] - vectors[i]).cross(vectors[k] - vectors[i])
                    if n.Length > best_magnitude:
                        best_magnitude = n.Length
                        best_normal = n.normalize()

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

    def create_standalone_sketch(self, temp_sketch, edges):
        """Create standalone sketch using current stable implementation"""
        doc = FreeCAD.ActiveDocument

        final_sketch = doc.addObject("Sketcher::SketchObject", "Sketch")
        final_sketch.Placement = temp_sketch.Placement
        doc.recompute()

        edge_map = {}
        tolerance = 0.001
        for edge in edges:
            v_start, v_end = edge.Vertexes[0].Point, edge.Vertexes[-1].Point
            if (v_start - v_end).Length < tolerance:
                continue  # Skip degenerate

            v_start_local = final_sketch.getGlobalPlacement().inverse().multVec(v_start)
            v_end_local = final_sketch.getGlobalPlacement().inverse().multVec(v_end)

            geo_index = final_sketch.addGeometry(Part.LineSegment(v_start_local, v_end_local), False)
            final_sketch.setConstruction(geo_index, True)

            for point, vid in [(v_start, 1), (v_end, 2)]:
                key = (round(point.x, 5), round(point.y, 5), round(point.z, 5))
                edge_map.setdefault(key, []).append((geo_index, vid))

        for group in edge_map.values():
            base = group[0]
            for other in group[1:]:
                try:
                    final_sketch.addConstraint(Sketcher.Constraint('Coincident', base[0], base[1], other[0], other[1]))
                except Exception as e:
                    FreeCAD.Console.PrintWarning(f"Coincident error: {e}\n")

        return final_sketch

    def create_body_sketch(self, temp_sketch, edges, target_body):
        """Create sketch attached to PartDesign body"""
        doc = FreeCAD.ActiveDocument

        final_sketch = doc.addObject("Sketcher::SketchObject", "Sketch")

        # Add sketch to body
        target_body.ViewObject.dropObject(final_sketch, None, '', [])
        doc.recompute()

        # Set up attachment to body origin
        final_sketch.AttachmentSupport = [(target_body.Origin.OriginFeatures[0], '')]
        final_sketch.MapMode = 'ObjectXY'
        final_sketch.AttachmentOffset.Base = temp_sketch.Placement.Base
        final_sketch.AttachmentOffset.Rotation = temp_sketch.Placement.Rotation
        final_sketch.Placement = FreeCAD.Placement()

        doc.recompute()

        # Add geometry (same as standalone)
        edge_map = {}
        tolerance = 0.001
        for edge in edges:
            v_start, v_end = edge.Vertexes[0].Point, edge.Vertexes[-1].Point
            if (v_start - v_end).Length < tolerance:
                continue  # Skip degenerate

            v_start_local = final_sketch.getGlobalPlacement().inverse().multVec(v_start)
            v_end_local = final_sketch.getGlobalPlacement().inverse().multVec(v_end)

            geo_index = final_sketch.addGeometry(Part.LineSegment(v_start_local, v_end_local), False)
            final_sketch.setConstruction(geo_index, True)

            for point, vid in [(v_start, 1), (v_end, 2)]:
                key = (round(point.x, 5), round(point.y, 5), round(point.z, 5))
                edge_map.setdefault(key, []).append((geo_index, vid))

        for group in edge_map.values():
            base = group[0]
            for other in group[1:]:
                try:
                    final_sketch.addConstraint(Sketcher.Constraint('Coincident', base[0], base[1], other[0], other[1]))
                except Exception as e:
                    FreeCAD.Console.PrintWarning(f"Coincident error: {e}\n")

        return final_sketch

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

            all_vertices = [v.Point for edge in selected_edges for v in edge.Vertexes]
            unique_vertices = []
            for p in all_vertices:
                if not any((p - q).Length < 1e-4 for q in unique_vertices):
                    unique_vertices.append(p)

            FreeCAD.Console.PrintMessage(f"DEBUG: Found {len(unique_vertices)} unique vertices\n")

            normal, center = self.calculate_robust_plane_normal_and_placement(unique_vertices, source_object)
            placement = self.create_robust_placement(normal, center)

            FreeCAD.Console.PrintMessage("DEBUG: Calculated placement\n")

            # Create temporary sketch for placement calculation
            temp_sketch = doc.addObject("Sketcher::SketchObject", "TempSketch")
            temp_sketch.Placement = placement
            doc.recompute()

            FreeCAD.Console.PrintMessage("DEBUG: Created temp sketch, about to call dialog\n")

            # Show destination dialog
            choice = self.show_destination_dialog()
            FreeCAD.Console.PrintMessage(f"DEBUG: Dialog returned: {choice}\n")

            if not choice:
                self.info_display.append("Sketch creation cancelled by user.")
                if temp_sketch:
                    doc.removeObject(temp_sketch.Name)
                doc.abortTransaction()
                return

            # Create sketch based on user choice
            if choice["type"] == "standalone":
                final_sketch = self.create_standalone_sketch(temp_sketch, selected_edges)
            elif choice["type"] == "new_body":
                # Create new body first
                target_body = doc.addObject("PartDesign::Body", "NewBody")
                doc.recompute()
                final_sketch = self.create_body_sketch(temp_sketch, selected_edges, target_body)
            elif choice["type"] == "existing_body":
                # Get existing body
                target_body = doc.getObject(choice["body_name"])
                if not target_body:
                    self.info_display.append(f"Error: Body {choice['body_name']} not found.")
                    doc.abortTransaction()
                    return
                final_sketch = self.create_body_sketch(temp_sketch, selected_edges, target_body)

            doc.recompute()
            FreeCADGui.Selection.clearSelection()
            FreeCADGui.Selection.addSelection(final_sketch)
            FreeCADGui.activeDocument().activeView().viewAxonometric()
            FreeCADGui.activeDocument().activeView().fitAll()

            duration = time.time() - start_time
            self.info_display.append("Sketch created with destination dialog selection.")
            self.info_display.append(f"Elapsed time: {duration:.4f} seconds.\n")
            doc.commitTransaction()

        except Exception as e:
            doc.abortTransaction()
            self.info_display.append(f"Sketch creation failed:\n{e}")
            FreeCAD.Console.PrintError(f"Sketch error: {e}\n")
        finally:
            # Clean up temporary sketch
            if temp_sketch and hasattr(temp_sketch, 'Name'):
                try:
                    doc.removeObject(temp_sketch.Name)
                    doc.recompute()
                except Exception as e:
                    FreeCAD.Console.PrintWarning(f"Could not remove temporary sketch: {e}\n")

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
