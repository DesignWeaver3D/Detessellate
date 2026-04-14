import FreeCAD as App
import FreeCADGui as Gui
from PySide import QtCore, QtGui
import Part
import math
from typing import List, Dict, Tuple, Optional, Any


class GeometryMatcher:
    """Handles geometric matching logic for faces, edges, and vertices."""

    TOLERANCE = 1e-6

    @staticmethod
    def faces_exact_match(face1: Part.Face, face2: Part.Face) -> bool:
        """Check if two faces are exact matches."""
        try:
            # Check surface type
            if face1.Surface.TypeId != face2.Surface.TypeId:
                return False

            # Check area
            if abs(face1.Area - face2.Area) > GeometryMatcher.TOLERANCE:
                return False

            # Check center of mass
            com1 = face1.CenterOfMass
            com2 = face2.CenterOfMass
            if com1.distanceToPoint(com2) > GeometryMatcher.TOLERANCE:
                return False

            # Check normal vectors for planar faces
            if hasattr(face1.Surface, 'Axis') and hasattr(face2.Surface, 'Axis'):
                if abs(face1.Surface.Axis.dot(face2.Surface.Axis)) < (1 - GeometryMatcher.TOLERANCE):
                    return False

            return True

        except Exception:
            return False

    @staticmethod
    def faces_similar_match(face1: Part.Face, face2: Part.Face) -> bool:
        """
        Check if two faces are similar.
        For planar faces: only coplanarity.
        For non-planar faces: 3D bounding box overlap.
        """
        try:
            is_face1_planar = (face1.Surface.TypeId == "Part::GeomPlane")
            is_face2_planar = (face2.Surface.TypeId == "Part::GeomPlane")

            if is_face1_planar and is_face2_planar:
                # Both faces are planar: check only coplanarity as requested
                # Check if coplanar (normal vectors parallel)
                normal1 = face1.Surface.Axis
                normal2 = face2.Surface.Axis
                if abs(normal1.dot(normal2)) < (1 - GeometryMatcher.TOLERANCE):
                    return False

                # Check if on same plane
                point1 = face1.Surface.Position
                point2 = face2.Surface.Position
                distance = abs(normal1.dot(point2 - point1))
                if distance > GeometryMatcher.TOLERANCE:
                    return False

                return True # Coplanar faces are considered similar as per new relaxed criteria

            else:
                # At least one face is non-planar (or both are non-planar): use 3D bounding box overlap
                bb1 = face1.BoundBox
                bb2 = face2.BoundBox

                if bb1.isNull() or bb2.isNull():
                    return False
                
                intersected_bb = bb1.intersect(bb2)
                return not intersected_bb.isNull()

        except Exception:
            return False

    @staticmethod
    def edges_exact_match(edge1: Part.Edge, edge2: Part.Edge) -> bool:
        """Check if two edges are exact matches."""
        try:
            is_circle1 = edge1.Curve.TypeId == "Part::GeomCircle"
            is_circle2 = edge2.Curve.TypeId == "Part::GeomCircle"

            if is_circle1 and is_circle2:
                center1 = edge1.Curve.Center
                center2 = edge2.Curve.Center
                if center1.distanceToPoint(center2) > GeometryMatcher.TOLERANCE:
                    return False

                radius1 = edge1.Curve.Radius
                radius2 = edge2.Curve.Radius
                if abs(radius1 - radius2) > GeometryMatcher.TOLERANCE:
                    return False

                if hasattr(edge1.Curve, 'Axis') and hasattr(edge2.Curve, 'Axis'):
                    axis1 = edge1.Curve.Axis
                    axis2 = edge2.Curve.Axis
                    if abs(axis1.dot(axis2)) < (1 - GeometryMatcher.TOLERANCE):
                        return False

                # Compare angular span
                a1s = edge1.Curve.FirstParameter
                a1e = edge1.Curve.LastParameter
                a2s = edge2.Curve.FirstParameter
                a2e = edge2.Curve.LastParameter

                # Normalize angles
                while a1e < a1s:
                    a1e += 2 * math.pi
                while a2e < a2s:
                    a2e += 2 * math.pi

                angle1 = abs(a1e - a1s)
                angle2 = abs(a2e - a2s)

                if abs(angle1 - angle2) > GeometryMatcher.TOLERANCE:
                    return False

                # Ensure endpoints match in either direction
                if len(edge1.Vertexes) < 2 or len(edge2.Vertexes) < 2:
                    return False

                start1, end1 = edge1.Vertexes[0].Point, edge1.Vertexes[1].Point
                start2, end2 = edge2.Vertexes[0].Point, edge2.Vertexes[1].Point

                forward = (start1.distanceToPoint(start2) < GeometryMatcher.TOLERANCE and
                           end1.distanceToPoint(end2) < GeometryMatcher.TOLERANCE)
                reverse = (start1.distanceToPoint(end2) < GeometryMatcher.TOLERANCE and
                           end1.distanceToPoint(start2) < GeometryMatcher.TOLERANCE)

                return forward or reverse

            # Fallback for non-circular edges
            if edge1.Curve.TypeId != edge2.Curve.TypeId:
                return False

            if abs(edge1.Length - edge2.Length) > GeometryMatcher.TOLERANCE:
                return False

            if len(edge1.Vertexes) < 2 or len(edge2.Vertexes) < 2:
                return False

            start1, end1 = edge1.Vertexes[0].Point, edge1.Vertexes[1].Point
            start2, end2 = edge2.Vertexes[0].Point, edge2.Vertexes[1].Point

            forward_match = (start1.distanceToPoint(start2) < GeometryMatcher.TOLERANCE and
                             end1.distanceToPoint(end2) < GeometryMatcher.TOLERANCE)
            reverse_match = (start1.distanceToPoint(end2) < GeometryMatcher.TOLERANCE and
                             end1.distanceToPoint(start2) < GeometryMatcher.TOLERANCE)

            return forward_match or reverse_match

        except Exception:
            return False


    @staticmethod
    def edges_similar_match(edge1: Part.Edge, edge2: Part.Edge) -> bool:
        """Check if two edges are similar (collinear with overlapping positions for lines, or same plane, radius, center for arcs)."""
        try:
            # Check if both are lines
            if edge1.Curve.TypeId == "Part::GeomLine" and edge2.Curve.TypeId == "Part::GeomLine":
                return GeometryMatcher._check_line_similarity(edge1, edge2)

            # Check if both are circles/arcs
            elif edge1.Curve.TypeId == "Part::GeomCircle" and edge2.Curve.TypeId == "Part::GeomCircle":
                return GeometryMatcher._check_circle_similarity(edge1, edge2)

            # For other curve types, they are not considered "similar" based on the strict criteria.
            return False

        except Exception:
            return False

    @staticmethod
    def _check_line_similarity(edge1: Part.Edge, edge2: Part.Edge) -> bool:
        """Check similarity for linear edges (exactly collinear with overlap)."""
        try:
            # Check if collinear (directions are parallel or anti-parallel)
            dir1 = edge1.Curve.Direction
            dir2 = edge2.Curve.Direction
            
            # Use abs() to account for anti-parallel directions
            if abs(dir1.dot(dir2)) < (1 - GeometryMatcher.TOLERANCE):
                return False

            # Check if they lie on the same line (origin of one is on the other's line)
            line_origin1 = edge1.Curve.Location
            line_origin2 = edge2.Curve.Location
            
            # Use cross product magnitude to check if line_origin2 is on the line of edge1
            vec_to_origin2 = line_origin2 - line_origin1
            cross_product_magnitude = vec_to_origin2.cross(dir1).Length
            
            if cross_product_magnitude > GeometryMatcher.TOLERANCE:
                return False

            # Get edge endpoints
            start1, end1 = edge1.Vertexes[0].Point, edge1.Vertexes[1].Point
            start2, end2 = edge2.Vertexes[0].Point, edge2.Vertexes[1].Point

            # Project all points onto the common line direction
            proj_start1 = dir1.dot(start1 - line_origin1)
            proj_end1 = dir1.dot(end1 - line_origin1)
            proj_start2 = dir1.dot(start2 - line_origin1)
            proj_end2 = dir1.dot(end2 - line_origin1)

            # Ensure proper ordering (start < end) for intervals
            if proj_end1 < proj_start1:
                proj_start1, proj_end1 = proj_end1, proj_start1
            if proj_end2 < proj_start2:
                proj_start2, proj_end2 = proj_end2, proj_start2

            # Check for overlap in the projected 1D intervals
            overlap_start = max(proj_start1, proj_start2)
            overlap_end = min(proj_end1, proj_end2)
            
            return (overlap_end - overlap_start) > GeometryMatcher.TOLERANCE

        except Exception:
            return False

    @staticmethod
    def _check_circle_similarity(edge1: Part.Edge, edge2: Part.Edge) -> bool:
        """Check similarity for circular edges (same plane, radius, and center point with angular overlap)."""
        try:
            # Check if centers are the same
            center1 = edge1.Curve.Center
            center2 = edge2.Curve.Center
            if center1.distanceToPoint(center2) > GeometryMatcher.TOLERANCE:
                return False

            # Check if radii are effectively the same
            radius1 = edge1.Curve.Radius
            radius2 = edge2.Curve.Radius
            if abs(radius1 - radius2) > GeometryMatcher.TOLERANCE:
                return False

            # Check if their planes are parallel (normals are aligned)
            if hasattr(edge1.Curve, 'Axis') and hasattr(edge2.Curve, 'Axis'):
                axis1 = edge1.Curve.Axis
                axis2 = edge2.Curve.Axis
                if abs(axis1.dot(axis2)) < (1 - GeometryMatcher.TOLERANCE):
                    return False
            else:
                return False

            # If they are full circles with same center, radius, and plane, they are similar
            is_full_circle1 = abs(edge1.Curve.LastParameter - edge1.Curve.FirstParameter - 2 * math.pi) < GeometryMatcher.TOLERANCE
            is_full_circle2 = abs(edge2.Curve.LastParameter - edge2.Curve.FirstParameter - 2 * math.pi) < GeometryMatcher.TOLERANCE

            if is_full_circle1 and is_full_circle2:
                return True

            # If both are arcs, check for angular overlap
            if hasattr(edge1.Curve, 'FirstParameter') and hasattr(edge2.Curve, 'FirstParameter'):
                angle1_start = edge1.Curve.FirstParameter
                angle1_end = edge1.Curve.LastParameter
                angle2_start = edge2.Curve.FirstParameter
                angle2_end = edge2.Curve.LastParameter

                # Normalize angles to handle wraparound (e.g., arc from 300 to 30 degrees).
                while angle1_end < angle1_start:
                    angle1_end += 2 * math.pi
                while angle2_end < angle2_start:
                    angle2_end += 2 * math.pi

                overlap_start = max(angle1_start, angle2_start)
                overlap_end = min(angle1_end, angle2_end)

                return (overlap_end - overlap_start) > GeometryMatcher.TOLERANCE

            return False

        except Exception:
            return False

    @staticmethod
    def vertices_exact_match(vertex1: Part.Vertex, vertex2: Part.Vertex) -> bool:
        """Check if two vertices are exact matches."""
        try:
            return vertex1.Point.distanceToPoint(vertex2.Point) < GeometryMatcher.TOLERANCE
        except Exception:
            return False

class FeatureAnalyzer:
    """Analyzes features in a body to extract geometric elements."""

    @staticmethod
    def get_body_features(body: Any) -> List[Any]:
        """Get all features in a body ordered by creation sequence."""
        features = []
        if hasattr(body, 'Group'):
            for obj in body.Group:
                if hasattr(obj, 'Shape') and obj.Shape:
                    features.append(obj)
        return features

    @staticmethod
    def extract_faces(feature: Any) -> List[Tuple[Part.Face, str]]:
        """Extract all faces from a feature with their names."""
        faces = []
        if hasattr(feature, 'Shape') and feature.Shape:
            for i, face in enumerate(feature.Shape.Faces):
                face_name = f"Face{i+1}"
                faces.append((face, face_name))
        return faces

    @staticmethod
    def extract_edges(feature: Any) -> List[Tuple[Part.Edge, str]]:
        """Extract all edges from a feature with their names."""
        edges = []
        if hasattr(feature, 'Shape') and feature.Shape:
            for i, edge in enumerate(feature.Shape.Edges):
                edge_name = f"Edge{i+1}"
                edges.append((edge, edge_name))
        return edges

    @staticmethod
    def extract_vertices(feature: Any) -> List[Tuple[Part.Vertex, str]]:
        """Extract all vertices from a feature with their names."""
        vertices = []
        if hasattr(feature, 'Shape') and feature.Shape:
            for i, vertex in enumerate(feature.Shape.Vertexes):
                vertex_name = f"Vertex{i+1}"
                vertices.append((vertex, vertex_name))
        return vertices


class SelectionTracker(QtCore.QObject):
    """Tracks selection changes in the 3D view."""

    selection_changed = QtCore.Signal(object)

    def __init__(self):
        super().__init__()
        self.current_selection = None
        self.selection_observer = None

    def start_tracking(self):
        """Start tracking selection changes."""
        self.selection_observer = SelectionObserver(self)
        Gui.Selection.addObserver(self.selection_observer)

    def stop_tracking(self):
        """Stop tracking selection changes."""
        if self.selection_observer:
            Gui.Selection.removeObserver(self.selection_observer)
            self.selection_observer = None

    def update_selection(self, selection_info):
        """Update current selection and emit signal."""
        self.current_selection = selection_info
        self.selection_changed.emit(selection_info)


class SelectionObserver:
    """Observer for FreeCAD selection changes."""

    def __init__(self, tracker: SelectionTracker):
        self.tracker = tracker

    def addSelection(self, doc_name, obj_name, sub_name, pos):
        """Called when selection is added."""
        self.update_selection(doc_name, obj_name, sub_name)

    def removeSelection(self, doc_name, obj_name, sub_name):
        """Called when selection is removed."""
        self.update_selection(doc_name, obj_name, sub_name)

    def setSelection(self, doc_name, obj_name, sub_name):
        """Called when selection is set."""
        self.update_selection(doc_name, obj_name, sub_name)

    def clearSelection(self, doc_name):
        """Called when selection is cleared."""
        self.tracker.update_selection(None)

    def update_selection(self, doc_name, obj_name, sub_name):
        """Process selection update."""
        try:
            if not sub_name:
                self.tracker.update_selection(None)
                return

            doc = App.getDocument(doc_name)
            obj = doc.getObject(obj_name)

            if not obj or not hasattr(obj, 'Shape'):
                self.tracker.update_selection(None)
                return

            # Determine selection type
            selection_type = None
            if sub_name.startswith('Face'):
                selection_type = 'Face'
            elif sub_name.startswith('Edge'):
                selection_type = 'Edge'
            elif sub_name.startswith('Vertex'):
                selection_type = 'Vertex'

            if selection_type:
                selection_info = {
                    'object': obj,
                    'sub_name': sub_name,
                    'type': selection_type,
                    'shape': getattr(obj.Shape, sub_name)
                }
                self.tracker.update_selection(selection_info)
            else:
                self.tracker.update_selection(None)

        except Exception:
            self.tracker.update_selection(None)


class TopoMatchSelectorWidget(QtGui.QWidget):
    """Main widget for the TopoMatchSelector docker."""

    def __init__(self):
        super().__init__()
        self.selection_tracker = SelectionTracker()
        self.current_body = None
        self.current_selection = None
        self.setup_ui()
        self.connect_signals()
        self.selection_tracker.start_tracking()

    def setup_ui(self):
        """Setup the user interface."""
        layout = QtGui.QVBoxLayout(self)

        # Selection info
        self.selection_label = QtGui.QLabel("No selection")
        self.selection_label.setWordWrap(True)
        layout.addWidget(self.selection_label)

        # Exact matches section
        exact_label = QtGui.QLabel("Exact Matches:")
        exact_label.setFont(QtGui.QFont("", -1, QtGui.QFont.Bold))
        layout.addWidget(exact_label)

        self.exact_list = QtGui.QListWidget()
        self.exact_list.setMaximumHeight(150)
        layout.addWidget(self.exact_list)

        # Similar matches section
        similar_label = QtGui.QLabel("Similar Matches:")
        similar_label.setFont(QtGui.QFont("", -1, QtGui.QFont.Bold))
        layout.addWidget(similar_label)

        self.similar_list = QtGui.QListWidget()
        self.similar_list.setMaximumHeight(150)
        layout.addWidget(self.similar_list)

        # Status / Log messages - now a QTextBrowser for selectable text
        self.status_browser = QtGui.QTextBrowser()
        self.status_browser.setReadOnly(True)
        self.status_browser.setAcceptRichText(False) # For plain text
        self.status_browser.setMinimumHeight(50) # Give it some space
        layout.addWidget(self.status_browser)

        layout.addStretch()

    def connect_signals(self):
        """Connect UI signals."""
        self.selection_tracker.selection_changed.connect(self.on_selection_changed)
        self.exact_list.itemClicked.connect(self.on_exact_item_clicked)
        self.similar_list.itemClicked.connect(self.on_similar_item_clicked)

    def on_selection_changed(self, selection_info):
        """Handle selection changes."""
        self.current_selection = selection_info
        self.update_display()

    def on_exact_item_clicked(self, item):
        """Handle exact match item selection."""
        self.select_item_geometry(item)

    def on_similar_item_clicked(self, item):
        """Handle similar match item selection."""
        self.select_item_geometry(item)

    def select_item_geometry(self, item):
        """Select the geometry represented by the list item."""
        try:
            item_data = item.data(QtCore.Qt.UserRole)
            if item_data:
                obj_name = item_data['object']
                sub_name = item_data['sub_name']

                # Get the document name
                doc_name = App.ActiveDocument.Name

                # Clear current selection and select new item
                Gui.Selection.clearSelection()
                Gui.Selection.addSelection(doc_name, obj_name, sub_name)

        except Exception as e:
            self.status_browser.setText(f"Error selecting item: {str(e)}")

    def update_display(self):
        """Update the display based on current selection."""
        self.exact_list.clear()
        self.similar_list.clear()
        self.status_browser.clear() # Clear status messages for new selection

        if not self.current_selection:
            self.selection_label.setText("No selection")
            self.status_browser.setText("")
            return

        # Get current body
        self.current_body = self.find_parent_body(self.current_selection['object'])
        if not self.current_body:
            self.selection_label.setText("Selection not in a body")
            self.status_browser.setText("")
            return

        # Update selection info
        obj_name = self.current_selection['object'].Label
        sub_name = self.current_selection['sub_name']
        selection_type = self.current_selection['type']
        self.selection_label.setText(f"Selected: {obj_name}.{sub_name} ({selection_type})")

        # Find matches
        self.find_and_display_matches()

    def find_parent_body(self, obj):
        """Find the parent body of an object."""
        try:
            # Check if object is directly in a body
            for body in App.ActiveDocument.Objects:
                if hasattr(body, 'TypeId') and body.TypeId == 'PartDesign::Body':
                    if hasattr(body, 'Group') and obj in body.Group:
                        return body
            return None
        except Exception:
            return None

    def find_and_display_matches(self):
        """Find and display matching geometry."""
        if not self.current_body or not self.current_selection:
            return

        try:
            # Get all features in the body
            features = FeatureAnalyzer.get_body_features(self.current_body)
            current_obj = self.current_selection['object']

            # Find features that come before the current one
            try:
                current_index = features.index(current_obj)
                earlier_features = features[:current_index]
            except ValueError:
                earlier_features = features

            selection_type = self.current_selection['type']
            current_shape = self.current_selection['shape']

            exact_matches = []
            similar_matches = []

            # Search through earlier features
            for feature in earlier_features:
                if selection_type == 'Face':
                    matches = self.find_face_matches(feature, current_shape)
                elif selection_type == 'Edge':
                    matches = self.find_edge_matches(feature, current_shape)
                elif selection_type == 'Vertex':
                    matches = self.find_vertex_matches(feature, current_shape)
                else:
                    continue

                exact_matches.extend(matches['exact'])
                similar_matches.extend(matches['similar'])

            # Populate lists
            self.populate_match_list(self.exact_list, exact_matches)
            self.populate_match_list(self.similar_list, similar_matches)

            # Update status
            exact_count = len(exact_matches)
            similar_count = len(similar_matches)
            self.status_browser.setText(f"Found {exact_count} exact, {similar_count} similar matches")

        except Exception as e:
            self.status_browser.setText(f"Error finding matches: {str(e)}")

    def find_face_matches(self, feature, current_face):
        """Find face matches in a feature."""
        exact_matches = []
        similar_matches = []

        faces = FeatureAnalyzer.extract_faces(feature)
        for face, face_name in faces:
            if GeometryMatcher.faces_exact_match(current_face, face):
                exact_matches.append({
                    'feature': feature,
                    'name': face_name,
                    'shape': face
                })
            elif GeometryMatcher.faces_similar_match(current_face, face):
                similar_matches.append({
                    'feature': feature,
                    'name': face_name,
                    'shape': face
                })

        return {'exact': exact_matches, 'similar': similar_matches}

    def find_edge_matches(self, feature, current_edge):
        """Find edge matches in a feature."""
        exact_matches = []
        similar_matches = []

        edges = FeatureAnalyzer.extract_edges(feature)
        for edge, edge_name in edges:
            if GeometryMatcher.edges_exact_match(current_edge, edge):
                exact_matches.append({
                    'feature': feature,
                    'name': edge_name,
                    'shape': edge
                })
            elif GeometryMatcher.edges_similar_match(current_edge, edge):
                similar_matches.append({
                    'feature': feature,
                    'name': edge_name,
                    'shape': edge
                })

        return {'exact': exact_matches, 'similar': similar_matches}

    def find_vertex_matches(self, feature, current_vertex):
        """Find vertex matches in a feature."""
        exact_matches = []
        similar_matches = []

        vertices = FeatureAnalyzer.extract_vertices(feature)
        for vertex, vertex_name in vertices:
            if GeometryMatcher.vertices_exact_match(current_vertex, vertex):
                exact_matches.append({
                    'feature': feature,
                    'name': vertex_name,
                    'shape': vertex
                })

        return {'exact': exact_matches, 'similar': similar_matches}

    def populate_match_list(self, list_widget, matches):
        """Populate a list widget with matches."""
        for match in matches:
            feature_label = match['feature'].Label
            item_name = match['name']
            item_text = f"{feature_label}.{item_name}"

            item = QtGui.QListWidgetItem(item_text)
            item.setData(QtCore.Qt.UserRole, {
                'object': match['feature'].Name,
                'sub_name': item_name
            })
            list_widget.addItem(item)

    def closeEvent(self, event):
        """Handle widget close event."""
        self.selection_tracker.stop_tracking()
        event.accept()


class TopoMatchSelectorDockWidget(QtGui.QDockWidget):
    """Docker widget for the TopoMatchSelector macro."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("TopoMatchSelector")
        self.setObjectName("TopoMatchSelector")

        # Create main widget
        self.main_widget = TopoMatchSelectorWidget()
        self.setWidget(self.main_widget)

        # Set dock properties
        self.setFeatures(QtGui.QDockWidget.DockWidgetMovable |
                        QtGui.QDockWidget.DockWidgetFloatable |
                        QtGui.QDockWidget.DockWidgetClosable)
        self.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea |
                            QtCore.Qt.RightDockWidgetArea)

    def closeEvent(self, event):
        """Handle dock close event."""
        self.main_widget.closeEvent(event)


def create_topo_match_selector():
    """Create and show the TopoMatchSelector docker."""
    # Remove existing docker if it exists
    main_window = Gui.getMainWindow()
    existing_dock = main_window.findChild(QtGui.QDockWidget, "TopoMatchSelector")
    if existing_dock:
        existing_dock.close()
        existing_dock.deleteLater()

    # Create new docker
    dock = TopoMatchSelectorDockWidget()
    main_window.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
    dock.show()

    return dock


# Main execution
if __name__ == "__main__":
    try:
        # Start transaction
        App.ActiveDocument.openTransaction("TopoMatchSelector")

        # Create the docker
        topo_match_selector_dock = create_topo_match_selector()

        # Commit transaction
        App.ActiveDocument.commitTransaction()

        print("TopoMatchSelector v1.0 docker created successfully!")

    except Exception as e:
        App.ActiveDocument.abortTransaction()
        print(f"Error creating TopoMatchSelector: {str(e)}")
        raise
