#!/usr/bin/env python3
"""
Point Cloud Plane Sketch - FreeCAD Macro

Workflow:
1. User selects 3+ vertices from point cloud to define approximate plane
2. Interactive docker shows tolerance adjustment with live preview
3. All points within tolerance are selected and highlighted
4. RANSAC fits best plane to selected points
5. User chooses destination (Standalone/New Body/Existing Body)
6. Creates datum plane and sketch with projected construction points

Author: Based on SketcherWireDoctor and CoplanarSketch patterns
Version: 1.1
"""

from __future__ import annotations

# Standard library
import math
import traceback

# Third-party
import numpy as np
from PySide import QtCore, QtGui
from PySide.QtWidgets import (QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, 
                               QPushButton, QLabel, QDoubleSpinBox, QInputDialog, 
                               QMessageBox, QTextEdit)

# FreeCAD
import FreeCAD as App
import FreeCADGui as Gui
import Part
import Sketcher


# Configuration
class Config:
    """Configuration constants."""
    DEFAULT_TOLERANCE = 0.1  # mm
    MIN_TOLERANCE = 0.01
    MAX_TOLERANCE = 50.0
    TOLERANCE_STEP = 0.1
    TOLERANCE_DECIMALS = 3
    
    RANSAC_ITERATIONS = 200
    
    HIGHLIGHT_POINT_SIZE = 8.0
    HIGHLIGHT_TRANSPARENCY = 30
    MAX_HIGHLIGHT_POINTS = 10000  # Skip visualization above this threshold for performance
    HIGHLIGHT_BATCH_SIZE = 5000   # Batch size for creating highlight objects
    
    # Default highlight color index (Yellow)
    DEFAULT_HIGHLIGHT_COLOR_INDEX = 3
    
    # Default highlight colors (R, G, B)
    HIGHLIGHT_COLORS = [
        (1.0, 0.0, 0.0),    # Red
        (0.0, 1.0, 0.0),    # Green
        (0.0, 0.0, 1.0),    # Blue
        (1.0, 1.0, 0.0),    # Yellow
        (1.0, 0.0, 1.0),    # Magenta
        (0.0, 1.0, 1.0),    # Cyan
        (1.0, 0.5, 0.0),    # Orange
        (1.0, 1.0, 1.0),    # White
    ]
    
    COLOR_BUTTON_SIZE = (25, 25, 35, 35)  # min_w, min_h, max_w, max_h

class PointCloudAnalyzer:
    """Analyze point clouds and fit planes with viewer-facing orientation."""

    @staticmethod
    def get_all_points_from_object(obj) -> list[App.Vector]:
        """Extract all vertex points from a FreeCAD object."""
        if not hasattr(obj, "Shape"):
            return []
        return [v.Point for v in obj.Shape.Vertexes]

    @staticmethod
    def orient_normal_toward_viewer(normal: App.Vector, plane_point: App.Vector) -> App.Vector:
        """Ensure the plane normal points toward the viewer (out of the screen)."""
        if normal.Length < 1e-9:
            return App.Vector(0, 0, 1)

        normal = normal.normalize()
        
        # Get view direction (points INTO the screen, away from camera)
        try:
            view_dir = Gui.ActiveDocument.ActiveView.getViewDirection()
        except Exception:
            view_dir = App.Vector(0, 0, 1)
        
        # view_dir points INTO screen, we want normal to point OUT (opposite direction)
        # If dot product < 0, they already point opposite ways - keep normal as is
        # If dot product > 0, they point same way - flip normal
        if normal.dot(view_dir) > 0:
            normal = -normal
            
        return normal

    @staticmethod
    def calculate_plane_from_3_points(p1: App.Vector, p2: App.Vector, p3: App.Vector) -> tuple[App.Vector, App.Vector]:
        """Calculate a plane from three points."""
        v1 = p2 - p1
        v2 = p3 - p1
        normal = v1.cross(v2)
        if normal.Length < 1e-9:
            raise ValueError("Selected points are collinear - cannot define a plane")
        return normal.normalize(), p1

    @staticmethod
    def fit_plane_least_squares(points_np: np.ndarray) -> tuple[App.Vector, App.Vector]:
        """
        Fit a plane through all points using least-squares (PCA).
        Every selected point contributes equally.
        """
        if len(points_np) < 3:
            raise ValueError("Need at least 3 points to fit a plane")

        centroid = np.mean(points_np, axis=0)
        centered = points_np - centroid
        _, _, vh = np.linalg.svd(centered)
        normal = vh[2]

        normal_fc = App.Vector(*normal)
        centroid_fc = App.Vector(*centroid)
        # Don't orient here - let caller decide when to orient based on camera

        return normal_fc, centroid_fc

    @staticmethod
    def fit_plane_ransac(points_np: np.ndarray, threshold: float, iterations: int = 20) -> tuple[App.Vector, App.Vector, list[int]]:
        """
        Fit a plane to noisy points using RANSAC, oriented toward the viewer.

        Args:
            points_np: Numpy array of points (Nx3).
            threshold: Max distance for inliers (mm).
            iterations: Number of RANSAC iterations.

        Returns:
            (normal_vector, point_on_plane, inlier_indices)
        """
        n = len(points_np)
        if n < 3:
            raise ValueError("Need at least 3 points to fit a plane")

        best_inliers: list[int] = []
        best_plane = None

        for _ in range(iterations):
            sample_idx = np.random.choice(n, 3, replace=False)
            sample = points_np[sample_idx]

            v1 = sample[1] - sample[0]
            v2 = sample[2] - sample[0]
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm < 1e-6:
                continue
            normal = normal / norm
            d = -np.dot(normal, sample[0])

            distances = np.abs(np.dot(points_np, normal) + d)
            inliers = np.where(distances < threshold)[0]

            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_plane = (normal, sample[0], d)

        if best_plane is None:
            raise ValueError("Could not fit plane - try increasing threshold")

        inlier_points = points_np[best_inliers]
        centroid = np.mean(inlier_points, axis=0)
        centered = inlier_points - centroid
        _, _, vh = np.linalg.svd(centered)
        normal = vh[2]

        normal_fc = App.Vector(*normal).normalize()
        centroid_fc = App.Vector(*centroid)
        # Don't orient here - let caller decide when to orient based on camera

        return normal_fc, centroid_fc, best_inliers.tolist()

    @staticmethod
    def select_points_within_tolerance(points_np: np.ndarray, normal: App.Vector, plane_point: App.Vector, tolerance: float) -> list[int]:
        """
        Select all points within tolerance of a plane.

        Args:
            points_np: Numpy array of points (Nx3).
            normal: Plane normal vector.
            plane_point: A point on the plane.
            tolerance: Distance threshold.

        Returns:
            Indices of points within tolerance.
        """
        plane_point_array = np.array([plane_point.x, plane_point.y, plane_point.z])
        normal_array = np.array([normal.x, normal.y, normal.z])
        distances = np.abs(np.dot(points_np - plane_point_array, normal_array))
        return np.where(distances <= tolerance)[0].tolist()

class PointHighlighter:
    """Handles point highlighting in the 3D view."""
    
    def __init__(self):
        self.highlight_objects = []
        self.profile_highlight_objects = []  # Separate list for profile points
        self.normal_arrow_object = None
        self.current_color = Config.HIGHLIGHT_COLORS[Config.DEFAULT_HIGHLIGHT_COLOR_INDEX]
    
    def set_color(self, color):
        """Set the highlight color and update existing highlights."""
        self.current_color = color
        self._update_existing_highlights()
    
    def _update_existing_highlights(self):
        """Update color of existing highlight objects."""
        for highlight_obj in self.highlight_objects:
            try:
                view_obj = highlight_obj.ViewObject
                view_obj.PointColor = self.current_color
            except Exception:
                pass
        Gui.updateGui()
    
    def highlight_points(self, points, skip_if_too_many=True):
        """
        Highlight multiple points efficiently.
        For large point sets, creates multiple batched highlight objects.
        
        Args:
            points: List of App.Vector points to highlight
            skip_if_too_many: If True, skip visualization if points exceed MAX_HIGHLIGHT_POINTS
        
        Returns:
            True if highlighted, False if skipped
        """
        import time
        
        self.clear_highlights()
        
        doc = App.ActiveDocument
        if not doc:
            return False
        
        if not points:
            return False
        
        num_points = len(points)
        
        # Skip visualization for very large datasets
        if skip_if_too_many and num_points > Config.MAX_HIGHLIGHT_POINTS:
            print(f"Skipping visualization of {num_points} points (exceeds {Config.MAX_HIGHLIGHT_POINTS} limit)")
            return False
        
        # === TIMING INSTRUMENTATION ===
        timing_results = []
        timing_results.append(f"\n=== HIGHLIGHT TIMING ({num_points} points) ===")
        t_start = time.time()
        
        try:
            batch_size = Config.HIGHLIGHT_BATCH_SIZE
            
            # If we have a large number of points, batch them
            if num_points > batch_size:
                # Create multiple batched objects for better performance
                num_batches = (num_points + batch_size - 1) // batch_size
                
                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = min(start_idx + batch_size, num_points)
                    batch_points = points[start_idx:end_idx]
                    
                    t1 = time.time()
                    vertices = [Part.Vertex(p) for p in batch_points]
                    t2 = time.time()
                    compound = Part.Compound(vertices)
                    t3 = time.time()
                    
                    highlight_obj = doc.addObject("Part::Feature", f"PointHighlight_{i}")
                    t4 = time.time()
                    highlight_obj.Shape = compound
                    t5 = time.time()
                    
                    view_obj = highlight_obj.ViewObject
                    view_obj.PointColor = self.current_color
                    view_obj.PointSize = Config.HIGHLIGHT_POINT_SIZE
                    view_obj.Transparency = Config.HIGHLIGHT_TRANSPARENCY
                    t6 = time.time()
                    
                    self.highlight_objects.append(highlight_obj)
                    
                    timing_results.append(f"Batch {i+1}/{num_batches} ({len(batch_points)} pts): "
                                        f"vertices={t2-t1:.3f}s, compound={t3-t2:.3f}s, "
                                        f"addObj={t4-t3:.3f}s, setShape={t5-t4:.3f}s, viewObj={t6-t5:.3f}s")
            else:
                # Single object for smaller point sets
                t1 = time.time()
                vertices = [Part.Vertex(p) for p in points]
                t2 = time.time()
                
                compound = Part.Compound(vertices)
                t3 = time.time()
                
                highlight_obj = doc.addObject("Part::Feature", "PointHighlight")
                t4 = time.time()
                
                highlight_obj.Shape = compound
                t5 = time.time()
                
                view_obj = highlight_obj.ViewObject
                view_obj.PointColor = self.current_color
                view_obj.PointSize = Config.HIGHLIGHT_POINT_SIZE
                view_obj.Transparency = Config.HIGHLIGHT_TRANSPARENCY
                t6 = time.time()
                
                self.highlight_objects.append(highlight_obj)
                
                timing_results.append(f"1. Create vertices: {t2-t1:.3f}s")
                timing_results.append(f"2. Create compound: {t3-t2:.3f}s")
                timing_results.append(f"3. Add to document: {t4-t3:.3f}s")
                timing_results.append(f"4. Assign shape: {t5-t4:.3f}s")
                timing_results.append(f"5. Setup appearance: {t6-t5:.3f}s")
            
            # Single recompute and GUI update at the end
            t_recompute = time.time()
            doc.recompute()
            t_gui = time.time()
            Gui.updateGui()
            t_end = time.time()
            
            timing_results.append(f"6. Recompute: {t_gui-t_recompute:.3f}s")
            timing_results.append(f"7. Update GUI: {t_end-t_gui:.3f}s")
            timing_results.append(f"\n*** TOTAL TIME: {t_end-t_start:.3f}s ***\n")
            
            # Print timing results
            print("\n".join(timing_results))
            
            return True
            
        except Exception as e:
            print(f"Highlighting error: {e}")
            traceback.print_exc()
            return False
    
    def clear_highlights(self, clear_profile=True):
        """Clear highlight objects. Optionally preserve profile highlights."""
        doc = App.ActiveDocument
        if not doc:
            return
        
        for highlight_obj in self.highlight_objects[:]:
            try:
                if highlight_obj in doc.Objects:
                    doc.removeObject(highlight_obj.Name)
            except Exception:
                pass
        
        self.highlight_objects.clear()
        
        # Clear profile highlights only if requested
        if clear_profile:
            for highlight_obj in self.profile_highlight_objects[:]:
                try:
                    if highlight_obj in doc.Objects:
                        doc.removeObject(highlight_obj.Name)
                except Exception:
                    pass
            
            self.profile_highlight_objects.clear()
        
        # Clear normal arrow if it exists
        if self.normal_arrow_object:
            try:
                if self.normal_arrow_object in doc.Objects:
                    doc.removeObject(self.normal_arrow_object.Name)
            except Exception:
                pass
            self.normal_arrow_object = None
    
    def show_normal_arrow(self, origin: App.Vector, normal: App.Vector, length: float = 50.0):
        """
        Show an arrow indicating the plane normal direction.
        
        Args:
            origin: Starting point of the arrow (plane origin)
            normal: Normal vector direction
            length: Length of the arrow in mm
        """
        doc = App.ActiveDocument
        if not doc:
            return
        
        # Clear existing arrow
        if self.normal_arrow_object:
            try:
                if self.normal_arrow_object in doc.Objects:
                    doc.removeObject(self.normal_arrow_object.Name)
            except Exception:
                pass
        
        try:
            # Create arrow as a line
            # Don't use .normalize() or .multiply() as they modify in place!
            normal_unit = App.Vector(normal.x, normal.y, normal.z).normalize()
            end_point = origin + (normal_unit * length)
            
            line = Part.LineSegment(origin, end_point)
            
            self.normal_arrow_object = doc.addObject("Part::Feature", "PlaneNormalArrow")
            self.normal_arrow_object.Shape = line.toShape()
            
            # Style the arrow
            view_obj = self.normal_arrow_object.ViewObject
            view_obj.LineColor = self.current_color
            view_obj.LineWidth = 5.0
            view_obj.PointSize = 10.0
            
            doc.recompute()
            Gui.updateGui()
            
        except Exception as e:
            print(f"Error creating normal arrow: {e}")
            traceback.print_exc()
    
    def highlight_profile_points(self, points, color, skip_if_too_many=True):
        """
        Highlight profile plane points in a specific color.
        Similar to highlight_points but uses profile_highlight_objects list.
        """
        import time
        
        # Clear existing profile highlights
        doc = App.ActiveDocument
        if not doc:
            return False
        
        for highlight_obj in self.profile_highlight_objects[:]:
            try:
                if highlight_obj in doc.Objects:
                    doc.removeObject(highlight_obj.Name)
            except Exception:
                pass
        self.profile_highlight_objects.clear()
        
        if not points:
            return False
        
        num_points = len(points)
        
        # Skip visualization for very large datasets
        if skip_if_too_many and num_points > Config.MAX_HIGHLIGHT_POINTS:
            print(f"Skipping profile visualization of {num_points} points")
            return False
        
        try:
            batch_size = Config.HIGHLIGHT_BATCH_SIZE
            
            if num_points > batch_size:
                # Batched creation for large point sets
                num_batches = (num_points + batch_size - 1) // batch_size
                
                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = min(start_idx + batch_size, num_points)
                    batch_points = points[start_idx:end_idx]
                    
                    vertices = [Part.Vertex(p) for p in batch_points]
                    compound = Part.Compound(vertices)
                    
                    highlight_obj = doc.addObject("Part::Feature", f"ProfileHighlight_{i}")
                    highlight_obj.Shape = compound
                    
                    view_obj = highlight_obj.ViewObject
                    view_obj.PointColor = color
                    view_obj.PointSize = Config.HIGHLIGHT_POINT_SIZE
                    view_obj.Transparency = Config.HIGHLIGHT_TRANSPARENCY
                    
                    self.profile_highlight_objects.append(highlight_obj)
            else:
                # Single object for smaller point sets
                vertices = [Part.Vertex(p) for p in points]
                compound = Part.Compound(vertices)
                
                highlight_obj = doc.addObject("Part::Feature", "ProfileHighlight")
                highlight_obj.Shape = compound
                
                view_obj = highlight_obj.ViewObject
                view_obj.PointColor = color
                view_obj.PointSize = Config.HIGHLIGHT_POINT_SIZE
                view_obj.Transparency = Config.HIGHLIGHT_TRANSPARENCY
                
                self.profile_highlight_objects.append(highlight_obj)
            
            doc.recompute()
            Gui.updateGui()
            
            return True
            
        except Exception as e:
            print(f"Profile highlighting error: {e}")
            traceback.print_exc()
            return False

class SketchCreator:
    """Create datum planes and sketches with viewer-facing orientation."""

    # ---------------------------
    # Orientation helpers
    # ---------------------------
    @staticmethod
    def create_placement_from_plane(normal: App.Vector, point: App.Vector) -> App.Placement:
        """
        Build a view-aligned basis on the plane:
        - Z = plane normal (facing viewer)
        - X = camera right projected into the plane (or camera up if right degenerates)
        - Y = Z × X (right-handed), flipped to align with projected camera up
        """
        # Normal is already correctly oriented and normalized - make a copy to avoid modifying input
        n = App.Vector(normal.x, normal.y, normal.z)

        try:
            right_vec, up_vec, fwd_vec = Gui.ActiveDocument.ActiveView.getCameraOrientation()
            cam_right = App.Vector(*right_vec)
            cam_up    = App.Vector(*up_vec)
            cam_fwd   = App.Vector(*fwd_vec)
        except Exception:
            cam_fwd   = App.Vector(0, 0, -1)
            cam_up    = App.Vector(0, 1, 0)
            cam_right = cam_up.cross(cam_fwd)

        if cam_right.Length > 1e-9: cam_right = cam_right.normalize()
        if cam_up.Length    > 1e-9: cam_up    = cam_up.normalize()
        if cam_fwd.Length   > 1e-9: cam_fwd   = cam_fwd.normalize()

        r_proj = cam_right - (n * cam_right.dot(n))
        u_proj = cam_up - (n * cam_up.dot(n))

        r_len = r_proj.Length
        u_len = u_proj.Length

        if r_len >= 1e-6:
            x_axis = r_proj.normalize()
        elif u_len >= 1e-6:
            x_axis = u_proj.normalize()
        else:
            fallback = App.Vector(1, 0, 0) if abs(n.dot(App.Vector(1, 0, 0))) < 0.95 else App.Vector(0, 1, 0)
            x_proj = fallback - (n * fallback.dot(n))
            x_axis = x_proj.normalize()

        y_axis = n.cross(x_axis)
        if y_axis.Length < 1e-9:
            alt = App.Vector(0, 1, 0) if abs(n.dot(App.Vector(0, 1, 0))) < 0.95 else App.Vector(1, 0, 0)
            x_axis = (alt - (n * alt.dot(n))).normalize()
            y_axis = n.cross(x_axis)
        y_axis = y_axis.normalize()

        # Align Y-axis with camera up if possible
        if u_len >= 1e-6:
            u_dir = u_proj.normalize()
            if y_axis.dot(u_dir) < 0:
                # Y is backwards - flip X to reverse Y (don't flip both, that inverts the normal!)
                x_axis = -x_axis
                y_axis = n.cross(x_axis).normalize()

        # Create rotation from basis vectors using matrix
        # FreeCAD expects column vectors in the matrix
        matrix = App.Matrix(
            x_axis.x, y_axis.x, n.x, 0,
            x_axis.y, y_axis.y, n.y, 0,
            x_axis.z, y_axis.z, n.z, 0,
            0, 0, 0, 1
        )
        rot = App.Placement(matrix).Rotation
        
        return App.Placement(point, rot)

    # ---------------------------
    # Destination dialog
    # ---------------------------
    @staticmethod
    def show_destination_dialog():
        """Show destination dialog and return choice info."""
        doc = App.ActiveDocument
        body_names = [o.Name for o in doc.Objects if o.isDerivedFrom("PartDesign::Body")]
        options = ["<Standalone (Part Workbench)>", "<Create New Body (PartDesign)>"] + body_names

        item, ok = QtGui.QInputDialog.getItem(
            Gui.getMainWindow(),
            "Sketch Destination",
            "Choose where to create the datum plane and sketch:",
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

    # ---------------------------
    # Body helpers
    # ---------------------------
    @staticmethod
    def _get_body_xy_plane(body) -> App.DocumentObject | None:
        """Return the body's XY origin plane feature, if available."""
        if not hasattr(body, 'Origin'):
            return None
        origin = body.Origin
        if not hasattr(origin, 'OriginFeatures'):
            return None
        
        # Look for XY_Plane in the origin features
        for feat in origin.OriginFeatures:
            # Check both Name and Label for XY_Plane
            if 'XY_Plane' in feat.Name or feat.Label == 'XY_Plane':
                return feat
        
        # Fallback: return first origin feature if it exists
        return origin.OriginFeatures[0] if origin.OriginFeatures else None

    # ---------------------------
    # Creation methods
    # ---------------------------
    @staticmethod
    def create_standalone(normal: App.Vector, plane_point: App.Vector, points: list[App.Vector]):
        """Create standalone datum plane and sketch (Part workbench)."""
        doc = App.ActiveDocument
        placement = SketchCreator.create_placement_from_plane(normal, plane_point)

        datum = doc.addObject("PartDesign::Plane", "DatumPlane")
        datum.Placement = placement
        doc.recompute()

        sketch = doc.addObject("Sketcher::SketchObject", "PointCloudSketch")
        sketch.Placement = placement
        doc.recompute()

        SketchCreator._add_construction_points(sketch, points, placement)
        doc.recompute()

        return datum, sketch

    @staticmethod
    def create_in_body(normal: App.Vector, plane_point: App.Vector, points: list[App.Vector], body=None):
        """Create datum plane and sketch in a PartDesign body."""
        doc = App.ActiveDocument
        if body is None:
            body = doc.addObject("PartDesign::Body", "Body")
            doc.recompute()
    
        placement = SketchCreator.create_placement_from_plane(normal, plane_point)
    
        # Create proper PartDesign datum plane inside the body
        datum = body.newObject("PartDesign::Plane", "DatumPlane")
        doc.recompute()
        
        # Attach datum plane to body's XY origin plane with attachment offset
        xy_plane = SketchCreator._get_body_xy_plane(body)
        if xy_plane:
            datum.AttachmentSupport = [(xy_plane, '')]
            datum.MapMode = 'FlatFace'
            datum.AttachmentOffset = placement
        else:
            # Fallback if no origin plane available
            datum.MapMode = 'Deactivated'
            datum.Placement = placement
        
        datum.recompute()
        doc.recompute()
    
        # Create proper PartDesign sketch inside the body  
        sketch = body.newObject("Sketcher::SketchObject", "PointCloudSketch")
        sketch.AttachmentSupport = [(datum, '')]
        sketch.MapMode = 'FlatFace'
        doc.recompute()
    
        SketchCreator._add_construction_points(sketch, points, placement)
        doc.recompute()
    
        return datum, sketch, body

    # ---------------------------
    # Geometry helpers
    # ---------------------------
    @staticmethod
    def _add_construction_points(sketch, points: list[App.Vector], placement: App.Placement) -> None:
        """Add points as construction geometry to a sketch."""
        # Transform points from global space to sketch's local coordinate system
        # The sketch is attached to the datum plane, so we need to use the placement
        # that was used to position the datum plane
        inverse_placement = placement.inverse()
        for point in points:
            local_point = inverse_placement.multVec(point)
            # Project onto XY plane (Z=0) in sketch coordinates
            geo_point = Part.Point(App.Vector(local_point.x, local_point.y, 0))
            sketch.addGeometry(geo_point, True)

class PointCloudPlaneWidget(QWidget):
    """Main widget for the Point Cloud Plane Sketch docker."""
    
    def __init__(self):
        super().__init__()
        
        self.all_points_np = None  # Numpy array of all points (Nx3)
        self.source_object = None
        self.source_object_visibility = None  # Store original visibility state
        self.initial_normal = None
        self.initial_plane_point = None
        
        self.selected_indices = []
        self.refined_normal = None
        self.refined_plane_point = None
        
        # Profile plane selection
        self.profile_indices = []
        self.profile_color_index = 4  # Default to Magenta (index 4) for profile points
        self.color_mode = "base"  # "base" or "profile" - which colors the swatches control
        
        self.highlighter = PointHighlighter()
        self.current_color_index = Config.DEFAULT_HIGHLIGHT_COLOR_INDEX
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        
        # Instructions
        instructions = QLabel(
            "1. Select 3+ vertices from point cloud\n"
            "2. Click 'Collect Vertex Data' (one time only)\n"
            "3. Adjust tolerance and preview\n"
            "4. Click 'Create Sketch' when ready\n"
            "5. Click 'New Selection' to create another sketch"
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Initialize button
        self.init_button = QPushButton("Collect Vertex Data")
        self.init_button.clicked.connect(self._on_collect_button_pressed)
        layout.addWidget(self.init_button)
        
        # New selection button (initially hidden)
        self.new_selection_button = QPushButton("New Selection (Select 3+ vertices)")
        self.new_selection_button.clicked.connect(self._on_new_selection)
        self.new_selection_button.setVisible(False)
        layout.addWidget(self.new_selection_button)
        
        # Tolerance section (initially hidden)
        self.tolerance_widget = QWidget()
        tolerance_layout = QVBoxLayout(self.tolerance_widget)
        tolerance_layout.setContentsMargins(0, 0, 0, 0)
        
        # Tolerance input
        tol_input_layout = QHBoxLayout()
        tol_input_layout.addWidget(QLabel("Tolerance (mm):"))
        
        self.tolerance_spin = QDoubleSpinBox()
        self.tolerance_spin.setRange(Config.MIN_TOLERANCE, Config.MAX_TOLERANCE)
        self.tolerance_spin.setValue(Config.DEFAULT_TOLERANCE)
        self.tolerance_spin.setDecimals(Config.TOLERANCE_DECIMALS)
        self.tolerance_spin.setSingleStep(Config.TOLERANCE_STEP)
        self.tolerance_spin.setMinimumWidth(100)
        tol_input_layout.addWidget(self.tolerance_spin)
        
        self.update_button = QPushButton("Update Preview")
        self.update_button.clicked.connect(self._update_preview)
        tol_input_layout.addWidget(self.update_button)
        
        tol_input_layout.addStretch()
        tolerance_layout.addLayout(tol_input_layout)
        
        # Point count display
        self.count_label = QLabel("Selected: 0 points")
        self.count_label.setStyleSheet("font-weight: bold; font-size: 11pt;")
        tolerance_layout.addWidget(self.count_label)
        
        # Color selection
        self._setup_color_selector(tolerance_layout)
        
        # Profile plane section
        profile_label = QLabel("Profile Plane (Optional):")
        profile_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        tolerance_layout.addWidget(profile_label)
        
        # Profile distance
        profile_dist_layout = QHBoxLayout()
        profile_dist_layout.addWidget(QLabel("Offset Distance (mm):"))
        self.profile_distance_edit = QtGui.QLineEdit()
        self.profile_distance_edit.setText("5.0")
        self.profile_distance_edit.setMinimumWidth(80)
        self.profile_distance_edit.setMaximumWidth(100)
        profile_dist_layout.addWidget(self.profile_distance_edit)
        profile_dist_layout.addStretch()
        tolerance_layout.addLayout(profile_dist_layout)
        
        # Profile tolerance
        profile_tol_layout = QHBoxLayout()
        profile_tol_layout.addWidget(QLabel("Tolerance (mm):"))
        self.profile_tolerance_edit = QtGui.QLineEdit()
        self.profile_tolerance_edit.setText("0.1")
        self.profile_tolerance_edit.setMinimumWidth(80)
        self.profile_tolerance_edit.setMaximumWidth(100)
        profile_tol_layout.addWidget(self.profile_tolerance_edit)
        profile_tol_layout.addStretch()
        tolerance_layout.addLayout(profile_tol_layout)
        
        # Add profile points button
        self.add_profile_button = QPushButton("Add Profile Plane Points")
        self.add_profile_button.clicked.connect(self._add_profile_points)
        self.add_profile_button.setEnabled(False)
        tolerance_layout.addWidget(self.add_profile_button)
        
        # Profile count display
        self.profile_count_label = QLabel("")
        self.profile_count_label.setStyleSheet("font-style: italic;")
        tolerance_layout.addWidget(self.profile_count_label)
        
        # Create sketch button
        self.create_button = QPushButton("Create Sketch")
        self.create_button.clicked.connect(self._create_sketch)
        self.create_button.setEnabled(False)
        tolerance_layout.addWidget(self.create_button)
        
        self.tolerance_widget.setVisible(False)
        layout.addWidget(self.tolerance_widget)
        
        # Info display
        self.info_display = QTextEdit()
        self.info_display.setReadOnly(True)
        self.info_display.setMaximumHeight(150)
        layout.addWidget(self.info_display)
        
        # Clear button
        clear_button = QPushButton("Clear Messages")
        clear_button.clicked.connect(self.info_display.clear)
        layout.addWidget(clear_button)
        
        layout.addStretch()
    
    def _setup_color_selector(self, layout):
        """Setup color selection widget."""
        color_box = QHBoxLayout()
        self.color_mode_label = QLabel("Highlight Color:")
        color_box.addWidget(self.color_mode_label)
        
        self.color_buttons = []
        
        for i, color in enumerate(Config.HIGHLIGHT_COLORS):
            button = self._create_color_button(color, i)
            self.color_buttons.append(button)
            color_box.addWidget(button)
        
        color_box.addStretch()
        layout.addLayout(color_box)
    
    def _create_color_button(self, color, index):
        """Create a single color button."""
        button = QtGui.QToolButton()
        min_w, min_h, max_w, max_h = Config.COLOR_BUTTON_SIZE
        button.setMinimumSize(min_w, min_h)
        button.setMaximumSize(max_w, max_h)
        
        self._update_color_button_style(button, color, index == self.current_color_index)
        button.clicked.connect(lambda checked=False, c=color, idx=index: self._set_highlight_color(c, idx))
        
        return button
    
    def _update_color_button_style(self, button, color, is_selected):
        """Update the style of a color button."""
        rgb = f"rgb({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)})"
        border = "3px solid white" if is_selected else "2px solid black"
        button.setStyleSheet(f"background-color: {rgb}; border: {border};")
    
    def _set_highlight_color(self, color, index):
        """Set the highlight color based on current mode (base or profile)."""
        if self.color_mode == "base":
            self.current_color_index = index
            self.highlighter.set_color(color)
            # Also update normal arrow color immediately if it exists
            if self.refined_normal and self.refined_plane_point:
                self.highlighter.show_normal_arrow(self.refined_plane_point, self.refined_normal, length=50.0)
            self.info_display.append(f"Changed base highlight color\n")
        else:  # profile mode
            self.profile_color_index = index
            # Re-highlight profile points if they exist
            if self.profile_indices:
                profile_points_np = self.all_points_np[self.profile_indices]
                profile_points_fc = [App.Vector(float(pt[0]), float(pt[1]), float(pt[2])) 
                                      for pt in profile_points_np]
                self.highlighter.highlight_profile_points(profile_points_fc, color, skip_if_too_many=True)
            self.info_display.append(f"Changed profile highlight color\n")
        
        # Update button styles
        for i, button in enumerate(self.color_buttons):
            button_color = Config.HIGHLIGHT_COLORS[i]
            self._update_color_button_style(button, button_color, i == index)
    
    def _on_new_selection(self):
        """Prepare for a new manual vertex selection without recollecting all points."""
        # Clear current highlights
        self.highlighter.clear_highlights()
    
        # Show the source object again so user can pick vertices
        if self.source_object and hasattr(self.source_object, 'ViewObject'):
            self.source_object.ViewObject.Visibility = True
            self.info_display.append(f"Showing {self.source_object.Label} (Name: {self.source_object.Name}) for new selection\n")
    
        # Reset only the selection state (keep all_points_np cached!)
        self.initial_normal = None
        self.initial_plane_point = None
        self.user_selected_vertices = []
        self.selected_indices = []
        self.refined_normal = None
        self.refined_plane_point = None
        self.profile_indices = []  # Clear profile points
    
        # Reset tolerance to default
        self.tolerance_spin.setValue(Config.DEFAULT_TOLERANCE)
    
        # Clear display
        self.count_label.setText("Selected: 0 points")
        self.profile_count_label.setText("")
        self.create_button.setEnabled(False)
        self.add_profile_button.setEnabled(False)
    
        # Reset color mode back to base
        self.color_mode = "base"
        self.color_mode_label.setText("Highlight Color:")
        # Update button selection to show current base color
        for i, button in enumerate(self.color_buttons):
            button_color = Config.HIGHLIGHT_COLORS[i]
            self._update_color_button_style(button, button_color, i == self.current_color_index)
    
        self.info_display.append("\n--- Ready for new selection ---")
        self.info_display.append(f"Select 3+ vertices from {self.source_object.Label}, then click 'New Selection' again\n")
    
        # Change button text to indicate we're waiting for selection
        self.new_selection_button.setText("Process New Selection")
        self.new_selection_button.clicked.disconnect()
        self.new_selection_button.clicked.connect(self._process_new_selection)
    
    def _process_new_selection(self):
        """Process the new vertex selection - optimized to skip full initialization."""
        # Validate and process new selection
        selection = Gui.Selection.getSelectionEx()
        if not selection:
            self.info_display.append("Error: No selection made. Please select 3+ vertices.\n")
            return
        
        # Check if selection is from the same source object
        selected_object = selection[0].Object
        if selected_object != self.source_object:
            self.info_display.append(
                f"Error: New selection must be from {self.source_object.Label} (Name: {self.source_object.Name}).\n"
                f"You selected from {selected_object.Label} (Name: {selected_object.Name}).\n"
                f"To work with a different object, close this docker and restart the macro.\n"
            )
            return
        
        # Collect selected vertices directly (optimized - cache Vertexes array)
        selected_vertices = []
        vertices_from_other_objects = 0
        
        for sel in selection:
            # Additional safety check - only process vertices from source object
            if sel.Object != self.source_object:
                # Count vertices from other objects for info message
                if hasattr(sel.Object, 'Shape'):
                    for sub_name in sel.SubElementNames:
                        if sub_name.startswith("Vertex"):
                            vertices_from_other_objects += 1
                continue
            if hasattr(sel.Object, 'Shape'):
                vertexes = sel.Object.Shape.Vertexes  # Cache once!
                sub_names = sel.SubElementNames  # Cache once!
                for sub_name in sub_names:
                    if sub_name.startswith("Vertex"):
                        vertex_idx = int(sub_name[6:]) - 1
                        if vertex_idx < len(vertexes):
                            selected_vertices.append(vertexes[vertex_idx].Point)
        
        # Inform user if vertices from other objects were ignored
        if vertices_from_other_objects > 0:
            self.info_display.append(
                f"Note: Ignored {vertices_from_other_objects} vertex/vertices from other objects. "
                f"Only using vertices from {self.source_object.Label}\n"
            )
        
        if len(selected_vertices) < 3:
            self.info_display.append(f"Error: Only {len(selected_vertices)} vertex/vertices selected. Please select at least 3 vertices.\n")
            return
        
        # Store user-selected vertices for plane origin calculation
        self.user_selected_vertices = selected_vertices
        
        # Hide the source object again (only our source object)
        if self.source_object and hasattr(self.source_object, 'ViewObject'):
            self.source_object.ViewObject.Visibility = False
        
        # Change button back to "New Selection"
        self.new_selection_button.setText("New Selection (Select 3+ vertices)")
        self.new_selection_button.clicked.disconnect()
        self.new_selection_button.clicked.connect(self._on_new_selection)
        
        try:
            # Calculate initial plane from selected vertices (fast - no full point cloud processing)
            if len(selected_vertices) == 3:
                self.initial_normal, self.initial_plane_point = PointCloudAnalyzer.calculate_plane_from_3_points(
                    selected_vertices[0], selected_vertices[1], selected_vertices[2]
                )
                self.info_display.append("Calculated initial plane from 3 selected vertices\n")
            else:
                selected_np = np.array([[p.x, p.y, p.z] for p in selected_vertices])
                self.initial_normal, self.initial_plane_point = PointCloudAnalyzer.fit_plane_least_squares(selected_np)
                self.info_display.append(
                    f"Calculated initial plane from {len(selected_vertices)} vertices using least-squares fit\n"
                )
            
            # Show tolerance controls and preview
            self.tolerance_widget.setVisible(True)
            self._update_preview()
            
        except Exception as e:
            self.info_display.append(f"Error: {str(e)}\n")
            traceback.print_exc()
    
    def _on_collect_button_pressed(self):
        """Handle collect button press - validate selection first."""
        # First validate selection before showing any messages
        selection = Gui.Selection.getSelectionEx()
        if not selection:
            self.info_display.append("Error: No selection made. Please select 3+ vertices.\n")
            return
        
        # Count selected vertices
        selected_vertex_count = 0
        for sel in selection:
            for sub_name in sel.SubElementNames:
                if sub_name.startswith("Vertex"):
                    selected_vertex_count += 1
        
        if selected_vertex_count < 3:
            self.info_display.append(f"Error: Only {selected_vertex_count} vertex/vertices selected. Please select at least 3 vertices.\n")
            return
        
        # Show immediate feedback
        self.info_display.append("Collecting vertex data, please wait...")
        QtGui.QApplication.processEvents()  # Force UI update
        
        # Use QTimer to defer actual collection so message can display
        QtCore.QTimer.singleShot(100, self._initialize_from_selection)
    
    def _initialize_from_selection(self):
        """Initialize plane from selected vertices."""
        import time
        
        timing_results = []
        t_start = time.time()
        
        selection = Gui.Selection.getSelectionEx()
        if not selection:
            self.info_display.append("Error: No selection made.\n")
            return
    
        # Collect selected vertices
        t1 = time.time()
        selected_vertices = []
        self.source_object = selection[0].Object
        vertices_from_other_objects = 0
        
        for sel in selection:
            # Only collect vertices from the source object
            if sel.Object != self.source_object:
                # Count vertices from other objects for info message
                if hasattr(sel.Object, 'Shape'):
                    for sub_name in sel.SubElementNames:
                        if sub_name.startswith("Vertex"):
                            vertices_from_other_objects += 1
                continue
            if hasattr(sel.Object, 'Shape'):
                vertexes = sel.Object.Shape.Vertexes  # Cache once!
                sub_names = sel.SubElementNames  # Cache once!
                for sub_name in sub_names:
                    if sub_name.startswith("Vertex"):
                        vertex_idx = int(sub_name[6:]) - 1
                        if vertex_idx < len(vertexes):
                            selected_vertices.append(vertexes[vertex_idx].Point)
        
        # Inform user if vertices from other objects were ignored
        if vertices_from_other_objects > 0:
            self.info_display.append(
                f"Note: Ignored {vertices_from_other_objects} vertex/vertices from other objects. "
                f"Only using vertices from {self.source_object.Label}\n"
            )
        
        # Store user-selected vertices for plane origin calculation
        self.user_selected_vertices = selected_vertices
        
        t2 = time.time()
        timing_results.append(f"1. Collect selected vertices: {t2-t1:.3f}s")
    
        if len(selected_vertices) < 3:
            self.info_display.append("Error: Please select at least 3 vertices.\n")
            return
    
        try:
            # Collect all points only once
            if self.all_points_np is None or len(self.all_points_np) == 0:
                t3 = time.time()
                # Optimized: pre-allocate numpy array and fill directly
                if hasattr(self.source_object, 'Shape'):
                    vertexes = self.source_object.Shape.Vertexes
                    num_vertices = len(vertexes)
                    # Pre-allocate array
                    self.all_points_np = np.empty((num_vertices, 3), dtype=np.float64)
                    # Fill array directly
                    for i, v in enumerate(vertexes):
                        p = v.Point
                        self.all_points_np[i] = [p.x, p.y, p.z]
                else:
                    self.info_display.append("Error: Object does not have a Shape.\n")
                    return
                t4 = time.time()
                timing_results.append(f"2. Get all points and convert to numpy: {t4-t3:.3f}s ({num_vertices} points)")
                
                if len(self.all_points_np) < 3:
                    self.info_display.append("Error: Object does not contain enough vertices.\n")
                    return
                
                if hasattr(self.source_object, 'ViewObject'):
                    self.source_object_visibility = self.source_object.ViewObject.Visibility
                    self.source_object.ViewObject.Visibility = False
                    self.info_display.append(f"Hid {self.source_object.Label} (Name: {self.source_object.Name}) to show highlights\n")
                self.init_button.setVisible(False)
                self.new_selection_button.setVisible(True)
    
            # Plane fitting logic
            t5 = time.time()
            if len(selected_vertices) == 3:
                self.initial_normal, self.initial_plane_point = PointCloudAnalyzer.calculate_plane_from_3_points(
                    selected_vertices[0], selected_vertices[1], selected_vertices[2]
                )
                self.info_display.append("Calculated initial plane from 3 selected vertices\n")
            else:
                selected_np = np.array([[p.x, p.y, p.z] for p in selected_vertices])
                self.initial_normal, self.initial_plane_point = PointCloudAnalyzer.fit_plane_least_squares(selected_np)
                self.info_display.append(
                    f"Calculated initial plane from {len(selected_vertices)} vertices using least-squares fit\n"
                )
            t6 = time.time()
            timing_results.append(f"3. Calculate initial plane: {t6-t5:.3f}s")
    
            # Show tolerance controls and preview
            self.tolerance_widget.setVisible(True)
            self._update_preview()
            
            # Print timing at the very end
            t_end = time.time()
            timing_results.append(f"\n=== INITIALIZE TOTAL: {t_end-t_start:.3f}s ===")
            print("\n=== INITIALIZATION TIMING ===")
            print("\n".join(timing_results))
    
        except Exception as e:
            self.info_display.append(f"Error: {str(e)}\n")
            traceback.print_exc()
    
    def _update_preview(self):
        """Update the preview with current tolerance."""
        import time
        
        if self.all_points_np is None or self.initial_normal is None:
            self.info_display.append("Error: Collect vertex data first.\n")
            return
        
        tolerance = self.tolerance_spin.value()
        
        # === TIMING INSTRUMENTATION ===
        timing_results = []
        t_start = time.time()
        
        try:
            # Select points within tolerance of initial plane
            t1 = time.time()
            self.selected_indices = PointCloudAnalyzer.select_points_within_tolerance(
                self.all_points_np,
                self.initial_normal,
                self.initial_plane_point,
                tolerance
            )
            t2 = time.time()
            timing_results.append(f"1. Select points within tolerance: {t2-t1:.3f}s")
            
            if len(self.selected_indices) < 3:
                self.count_label.setText(f"Selected: {len(self.selected_indices)} points (need at least 3)")
                self.create_button.setEnabled(False)
                self.highlighter.clear_highlights(clear_profile=False)  # Keep profile highlights
                self.info_display.append(f"Tolerance {tolerance}mm: only {len(self.selected_indices)} points - too few\n")
                return
            
            # Get selected points as numpy array
            selected_points_np = self.all_points_np[self.selected_indices]
            num_selected = len(selected_points_np)
            t3 = time.time()
            timing_results.append(f"2. Extract numpy subset: {t3-t2:.3f}s")
            
            # Update status for large datasets
            if num_selected > 1000:
                self.info_display.append(f"Processing {num_selected} points...\n")
                Gui.updateGui()  # Force UI update to show message
            
            # Refine plane using RANSAC on selected points
            self.refined_normal, ransac_centroid, inlier_indices = \
                PointCloudAnalyzer.fit_plane_ransac(
                    selected_points_np,
                    tolerance,
                    Config.RANSAC_ITERATIONS
                )
            
            # Re-orient normal based on CURRENT camera position (not initial camera position)
            # This ensures the normal always points toward the current view
            self.refined_normal = PointCloudAnalyzer.orient_normal_toward_viewer(
                self.refined_normal, 
                ransac_centroid
            )
            
            # Use centroid of user-selected vertices, but project it onto the fitted plane
            user_selected_np = np.array([[p.x, p.y, p.z] for p in self.user_selected_vertices])
            user_centroid_np = np.mean(user_selected_np, axis=0)
            user_centroid = App.Vector(user_centroid_np[0], user_centroid_np[1], user_centroid_np[2])
            
            # Project user centroid onto the fitted plane
            to_user_centroid = user_centroid - ransac_centroid
            distance_to_plane = to_user_centroid.dot(self.refined_normal)
            self.refined_plane_point = user_centroid - (self.refined_normal * distance_to_plane)
            
            t4 = time.time()
            timing_results.append(f"3. RANSAC plane fitting: {t4-t3:.3f}s")
            
            # Convert numpy points to FreeCAD vectors for highlighting
            # Skip visualization for very large datasets
            visualization_skipped = False
            if num_selected > Config.MAX_HIGHLIGHT_POINTS:
                self.info_display.append(
                    f"Skipping visualization ({num_selected} points exceeds {Config.MAX_HIGHLIGHT_POINTS} limit)\n"
                )
                visualization_skipped = True
                self.highlighter.clear_highlights(clear_profile=False)  # Keep profile highlights
            else:
                if num_selected > 1000:
                    self.info_display.append(f"Creating highlight visualization...\n")
                    Gui.updateGui()
                
                # More efficient conversion using direct indexing
                t5 = time.time()
                selected_points_fc = [App.Vector(float(pt[0]), float(pt[1]), float(pt[2])) 
                                      for pt in selected_points_np]
                t6 = time.time()
                timing_results.append(f"4. Convert numpy to FreeCAD Vectors: {t6-t5:.3f}s")
                
                # Highlight the selected points
                highlighted = self.highlighter.highlight_points(selected_points_fc, skip_if_too_many=True)
                if not highlighted:
                    visualization_skipped = True
                    self.info_display.append("Visualization skipped for performance\n")
            
            # Show normal arrow indicator (always, even if point visualization is skipped)
            self.highlighter.show_normal_arrow(self.refined_plane_point, self.refined_normal, length=50.0)
            
            # Update profile points if they were previously added
            if self.profile_indices:
                # Parse profile settings
                try:
                    distance = float(self.profile_distance_edit.text())
                    profile_tolerance = float(self.profile_tolerance_edit.text())
                    
                    if profile_tolerance > 0:  # Only tolerance must be positive
                        # Recalculate offset plane with current settings (distance can be negative)
                        offset_plane_point = self.refined_plane_point - (self.refined_normal * distance)
                        
                        # Re-select points within tolerance of offset plane
                        self.profile_indices = PointCloudAnalyzer.select_points_within_tolerance(
                            self.all_points_np,
                            self.refined_normal,
                            offset_plane_point,
                            profile_tolerance
                        )
                        
                        num_profile = len(self.profile_indices)
                        
                        if num_profile > 0:
                            # Re-highlight profile points with current color
                            profile_points_np = self.all_points_np[self.profile_indices]
                            profile_points_fc = [App.Vector(float(pt[0]), float(pt[1]), float(pt[2])) 
                                                 for pt in profile_points_np]
                            profile_color = Config.HIGHLIGHT_COLORS[self.profile_color_index]
                            highlighted = self.highlighter.highlight_profile_points(profile_points_fc, profile_color, skip_if_too_many=True)
                            
                            if highlighted:
                                self.profile_count_label.setText(f"Profile: {num_profile} points at {distance}mm offset")
                            else:
                                self.profile_count_label.setText(f"Profile: {num_profile} points (viz skipped)")
                        else:
                            self.profile_count_label.setText("Profile: 0 points")
                            self.profile_indices = []
                except ValueError:
                    # Invalid profile settings - ignore profile update
                    pass
            
            # Update count display
            inlier_percentage = (len(inlier_indices) / len(selected_points_np)) * 100
            viz_note = " (viz skipped)" if visualization_skipped else ""
            self.count_label.setText(
                f"Selected: {len(self.selected_indices)} points "
                f"({len(inlier_indices)} inliers, {inlier_percentage:.1f}%){viz_note}"
            )
            
            self.info_display.append(
                f"Tolerance {tolerance}mm: {len(self.selected_indices)} points "
                f"({inlier_percentage:.1f}% inliers)\n"
            )
            
            self.create_button.setEnabled(True)
            self.add_profile_button.setEnabled(True)  # Enable profile button when base plane is ready
            
            # Print timing summary
            t_end = time.time()
            timing_results.append(f"\n=== UPDATE PREVIEW TOTAL: {t_end-t_start:.3f}s ===")
            print("\n".join(timing_results))
            
        except Exception as e:
            self.count_label.setText(f"Error: {str(e)}")
            self.create_button.setEnabled(False)
            self.highlighter.clear_highlights()
            self.info_display.append(f"Error: {str(e)}\n")
            traceback.print_exc()
    
    def _add_profile_points(self):
        """Add profile plane points at offset distance from base plane."""
        if not self.refined_normal or not self.refined_plane_point:
            self.info_display.append("Error: No base plane defined yet.\n")
            return
        
        # Parse distance from text input (can be positive or negative)
        try:
            distance = float(self.profile_distance_edit.text())
        except ValueError:
            self.info_display.append("Error: Invalid distance value.\n")
            return
        
        # Parse tolerance from text input
        try:
            tolerance = float(self.profile_tolerance_edit.text())
            if tolerance <= 0:
                self.info_display.append("Error: Tolerance must be positive.\n")
                return
        except ValueError:
            self.info_display.append("Error: Invalid tolerance value.\n")
            return
        
        # Create offset plane (move away from camera, opposite of normal)
        offset_plane_point = self.refined_plane_point - (self.refined_normal * distance)
        
        # Select points within tolerance of offset plane
        self.profile_indices = PointCloudAnalyzer.select_points_within_tolerance(
            self.all_points_np,
            self.refined_normal,  # Same normal as base plane
            offset_plane_point,
            tolerance
        )
        
        num_profile = len(self.profile_indices)
        
        if num_profile == 0:
            self.info_display.append(f"No points found at offset {distance}mm with tolerance {tolerance}mm\n")
            self.profile_count_label.setText("Profile: 0 points")
            return
        
        # Convert to FreeCAD vectors and highlight in profile color
        profile_points_np = self.all_points_np[self.profile_indices]
        profile_points_fc = [App.Vector(float(pt[0]), float(pt[1]), float(pt[2])) 
                             for pt in profile_points_np]
        
        # Highlight in profile color (magenta by default)
        profile_color = Config.HIGHLIGHT_COLORS[self.profile_color_index]
        highlighted = self.highlighter.highlight_profile_points(profile_points_fc, profile_color, skip_if_too_many=True)
        
        # Switch color mode to profile so swatches now control profile color
        self.color_mode = "profile"
        self.color_mode_label.setText("Profile Highlight Color:")
        
        # Update color button selection to show current profile color
        for i, button in enumerate(self.color_buttons):
            button_color = Config.HIGHLIGHT_COLORS[i]
            self._update_color_button_style(button, button_color, i == self.profile_color_index)
        
        if highlighted:
            self.profile_count_label.setText(f"Profile: {num_profile} points at {distance}mm offset")
            self.info_display.append(f"Added {num_profile} profile points at {distance}mm offset (tolerance {tolerance}mm)\n")
        else:
            self.profile_count_label.setText(f"Profile: {num_profile} points (viz skipped)")
            self.info_display.append(f"Added {num_profile} profile points (visualization skipped)\n")
    
    def _create_sketch(self):
        """Create the sketch with datum plane."""
        if not self.selected_indices or not self.refined_normal:
            self.info_display.append("Error: No valid plane to create sketch from.\n")
            return
        
        # Get base plane points
        selected_points_np = self.all_points_np[self.selected_indices]
        selected_points = [App.Vector(pt[0], pt[1], pt[2]) for pt in selected_points_np]
        
        # Add profile points if they exist
        if self.profile_indices:
            profile_points_np = self.all_points_np[self.profile_indices]
            profile_points = [App.Vector(pt[0], pt[1], pt[2]) for pt in profile_points_np]
            # Combine both sets
            all_points = selected_points + profile_points
            self.info_display.append(f"Including {len(selected_points)} base + {len(profile_points)} profile points\n")
        else:
            all_points = selected_points
        
        # Show destination dialog
        destination = SketchCreator.show_destination_dialog()
        
        if not destination:
            return
        
        # Create sketch based on destination
        doc = App.ActiveDocument
        doc.openTransaction("Create Point Cloud Sketch")
        
        try:
            if destination["type"] == "standalone":
                datum, sketch = SketchCreator.create_standalone(
                    self.refined_normal,
                    self.refined_plane_point,
                    all_points
                )
                self.info_display.append("Created standalone datum plane and sketch\n")
                
            elif destination["type"] == "new_body":
                datum, sketch, body = SketchCreator.create_in_body(
                    self.refined_normal,
                    self.refined_plane_point,
                    all_points
                )
                self.info_display.append(f"Created datum plane and sketch in new body: {body.Name}\n")
                
            elif destination["type"] == "existing_body":
                body = doc.getObject(destination["body_name"])
                if not body:
                    raise ValueError(f"Body {destination['body_name']} not found")
                
                datum, sketch, _ = SketchCreator.create_in_body(
                    self.refined_normal,
                    self.refined_plane_point,
                    all_points,
                    body
                )
                self.info_display.append(f"Created datum plane and sketch in existing body: {body.Name}\n")
            
            # Select the new sketch and fit view
            Gui.Selection.clearSelection()
            Gui.Selection.addSelection(sketch)
            Gui.activeDocument().activeView().viewAxonometric()
            Gui.activeDocument().activeView().fitAll()
            
            doc.commitTransaction()
            
            self.info_display.append(
                f"Success! Created sketch with {len(all_points)} construction points.\n"
            )
            
            # Clear highlights after successful creation
            self.highlighter.clear_highlights()
            
        except Exception as e:
            doc.abortTransaction()
            self.info_display.append(f"Error creating sketch: {str(e)}\n")
            import traceback
            traceback.print_exc()
    
    def closeEvent(self, event):
        """Handle widget close event."""
        self.highlighter.clear_highlights()
        
        # Restore source object visibility
        if self.source_object and self.source_object_visibility is not None:
            try:
                if hasattr(self.source_object, 'ViewObject'):
                    self.source_object.ViewObject.Visibility = self.source_object_visibility
            except Exception:
                pass
        
        event.accept()


class PointCloudPlaneDockWidget(QDockWidget):
    """Docker widget for Point Cloud Plane Sketch."""
    
    def __init__(self):
        super().__init__()
        self._setup_dock_properties()
        self._setup_main_widget()
    
    def _setup_dock_properties(self):
        """Setup dock widget properties."""
        self.setWindowTitle("Point Cloud Plane Sketch")
        self.setObjectName("PointCloudPlaneSketch")
        
        self.setFeatures(
            QDockWidget.DockWidgetMovable |
            QDockWidget.DockWidgetFloatable |
            QDockWidget.DockWidgetClosable
        )
        self.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
    
    def _setup_main_widget(self):
        """Setup the main widget."""
        self.main_widget = PointCloudPlaneWidget()
        self.setWidget(self.main_widget)
    
    def closeEvent(self, event):
        """Handle dock close event."""
        self.main_widget.closeEvent(event)
        
        # Clear global reference when closing
        global point_cloud_plane_dock
        point_cloud_plane_dock = None
        
        event.accept()


# Global reference to prevent garbage collection
point_cloud_plane_dock = None


def show_point_cloud_plane_sketch():
    """Show the Point Cloud Plane Sketch docker."""
    global point_cloud_plane_dock
    
    # Check for active document
    if not App.ActiveDocument:
        QMessageBox.warning(
            Gui.getMainWindow(),
            "No Document",
            "Please open or create a document first."
        )
        return
    
    # Clean up existing dock
    main_window = Gui.getMainWindow()
    existing_docks = main_window.findChildren(QDockWidget)
    
    for dock in existing_docks:
        if dock.objectName() == "PointCloudPlaneSketch":
            dock.close()
            dock.deleteLater()
    
    # Create new dock
    point_cloud_plane_dock = PointCloudPlaneDockWidget()
    main_window.addDockWidget(QtCore.Qt.RightDockWidgetArea, point_cloud_plane_dock)
    point_cloud_plane_dock.show()


# Run the macro
if __name__ == "__main__":
    show_point_cloud_plane_sketch()