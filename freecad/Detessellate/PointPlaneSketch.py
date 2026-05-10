#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-2.1-or-later
# SPDX-FileCopyrightText: 2024 DesignWeaver3D
# SPDX-FileNotice: Part of the Detessellate addon.

"""
Point Plane Sketch - FreeCAD Macro

Workflow:
1. Using Part workbench convert Mesh to Points.
2. User selects 3+ vertices from points shape object to define approximate plane
3. Interactive docker shows tolerance adjustment with live preview
4. All points within tolerance are selected and highlighted
5. RANSAC fits best plane to selected points
6. User defines voxel filter size if desired (recommended if selection is >1k points)
7. User chooses new sketch destination (Standalone/New Body/Existing Body)
8. Creates datum plane and sketch with projected construction points

"""

from __future__ import annotations

# Standard library
import math
import re
import traceback

# Third-party
import numpy as np
from pivy import coin
from PySide6 import QtCore
from PySide6.QtWidgets import (QApplication, QDockWidget, QDoubleSpinBox,
                               QHBoxLayout, QInputDialog, QLabel, QLineEdit,
                               QMessageBox, QPushButton, QTextEdit, QToolButton,
                               QVBoxLayout, QWidget)

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
    MAX_RANSAC_POINTS = 10000
    
    HIGHLIGHT_POINT_SIZE = 8.0

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

    @staticmethod
    def voxel_filter(base_np: np.ndarray, profile_np: np.ndarray,
                     placement: App.Placement, cell_size: float):
        """
        Reduce base and profile point sets using a 3D voxel grid aligned to the
        fitted plane's local coordinate system.

        Points from both sets are pooled and binned together. Within each occupied
        cell the point whose local XY position is closest to the cell's XY center
        is retained, regardless of which set it came from. Survivors are then split
        back into base and profile sets using their original source tags.

        Args:
            base_np:    Nx3 numpy array of base plane points in global coords.
            profile_np: Mx3 numpy array of profile plane points in global coords
                        (may be empty).
            placement:  The sketch placement whose inverse transforms global →
                        plane-local coordinates.
            cell_size:  Voxel cell size in mm (cube side length).

        Returns:
            (filtered_base_np, filtered_profile_np) both in global coordinates.
        """
        # Build combined array with source tag: 0 = base, 1 = profile
        has_profile = profile_np is not None and len(profile_np) > 0
        if has_profile:
            combined = np.vstack([base_np, profile_np])
            tags = np.array([0] * len(base_np) + [1] * len(profile_np), dtype=np.int8)
        else:
            combined = base_np
            tags = np.zeros(len(base_np), dtype=np.int8)

        # Transform all points to plane-local space in one vectorised operation
        inv = placement.inverse()
        mat = np.array(inv.Matrix.A).reshape(4, 4)
        ones = np.ones((len(combined), 1))
        homogeneous = np.hstack([combined, ones])          # Nx4
        local = (mat @ homogeneous.T).T[:, :3]            # Nx3 in local space

        # Compute cell indices from local XY only (Z ignored for binning)
        ix = np.floor(local[:, 0] / cell_size).astype(np.int32)
        iy = np.floor(local[:, 1] / cell_size).astype(np.int32)

        # Cell center XY in local space
        cx = (ix + 0.5) * cell_size
        cy = (iy + 0.5) * cell_size

        # 2D distance from each point to its cell center (local XY only)
        dx = local[:, 0] - cx
        dy = local[:, 1] - cy
        dist2 = dx * dx + dy * dy

        # For each cell keep the index of the point closest to XY center
        cell_map: dict[tuple[int, int], tuple[float, int]] = {}
        for i, (key_x, key_y, d2) in enumerate(zip(ix, iy, dist2)):
            key = (int(key_x), int(key_y))
            if key not in cell_map or d2 < cell_map[key][0]:
                cell_map[key] = (d2, i)

        kept = np.array([idx for _, idx in cell_map.values()], dtype=np.int64)

        # Split survivors back by source tag
        kept_tags = tags[kept]
        filtered_base = combined[kept[kept_tags == 0]]
        filtered_profile = combined[kept[kept_tags == 1]] if has_profile else np.empty((0, 3))

        return filtered_base, filtered_profile


class PointHighlighter:
    """Handles point highlighting in the 3D view via Coin3D scene graph nodes."""

    def __init__(self):
        self._base_sep = None       # SoSeparator for base plane points
        self._profile_sep = None    # SoSeparator for profile plane points
        self._arrow_sep = None      # SoSeparator for normal direction indicator
        self._base_color = None     # SoBaseColor node for live color updates
        self._arrow_color = None    # SoBaseColor node for arrow color updates
        self.current_color = Config.HIGHLIGHT_COLORS[Config.DEFAULT_HIGHLIGHT_COLOR_INDEX]

    def _get_root(self):
        """Return the scene graph root, or None if unavailable."""
        try:
            return Gui.ActiveDocument.ActiveView.getSceneGraph()
        except Exception:
            return None

    def _make_separator(self, points_np, color):
        """
        Build a Coin3D SoSeparator containing a colored SoPointSet.

        Args:
            points_np: numpy array (Nx3) or list of (x,y,z) tuples
            color: (R, G, B) float tuple

        Returns:
            (SoSeparator, SoBaseColor) so the caller can update color later
        """
        sep = coin.SoSeparator()

        base_color = coin.SoBaseColor()
        base_color.rgb = color

        style = coin.SoDrawStyle()
        style.pointSize = Config.HIGHLIGHT_POINT_SIZE

        coords = coin.SoCoordinate3()
        pts = [(float(p[0]), float(p[1]), float(p[2])) for p in points_np]
        coords.point.setValues(0, len(pts), pts)

        ps = coin.SoPointSet()

        sep.addChild(base_color)
        sep.addChild(style)
        sep.addChild(coords)
        sep.addChild(ps)

        return sep, base_color

    def set_color(self, color):
        """Set the highlight color and update existing base highlight and arrow."""
        self.current_color = color
        if self._base_color is not None:
            self._base_color.rgb = color
        if self._arrow_color is not None:
            self._arrow_color.rgb = color

    def highlight_points(self, points_np):
        """
        Highlight base plane points via a Coin3D SoPointSet.

        Args:
            points_np: numpy array (Nx3) or list of (x,y,z) tuples

        Returns:
            True if highlighted, False on error or empty input
        """
        self.clear_highlights(clear_profile=False)

        if points_np is None or len(points_np) == 0:
            return False

        root = self._get_root()
        if root is None:
            return False

        try:
            sep, base_color = self._make_separator(points_np, self.current_color)
            root.addChild(sep)
            self._base_sep = sep
            self._base_color = base_color
            return True
        except Exception as e:
            print(f"Highlighting error: {e}")
            traceback.print_exc()
            return False

    def highlight_profile_points(self, points_np, color):
        """
        Highlight profile plane points via a Coin3D SoPointSet.

        Args:
            points_np: numpy array (Nx3) or list of (x,y,z) tuples
            color: (R, G, B) float tuple

        Returns:
            True if highlighted, False on error or empty input
        """
        # Clear existing profile highlight only
        root = self._get_root()
        if root is not None and self._profile_sep is not None:
            try:
                root.removeChild(self._profile_sep)
            except Exception:
                pass
        self._profile_sep = None

        if points_np is None or len(points_np) == 0:
            return False

        if root is None:
            return False

        try:
            sep, _ = self._make_separator(points_np, color)
            root.addChild(sep)
            self._profile_sep = sep
            return True
        except Exception as e:
            print(f"Profile highlighting error: {e}")
            traceback.print_exc()
            return False

    def clear_highlights(self, clear_profile=True):
        """Remove Coin3D highlight nodes from the scene graph."""
        root = self._get_root()

        if root is not None and self._base_sep is not None:
            try:
                root.removeChild(self._base_sep)
            except Exception:
                pass
        self._base_sep = None
        self._base_color = None

        if clear_profile:
            if root is not None and self._profile_sep is not None:
                try:
                    root.removeChild(self._profile_sep)
                except Exception:
                    pass
            self._profile_sep = None

        if root is not None and self._arrow_sep is not None:
            try:
                root.removeChild(self._arrow_sep)
            except Exception:
                pass
        self._arrow_sep = None
        self._arrow_color = None
    
    def show_normal_arrow(self, origin: App.Vector, normal: App.Vector, length: float = 50.0):
        """
        Show a line indicating the plane normal direction via Coin3D.

        Args:
            origin: Starting point of the arrow (plane origin)
            normal: Normal vector direction
            length: Length of the line in mm
        """
        root = self._get_root()

        # Clear existing arrow node
        if root is not None and self._arrow_sep is not None:
            try:
                root.removeChild(self._arrow_sep)
            except Exception:
                pass
        self._arrow_sep = None
        self._arrow_color = None

        if root is None:
            return

        try:
            normal_unit = App.Vector(normal.x, normal.y, normal.z).normalize()
            tip = origin + (normal_unit * length)

            sep = coin.SoSeparator()

            arrow_color = coin.SoBaseColor()
            arrow_color.rgb = self.current_color

            style = coin.SoDrawStyle()
            style.lineWidth = 3.0

            coords = coin.SoCoordinate3()
            coords.point.setValues(0, 2, [
                (origin.x, origin.y, origin.z),
                (tip.x,    tip.y,    tip.z),
            ])

            line = coin.SoLineSet()
            line.numVertices.setValue(2)

            # Small sphere at the tip to indicate direction
            tip_transform = coin.SoTranslation()
            tip_transform.translation = (tip.x, tip.y, tip.z)
            tip_sphere = coin.SoSphere()
            tip_sphere.radius = length * 0.04

            sep.addChild(arrow_color)
            sep.addChild(style)
            sep.addChild(coords)
            sep.addChild(line)
            sep.addChild(tip_transform)
            sep.addChild(tip_sphere)

            root.addChild(sep)
            self._arrow_sep = sep
            self._arrow_color = arrow_color

        except Exception as e:
            print(f"Error creating normal arrow: {e}")
            traceback.print_exc()
    
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

        item, ok = QInputDialog.getItem(
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
        if not points:
            return

        # Vectorise the global → local coordinate transform using the inverse placement matrix
        inv = placement.inverse()
        mat = np.array(inv.Matrix.A).reshape(4, 4)
        pts_np = np.array([[p.x, p.y, p.z] for p in points])
        ones = np.ones((len(pts_np), 1))
        local = (mat @ np.hstack([pts_np, ones]).T).T[:, :2]  # Nx2 local XY, Z discarded

        for lx, ly in local:
            sketch.addGeometry(Part.Point(App.Vector(float(lx), float(ly), 0)), True)

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

        # Voxel filter state
        self.filter_active = False
        self.filtered_base_np = None    # Nx3 global coords after filtering
        self.filtered_profile_np = None # Mx3 global coords after filtering
        
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
        self.profile_distance_edit = QLineEdit()
        self.profile_distance_edit.setText("5.0")
        self.profile_distance_edit.setMinimumWidth(80)
        self.profile_distance_edit.setMaximumWidth(100)
        profile_dist_layout.addWidget(self.profile_distance_edit)
        profile_dist_layout.addStretch()
        tolerance_layout.addLayout(profile_dist_layout)
        
        # Profile tolerance
        profile_tol_layout = QHBoxLayout()
        profile_tol_layout.addWidget(QLabel("Tolerance (mm):"))
        self.profile_tolerance_edit = QLineEdit()
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

        # Voxel filter section
        voxel_label = QLabel("Voxel Filter (Optional):")
        voxel_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        tolerance_layout.addWidget(voxel_label)

        voxel_input_layout = QHBoxLayout()
        voxel_input_layout.addWidget(QLabel("Cell Size (mm):"))
        self.voxel_size_spin = QDoubleSpinBox()
        self.voxel_size_spin.setRange(0.1, 100.0)
        self.voxel_size_spin.setValue(1.0)
        self.voxel_size_spin.setDecimals(1)
        self.voxel_size_spin.setSingleStep(0.1)
        self.voxel_size_spin.setMinimumWidth(80)
        voxel_input_layout.addWidget(self.voxel_size_spin)

        self.apply_filter_button = QPushButton("Apply Filter")
        self.apply_filter_button.clicked.connect(self._apply_voxel_filter)
        self.apply_filter_button.setEnabled(False)
        voxel_input_layout.addWidget(self.apply_filter_button)

        self.clear_filter_button = QPushButton("Clear Filter")
        self.clear_filter_button.clicked.connect(self._clear_voxel_filter)
        self.clear_filter_button.setEnabled(False)
        voxel_input_layout.addWidget(self.clear_filter_button)

        voxel_input_layout.addStretch()
        tolerance_layout.addLayout(voxel_input_layout)

        self.voxel_status_label = QLabel("")
        self.voxel_status_label.setStyleSheet("font-style: italic;")
        tolerance_layout.addWidget(self.voxel_status_label)

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
        button = QToolButton()
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
            self.info_display.append("Changed base highlight color\n")
        else:  # profile mode
            self.profile_color_index = index
            # Re-highlight profile points if they exist
            if self.profile_indices:
                profile_points_np = self.all_points_np[self.profile_indices]
                self.highlighter.highlight_profile_points(profile_points_np, color)
            self.info_display.append("Changed profile highlight color\n")
        
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

        # Reset filter state
        self.filter_active = False
        self.filtered_base_np = None
        self.filtered_profile_np = None
        self.voxel_status_label.setText("")
        self.apply_filter_button.setEnabled(False)
        self.clear_filter_button.setEnabled(False)
    
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
                "To work with a different object, close this docker and restart the macro.\n"
            )
            return
        
        # Collect selected vertices via PickedPoints — avoids materializing OCCT vertex list
        selected_vertices = []
        vertices_from_other_objects = 0

        for sel in selection:
            if sel.Object != self.source_object:
                if hasattr(sel.Object, 'Shape'):
                    for sub_name in sel.SubElementNames:
                        if sub_name.startswith("Vertex"):
                            vertices_from_other_objects += 1
                continue
            for pt in sel.PickedPoints:
                selected_vertices.append(App.Vector(pt.x, pt.y, pt.z))
        
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
        QApplication.processEvents()  # Force UI update
        
        # Use QTimer to defer actual collection so message can display
        QtCore.QTimer.singleShot(100, self._initialize_from_selection)
    
    def _initialize_from_selection(self):
        """Initialize plane from selected vertices."""
        selection = Gui.Selection.getSelectionEx()
        if not selection:
            self.info_display.append("Error: No selection made.\n")
            return

        # Collect selected vertices from BREP by index — avoids materializing all OCCT wrappers
        selected_vertices = []
        self.source_object = selection[0].Object
        vertices_from_other_objects = 0

        for sel in selection:
            if sel.Object != self.source_object:
                if hasattr(sel.Object, 'Shape'):
                    for sub_name in sel.SubElementNames:
                        if sub_name.startswith("Vertex"):
                            vertices_from_other_objects += 1
                continue
            for pt in sel.PickedPoints:
                selected_vertices.append(App.Vector(pt.x, pt.y, pt.z))

        # Inform user if vertices from other objects were ignored
        if vertices_from_other_objects > 0:
            self.info_display.append(
                f"Note: Ignored {vertices_from_other_objects} vertex/vertices from other objects. "
                f"Only using vertices from {self.source_object.Label}\n"
            )

        # Store user-selected vertices for plane origin calculation
        self.user_selected_vertices = selected_vertices

        if len(selected_vertices) < 3:
            self.info_display.append("Error: Please select at least 3 vertices.\n")
            return

        try:
            # Collect all points only once via BREP string parsing —
            # avoids materializing 400k+ OCCT vertex wrappers simultaneously
            if self.all_points_np is None or len(self.all_points_np) == 0:
                if not hasattr(self.source_object, 'Shape'):
                    self.info_display.append("Error: Object does not have a Shape.\n")
                    return

                self.info_display.append("Parsing point data from shape...\n")

                brep = self.source_object.Shape.exportBrepToString()
                coords = re.findall(
                    r'Ve\n[\d.e+-]+\n([-\d.e+]+) ([-\d.e+]+) ([-\d.e+]+)',
                    brep
                )
                del brep  # Free the string immediately

                if len(coords) < 3:
                    self.info_display.append("Error: Object does not contain enough vertices.\n")
                    return

                self.all_points_np = np.array(coords, dtype=np.float64)
                del coords

                self.info_display.append(f"Loaded {len(self.all_points_np)} points\n")

                if hasattr(self.source_object, 'ViewObject'):
                    self.source_object_visibility = self.source_object.ViewObject.Visibility
                    self.source_object.ViewObject.Visibility = False
                    self.info_display.append(f"Hid {self.source_object.Label} (Name: {self.source_object.Name}) to show highlights\n")
                self.init_button.setVisible(False)
                self.new_selection_button.setVisible(True)

            # Plane fitting
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
    
    def _update_preview(self):
        """Update the preview with current tolerance."""
        if self.all_points_np is None or self.initial_normal is None:
            self.info_display.append("Error: Collect vertex data first.\n")
            return
        
        tolerance = self.tolerance_spin.value()

        try:
            self.selected_indices = PointCloudAnalyzer.select_points_within_tolerance(
                self.all_points_np,
                self.initial_normal,
                self.initial_plane_point,
                tolerance
            )

            if len(self.selected_indices) < 3:
                self.count_label.setText(f"Selected: {len(self.selected_indices)} points (need at least 3)")
                self.create_button.setEnabled(False)
                self.highlighter.clear_highlights(clear_profile=False)
                self.info_display.append(f"Tolerance {tolerance}mm: only {len(self.selected_indices)} points - too few\n")
                return

            selected_points_np = self.all_points_np[self.selected_indices]

            # Subsample for RANSAC — full set not needed for plane fitting
            if len(selected_points_np) > Config.MAX_RANSAC_POINTS:
                ransac_indices = np.random.choice(len(selected_points_np), Config.MAX_RANSAC_POINTS, replace=False)
                ransac_points = selected_points_np[ransac_indices]
            else:
                ransac_points = selected_points_np

            # Refine plane using RANSAC on selected points
            self.refined_normal, ransac_centroid, inlier_indices = \
                PointCloudAnalyzer.fit_plane_ransac(
                    ransac_points,
                    tolerance,
                    Config.RANSAC_ITERATIONS
                )

            # Re-orient normal based on current camera position
            self.refined_normal = PointCloudAnalyzer.orient_normal_toward_viewer(
                self.refined_normal,
                ransac_centroid
            )

            # Project user-selected centroid onto the fitted plane
            user_selected_np = np.array([[p.x, p.y, p.z] for p in self.user_selected_vertices])
            user_centroid_np = np.mean(user_selected_np, axis=0)
            user_centroid = App.Vector(user_centroid_np[0], user_centroid_np[1], user_centroid_np[2])
            to_user_centroid = user_centroid - ransac_centroid
            distance_to_plane = to_user_centroid.dot(self.refined_normal)
            self.refined_plane_point = user_centroid - (self.refined_normal * distance_to_plane)

            # Highlight selected points directly from numpy array (no conversion needed)
            self.highlighter.highlight_points(selected_points_np)

            # Show normal arrow indicator
            self.highlighter.show_normal_arrow(self.refined_plane_point, self.refined_normal, length=50.0)

            # Update profile points if they were previously added
            if self.profile_indices:
                try:
                    distance = float(self.profile_distance_edit.text())
                    profile_tolerance = float(self.profile_tolerance_edit.text())

                    if profile_tolerance > 0:
                        offset_plane_point = self.refined_plane_point - (self.refined_normal * distance)

                        self.profile_indices = PointCloudAnalyzer.select_points_within_tolerance(
                            self.all_points_np,
                            self.refined_normal,
                            offset_plane_point,
                            profile_tolerance
                        )

                        num_profile = len(self.profile_indices)

                        if num_profile > 0:
                            profile_points_np = self.all_points_np[self.profile_indices]
                            profile_color = Config.HIGHLIGHT_COLORS[self.profile_color_index]
                            self.highlighter.highlight_profile_points(profile_points_np, profile_color)
                            self.profile_count_label.setText(f"Profile: {num_profile} points at {distance}mm offset")
                        else:
                            self.profile_count_label.setText("Profile: 0 points")
                            self.profile_indices = []
                except ValueError:
                    pass

            inlier_percentage = (len(inlier_indices) / len(ransac_points)) * 100
            self.count_label.setText(
                f"Selected: {len(self.selected_indices)} points "
                f"({len(inlier_indices)} inliers, {inlier_percentage:.1f}%)"
            )
            self.info_display.append(
                f"Tolerance {tolerance}mm: {len(self.selected_indices)} points "
                f"({inlier_percentage:.1f}% inliers)\n"
            )

            self.create_button.setEnabled(True)
            self.add_profile_button.setEnabled(True)
            self.apply_filter_button.setEnabled(True)

            # Clear any stale filter — selection has changed
            if self.filter_active:
                self._clear_voxel_filter(silent=True)

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
        
        # Highlight in profile color (magenta by default)
        profile_points_np = self.all_points_np[self.profile_indices]
        profile_color = Config.HIGHLIGHT_COLORS[self.profile_color_index]
        self.highlighter.highlight_profile_points(profile_points_np, profile_color)

        # Switch color mode to profile so swatches now control profile color
        self.color_mode = "profile"
        self.color_mode_label.setText("Profile Highlight Color:")

        # Update color button selection to show current profile color
        for i, button in enumerate(self.color_buttons):
            button_color = Config.HIGHLIGHT_COLORS[i]
            self._update_color_button_style(button, button_color, i == self.profile_color_index)

        self.profile_count_label.setText(f"Profile: {num_profile} points at {distance}mm offset")
        self.info_display.append(f"Added {num_profile} profile points at {distance}mm offset (tolerance {tolerance}mm)\n")
    
    def _apply_voxel_filter(self):
        """Apply 3D voxel filter to the current base and profile point selections."""
        if not self.selected_indices or self.refined_normal is None:
            self.info_display.append("Error: No valid selection to filter.\n")
            return

        cell_size = self.voxel_size_spin.value()
        base_np = self.all_points_np[self.selected_indices]
        profile_np = self.all_points_np[self.profile_indices] if self.profile_indices else np.empty((0, 3))

        placement = SketchCreator.create_placement_from_plane(self.refined_normal, self.refined_plane_point)

        try:
            filtered_base, filtered_profile = PointCloudAnalyzer.voxel_filter(
                base_np, profile_np, placement, cell_size
            )
        except Exception as e:
            self.info_display.append(f"Filter error: {e}\n")
            traceback.print_exc()
            return

        self.filtered_base_np = filtered_base
        self.filtered_profile_np = filtered_profile
        self.filter_active = True
        self.clear_filter_button.setEnabled(True)

        # Update highlights to show filtered sets
        self.highlighter.highlight_points(filtered_base)
        self.highlighter.show_normal_arrow(self.refined_plane_point, self.refined_normal, length=50.0)
        if self.profile_indices and len(filtered_profile) > 0:
            profile_color = Config.HIGHLIGHT_COLORS[self.profile_color_index]
            self.highlighter.highlight_profile_points(filtered_profile, profile_color)

        # Status label
        base_orig = len(self.selected_indices)
        base_filt = len(filtered_base)
        status = f"Base: {base_orig} → {base_filt} points"
        if self.profile_indices:
            prof_orig = len(self.profile_indices)
            prof_filt = len(filtered_profile)
            status += f" | Profile: {prof_orig} → {prof_filt} points"
        self.voxel_status_label.setText(status)
        self.info_display.append(f"Voxel filter applied ({cell_size}mm): {status}\n")

    def _clear_voxel_filter(self, silent=False):
        """Remove voxel filter and restore full point highlights."""
        self.filter_active = False
        self.filtered_base_np = None
        self.filtered_profile_np = None
        self.voxel_status_label.setText("")
        self.clear_filter_button.setEnabled(False)

        # Restore full highlights
        if self.selected_indices:
            self.highlighter.highlight_points(self.all_points_np[self.selected_indices])
            self.highlighter.show_normal_arrow(self.refined_plane_point, self.refined_normal, length=50.0)
        if self.profile_indices:
            profile_color = Config.HIGHLIGHT_COLORS[self.profile_color_index]
            self.highlighter.highlight_profile_points(
                self.all_points_np[self.profile_indices], profile_color
            )

        if not silent:
            self.info_display.append("Voxel filter cleared.\n")

    def _create_sketch(self):
        """Create the sketch with datum plane."""
        if not self.selected_indices or not self.refined_normal:
            self.info_display.append("Error: No valid plane to create sketch from.\n")
            return

        # Use filtered sets if filter is active, otherwise full selections
        if self.filter_active and self.filtered_base_np is not None:
            selected_points = [App.Vector(pt[0], pt[1], pt[2]) for pt in self.filtered_base_np]
            if self.filtered_profile_np is not None and len(self.filtered_profile_np) > 0:
                profile_points = [App.Vector(pt[0], pt[1], pt[2]) for pt in self.filtered_profile_np]
                all_points = selected_points + profile_points
                self.info_display.append(
                    f"Using filtered points: {len(selected_points)} base + {len(profile_points)} profile\n"
                )
            else:
                all_points = selected_points
                self.info_display.append(f"Using filtered points: {len(selected_points)} base\n")
        else:
            selected_points_np = self.all_points_np[self.selected_indices]
            selected_points = [App.Vector(pt[0], pt[1], pt[2]) for pt in selected_points_np]
            if self.profile_indices:
                profile_points_np = self.all_points_np[self.profile_indices]
                profile_points = [App.Vector(pt[0], pt[1], pt[2]) for pt in profile_points_np]
                all_points = selected_points + profile_points
                self.info_display.append(
                    f"Including {len(selected_points)} base + {len(profile_points)} profile points\n"
                )
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



def run():
    show_point_cloud_plane_sketch()


# Run the macro
if __name__ == "__main__":
    show_point_cloud_plane_sketch()