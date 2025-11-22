#!/usr/bin/env python3
"""
SketcherWireDoctor - FreeCAD Sketcher Analysis Tool - Main Module

INSTALLATION INSTRUCTIONS:
After downloading, rename the files to:
- wire_doctor_main.py → SketcherWireDoctor_Main.py
- wire_doctor_tab1.py → SketcherWireDoctor_Tab1.py  
- wire_doctor_tab2.py → SketcherWireDoctor_Tab2.py
- wire_doctor_tab3.py → SketcherWireDoctor_Tab3.py
- wire_doctor_tab4.py → SketcherWireDoctor_Tab4.py

Place all 5 files in your FreeCAD macro directory.
Run SketcherWireDoctor_Main.py to start the tool.

A comprehensive tool for analyzing and fixing wire closure issues in FreeCAD sketches.
Identifies zero-length lines, duplicate geometry, non-coincident vertices, and
problematic intersections.

Author: Original implementation refactored for best practices
Version: 0.72 (Minimal debug messages)
"""

import math
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Set, Any, Union

import FreeCAD as App
import FreeCADGui as Gui
import Part
import Sketcher
from PySide import QtCore, QtGui


# Constants
class Config:
    """Configuration constants for the application."""
    TOLERANCE = 1e-6
    COORDINATE_PRECISION = 8
    DISPLAY_PRECISION = 3
    MAX_PATH_LENGTH = 10  # Prevent infinite recursion in graph traversal
    HIGHLIGHT_LINE_WIDTH = 5.0
    HIGHLIGHT_TRANSPARENCY = 30

    # UI Constants
    COLOR_BUTTON_SIZE = (20, 20, 30, 30)  # min_w, min_h, max_w, max_h

    # Geometry type mappings
    GEOMETRY_TYPE_MAP = {
        'Part::GeomLineSegment': 'Line',
        'Part::GeomArcOfCircle': 'Arc',
        'Part::GeomCircle': 'Circle',
        'Part::GeomBSplineCurve': 'BSpline',
        'Part::GeomEllipse': 'Ellipse'
    }

    # Supported geometry types
    ARC_TYPES = [
        "Part::GeomArcOfCircle",
        "Part::GeomArcOfEllipse",
        "Part::GeomArcOfHyperbola",
        "Part::GeomArcOfParabola"
    ]

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


def round_coord(point: App.Vector, digits: int = 8) -> Tuple[float, float]:
    """Round coordinates for precision comparison."""
    return (round(point.x, digits), round(point.y, digits))


class GeometryUtils:
    """Utility functions for geometry operations."""

    @staticmethod
    def get_geometry_name(geo_idx: int, geometry: Any) -> str:
        """Get the FreeCAD display name for geometry (1-based indexing)."""
        geo_type = Config.GEOMETRY_TYPE_MAP.get(geometry.TypeId, 'Geometry')
        return f"{geo_idx + 1}-{geo_type}"

    @staticmethod
    def get_geometry_endpoints(geometry: Any) -> Tuple[Optional[Tuple[float, float]],
                                                       Optional[Tuple[float, float]]]:
        """Get start and end coordinates for any geometry type."""
        start_coord = None
        end_coord = None

        if geometry.TypeId == "Part::GeomLineSegment":
            start_coord = round_coord(geometry.StartPoint)
            end_coord = round_coord(geometry.EndPoint)

        elif geometry.TypeId in Config.ARC_TYPES:
            # Try direct properties first
            if hasattr(geometry, 'StartPoint') and hasattr(geometry, 'EndPoint'):
                start_coord = round_coord(geometry.StartPoint)
                end_coord = round_coord(geometry.EndPoint)
            # Fallback to parametric evaluation
            elif (hasattr(geometry, 'value') and
                  hasattr(geometry, 'FirstParameter') and
                  hasattr(geometry, 'LastParameter')):
                try:
                    start_point = geometry.value(geometry.FirstParameter)
                    end_point = geometry.value(geometry.LastParameter)
                    start_coord = round_coord(start_point)
                    end_coord = round_coord(end_point)
                except Exception:
                    pass

        elif geometry.TypeId == "Part::GeomCircle":
            # Handle circles that are actually arcs
            try:
                if hasattr(geometry, 'StartPoint') and hasattr(geometry, 'EndPoint'):
                    start_coord = round_coord(geometry.StartPoint)
                    end_coord = round_coord(geometry.EndPoint)
                    # Skip full circles
                    if start_coord == end_coord:
                        return None, None
            except Exception:
                pass

        elif geometry.TypeId in ["Part::GeomBSplineCurve", "Part::GeomEllipse"]:
            if hasattr(geometry, 'StartPoint') and hasattr(geometry, 'EndPoint'):
                start_coord = round_coord(geometry.StartPoint)
                end_coord = round_coord(geometry.EndPoint)
                # For ellipses, skip if it's a full ellipse
                if geometry.TypeId == "Part::GeomEllipse" and start_coord == end_coord:
                    return None, None

        return start_coord, end_coord

    @staticmethod
    def get_geometry_midpoint(geometry: Any) -> Tuple[float, float]:
        """Get the midpoint of geometry for all types."""
        try:
            if geometry.TypeId == "Part::GeomLineSegment":
                start = geometry.StartPoint
                end = geometry.EndPoint
                return ((start.x + end.x) / 2, (start.y + end.y) / 2)

            elif geometry.TypeId == "Part::GeomArcOfCircle":
                # For arcs, get the midpoint along the arc path
                if (hasattr(geometry, 'FirstParameter') and
                    hasattr(geometry, 'LastParameter')):
                    mid_param = (geometry.FirstParameter + geometry.LastParameter) / 2
                    mid_point = geometry.value(mid_param)
                    return (mid_point.x, mid_point.y)
                else:
                    # Fallback to geometric midpoint of endpoints
                    start = geometry.StartPoint
                    end = geometry.EndPoint
                    return ((start.x + end.x) / 2, (start.y + end.y) / 2)

            elif geometry.TypeId == "Part::GeomBSplineCurve":
                # For B-splines, get midpoint along the curve
                try:
                    if (hasattr(geometry, 'FirstParameter') and
                        hasattr(geometry, 'LastParameter')):
                        mid_param = (geometry.FirstParameter + geometry.LastParameter) / 2
                        mid_point = geometry.value(mid_param)
                        return (mid_point.x, mid_point.y)
                except Exception:
                    pass
                # Fallback to geometric midpoint
                if hasattr(geometry, 'StartPoint') and hasattr(geometry, 'EndPoint'):
                    start = geometry.StartPoint
                    end = geometry.EndPoint
                    return ((start.x + end.x) / 2, (start.y + end.y) / 2)

            elif geometry.TypeId in ["Part::GeomCircle", "Part::GeomEllipse"]:
                # Use center point if available
                if hasattr(geometry, 'Center'):
                    center = geometry.Center
                    return (center.x, center.y)
                elif hasattr(geometry, 'StartPoint') and hasattr(geometry, 'EndPoint'):
                    start = geometry.StartPoint
                    end = geometry.EndPoint
                    return ((start.x + end.x) / 2, (start.y + end.y) / 2)

            # Generic fallback for any geometry with start/end points
            if hasattr(geometry, 'StartPoint') and hasattr(geometry, 'EndPoint'):
                start = geometry.StartPoint
                end = geometry.EndPoint
                return ((start.x + end.x) / 2, (start.y + end.y) / 2)

        except Exception:
            pass

        return (0, 0)


class GeometryAnalyzer:
    """Analyzes sketch geometry for wire closure issues."""

    def __init__(self, sketch):
        self.sketch = sketch
        self.all_geometry: List[Tuple[int, Any]] = []  # All geometry including construction
        self.normal_geometry: List[Tuple[int, Any]] = []  # Only normal geometry
        self.constraints: List[Any] = []

    def collect_data(self) -> None:
        """Collect all geometry and constraints from the sketch."""
        self._reset_data()
        self._collect_geometry()
        self._collect_constraints()

    def _reset_data(self) -> None:
        """Reset all internal data structures."""
        self.all_geometry.clear()
        self.normal_geometry.clear()
        self.constraints.clear()

    def _collect_geometry(self) -> None:
        """Collect all geometry and separate normal vs construction."""
        for i, geo in enumerate(self.sketch.Geometry):
            self.all_geometry.append((i, geo))
            if not self.sketch.getConstruction(i):
                self.normal_geometry.append((i, geo))

    def _collect_constraints(self) -> None:
        """Collect constraints."""
        self.constraints = list(self.sketch.Constraints)

    def find_zero_length_lines(self) -> List[Dict[str, Any]]:
        """Find all zero-length line segments - delegated to Tab1."""
        try:
            import importlib
            import sys
            
            # Force reload if module already exists to avoid stale imports
            if 'SketcherWireDoctor_Tab1' in sys.modules:
                importlib.reload(sys.modules['SketcherWireDoctor_Tab1'])
            
            import SketcherWireDoctor_Tab1
            return SketcherWireDoctor_Tab1.find_zero_length_lines(self)
        except ImportError:
            return []
        except Exception:
            return []

    def find_duplicate_geometry(self) -> List[List[Dict[str, Any]]]:
        """Find duplicate or overlapping geometry - delegated to Tab2."""
        try:
            import importlib
            import sys
            
            if 'SketcherWireDoctor_Tab2' in sys.modules:
                importlib.reload(sys.modules['SketcherWireDoctor_Tab2'])
                
            import SketcherWireDoctor_Tab2
            return SketcherWireDoctor_Tab2.find_duplicate_geometry(self)
        except ImportError:
            return []
        except Exception:
            return []

    def find_non_coincident_vertices(self) -> List[Dict[str, Any]]:
        """Find vertices that should be coincident but aren't - delegated to Tab3."""
        try:
            import importlib
            import sys
            
            if 'SketcherWireDoctor_Tab3' in sys.modules:
                importlib.reload(sys.modules['SketcherWireDoctor_Tab3'])
                
            import SketcherWireDoctor_Tab3
            return SketcherWireDoctor_Tab3.find_non_coincident_vertices(self)
        except ImportError:
            return []
        except Exception:
            return []

    def find_problematic_intersections(self) -> List[Dict[str, Any]]:
        """Find T-sections and bridge edges - delegated to Tab4."""
        try:
            import importlib
            import sys
            
            if 'SketcherWireDoctor_Tab4' in sys.modules:
                importlib.reload(sys.modules['SketcherWireDoctor_Tab4'])
                
            import SketcherWireDoctor_Tab4
            return SketcherWireDoctor_Tab4.find_problematic_intersections(self)
        except ImportError:
            return []
        except Exception:
            return []


class GeometryHighlighter:
    """Handles geometry highlighting in the 3D view."""

    def __init__(self, sketch):
        self.sketch = sketch
        self.highlight_objects: List[Any] = []
        self.current_color: Tuple[float, float, float] = Config.HIGHLIGHT_COLORS[0]

    def set_color(self, color: Tuple[float, float, float]) -> None:
        """Set the highlight color and update existing highlights."""
        self.current_color = color
        self._update_existing_highlights()

    def _update_existing_highlights(self) -> None:
        """Update color of existing highlight objects."""
        for highlight_obj in self.highlight_objects:
            try:
                view_obj = highlight_obj.ViewObject
                view_obj.ShapeColor = self.current_color
                view_obj.LineColor = self.current_color
            except Exception:
                pass
        Gui.updateGui()

    def highlight_geometry(self, geo_idx: int) -> None:
        """Highlight a specific geometry element."""
        self.clear_highlights()

        try:
            doc = App.ActiveDocument
            if not doc:
                return

            geometry = self.sketch.Geometry[geo_idx]
            highlight_obj = self._create_highlight_object(doc, geo_idx, geometry)

            if highlight_obj:
                self._setup_highlight_appearance(highlight_obj)
                self.highlight_objects.append(highlight_obj)
                doc.recompute()
                Gui.updateGui()

        except Exception:
            pass

    def _create_highlight_object(self, doc: Any, geo_idx: int, geometry: Any) -> Optional[Any]:
        """Create a highlight object for the given geometry."""
        highlight_obj = doc.addObject("Part::Feature", f"Highlight_Geo{geo_idx}")

        try:
            if geometry.TypeId == "Part::GeomLineSegment":
                highlight_obj.Shape = Part.makeLine(geometry.StartPoint, geometry.EndPoint)
                
            elif geometry.TypeId == "Part::GeomPoint":
                # Create a vertex to highlight points
                vertex = Part.Vertex(App.Vector(geometry.X, geometry.Y, 0))
                highlight_obj.Shape = vertex
                
            elif geometry.TypeId == "Part::GeomArcOfCircle":
                highlight_obj.Shape = Part.Edge(geometry)
                
            elif geometry.TypeId == "Part::GeomCircle":
                highlight_obj.Shape = Part.Edge(geometry)
                
            elif geometry.TypeId == "Part::GeomEllipse":
                try:
                    highlight_obj.Shape = Part.Edge(geometry)
                except Exception:
                    # Fallback: create ellipse manually if direct Edge creation fails
                    center = geometry.Center
                    major_radius = geometry.MajorRadius
                    minor_radius = geometry.MinorRadius
                    ellipse = Part.Ellipse(center, major_radius, minor_radius)
                    highlight_obj.Shape = Part.Edge(ellipse)
                    
            elif geometry.TypeId == "Part::GeomArcOfEllipse":
                try:
                    highlight_obj.Shape = Part.Edge(geometry)
                except Exception:
                    # Fallback for arc of ellipse
                    if hasattr(geometry, 'StartPoint') and hasattr(geometry, 'EndPoint'):
                        highlight_obj.Shape = Part.makeLine(geometry.StartPoint, geometry.EndPoint)
                    
            elif geometry.TypeId == "Part::GeomArcOfHyperbola":
                try:
                    highlight_obj.Shape = Part.Edge(geometry)
                except Exception:
                    # Fallback for arc of hyperbola
                    if hasattr(geometry, 'StartPoint') and hasattr(geometry, 'EndPoint'):
                        highlight_obj.Shape = Part.makeLine(geometry.StartPoint, geometry.EndPoint)
                        
            elif geometry.TypeId == "Part::GeomArcOfParabola":
                try:
                    highlight_obj.Shape = Part.Edge(geometry)
                except Exception:
                    # Fallback for arc of parabola
                    if hasattr(geometry, 'StartPoint') and hasattr(geometry, 'EndPoint'):
                        highlight_obj.Shape = Part.makeLine(geometry.StartPoint, geometry.EndPoint)
                        
            elif geometry.TypeId == "Part::GeomBSplineCurve":
                try:
                    highlight_obj.Shape = Part.Edge(geometry)
                except Exception:
                    # Fallback: create line between start and end points
                    if hasattr(geometry, 'StartPoint') and hasattr(geometry, 'EndPoint'):
                        highlight_obj.Shape = Part.makeLine(geometry.StartPoint, geometry.EndPoint)
                        
            else:
                # Generic fallback for any other geometry type
                try:
                    highlight_obj.Shape = Part.Edge(geometry)
                except Exception:
                    # Last resort: try to get start/end points and create a line
                    if hasattr(geometry, 'StartPoint') and hasattr(geometry, 'EndPoint'):
                        highlight_obj.Shape = Part.makeLine(geometry.StartPoint, geometry.EndPoint)
                    else:
                        # If all else fails, return None to clean up the object
                        doc.removeObject(highlight_obj.Name)
                        return None

            return highlight_obj

        except Exception:
            # Clean up failed object
            try:
                doc.removeObject(highlight_obj.Name)
            except Exception:
                pass
            return None

    def _setup_highlight_appearance(self, highlight_obj: Any) -> None:
        """Setup the appearance of a highlight object."""
        view_obj = highlight_obj.ViewObject
        
        # Check if this is a vertex (point) or edge/face geometry
        shape = highlight_obj.Shape
        is_vertex = hasattr(shape, 'ShapeType') and shape.ShapeType == 'Vertex'
        
        if is_vertex:
            # Use point properties for vertex highlighting
            view_obj.PointColor = self.current_color
            view_obj.PointSize = 15.0  # Make points much larger for visibility
        else:
            # Use line properties for edge/face highlighting
            view_obj.ShapeColor = self.current_color
            view_obj.LineColor = self.current_color
            view_obj.LineWidth = Config.HIGHLIGHT_LINE_WIDTH
            
        view_obj.Transparency = Config.HIGHLIGHT_TRANSPARENCY

    def clear_highlights(self) -> None:
        """Clear all highlight objects."""
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


class SketchAnalysisData:
    """Container for sketch analysis results."""

    def __init__(self):
        self.zero_length: List[Dict[str, Any]] = []
        self.duplicates: List[List[Dict[str, Any]]] = []
        self.non_coincident: List[Dict[str, Any]] = []
        self.problematic: List[Dict[str, Any]] = []

    def get_total_issues(self) -> int:
        """Get the total number of issues found."""
        return (len(self.zero_length) +
                len(self.duplicates) +
                len(self.non_coincident) +
                len(self.problematic))

    def update(self, analyzer: GeometryAnalyzer) -> None:
        """Update analysis data from analyzer."""
        self.zero_length = analyzer.find_zero_length_lines()
        self.duplicates = analyzer.find_duplicate_geometry()
        self.non_coincident = analyzer.find_non_coincident_vertices()
        self.problematic = analyzer.find_problematic_intersections()


class SketcherWireDoctorWidget(QtGui.QWidget):
    """Main widget for the SketcherWireDoctor docker."""

    def __init__(self):
        super().__init__()
        self.sketch: Optional[Any] = None
        self.analyzer: Optional[GeometryAnalyzer] = None
        self.highlighter: Optional[GeometryHighlighter] = None
        self.analysis_data = SketchAnalysisData()
        self.current_color_index = 3  # Start with yellow instead of red

        self._setup_ui()
        self.analyze_sketch()

    def _setup_ui(self) -> None:
        """Setup the user interface."""
        layout = QtGui.QVBoxLayout(self)

        # Status label
        self.status_label = QtGui.QLabel("Initializing...")
        layout.addWidget(self.status_label)

        # Analyze button
        self.analyze_button = QtGui.QPushButton("Re-Analyze Sketch")
        self.analyze_button.clicked.connect(self.analyze_sketch)
        layout.addWidget(self.analyze_button)

        # Create tabs
        self.tab_widget = QtGui.QTabWidget()
        layout.addWidget(self.tab_widget)

        self._setup_tabs()

        # Color selection
        self._setup_color_selector(layout)

        # Initially disable tabs
        self.tab_widget.setEnabled(False)

    def _setup_tabs(self) -> None:
        """Setup all tab widgets by importing and calling tab modules."""
        import sys
        import os
        
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Add to sys.path if not already there
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        
        # Check for required tab modules
        required_modules = [
            'SketcherWireDoctor_Tab1',
            'SketcherWireDoctor_Tab2', 
            'SketcherWireDoctor_Tab3',
            'SketcherWireDoctor_Tab4'
        ]
        
        available_modules = {}
        
        # Check each module
        for module_name in required_modules:
            try:
                module = __import__(module_name)
                available_modules[module_name] = module
            except ImportError:
                pass
        
        # Setup available tabs
        if 'SketcherWireDoctor_Tab1' in available_modules:
            try:
                available_modules['SketcherWireDoctor_Tab1'].setup_zero_length_tab(self)
                print("[SketcherWireDoctor] Loaded: Tab1")
            except Exception as e:
                print(f"[SketcherWireDoctor] ERROR: Failed to load Tab1: {type(e).__name__}: {str(e)}")
                self._setup_missing_tab("Zero-Length Lines")
        else:
            print("[SketcherWireDoctor] ERROR: Failed to load Tab1: Module not found")
            self._setup_missing_tab("Zero-Length Lines")
            
        if 'SketcherWireDoctor_Tab2' in available_modules:
            try:
                available_modules['SketcherWireDoctor_Tab2'].setup_duplicate_tab(self)
                print("[SketcherWireDoctor] Loaded: Tab2")
            except Exception as e:
                print(f"[SketcherWireDoctor] ERROR: Failed to load Tab2: {type(e).__name__}: {str(e)}")
                self._setup_missing_tab("Duplicate Geometry")
        else:
            print("[SketcherWireDoctor] ERROR: Failed to load Tab2: Module not found")
            self._setup_missing_tab("Duplicate Geometry")
            
        if 'SketcherWireDoctor_Tab3' in available_modules:
            try:
                available_modules['SketcherWireDoctor_Tab3'].setup_coincident_tab(self)
                print("[SketcherWireDoctor] Loaded: Tab3")
            except Exception as e:
                print(f"[SketcherWireDoctor] ERROR: Failed to load Tab3: {type(e).__name__}: {str(e)}")
                self._setup_missing_tab("Non-Coincident Vertices")
        else:
            print("[SketcherWireDoctor] ERROR: Failed to load Tab3: Module not found")
            self._setup_missing_tab("Non-Coincident Vertices")
            
        if 'SketcherWireDoctor_Tab4' in available_modules:
            try:
                available_modules['SketcherWireDoctor_Tab4'].setup_intersections_tab(self)
                print("[SketcherWireDoctor] Loaded: Tab4")
            except Exception as e:
                print(f"[SketcherWireDoctor] ERROR: Failed to load Tab4: {type(e).__name__}: {str(e)}")
                self._setup_missing_tab("Problematic Intersections")
        else:
            print("[SketcherWireDoctor] ERROR: Failed to load Tab4: Module not found")
            self._setup_missing_tab("Problematic Intersections")
        
        # Connect hover handlers for all available list widgets
        self._connect_hover_handlers()

    def _setup_missing_tab(self, tab_name: str) -> None:
        """Setup a placeholder tab for missing modules."""
        tab = QtGui.QWidget()
        layout = QtGui.QVBoxLayout(tab)
        
        label = QtGui.QLabel(f"Module not found for {tab_name}\n\n"
                           f"Please ensure SketcherWireDoctor_Tab files are installed.")
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.setStyleSheet("color: red; font-weight: bold;")
        layout.addWidget(label)
        
        self.tab_widget.addTab(tab, tab_name)

    def _connect_hover_handlers(self) -> None:
        """Connect all list widgets to the unified hover handler."""
        list_widgets = [
            getattr(self, 'zero_length_list', None),
            getattr(self, 'duplicate_list', None),
            getattr(self, 'coincident_list', None),
            getattr(self, 'intersections_list', None)
        ]
        
        for list_widget in list_widgets:
            if list_widget:
                try:
                    list_widget.itemEntered.connect(self._on_hover)
                    # Install event filter to catch mouse leave events
                    list_widget.installEventFilter(self)
                except Exception:
                    pass

    def _setup_color_selector(self, layout: QtGui.QVBoxLayout) -> None:
        """Setup color selection widget."""
        color_box = QtGui.QHBoxLayout()
        label = QtGui.QLabel("Highlight Color:")
        color_box.addWidget(label)

        self.color_buttons = []

        for i, color in enumerate(Config.HIGHLIGHT_COLORS):
            button = self._create_color_button(color, i)
            self.color_buttons.append(button)
            color_box.addWidget(button)

        color_box.addStretch()
        layout.addLayout(color_box)

    def _create_color_button(self, color: Tuple[float, float, float], index: int) -> QtGui.QToolButton:
        """Create a single color button."""
        button = QtGui.QToolButton()
        min_w, min_h, max_w, max_h = Config.COLOR_BUTTON_SIZE
        button.setMinimumSize(min_w, min_h)
        button.setMaximumSize(max_w, max_h)

        self._update_color_button_style(button, color, index == self.current_color_index)
        button.clicked.connect(lambda checked=False, c=color, idx=index: self._set_highlight_color(c, idx))

        return button

    def _update_color_button_style(self, button: QtGui.QToolButton, color: Tuple[float, float, float],
                                  is_selected: bool) -> None:
        """Update the style of a color button."""
        rgb = f"rgb({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)})"
        border = "3px solid white" if is_selected else "2px solid black"
        button.setStyleSheet(f"background-color: {rgb}; border: {border};")

    def _get_active_sketch(self) -> Optional[Any]:
        """Get the currently active sketch."""
        edit_obj = Gui.ActiveDocument.getInEdit()
        if edit_obj and hasattr(edit_obj, 'Object'):
            obj = edit_obj.Object
            if hasattr(obj, 'TypeId') and 'Sketch' in obj.TypeId:
                return obj
        return None

    def analyze_sketch(self) -> None:
        """Analyze the current sketch for wire closure issues."""
        self.sketch = self._get_active_sketch()
        if not self.sketch:
            self.status_label.setText("No sketch open for editing")
            self.tab_widget.setEnabled(False)
            return

        self.sketch.recompute()
        App.ActiveDocument.recompute()
        self._setup_analysis_tools()
        self._perform_analysis()
        self._update_ui_with_results()

    def _setup_analysis_tools(self) -> None:
        """Setup analyzer and highlighter."""
        if self.highlighter:
            self.highlighter.clear_highlights()

        self.analyzer = GeometryAnalyzer(self.sketch)
        self.highlighter = GeometryHighlighter(self.sketch)
        self.highlighter.set_color(Config.HIGHLIGHT_COLORS[self.current_color_index])

    def _perform_analysis(self) -> None:
        """Perform the actual analysis."""
        self.analyzer.collect_data()
        self.analysis_data.update(self.analyzer)

    def _update_ui_with_results(self) -> None:
        """Update UI with analysis results."""
        self._populate_all_tabs()
        self.tab_widget.setEnabled(True)

        total_issues = self.analysis_data.get_total_issues()
        self.status_label.setText(f"{total_issues} issues detected in {self.sketch.Label}")

    def _populate_all_tabs(self) -> None:
        """Populate all list widgets with analysis results."""
        import sys
        import os
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
            
        # Try to use tab module functions with individual error handling
        try:
            tab1_module = __import__('SketcherWireDoctor_Tab1')
            tab1_module.populate_zero_length_list(self)
        except ImportError:
            pass  # Tab not available
        except Exception:
            pass
            
        try:
            tab2_module = __import__('SketcherWireDoctor_Tab2')
            tab2_module.populate_duplicate_list(self)
        except ImportError:
            pass  # Tab not available
        except Exception:
            pass
            
        try:
            tab3_module = __import__('SketcherWireDoctor_Tab3')
            tab3_module.populate_tab3_list(self)
        except ImportError:
            pass  # Tab not available
        except Exception:
            pass
            
        try:
            tab4_module = __import__('SketcherWireDoctor_Tab4')
            tab4_module.populate_intersections_list(self)
        except ImportError:
            pass  # Tab not available
        except Exception:
            pass

    # Unified hover handler
    def _on_hover(self, item: QtGui.QListWidgetItem) -> None:
        """Unified hover handler for all list items."""
        if not self.highlighter:
            return
            
        data = item.data(QtCore.Qt.UserRole)
        geo_idx = None
        is_tab1_zero_length = False
        
        # Check if this item is from the zero-length list (Tab1)
        sender_widget = self.sender()
        if sender_widget == getattr(self, 'zero_length_list', None):
            is_tab1_zero_length = True
        
        # Tab1: Direct integer geo_idx
        if isinstance(data, int):
            geo_idx = data
            
        # Tab2: Complex nested structure
        elif isinstance(data, dict):
            if data.get('type') == 'geometry' and 'data' in data:
                geo_idx = data['data'].get('geo_idx')
            # Tab3: Edge type with direct geo_idx
            elif data.get('type') == 'edge':
                geo_idx = data.get('geo_idx')
            # Tab3: Location type - use first vertex
            elif data.get('type') == 'location' and 'data' in data:
                vertex_data = data['data']
                vertices = vertex_data.get('vertices', [])
                if vertices:
                    geo_idx = vertices[0][0]  # First vertex's geometry index
            # Tab4: Direct geo_idx in dict
            elif 'geo_idx' in data:
                geo_idx = data['geo_idx']
        
        # Highlight if we found a valid geometry index
        if isinstance(geo_idx, int) and geo_idx >= 0:
            try:
                if is_tab1_zero_length:
                    # Tab1 items are always zero-length - highlight start vertex
                    self._highlight_zero_length_vertex(geo_idx)
                else:
                    # Normal geometry highlighting
                    self.highlighter.highlight_geometry(geo_idx)
            except Exception:
                pass

    def _highlight_zero_length_vertex(self, geo_idx: int) -> None:
        """Highlight the start vertex of a zero-length line."""
        try:
            self.highlighter.clear_highlights()
            
            doc = App.ActiveDocument
            if not doc:
                return

            geometry = self.sketch.Geometry[geo_idx]
            if geometry.TypeId == "Part::GeomLineSegment":
                # Create a vertex at the start point location
                highlight_obj = doc.addObject("Part::Feature", f"Highlight_ZeroLength_Geo{geo_idx}")
                vertex = Part.Vertex(App.Vector(geometry.StartPoint.x, geometry.StartPoint.y, 0))
                highlight_obj.Shape = vertex
                
                # Apply vertex appearance
                view_obj = highlight_obj.ViewObject
                view_obj.PointColor = self.highlighter.current_color
                view_obj.PointSize = 15.0  # Same as other vertex highlighting
                view_obj.Transparency = Config.HIGHLIGHT_TRANSPARENCY
                
                self.highlighter.highlight_objects.append(highlight_obj)
                doc.recompute()
                Gui.updateGui()
                
        except Exception:
            pass

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        """Event filter to handle mouse leave events for clearing highlights."""
        if event.type() == QtCore.QEvent.Leave and self.highlighter:
            # Check if the object is one of our list widgets
            list_widgets = [
                getattr(self, 'zero_length_list', None),
                getattr(self, 'duplicate_list', None),
                getattr(self, 'coincident_list', None),
                getattr(self, 'intersections_list', None)
            ]
            
            if obj in list_widgets:
                self.highlighter.clear_highlights()
        
        # Let the event continue to be processed normally
        return False

    def _on_hover_exit(self, item: QtGui.QListWidgetItem) -> None:
        """Clear highlights when mouse exits any list item."""
        if self.highlighter:
            self.highlighter.clear_highlights()

    # Selection handlers
    def _on_duplicate_selected(self, item: QtGui.QListWidgetItem) -> None:
        """Handle selection of duplicate item."""
        self._on_hover(item)

    def _on_coincident_selected(self, item: QtGui.QListWidgetItem) -> None:
        """Handle selection of coincident item."""
        data = item.data(QtCore.Qt.UserRole)
        if data and self.highlighter:
            if data.get('type') == 'edge':
                self.highlighter.highlight_geometry(data['geo_idx'])
            elif data.get('type') == 'location':
                vertex_data = data['data']
                if vertex_data['vertices']:
                    self.highlighter.highlight_geometry(vertex_data['vertices'][0][0])

    def _on_intersection_selected(self, item: QtGui.QListWidgetItem) -> None:
        """Handle selection of intersection item."""
        self._on_hover(item)

    # Color management
    def _set_highlight_color(self, color: Tuple[float, float, float], index: int) -> None:
        """Set the highlight color and update button appearance."""
        self.current_color_index = index

        if self.highlighter:
            self.highlighter.set_color(color)

        # Update button styles
        for i, button in enumerate(self.color_buttons):
            button_color = Config.HIGHLIGHT_COLORS[i]
            self._update_color_button_style(button, button_color, i == index)

    # Shared utility methods for tabs
    def _delete_geometries(self, indices: List[int], transaction_name: str) -> None:
        """Delete multiple geometries in a single transaction."""
        if not indices:
            return

        self.sketch.Document.openTransaction(transaction_name)

        try:
            # Sort in reverse order to avoid index shifting
            for geo_idx in sorted(indices, reverse=True):
                self.sketch.delGeometry(geo_idx)

            # Add sketch recompute after geometry changes
            self.sketch.recompute()
            self.sketch.Document.recompute()
            Gui.updateGui()

            self.sketch.Document.commitTransaction()
            self.analyze_sketch()  # Re-analyze

        except Exception:
            self.sketch.Document.abortTransaction()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Handle widget close event."""
        if self.highlighter:
            self.highlighter.clear_highlights()
        event.accept()


class SketcherWireDoctorDockWidget(QtGui.QDockWidget):
    """Docker widget for the SketcherWireDoctor macro."""

    def __init__(self):
        super().__init__()
        self._setup_dock_properties()
        self._setup_main_widget()

    def _setup_dock_properties(self) -> None:
        """Setup dock widget properties."""
        self.setWindowTitle("SketcherWireDoctor")
        self.setObjectName("SketcherWireDoctor")

        self.setFeatures(QtGui.QDockWidget.DockWidgetMovable |
                        QtGui.QDockWidget.DockWidgetFloatable |
                        QtGui.QDockWidget.DockWidgetClosable)
        self.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea |
                            QtCore.Qt.RightDockWidgetArea)

    def _setup_main_widget(self) -> None:
        """Setup the main widget."""
        self.main_widget = SketcherWireDoctorWidget()
        self.setWidget(self.main_widget)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Handle dock close event."""
        self.main_widget.closeEvent(event)

        # Clear global reference when closing
        global sketcher_wire_doctor_dock
        sketcher_wire_doctor_dock = None

        event.accept()


class DockWidgetManager:
    """Manages dock widget lifecycle and cleanup."""

    @staticmethod
    def cleanup_existing_docks() -> None:
        """Clean up any existing SketcherWireDoctor docks."""
        main_window = Gui.getMainWindow()
        existing_docks = DockWidgetManager._find_existing_docks(main_window)

        global sketcher_wire_doctor_dock
        if sketcher_wire_doctor_dock:
            parent_dock = DockWidgetManager._find_parent_dock(sketcher_wire_doctor_dock)
            if parent_dock and parent_dock not in existing_docks:
                existing_docks.append(parent_dock)

        # Clean up existing dockers
        for dock in existing_docks:
            DockWidgetManager._cleanup_single_dock(dock, main_window)

        sketcher_wire_doctor_dock = None
        QtGui.QApplication.processEvents()  # Force Qt to process pending deletions

    @staticmethod
    def _find_existing_docks(main_window: QtGui.QMainWindow) -> List[QtGui.QDockWidget]:
        """Find all existing SketcherWireDoctor dock widgets."""
        existing_docks = []

        # Find by window title and object name
        for dock in main_window.findChildren(QtGui.QDockWidget):
            if (dock.windowTitle() in ["SketcherWireDoctor", "SketcherWireDoctor v0.71"] or
                dock.objectName() == "SketcherWireDoctor"):
                existing_docks.append(dock)

        # Find by widget type
        for dock in main_window.findChildren(QtGui.QDockWidget):
            widget = dock.widget()
            if widget and isinstance(widget, SketcherWireDoctorWidget):
                if dock not in existing_docks:
                    existing_docks.append(dock)

        return existing_docks

    @staticmethod
    def _find_parent_dock(widget: QtGui.QWidget) -> Optional[QtGui.QDockWidget]:
        """Find the parent dock widget of a given widget."""
        parent_dock = widget.parent()
        while parent_dock and not isinstance(parent_dock, QtGui.QDockWidget):
            parent_dock = parent_dock.parent()
        return parent_dock

    @staticmethod
    def _cleanup_single_dock(dock: QtGui.QDockWidget, main_window: QtGui.QMainWindow) -> None:
        """Clean up a single dock widget."""
        try:
            # Clean up any active highlighting
            widget = dock.widget()
            if widget and hasattr(widget, 'closeEvent'):
                widget.closeEvent(QtGui.QCloseEvent())

            # Remove from main window first, then clean up
            main_window.removeDockWidget(dock)
            dock.setParent(None)
            dock.close()
            dock.deleteLater()

        except Exception:
            pass

    @staticmethod
    def create_new_dock() -> SketcherWireDoctorDockWidget:
        """Create and setup a new dock widget."""
        main_window = Gui.getMainWindow()
        dock = SketcherWireDoctorDockWidget()
        main_window.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
        dock.show()
        return dock


# Global reference to prevent garbage collection
sketcher_wire_doctor_dock: Optional[SketcherWireDoctorDockWidget] = None


def show_sketcher_wire_doctor() -> None:
    """Show the SketcherWireDoctor docker."""
    global sketcher_wire_doctor_dock

    print("[SketcherWireDoctor] Starting...")

    try:
        # Clear module cache for fresh reload
        _clear_module_cache()
        
        # Clean up existing docks
        DockWidgetManager.cleanup_existing_docks()

        # Create new dock
        dock = DockWidgetManager.create_new_dock()
        sketcher_wire_doctor_dock = dock

    except Exception as e:
        print(f"[SketcherWireDoctor] ERROR: Failed to launch: {str(e)}")


def _clear_module_cache() -> None:
    """Clear SketcherWireDoctor module cache for fresh reload."""
    import sys
    
    # Clear all SketcherWireDoctor modules from cache
    modules_to_clear = [name for name in sys.modules.keys() if name.startswith('SketcherWireDoctor')]
    
    for module_name in modules_to_clear:
        try:
            del sys.modules[module_name]
        except Exception:
            pass  # Module might not exist or be locked


# Main execution
if __name__ == "__main__":
    show_sketcher_wire_doctor()