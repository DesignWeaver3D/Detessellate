#!/usr/bin/env python3
"""
SketcherWireDoctor - Tab1: Zero-Length Lines

Handles the zero-length lines detection tab functionality.

Author: Original implementation refactored for best practices
Version: 0.71 (Face-based area calculation with universal geometry support)
"""

import FreeCAD as App
from PySide import QtCore, QtGui
from typing import Any, List, Dict


# Local constants to avoid circular imports
TOLERANCE = 1e-6

GEOMETRY_TYPE_MAP = {
    'Part::GeomLineSegment': 'Line',
    'Part::GeomArcOfCircle': 'Arc',
    'Part::GeomCircle': 'Circle',
    'Part::GeomBSplineCurve': 'BSpline',
    'Part::GeomEllipse': 'Ellipse'
}


def get_geometry_name(geo_idx: int, geometry: Any) -> str:
    """Get the FreeCAD display name for geometry (1-based indexing)."""
    geo_type = GEOMETRY_TYPE_MAP.get(geometry.TypeId, 'Geometry')
    return f"{geo_idx + 1}-{geo_type}"


def find_zero_length_lines(analyzer: Any) -> List[Dict[str, Any]]:
    """Find all zero-length line segments (including construction)."""
    zero_length = []

    # Check ALL geometry for zero-length lines (including construction)
    for geo_idx, geometry in analyzer.all_geometry:
        if geometry.TypeId == "Part::GeomLineSegment":
            length = geometry.StartPoint.distanceToPoint(geometry.EndPoint)
            if length < TOLERANCE:
                zero_length.append({
                    'geo_idx': geo_idx,
                    'geometry': geometry,
                    'length': length,
                    'is_construction': analyzer.sketch.getConstruction(geo_idx)
                })

    return zero_length


def setup_zero_length_tab(widget: Any) -> None:
    """Setup the zero-length lines tab."""
    tab = QtGui.QWidget()
    layout = QtGui.QVBoxLayout(tab)

    widget.zero_length_list = QtGui.QListWidget()
    widget.zero_length_list.itemEntered.connect(widget._on_hover)
    layout.addWidget(widget.zero_length_list)

    delete_all_btn = QtGui.QPushButton("Delete All Zero-Length Lines")
    delete_all_btn.clicked.connect(lambda: delete_all_zero_length(widget))
    layout.addWidget(delete_all_btn)

    widget.tab_widget.addTab(tab, "Zero-Length Lines")


def populate_zero_length_list(widget: Any) -> None:
    """Populate the zero-length lines list."""
    widget.zero_length_list.clear()

    for item in widget.analysis_data.zero_length:
        geo_name = get_geometry_name(item['geo_idx'], item['geometry'])
        construction_tag = " (Construction)" if item.get('is_construction', False) else ""
        list_item = QtGui.QListWidgetItem(f"{geo_name}: Length {item['length']:.6f}{construction_tag}")
        list_item.setData(QtCore.Qt.UserRole, item['geo_idx'])
        widget.zero_length_list.addItem(list_item)


def delete_all_zero_length(widget: Any) -> None:
    """Delete all zero-length lines (including construction)."""
    if not widget.analysis_data.zero_length:
        return

    indices = [item['geo_idx'] for item in widget.analysis_data.zero_length]
    widget._delete_geometries(indices, "Delete Zero-Length Lines")