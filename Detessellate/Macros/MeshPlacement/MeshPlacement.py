# -*- coding: utf-8 -*-
"""
***************************************************************************
*   FreeCAD Addon Manager Macro                                           *
*   Name:        MeshPlacement                                            *
*   Author:      NSUBB (aka DesignWeaver)                                *
*   License:     GNU GPL v3.0                                             *
*   Version:     1.0.0                                                    *
*   Date:        2025-11-21                                               *
*   FreeCAD:     1.0.2 or later                                           *
*   Description:                                                          *
*       A FreeCAD macro to position one or more selected meshes at the    *
*       global origin based on their bounding box.                        *
*                                                                         *
*   This macro provides a dock widget with buttons to center or align     *
*   meshes along X, Y, Z axes or combinations (XY, XYZ). Works on one     *
*   or multiple selected meshes, with undo safety and clean UI grouping.  *
***************************************************************************
"""

__title__   = "MeshPlacement"
__author__  = "NSUBB (aka DesignWeaver)"
__license__ = "GNU GPL v3.0"
__version__ = "1.0.0"
__date__    = "2025-11-21"
__FreeCAD__ = "1.0.2 or later"
__url__     = "https://github.com/NSUBB/MeshPlacement"
__doc__     = "A FreeCAD macro to position selected meshes at the global origin using bounding box alignment."

import FreeCAD, FreeCADGui
from PySide import QtGui, QtCore

class MeshPlacementDock(QtGui.QDockWidget):
    def __init__(self):
        super(MeshPlacementDock, self).__init__("MeshPlacement")
        self.setObjectName("MeshPlacement")
        self.setWindowTitle("MeshPlacement")
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        self.widget = QtGui.QWidget()
        main_layout = QtGui.QVBoxLayout()

        # --- Center XYZ ---
        main_layout.addWidget(QtGui.QPushButton("Center XYZ", clicked=self.centerXYZ))

        # --- Divider line ---
        line = QtGui.QFrame()
        line.setFrameShape(QtGui.QFrame.HLine)
        line.setFrameShadow(QtGui.QFrame.Sunken)
        main_layout.addWidget(line)

        # --- Z stack group ---
        z_group = QtGui.QGroupBox("Z Controls")
        z_layout = QtGui.QVBoxLayout()
        z_layout.addWidget(QtGui.QPushButton("Align Bottom", clicked=self.alignBottom))
        z_layout.addWidget(QtGui.QPushButton("Center Z", clicked=self.centerZ))
        z_layout.addWidget(QtGui.QPushButton("Align Top", clicked=self.alignTop))
        z_group.setLayout(z_layout)
        main_layout.addWidget(z_group)

        # --- XY cross group ---
        xy_group = QtGui.QGroupBox("XY Controls")
        xy_layout = QtGui.QGridLayout()

        # Row 0: Center XY
        xy_layout.addWidget(QtGui.QPushButton("Center XY", clicked=self.centerXY), 0, 1)

        # Row 1: Back - Center Y - Front
        xy_layout.addWidget(QtGui.QPushButton("Align Back", clicked=self.alignBack), 1, 0)
        xy_layout.addWidget(QtGui.QPushButton("Center Y", clicked=self.centerY), 1, 1)
        xy_layout.addWidget(QtGui.QPushButton("Align Front", clicked=self.alignFront), 1, 2)

        # Row 2: Right - Center X - Left
        xy_layout.addWidget(QtGui.QPushButton("Align Right", clicked=self.alignRight), 2, 0)
        xy_layout.addWidget(QtGui.QPushButton("Center X", clicked=self.centerX), 2, 1)
        xy_layout.addWidget(QtGui.QPushButton("Align Left", clicked=self.alignLeft), 2, 2)

        xy_group.setLayout(xy_layout)
        main_layout.addWidget(xy_group)

        self.widget.setLayout(main_layout)
        self.setWidget(self.widget)

    def getMeshes(self):
        sel = FreeCADGui.Selection.getSelection()
        meshes = [obj for obj in sel if obj.TypeId.startswith("Mesh::Feature")]
        if not meshes:
            FreeCAD.Console.PrintError("Select one or more Mesh objects.\n")
        return meshes

    # --- Centering methods ---
    def centerXYZ(self): self._applyShift("xyz")
    def centerXY(self):  self._applyShift("xy")
    def centerX(self):   self._applyShift("x")
    def centerY(self):   self._applyShift("y")
    def centerZ(self):   self._applyShift("z")

    # --- Alignment methods ---
    def alignTop(self):    self._applyAlign("top")
    def alignBottom(self): self._applyAlign("bottom")
    def alignLeft(self):   self._applyAlign("left")
    def alignRight(self):  self._applyAlign("right")
    def alignFront(self):  self._applyAlign("front")
    def alignBack(self):   self._applyAlign("back")

    def _applyShift(self, mode):
        meshes = self.getMeshes()
        if not meshes: return
        doc = FreeCAD.ActiveDocument
        doc.openTransaction(f"Center {mode.upper()}")

        for obj in meshes:
            bb = obj.Mesh.BoundBox
            cx, cy, cz = (bb.XMin+bb.XMax)/2, (bb.YMin+bb.YMax)/2, (bb.ZMin+bb.ZMax)/2
            x,y,z = obj.Placement.Base
            if mode=="xyz": obj.Placement.Base = FreeCAD.Vector(x-cx,y-cy,z-cz)
            if mode=="xy":  obj.Placement.Base = FreeCAD.Vector(x-cx,y-cy,z)
            if mode=="x":   obj.Placement.Base = FreeCAD.Vector(x-cx,y,z)
            if mode=="y":   obj.Placement.Base = FreeCAD.Vector(x,y-cy,z)
            if mode=="z":   obj.Placement.Base = FreeCAD.Vector(x,y,z-cz)

        doc.recompute()
        doc.commitTransaction()
        FreeCAD.Console.PrintMessage(f"Meshes recentered ({mode.upper()}).\n")

    def _applyAlign(self, mode):
        meshes = self.getMeshes()
        if not meshes: return
        doc = FreeCAD.ActiveDocument
        doc.openTransaction(f"Align {mode.capitalize()}")

        for obj in meshes:
            bb = obj.Mesh.BoundBox
            x,y,z = obj.Placement.Base
            if mode=="top":    obj.Placement.Base = FreeCAD.Vector(x,y,z-bb.ZMax)
            if mode=="bottom": obj.Placement.Base = FreeCAD.Vector(x,y,z-bb.ZMin)
            if mode=="left":   obj.Placement.Base = FreeCAD.Vector(x-bb.XMin,y,z)
            if mode=="right":  obj.Placement.Base = FreeCAD.Vector(x-bb.XMax,y,z)
            if mode=="front":  obj.Placement.Base = FreeCAD.Vector(x,y-bb.YMin,z)
            if mode=="back":   obj.Placement.Base = FreeCAD.Vector(x,y-bb.YMax,z)

        doc.recompute()
        doc.commitTransaction()
        FreeCAD.Console.PrintMessage(f"Meshes aligned ({mode}).\n")

# --- Ensure only one dock instance ---
mw = FreeCADGui.getMainWindow()
for dock in mw.findChildren(QtGui.QDockWidget):
    if dock.objectName() == "MeshPlacement":
        mw.removeDockWidget(dock)
        dock.deleteLater()

dock = MeshPlacementDock()
mw.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
