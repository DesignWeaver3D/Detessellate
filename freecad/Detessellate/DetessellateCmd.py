# SPDX-License-Identifier: LGPL-2.1-or-later
# SPDX-FileCopyrightText: 2024 DesignWeaver3D
# SPDX-FileNotice: Part of the Detessellate addon.

from pathlib import Path
import sys

import FreeCAD
import FreeCADGui
from PySide6 import QtGui, QtCore, QtWidgets
from freecad.Detessellate.Misc.Resources import asIcon


# ---------------------------------------------------------------------------
# Mesh Tools
# ---------------------------------------------------------------------------

class MeshPlacementCommand:
    base_path: Path = Path(__file__).parent / "Macros/MeshPlacement"

    def GetResources(self):
        return {
            'Pixmap': asIcon('MeshPlacement'),
            'MenuText': 'Mesh Placement',
            'ToolTip': 'Center and align meshes at origin'
        }

    def Activated(self):
        if str(self.base_path) not in sys.path:
            sys.path.append(str(self.base_path))
        try:
            import importlib
            if 'MeshPlacement' in sys.modules:
                import MeshPlacement
                importlib.reload(MeshPlacement)
            else:
                import MeshPlacement
        except Exception as e:
            FreeCAD.Console.PrintError(f"Error running MeshPlacement: {e}\n")
            import traceback
            traceback.print_exc()

    def IsActive(self):
        return FreeCAD.ActiveDocument is not None


class MeshToBodyCommand:
    base_path: Path = Path(__file__).parent / "Macros/MeshToBody"

    def GetResources(self):
        return {
            'Pixmap': asIcon('MeshToBody'),
            'MenuText': 'Mesh To Body',
            'ToolTip': 'Convert meshes to solid body'
        }

    def Activated(self):
        if str(self.base_path) not in sys.path:
            sys.path.append(str(self.base_path))
        try:
            import importlib
            if 'MeshToBody' in sys.modules:
                import MeshToBody
                importlib.reload(MeshToBody)
            else:
                import MeshToBody
            MeshToBody.run_unified_macro(auto_mode=True)
        except Exception as e:
            FreeCAD.Console.PrintError(f"Error running MeshToBody: {e}\n")
            import traceback
            traceback.print_exc()

    def IsActive(self):
        return FreeCAD.ActiveDocument is not None


# ---------------------------------------------------------------------------
# Sketch Tools
# ---------------------------------------------------------------------------

class CoplanarSketchCommand:
    base_path: Path = Path(__file__).parent / "Macros/CoplanarSketch"

    def GetResources(self):
        return {
            'Pixmap': asIcon('CoplanarSketch'),
            'MenuText': 'Coplanar Sketch',
            'ToolTip': 'Create sketches from selected coplanar edges or faces from a tessellated solid.'
        }

    def Activated(self):
        if str(self.base_path) not in sys.path:
            sys.path.append(str(self.base_path))
        try:
            import importlib
            if 'CoplanarSketch' in sys.modules:
                import CoplanarSketch
                importlib.reload(CoplanarSketch)
            else:
                import CoplanarSketch
        except Exception as e:
            FreeCAD.Console.PrintError(f"Error running CoplanarSketch: {e}\n")
            import traceback
            traceback.print_exc()

    def IsActive(self):
        return FreeCAD.ActiveDocument is not None


class PointPlaneSketchCommand:
    base_path: Path = Path(__file__).parent / "Macros/PointPlaneSketch"

    def GetResources(self):
        return {
            'Pixmap': asIcon('PointPlaneSketch'),
            'MenuText': 'Point Plane Sketch',
            'ToolTip': 'Create sketch from points and their derived plane'
        }

    def Activated(self):
        if str(self.base_path) not in sys.path:
            sys.path.append(str(self.base_path))
        try:
            import importlib
            if 'PointPlaneSketch' in sys.modules:
                import PointPlaneSketch
                importlib.reload(PointPlaneSketch)
            else:
                import PointPlaneSketch
            PointPlaneSketch.show_point_cloud_plane_sketch()
        except Exception as e:
            FreeCAD.Console.PrintError(f"Error running PointPlaneSketch: {e}\n")
            import traceback
            traceback.print_exc()

    def IsActive(self):
        return FreeCAD.ActiveDocument is not None


class SketchReProfileCommand:
    base_path: Path = Path(__file__).parent / "Macros/SketchReProfile"

    def GetResources(self):
        return {
            'Pixmap': asIcon('SketchReProfile'),
            'MenuText': 'Sketch ReProfile',
            'ToolTip': 'Reprocess sketch profiles - converts construction lines to circles, arcs, and splines'
        }

    def Activated(self):
        if str(self.base_path) not in sys.path:
            sys.path.append(str(self.base_path))
        try:
            import importlib
            if 'SketchReProfile' in sys.modules:
                import SketchReProfile
                importlib.reload(SketchReProfile)
            else:
                import SketchReProfile
            SketchReProfile.main()
        except Exception as e:
            FreeCAD.Console.PrintError(f"Error running SketchReProfile: {e}\n")
            import traceback
            traceback.print_exc()

    def IsActive(self):
        return FreeCAD.ActiveDocument is not None


class ConstrainAllPointOnPointCommand:
    base_path: Path = Path(__file__).parent / "Macros/ConstrainAllPointOnPoint"

    def GetResources(self):
        return {
            'Pixmap': asIcon('ConstrainAllPointOnPoint'),
            'MenuText': 'Constrain All Point-On-Point',
            'ToolTip': 'Automatically add missing coincident constraints using built-in FreeCAD detection'
        }

    def Activated(self):
        if str(self.base_path) not in sys.path:
            sys.path.append(str(self.base_path))
        try:
            import importlib
            if 'ConstrainAllPointOnPoint' in sys.modules:
                import ConstrainAllPointOnPoint
                importlib.reload(ConstrainAllPointOnPoint)
            else:
                import ConstrainAllPointOnPoint
            ConstrainAllPointOnPoint.main()
        except Exception as e:
            FreeCAD.Console.PrintError(f"Error running ConstrainAllPointOnPoint: {e}\n")
            import traceback
            traceback.print_exc()

    def IsActive(self):
        try:
            doc = FreeCADGui.activeDocument()
            if doc is None:
                return False
            edit_obj = doc.getInEdit()
            if edit_obj is None:
                return False
            if hasattr(edit_obj, 'Object'):
                obj = edit_obj.Object
                return hasattr(obj, 'TypeId') and 'Sketch' in obj.TypeId
            return False
        except:
            return False


class SketcherWireDoctorCommand:
    base_path: Path = Path(__file__).parent / "Macros/SketcherWireDoctor"

    def GetResources(self):
        return {
            'Pixmap': asIcon('SketcherWireDoctor'),
            'MenuText': 'Sketcher Wire Doctor',
            'ToolTip': 'Fix sketch wire not closed issues'
        }

    def Activated(self):
        if str(self.base_path) not in sys.path:
            sys.path.append(str(self.base_path))
        try:
            import importlib
            if 'SketcherWireDoctor_Main' in sys.modules:
                import SketcherWireDoctor_Main
                importlib.reload(SketcherWireDoctor_Main)
            else:
                import SketcherWireDoctor_Main
            SketcherWireDoctor_Main.show_sketcher_wire_doctor()
        except Exception as e:
            FreeCAD.Console.PrintError(f"Error running SketcherWireDoctor: {e}\n")
            import traceback
            traceback.print_exc()

    def IsActive(self):
        return FreeCAD.ActiveDocument is not None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

class EdgeLoopSelectorCommand:
    base_path: Path = Path(__file__).parent / "Macros/EdgeLoopSelector"

    def GetResources(self):
        return {
            'Pixmap': asIcon('EdgeLoopSelector'),
            'MenuText': 'Edge Loop Selector',
            'ToolTip': 'Select connected edge loops from sketches or faces'
        }

    def Activated(self):
        if str(self.base_path) not in sys.path:
            sys.path.append(str(self.base_path))
        try:
            import importlib
            if 'EdgeLoopSelector' in sys.modules:
                import EdgeLoopSelector
                importlib.reload(EdgeLoopSelector)
            else:
                import EdgeLoopSelector
        except Exception as e:
            FreeCAD.Console.PrintError(f"Error running EdgeLoopSelector: {e}\n")
            import traceback
            traceback.print_exc()

    def IsActive(self):
        return FreeCAD.ActiveDocument is not None


class EdgeLoopToSketchCommand:
    base_path: Path = Path(__file__).parent / "Macros/EdgeLoopToSketch"

    def GetResources(self):
        return {
            'Pixmap': asIcon('EdgeLoopToSketch'),
            'MenuText': 'Edge Loop to Sketch',
            'ToolTip': 'Convert selected coplanar faces or edges to parametric sketch'
        }

    def Activated(self):
        if str(self.base_path) not in sys.path:
            sys.path.append(str(self.base_path))
        try:
            import importlib
            if 'EdgeLoopToSketch' in sys.modules:
                import EdgeLoopToSketch
                importlib.reload(EdgeLoopToSketch)
            else:
                import EdgeLoopToSketch
        except Exception as e:
            FreeCAD.Console.PrintError(f"Error running EdgeLoopToSketch: {e}\n")
            import traceback
            traceback.print_exc()

    def IsActive(self):
        return FreeCAD.ActiveDocument is not None


class ReconstructSolidCommand:
    base_path: Path = Path(__file__).parent / "Macros/ReconstructSolid"

    def GetResources(self):
        return {
            'Pixmap': asIcon('ReconstructSolid'),
            'MenuText': 'Reconstruct Solid',
            'ToolTip': 'Reconstruct a simple solid to change its geometric origin'
        }

    def Activated(self):
        if str(self.base_path) not in sys.path:
            sys.path.append(str(self.base_path))
        try:
            import importlib
            if 'ReconstructSolid' in sys.modules:
                import ReconstructSolid
                importlib.reload(ReconstructSolid)
            else:
                import ReconstructSolid
        except Exception as e:
            FreeCAD.Console.PrintError(f"Error running ReconstructSolid: {e}\n")
            import traceback
            traceback.print_exc()

    def IsActive(self):
        return FreeCAD.ActiveDocument is not None


class TopoMatchSelectorCommand:
    base_path: Path = Path(__file__).parent / "Macros/TopoMatchSelector"

    def GetResources(self):
        return {
            'Pixmap': asIcon('TopoMatchSelector'),
            'MenuText': 'Topo Match Selector',
            'ToolTip': 'Select topology matching elements'
        }

    def Activated(self):
        if str(self.base_path) not in sys.path:
            sys.path.append(str(self.base_path))
        try:
            import importlib
            if 'TopoMatchSelector' in sys.modules:
                import TopoMatchSelector
                importlib.reload(TopoMatchSelector)
            else:
                import TopoMatchSelector
            TopoMatchSelector.create_topo_match_selector()
        except Exception as e:
            FreeCAD.Console.PrintError(f"Error running TopoMatchSelector: {e}\n")
            import traceback
            traceback.print_exc()

    def IsActive(self):
        return FreeCAD.ActiveDocument is not None


class VarSetUpdateCommand:
    base_path: Path = Path(__file__).parent / "Macros/VarSet-Update"

    def GetResources(self):
        return {
            'Pixmap': asIcon('VarSetUpdate'),
            'MenuText': 'VarSet Update',
            'ToolTip': 'Update VarSet Properties'
        }

    def Activated(self):
        if str(self.base_path) not in sys.path:
            sys.path.append(str(self.base_path))
        try:
            import importlib
            if 'VarSetUpdate' in sys.modules:
                import VarSetUpdate
                importlib.reload(VarSetUpdate)
            else:
                import VarSetUpdate
        except Exception as e:
            FreeCAD.Console.PrintError(f"Error running VarSet Update: {e}\n")
            import traceback
            traceback.print_exc()

    def IsActive(self):
        return FreeCAD.ActiveDocument is not None


# ---------------------------------------------------------------------------
# Toolbar Creators
# ---------------------------------------------------------------------------

class CreateSketchToolbarCommand:
    wb_path: Path = Path(__file__).parent

    def GetResources(self):
        return {
            'Pixmap': '',
            'MenuText': 'Create Sketch Toolbar',
            'ToolTip': 'Create a toolbar with sketch tools that appears only in Sketcher workbench'
        }

    def Activated(self):
        try:
            mw = FreeCADGui.getMainWindow()
            toolbar_name = "Detessellate_Sketch_Tools"

            for toolbar in mw.findChildren(QtWidgets.QToolBar):
                if toolbar.objectName() == toolbar_name:
                    FreeCAD.Console.PrintMessage("Detessellate Sketch Tools toolbar already exists.\n")
                    return

            custom_toolbar = QtWidgets.QToolBar("Detessellate Sketch Tools", mw)
            custom_toolbar.setObjectName(toolbar_name)
            mw.addToolBar(QtCore.Qt.TopToolBarArea, custom_toolbar)

            self._add_button(custom_toolbar, 'SketchReProfile', "Sketch ReProfile",
                "<b>SketchReProfile</b><br><br>"
                "Draws lines, circles, arcs, and splines over construction geometry<br><br>"
                "<i>Detessellate_SketchReProfile</i>",
                lambda: self._run_macro('SketchReProfile', 'SketchReProfile', 'main'))
            self._add_button(custom_toolbar, 'ConstrainAllPointOnPoint', "Constrain All Point-On-Point",
                "<b>Constrain All Point-On-Point</b><br><br>"
                "Automatically add missing coincident constraints<br><br>"
                "<i>Detessellate_ConstrainAllPointOnPoint</i>",
                lambda: self._run_macro('ConstrainAllPointOnPoint', 'ConstrainAllPointOnPoint', 'main'))
            self._add_button(custom_toolbar, 'SketcherWireDoctor', "Sketcher Wire Doctor",
                "<b>SketcherWireDoctor</b><br><br>"
                "Detects sketch issues and provides repair tools<br><br>"
                "<i>Detessellate_SketcherWireDoctor</i>",
                lambda: self._run_macro('SketcherWireDoctor', 'SketcherWireDoctor_Main', 'show_sketcher_wire_doctor'))

            self._connect_workbench_toggle(custom_toolbar, "SketcherWorkbench",
                                           '_detessellate_sketch_toolbars')

            current_wb = FreeCADGui.activeWorkbench()
            custom_toolbar.setVisible(
                current_wb and current_wb.__class__.__name__ == "SketcherWorkbench")

        except Exception as e:
            FreeCAD.Console.PrintError(f"Error creating sketch toolbar: {e}\n")
            import traceback
            traceback.print_exc()

    def _add_button(self, toolbar, icon_name, label, tooltip, callback):
        try:
            icon = QtGui.QIcon(asIcon(icon_name))
            action = QtGui.QAction(icon, label, toolbar)
            action.setToolTip(tooltip)
            action.triggered.connect(callback)
            toolbar.addAction(action)
        except Exception as e:
            FreeCAD.Console.PrintError(f"Error adding {label} button: {e}\n")

    def _run_macro(self, macro_folder, module_name, func_name):
        macro_path = self.wb_path / "Macros" / macro_folder
        try:
            if str(macro_path) not in sys.path:
                sys.path.append(str(macro_path))
            import importlib
            mod = sys.modules.get(module_name)
            if mod:
                importlib.reload(mod)
            else:
                import importlib as il
                mod = il.import_module(module_name)
            getattr(mod, func_name)()
        except Exception as e:
            FreeCAD.Console.PrintError(f"Error running {module_name}: {e}\n")
            import traceback
            traceback.print_exc()

    def _connect_workbench_toggle(self, toolbar, wb_class_name, attr_name):
        try:
            mw = FreeCADGui.getMainWindow()
            if not hasattr(mw, attr_name):
                setattr(mw, attr_name, [])
            getattr(mw, attr_name).append(toolbar)

            def on_workbench_changed():
                try:
                    current_wb = FreeCADGui.activeWorkbench()
                    toolbar.setVisible(
                        current_wb and current_wb.__class__.__name__ == wb_class_name)
                except Exception as e:
                    FreeCAD.Console.PrintWarning(f"Error in workbench toggle: {e}\n")

            mw.workbenchActivated.connect(on_workbench_changed)
        except Exception as e:
            FreeCAD.Console.PrintWarning(f"Could not connect workbench toggle: {e}\n")

    def IsActive(self):
        return True


class CreatePartDesignToolbarCommand:
    wb_path: Path = Path(__file__).parent

    def GetResources(self):
        return {
            'Pixmap': '',
            'MenuText': 'Create PartDesign Toolbar',
            'ToolTip': 'Create a toolbar with PartDesign-specific tools that appears only in PartDesign workbench'
        }

    def Activated(self):
        try:
            mw = FreeCADGui.getMainWindow()
            toolbar_name = "Detessellate_PartDesign_Tools"

            for toolbar in mw.findChildren(QtWidgets.QToolBar):
                if toolbar.objectName() == toolbar_name:
                    FreeCAD.Console.PrintMessage("Detessellate PartDesign Tools toolbar already exists.\n")
                    return

            custom_toolbar = QtWidgets.QToolBar("Detessellate PartDesign Tools", mw)
            custom_toolbar.setObjectName(toolbar_name)
            mw.addToolBar(QtCore.Qt.TopToolBarArea, custom_toolbar)

            self._add_topomatch_button(custom_toolbar)
            self._connect_workbench_toggle(custom_toolbar)

            current_wb = FreeCADGui.activeWorkbench()
            custom_toolbar.setVisible(
                current_wb and current_wb.__class__.__name__ == "PartDesignWorkbench")

        except Exception as e:
            FreeCAD.Console.PrintError(f"Error creating PartDesign toolbar: {e}\n")
            import traceback
            traceback.print_exc()

    def _add_topomatch_button(self, toolbar):
        try:
            mw = FreeCADGui.getMainWindow()
            for action in mw.findChildren(QtGui.QAction):
                if action.objectName() == "Detessellate_TopoMatchSelector" or \
                   action.data() == "Detessellate_TopoMatchSelector" or \
                   action.text() == "Topo Match Selector":
                    toolbar.addAction(action)
                    return
            FreeCAD.Console.PrintWarning("Could not find TopoMatchSelector action\n")
        except Exception as e:
            FreeCAD.Console.PrintError(f"Error adding TopoMatchSelector button: {e}\n")
            import traceback
            traceback.print_exc()

    def _connect_workbench_toggle(self, toolbar):
        try:
            mw = FreeCADGui.getMainWindow()
            if not hasattr(mw, '_detessellate_partdesign_toolbars'):
                mw._detessellate_partdesign_toolbars = []
            mw._detessellate_partdesign_toolbars.append(toolbar)

            def on_workbench_changed():
                try:
                    current_wb = FreeCADGui.activeWorkbench()
                    toolbar.setVisible(
                        current_wb and current_wb.__class__.__name__ == "PartDesignWorkbench")
                except Exception as e:
                    FreeCAD.Console.PrintWarning(f"Error in workbench toggle: {e}\n")

            mw.workbenchActivated.connect(on_workbench_changed)
        except Exception as e:
            FreeCAD.Console.PrintWarning(f"Could not connect workbench toggle: {e}\n")

    def IsActive(self):
        return True


class CreateGlobalToolbarCommand:
    wb_path: Path = Path(__file__).parent

    def GetResources(self):
        return {
            'Pixmap': '',
            'MenuText': 'Create Global Toolbar',
            'ToolTip': 'Create a toolbar with universal tools that appears in all workbenches'
        }

    def Activated(self):
        try:
            mw = FreeCADGui.getMainWindow()
            toolbar_name = "Detessellate_Global_Tools"

            for toolbar in mw.findChildren(QtWidgets.QToolBar):
                if toolbar.objectName() == toolbar_name:
                    FreeCAD.Console.PrintMessage("Detessellate Global Tools toolbar already exists.\n")
                    return

            custom_toolbar = QtWidgets.QToolBar("Detessellate Global", mw)
            custom_toolbar.setObjectName(toolbar_name)
            mw.addToolBar(QtCore.Qt.TopToolBarArea, custom_toolbar)

            for cmd_name in [
                "Detessellate_CoplanarSketch",
                "Detessellate_EdgeLoopSelector",
                "Detessellate_EdgeLoopToSketch",
                "Detessellate_VarSetUpdate",
            ]:
                self._add_command_button(custom_toolbar, cmd_name)

            custom_toolbar.setVisible(True)

        except Exception as e:
            FreeCAD.Console.PrintError(f"Error creating Global toolbar: {e}\n")
            import traceback
            traceback.print_exc()

    def _add_command_button(self, toolbar, command_name):
        try:
            mw = FreeCADGui.getMainWindow()
            for action in mw.findChildren(QtGui.QAction):
                if action.objectName() == command_name:
                    toolbar.addAction(action)
                    return
            FreeCAD.Console.PrintWarning(f"Could not find {command_name} action\n")
        except Exception as e:
            FreeCAD.Console.PrintError(f"Error adding {command_name} button: {e}\n")
            import traceback
            traceback.print_exc()

    def IsActive(self):
        return True


# ---------------------------------------------------------------------------
# Command registration
# ---------------------------------------------------------------------------

command_list = [
    ("Detessellate_MeshPlacement",          MeshPlacementCommand),
    ("Detessellate_MeshToBody",             MeshToBodyCommand),
    ("Detessellate_CoplanarSketch",         CoplanarSketchCommand),
    ("Detessellate_PointPlaneSketch",       PointPlaneSketchCommand),
    ("Detessellate_SketchReProfile",        SketchReProfileCommand),
    ("Detessellate_ConstrainAllPointOnPoint", ConstrainAllPointOnPointCommand),
    ("Detessellate_SketcherWireDoctor",     SketcherWireDoctorCommand),
    ("Detessellate_EdgeLoopSelector",       EdgeLoopSelectorCommand),
    ("Detessellate_EdgeLoopToSketch",       EdgeLoopToSketchCommand),
    ("Detessellate_ReconstructSolid",       ReconstructSolidCommand),
    ("Detessellate_TopoMatchSelector",      TopoMatchSelectorCommand),
    ("Detessellate_VarSetUpdate",           VarSetUpdateCommand),
    ("CreateSketchToolbar",                 CreateSketchToolbarCommand),
    ("CreatePartDesignToolbar",             CreatePartDesignToolbarCommand),
    ("CreateGlobalToolbar",                 CreateGlobalToolbarCommand),
]

for cmd_name, cmd_class in command_list:
    try:
        FreeCADGui.addCommand(cmd_name, cmd_class())
    except Exception as e:
        FreeCAD.Console.PrintError(f"ERROR registering {cmd_name}: {e}\n")
