# SPDX-License-Identifier: LGPL-2.1-or-later
# SPDX-FileNotice: Part of the Detessellate addon.

import sys
from pathlib import Path

import FreeCAD
import FreeCADGui
from PySide6 import QtGui, QtCore
from freecad.Detessellate.Misc.Resources import asIcon


class CreateSketchToolbarCommand:
    wb_path: Path = Path(__file__).parent.parent

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

            existing_toolbar = None
            for toolbar in mw.findChildren(QtGui.QToolBar):
                if toolbar.objectName() == toolbar_name:
                    existing_toolbar = toolbar
                    break

            if existing_toolbar:
                FreeCAD.Console.PrintMessage("Detessellate Sketch Tools toolbar already exists.\n")
                return

            custom_toolbar = QtGui.QToolBar("Detessellate Sketch Tools", mw)
            custom_toolbar.setObjectName(toolbar_name)
            mw.addToolBar(QtCore.Qt.TopToolBarArea, custom_toolbar)

            self.add_sketch_reprofile_button(custom_toolbar)
            self.add_constrain_all_pointonpoint_button(custom_toolbar)
            self.add_sketcher_wiredoctor_button(custom_toolbar)

            self.connect_workbench_toggle(custom_toolbar)

            current_wb = FreeCADGui.activeWorkbench()
            is_sketcher = current_wb and current_wb.__class__.__name__ == "SketcherWorkbench"
            custom_toolbar.setVisible(is_sketcher)

        except Exception as e:
            FreeCAD.Console.PrintError(f"Error creating sketch toolbar: {e}\n")
            import traceback
            traceback.print_exc()

    def connect_workbench_toggle(self, toolbar):
        try:
            mw = FreeCADGui.getMainWindow()
            if not hasattr(mw, '_detessellate_sketch_toolbars'):
                mw._detessellate_sketch_toolbars = []
            mw._detessellate_sketch_toolbars.append(toolbar)

            def on_workbench_changed():
                try:
                    current_wb = FreeCADGui.activeWorkbench()
                    is_sketcher = current_wb and current_wb.__class__.__name__ == "SketcherWorkbench"
                    toolbar.setVisible(is_sketcher)
                except Exception as e:
                    FreeCAD.Console.PrintWarning(f"Error in workbench toggle: {e}\n")

            mw.workbenchActivated.connect(on_workbench_changed)

        except Exception as e:
            FreeCAD.Console.PrintWarning(f"Could not connect workbench toggle: {e}\n")

    def add_sketch_reprofile_button(self, toolbar):
        try:
            macro_path = self.wb_path / "Macros" / "SketchReProfile"
            icon = QtGui.QIcon(asIcon('SketchReProfile'))
            action = QtGui.QAction(icon, "Sketch ReProfile", toolbar)
            action.setToolTip(
                "<b>SketchReProfile</b><br><br>"
                "Draws lines, circles, arcs, and splines over construction geometry<br><br>"
                "<i>Detessellate_SketchReProfile</i>"
            )
            action.triggered.connect(lambda: self.run_sketch_reprofile(macro_path))
            toolbar.addAction(action)
        except Exception as e:
            FreeCAD.Console.PrintError(f"Error adding SketchReProfile button: {e}\n")

    def add_constrain_all_pointonpoint_button(self, toolbar):
        try:
            macro_path = self.wb_path / "Macros" / "ConstrainAllPointOnPoint"
            icon = QtGui.QIcon(asIcon('ConstrainAllPointOnPoint'))
            action = QtGui.QAction(icon, "Constrain All Point-On-Point", toolbar)
            action.setToolTip(
                "<b>Constrain All Point-On-Point</b><br><br>"
                "Automatically add missing coincident constraints<br><br>"
                "<i>Detessellate_ConstrainAllPointOnPoint</i>"
            )
            action.triggered.connect(lambda: self.run_constrain_all_pointonpoint(macro_path))
            toolbar.addAction(action)
        except Exception as e:
            FreeCAD.Console.PrintError(f"Error adding ConstrainAllPointOnPoint button: {e}\n")

    def add_sketcher_wiredoctor_button(self, toolbar):
        try:
            macro_path = self.wb_path / "Macros" / "SketcherWireDoctor"
            icon = QtGui.QIcon(asIcon('SketcherWireDoctor'))
            action = QtGui.QAction(icon, "Sketcher Wire Doctor", toolbar)
            action.setToolTip(
                "<b>SketcherWireDoctor</b><br><br>"
                "Detects sketch issues and provides repair tools<br><br>"
                "<i>Detessellate_SketcherWireDoctor</i>"
            )
            action.triggered.connect(lambda: self.run_sketcher_wiredoctor(macro_path))
            toolbar.addAction(action)
        except Exception as e:
            FreeCAD.Console.PrintError(f"Error adding SketcherWireDoctor button: {e}\n")

    def run_sketch_reprofile(self, macro_path: Path) -> None:
        try:
            if str(macro_path) not in sys.path:
                sys.path.append(str(macro_path))
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

    def run_constrain_all_pointonpoint(self, macro_path: Path) -> None:
        try:
            if str(macro_path) not in sys.path:
                sys.path.append(str(macro_path))
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

    def run_sketcher_wiredoctor(self, macro_path: Path) -> None:
        try:
            if str(macro_path) not in sys.path:
                sys.path.append(str(macro_path))
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
        return True
