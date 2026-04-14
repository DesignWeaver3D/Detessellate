# SPDX-License-Identifier: LGPL-2.1-or-later
# SPDX-FileNotice: Part of the Detessellate addon.

from pathlib import Path
import sys

import FreeCAD
import FreeCADGui
from PySide6 import QtGui, QtCore

class CreateGlobalToolbarCommand:
    wb_path: Path = Path(__file__).parent.parent

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

            existing_toolbar = None
            for toolbar in mw.findChildren(QtGui.QToolBar):
                if toolbar.objectName() == toolbar_name:
                    existing_toolbar = toolbar
                    break

            if existing_toolbar:
                FreeCAD.Console.PrintMessage("Detessellate Global Tools toolbar already exists.\n")
                return

            custom_toolbar = QtGui.QToolBar("Detessellate Global", mw)
            custom_toolbar.setObjectName(toolbar_name)
            mw.addToolBar(QtCore.Qt.TopToolBarArea, custom_toolbar)

            self.add_command_button(custom_toolbar, "Detessellate_CoplanarSketch")
            self.add_command_button(custom_toolbar, "Detessellate_EdgeLoopSelector")
            self.add_command_button(custom_toolbar, "Detessellate_EdgeLoopToSketch")
            self.add_command_button(custom_toolbar, "Detessellate_VarSetUpdate")

            custom_toolbar.setVisible(True)

        except Exception as e:
            FreeCAD.Console.PrintError(f"Error creating Global toolbar: {e}\n")
            import traceback
            traceback.print_exc()

    def add_command_button(self, toolbar, command_name):
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
