from pathlib import Path
import sys

import FreeCAD
import FreeCADGui
from PySide import QtGui, QtCore

class CreateGlobalToolbarCommand:
    wb_path: Path = Path(__file__).parent.parent

    def GetResources(self):
        return {
            'Pixmap': '',  # No icon
            'MenuText': 'Create Global Toolbar',
            'ToolTip': 'Create a toolbar with universal tools that appears in all workbenches'
        }

    def Activated(self):
        try:
            mw = FreeCADGui.getMainWindow()
            toolbar_name = "Detessellate_Global_Tools"

            # Check if toolbar already exists
            existing_toolbar = None
            for toolbar in mw.findChildren(QtGui.QToolBar):
                if toolbar.objectName() == toolbar_name:
                    existing_toolbar = toolbar
                    break

            if existing_toolbar:
                FreeCAD.Console.PrintMessage("Detessellate Global Tools toolbar already exists.\n")
                return

            # Create new toolbar
            custom_toolbar = QtGui.QToolBar("Detessellate Global", mw)
            custom_toolbar.setObjectName(toolbar_name)
            mw.addToolBar(QtCore.Qt.TopToolBarArea, custom_toolbar)

            # Add buttons using registered commands - gets proper tooltips automatically
            self.add_command_button(custom_toolbar, "Detessellate_CoplanarSketch")
            self.add_command_button(custom_toolbar, "Detessellate_EdgeLoopSelector")
            self.add_command_button(custom_toolbar, "Detessellate_EdgeLoopToSketch")
            self.add_command_button(custom_toolbar, "Detessellate_VarSetUpdate")

            # No workbench toggle needed - always visible
            custom_toolbar.setVisible(True)

            #FreeCAD.Console.PrintMessage("✓ Created 'Detessellate Global' toolbar.\n")
            #FreeCAD.Console.PrintMessage("This toolbar is visible in all workbenches.\n")

        except Exception as e:
            FreeCAD.Console.PrintError(f"Error creating Global toolbar: {e}\n")
            import traceback
            traceback.print_exc()

    def add_command_button(self, toolbar, command_name):
        """Add a button using the registered FreeCAD command"""
        try:
            mw = FreeCADGui.getMainWindow()

            # Find the action created by FreeCAD when command was registered
            for action in mw.findChildren(QtGui.QAction):
                if action.objectName() == command_name:
                    toolbar.addAction(action)
                    #FreeCAD.Console.PrintMessage(f"✓ Added {command_name} button\n")
                    return

            FreeCAD.Console.PrintWarning(f"Could not find {command_name} action\n")

        except Exception as e:
            FreeCAD.Console.PrintError(f"Error adding {command_name} button: {e}\n")
            import traceback
            traceback.print_exc()

    def IsActive(self):
        return True
