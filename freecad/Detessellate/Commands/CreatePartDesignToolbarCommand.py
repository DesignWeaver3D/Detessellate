from pathlib import Path
import sys

import FreeCAD
import FreeCADGui
from PySide import QtGui, QtCore

class CreatePartDesignToolbarCommand:
    wb_path: Path = Path(__file__).parent.parent

    def GetResources(self):
        return {
            'Pixmap': '',  # No icon
            'MenuText': 'Create PartDesign Toolbar',
            'ToolTip': 'Create a toolbar with PartDesign-specific tools that appears only in PartDesign workbench'
        }

    def Activated(self):
        try:
            mw = FreeCADGui.getMainWindow()
            toolbar_name = "Detessellate_PartDesign_Tools"

            # Check if toolbar already exists
            existing_toolbar = None
            for toolbar in mw.findChildren(QtGui.QToolBar):
                if toolbar.objectName() == toolbar_name:
                    existing_toolbar = toolbar
                    break

            if existing_toolbar:
                FreeCAD.Console.PrintMessage("Detessellate PartDesign Tools toolbar already exists.\n")
                return

            # Create new toolbar
            custom_toolbar = QtGui.QToolBar("Detessellate PartDesign Tools", mw)
            custom_toolbar.setObjectName(toolbar_name)
            mw.addToolBar(QtCore.Qt.TopToolBarArea, custom_toolbar)

            # Add buttons
            self.add_topomatch_selector_button(custom_toolbar)

            # Connect to workbench changes to show/hide toolbar
            self.connect_workbench_toggle(custom_toolbar)

            # Set initial visibility based on current workbench
            current_wb = FreeCADGui.activeWorkbench()
            is_partdesign = current_wb and current_wb.__class__.__name__ == "PartDesignWorkbench"
            custom_toolbar.setVisible(is_partdesign)

            #FreeCAD.Console.PrintMessage("✓ Created 'Detessellate PartDesign Tools' toolbar.\n")
            #FreeCAD.Console.PrintMessage("This toolbar appears only in PartDesign workbench.\n")

        except Exception as e:
            FreeCAD.Console.PrintError(f"Error creating PartDesign toolbar: {e}\n")
            import traceback
            traceback.print_exc()

    def connect_workbench_toggle(self, toolbar):
        """Connect to workbench activation to show/hide toolbar"""
        try:
            mw = FreeCADGui.getMainWindow()

            # Store reference to toolbar
            if not hasattr(mw, '_detessellate_partdesign_toolbars'):
                mw._detessellate_partdesign_toolbars = []
            mw._detessellate_partdesign_toolbars.append(toolbar)

            # Create a callback that shows/hides the toolbar
            def on_workbench_changed():
                try:
                    current_wb = FreeCADGui.activeWorkbench()
                    is_partdesign = current_wb and current_wb.__class__.__name__ == "PartDesignWorkbench"
                    toolbar.setVisible(is_partdesign)
                except Exception as e:
                    FreeCAD.Console.PrintWarning(f"Error in workbench toggle: {e}\n")

            # Connect to workbench activated signal
            mw.workbenchActivated.connect(on_workbench_changed)

        except Exception as e:
            FreeCAD.Console.PrintWarning(f"Could not connect workbench toggle: {e}\n")

    def add_topomatch_selector_button(self, toolbar):
        """Add TopoMatchSelector button using its registered action"""
        try:
            # Method 1: Try getting via command manager
            mw = FreeCADGui.getMainWindow()

            # Search all QActions for one with our command name
            for action in mw.findChildren(QtGui.QAction):
                # FreeCAD actions have objectName set to the command name
                if action.objectName() == "Detessellate_TopoMatchSelector" or \
                   action.data() == "Detessellate_TopoMatchSelector" or \
                   action.text() == "Topo Match Selector":  # MenuText from GetResources
                    #FreeCAD.Console.PrintMessage(f"Found action: {action.objectName()}, text: {action.text()}\n")
                    toolbar.addAction(action)
                    return

            FreeCAD.Console.PrintWarning("Could not find TopoMatchSelector action\n")

        except Exception as e:
            FreeCAD.Console.PrintError(f"Error: {e}\n")
            import traceback
            traceback.print_exc()

    def IsActive(self):
        return True
