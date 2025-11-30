import FreeCAD
import FreeCADGui
from PySide import QtGui, QtCore
import os
import sys

class CreateSketchToolbarCommand:
    def GetResources(self):
        return {
            'Pixmap': '',  # No icon
            'MenuText': 'Create Sketch Toolbar',
            'ToolTip': 'Create a toolbar with sketch tools that appears only in Sketcher workbench'
        }

    def Activated(self):
        try:
            mw = FreeCADGui.getMainWindow()
            toolbar_name = "Detessellate_Sketch_Tools"

            # Check if toolbar already exists
            existing_toolbar = None
            for toolbar in mw.findChildren(QtGui.QToolBar):
                if toolbar.objectName() == toolbar_name:
                    existing_toolbar = toolbar
                    break

            if existing_toolbar:
                FreeCAD.Console.PrintMessage("Detessellate Sketch Tools toolbar already exists.\n")
                return

            # Create new toolbar
            custom_toolbar = QtGui.QToolBar("Detessellate Sketch Tools", mw)
            custom_toolbar.setObjectName(toolbar_name)
            mw.addToolBar(QtCore.Qt.TopToolBarArea, custom_toolbar)

            # Add buttons
            self.add_sketch_reprofile_button(custom_toolbar)
            self.add_sketcher_wiredoctor_button(custom_toolbar)

            # Connect to workbench changes to show/hide toolbar
            self.connect_workbench_toggle(custom_toolbar)

            # Set initial visibility based on current workbench
            current_wb = FreeCADGui.activeWorkbench()
            is_sketcher = current_wb and current_wb.name() == "SketcherWorkbench"
            custom_toolbar.setVisible(is_sketcher)

            FreeCAD.Console.PrintMessage("✓ Created 'Detessellate Sketch Tools' toolbar.\n")
            FreeCAD.Console.PrintMessage("This toolbar appears only in Sketcher workbench.\n")
            FreeCAD.Console.PrintMessage("Note: Run this command again after restarting FreeCAD if needed.\n")

        except Exception as e:
            FreeCAD.Console.PrintError(f"Error creating sketch toolbar: {e}\n")
            import traceback
            traceback.print_exc()

    def connect_workbench_toggle(self, toolbar):
        """Connect to workbench activation to show/hide toolbar"""
        try:
            mw = FreeCADGui.getMainWindow()

            # Store reference to toolbar in the main window so the lambda can access it
            if not hasattr(mw, '_detessellate_sketch_toolbars'):
                mw._detessellate_sketch_toolbars = []
            mw._detessellate_sketch_toolbars.append(toolbar)

            # Create a callback that shows/hides the toolbar
            def on_workbench_changed():
                try:
                    current_wb = FreeCADGui.activeWorkbench()
                    # More robust check - use class name instead of name() method
                    is_sketcher = current_wb and current_wb.__class__.__name__ == "SketcherWorkbench"
                    toolbar.setVisible(is_sketcher)
                except Exception as e:
                    FreeCAD.Console.PrintWarning(f"Error in workbench toggle: {e}\n")

            # Connect to workbench activated signal
            mw.workbenchActivated.connect(on_workbench_changed)

        except Exception as e:
            FreeCAD.Console.PrintWarning(f"Could not connect workbench toggle: {e}\n")

    def add_sketch_reprofile_button(self, toolbar):
        """Add SketchReProfile button that directly calls the macro"""
        try:
            macro_path = os.path.join(
                FreeCAD.getUserAppDataDir(),
                "Mod",
                "Detessellate",
                "Macros",
                "SketchReProfile"
            )

            icon_path = os.path.join(macro_path, "sketchreprofile.svg")
            icon = QtGui.QIcon(icon_path) if os.path.exists(icon_path) else QtGui.QIcon()

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

    def add_sketcher_wiredoctor_button(self, toolbar):
        """Add SketcherWireDoctor button that directly calls the macro"""
        try:
            macro_path = os.path.join(
                FreeCAD.getUserAppDataDir(),
                "Mod",
                "Detessellate",
                "Macros",
                "SketcherWireDoctor"
            )

            # Load icon - adjust the filename to match your actual icon file
            icon_path = os.path.join(macro_path, "sketcherwiredoctor.svg")  # or .png, whatever you have
            icon = QtGui.QIcon(icon_path) if os.path.exists(icon_path) else QtGui.QIcon()

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

    def run_sketch_reprofile(self, macro_path):
        """Directly run the SketchReProfile macro"""
        try:
            if macro_path not in sys.path:
                sys.path.append(macro_path)

            import importlib
            if 'SketchReProfile' in sys.modules:
                import SketchReProfile
                importlib.reload(SketchReProfile)
            else:
                import SketchReProfile

            SketchReProfile.final_sketcher_main()

        except Exception as e:
            FreeCAD.Console.PrintError(f"Error running SketchReProfile: {e}\n")
            import traceback
            traceback.print_exc()

    def run_sketcher_wiredoctor(self, macro_path):
        """Directly run the SketcherWireDoctor macro"""
        try:
            if macro_path not in sys.path:
                sys.path.append(macro_path)

            import importlib
            if 'SketcherWireDoctor_Main' in sys.modules:
                import SketcherWireDoctor_Main
                importlib.reload(SketcherWireDoctor_Main)
            else:
                import SketcherWireDoctor_Main

            # Call the main function to show the docker
            SketcherWireDoctor_Main.show_sketcher_wire_doctor()

        except Exception as e:
            FreeCAD.Console.PrintError(f"Error running SketcherWireDoctor: {e}\n")
            import traceback
            traceback.print_exc()

    def IsActive(self):
        return True
