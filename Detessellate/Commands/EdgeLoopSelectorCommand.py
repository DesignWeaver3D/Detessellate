import os, sys
import FreeCAD
import FreeCADGui

class EdgeLoopSelectorCommand:
    def GetResources(self):
        icon_path = os.path.join(
            os.path.dirname(__file__), "..", "Macros", "EdgeLoopSelector", "edgeloopselector.svg"
        )
        return {
            'Pixmap': icon_path if os.path.exists(icon_path) else '',  # Empty string = text button fallback
            'MenuText': 'Edge Loop Selector',
            'ToolTip': 'Select connected edge loops'
        }

    def Activated(self):
        macro_path = os.path.join(os.path.dirname(__file__), "..", "Macros", "EdgeLoopSelector")
        macro_path = os.path.abspath(macro_path)
        if macro_path not in sys.path:
            sys.path.append(macro_path)

        try:
            import importlib
            if 'EdgeLoopSelector' in sys.modules:
                import EdgeLoopSelector
                importlib.reload(EdgeLoopSelector)
            else:
                import EdgeLoopSelector

            # The macro will execute on import or call the appropriate function
            # Adjust based on how EdgeLoopSelector.py is structured

        except Exception as e:
            FreeCAD.Console.PrintError(f"Error running EdgeLoopSelector: {e}\n")
            import traceback
            traceback.print_exc()

    def IsActive(self):
        return True  # Adjust if it needs specific conditions
