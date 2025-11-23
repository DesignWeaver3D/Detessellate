import os, sys
import FreeCAD
import FreeCADGui

class TopoMatchSelectorCommand:
    def GetResources(self):
        icon_path = os.path.join(
            os.path.dirname(__file__), "..", "Macros", "TopoMatchSelector", "topomatchselector.svg"
        )
        return {
            'Pixmap': icon_path if os.path.exists(icon_path) else '',  # Empty string = text button fallback
            'MenuText': 'Topo Match Selector',
            'ToolTip': 'Select topology matching elements'
        }

    def Activated(self):
        macro_path = os.path.join(os.path.dirname(__file__), "..", "Macros", "TopoMatchSelector")
        macro_path = os.path.abspath(macro_path)
        if macro_path not in sys.path:
            sys.path.append(macro_path)

        try:
            import importlib
            if 'TopoMatchSelector' in sys.modules:
                import TopoMatchSelector
                importlib.reload(TopoMatchSelector)
            else:
                import TopoMatchSelector

            # The macro will execute on import or call the appropriate function
            # Adjust based on how TopoMatchSelector.py is structured

        except Exception as e:
            FreeCAD.Console.PrintError(f"Error running TopoMatchSelector: {e}\n")
            import traceback
            traceback.print_exc()

    def IsActive(self):
        return True  # Adjust if it needs specific conditions
