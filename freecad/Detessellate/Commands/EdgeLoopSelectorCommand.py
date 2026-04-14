from pathlib import Path
import sys

import FreeCAD
import FreeCADGui

class EdgeLoopSelectorCommand:
    base_path: Path = Path(__file__).parent.parent / "Macros/EdgeLoopSelector"

    def GetResources(self):
        icon_path = self.base_path / "EdgeLoopSelector.svg"
        return {
            'Pixmap': str(icon_path),
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

            # The macro will execute on import or call the appropriate function
            # Adjust based on how EdgeLoopSelector.py is structured

        except Exception as e:
            FreeCAD.Console.PrintError(f"Error running EdgeLoopSelector: {e}\n")
            import traceback
            traceback.print_exc()

    def IsActive(self):
        # Only active when a document is
        return FreeCAD.ActiveDocument is not None
