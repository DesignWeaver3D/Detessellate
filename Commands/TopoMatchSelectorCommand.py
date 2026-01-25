from pathlib import Path
import sys

import FreeCAD
import FreeCADGui

class TopoMatchSelectorCommand:
    base_path: Path = Path(__file__).parent.parent / "Macros/TopoMatchSelector"

    def GetResources(self):
        icon_path = self.base_path / "TopoMatchSelector.svg"
        return {
            'Pixmap': str(icon_path),
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

            # Call the function that creates the docker
            TopoMatchSelector.create_topo_match_selector()

        except Exception as e:
            FreeCAD.Console.PrintError(f"Error running TopoMatchSelector: {e}\n")
            import traceback
            traceback.print_exc()

    def IsActive(self):
        # Only active when a document is
        return FreeCAD.ActiveDocument is not None
