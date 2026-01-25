from pathlib import Path
import sys

import FreeCAD
import FreeCADGui

class VarSetUpdateCommand:
    base_path: Path = Path(__file__).parent.parent / "Macros/VarSet-Update"

    def GetResources(self):
        icon_path = self.base_path / "VarSetUpdate.svg"
        return {
            'Pixmap': str(icon_path),
            'MenuText': 'VarSet Update',
            'ToolTip': 'Update VarSet Properties'
        }

    def Activated(self):
        if str(self.base_path) not in sys.path:
            sys.path.append(str(self.base_path))

        try:
            import importlib
            if 'VarSetUpdate' in sys.modules:  # Note: Python converts hyphens to underscores in module names
                import VarSetUpdate
                importlib.reload(VarSetUpdate)
            else:
                import VarSetUpdate

            # The macro will execute on import or call the appropriate function
            # Adjust based on how VarSetUpdate.py is structured

        except Exception as e:
            FreeCAD.Console.PrintError(f"Error running VarSet Update: {e}\n")
            import traceback
            traceback.print_exc()

    def IsActive(self):
        # Only active when a document is
        return FreeCAD.ActiveDocument is not None
