from pathlib import Path
import sys

import FreeCAD
import FreeCADGui

class ReconstructSolidCommand:
    base_path: Path = Path(__file__).parent.parent / "Macros/ReconstructSolid"

    def GetResources(self):
        icon_path = self.base_path / "ReconstructSolid.svg"
        return {
            'Pixmap': str(icon_path),
            'MenuText': 'Reconstruct Solid',
            'ToolTip': 'Reconstruct a simple solid to change its geometric origin'
        }

    def Activated(self):
        if str(self.base_path) not in sys.path:
            sys.path.append(str(self.base_path))

        try:
            import importlib
            if 'ReconstructSolid' in sys.modules:
                import ReconstructSolid
                importlib.reload(ReconstructSolid)
            else:
                import ReconstructSolid

            # The macro will execute on import or call the appropriate function
            # Adjust based on how ReconstructSolid.py is structured

        except Exception as e:
            FreeCAD.Console.PrintError(f"Error running ReconstructSolid: {e}\n")
            import traceback
            traceback.print_exc()

    def IsActive(self):
        # Only active when a document is
        return FreeCAD.ActiveDocument is not None
