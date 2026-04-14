from pathlib import Path
import sys

import FreeCAD
import FreeCADGui

class MeshToBodyCommand:
    base_path: Path = Path(__file__).parent.parent / "Macros/MeshToBody"

    def GetResources(self):
        icon_path = self.base_path / "MeshToBody.svg"
        return {
            'Pixmap': str(icon_path),
            'MenuText': 'Mesh To Body',
            'ToolTip': 'Convert meshes to solid body'
        }

    def Activated(self):
        if str(self.base_path) not in sys.path:
            sys.path.append(str(self.base_path))

        try:
            import importlib
            if 'MeshToBody' in sys.modules:
                import MeshToBody
                importlib.reload(MeshToBody)
            else:
                import MeshToBody

            # Call the function directly
            MeshToBody.run_unified_macro(auto_mode=True)

        except Exception as e:
            FreeCAD.Console.PrintError(f"Error running MeshToBody: {e}\n")
            import traceback
            traceback.print_exc()

    def IsActive(self):
        # Only active when a document is
        return FreeCAD.ActiveDocument is not None
