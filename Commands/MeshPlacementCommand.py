from pathlib import Path
import  sys

import FreeCAD
import FreeCADGui

class MeshPlacementCommand:
    base_path: Path = Path(__file__).parent.parent / "Macros/MeshPlacement"

    def GetResources(self):
        icon_path = self.base_path / "MeshPlacement.svg"
        return {
            'Pixmap': str(icon_path),
            'MenuText': 'Mesh Placement',
            'ToolTip': 'Center and align meshes at origin'
        }

    def Activated(self):
        if str(self.base_path) not in sys.path:
            sys.path.append(str(self.base_path))
        try:
            # Force reload if already imported
            import importlib
            if 'MeshPlacement' in sys.modules:
                import MeshPlacement
                importlib.reload(MeshPlacement)
            else:
                import MeshPlacement
        except Exception as e:
            FreeCAD.Console.PrintError(f"Error running MeshPlacement: {e}\n")
            import traceback
            traceback.print_exc()

    def IsActive(self):
        # Only active when a document is
        return FreeCAD.ActiveDocument is not None
