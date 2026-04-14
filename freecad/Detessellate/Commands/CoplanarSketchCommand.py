from pathlib import Path
import sys

import FreeCAD
import FreeCADGui

class CoplanarSketchCommand:
    base_path: Path = Path(__file__).parent.parent / "Macros/CoplanarSketch"

    def GetResources(self):
        icon_path = self.base_path / "CoplanarSketch.svg"
        return {
            'Pixmap': str(icon_path),
            'MenuText': 'Coplanar Sketch',
            'ToolTip': 'Create sketches from selected coplanar edges or faces from a tessellated solid.'
        }

    def Activated(self):
        if str(self.base_path) not in sys.path:
            sys.path.append(str(self.base_path))

        try:
            import importlib
            if 'CoplanarSketch' in sys.modules:
                import CoplanarSketch
                importlib.reload(CoplanarSketch)
            else:
                import CoplanarSketch

            # The show_edge_data_collector_docker() runs on import/reload

        except Exception as e:
            FreeCAD.Console.PrintError(f"Error running CoplanarSketch: {e}\n")
            import traceback
            traceback.print_exc()

    def IsActive(self):
        # Only active when a document is
        return FreeCAD.ActiveDocument is not None
