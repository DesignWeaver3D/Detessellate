from pathlib import Path
import sys

import FreeCAD
import FreeCADGui

class PointPlaneSketchCommand:
    base_path: Path = Path(__file__).parent.parent / "Macros/PointPlaneSketch"

    def GetResources(self):
        icon_path = self.base_path / "PointPlaneSketch.svg"
        return {
            'Pixmap': str(icon_path),
            'MenuText': 'Point Plane Sketch',
            'ToolTip': 'Create sketch from points and ther derived plane'
        }

    def Activated(self):
        if str(self.base_path) not in sys.path:
            sys.path.append(str(self.base_path))

        try:
            import importlib
            if 'PointPlaneSketch' in sys.modules:
                import PointPlaneSketch
                importlib.reload(PointPlaneSketch)
            else:
                import PointPlaneSketch

            # Execute the show function from the macro
            PointPlaneSketch.show_point_cloud_plane_sketch()

        except Exception as e:
            FreeCAD.Console.PrintError(f"Error running PointPlaneSketch: {e}\n")
            import traceback
            traceback.print_exc()

    def IsActive(self):
        # Only active when a document is
        return FreeCAD.ActiveDocument is not None
