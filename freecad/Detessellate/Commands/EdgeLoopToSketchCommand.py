from pathlib import Path
import sys

import FreeCAD
import FreeCADGui

class EdgeLoopToSketchCommand:
    base_path: Path = Path(__file__).parent.parent / "Macros/EdgeLoopToSketch"

    def GetResources(self):
        icon_path = self.base_path / "EdgeLoopToSketch.svg"
        return {
            'Pixmap': str(icon_path),
            'MenuText': 'Edge Loop to Sketch',
            'ToolTip': 'Convert selected coplanar faces or edges to parametric sketch'
        }

    def Activated(self):
        if str(self.base_path) not in sys.path:
            sys.path.append(str(self.base_path))

        try:
            import importlib
            if 'EdgeLoopToSketch' in sys.modules:
                import EdgeLoopToSketch
                importlib.reload(EdgeLoopToSketch)
            else:
                import EdgeLoopToSketch

            # The macro executes on import via edge_loop_to_sketch() call at end

        except Exception as e:
            FreeCAD.Console.PrintError(f"Error running EdgeLoopToSketch: {e}\n")
            import traceback
            traceback.print_exc()

    def IsActive(self):
        # Only active when a document is
        return FreeCAD.ActiveDocument is not None
