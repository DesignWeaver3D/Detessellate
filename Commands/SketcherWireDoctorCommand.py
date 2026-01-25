from pathlib import Path
import sys

import FreeCAD
import FreeCADGui

class SketcherWireDoctorCommand:
    base_path: Path = Path(__file__).parent.parent / "Macros/SketcherWireDoctor"

    def GetResources(self):
        icon_path = self.base_path / "SketcherWireDoctor.svg"
        return {
            'Pixmap': str(icon_path),
            'MenuText': 'Sketcher Wire Doctor',
            'ToolTip': 'Fix sketch wire not closed issues'
        }

    def Activated(self):
        if str(self.base_path) not in sys.path:
            sys.path.append(str(self.base_path))

        try:
            import importlib
            if 'SketcherWireDoctor' in sys.modules:
                import SketcherWireDoctor
                importlib.reload(SketcherWireDoctor)
            else:
                import SketcherWireDoctor

            # Call the appropriate function based on the macro

        except Exception as e:
            FreeCAD.Console.PrintError(f"Error running SketcherWireDoctor: {e}\n")
            import traceback
            traceback.print_exc()

    def IsActive(self):
        # Only active when a document is
        return FreeCAD.ActiveDocument is not None
