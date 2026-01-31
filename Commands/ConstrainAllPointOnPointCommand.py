from pathlib import Path
import sys

import FreeCAD
import FreeCADGui


class ConstrainAllPointOnPointCommand:
    base_path: Path = Path(__file__).parent.parent / "Macros/ConstrainAllPointOnPoint"

    def GetResources(self):
        icon_path = self.base_path / "ConstrainAllPointOnPoint.svg"
        return {
            'Pixmap': str(icon_path),
            'MenuText': 'Constrain All Point-On-Point',
            'ToolTip': 'Automatically add missing coincident constraints using built-in FreeCAD detection'
        }

    def Activated(self):
        if str(self.base_path) not in sys.path:
            sys.path.append(str(self.base_path))

        try:
            import importlib
            if 'ConstrainAllPointOnPoint' in sys.modules:
                import ConstrainAllPointOnPoint
                importlib.reload(ConstrainAllPointOnPoint)
            else:
                import ConstrainAllPointOnPoint

            # Call the main function
            ConstrainAllPointOnPoint.main()

        except Exception as e:
            FreeCAD.Console.PrintError(f"Error running ConstrainAllPointOnPoint: {e}\n")
            import traceback
            traceback.print_exc()

    def IsActive(self):
        # Active when a sketch is being edited
        try:
            doc = FreeCADGui.activeDocument()
            if doc is None:
                return False

            edit_obj = doc.getInEdit()
            if edit_obj is None:
                return False

            # Check if it's a sketch object
            if hasattr(edit_obj, 'Object'):
                obj = edit_obj.Object
                return hasattr(obj, 'TypeId') and 'Sketch' in obj.TypeId

            return False
        except:
            return False
