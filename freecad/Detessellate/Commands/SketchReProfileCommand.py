from pathlib import Path
import sys

import FreeCAD
import FreeCADGui

class SketchReProfileCommand:
    base_path: Path = Path(__file__).parent.parent / "Macros/SketchReProfile"

    def GetResources(self):
        icon_path = self.base_path / "SketchReProfile.svg"
        return {
            'Pixmap': str(icon_path),
            'MenuText': 'Sketch ReProfile',
            'ToolTip': 'Reprocess sketch profiles - converts construction lines to circles, arcs, and splines'
        }

    def Activated(self):
        if str(self.base_path) not in sys.path:
            sys.path.append(str(self.base_path))

        try:
            import importlib
            if 'SketchReProfile' in sys.modules:
                import SketchReProfile
                importlib.reload(SketchReProfile)
            else:
                import SketchReProfile

            # Call the main function
            SketchReProfile.main()

        except Exception as e:
            FreeCAD.Console.PrintError(f"Error running SketchReProfile: {e}\n")
            import traceback
            traceback.print_exc()

    def IsActive(self):
        # Only active when a document is
        return FreeCAD.ActiveDocument is not None

        '''
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
        ''''
