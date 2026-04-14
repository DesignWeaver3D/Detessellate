# SPDX-License-Identifier: LGPL-2.1-or-later
# SPDX-FileNotice: Part of the Detessellate addon.

from pathlib import Path
import sys

import FreeCAD
import FreeCADGui
from freecad.Detessellate.Misc.Resources import asIcon

class CoplanarSketchCommand:
    base_path: Path = Path(__file__).parent.parent / "Macros/CoplanarSketch"

    def GetResources(self):
        return {
            'Pixmap': asIcon('CoplanarSketch'),
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

        except Exception as e:
            FreeCAD.Console.PrintError(f"Error running CoplanarSketch: {e}\n")
            import traceback
            traceback.print_exc()

    def IsActive(self):
        return FreeCAD.ActiveDocument is not None
