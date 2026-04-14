# SPDX-License-Identifier: LGPL-2.1-or-later
# SPDX-FileNotice: Part of the Detessellate addon.

from pathlib import Path
import sys

import FreeCAD
import FreeCADGui
from freecad.Detessellate.Misc.Resources import asIcon


class EdgeLoopToSketchCommand:
    base_path: Path = Path(__file__).parent.parent / "Macros/EdgeLoopToSketch"

    def GetResources(self):
        return {
            'Pixmap': asIcon('EdgeLoopToSketch'),
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

        except Exception as e:
            FreeCAD.Console.PrintError(f"Error running EdgeLoopToSketch: {e}\n")
            import traceback
            traceback.print_exc()

    def IsActive(self):
        return FreeCAD.ActiveDocument is not None
