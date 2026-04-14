# SPDX-License-Identifier: LGPL-2.1-or-later
# SPDX-FileNotice: Part of the Detessellate addon.

from pathlib import Path
import sys

import FreeCAD
import FreeCADGui
from freecad.Detessellate.Misc.Resources import asIcon


class EdgeLoopSelectorCommand:
    base_path: Path = Path(__file__).parent.parent / "Macros/EdgeLoopSelector"

    def GetResources(self):
        return {
            'Pixmap': asIcon('EdgeLoopSelector'),
            'MenuText': 'Edge Loop Selector',
            'ToolTip': 'Select connected edge loops from sketches or faces'
        }

    def Activated(self):
        if str(self.base_path) not in sys.path:
            sys.path.append(str(self.base_path))

        try:
            import importlib
            if 'EdgeLoopSelector' in sys.modules:
                import EdgeLoopSelector
                importlib.reload(EdgeLoopSelector)
            else:
                import EdgeLoopSelector

        except Exception as e:
            FreeCAD.Console.PrintError(f"Error running EdgeLoopSelector: {e}\n")
            import traceback
            traceback.print_exc()

    def IsActive(self):
        return FreeCAD.ActiveDocument is not None
