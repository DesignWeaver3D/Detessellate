# SPDX-License-Identifier: LGPL-2.1-or-later
# SPDX-FileNotice: Part of the Detessellate addon.

from pathlib import Path
import sys

import FreeCAD
import FreeCADGui
from freecad.Detessellate.Misc.Resources import asIcon


class TopoMatchSelectorCommand:
    base_path: Path = Path(__file__).parent.parent / "Macros/TopoMatchSelector"

    def GetResources(self):
        return {
            'Pixmap': asIcon('TopoMatchSelector'),
            'MenuText': 'Topo Match Selector',
            'ToolTip': 'Select topology matching elements'
        }

    def Activated(self):
        if str(self.base_path) not in sys.path:
            sys.path.append(str(self.base_path))

        try:
            import importlib
            if 'TopoMatchSelector' in sys.modules:
                import TopoMatchSelector
                importlib.reload(TopoMatchSelector)
            else:
                import TopoMatchSelector

            TopoMatchSelector.create_topo_match_selector()

        except Exception as e:
            FreeCAD.Console.PrintError(f"Error running TopoMatchSelector: {e}\n")
            import traceback
            traceback.print_exc()

    def IsActive(self):
        return FreeCAD.ActiveDocument is not None
