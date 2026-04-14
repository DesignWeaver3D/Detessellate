# SPDX-License-Identifier: LGPL-2.1-or-later
# SPDX-FileNotice: Part of the Detessellate addon.

from pathlib import Path
import sys

import FreeCAD
import FreeCADGui
from freecad.Detessellate.Misc.Resources import asIcon


class VarSetUpdateCommand:
    base_path: Path = Path(__file__).parent.parent / "Macros/VarSet-Update"

    def GetResources(self):
        return {
            'Pixmap': asIcon('VarSetUpdate'),
            'MenuText': 'VarSet Update',
            'ToolTip': 'Update VarSet Properties'
        }

    def Activated(self):
        if str(self.base_path) not in sys.path:
            sys.path.append(str(self.base_path))

        try:
            import importlib
            if 'VarSetUpdate' in sys.modules:
                import VarSetUpdate
                importlib.reload(VarSetUpdate)
            else:
                import VarSetUpdate

        except Exception as e:
            FreeCAD.Console.PrintError(f"Error running VarSet Update: {e}\n")
            import traceback
            traceback.print_exc()

    def IsActive(self):
        return FreeCAD.ActiveDocument is not None
