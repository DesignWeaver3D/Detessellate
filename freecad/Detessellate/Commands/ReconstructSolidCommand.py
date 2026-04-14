# SPDX-License-Identifier: LGPL-2.1-or-later
# SPDX-FileNotice: Part of the Detessellate addon.

from pathlib import Path
import sys

import FreeCAD
import FreeCADGui
from freecad.Detessellate.Misc.Resources import asIcon


class ReconstructSolidCommand:
    base_path: Path = Path(__file__).parent.parent / "Macros/ReconstructSolid"

    def GetResources(self):
        return {
            'Pixmap': asIcon('ReconstructSolid'),
            'MenuText': 'Reconstruct Solid',
            'ToolTip': 'Reconstruct a simple solid to change its geometric origin'
        }

    def Activated(self):
        if str(self.base_path) not in sys.path:
            sys.path.append(str(self.base_path))

        try:
            import importlib
            if 'ReconstructSolid' in sys.modules:
                import ReconstructSolid
                importlib.reload(ReconstructSolid)
            else:
                import ReconstructSolid

        except Exception as e:
            FreeCAD.Console.PrintError(f"Error running ReconstructSolid: {e}\n")
            import traceback
            traceback.print_exc()

    def IsActive(self):
        return FreeCAD.ActiveDocument is not None
