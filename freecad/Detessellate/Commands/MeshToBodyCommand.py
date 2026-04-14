# SPDX-License-Identifier: LGPL-2.1-or-later
# SPDX-FileNotice: Part of the Detessellate addon.

from pathlib import Path
import sys

import FreeCAD
import FreeCADGui
from freecad.Detessellate.Misc.Resources import asIcon


class MeshToBodyCommand:
    base_path: Path = Path(__file__).parent.parent / "Macros/MeshToBody"

    def GetResources(self):
        return {
            'Pixmap': asIcon('MeshToBody'),
            'MenuText': 'Mesh To Body',
            'ToolTip': 'Convert meshes to solid body'
        }

    def Activated(self):
        if str(self.base_path) not in sys.path:
            sys.path.append(str(self.base_path))

        try:
            import importlib
            if 'MeshToBody' in sys.modules:
                import MeshToBody
                importlib.reload(MeshToBody)
            else:
                import MeshToBody

            MeshToBody.run_unified_macro(auto_mode=True)

        except Exception as e:
            FreeCAD.Console.PrintError(f"Error running MeshToBody: {e}\n")
            import traceback
            traceback.print_exc()

    def IsActive(self):
        return FreeCAD.ActiveDocument is not None
