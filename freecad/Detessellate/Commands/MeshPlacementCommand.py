# SPDX-License-Identifier: LGPL-2.1-or-later
# SPDX-FileNotice: Part of the Detessellate addon.

from pathlib import Path
import sys

import FreeCAD
import FreeCADGui
from freecad.Detessellate.Misc.Resources import asIcon


class MeshPlacementCommand:
    base_path: Path = Path(__file__).parent.parent / "Macros/MeshPlacement"

    def GetResources(self):
        return {
            'Pixmap': asIcon('MeshPlacement'),
            'MenuText': 'Mesh Placement',
            'ToolTip': 'Center and align meshes at origin'
        }

    def Activated(self):
        if str(self.base_path) not in sys.path:
            sys.path.append(str(self.base_path))
        try:
            import importlib
            if 'MeshPlacement' in sys.modules:
                import MeshPlacement
                importlib.reload(MeshPlacement)
            else:
                import MeshPlacement
        except Exception as e:
            FreeCAD.Console.PrintError(f"Error running MeshPlacement: {e}\n")
            import traceback
            traceback.print_exc()

    def IsActive(self):
        return FreeCAD.ActiveDocument is not None
