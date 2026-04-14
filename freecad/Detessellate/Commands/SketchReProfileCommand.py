# SPDX-License-Identifier: LGPL-2.1-or-later
# SPDX-FileNotice: Part of the Detessellate addon.

from pathlib import Path
import sys

import FreeCAD
import FreeCADGui
from freecad.Detessellate.Misc.Resources import asIcon


class SketchReProfileCommand:
    base_path: Path = Path(__file__).parent.parent / "Macros/SketchReProfile"

    def GetResources(self):
        return {
            'Pixmap': asIcon('SketchReProfile'),
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

            SketchReProfile.main()

        except Exception as e:
            FreeCAD.Console.PrintError(f"Error running SketchReProfile: {e}\n")
            import traceback
            traceback.print_exc()

    def IsActive(self):
        return FreeCAD.ActiveDocument is not None
