# SPDX-License-Identifier: LGPL-2.1-or-later
# SPDX-FileNotice: Part of the Detessellate addon.

from pathlib import Path
import sys

import FreeCAD
import FreeCADGui
from freecad.Detessellate.Misc.Resources import asIcon


class SketcherWireDoctorCommand:
    base_path: Path = Path(__file__).parent.parent / "Macros/SketcherWireDoctor"

    def GetResources(self):
        return {
            'Pixmap': asIcon('SketcherWireDoctor'),
            'MenuText': 'Sketcher Wire Doctor',
            'ToolTip': 'Fix sketch wire not closed issues'
        }

    def Activated(self):
        if str(self.base_path) not in sys.path:
            sys.path.append(str(self.base_path))

        try:
            import importlib
            if 'SketcherWireDoctor_Main' in sys.modules:
                import SketcherWireDoctor_Main
                importlib.reload(SketcherWireDoctor_Main)
            else:
                import SketcherWireDoctor_Main

            SketcherWireDoctor_Main.show_sketcher_wire_doctor()

        except Exception as e:
            FreeCAD.Console.PrintError(f"Error running SketcherWireDoctor: {e}\n")
            import traceback
            traceback.print_exc()

    def IsActive(self):
        return FreeCAD.ActiveDocument is not None
