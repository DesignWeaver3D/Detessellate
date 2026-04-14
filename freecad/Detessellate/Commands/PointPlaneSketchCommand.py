# SPDX-License-Identifier: LGPL-2.1-or-later
# SPDX-FileNotice: Part of the Detessellate addon.

from pathlib import Path
import sys

import FreeCAD
import FreeCADGui
from freecad.Detessellate.Misc.Resources import asIcon


class PointPlaneSketchCommand:
    base_path: Path = Path(__file__).parent.parent / "Macros/PointPlaneSketch"

    def GetResources(self):
        return {
            'Pixmap': asIcon('PointPlaneSketch'),
            'MenuText': 'Point Plane Sketch',
            'ToolTip': 'Create sketch from points and their derived plane'
        }

    def Activated(self):
        if str(self.base_path) not in sys.path:
            sys.path.append(str(self.base_path))

        try:
            import importlib
            if 'PointPlaneSketch' in sys.modules:
                import PointPlaneSketch
                importlib.reload(PointPlaneSketch)
            else:
                import PointPlaneSketch

            PointPlaneSketch.show_point_cloud_plane_sketch()

        except Exception as e:
            FreeCAD.Console.PrintError(f"Error running PointPlaneSketch: {e}\n")
            import traceback
            traceback.print_exc()

    def IsActive(self):
        return FreeCAD.ActiveDocument is not None
