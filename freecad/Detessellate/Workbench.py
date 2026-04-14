# SPDX-License-Identifier: LGPL-2.1-or-later
# SPDX-FileNotice: Part of the Detessellate addon.

import FreeCAD
import FreeCADGui
from freecad.Detessellate.Misc.Resources import asIcon


class DetessellateWorkbench(FreeCADGui.Workbench):

    MenuText = "Detessellate"
    ToolTip = "Tools to reverse engineer meshes"
    Icon = asIcon('Detessellate')

    def __init__(self):
        self._toolbar_created = False

    def Initialize(self):
        mesh_cmds = [
            "Detessellate_MeshPlacement",
            "Detessellate_MeshToBody",
        ]
        sketch_cmds = [
            "Detessellate_CoplanarSketch",
            "Detessellate_PointPlaneSketch",
        ]
        utility_cmds = [
            "Detessellate_EdgeLoopSelector",
            "Detessellate_EdgeLoopToSketch",
            "Detessellate_ReconstructSolid",
            "Detessellate_VarSetUpdate",
        ]

        self.appendToolbar("Detessellate Mesh", mesh_cmds)
        self.appendToolbar("Detessellate Sketch", sketch_cmds)
        self.appendToolbar("Detessellate Utilities", utility_cmds)

        all_cmds = mesh_cmds + sketch_cmds + utility_cmds + [
            "Detessellate_TopoMatchSelector",
            "Detessellate_SketchReProfile",
            "Detessellate_ConstrainAllPointOnPoint",
            "Detessellate_SketcherWireDoctor",
        ]
        self.appendMenu("Detessellate", all_cmds)

    def Activated(self):
        if not self._toolbar_created:
            self._auto_create_sketch_toolbar()
            self._auto_create_partdesign_toolbar()
            self._auto_create_global_toolbar()
            self._toolbar_created = True

    def Deactivated(self):
        pass

    def _auto_create_sketch_toolbar(self):
        try:
            FreeCADGui.runCommand('CreateSketchToolbar')
        except Exception as e:
            FreeCAD.Console.PrintWarning(f"Could not auto-create sketch toolbar: {e}\n")

    def _auto_create_partdesign_toolbar(self):
        try:
            FreeCADGui.runCommand('CreatePartDesignToolbar')
        except Exception as e:
            FreeCAD.Console.PrintWarning(f"Could not auto-create PartDesign toolbar: {e}\n")

    def _auto_create_global_toolbar(self):
        try:
            FreeCADGui.runCommand('CreateGlobalToolbar')
        except Exception as e:
            FreeCAD.Console.PrintWarning(f"Could not auto-create Global toolbar: {e}\n")

    def GetClassName(self):
        return "Gui::PythonWorkbench"


FreeCADGui.addWorkbench(DetessellateWorkbench())
