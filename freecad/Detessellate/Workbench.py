# SPDX-License-Identifier: LGPL-2.1-or-later
# SPDX-FileNotice: Part of the Detessellate addon.

import FreeCAD
import FreeCADGui
import traceback

command_specs = [
    ("Detessellate_MeshPlacement", "freecad.Detessellate.Commands.MeshPlacementCommand", "MeshPlacementCommand", "Detessellate Mesh", True),
    ("Detessellate_MeshToBody", "freecad.Detessellate.Commands.MeshToBodyCommand", "MeshToBodyCommand", "Detessellate Mesh", True),
    ("Detessellate_CoplanarSketch", "freecad.Detessellate.Commands.CoplanarSketchCommand", "CoplanarSketchCommand", "Detessellate Sketch", True),
    ("Detessellate_EdgeLoopSelector", "freecad.Detessellate.Commands.EdgeLoopSelectorCommand", "EdgeLoopSelectorCommand", "Detessellate Utilities", True),
    ("Detessellate_EdgeLoopToSketch", "freecad.Detessellate.Commands.EdgeLoopToSketchCommand", "EdgeLoopToSketchCommand", "Detessellate Utilities", True),
    ("Detessellate_PointPlaneSketch", "freecad.Detessellate.Commands.PointPlaneSketchCommand", "PointPlaneSketchCommand", "Detessellate Sketch", True),
    ("Detessellate_ReconstructSolid", "freecad.Detessellate.Commands.ReconstructSolidCommand", "ReconstructSolidCommand", "Detessellate Utilities", True),
    ("Detessellate_TopoMatchSelector", "freecad.Detessellate.Commands.TopoMatchSelectorCommand", "TopoMatchSelectorCommand", "Detessellate Utilities", False),
    ("Detessellate_VarSetUpdate", "freecad.Detessellate.Commands.VarSetUpdateCommand", "VarSetUpdateCommand", "Detessellate Utilities", True),
    ("CreateSketchToolbar", "freecad.Detessellate.Commands.CreateSketchToolbarCommand", "CreateSketchToolbarCommand", "Detessellate Sketch", False),
    ("CreatePartDesignToolbar", "freecad.Detessellate.Commands.CreatePartDesignToolbarCommand", "CreatePartDesignToolbarCommand", "Detessellate Utilities", False),
    ("CreateGlobalToolbar", "freecad.Detessellate.Commands.CreateGlobalToolbarCommand", "CreateGlobalToolbarCommand", "Detessellate Utilities", False),
]

commands = {}

for cmd_name, module_path, class_name, toolbar, show_in_toolbar in command_specs:
    try:
        module = __import__(module_path, fromlist=[class_name])
        cmd_class = getattr(module, class_name)
        commands[cmd_name] = (cmd_class, toolbar, show_in_toolbar)
        FreeCADGui.addCommand(cmd_name, cmd_class())
    except Exception as e:
        print(f"ERROR importing {cmd_name}: {e}")
        traceback.print_exc()


class DetessellateWorkbench(FreeCADGui.Workbench):
    from pathlib import Path
    MenuText = "Detessellate"
    ToolTip = "Tools to reverse engineer meshes"
    Icon = str(Path(__file__).parent / "Resources/icons/Detessellate.svg")

    def __init__(self):
        self._toolbar_created = False

    def Initialize(self):
        global commands
        for cmd_name, (cmd_class, toolbar, show_in_toolbar) in commands.items():
            if show_in_toolbar:
                self.appendToolbar(toolbar, [cmd_name])
            self.appendMenu("Detessellate", [cmd_name])

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
