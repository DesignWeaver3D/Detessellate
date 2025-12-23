import FreeCAD
import FreeCADGui
import traceback

#print("Detessellate InitGui.py starting to load")

# Command specs: (command name, module path, class name, toolbar group, show_in_toolbar)
command_specs = [
    ("Detessellate_MeshPlacement", "Commands.MeshPlacementCommand", "MeshPlacementCommand", "Detessellate Mesh", True),
    ("Detessellate_MeshToBody", "Commands.MeshToBodyCommand", "MeshToBodyCommand", "Detessellate Mesh", True),
    ("Detessellate_CoplanarSketch", "Commands.CoplanarSketchCommand", "CoplanarSketchCommand", "Detessellate Sketch", True),
    ("Detessellate_EdgeLoopSelector", "Commands.EdgeLoopSelectorCommand", "EdgeLoopSelectorCommand", "Detessellate Utilities", True),
    ("Detessellate_EdgeLoopToSketch", "Commands.EdgeLoopToSketchCommand", "EdgeLoopToSketchCommand", "Detessellate Utilities", True),
    ("Detessellate_PointPlaneSketch", "Commands.PointPlaneSketchCommand", "PointPlaneSketchCommand", "Detessellate Sketch", True),
    ("Detessellate_ReconstructSolid", "Commands.ReconstructSolidCommand", "ReconstructSolidCommand", "Detessellate Utilities", True),
    ("Detessellate_TopoMatchSelector", "Commands.TopoMatchSelectorCommand", "TopoMatchSelectorCommand", "Detessellate Utilities", False),  # Menu only
    ("Detessellate_VarSetUpdate", "Commands.VarSetUpdateCommand", "VarSetUpdateCommand", "Detessellate Utilities", True),
    ("CreateSketchToolbar", "Commands.CreateSketchToolbarCommand", "CreateSketchToolbarCommand", "Detessellate Sketch", False),  # Menu only
    ("CreatePartDesignToolbar", "Commands.CreatePartDesignToolbarCommand", "CreatePartDesignToolbarCommand", "Detessellate Utilities", False),  # Menu only
    ("CreateGlobalToolbar", "Commands.CreateGlobalToolbarCommand", "CreateGlobalToolbarCommand", "Detessellate Utilities", False),  # Menu only
]

commands = {}

for cmd_name, module_path, class_name, toolbar, show_in_toolbar in command_specs:
    try:
        module = __import__(module_path, fromlist=[class_name])
        cmd_class = getattr(module, class_name)
        commands[cmd_name] = (cmd_class, toolbar, show_in_toolbar)

        # Register commands globally BEFORE workbench initialization
        FreeCADGui.addCommand(cmd_name, cmd_class())

        #print(f"{cmd_name} imported successfully")
    except Exception as e:
        print(f"ERROR importing {cmd_name}: {e}")
        traceback.print_exc()

#print("Detessellate workbench loaded")

class DetessellateWorkbench(FreeCADGui.Workbench):
    # Must be imported here otherwise "name 'Path' is not defined".
    from pathlib import Path

    MenuText = "Detessellate"
    ToolTip = "Tools to reverse engineering meshes"
    Icon = str(Path(FreeCAD.getUserAppDataDir()) / "Mod/Detessellate/Resources/icons/Detessellate.svg")

    def __init__(self):
        self._toolbar_created = False

    def Initialize(self):
        global commands
        # Commands are already registered globally, just add to toolbars/menus
        for cmd_name, (cmd_class, toolbar, show_in_toolbar) in commands.items():
            # Add to toolbar only if flagged True
            if show_in_toolbar:
                self.appendToolbar(toolbar, [cmd_name])

            # Always add to menu
            self.appendMenu("Detessellate", [cmd_name])

    def Activated(self):
        # Auto-create toolbars on first activation
        if not self._toolbar_created:
            self._auto_create_sketch_toolbar()
            self._auto_create_partdesign_toolbar()
            self._auto_create_global_toolbar()
            self._toolbar_created = True

    def Deactivated(self):
        pass

    def _auto_create_sketch_toolbar(self):
        """Automatically create the Sketcher toolbar"""
        try:
            # Directly run the command by name
            FreeCADGui.runCommand('CreateSketchToolbar')
            #FreeCAD.Console.PrintMessage("✓ Auto-created Detessellate Sketch Tools toolbar\n")
        except Exception as e:
            FreeCAD.Console.PrintWarning(f"Could not auto-create sketch toolbar: {e}\n")
            import traceback
            traceback.print_exc()

    def _auto_create_partdesign_toolbar(self):
        """Automatically create the PartDesign toolbar"""
        try:
            FreeCADGui.runCommand('CreatePartDesignToolbar')
            #FreeCAD.Console.PrintMessage("✓ Auto-created Detessellate PartDesign Tools toolbar\n")
        except Exception as e:
            FreeCAD.Console.PrintWarning(f"Could not auto-create PartDesign toolbar: {e}\n")

    def _auto_create_global_toolbar(self):
        """Automatically create the Global toolbar"""
        try:
            FreeCADGui.runCommand('CreateGlobalToolbar')
            #FreeCAD.Console.PrintMessage("✓ Auto-created Detessellate Global toolbar\n")
        except Exception as e:
            FreeCAD.Console.PrintWarning(f"Could not auto-create Global toolbar: {e}\n")

FreeCADGui.addWorkbench(DetessellateWorkbench())
