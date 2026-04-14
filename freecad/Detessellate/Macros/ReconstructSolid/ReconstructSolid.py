"""
ReconstructSolid - Rebuilds selected solid with geometric origin reset to match placement.
Useful for imported STEP files with embedded origin offsets.
"""

import FreeCAD, FreeCADGui, Draft

doc = FreeCAD.ActiveDocument
sel = FreeCADGui.Selection.getSelection()

if not sel:
    FreeCAD.Console.PrintError("⚠️ No object selected.\n")
else:
    obj = sel[0]
    doc.openTransaction("Rebuild Solid with Reset Origin")
    
    try:
        # Downgrade returns list of created objects
        faces_result = Draft.downgrade([obj], delete=True)
        doc.recompute()
        
        # faces_result is typically [list_of_new_objects, command_used]
        if faces_result and faces_result[0]:
            faces = faces_result[0]
            
            # Upgrade faces → shell
            shell_result = Draft.upgrade(faces, delete=True)
            doc.recompute()
            
            if shell_result and shell_result[0]:
                shell = shell_result[0]
                
                # Upgrade shell → solid
                solid_result = Draft.upgrade(shell, delete=True)
                doc.recompute()
                
                FreeCAD.Console.PrintMessage("✅ Solid reconstructed via Downgrade → Upgrade → Upgrade\n")
        
        doc.commitTransaction()
        
    except Exception as e:
        doc.abortTransaction()
        FreeCAD.Console.PrintError(f"❌ Error: {e}\n")