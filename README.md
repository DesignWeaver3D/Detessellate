# Detessellate  
FreeCAD workbench of tools to reverse engineer meshes  

<img width="128" height="128" alt="Detessellate" src="https://github.com/user-attachments/assets/0c7ede91-acdf-4160-bc04-fc37f76c0e3c" />

Detessellate is a collection of FreeCAD macros that introduce an **algorithm-assisted workflow** for reverse engineering mesh models such as imported STL, OBJ, or 3MF files.  

## ✨ Workflow
1. Use **MeshPlacement** and **MeshToBody** to align and convert meshes to solids.  
2. Use **CoplanarSketch** to generate construction sketches for reconstructive solid features.  
3. Manually sketch or use **SketchReProfile** to automatically convert construction sketches to cleaner geometry. 
    - Potentially use **SketcherWireDoctor** (edge case only) to repair sketch errors prior to 3D feature creation.
4. Finish features using either **Part** or **PartDesign** workbenches as desired.  

## 📦 Included Macros
- [MeshPlacement](https://github.com/NSUBB/MeshPlacement) – recenter and align meshes to origin 
- [MeshToBody](https://github.com/NSUBB/MeshToBody) – convert meshes into solids and bodies  
- [CoplanarSketch](https://github.com/NSUBB/CoplanarSketch) – generate construction sketches from coplanar edges on tessellated solids  
- [SketchReProfile](https://github.com/NSUBB/SketchReProfile) – rebuild normal geometry profiles from construction sketches  
- [SketcherWireDoctor](https://github.com/NSUBB/SketcherWireDoctor) – repair and clean sketch wires  
- [EdgeLoopSelector](https://github.com/NSUBB/EdgeLoopSelector) – select and process edge loops  
- [VarSet-Update](https://github.com/NSUBB/VarSet-Update) – update variable sets properties  
- [TopoMatchSelector](https://github.com/NSUBB/TopoMatchSelector) – match and select topology from earlier body features  

> Some of these macros are included for convenience and are not strictly part of the Detessellate workflow.

## 🚀 Getting Started
1. ~~Install via **FreeCAD Addon Manager**~~ (hopefully coming soon) or download the Detessellate folder from this repo.  
2. Place the folder in your FreeCAD `Mod` directory.
   - Windows:  C:\Users\<username>\AppData\Roaming\FreeCAD\Mod
   - Linux: /home/<username>/.FreeCAD/Mod
   - macOS: /Users/<username>/Library/Preferences/FreeCAD/Mod
3. Restart FreeCAD.
4. Access tools from the **Detessellate workbench** and/or the custom toolbars that the workbench creates.   

## 📖 Roadmap
- 📚 Expanded documentation and tutorials  
- 🛠️ Additional utilities for Detessellate workflows  
- 🎯 Integration with FreeCAD Addon Manager  
