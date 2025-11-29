# Detessellate  
FreeCAD workbench of tools to reverse engineer meshes  

<img width="128" height="128" alt="Detessellate" src="https://github.com/user-attachments/assets/0c7ede91-acdf-4160-bc04-fc37f76c0e3c" />

FreeCAD Forum: https://forum.freecad.org/viewtopic.php?t=101467

Detessellate is a collection of FreeCAD macros that introduce an **algorithm-assisted workflow** for reverse engineering mesh models such as imported STL, OBJ, or 3MF files.  

## ✨ Workflow
1. Use **MeshPlacement** and **MeshToBody** to align and convert meshes to solids.  
2. Use **CoplanarSketch** to generate construction sketches for reconstructive solid features.  
3. Manually sketch or use **SketchReProfile** to automatically convert construction sketches to cleaner geometry. 
    - Potentially use **SketcherWireDoctor** (edge case only) to repair sketch errors prior to 3D feature creation.
4. Finish features using either **Part** or **PartDesign** workbenches as desired.  

## 📦 Included Macros
- <img width="25" height="25" alt="MeshPlacementIcon" src="https://github.com/user-attachments/assets/57233128-99af-42cd-b23f-17bc44b23b97" /> [MeshPlacement](https://github.com/NSUBB/MeshPlacement) – recenter and align meshes to origin 
- <img width="25" height="25" alt="MeshToBodyIcon" src="https://github.com/user-attachments/assets/5ead9567-3c8c-40a1-a8f5-066e9259917e" /> [MeshToBody](https://github.com/NSUBB/MeshToBody) – convert meshes into solids and bodies  
- <img width="25" height="25" alt="CoplanarSketch" src="https://github.com/user-attachments/assets/a941d04a-1707-400b-bd9c-d0751c8ea021" /> [CoplanarSketch](https://github.com/NSUBB/CoplanarSketch) – generate construction sketches from coplanar edges on tessellated solids  
- <img width="25" height="25" alt="SketchReProfileIcon" src="https://github.com/user-attachments/assets/b21b52fa-843c-4c4d-8b63-4600f9488f41" /> [SketchReProfile](https://github.com/NSUBB/SketchReProfile) – rebuild normal geometry profiles from construction sketches  
- <img width="25" height="25" alt="SketcherWireDoctorIcon" src="https://github.com/user-attachments/assets/21fd3989-5f19-4127-a680-0e17d17534ec" /> [SketcherWireDoctor](https://github.com/NSUBB/SketcherWireDoctor) – repair and clean sketch wires  
- <img width="25" height="25" alt="EdgeLoopSelectorIcon" src="https://github.com/user-attachments/assets/e2de31f8-bf12-46f7-b5eb-5a7271ec598b" /> [EdgeLoopSelector](https://github.com/NSUBB/EdgeLoopSelector) – select and process edge loops  
- <img width="25" height="25" alt="VarSetUpdateIcon" src="https://github.com/user-attachments/assets/9634b68f-6d81-4f1b-a367-122271b6bdc5" /> [VarSet-Update](https://github.com/NSUBB/VarSet-Update) – update variable sets properties  
- <img width="25" height="25" alt="TopomatchSelectorIcon" src="https://github.com/user-attachments/assets/9bc7beca-cde8-40ee-b50c-679865858a58" /> [TopoMatchSelector](https://github.com/NSUBB/TopoMatchSelector) – match and select topology from earlier body features  

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
