# Detessellate  
FreeCAD workbench of tools to reverse engineer meshes  

<img width="128" height="128" alt="Detessellate" src="https://github.com/user-attachments/assets/0c7ede91-acdf-4160-bc04-fc37f76c0e3c" />

FreeCAD Forum: https://forum.freecad.org/viewtopic.php?t=101467

Detessellate is a collection of FreeCAD macros that introduce an **algorithm-assisted workflow** for reverse engineering mesh models such as imported STL, OBJ, or 3MF files.  

📺 Click the image below to watch the demo video on YouTube

[![Watch the demo video](https://img.youtube.com/vi/QLw4me9nutA/maxresdefault.jpg)](https://www.youtube.com/watch?v=QLw4me9nutA)

## ✨ Workflow
1. Use **MeshPlacement** and **MeshToBody** to align and convert meshes to solids.  
2. Use **CoplanarSketch** to generate construction sketches for reconstructive solid features.  
3. Manually sketch or use **SketchReProfile** to automatically convert construction sketches to cleaner geometry. 
    - Potentially use **SketcherWireDoctor** (edge case only) to repair sketch errors prior to 3D feature creation.
4. Finish features using either **Part** or **PartDesign** workbenches as desired.  

## 📦 Included Macros
- <img width="25" height="25" alt="MeshPlacementIcon" src="https://github.com/user-attachments/assets/57233128-99af-42cd-b23f-17bc44b23b97" /> [MeshPlacement](https://github.com/DesignWeaver3D/Detessellate/blob/main/Macros/MeshPlacement/README.md) – recenter and align meshes to origin 
- <img width="25" height="25" alt="MeshToBodyIcon" src="https://github.com/user-attachments/assets/5ead9567-3c8c-40a1-a8f5-066e9259917e" /> [MeshToBody](https://github.com/DesignWeaver3D/Detessellate/blob/main/Macros/MeshToBody/README.md) – convert meshes into solids and bodies  
- <img width="25" height="25" alt="CoplanarSketch" src="https://github.com/user-attachments/assets/a941d04a-1707-400b-bd9c-d0751c8ea021" /> [CoplanarSketch](https://github.com/DesignWeaver3D/Detessellate/blob/main/Macros/CoplanarSketch/README.md) – generate construction sketches from coplanar edges on tessellated solids  
- <img width="25" height="25" alt="SketchReProfileIcon" src="https://github.com/user-attachments/assets/b21b52fa-843c-4c4d-8b63-4600f9488f41" /> [SketchReProfile](https://github.com/DesignWeaver3D/Detessellate/blob/main/Macros/SketchReProfile/README.md) – rebuild normal geometry profiles from construction sketches  
- <img width="25" height="25" alt="SketcherWireDoctorIcon" src="https://github.com/user-attachments/assets/21fd3989-5f19-4127-a680-0e17d17534ec" /> [SketcherWireDoctor](https://github.com/DesignWeaver3D/Detessellate/blob/main/Macros/SketcherWireDoctor/README.md) – repair and clean sketch wires  
- <img width="25" height="25" alt="EdgeLoopSelectorIcon" src="https://github.com/user-attachments/assets/e2de31f8-bf12-46f7-b5eb-5a7271ec598b" /> [EdgeLoopSelector](https://github.com/DesignWeaver3D/Detessellate/blob/main/Macros/EdgeLoopSelector/README.md) – select and process edge loops
- [EdgeLoopToSketch](https://github.com/DesignWeaver3D/Detessellate/blob/main/Macros/EdgeLoopToSketch/README.md) - Generate sketches from selected coplanar 3D edges, preserving edge type.
- <img width="25" height="25" alt="VarSetUpdateIcon" src="https://github.com/user-attachments/assets/9634b68f-6d81-4f1b-a367-122271b6bdc5" /> [VarSet-Update](https://github.com/DesignWeaver3D/Detessellate/blob/main/Macros/VarSet-Update/README.md) – update variable sets properties  
- <img width="25" height="25" alt="TopomatchSelectorIcon" src="https://github.com/user-attachments/assets/9bc7beca-cde8-40ee-b50c-679865858a58" /> [TopoMatchSelector](https://github.com/DesignWeaver3D/Detessellate/blob/main/Macros/TopoMatchSelector/README.md) – match and select topology from earlier body features  

> Some of these macros are included for convenience and are not strictly part of the Detessellate workflow.

## 🚀 Getting Started
Install via **FreeCAD Addon Manager** by adding the Detessellate Repository to the custom repositories list.
1. Open the Preferences via **Edit > Preferences**.
2. Go to **Addon Manager Options**.
3. Click the green **Plus** button under the **Custom Repository** list window to add to the list.
4. Paste the **Repository URL** into the **Custom Repository** dialog box: **https://github.com/DesignWeaver3D/Detessellate**
5. Type in the **Branch** name: **main**
6. Click **OK**
7. Open **Addon Manager** and find **Detessellate** in the list.
8. Click **Install**
10. **Restart** FreeCAD.
11. Access tools from the **Detessellate workbench** and/or the custom toolbars that the workbench creates.

<img width="885" height="932" alt="image" src="https://github.com/user-attachments/assets/8a0a0d23-7a0b-46d9-a032-3d1cb1f87fb2" />

## 🛣️ Roadmap
- 📚 Expanded documentation and tutorials  
- 🛠️ Additional utilities for Detessellate workflows  
- 🎯 Integration with FreeCAD Addon Manager

## 📜 Changelog
- **v0.1.6** (2025‑12-07) - Added EdgeLoopToSketch macro for creating sketches from EdgeLoopSelector
- **v0.1.5** (2025‑12-07) - Added multiple loop selection to EdgeLoopSelector
- **v0.1.4** (2025‑12‑02) – Fixed regression in CoplanarSketch Macro
- **v0.1.3** (2025‑12‑02) – Optimized CoplanarSketch Macro for better handling of larger edge counts
- **v0.1.2** (2025‑11‑30) – Improved tooltips for all toolbars  
- **v0.1.1** (2025‑11‑30) – Bug fix for `package.xml` for Addon Manager  
- **v0.1.0** (2025‑11‑26) – Initial release  

