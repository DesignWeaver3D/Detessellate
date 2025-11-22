# MeshToBody FreeCAD Macro (v2.0)
![MeshToBody](https://github.com/user-attachments/assets/5ead9567-3c8c-40a1-a8f5-066e9259917e)

The **MeshToBody** FreeCAD macro converts the selected or all mesh objects into a refined, simple solid or fusion of solids and integrates it into a **PartDesign Body**. 
Version **2.0** introduces a unified workflow with improved report messaging and robust cleanup.

📺 Click the image below to watch the demo video on YouTube

[![Watch the demo video](https://img.youtube.com/vi/EWC2T1qP_OI/maxresdefault.jpg)](https://www.youtube.com/watch?v=EWC2T1qP_OI)

https://forum.freecad.org/viewtopic.php?t=97579

## Features

- **Unified entry point**: Works on selected meshes or all meshes in the document
- **Pre‑collection & schedule**: Gathers facet/component counts and prints a sorted conversion plan
- **Smart evaluation**: Decides whether to proceed, split, repair, or fuse based on mesh state
- **Component splitting**: Handles multi‑component meshes safely, assembling solids into fusions
- **Conservative repair**: Attempts non‑destructive fixes (duplicate removal, hole filling, harmonized normals)
- **Guardrails**: Skips meshes with >50 components when multiple meshes are present
- **Granular feedback**: Console shows facet/component counts, warnings for large meshes, start markers, and elapsed times
- **Automatic cleanup**: Interim objects are removed on both success and failure
- Supports **undo transactions** for safe modifications.

## Requirements

- **FreeCAD** v1.0.2+

## Installation

1. Download `MeshToBody.py`  
2. Place it in your FreeCAD macros directory which can be determined in these locations:
   - `Edit > Preferences > Python > Macro > Macro path`
   - `Macro > Macros... > User macros location:`
3. Restart FreeCAD if already running

## Usage

1. Open a FreeCAD project  
2. Select one or more mesh objects, or leave no selection to process all meshes  
3. Run the macro via `Macro > Macros... > MeshToBody.py > Execute`
   - Optional: Use the included SVG icon for custom toolbar.
5. Watch the Report Viewer for the conversion schedule, progress updates, and summary

### Notes

- Meshes with >10k facets may take significant time  
- Meshes with >50 components are skipped when multiple meshes are present — rerun them individually  
- Failed conversions are logged and cleaned up automatically

## Output

- A new **PartDesign Body** per original mesh, containing a BaseFeature to the converted solid or fusion of solids 
- Skipped meshes remain in the document with reasons logged  
- Final summary reports converted vs. skipped counts and total runtime

## License

Licensed under the [GNU GPL v3.0](LICENSE)


## Contributing

Contributions are welcome! Feel free to submit **issues or pull requests** to improve the macro.

## Disclaimer

This macro is provided **"as is"** without any warranty. Use it at your own risk.

## Change Log
- v2.0.1 - 2025.11.18 Changed from making Part Compound to Part Fusion for better PartDesign body compatibility.
- v2.0.0 - 2025.11.16 Initial release of v2.0.
