# ReconstructSolid

ReconstructSolid rebuilds a selected solid so its geometric origin is reset to match its current placement.
This is especially useful for imported STEP files that contain embedded origin offsets.

<img width="128" height="128" alt="ReconstructSolid" src="https://github.com/user-attachments/assets/4ee570fd-bbea-452f-95af-8e8e17ed9ac9" />

## Overview

Some STEP imports contain solids whose internal geometry is offset relative to the global origin.
ReconstructSolid resolves this by performing a controlled Downgrade → Upgrade → Upgrade sequence,
producing a new solid whose geometric origin is aligned to its placement transform relative to the global origin.

The original object is removed during the downgrade step, and a new solid is created in its place.

## Features

- Useful for STEP files with offset geometric origin
- Resets geometric origin to match the object’s placement
- Rebuilds solids using Draft’s downgrade/upgrade pipeline
- Runs inside a single document transaction for clean undo/redo
- Provides clear console messages for success or failure

## Usage

1. Select a solid object in the Model Tree.
2. From Detessellate wokbench, click the ReconstructSolid icon or select from the menu Detessellate > ReconstructSolid.
3. A new solid will be created with corrected geometry.
4. Check the Report View for status messages.

## Notes

- Behavior depends on the topology of the selected object.
- Works best on clean STEP solids with well‑formed faces.
- If the operation fails, inspect the Report View for details.

## Installation

This macro is bundled with the [Detessellate Workbench](https://github.com/yourusername/Detessellate), but can also be installed separately.

### Manual Installation

1. Download `ReconstructSolid.py`
2. Place or Copy the downloaded file in the Macro folder
  - In FreeCAD, Macro folder path can be found by going to: 
      - `Macro → Macros...` shown in `User macros location`
      - Or via Preferences > Python > Macro > Macro Path
3. Close and reopen the Macro dialog or restart FreeCAD

