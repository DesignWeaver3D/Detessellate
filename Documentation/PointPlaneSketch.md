# PointPlaneSketch

A FreeCAD macro for creating datum planes and sketches from mesh point data. It uses RANSAC plane fitting on user‑selected points, making it easier to turn scan data into usable geometry.

<img width="128" height="128" alt="PointPlaneSketch" src="https://github.com/user-attachments/assets/4ed28d6d-6908-47f5-bf3a-f589f030016a" />

📺 Click the image below to watch the demo video on YouTube

[![Watch the demo video](https://img.youtube.com/vi/jcQCEt7tGA4/maxresdefault.jpg)](https://www.youtube.com/watch?v=jcQCEt7tGA4)

## What it does

- Fits planes to noisy point clouds using RANSAC.
- Lets you select 3+ vertices and refine with tolerance controls.
- Aligns the plane normal toward the camera for consistent orientation.
- Supports offset profile planes for capturing edges and fillets.
- Provides visual feedback with highlighted points and normal indicators.
- Outputs sketches either standalone or inside PartDesign bodies.

## Why it's useful

PointPlaneSketch reduces the manual effort of aligning datum planes to scan data. Instead of guessing or adjusting geometry by hand, you can interactively select points, preview the fit, and generate sketches that match your model's orientation.

## Quick Start

1. **Import mesh or point cloud**: Points shape object can be derived from either 
2. **Create points shape**: Use `Part > Points from Shape` to create selectabe points shape object
3. **Select vertices**: Pick 3 or more vertices that roughly define your plane
4. **Run macro**: Execute PointPlaneSketch—a docker window appears
5. **Adjust tolerance**: Fine-tune which points are included in the fit
6. **Update preview**: See highlighted points that will be used
7. **Create sketch**: Generate the datum plane and sketch with construction points for either Part or PartDesign

## Profile Plane Points (Optional)

For objects with filleted edges where the outer profile sits offset from the base:

1. Enter an **offset distance** (positive = away from camera, negative = toward)
2. Set a **profile tolerance**
3. Click **"Add Profile Plane Points"**—a second set of points highlights in a different color
4. Click **"Create Sketch"** to include both base and profile points as construction geometry

## Controls

- **Tolerance**: Distance threshold (mm) for including points in the base plane
- **Offset Distance**: How far (mm) to offset the profile plane from the base (accepts negative values)
- **Profile Tolerance**: Distance threshold (mm) for profile plane points
- **Highlight Color**: Click swatches to change base or profile point colors
- **Update Preview**: Recalculate everything based on current settings
- **New Selection**: Start over with a fresh selection

## Output Options

- **Standalone (Part Workbench)**: Independent datum plane and sketch using Placement
- **New Body (PartDesign)**: Creates a new PartDesign body containing the datum and sketch using Attachment Offset
- **Existing Body**: Adds the datum and sketch to a body you select

## Requirements

- FreeCAD 1.0 or later
- Python numpy

## Alternative Installation

This macro is bundled with the Detessellate Workbench, but can also be installed as a standalone macro.

1. Download `PointPlaneSketch.py` from this repository.
2. Place it in your FreeCAD macros directory:
   - `Macro → Macros...` → **User macros location**, or
   - `Edit → Preferences → Python → Macro → Macro path`
3. Restart FreeCAD, or click **Refresh** in the Macro dialog.
4. (Optional) Copy the matching icon from `Resources/Icons/PointPlaneSketch.svg` for use as a custom toolbar button.

## Tips

- Selected vertices will define the the created sketch origin
- Increase tolerance to include more points from noisy scans
- Decrease tolerance for tighter plane definitions
- Use negative offset distances to capture profiles toward the camera
- Profile points are particularly useful for rounded edges and chamfers
- For easier point selection
  - Set original mesh or points object `View Property > Selectable` to `No`
  - Set point object `View Property > Point Size` to ≥ `8`
  - Set point object `View Property > On Top When Selected` to `Enabled`
