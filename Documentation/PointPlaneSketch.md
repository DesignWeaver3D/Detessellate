# PointPlaneSketch
*Fit a sketch to scan points using RANSAC plane fitting*

A FreeCAD tool for creating datum planes and sketches from mesh point data. Select a handful of vertices from a scanned surface, and PointPlaneSketch fits a best-fit plane through them — handling noise and outliers — then creates a sketch with construction points you can trace over. Available as part of the Detessellate workbench, installable via the FreeCAD Addon Manager.

<img width="128" height="128" alt="PointPlaneSketch" src="https://github.com/user-attachments/assets/4ed28d6d-6908-47f5-bf3a-f589f030016a" />

📺 Click the image below to watch the demo video on YouTube

[![Watch the demo video](https://img.youtube.com/vi/jcQCEt7tGA4/maxresdefault.jpg)](https://www.youtube.com/watch?v=jcQCEt7tGA4)

## What it does

- Fits planes to noisy point clouds using RANSAC
- Lets you select 3+ vertices and refine with tolerance controls
- Aligns the plane normal toward the camera for consistent orientation
- Supports offset profile planes for capturing edge profiles through filleted geometry
- Provides visual feedback with highlighted points and normal indicators
- Outputs sketches & datum planes standalone or inside PartDesign bodies

The fitted plane aligns to your actual scan geometry regardless of where the mesh sits in 3D space — no need to align the mesh to the global origin first.

## Why not just manually place a datum plane?

With clean CAD geometry you can snap to faces and edges. With scan data, surfaces are noisy and there is nothing to snap to. RANSAC fitting finds the statistically best plane through scattered points, ignoring outliers. You get a plane that actually matches the surface, not one you eyeballed.

## Quick Start

1. **Import mesh or point cloud** into FreeCAD
2. **Create a points shape**: use `Part > Points from Shape` to create a selectable points object
3. **Select vertices**: pick 3 or more vertices that roughly define your plane
4. **Click the PointPlaneSketch icon in the Detessellate workbench**: a panel appears
5. **Adjust Tolerance**: control which points are included in the fit
6. **Update Preview**: see highlighted points that will be used
7. **Create Sketch**: generate the datum plane and sketch with construction points

## Profile Plane Points (Optional)

When a face has a filleted or rounded edge, the edge itself curves away from the face. If you want a sketch that traces the intended profile of that face — the boundary as if the fillet were not there — you need points sampled from the adjacent sides.

Profile Plane Points solves this by sampling a second set of points from a plane offset parallel to the base plane, then projecting them onto the base sketch.

1. Enter an **Offset Distance** (mm) — positive moves away from camera, negative moves toward
2. Set a **Profile Tolerance** for how thick the sampling band is
3. Click **Add Profile Plane Points** — a second color shows the captured points
4. Click **Create Sketch** — both sets appear as construction geometry in the sketch

The result: base points define the surface plane, profile points trace the actual edge boundary through filleted geometry.

## Controls

- **Tolerance**: how far from the fitted plane a point can be and still be included (mm)
- **Offset Distance**: distance from the base plane to the profile sampling plane (mm, accepts negative)
- **Profile Tolerance**: thickness of the profile sampling band (mm)
- **Highlight Color**: click swatches to change base or profile point preview colors
- **Update Preview**: recalculate with current settings
- **New Selection**: start over with a fresh vertex selection

## Output Options

- **Standalone (Part Workbench)**: independent datum plane and sketch using Placement
- **New Body (PartDesign)**: creates a new PartDesign body containing the datum plane and sketch
- **Existing Body**: adds the datum plane and sketch to a body you select

## Tips

- The centroid of your selected vertices becomes the sketch origin
- Increase tolerance to include more points from noisy scans; decrease for tighter fits
- Use negative offset distances to capture profiles on the camera-facing side
- For easier point selection:
  - Set the mesh object `View Property > Selectable` to `No`
  - Set the points object `View Property > Point Size` to `8` or as desired
  - Set the points object `View Property > On Top When Selected` to `Enabled`
- To align a mesh to the global origin: put the sketch and mesh in a Part container, then use the sketch as the alignment reference with the Transform tool
