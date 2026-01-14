import FreeCAD
import FreeCADGui
from collections import defaultdict


def selectEdges(obj, edges):
    FreeCADGui.Selection.clearSelection()
    names = [f"Edge{idx+1}" for idx in sorted(edges)]
    FreeCADGui.Selection.addSelection(obj, names)

def getPlaneNormal(unique_points, tolerance):
    if len(unique_points) < 3:
        return
    plane_point = unique_points[0]
    plane_normal = None
    for i in range(1, len(unique_points)):
        v1 = unique_points[i] - plane_point
        for j in range(i + 1, len(unique_points)):
            v2 = unique_points[j] - plane_point
            cross = v1.cross(v2)
            if cross.Length > tolerance:
                plane_normal = cross.normalize()
                break
        if plane_normal:
            return plane_normal

    return

def select_connected_loop_or_sketch():
    tolerance = 1e-6

    selection = FreeCADGui.Selection.getSelectionEx()
    if not selection:
        FreeCAD.Console.PrintError("Error: Please select one or more edges or faces.\n")
        return

    if len(selection) > 1:
        FreeCAD.Console.PrintError("Error: Please select elements from only one object.\n")
        return

    obj = selection[0].Object
    subObjects = selection[0].SubObjects
    all_obj_edges = obj.Shape.Edges
    all_obj_edges_hash = [e.hashCode() for e in all_obj_edges]

    # --- Check if faces are selected ---
    selected_faces = [sub for sub in subObjects if sub.ShapeType == "Face"]

    if selected_faces:
        # Handle face selection

        # Get plane from first face
        first_face = selected_faces[0]
        plane_normal = first_face.normalAt(0, 0).normalize()
        plane_point = first_face.valueAt(0, 0)

        # Validate all faces are coplanar (if multiple selected)
        if len(selected_faces) > 1:
            for face in selected_faces[1:]:
                face_normal = face.normalAt(0, 0).normalize()
                face_point = face.valueAt(0, 0)

                # Check if normals are parallel
                dot = abs(face_normal.dot(plane_normal))
                if abs(dot - 1.0) > tolerance:
                    FreeCAD.Console.PrintError("Error: Selected faces are not coplanar.\n")
                    return

                # Check if face lies on same plane
                distance = abs((face_point - plane_point).dot(plane_normal))
                if distance > tolerance:
                    FreeCAD.Console.PrintError("Error: Selected faces are not coplanar.\n")
                    return

        # Collect all edge loops from selected faces
        all_edges_to_select = set()
        for face in selected_faces:
            for edge in face.Edges:
                edge_hash = edge.hashCode()
                if edge_hash in all_obj_edges_hash:
                    idx = all_obj_edges_hash.index(edge_hash)
                    all_edges_to_select.add(idx)

        # Select all edges
        selectEdges(obj, all_edges_to_select)

        FreeCAD.Console.PrintMessage(f"Selected {len(all_edges_to_select)} edges from {len(selected_faces)} face(s).\n")
        return

    # --- Handle objects with Wires but no Faces (e.g., SubShapeBinder from sketch) ---
    if not obj.Shape.Faces and obj.Shape.Wires:

        selected_edge_objects = [sub for sub in subObjects if sub.ShapeType == "Edge"]
        if not selected_edge_objects:
            FreeCAD.Console.PrintError("Error: No valid edges were selected.\n")
            return

        # Find which wires contain the selected edges
        wires_to_select = set()
        for selected_edge in selected_edge_objects:
            selected_edge_hash = selected_edge.hashCode()
            for wire in obj.Shape.Wires:
                wire_edges_hash = [e.hashCode() for e in wire.Edges]
                # Check if this edge is in this wire
                if selected_edge_hash in wire_edges_hash:
                    wires_to_select.add(wire)

        if not wires_to_select:
            FreeCAD.Console.PrintError("Error: Could not find wires containing the selected edges.\n")
            return

        # Collect all edge indices from selected wires
        all_edges_to_select = set()
        for wire in wires_to_select:
            wire_edges_hash = [e.hashCode() for e in wire.Edges]
            for wire_edge_hash in wire_edges_hash:
                if wire_edge_hash in all_obj_edges_hash:
                    idx = all_obj_edges_hash.index(wire_edge_hash)
                    all_edges_to_select.add(idx)

        
        # Select all edges
        selectEdges(obj, all_edges_to_select)
        
        # Perform coplanarity check AFTER selection for warning purposes
        if len(wires_to_select) > 1:
            # Collect all points from all selected wires
            all_wire_points = [p for w in wires_to_select for e in w.Edges for p in e.discretize(4)]

            # Try to find a plane from all wire points
            unique_points = []
            for pt in all_wire_points:
                if not any(pt.isEqual(existing, tolerance) for existing in unique_points):
                    unique_points.append(pt)
                    if len(unique_points) >= 100:  # Limit for performance
                        break
            
            plane_normal = getPlaneNormal(unique_points, tolerance)

            # Check if all wires are coplanar
            if plane_normal:
                plane_point = unique_points[0]
                all_coplanar = True
                for pt in all_wire_points:
                    distance = abs((pt - plane_point).dot(plane_normal))
                    if distance > tolerance:
                        all_coplanar = False
                        break
                
                if not all_coplanar:
                    FreeCAD.Console.PrintWarning("Warning: Selected wires are not coplanar.\n")
        
        FreeCAD.Console.PrintMessage(f"Selected {len(all_edges_to_select)} edges from {len(wires_to_select)} wire(s).\n")
        return

    # --- Handle Sketches in 3D view ---
    if obj.TypeId.startswith("Sketcher::SketchObject"):
        selected_indices = []
        for sel_ex in selection:
            for edge_name in sel_ex.SubElementNames:
                if edge_name.startswith("Edge"):
                    edge_idx = int(edge_name[4:]) - 1
                    selected_indices.append(edge_idx)

        if not selected_indices:
            FreeCAD.Console.PrintError("Error: No valid sketch edges were selected.\n")
            return

        # Build vertex-to-edge map using hashable keys
        vertex_map = defaultdict(list)
        for i, edge in enumerate(all_obj_edges):
            for v in [edge.Vertexes[0], edge.Vertexes[-1]]:
                vertex_map[tuple(v.Point)].append(i)

        # Build connectivity graph
        edge_graph = defaultdict(set)
        for i, edge in enumerate(all_obj_edges):
            v1 = tuple(edge.Vertexes[0].Point)
            v2 = tuple(edge.Vertexes[-1].Point)
            for neighbor in vertex_map[v1]:
                if neighbor != i:
                    edge_graph[i].add(neighbor)
            for neighbor in vertex_map[v2]:
                if neighbor != i:
                    edge_graph[i].add(neighbor)

        # Find connected components using DFS
        def dfs(start, visited):
            group = []
            stack = [start]
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                group.append(current)
                stack.extend(edge_graph[current])
            return group

        visited = set()
        components = []
        for i in range(len(all_obj_edges)):
            if i not in visited:
                comp = dfs(i, visited)
                components.append(comp)

        # Find all unique loops containing selected edges
        unique_loops = []
        seen_loop_sets = []

        for edge_idx in selected_indices:
            matching_comp = next((comp for comp in components if edge_idx in comp), None)
            if matching_comp:
                loop_set = frozenset(matching_comp)
                if loop_set not in seen_loop_sets:
                    seen_loop_sets.append(loop_set)
                    unique_loops.append(matching_comp)

        if not unique_loops:
            FreeCAD.Console.PrintError("Error: Could not find a matching loop for the selected edge(s).\n")
            return

        # Select all edges from all unique loops
        all_edges_to_select = set()
        for loop in unique_loops:
            all_edges_to_select.update(loop)

        selectEdges(obj, all_edges_to_select)

        FreeCAD.Console.PrintMessage(f"Selected {len(all_edges_to_select)} edges from {len(unique_loops)} loop(s).\n")
        return

    # --- Handle Solids and Part shapes ---
    selected_edge_objects = []
    try:
        for sel_ex in selection:
            for edge_name in sel_ex.SubElementNames:
                if edge_name.startswith("Edge"):
                    edge_idx = int(edge_name[4:]) - 1
                    selected_edge_objects.append(all_obj_edges[edge_idx])
        if not selected_edge_objects:
            FreeCAD.Console.PrintError("Error: No valid edges were selected.\n")
            return
    except Exception as e:
        FreeCAD.Console.PrintError(f"Error processing selection: {e}\n")
        return

    # Validate at least 2 edges selected for 3D objects
    if len(selected_edge_objects) < 2:
        FreeCAD.Console.PrintError("Error: Please select at least 2 edges from a 3D object to define the plane.\n")
        return

    # Collect unique vertex points from selected edges
    unique_points = []
    all_points = []
    for edge in selected_edge_objects:
        for vertex in edge.Vertexes:
            pt = vertex.Point
            all_points.append(pt)
            if not any(pt.isEqual(existing, tolerance) for existing in unique_points):
                unique_points.append(pt)

    if len(unique_points) < 3:
        FreeCAD.Console.PrintError("Error: Selected edges do not provide enough unique points to define a plane.\n")
        return

    # Find 3 non-collinear points to define plane
    plane_normal = getPlaneNormal(unique_points, tolerance)

    if not plane_normal:
        FreeCAD.Console.PrintError("Error: Selected edges are collinear and cannot define a plane.\n")
        return

    # Check all points lie on the defined plane
    plane_point = unique_points[0]
    for pt in all_points:
        distance = abs((pt - plane_point).dot(plane_normal))
        if distance > tolerance:
            FreeCAD.Console.PrintError("Error: Selected edges are not coplanar.\n")
            return

    # Find all unique loops containing selected edges on coplanar faces
    unique_loop_sets = []

    for start_edge in selected_edge_objects:
        # Find parent faces that contain this edge
        parent_faces = []
        for face in obj.Shape.Faces:
            for face_edge in face.Edges:
                if face_edge.isSame(start_edge):
                    parent_faces.append(face)
                    break

        if not parent_faces:
            FreeCAD.Console.PrintWarning("Warning: Could not find a parent face for a selected edge.\n")
            continue

        # Filter to only coplanar faces
        coplanar_faces = []
        for face in parent_faces:
            # Get face normal and a point on the face
            face_normal = face.normalAt(0, 0)
            face_point = face.valueAt(0, 0)

            # Check if normals are parallel (dot product near ±1)
            dot = abs(face_normal.normalize().dot(plane_normal))
            if abs(dot - 1.0) < tolerance:
                # Check if face lies on same plane
                distance = abs((face_point - plane_point).dot(plane_normal))
                if distance < tolerance:
                    coplanar_faces.append(face)

        if not coplanar_faces:
            FreeCAD.Console.PrintWarning("Warning: Could not find a coplanar face for a selected edge.\n")
            continue

        # Find wire on coplanar faces containing this edge
        found_loop = None
        for face in coplanar_faces:
            if found_loop:
                break
            for wire in face.Wires:
                wire_edges = wire.Edges
                if any(edge.isSame(start_edge) for edge in wire_edges):
                    found_loop = wire
                    break

        if found_loop:
            # Convert wire to set of edge indices
            loop_indices = set()
            for edge_in_loop in found_loop.Edges:
                edge_in_loop_hash = edge_in_loop.hashCode()
                if edge_in_loop_hash in all_obj_edges_hash:
                    idx = all_obj_edges_hash.index(edge_in_loop_hash)
                    loop_indices.add(idx)

            # Check if we've already found this loop
            loop_frozen = frozenset(loop_indices)
            if loop_frozen not in unique_loop_sets:
                unique_loop_sets.append(loop_frozen)

    if not unique_loop_sets:
        FreeCAD.Console.PrintError("Error: Could not find loops for the selected edges.\n")
        return

    # Select all edges from all unique loops
    all_edges_to_select = set()
    for loop_set in unique_loop_sets:
        all_edges_to_select.update(loop_set)

    selectEdges(obj, all_edges_to_select)

    FreeCAD.Console.PrintMessage(f"Selected {len(all_edges_to_select)} edges from {len(unique_loop_sets)} loop(s).\n")

# --- Run the macro ---
select_connected_loop_or_sketch()
