import FreeCAD
import FreeCADGui
from collections import defaultdict

def select_connected_loop_or_sketch():
    selection = FreeCADGui.Selection.getSelectionEx()
    if not selection:
        FreeCAD.Console.PrintError("Error: Please select one or more edges from a single loop.\n")
        return

    obj = selection[0].Object
    all_obj_edges = obj.Shape.Edges

    # --- Handle Sketches in 3D view ---
    if obj.TypeId.startswith("Sketcher::SketchObject"):
        selected_indices = []
        for sel_ex in selection:
            if sel_ex.Object.Name != obj.Name:
                FreeCAD.Console.PrintError("Error: Please select edges from only one sketch.\n")
                return
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

        # Find connected components
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

        # If only one edge selected, pick its loop
        if len(selected_indices) == 1:
            target = selected_indices[0]
            matching = next((comp for comp in components if target in comp), None)
        else:
            # Multiple edges selected: find loop that contains all
            selected_set = set(selected_indices)
            matching = next((comp for comp in components if selected_set.issubset(set(comp))), None)

        if not matching:
            FreeCAD.Console.PrintError("Error: Could not find a matching loop for the selected edge(s).\n")
            return

        FreeCADGui.Selection.clearSelection()
        for i in matching:
            FreeCADGui.Selection.addSelection(obj, f"Edge{i+1}")
        FreeCAD.Console.PrintMessage(f"Selected {len(matching)} edges from the sketch loop.\n")
        return

    # --- Handle Solids and Part shapes ---
    selected_edge_objects = []
    try:
        for sel_ex in selection:
            if sel_ex.Object.Name != obj.Name:
                FreeCAD.Console.PrintError("Error: Please select edges from only one object.\n")
                return
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

    start_edge = selected_edge_objects[0]
    parent_faces = []
    for face in obj.Shape.Faces:
        for face_edge in face.Edges:
            if face_edge.isSame(start_edge):
                parent_faces.append(face)
                break

    if not parent_faces:
        FreeCAD.Console.PrintError("Error: Could not find a parent face for the selected edge.\n")
        return

    found_loop = None
    for face in parent_faces:
        if found_loop:
            break
        for wire in face.Wires:
            wire_edges = wire.Edges
            if not any(edge.isSame(start_edge) for edge in wire_edges):
                continue
            if all(any(edge.isSame(e) for edge in wire_edges) for e in selected_edge_objects):
                found_loop = wire
                break

    if not found_loop:
        FreeCAD.Console.PrintError("Error: The selected edges do not all belong to the same continuous loop.\n")
        return

    FreeCADGui.Selection.clearSelection()
    selected_count = 0
    for edge_in_loop in found_loop.Edges:
        for idx, original_edge in enumerate(all_obj_edges):
            if original_edge.isSame(edge_in_loop):
                FreeCADGui.Selection.addSelection(obj, f"Edge{idx+1}")
                selected_count += 1
                break

    FreeCAD.Console.PrintMessage(f"Selected {selected_count} edges in the connected loop.\n")

# --- Run the macro ---
select_connected_loop_or_sketch()