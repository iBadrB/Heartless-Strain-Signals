import re
import numpy as np
import xml.etree.ElementTree as et
from matplotlib import pyplot as plt
from plotly import graph_objects as go
from pathlib import Path

from GEOSMesh import GEOSMesh
from GEOSBox import GEOSBox
from GEOSThickPlane import GEOSThickPlane
from GEOSRectangle import GEOSRectangle


# -----------------------------------
# |         XML Decoding            |
# -----------------------------------

def parse_str(input_str: str) -> list:
    """
    Parse GEOS strings from XML inputs if it matches '{ +-n1, +-n2, +-n3 }.

    Args:
        input_str (str): String from XML input.

    Returns:
        list: List of values converted from the input string.
        or
        str: String if the input does not match the hardcoded pattern.
    """
    pattern = r"\{\s*([-\w.]+(?:\s*,\s*[-\w.]+)*)\s*\}" # this pattern matches '{ (int, float, word), (int, float, word), ..... }'
    match = re.search(pattern, input_str) # check if input string matches the pattern

    if match is not None: # if there is a match
        temp_str = match.group(1) # returns the first group of elements to match the pattern without the parentheses
        arr = [v for v in temp_str.split(", ")] # split the string into individual elements
        
        ret_arr = []
        for elm in arr:
            # attempt conversion to numbers
            try:
                num = float(elm) # convert to float
                if num.is_integer(): # if it is an int, append the int
                    ret_arr.append(int(num))
                else: # else append the float
                    ret_arr.append(num) 
            # if it fails, keep the string
            except ValueError:
                ret_arr.append(elm)
        return ret_arr
    # no match, return input as is
    return input_str

def parse_xml(xml_str: str) -> list[dict]:
    """
    Parse XML GEOS string for later use.

    Args:
        xml_str (str): String of GEOS XML snippet.

    Returns:
        list[dict]: List of dictionaries containing parsed GEOS parameters.
    """
    options = {
        "InternalMesh":["name", "xCoords", "yCoords", "zCoords", "nx", "ny", "nz", "xBias", "yBias", "zBias", "cellBlockNames", "elementTypes"], 
        "Box": ["name", "xMin", "xMax"], 
        "ThickPlane": ["name", "normal", "origin", "thickness"],
        "Rectangle": ["name", "normal", "origin", "lengthVector", "widthVector", "dimensions"],
    }
    root = et.fromstring(xml_str)
    ret_val, arr = [], []
    
    for k, v in options.items():
        arr = root.findall(f".//{k}")

        if len(arr) > 0:
            for elm in arr:
                _dict = {"tag": elm.tag}
                for key in v:
                    if "Bias" in key:
                        try:
                            _dict[key] = parse_str(elm.get(key))
                        except:
                            _dict[key] = None
                    else:
                        _dict[key] = parse_str(elm.get(key))
                ret_val.append(_dict)
            arr = []
            
    return ret_val


# -----------------------------------
# |         Mesh Construction       |
# -----------------------------------

def _as_list(value: object | list[object] | tuple[object, ...] | None) -> list[object]:
    """
    Convert an optional scalar/list input into a list.

    Args:
        value (object | list[object] | tuple[object, ...] | None): Input value.

    Returns:
        list[object]: List representation of the input.
    """
    # convert a single value into a one-item list for unified processing
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]

def _plane_config(plane: str) -> tuple[str, list[int], int, list[str], str]:
    """
    Resolve plotting-axis configuration for a requested slice plane.

    Args:
        plane (str): Plane string ('xy', 'xz', or 'yz').

    Returns:
        tuple[str, list[int], int, list[str], str]:
            Plane key, in-plane axis indices, perpendicular axis index, axis labels,
            and perpendicular axis label.
    """
    # normalize and validate requested slice plane
    plane = plane.lower()
    if plane not in ["xy", "xz", "yz"]:
        raise ValueError("plane must be 'xy', 'xz', or 'yz'")

    # map selected 2D plane to in-plane axes and perpendicular axis
    if plane == "xy":
        return plane, [0, 1], 2, ["X", "Y"], "Z"
    if plane == "xz":
        return plane, [0, 2], 1, ["X", "Z"], "Y"
    return plane, [1, 2], 0, ["Y", "Z"], "X"


def _element_faces_by_type(element_node_count: int) -> list[list[int]]:
    """
    Return local face connectivity for supported solid element topologies.

    Args:
        element_node_count (int): Number of nodes in the element connectivity.

    Returns:
        list[list[int]]: Faces as local node-index lists.
    """
    if element_node_count == 8:
        # C3D8 hexahedron: 6 quad faces
        return [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [1, 2, 6, 5],
            [2, 3, 7, 6],
            [3, 0, 4, 7],
        ]
    if element_node_count == 6:
        # C3D6 wedge/prism: 2 triangular and 3 quad faces
        return [
            [0, 1, 2],
            [3, 4, 5],
            [0, 1, 4, 3],
            [1, 2, 5, 4],
            [2, 0, 3, 5],
        ]
    raise ValueError(f"Unsupported element with {element_node_count} nodes (supported: C3D8, C3D6)")


def _face_edges(face_nodes: list[int]) -> list[tuple[int, int]]:
    """Build local edges from one face definition."""
    ret = []
    n = len(face_nodes)
    for i in range(n):
        i0 = int(face_nodes[i])
        i1 = int(face_nodes[(i + 1) % n])
        ret.append((min(i0, i1), max(i0, i1)))
    return ret


def _mesh_element_slice_polygon(
    elem_nodes: np.ndarray,
    axis_indices: list[int],
    perp_axis: int,
    position: float,
    tol: float = 1e-9,
) -> np.ndarray | None:
    """
    Intersect one solid element (C3D8 or C3D6) with a slice plane and return 2D polygon.

    Args:
        elem_nodes (np.ndarray): Element node coordinates, shape (N, 3), with N in {6, 8}.
        axis_indices (list[int]): In-plane axis indices.
        perp_axis (int): Perpendicular axis index.
        position (float): Slice location on perpendicular axis.
        tol (float): Numerical tolerance.

    Returns:
        np.ndarray | None: Ordered 2D polygon points or None when no intersection exists.
    """
    elem_nodes = np.asarray(elem_nodes, dtype=float)
    faces = _element_faces_by_type(elem_nodes.shape[0])

    # collect unique element edges from all faces
    edge_set = set()
    for face in faces:
        edge_set.update(_face_edges(face))

    intersections = []
    for i0, i1 in edge_set:
        p0 = elem_nodes[i0]
        p1 = elem_nodes[i1]
        d0 = p0[perp_axis] - position
        d1 = p1[perp_axis] - position

        if abs(d0) < tol and abs(d1) < tol:
            intersections.append(p0[axis_indices])
            intersections.append(p1[axis_indices])
        elif d0 * d1 < 0 or abs(d0) < tol or abs(d1) < tol:
            t = (position - p0[perp_axis]) / (p1[perp_axis] - p0[perp_axis] + 1e-30)
            if -tol <= t <= 1 + tol:
                intersections.append((p0 + t * (p1 - p0))[axis_indices])

    return _order_unique_polygon(intersections)

def _order_unique_polygon(points: list[np.ndarray]) -> np.ndarray | None:
    """
    De-duplicate and angularly order 2D points into a polygon.

    Args:
        points (list[np.ndarray]): Unordered 2D points.

    Returns:
        np.ndarray | None: Ordered polygon points or None if fewer than 3 unique points.
    """
    # return no polygon when there are no points
    if not points:
        return None

    # remove duplicate points introduced by shared edges or faces
    pts = np.asarray(points, dtype=float)
    rounded = np.round(pts, 12)
    _, unique_idx = np.unique(rounded, axis=0, return_index=True)
    pts = pts[np.sort(unique_idx)]

    # need at least 3 points to define a polygon
    if len(pts) < 3:
        return None

    # order points around centroid for a valid polygon path
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    return pts[np.argsort(angles)]

def _box_slice_polygon(
    box_obj: GEOSBox,
    axis_indices: list[int],
    perp_axis: int,
    position: float,
    tol: float = 1e-9,
) -> np.ndarray | None:
    """
    Compute the 2D polygon from intersecting a GEOS box with a slice plane.

    Args:
        box_obj (GEOSBox): Box object to be sliced.
        axis_indices (list[int]): In-plane axis indices.
        perp_axis (int): Perpendicular axis index.
        position (float): Slice location on the perpendicular axis.
        tol (float): Numerical tolerance for intersection tests.

    Returns:
        np.ndarray | None: Ordered polygon points in 2D or None if no intersection.
    """
    # retrieve box vertices and edge connectivity
    vertices = np.asarray(box_obj.get_vertices(), dtype=float)
    edges = np.asarray(box_obj.get_edges())
    intersections = []

    # intersect each box edge with the slice
    for edge in edges:
        i0 = int(edge[0])
        i1 = int(edge[1])
        p0 = vertices[i0]
        p1 = vertices[i1]
        d0 = p0[perp_axis] - position
        d1 = p1[perp_axis] - position

        # entire edge lies on the slicing plane
        if abs(d0) < tol and abs(d1) < tol:
            intersections.append(p0[axis_indices])
            intersections.append(p1[axis_indices])
        # edge crosses or touches the slicing plane
        elif d0 * d1 < 0 or abs(d0) < tol or abs(d1) < tol:
            t = (position - p0[perp_axis]) / (p1[perp_axis] - p0[perp_axis] + 1e-30)
            if -tol <= t <= 1 + tol:
                intersections.append((p0 + t * (p1 - p0))[axis_indices])

    # create an ordered polygon from intersection points
    return _order_unique_polygon(intersections)

def _thick_plane_slice_polygon(
    thick_plane_obj: GEOSThickPlane,
    axis_indices: list[int],
    perp_axis: int,
    position: float,
    size: float = 10.0,
    tol: float = 1e-9,
) -> np.ndarray | None:
    """
    Compute the 2D polygon from intersecting a GEOS thick plane with a slice plane.

    Args:
        thick_plane_obj (GEOSThickPlane): Thick-plane object to be sliced.
        axis_indices (list[int]): In-plane axis indices.
        perp_axis (int): Perpendicular axis index.
        position (float): Slice location on the perpendicular axis.
        size (float): Rectangular patch half-size used for thick-plane rendering.
        tol (float): Numerical tolerance for intersection tests.

    Returns:
        np.ndarray | None: Ordered polygon points in 2D or None if no intersection.
    """
    # retrieve quad faces for the thick-plane prism
    faces = thick_plane_obj.get_faces(size=size)
    intersections = []

    # intersect each face edge with the slice
    for face in faces:
        face = np.asarray(face, dtype=float)
        n_face = len(face)
        for i in range(n_face):
            p0 = face[i]
            p1 = face[(i + 1) % n_face]
            d0 = p0[perp_axis] - position
            d1 = p1[perp_axis] - position

            # entire edge lies on the slicing plane
            if abs(d0) < tol and abs(d1) < tol:
                intersections.append(p0[axis_indices])
                intersections.append(p1[axis_indices])
            # edge crosses or touches the slicing plane
            elif d0 * d1 < 0 or abs(d0) < tol or abs(d1) < tol:
                t = (position - p0[perp_axis]) / (p1[perp_axis] - p0[perp_axis] + 1e-30)
                if -tol <= t <= 1 + tol:
                    intersections.append((p0 + t * (p1 - p0))[axis_indices])

    # create an ordered polygon from intersection points
    return _order_unique_polygon(intersections)


def _rectangle_slice_polygon(
    rectangle_obj: GEOSRectangle,
    axis_indices: list[int],
    perp_axis: int,
    position: float,
    tol: float = 1e-9,
) -> np.ndarray | None:
    """
    Compute the 2D polygon from intersecting a GEOS rectangle with a slice plane.

    Args:
        rectangle_obj (GEOSRectangle): Rectangle object to be sliced.
        axis_indices (list[int]): In-plane axis indices.
        perp_axis (int): Perpendicular axis index.
        position (float): Slice location on the perpendicular axis.
        tol (float): Numerical tolerance for intersection tests.

    Returns:
        np.ndarray | None: Ordered polygon points in 2D or None if no intersection.
    """
    corners = np.asarray(rectangle_obj.get_corners(), dtype=float)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    intersections = []

    # intersect each rectangle edge with the slice
    for i0, i1 in edges:
        p0 = corners[i0]
        p1 = corners[i1]
        d0 = p0[perp_axis] - position
        d1 = p1[perp_axis] - position

        # entire edge lies on the slicing plane
        if abs(d0) < tol and abs(d1) < tol:
            intersections.append(p0[axis_indices])
            intersections.append(p1[axis_indices])
        # edge crosses or touches the slicing plane
        elif d0 * d1 < 0 or abs(d0) < tol or abs(d1) < tol:
            t = (position - p0[perp_axis]) / (p1[perp_axis] - p0[perp_axis] + 1e-30)
            if -tol <= t <= 1 + tol:
                intersections.append((p0 + t * (p1 - p0))[axis_indices])

    # create an ordered polygon from intersection points
    return _order_unique_polygon(intersections)

def _compute_slice_geometry(
    geometry: GEOSMesh | GEOSBox | GEOSThickPlane | GEOSRectangle,
    boxes: GEOSBox | list[GEOSBox] | None,
    thick_planes: GEOSThickPlane | list[GEOSThickPlane] | None,
    rectangles: GEOSRectangle | list[GEOSRectangle] | None,
    plane: str,
    position: float | None,
    thick_plane_size: float = 10.0,
) -> dict[str, object]:
    """
    Build shared 2D slice geometry data for matplotlib and plotly renderers.

    Args:
        geometry (GEOSMesh | GEOSBox | GEOSThickPlane | GEOSRectangle): Primary geometry object.
        boxes (GEOSBox | list[GEOSBox] | None): Optional overlay box or boxes.
        thick_planes (GEOSThickPlane | list[GEOSThickPlane] | None): Optional thick-plane overlay(s).
        rectangles (GEOSRectangle | list[GEOSRectangle] | None): Optional rectangle overlay(s).
        plane (str): Plane string ('xy', 'xz', or 'yz').
        position (float | None): Slice location along perpendicular axis.
        thick_plane_size (float): Rectangular patch half-size for thick-plane faces.

    Returns:
        dict[str, object]: Geometry payload including polygons, labels, and bounds.
    """
    # derive axis config for the selected plane
    plane, axis_indices, perp_axis, axis_labels, perp_label = _plane_config(plane)

    # determine whether primary geometry behaves like mesh, box, or thick plane
    is_mesh = hasattr(geometry, "nodes") and hasattr(geometry, "elements")
    is_box = hasattr(geometry, "get_vertices") and hasattr(geometry, "get_edges")
    is_thick_plane = hasattr(geometry, "get_faces") and hasattr(geometry, "normal")
    is_rectangle = hasattr(geometry, "get_corners") and hasattr(geometry, "dimensions")

    if not (is_mesh or is_box or is_thick_plane or is_rectangle):
        raise TypeError("geometry must be a GEOSMesh/GEOSBox/GEOSThickPlane/GEOSRectangle-like object")

    # normalize overlay inputs
    overlay_boxes = _as_list(boxes)
    overlay_thick_planes = _as_list(thick_planes)
    overlay_rectangles = _as_list(rectangles)

    # split primary input into mesh, box, or thick-plane mode
    if is_mesh:
        mesh_obj = geometry
        primary_box = None
        primary_thick_plane = None
        primary_rectangle = None
    elif is_box:
        mesh_obj = None
        primary_box = geometry
        primary_thick_plane = None
        primary_rectangle = None
    elif is_thick_plane:
        mesh_obj = None
        primary_box = None
        primary_thick_plane = geometry
        primary_rectangle = None
    else:
        mesh_obj = None
        primary_box = None
        primary_thick_plane = None
        primary_rectangle = geometry

    # if only a primary box was supplied, reuse mesh style for that box
    if primary_box is not None and not overlay_boxes:
        overlay_boxes = [primary_box]
        use_primary_style_for_boxes = True
    else:
        use_primary_style_for_boxes = False

    # if only a primary thick plane was supplied, reuse mesh style for that plane
    if primary_thick_plane is not None and not overlay_thick_planes:
        overlay_thick_planes = [primary_thick_plane]
        use_primary_style_for_thick_planes = True
    else:
        use_primary_style_for_thick_planes = False

    # if only a primary rectangle was supplied, reuse mesh style for that rectangle
    if primary_rectangle is not None and not overlay_rectangles:
        overlay_rectangles = [primary_rectangle]
        use_primary_style_for_rectangles = True
    else:
        use_primary_style_for_rectangles = False

    # infer default slice location from mesh, box, or thick plane
    if position is None:
        if mesh_obj is not None:
            position = {
                "xy": mesh_obj.nodes[:, 2].mean(),
                "xz": mesh_obj.nodes[:, 1].mean(),
                "yz": mesh_obj.nodes[:, 0].mean(),
            }[plane]
        elif overlay_boxes:
            first_box = overlay_boxes[0]
            if not hasattr(first_box, "get_vertices"):
                raise TypeError("All items in boxes must be GEOSBox-like objects")
            position = float(np.asarray(first_box.get_vertices())[:, perp_axis].mean())
        elif overlay_thick_planes:
            first_plane = overlay_thick_planes[0]
            if hasattr(first_plane, "origin"):
                position = float(np.asarray(first_plane.origin)[perp_axis])
            else:
                faces = first_plane.get_faces(size=thick_plane_size)
                position = float(np.asarray(faces[0])[:, perp_axis].mean())
        elif overlay_rectangles:
            first_rectangle = overlay_rectangles[0]
            position = float(np.asarray(first_rectangle.origin)[perp_axis])
        else:
            raise ValueError("Could not infer default position")

    # compute mesh slice polygons
    mesh_polygons = []
    if mesh_obj is not None:
        tolerance = max(mesh_obj.nodes[:, perp_axis].std() * 0.01, 1e-6)
        for elem in mesh_obj.elements:
            elem_nodes = mesh_obj.nodes[elem]
            perp_coords = elem_nodes[:, perp_axis]
            if perp_coords.min() <= position <= perp_coords.max():
                polygon = _mesh_element_slice_polygon(
                    elem_nodes,
                    axis_indices,
                    perp_axis,
                    position,
                    tol=tolerance,
                )
                if polygon is not None:
                    mesh_polygons.append(polygon)

    # resolve mesh name for legends
    if mesh_obj is not None:
        mesh_name = str(getattr(mesh_obj, "name", "mesh"))
    else:
        mesh_name = "mesh"

    # compute box slice polygons
    box_polygons = []
    box_entries = []
    for idx, box_obj in enumerate(overlay_boxes, start=1):
        if not (hasattr(box_obj, "get_vertices") and hasattr(box_obj, "get_edges")):
            raise TypeError("All items in boxes must be GEOSBox-like objects")

        pts = _box_slice_polygon(box_obj, axis_indices, perp_axis, position)
        if pts is not None:
            box_polygons.append(pts)
            box_name = str(getattr(box_obj, "name", f"box{idx}"))
            box_entries.append({"name": box_name, "polygon": pts})

    # compute thick-plane slice polygons
    thick_plane_polygons = []
    thick_plane_entries = []
    for idx, thick_plane_obj in enumerate(overlay_thick_planes, start=1):
        if not hasattr(thick_plane_obj, "get_faces"):
            raise TypeError("All items in thick_planes must be GEOSThickPlane-like objects")

        pts = _thick_plane_slice_polygon(
            thick_plane_obj,
            axis_indices,
            perp_axis,
            position,
            size=thick_plane_size,
        )
        if pts is not None:
            thick_plane_polygons.append(pts)
            thick_plane_name = str(getattr(thick_plane_obj, "name", f"thick_plane{idx}"))
            thick_plane_entries.append({"name": thick_plane_name, "polygon": pts})

    # compute rectangle slice polygons
    rectangle_polygons = []
    rectangle_entries = []
    for idx, rectangle_obj in enumerate(overlay_rectangles, start=1):
        if not hasattr(rectangle_obj, "get_corners"):
            raise TypeError("All items in rectangles must be GEOSRectangle-like objects")

        pts = _rectangle_slice_polygon(rectangle_obj, axis_indices, perp_axis, position)
        if pts is not None:
            rectangle_polygons.append(pts)
            rectangle_name = str(getattr(rectangle_obj, "name", f"rectangle{idx}"))
            rectangle_entries.append({"name": rectangle_name, "polygon": pts})

    # combine coordinates for autoscaling
    all_coords = []
    for poly in mesh_polygons:
        all_coords.extend(poly)
    for poly in box_polygons:
        all_coords.extend(poly)
    for poly in thick_plane_polygons:
        all_coords.extend(poly)
    for poly in rectangle_polygons:
        all_coords.extend(poly)

    return {
        "plane": plane,
        "position": position,
        "axis_labels": axis_labels,
        "perp_label": perp_label,
        "mesh_name": mesh_name,
        "mesh_polygons": mesh_polygons,
        "box_polygons": box_polygons,
        "thick_plane_polygons": thick_plane_polygons,
        "rectangle_polygons": rectangle_polygons,
        "box_entries": box_entries,
        "thick_plane_entries": thick_plane_entries,
        "rectangle_entries": rectangle_entries,
        "all_coords": np.array(all_coords) if all_coords else np.array([]),
        "use_primary_style_for_boxes": use_primary_style_for_boxes,
        "use_primary_style_for_thick_planes": use_primary_style_for_thick_planes,
        "use_primary_style_for_rectangles": use_primary_style_for_rectangles,
    }

def plot_mesh_2d(
    geometry: GEOSMesh | GEOSBox | GEOSThickPlane | GEOSRectangle,
    boxes: GEOSBox | list[GEOSBox] | None = None,
    thick_planes: GEOSThickPlane | list[GEOSThickPlane] | None = None,
    rectangles: GEOSRectangle | list[GEOSRectangle] | None = None,
    plane: str = "xy",
    position: float | None = None,
    figsize: tuple[int, int] = (10, 8),
    show_polygons_fill: bool = True,
    show_edges: bool = True,
    show_nodes: bool = False,
    show_legend: bool = True,
    alpha: float = 0.6,
    edge_color: str = "black",
    edge_size: float = 1.0,
    face_color: str = "lightblue",
    box_face_color: str | list[str] = "lightsalmon",
    box_edge_color: str | list[str] = "darkred",
    box_alpha: float = 0.35,
    thick_plane_face_color: str = "mediumpurple",
    thick_plane_edge_color: str = "indigo",
    thick_plane_alpha: float = 0.35,
    rectangle_face_color: str = "khaki",
    rectangle_edge_color: str = "goldenrod",
    rectangle_alpha: float = 0.45,
    thick_plane_size: float = 10.0,
    node_color: str = "red",
    node_size: float = 20,
    grid: bool = True,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    fig_prev: plt.Figure | None = None,
    ax_prev: plt.Axes | None = None
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a 2D slice of a 3D GEOS mesh, with optional GEOS box, thick-plane, and rectangle overlays.

    Args:
        geometry (GEOSMesh | GEOSBox | GEOSThickPlane | GEOSRectangle): Primary geometry object.
        boxes (GEOSBox | list[GEOSBox] | None): Box overlay(s).
        thick_planes (GEOSThickPlane | list[GEOSThickPlane] | None): Thick-plane overlay(s).
        rectangles (GEOSRectangle | list[GEOSRectangle] | None): Rectangle overlay(s).
        plane (str): String denoting plane to be plotted (default: "xy").
        position (float): Shift value along 3rd axis (default: None).
        figsize (tuple[int, int]): Figure size (default: (10, 8)).
        show_polygons_fill (bool): Toggle polygon fills (default: True).
        show_edges (bool): Boolean value to control edges (default: True).
        show_nodes (bool): Show plotted nodes/intersection points.
        show_legend (bool): Show legend for mesh and overlay entries.
        alpha (float): Mesh polygon transparency.
        edge_color (str): Mesh edge color.
        edge_size (float): Mesh edge line width.
        face_color (str): Mesh face color.
        box_face_color (str): Overlay box face color.
        box_edge_color (str): Overlay box edge color.
        box_alpha (float): Overlay box transparency.
        thick_plane_face_color (str): Overlay thick-plane face color.
        thick_plane_edge_color (str): Overlay thick-plane edge color.
        thick_plane_alpha (float): Overlay thick-plane transparency.
        rectangle_face_color (str): Overlay rectangle face color.
        rectangle_edge_color (str): Overlay rectangle edge color.
        rectangle_alpha (float): Overlay rectangle transparency.
        thick_plane_size (float): Thick-plane patch half-size used for slicing.
        node_color (str): Node color.
        node_size (float): Node marker size.
        grid (bool): Toggle grid.
        title (str): Optional title.
        xlabel (str | None): Override label for the horizontal axis (e.g. 'X (meters)').
        ylabel (str | None): Override label for the vertical axis (e.g. 'Z (meters)').
        fig_prev (plt.Figure | None): Optional original figure to plot on.
        ax_prev (plt.Axes | None): Optional original axes to plot on.

    Returns:
        tuple: (fig, ax) matplotlib figure and axes.
    """
    # compute geometry once and render using matplotlib
    data = _compute_slice_geometry(geometry, boxes, thick_planes, rectangles, plane, position, thick_plane_size)

    # create matplotlib figure/axes for plotting
    fig, ax = plt.subplots(figsize=figsize) if (fig_prev is None and ax_prev is None) else (fig_prev, ax_prev)

    # draw mesh polygons
    for poly in data["mesh_polygons"]:
        ax.add_patch(
            plt.Polygon(
                poly,
                fill=show_polygons_fill,
                alpha=alpha,
                facecolor=face_color,
                edgecolor=edge_color if show_edges else face_color,
                linewidth=edge_size if show_edges else 0,
            )
        )

    # draw box polygons
    box_count = len(data["box_polygons"])
    if isinstance(box_face_color, str):
        box_face_colors = [box_face_color] * box_count
    elif box_face_color:
        box_face_colors = [box_face_color[i % len(box_face_color)] for i in range(box_count)]
    else:
        box_face_colors = ["lightsalmon"] * box_count

    if isinstance(box_edge_color, str):
        box_edge_colors = [box_edge_color] * box_count
    elif box_edge_color:
        box_edge_colors = [box_edge_color[i % len(box_edge_color)] for i in range(box_count)]
    else:
        box_edge_colors = ["darkred"] * box_count

    for idx, poly in enumerate(data["box_polygons"]):
        current_face = face_color if data["use_primary_style_for_boxes"] else box_face_colors[idx]
        current_edge = edge_color if data["use_primary_style_for_boxes"] else box_edge_colors[idx]
        current_alpha = alpha if data["use_primary_style_for_boxes"] else box_alpha
        ax.add_patch(
            plt.Polygon(
                poly,
                fill=True,
                alpha=current_alpha,
                facecolor=current_face,
                edgecolor=current_edge if show_edges else current_face,
                linewidth=1.0 if show_edges else 0,
            )
        )

    # draw thick-plane polygons
    for poly in data["thick_plane_polygons"]:
        current_face = face_color if data["use_primary_style_for_thick_planes"] else thick_plane_face_color
        current_edge = edge_color if data["use_primary_style_for_thick_planes"] else thick_plane_edge_color
        current_alpha = alpha if data["use_primary_style_for_thick_planes"] else thick_plane_alpha
        ax.add_patch(
            plt.Polygon(
                poly,
                fill=True,
                alpha=current_alpha,
                facecolor=current_face,
                edgecolor=current_edge if show_edges else current_face,
                linewidth=1.0 if show_edges else 0,
            )
        )

    # draw rectangle polygons
    for poly in data["rectangle_polygons"]:
        current_face = face_color if data["use_primary_style_for_rectangles"] else rectangle_face_color
        current_edge = edge_color if data["use_primary_style_for_rectangles"] else rectangle_edge_color
        current_alpha = alpha if data["use_primary_style_for_rectangles"] else rectangle_alpha
        ax.add_patch(
            plt.Polygon(
                poly,
                fill=True,
                alpha=current_alpha,
                facecolor=current_face,
                edgecolor=current_edge if show_edges else current_face,
                linewidth=1.0 if show_edges else 0,
            )
        )

    # optionally draw polygon points
    if show_nodes:
        points = []
        for poly in data["mesh_polygons"]:
            points.extend(poly)
        for poly in data["box_polygons"]:
            points.extend(poly)
        for poly in data["thick_plane_polygons"]:
            points.extend(poly)
        for poly in data["rectangle_polygons"]:
            points.extend(poly)
        if points:
            points = np.array(points)
            ax.scatter(points[:, 0], points[:, 1], c=node_color, s=node_size, zorder=5, alpha=0.8)

    # label axes, using caller-supplied labels when provided
    ax.set_xlabel(xlabel if xlabel is not None else f"{data['axis_labels'][0]} (m)", fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel if ylabel is not None else f"{data['axis_labels'][1]} (m)", fontsize=12, fontweight="bold")

    # build default title when none is supplied
    if title is None:
        title = f"2D View: {data['plane'].upper()} Plane ({data['perp_label']} = {data['position']:.3f})"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

    # autoscale bounds from plotted data with small padding
    if data["all_coords"].size > 0:
        x_min, x_max = data["all_coords"][:, 0].min(), data["all_coords"][:, 0].max()
        y_min, y_max = data["all_coords"][:, 1].min(), data["all_coords"][:, 1].max()
        x_pad = max((x_max - x_min) * 0.05, 1e-9)
        y_pad = max((y_max - y_min) * 0.05, 1e-9)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

    # keep equal axis scaling to avoid geometric distortion
    ax.set_aspect("equal", adjustable="box")
    if grid:
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    if show_legend:
        # build custom legend handles for mesh and overlay entries
        legend_handles = []
        if data["mesh_polygons"]:
            legend_handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    color=edge_color if show_edges else face_color,
                    marker="s",
                    markersize=10,
                    markerfacecolor=face_color,
                    label=data["mesh_name"],
                )
            )
        for idx, entry in enumerate(data["box_entries"]):
            if data["use_primary_style_for_boxes"]:
                current_face = face_color
                current_edge = edge_color
            else:
                current_face = box_face_colors[idx]
                current_edge = box_edge_colors[idx]
            legend_handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    color=current_edge if show_edges else current_face,
                    marker="s",
                    markersize=10,
                    markerfacecolor=current_face,
                    label=entry["name"],
                )
            )
        for entry in data["thick_plane_entries"]:
            current_face = face_color if data["use_primary_style_for_thick_planes"] else thick_plane_face_color
            current_edge = edge_color if data["use_primary_style_for_thick_planes"] else thick_plane_edge_color
            legend_handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    color=current_edge if show_edges else current_face,
                    marker="s",
                    markersize=10,
                    markerfacecolor=current_face,
                    label=entry["name"],
                )
            )
        for entry in data["rectangle_entries"]:
            current_face = face_color if data["use_primary_style_for_rectangles"] else rectangle_face_color
            current_edge = edge_color if data["use_primary_style_for_rectangles"] else rectangle_edge_color
            legend_handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    color=current_edge if show_edges else current_face,
                    marker="s",
                    markersize=10,
                    markerfacecolor=current_face,
                    label=entry["name"],
                )
            )

        ax.legend(handles=legend_handles, loc="best", fontsize=10)
    

    # improve layout spacing and return figure/axes
    plt.tight_layout()
    return fig, ax

def plot_mesh_2d_interactive(
    geometry: GEOSMesh | GEOSBox | GEOSThickPlane | GEOSRectangle,
    boxes: GEOSBox | list[GEOSBox] | None = None,
    thick_planes: GEOSThickPlane | list[GEOSThickPlane] | None = None,
    rectangles: GEOSRectangle | list[GEOSRectangle] | None = None,
    plane: str = "xy",
    position: float | None = None,
    face_color: str = "lightblue",
    edge_color: str = "black",
    opacity: float = 0.6,
    box_face_color: str | list[str] = "lightsalmon",
    box_edge_color: str | list[str] = "darkred",
    box_opacity: float = 0.35,
    thick_plane_face_color: str = "mediumpurple",
    thick_plane_edge_color: str = "indigo",
    thick_plane_opacity: float = 0.35,
    rectangle_face_color: str = "khaki",
    rectangle_edge_color: str = "goldenrod",
    rectangle_opacity: float = 0.45,
    thick_plane_size: float = 10.0,
    show_edges: bool = True,
    edge_width: float = 1.0,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    width: int = 800,
    height: int = 700,
) -> go.Figure:
    """
    Plot an interactive 2D slice of a GEOS mesh with optional box, thick-plane, and rectangle overlays.

    Args:
        geometry (GEOSMesh | GEOSBox | GEOSThickPlane | GEOSRectangle): Primary geometry to be plotted.
        boxes (GEOSBox | list[GEOSBox] | None): Box overlay(s).
        thick_planes (GEOSThickPlane | list[GEOSThickPlane] | None): Thick-plane overlay(s).
        rectangles (GEOSRectangle | list[GEOSRectangle] | None): Rectangle overlay(s).
        plane (str): Plane to be plotted (default: xy-plane).
        position (float): Slice position along perpendicular axis.
        face_color (str): Mesh fill color.
        edge_color (str): Mesh edge color.
        opacity (float): Mesh face opacity.
        box_face_color (str): Overlay box fill color.
        box_edge_color (str): Overlay box edge color.
        box_opacity (float): Overlay box opacity.
        thick_plane_face_color (str): Overlay thick-plane fill color.
        thick_plane_edge_color (str): Overlay thick-plane edge color.
        thick_plane_opacity (float): Overlay thick-plane opacity.
        rectangle_face_color (str): Overlay rectangle fill color.
        rectangle_edge_color (str): Overlay rectangle edge color.
        rectangle_opacity (float): Overlay rectangle opacity.
        thick_plane_size (float): Thick-plane patch half-size used for slicing.
        show_edges (bool): Toggle edge rendering.
        edge_width (float): Edge line width.
        title (str): Optional figure title.
        xlabel (str | None): Override label for the horizontal axis (e.g. 'X (meters)').
        ylabel (str | None): Override label for the vertical axis (e.g. 'Z (meters)').
        width (int): Figure width.
        height (int): Figure height.

    Returns:
        go.Figure: Plotly figure.
    """
    # compute geometry once and render using plotly
    data = _compute_slice_geometry(geometry, boxes, thick_planes, rectangles, plane, position, thick_plane_size)

    fig = go.Figure()

    # helper to render one polygon as a legend-capable trace
    def _add_polygon_trace(
        coords_2d: np.ndarray,
        name: str,
        legendgroup: str,
        fill_color: str,
        line_color: str,
        trace_opacity: float,
        show_legend: bool,
    ) -> None:
        x_vals = np.append(coords_2d[:, 0], coords_2d[0, 0])
        y_vals = np.append(coords_2d[:, 1], coords_2d[0, 1])
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines",
                fill="toself",
                fillcolor=fill_color,
                line=dict(color=line_color, width=edge_width if show_edges else 0),
                opacity=trace_opacity,
                name=name,
                legendgroup=legendgroup,
                showlegend=show_legend,
                hoverinfo="skip",
            )
        )

    # draw mesh polygons as traces grouped under one legend item
    for idx, poly in enumerate(data["mesh_polygons"]):
        _add_polygon_trace(
            coords_2d=poly,
            name=data["mesh_name"],
            legendgroup="mesh",
            fill_color=face_color,
            line_color=edge_color,
            trace_opacity=opacity,
            show_legend=(idx == 0),
        )

    # draw box polygons with one legend item per box object
    box_entry_count = len(data["box_entries"])
    if isinstance(box_face_color, str):
        box_face_colors = [box_face_color] * box_entry_count
    elif box_face_color:
        box_face_colors = [box_face_color[i % len(box_face_color)] for i in range(box_entry_count)]
    else:
        box_face_colors = ["lightsalmon"] * box_entry_count

    if isinstance(box_edge_color, str):
        box_edge_colors = [box_edge_color] * box_entry_count
    elif box_edge_color:
        box_edge_colors = [box_edge_color[i % len(box_edge_color)] for i in range(box_entry_count)]
    else:
        box_edge_colors = ["darkred"] * box_entry_count

    for idx, entry in enumerate(data["box_entries"]):
        # make wells stand out by using bright colors
        if "well" in entry["name"].lower():
            current_face = "gold"
        else:
            current_face = face_color if data["use_primary_style_for_boxes"] else box_face_colors[idx]
        current_edge = edge_color if data["use_primary_style_for_boxes"] else box_edge_colors[idx]
        current_opacity = opacity if data["use_primary_style_for_boxes"] else box_opacity
        _add_polygon_trace(
            coords_2d=entry["polygon"],
            name=entry["name"],
            legendgroup=f"box::{entry['name']}",
            fill_color=current_face,
            line_color=current_edge,
            trace_opacity=current_opacity,
            show_legend=True,
        )

    # draw thick-plane polygons with one legend item per thick-plane object
    for entry in data["thick_plane_entries"]:
        current_face = face_color if data["use_primary_style_for_thick_planes"] else thick_plane_face_color
        current_edge = edge_color if data["use_primary_style_for_thick_planes"] else thick_plane_edge_color
        current_opacity = opacity if data["use_primary_style_for_thick_planes"] else thick_plane_opacity
        _add_polygon_trace(
            coords_2d=entry["polygon"],
            name=entry["name"],
            legendgroup=f"thick_plane::{entry['name']}",
            fill_color=current_face,
            line_color=current_edge,
            trace_opacity=current_opacity,
            show_legend=True,
        )

    # draw rectangle polygons with one legend item per rectangle object
    for entry in data["rectangle_entries"]:
        current_face = face_color if data["use_primary_style_for_rectangles"] else rectangle_face_color
        current_edge = edge_color if data["use_primary_style_for_rectangles"] else rectangle_edge_color
        current_opacity = opacity if data["use_primary_style_for_rectangles"] else rectangle_opacity
        _add_polygon_trace(
            coords_2d=entry["polygon"],
            name=entry["name"],
            legendgroup=f"rectangle::{entry['name']}",
            fill_color=current_face,
            line_color=current_edge,
            trace_opacity=current_opacity,
            show_legend=True,
        )

    # build default title when none is supplied
    if title is None:
        title = f"2D View: {data['plane'].upper()} Plane ({data['perp_label']} = {data['position']:.3f})"

    # resolve axis labels, using caller-supplied labels when provided
    x_label = xlabel if xlabel is not None else f'{data["axis_labels"][0]} (m)'
    y_label = ylabel if ylabel is not None else f'{data["axis_labels"][1]} (m)'

    # keep viewport stable using computed bounds
    x_range = None
    y_range = None
    if data["all_coords"].size > 0:
        x_min, x_max = data["all_coords"][:, 0].min(), data["all_coords"][:, 0].max()
        y_min, y_max = data["all_coords"][:, 1].min(), data["all_coords"][:, 1].max()
        x_pad = max((x_max - x_min) * 0.05, 1e-9)
        y_pad = max((y_max - y_min) * 0.05, 1e-9)
        x_range = [x_min - x_pad, x_max + x_pad]
        y_range = [y_min - y_pad, y_max + y_pad]

    # apply layout, axis labels/scaling, and legend behavior
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis=dict(
            title=x_label,
            scaleanchor="y",
            scaleratio=1,
            showgrid=True,
            gridcolor="lightgray",
            range=x_range,
        ),
        yaxis=dict(
            title=y_label,
            showgrid=True,
            gridcolor="lightgray",
            range=y_range,
        ),
        width=width,
        height=height,
        hovermode="closest",
        legend=dict(
            title="Legend",
            x=1.02,
            y=1.0,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.85)",
            groupclick="toggleitem",
        ),
    )

    # return fully configured interactive figure
    return fig


# -----------------------------------
# |  Stress Field Construction      |
# -----------------------------------

def load_stress_tablefunction(
    base_dir: str | Path,
    component_files: dict[str, str] | None = None,
) -> dict[str, np.ndarray]:
    """
    Load GEOS TableFunction stress tensor files and coordinate axes.

    Args:
        base_dir (str | Path): Directory containing x.csv, y.csv, z.csv and sigma files.
        component_files (dict[str, str] | None): Mapping of component name to filename.

    Returns:
        dict[str, np.ndarray]: Dictionary with x, y, z and stress components in MPa.
    """
    if component_files is None:
        component_files = {
            "sigma_xx": "sigma_xx.csv",
            "sigma_yy": "sigma_yy.csv",
            "sigma_zz": "sigma_zz.csv",
        }

    base_dir = Path(base_dir)
    x = np.atleast_1d(np.loadtxt(base_dir / "x.csv", delimiter=",", dtype=float))
    y = np.atleast_1d(np.loadtxt(base_dir / "y.csv", delimiter=",", dtype=float))
    z = np.atleast_1d(np.loadtxt(base_dir / "z.csv", delimiter=",", dtype=float))

    nx, ny, nz = len(x), len(y), len(z)
    n_total = nx * ny * nz

    table = {"x": x, "y": y, "z": z}
    for comp, file_name in component_files.items():
        raw = np.atleast_1d(np.loadtxt(base_dir / file_name, delimiter=",", dtype=float))
        if len(raw) not in [nz, n_total]:
            raise ValueError(
                f"{comp}: expected {nz} (layer-cake) or {n_total} entries, got {len(raw)}"
            )
        if len(raw) == nz:
            # layer-cake input (1D in z), broadcast to full 3D table shape
            arr = np.broadcast_to(raw[None, None, :], (nx, ny, nz)).astype(float).copy()
        else:
            arr = raw.reshape((nx, ny, nz), order="C")

        # convert stress components from Pa to MPa with GEOS sign convention
        if comp.startswith("sigma_"):
            arr = arr * -1e-6

        table[comp] = arr

    return table

def _axis_edges(axis: np.ndarray) -> np.ndarray:
    """
    Build bin edges around axis coordinates using midpoint spacing.

    Args:
        axis (np.ndarray): Monotonic coordinate array.

    Returns:
        np.ndarray: Edge coordinates with length len(axis)+1.
    """
    axis = np.asarray(axis, dtype=float)
    if len(axis) == 1:
        return np.array([-np.inf, np.inf], dtype=float)

    edges = np.empty(len(axis) + 1, dtype=float)
    edges[1:-1] = 0.5 * (axis[:-1] + axis[1:])
    edges[0] = axis[0] - 0.5 * (axis[1] - axis[0])
    edges[-1] = axis[-1] + 0.5 * (axis[-1] - axis[-2])
    return edges

def _plot_edges(axis: np.ndarray) -> np.ndarray:
    """Build finite plotting edges for pcolormesh-style rendering."""
    axis = np.asarray(axis, dtype=float)
    if len(axis) == 1:
        return np.array([axis[0] - 0.5, axis[0] + 0.5], dtype=float)

    edges = np.empty(len(axis) + 1, dtype=float)
    edges[1:-1] = 0.5 * (axis[:-1] + axis[1:])
    edges[0] = axis[0] - 0.5 * (axis[1] - axis[0])
    edges[-1] = axis[-1] + 0.5 * (axis[-1] - axis[-2])
    return edges

def _slice_grid(
    points_xyz: np.ndarray,
    values: np.ndarray,
    plane: str,
    decimals: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Convert slice points to a regular 2D grid if possible.

    Args:
        points_xyz (np.ndarray): Points on the slice plane.
        values (np.ndarray): Scalar values at each point.
        plane (str): Slice plane ('xy', 'xz', 'yz').
        decimals (int): Rounding precision for robust unique indexing.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray] | None:
            (x_coords, y_coords, value_grid) or None if grid cannot be formed.
    """
    axis_indices = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}[plane.lower()]
    x = np.round(points_xyz[:, axis_indices[0]], decimals)
    y = np.round(points_xyz[:, axis_indices[1]], decimals)

    x_unique = np.unique(x)
    y_unique = np.unique(y)
    if len(x_unique) * len(y_unique) > len(values) * 1.05:
        return None

    x_idx = np.searchsorted(x_unique, x)
    y_idx = np.searchsorted(y_unique, y)

    value_sum = np.zeros((len(y_unique), len(x_unique)), dtype=float)
    value_count = np.zeros((len(y_unique), len(x_unique)), dtype=float)
    np.add.at(value_sum, (y_idx, x_idx), values)
    np.add.at(value_count, (y_idx, x_idx), 1.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        grid = value_sum / value_count
    grid[value_count == 0.0] = np.nan

    if np.all(np.isnan(grid)):
        return None

    return x_unique, y_unique, grid

def _table_indices(
    coords: np.ndarray,
    axis: np.ndarray,
    method: str = "upper",
) -> np.ndarray:
    """
    Convert coordinates to nearest table-bin indices with clipping.

    Args:
        coords (np.ndarray): Coordinate values to index.
        axis (np.ndarray): Table axis coordinates.
        method (str): Interpolation policy ('upper' or 'lower').

    Returns:
        np.ndarray: Integer indices into the axis domain.
    """
    coords = np.asarray(coords, dtype=float)
    axis = np.asarray(axis, dtype=float)

    if len(axis) == 1:
        return np.zeros_like(coords, dtype=int)

    edges = _axis_edges(axis)
    idx = np.searchsorted(edges, coords, side="right") - 1
    idx = np.clip(idx, 0, len(axis) - 1)

    if method == "lower":
        idx = np.maximum(idx - 1, 0)
    elif method != "upper":
        raise ValueError("method must be 'upper' or 'lower'")

    return idx.astype(int)

def sample_stress_at_points(
    points_xyz: np.ndarray,
    table: dict[str, np.ndarray],
    component: str = "sigma_xx",
    method: str = "upper",
) -> np.ndarray:
    """
    Sample a stress component at 3D points using GEOS TableFunction indexing.

    Args:
        points_xyz (np.ndarray): Array of shape (N, 3) with point coordinates.
        table (dict[str, np.ndarray]): Output dictionary from load_stress_tablefunction.
        component (str): Stress component key ('sigma_xx', 'sigma_yy', 'sigma_zz').
        method (str): Interpolation policy ('upper' or 'lower').

    Returns:
        np.ndarray: Sampled stress values at input points in MPa.
    """
    if component not in table:
        raise KeyError(f"component '{component}' not found in table")

    points_xyz = np.asarray(points_xyz, dtype=float)
    ix = _table_indices(points_xyz[:, 0], table["x"], method=method)
    iy = _table_indices(points_xyz[:, 1], table["y"], method=method)
    iz = _table_indices(points_xyz[:, 2], table["z"], method=method)
    return table[component][ix, iy, iz]

def _slice_mask(
    points_xyz: np.ndarray,
    plane: str,
    position: float,
    tol: float | None = None,
) -> np.ndarray:
    """
    Build a mask for points lying near a 2D slice plane.

    Args:
        points_xyz (np.ndarray): Array of shape (N, 3) of point coordinates.
        plane (str): Slice plane ('xy', 'xz', 'yz').
        position (float): Slice position along perpendicular axis.
        tol (float | None): Optional tolerance; auto-computed when None.

    Returns:
        np.ndarray: Boolean mask of selected points.
    """
    plane = plane.lower()
    perp_axis = {"xy": 2, "xz": 1, "yz": 0}[plane]
    coords = points_xyz[:, perp_axis]

    if tol is None:
        unique_vals = np.unique(np.round(coords, 12))
        if len(unique_vals) > 1:
            tol = 0.5 * np.min(np.diff(np.sort(unique_vals))) + 1e-12
        else:
            tol = 1e-9

    return np.abs(coords - position) <= tol

def add_stress_component_to_matplotlib_slice(
    ax: plt.Axes,
    mesh: GEOSMesh,
    table: dict[str, np.ndarray],
    component: str = "sigma_xx",
    plane: str = "xz",
    position: float = 0.0,
    method: str = "upper",
    cmap: str = "RdYlBu",
    alpha: float = 0.75,
    max_points: int = 20000,
    render_mode: str = "cell",
    clip_percentiles: tuple[float, float] = (2.0, 98.0),
) -> None:
    """
    Overlay sampled stress-component values on a matplotlib mesh slice.

    Args:
        ax (plt.Axes): Target axes.
        mesh (GEOSMesh): Mesh used for point sampling.
        table (dict[str, np.ndarray]): Stress table dictionary.
        component (str): Stress component key.
        plane (str): Slice plane ('xy', 'xz', 'yz').
        position (float): Slice position along perpendicular axis.
        method (str): Interpolation policy ('upper' or 'lower').
        cmap (str): Colormap name.
        alpha (float): Overlay transparency.
        max_points (int): Max plotted points after optional down-sampling.
        render_mode (str): 'cell', 'scatter', or 'auto'.
        clip_percentiles (tuple[float, float]): Percentiles for contrast scaling.

    Returns:
        None
    """
    nodes = mesh.nodes
    mask = _slice_mask(nodes, plane=plane, position=position)
    pts = nodes[mask]
    if len(pts) == 0:
        return

    vals = sample_stress_at_points(pts, table, component=component, method=method)

    if len(pts) > max_points:
        step = int(np.ceil(len(pts) / max_points))
        pts = pts[::step]
        vals = vals[::step]

    if np.all(np.isnan(vals)):
        return

    valid_vals = vals[np.isfinite(vals)]
    if len(valid_vals) > 0:
        vmin, vmax = np.nanpercentile(valid_vals, clip_percentiles)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmin, vmax):
            vmin, vmax = np.nanmin(valid_vals), np.nanmax(valid_vals)
    else:
        vmin = vmax = None

    use_cell = render_mode in {"cell", "auto"}
    if use_cell:
        grid_result = _slice_grid(pts, vals, plane=plane)
    else:
        grid_result = None

    mappable = None
    if grid_result is not None:
        gx, gy, gvals = grid_result
        gx_edges = _plot_edges(gx)
        gy_edges = _plot_edges(gy)
        mappable = ax.pcolormesh(
            gx_edges,
            gy_edges,
            gvals,
            cmap=cmap,
            alpha=alpha,
            shading="flat",
            vmin=vmin,
            vmax=vmax,
            zorder=1,
        )
    else:
        axis_indices = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}[plane.lower()]
        mappable = ax.scatter(
            pts[:, axis_indices[0]],
            pts[:, axis_indices[1]],
            c=vals,
            cmap=cmap,
            alpha=max(alpha, 0.55),
            s=24,
            marker="s",
            linewidths=0,
            vmin=vmin,
            vmax=vmax,
            zorder=1,
        )

    if not hasattr(ax, "_stress_colorbar") or ax._stress_colorbar is None:
        _dict = {
            "sigma_xx": r"$\sigma_{xx}$",
            "sigma_yy": r"$\sigma_{yy}$",
            "sigma_zz": r"$\sigma_{zz}$",
        }
        cbar = plt.colorbar(mappable, ax=ax, pad=0.02)
        cbar.set_label(f"{_dict.get(component, component)} [MPa]")
        ax._stress_colorbar = cbar


def add_stress_component_to_plotly_slice(
    fig: go.Figure,
    mesh: GEOSMesh,
    table: dict[str, np.ndarray],
    component: str = "sigma_xx",
    plane: str = "xz",
    position: float = 0.0,
    method: str = "upper",
    colorscale: str = "RdYlBu",
    opacity: float = 0.78,
    max_points: int = 30000,
    render_mode: str = "cell",
    clip_percentiles: tuple[float, float] = (2.0, 98.0),
) -> None:
    """
    Overlay sampled stress-component values on a plotly mesh slice.

    Args:
        fig (go.Figure): Target figure.
        mesh (GEOSMesh): Mesh used for point sampling.
        table (dict[str, np.ndarray]): Stress table dictionary.
        component (str): Stress component key.
        plane (str): Slice plane ('xy', 'xz', 'yz').
        position (float): Slice position along perpendicular axis.
        method (str): Interpolation policy ('upper' or 'lower').
        colorscale (str): Plotly colorscale.
        opacity (float): Overlay opacity.
        max_points (int): Max plotted points after optional down-sampling.
        render_mode (str): 'cell', 'scatter', or 'auto'.
        clip_percentiles (tuple[float, float]): Percentiles for contrast scaling.

    Returns:
        None
    """
    nodes = mesh.nodes
    mask = _slice_mask(nodes, plane=plane, position=position)
    pts = nodes[mask]
    if len(pts) == 0:
        return

    vals = sample_stress_at_points(pts, table, component=component, method=method)

    if len(pts) > max_points:
        step = int(np.ceil(len(pts) / max_points))
        pts = pts[::step]
        vals = vals[::step]

    if np.all(np.isnan(vals)):
        return

    valid_vals = vals[np.isfinite(vals)]
    if len(valid_vals) > 0:
        zmin, zmax = np.nanpercentile(valid_vals, clip_percentiles)
        if not np.isfinite(zmin) or not np.isfinite(zmax) or np.isclose(zmin, zmax):
            zmin, zmax = np.nanmin(valid_vals), np.nanmax(valid_vals)
    else:
        zmin = zmax = None

    use_cell = render_mode in {"cell", "auto"}
    if use_cell:
        grid_result = _slice_grid(pts, vals, plane=plane)
    else:
        grid_result = None

    if grid_result is not None:
        gx, gy, gvals = grid_result
        fig.add_trace(
            go.Heatmap(
                x=gx,
                y=gy,
                z=gvals,
                colorscale=colorscale,
                opacity=opacity,
                zmin=zmin,
                zmax=zmax,
                colorbar=dict(title=f"{component} [MPa]"),
                hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<br>value=%{z:.3f} MPa<extra></extra>",
                showscale=True,
                name=f"Stress: {component}",
                showlegend=True,
            )
        )
    else:
        axis_indices = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}[plane.lower()]
        fig.add_trace(
            go.Scattergl(
                x=pts[:, axis_indices[0]],
                y=pts[:, axis_indices[1]],
                mode="markers",
                marker=dict(
                    color=vals,
                    colorscale=colorscale,
                    opacity=max(opacity, 0.6),
                    size=7,
                    symbol="square",
                    colorbar=dict(title=f"{component} [MPa]"),
                    cmin=zmin,
                    cmax=zmax,
                ),
                name=f"Stress: {component}",
                hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<br>value=%{marker.color:.3f} MPa<extra></extra>",
                showlegend=True,
            )
        )
        