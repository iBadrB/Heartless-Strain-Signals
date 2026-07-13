import numpy as np
import numpy.typing as npt

class GEOSThickPlane:
    """Represent a finite-thickness plane (used for fracture-like geometry).

    Attributes:
        name (str): Name of thick plane object.
        normal (npt.NDArray[np.float32]): Array with 3 elements representing the plane normal.
        origin (npt.NDArray[np.float32]): Array with 3 elements representing the plane origin.
        thickness (float): Thickness of the plane.
        extent (npt.NDArray[np.float32]): Array with 3 elements representing the extent of the plane.
    """

    def __init__(
            self, 
            name, 
            normal, 
            origin, 
            thickness, 
            tag,
            extent=None
    ) -> None:
        """
        Initialize GEOSThickPlane class.
    
        Args:
            name (str): Plane name.
            normal (list[float]): Normal vector for plane.
            origin (list[float]): Origin point for plane.
            thickness (float): Thickness of plane.
            tag (str): Tag for the plane.
            extent (list[float], optional): Extent of plane in u, v directions. Defaults to None.
    
        Returns:
            None
        """
        if tag != "ThickPlane":
            raise ValueError(f"tag must be 'ThickPlane', got '{tag}'")
        self.name = name
        self.normal = np.array(normal, dtype=np.float32)
        self.normal = self.normal / np.linalg.norm(self.normal)
        self.origin = np.array(origin, dtype=np.float32)
        self.thickness = float(thickness)
        self.extent = extent

    def get_plane_basis(self) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """
        Return two orthonormal in-plane vectors `(u, v)`.
    
        Args:
            None
    
        Returns:
            tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]: Orthonormal basis vectors `u` and `v` in the plane.
        """
        # Pick an axis that is not parallel to the normal.
        if abs(self.normal[0]) < 0.9:
            seed = np.array([1.0, 0.0, 0.0])
        else:
            seed = np.array([0.0, 1.0, 0.0])

        u = np.cross(self.normal, seed)
        u = u / np.linalg.norm(u)
        v = np.cross(self.normal, u)
        v = v / np.linalg.norm(v)
        return u, v

    def get_rectangular_patch(self, size=10) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """
        Return `(lower, upper)` rectangular vertices for rendering the thick plane.
    
        Args:
            None
    
        Returns:
            tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]: Arrays with 4 vertices for lower and upper faces of the thick plane.
        """
        u, v = self.get_plane_basis()
        half_t = self.thickness / 2.0

        # Keep `extent` for future bounding support; use `size` for now.
        size_u = size
        size_v = size

        corners = [
            self.origin - size_u * u - size_v * v,
            self.origin + size_u * u - size_v * v,
            self.origin + size_u * u + size_v * v,
            self.origin - size_u * u + size_v * v,
        ]

        lower = np.array([corner - half_t * self.normal for corner in corners])
        upper = np.array([corner + half_t * self.normal for corner in corners])
        return lower, upper

    def get_faces(self, size: float = 10) -> list[npt.NDArray[np.float32]]:
        """
        Return all 6 quad faces for plotting as polygons.
    
        Args:
            None
    
        Returns:
            list[npt.NDArray[np.float32]]: List of 6 arrays with 4 vertices each for the faces of the thick plane.
        """
        lower, upper = self.get_rectangular_patch(size)
        return [
            lower,
            upper,
            np.array([lower[0], lower[1], upper[1], upper[0]]),
            np.array([lower[1], lower[2], upper[2], upper[1]]),
            np.array([lower[2], lower[3], upper[3], upper[2]]),
            np.array([lower[3], lower[0], upper[0], upper[3]]),
        ]

    def distance_to_point(self, point: npt.NDArray[np.float32]) -> float:
        """
        Return signed distance from `point` to the plane center.
    
        Args:
            None
    
        Returns:
            float: Signed distance from `point` to the plane center.
        """
        return np.dot(point - self.origin, self.normal)

    def __repr__(self) -> str:
        return (
            f"ThickPlane(name='{self.name}', normal={self.normal}, "
            f"origin={self.origin}, thickness={self.thickness})"
        )