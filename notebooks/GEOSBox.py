import numpy as np
import numpy.typing as npt

class GEOSBox:
    """Represents a GEOS box and its attributes.

    Attributes:
        name (str): Name of box object.
        xMin (npt.NDArray[np.float32]): Array with 3 elements representing minimum values for box.
        xMax (npt.NDArray[np.float32]): Array with 3 elements representing maximum values for box.
    """

    def __init__(
            self, 
            name: str, 
            xMin: list[float], 
            xMax: list[float], 
            tag: str
    ) -> None:
        """
        Initialize GEOSBox class.
    
        Args:
            name (str): Box name.
            xMin (list[float]): Minimum values for box.
            xMax (list[float]): Maximum values for box.
            tag (str): Tag used for verification.
        Returns:
            None
        """
        self.name = name
        self.xMin = np.array(xMin, dtype=np.float32)
        self.xMax = np.array(xMax, dtype=np.float32)
        if tag != "Box":
            raise ValueError(f"tag must be 'Box', got '{tag}'")

    def get_vertices(self) -> npt.NDArray[np.float32]:
        """
        Return the 8 box vertices in a consistent order.
    
        Args:
            None

        Returns:
            npt.NDArray[np.float32]: Array with 8 box verticies.
        """
        xmin, ymin, zmin = self.xMin
        xmax, ymax, zmax = self.xMax
        return np.array(
            [
                [xmin, ymin, zmin],
                [xmax, ymin, zmin],
                [xmax, ymax, zmin],
                [xmin, ymax, zmin],
                [xmin, ymin, zmax],
                [xmax, ymin, zmax],
                [xmax, ymax, zmax],
                [xmin, ymax, zmax],
            ],
            dtype=np.float32,
        )

    def get_faces(self) -> npt.NDArray[np.float32]:
        """
        Return 6 quad faces as vertex index lists.
    
        Args:
            None

        Returns:
            npt.NDArray[np.float32]: Array with 6 quad faces.
        """
        return np.array([
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [3, 2, 6, 7],
            [0, 3, 7, 4],
            [1, 2, 6, 5],
        ], dtype=np.float32)

    def get_edges(self) -> npt.NDArray[np.float32]:
        """
        Return 12 edges as index pairs.
    
        Args:
            None

        Returns:
            npt.NDArray[np.float32]: Array with 12 pairs representing box edges.
        """
        return np.array([
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7],
        ], dtype=np.float32)

    def contains_point(self, point: list[float]) -> bool:
        """
        Check whether a 3D point lies inside or on the box.
    
        Args:
            None

        Returns:
            bool: Truth value for presence of input point in box.
        """
        point_arr = np.array(point, dtype=float)
        return np.all(point_arr >= self.xMin) and np.all(point_arr <= self.xMax)

    def __repr__(self) -> str:
        return f"Box(name='{self.name}', xMin={self.xMin}, xMax={self.xMax})"