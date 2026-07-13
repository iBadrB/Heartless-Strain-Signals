import numpy as np
import numpy.typing as npt


class GEOSRectangle:
    """Represent a GEOS rectangle defined by center, orientation, and dimensions."""

    def __init__(
        self,
        name: str,
        normal: list[float],
        origin: list[float],
        tag: str,
        lengthVector: list[float] | None = None,
        widthVector: list[float] | None = None,
        dimensions: list[float] | tuple[float, float] | None = None,
    ) -> None:
        if tag != "Rectangle":
            raise ValueError(f"tag must be 'Rectangle', got '{tag}'")

        self.name = name
        self.normal = np.asarray(normal, dtype=np.float32)
        n_norm = np.linalg.norm(self.normal)
        if n_norm == 0.0:
            raise ValueError("normal must be non-zero")
        self.normal = self.normal / n_norm

        self.origin = np.asarray(origin, dtype=np.float32)

        if dimensions is None or len(dimensions) != 2:
            raise ValueError("dimensions must contain [length, width]")
        self.dimensions = np.asarray(dimensions, dtype=np.float32)
        if np.any(self.dimensions <= 0.0):
            raise ValueError("dimensions values must be positive")

        self.lengthVector = None if lengthVector is None else np.asarray(lengthVector, dtype=np.float32)
        self.widthVector = None if widthVector is None else np.asarray(widthVector, dtype=np.float32)

        self.u, self.v = self.get_plane_basis()

    def get_plane_basis(self) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Return orthonormal in-plane basis vectors `(u, v)` for the rectangle."""
        if self.lengthVector is not None and np.linalg.norm(self.lengthVector) > 0.0:
            u = self.lengthVector / np.linalg.norm(self.lengthVector)
        elif self.widthVector is not None and np.linalg.norm(self.widthVector) > 0.0:
            u = np.cross(self.widthVector, self.normal)
            u = u / np.linalg.norm(u)
        else:
            # fallback axis that is not parallel to the normal
            if abs(self.normal[0]) < 0.9:
                seed = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            else:
                seed = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            u = np.cross(self.normal, seed)
            u = u / np.linalg.norm(u)

        # project to ensure basis is in the rectangle plane
        u = u - np.dot(u, self.normal) * self.normal
        u_norm = np.linalg.norm(u)
        if u_norm == 0.0:
            raise ValueError("Could not construct a valid in-plane length direction")
        u = u / u_norm

        if self.widthVector is not None and np.linalg.norm(self.widthVector) > 0.0:
            v = self.widthVector - np.dot(self.widthVector, self.normal) * self.normal
            v_norm = np.linalg.norm(v)
            if v_norm > 0.0:
                v = v / v_norm
            else:
                v = np.cross(self.normal, u)
                v = v / np.linalg.norm(v)
        else:
            v = np.cross(self.normal, u)
            v = v / np.linalg.norm(v)

        # enforce orthonormal basis
        v = v - np.dot(v, u) * u
        v = v / np.linalg.norm(v)

        return u.astype(np.float32), v.astype(np.float32)

    def get_corners(self) -> npt.NDArray[np.float32]:
        """Return the 4 rectangle corners as an ordered polygon in 3D."""
        half_length = 0.5 * float(self.dimensions[0])
        half_width = 0.5 * float(self.dimensions[1])

        return np.array(
            [
                self.origin - half_length * self.u - half_width * self.v,
                self.origin + half_length * self.u - half_width * self.v,
                self.origin + half_length * self.u + half_width * self.v,
                self.origin - half_length * self.u + half_width * self.v,
            ],
            dtype=np.float32,
        )

    def __repr__(self) -> str:
        return (
            f"Rectangle(name='{self.name}', normal={self.normal}, origin={self.origin}, "
            f"dimensions={self.dimensions})"
        )
