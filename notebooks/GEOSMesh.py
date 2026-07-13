import numpy as np
import numpy.typing as npt

class GEOSMesh:
    """Represents a GEOS mesh and its attributes.

    Attributes:
        name (str): Mesh name.
        xCoords (npt.NDArray[np.float32]): X-axis coordinates array.
        yCoords (npt.NDArray[np.float32]): Y-axis coordinates array.
        zCoords (npt.NDArray[np.float32]): Z-axis coordinates array.
        nx (npt.NDArray[np.int32]): X-axis element count array.
        ny (npt.NDArray[np.int32]): Y-axis element count  array.
        nz (npt.NDArray[np.int32]): Z-axis element count array.
        xBias (npt.NDArray[np.float32]): X-axis bias.
        yBias (npt.NDArray[np.float32]): Y-axis bias.
        zBias (npt.NDArray[np.float32]): Z-axis bias.
        elementTypes (str): String denoting element types.
        nodes (npt.NDArray[np.float32]): Array with coordinates of mesh nodes.
        elements (npt.NDArray[np.int32]): Array with element verticies of mesh nodes.
        cellBlockNames (list[str]): List of strings for block names.
    """
    def __init__(
        self,
        xCoords: list[float],
        yCoords: list[float],
        zCoords: list[float],
        nx: list[int],
        ny: list[int],
        nz: list[int],
        tag: str,
        name: str = "mesh",
        xBias: list[float] = None,
        yBias: list[float] = None,
        zBias: list[float] = None,
        elementTypes: str = "C3D8",
        cellBlockNames: list[str] = None,
    ) -> None:
        """
        Initialize GEOSMesh class.
    
        Args:
            xCoords (list[float]): X-axis coordinates array.
            yCoords (list[float]): Y-axis coordinates array.
            zCoords (list[float]): Z-axis coordinates array.
            nx (list[int]): X-axis element count array.
            ny (list[int]): Y-axis element count  array.
            nz (list[int]): Z-axis element count array.
            tag (str): XML tag used for verification.
            name (str): Optional mesh name.
            xBias (list[float]): X-axis optional bias.
            yBias (list[float]): Y-axis optional bias.
            zBias (list[float]): Z-axis optional bias.
            elementTypes (str): Optional string denoting element types.
            cellBlockNames (list[str]): Block names.
    
        Returns:
            None
        """
        # store core mesh definitions.
        self.xCoords = np.array(xCoords, dtype=np.float32)
        self.yCoords = np.array(yCoords, dtype=np.float32)
        self.zCoords = np.array(zCoords, dtype=np.float32)
        self.nx = np.array(nx, dtype=np.int32)
        self.ny = np.array(ny, dtype=np.int32)
        self.nz = np.array(nz, dtype=np.int32)

        # chekc if tag is valied
        if tag != "InternalMesh":
            raise ValueError(f"tag must be 'InternalMesh', got '{tag}'")

        if isinstance(elementTypes, list) and len(elementTypes) == 1:
            self.elementTypes = elementTypes[0]
        else:
            self.elementTypes = "C3D8"
            
        self.name = name

        # check if coords and number of element arrays are valied
        self._validate_inputs()

        # check bias values
        self.xBias = self._check_bias(xBias, len(self.nx), "xBias")
        self.yBias = self._check_bias(yBias, len(self.ny), "yBias")
        self.zBias = self._check_bias(zBias, len(self.nz), "zBias")

        # set block names
        if cellBlockNames is None:
            self.cellBlockNames = ["block"]
        elif isinstance(cellBlockNames, str):
            self.cellBlockNames = [cellBlockNames]
        else:
            self.cellBlockNames = cellBlockNames

        # generate nodes and elements
        if self.elementTypes not in ["C3D8", "C3D6"]:
            raise ValueError("Only 'C3D8' and 'C3D6' elementTypes are supported at this time.")

        self.nodes, self.elements = self.generate_mesh()


    def _validate_inputs(self):
        """
        Validate coordinates and number of element arrays are valied i.e. (len(xCoord) == len(nx) + 1) and (all(nx) > 0).
    
        Args:
            None
    
        Returns:
            None
            or 
            ValueError
        """
        if len(self.xCoords) != len(self.nx) + 1:
            raise ValueError("xCoords must have one more entry than nx")
        if len(self.yCoords) != len(self.ny) + 1:
            raise ValueError("yCoords must have one more entry than ny")
        if len(self.zCoords) != len(self.nz) + 1:
            raise ValueError("zCoords must have one more entry than nz")

        if np.any(self.nx <= 0) or np.any(self.ny <= 0) or np.any(self.nz <= 0):
            raise ValueError("All element counts (nx, ny, nz) must be positive")

    
    @staticmethod
    def _check_bias(bias_values: list[float], n_blocks: int, label: str) -> npt.NDArray[np.float32]:
        """
        Checks if bias has valied values (-1, 1).
    
        Args:
            bias_values (list[flaot]): List of bias values for each block in current axis.
            n_blocks (int): number of block in current axis.
            label (str): String denoting the current axis used for error printing.
    
        Returns:
            npt.NDArray[np.float32]: Array of valid bias values.
            or 
            ValueError
        """
        # check if no bias was passed, create and return an array matching number of blocks in current axis
        if bias_values is None:           
            return np.zeros(n_blocks, dtype=np.float32)
    
        bias = np.array(bias_values, dtype=np.float32)
        # if length of input bias values doesn't match number of blocks, raise an error
        if len(bias) != n_blocks:
            raise ValueError(f"{label} must have {n_blocks} entries, got {len(bias)}")
        # if any of the biase values are outside of the allowed value range, raise an error
        if np.any(np.abs(bias) >= 1.0):
            raise ValueError(f"{label} values must be in (-1, 1)")
        # all passed, return bias array
        return bias

        
    @staticmethod
    def _generate_biased_coords(coords: npt.NDArray[np.float32], n_elements: npt.NDArray[np.int32], bias: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        Generate 1D nodal coordinates for piecewise blocks with optional bias.
    
        Args:
            coords (npt.NDArray[np.float32]): Array of a given axis coordinates.
            n_elements (npt.NDArray[np.int32]): Array of number of elements in each block.
            bias (npt.NDArray[np.float32]): Array of bias values for each block.
    
        Returns:
            npt.NDArray[np.float32]: Array of coordinates based on given inputs for current axis.
        """
        all_coords = []
        # chunck coords based on number of blocks
        for block_idx in range(len(n_elements)):
            # define start and end of the current chunck
            start = coords[block_idx]
            end = coords[block_idx + 1]
            # define the count of elements in current chunck and associated bias
            count = n_elements[block_idx]
            b = bias[block_idx]

            # if current bias below threshold, use linear spacing
            if abs(b) < 1e-12:
                block_coords = np.linspace(start, end, count + 1)
            else:
                # linearly vary element widths from left to right
                block_length = end - start
                avg_dx = block_length / count
                dx_left = (1 + b) * avg_dx
                dx_right = (1 - b) * avg_dx
                # calculate coordinates
                block_coords = [start]
                x = start
                for j in range(count):
                    t = j / count
                    dx = dx_left * (1 - t) + dx_right * t
                    x += dx
                    block_coords.append(x)

                # rescale so the final point is exactly the block end
                block_coords = np.array(block_coords)
                block_coords = start + (block_coords - start) * block_length / (block_coords[-1] - start)

            # avoid duplicate node at block boundaries
            if block_idx == 0:
                all_coords.extend(block_coords)
            else:
                all_coords.extend(block_coords[1:])

        return np.array(all_coords, dtype=np.float32)

    
    def generate_mesh(self) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int32]]:
        """
        Generate nodal coordinates and element connectivity.
    
        Args:
            None
    
        Returns:
            npt.NDArray[np.float32]: Array of node coordinates of mesh.
            npt.NDArray[np.int32]: Array of elements coordinates of mesh nodes.
        """
        # genersate nodes for each axis
        x_nodes = self._generate_biased_coords(self.xCoords, self.nx, self.xBias)
        y_nodes = self._generate_biased_coords(self.yCoords, self.ny, self.yBias)
        z_nodes = self._generate_biased_coords(self.zCoords, self.nz, self.zBias)
        
        # get axis elm numbers
        nx_total = len(x_nodes)
        ny_total = len(y_nodes)
        nz_total = len(z_nodes)

        # build node array in k-j-i order
        nodes = []
        for k in range(nz_total):
            for j in range(ny_total):
                for i in range(nx_total):
                    nodes.append([x_nodes[i], y_nodes[j], z_nodes[k]])
                    
        nodes = np.array(nodes, dtype=np.float32)

        # build element connectivity
        elements = []
        stride_xy = nx_total*ny_total
        for k in range(nz_total - 1):
            for j in range(ny_total - 1):
                for i in range(nx_total - 1):
                    # define 8 nodes for each element
                    n0 = i + j*nx_total + k*stride_xy
                    n1 = n0 + 1
                    n2 = n1 + nx_total
                    n3 = n0 + nx_total
                    n4 = n0 + stride_xy
                    n5 = n1 + stride_xy
                    n6 = n2 + stride_xy
                    n7 = n3 + stride_xy

                    if self.elementTypes == "C3D8":
                        # one hexahedral element per i-j-k cell
                        elements.append([n0, n1, n2, n3, n4, n5, n6, n7])
                    elif self.elementTypes == "C3D6":
                        # split each hexa cell by the XY diagonal (n0 -> n2)
                        # into two triangular prisms (wedges)
                        elements.append([n0, n1, n2, n4, n5, n6])
                        elements.append([n0, n2, n3, n4, n6, n7])
                    else:
                        raise ValueError(
                            f"Unsupported elementTypes '{self.elementTypes}'. Expected 'C3D8' or 'C3D6'."
                        )

        elements = np.array(elements, dtype=np.int32)
        
        return nodes, elements


    def get_mesh_info(self) -> dict[str, float | int]:
        """
        Return summary stats for quick inspection/plot labels.
    
        Args:
            None
    
        Returns:
            dict: information about the current mesh.
        """
        return {
            "name": self.name,
            "num_nodes": len(self.nodes),
            "num_elements": len(self.elements),
            "x_range": (self.nodes[:, 0].min(), self.nodes[:, 0].max()),
            "y_range": (self.nodes[:, 1].min(), self.nodes[:, 1].max()),
            "z_range": (self.nodes[:, 2].min(), self.nodes[:, 2].max()),
        }

    
    def __repr__(self) -> str:
        ret_str = ""
        for k, v in self.get_mesh_info().items():
            ret_str += f"{k}: {v}\n"
        return ret_str
    
        