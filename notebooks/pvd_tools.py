# This file is adapted from https://docs.pyvista.org/api/readers/_autosummary/pyvista.pvdreader#pyvista.PVDReader and GEOS's pyvistaTools.py + genericHelpers.py

# Imports
from typing import Optional, Union
from geos.mesh.utils import pyvistaTools as  pv_geos
from vtkmodules.vtkCommonDataModel import (vtkPlane, vtkPolyData, vtkUnstructuredGrid)
from vtkmodules.vtkCommonCore import vtkDataArray
from vtkmodules.vtkFiltersCore import vtk3DLinearGridPlaneCutter
from geos.mesh.utils.arrayModifiers import transferPointDataToCellData
from geos.mesh.utils.arrayHelpers import (getAttributeValuesAsDF, computeCellCenterCoordinates)
from geos.mesh.utils.genericHelpers import getMultiBlockBounds
from pyvista import PVDReader
import vtkmodules.util.numpy_support as vnp

import pyvista as pv
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib import pyplot as plt

COMPONENTS = ['xx', 'yy', 'zz', 'yz', 'xz', 'xy']


def get_array_names(m: pv.MultiBlock, arr: list=[], level: int=0):
    """
    Recursively get array names from a PyVista MultiBlock mesh.

    Parameters:
    -----------
        m (pv.MultiBlock): The PyVista MultiBlock mesh.
        arr (list): List to store array names.
        level (int): Current recursion level.   

    Returns:
    -----------
        int: Recursion termination flag.
        list: List of array names.
    """

    if isinstance(m, pv.MultiBlock):
        # print("  " * level + str(len(m)))
        for i in range(len(m)):
            leaf, arr = get_array_names(m[i], arr=arr, level=level+1)
            if leaf > 0:
                break
    else:
        # print("  " * level + str(m.array_names))
        arr.append(m.array_names) 
        return 1, arr
    return 0, arr

def get_info(file_path: str) -> list[float]:
    """
    Get the time values from a PVD file.

    Parameters:
    -----------
        file_path (str): Path to the PVD file.

    Returns:
    
        list[float]: List of time values.
        list[list[str]]: List of list of array names.
        list[list[float]]: List of list of bounds.
    """
    reader = pv.get_reader(file_path)
    time_steps = reader.time_values

    reader.set_active_time_point(0)
    m = reader.read()
    array_names = get_array_names(m, arr=[], level=0)[1]
    

    return time_steps, array_names, getMultiBlockBounds(m)

# original from GEOS genericHelpers.py
def extractSurfaceFromElevation( mesh: vtkUnstructuredGrid, normal_vector: tuple[float, float, float], origin_vector: tuple[float, float, float]) -> vtkPolyData:
    """Extract surface at a constant elevation from a mesh.

    Args:
        mesh (vtkUnstructuredGrid): input mesh
        normal_vector (tuple[float, float, float]): normal vector of the plane
        origin_vector (tuple[float, float, float]): origin point of the plane

    Returns:
        vtkPolyData: output surface
    """
    assert mesh is not None, "Input mesh is undefined."
    assert isinstance( mesh, vtkUnstructuredGrid ), "Wrong object type"

    bounds: tuple[ float, float, float, float, float, float ] = mesh.GetBounds()
    ooX: float = ( bounds[ 0 ] + bounds[ 1 ] ) / 2.0
    ooY: float = ( bounds[ 2 ] + bounds[ 3 ] ) / 2.0

    # check check origin_vector within bounds
    assert bounds[0] <= origin_vector[0] <= bounds[1], "Origin X is out of bounds."
    assert bounds[2] <= origin_vector[1] <= bounds[3], "Origin Y is out of bounds."
    assert bounds[4] <= origin_vector[2] <= bounds[5], "Origin Z is out of bounds."

    plane: vtkPlane = vtkPlane()
    # plane.SetNormal( 0.0, 0.0, 1.0 )
    plane.SetNormal( *normal_vector )
    # plane.SetOrigin( ooX, ooY, elevation )
    plane.SetOrigin( *origin_vector )

    cutter = vtk3DLinearGridPlaneCutter()
    cutter.SetInputDataObject( mesh )
    cutter.SetPlane( plane )
    cutter.SetInterpolateAttributes( True )
    cutter.Update()
    return cutter.GetOutputDataObject( 0 )

# original from  GEOS pyvistaTools.py
def loadDataSet(
    reader: pv.PVDReader,
    timeStepIndexes: list[ int ],
    normal_vector: tuple[ float, float, float ],
    origin_vector: tuple[ float, float, float ],
    properties: tuple[ str ],
    target: Optional[ str ] = "volume"
) -> tuple[ dict[ str, pd.DataFrame ], npt.NDArray[ np.float64 ] ]:
    """Load the data using pyvista and extract properties from horizontal slice.

    Args:
        reader (pv.PVDReader): Pyvista pvd reader.
        timeStepIndexes (list[int]): List of time step indexes to load.
        normal_vector (tuple[float, float, float]): Normal vector of the plane.
        origin_vector (tuple[float, float, float]): Origin point of the plane.
        properties (tuple[str]): List of properties to extract.
        target (str, optional): Target domain to extract properties from. Options: ["volume", "fracture"] Defaults to "volume".

    Returns:
        tuple[dict[str, pd.DataFrame], npt.NDArray[np.float64]]: Tuple containing
            a dictionary with times as keys and dataframe with properties as
            values, and an array with cell center coordinates of the slice.

    """
    timeToPropertyMap: dict[ str, pd.DataFrame ] = {}
    surface: vtkPolyData = None
    timeValues: list[ float ] = reader.time_values
    for index in timeStepIndexes:
        if index >= len( timeValues ):
            raise IndexError( "Time step index is out of range." )

        time: float = timeValues[ index ]
        reader.set_active_time_value( time )
        inputMesh: pv.Multiblock = reader.read()

        _kword =  pv_geos.GeosDomainNameEnum.VOLUME_DOMAIN_NAME.value
        if target == "fracture":
            _kword = pv_geos.GeosDomainNameEnum.FAULT_DOMAIN_NAME.value

        volMesh: Optional[ Union[ pv.MultiBlock, pv.UnstructuredGrid ] ] = pv_geos.getBlockByName(inputMesh, _kword )
        if not volMesh:
            raise AttributeError( "Volumic mesh was not found." )

        # Merge volume block
        mergedMesh: pv.UnstructuredGrid = volMesh.combine(
            merge_points=True ) if isinstance( volMesh, pv.MultiBlock ) else volMesh
        if not mergedMesh:
            raise ValueError( "Merged mesh is undefined." )

        # Extract data
        surface = extractSurfaceFromElevation( mergedMesh, normal_vector, origin_vector )
        # Transfer point data to cell center
        surface = vtkPolyData.SafeDownCast( transferPointDataToCellData( surface ) )
        timeToPropertyMap[ str( time ) ] = getAttributeValuesAsDF( surface, properties )

    # Get cell center coordinates
    if not surface:
        raise ValueError( "Surface are undefined." )
    pointsCoords: vtkDataArray = computeCellCenterCoordinates( surface )
    if not pointsCoords:
        raise ValueError( "Cell center are undefined." )
    pointsCoordsNp: npt.NDArray[ np.float64 ] = vnp.vtk_to_numpy( pointsCoords )
    return ( timeToPropertyMap, pointsCoordsNp )

def visualize_3d_data(data_dict, coordinates, property_name, time_step=None, clim=None):
    """
    Visualize 3D data at a specific time step
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary with times as keys and dataframes with properties as values
    coordinates : array-like
        Array with cell center coordinates (shape: [n_points, 3])
    property_name : str
        Name of the property to visualize
    time_step : float, optional
        Specific time step to visualize. If None, creates an interactive plot
    """
    # Create a PyVista PolyData object
    cloud = pv.PolyData(coordinates)
    
    if time_step is None:
        time_step = list(data_dict.keys())[0]

    if time_step not in data_dict:
        raise ValueError(f"Time step {time_step} not found in data")
        
    # Add the property data to the mesh
    cloud.point_data[property_name] = data_dict[time_step][property_name].values
    
    # Create the plotter
    plotter = pv.Plotter()
    plotter.add_mesh(cloud, scalars=property_name, cmap='viridis', 
                    clim=clim, show_edges=False, point_size=5)
    plotter.show()

def split_slices(time_dict, coord, n_slices=2, key_prefix="strain"):
    """
    Split and organize strain data into separate slices based on spatial coordinates.
    
    Parameters:
    -----------
    time_dict : dict
        Dictionary containing time steps as keys and DataFrames with strain components as values.
        Example format: {time_step: {strain_component: values}}
    coord : array-like
        Array of coordinates (x, y, z) for each point where strain is measured
    n_slices : int, default=2
        Number of slices to split the data into along the spatial dimension
    key_prefix : str, default="strain"
        Prefix for strain component keys in the time_dict
        
    Returns:
    --------
    ret_dict : dict
        Dictionary containing reorganized strain data where:
        - Keys are formatted as "strain_component_slice_number"
        - Values are DataFrames containing coordinates and strain values for each slice,
          sorted by x and y coordinates
    """
    strain_names =  {
        f"{key_prefix}_0": COMPONENTS[0],
        f"{key_prefix}_1": COMPONENTS[1],
        f"{key_prefix}_2": COMPONENTS[2],
        f"{key_prefix}_3": COMPONENTS[3],
        f"{key_prefix}_4": COMPONENTS[4],
        f"{key_prefix}_5": COMPONENTS[5]
        }
    ret_dict = {}
    for t, df in time_dict.items(): # get time slice key and dataframe
        for k, v in df.items(): # iterate through strain components (k) and their values (v)
            # create a dataframe with spatial coordinates and add strain values
            new_name = f"{t}_{strain_names[k]}"
            temp = pd.DataFrame(coord, columns=["x", "y", "z"])
            temp[new_name] = v
            
            # split data into specified number of slices
            for i in range(n_slices):
                # select every nth row to create each slice (where n = slices)
                temp_i = temp.iloc[i::n_slices]
                # sort values by x then y coordinates for proper organization
                temp_i_sorted = temp_i.sort_values(by=["x", "y"])

                # store sorted slice in return dictionary with key format: "strain_component_slice_number"
                ret_dict[f"{new_name}_{i}"] = temp_i_sorted 

    return ret_dict

def split_strain_components(strain_dict, reshape_lambda=None, time_steps=None, rotate=False, plot=True, clim=None, keep_surface=None, 
                            x_label=None, y_label=None):
    """
    Split strain into 6 components and organize them by time step with plotting option.

    Parameters:
    -----------
    strain_dict : dict
        Dictionary containing every component time step as keys and DataFrames with coordinates and strain values as datafreames
    reshape_lambda : function, optional
        Function to reshape the strain component arrays for plotting. If None, no reshaping is done
    time_steps : list, optional
        List of time steps to be processed. If None, all time steps are processed.
    rotate : bool, default=False
        If True, rotate the strain component arrays by 90 degrees if the data is 2D.
    plot : bool, default=True
        Whether to plot the strain components for each time step
    clim : tuple, optional
        Color limits for the plots. If None, automatic scaling is used.
    keep_surface : int, optional
        If specified, only process and plot the given surface number (0 or 1). If None, process both surfaces.
    x_label : str, optional
        Label for the x-axis in the plots. If None, no label is set.
    y_label : str, optional
        Label for the y-axis in the plots. If None, no label is set.

    Returns:
    --------
    ret_dict : dict
        Dictionary containing reorganized strain data with times steps as keys and components as values:
        - Keys are formatted as time step: "time_step_surface_number" -> e.g., "2.0_0"
        - Values are array with strain components as keys in the following order: ['xx', 'yy', 'zz', 'yz', 'xz', 'xy']

    """

    avaiable_times = [*{float(i.split('_')[0]) for i in strain_dict.keys()}]
    avaiable_times.sort()
    surfaces = ["0", "1"] if keep_surface is None else [str(keep_surface)]
    ret_dict = {}

    if reshape_lambda is None:
        reshape_lambda = lambda x: x  # No reshaping by default

    if time_steps is None:
        time_steps = avaiable_times
    else:
        for t in time_steps:
            if t not in avaiable_times:
                raise ValueError(f"Time step {t} not found in strain_dict")
    
    for t in time_steps:
        for s in surfaces:
            components_data = []
            for c in COMPONENTS:
                key = f"{t}_{c}_{s}"
                if key not in strain_dict:
                    raise ValueError(f"Key {key} not found in strain_dict")
                _slice = strain_dict[key][f"{t}_{c}"].values
                _slice = reshape_lambda(_slice)
                if rotate and _slice.ndim == 2:
                    _slice = np.moveaxis(_slice, 1, 0)
                components_data.append(_slice)
            ret_dict[f"{t}_{s}"] = components_data

            if plot:
                fig = plt.figure(figsize=(28, 4))
                for i, comp_data in enumerate(components_data):
                    if comp_data.ndim != 2:
                        print(f"Component data for {t}, {s}, {COMPONENTS[i]} is not 2D, skipping plot.")
                        continue
                    plt.subplot(161 + i)
                    plt.imshow(comp_data, cmap='seismic', clim=clim)
                    plt.title(f"{COMPONENTS[i]}")
                    if x_label is not None:
                            plt.xlabel(x_label)
                    if y_label is not None:
                            plt.ylabel(y_label)
               
                plt.colorbar(ax=plt.gcf().get_axes())
                plt.suptitle(f"Strain components at time {t}, surface {s}")
                plt.show()

    return ret_dict