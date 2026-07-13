# This file is modified from the original file created by the GEOS team. 
# Original file: GEOS/inputFiles/hydraulicFracturing/scripts/hydrofractureQueries.py

import sys
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import math
from math import sin, cos, tan, exp
from geos import hdf5_wrapper
import xml.etree.ElementTree as ElementTree
from glob import glob
from typing import Optional, Union
import numpy.typing as npt
# import pvd_tools

COMPONENTS = ['xx', 'yy', 'zz', 'yz', 'xz', 'xy']

def read_hdf5(file_path: str) -> tuple[hdf5_wrapper.hdf5_wrapper, str]:
    """
    Read HDF5 file and return an HDF5Wrapper object.

    Parameters:
    -----------
        file_path (str): The path to the HDF5 file.

    Returns:
    -----------
        hdf5_wrapper.HDF5Wrapper: An HDF5Wrapper object.
        str: File name.
    """
    file =  hdf5_wrapper.hdf5_wrapper(file_path)
    file_name = f"{file_path.split("/")[-2]}/{file_path.split("/")[-1].split(".")[0]}"
    return file, file_name

def get_attributes(hdf5_ds: hdf5_wrapper.hdf5_wrapper, attributes_list: list=None, add_renamed: bool=True, return_all: bool=False) -> dict:
    """
    Get attributes from an HDF5 dataset.

    Parameters:
    -----------
        hdf5_ds (hdf5_wrapper.HDF5Wrapper): An HDF5Wrapper object.
        attributes_list (list, optional): A list of attribute names to retrieve.
        add_renamed (bool, optional): Whether to add renamed attributes when attributes_list is not provided. Defaults to True.
        return_all (bool, optional): Return all avaiable attributes. Defaults to False. 
    Returns:
    -----------
        dict: A dictionary containing the attributes of the dataset.
    """
    def _find_sufix(keys):
        allowed_sufixes = {"elementCenter", "elementAperture", "Time", "pressure", "elementArea", "hydraulicAperture"}
        sufixes = set()
        for key in keys:
            key_sufix = key.split(" ")[-1] if " " in key else key
            if key_sufix in allowed_sufixes:
                continue
            else:
                sufixes.add(key_sufix)
        
        if len(sufixes) == 0:
            return ""
        
        assert len(sufixes) == 1, f"Multiple sufixes found: {sufixes}"
        return sufixes.pop()
    _rename_dict = {
        "pressure elementCenter": "elementCenter",
        "pressure Time": "Time",
        "averageStrain elementCenter": "elementCenter",
        "averageStrain Time": "Time"
    }

    _rename_dict = None if attributes_list is not None and add_renamed == False else _rename_dict

    if attributes_list is None:
        attributes_list = [
            "pressure elementCenter",
            "pressure Time",

            "elementAperture",
            "elementArea",
            "hydraulicAperture",
            "pressure"
        ]

    attributes_sufix = _find_sufix(hdf5_ds.keys())
    # print(f"Identified sufix: '{attributes_sufix}' in dataset keys. Will attempt to match attributes by removing this sufix from target attribute names.") if attributes_sufix else print("No sufix identified in dataset keys. Attempting to match attributes by exact name.")

    attributes = {}
    keys = hdf5_ds.keys()

    if return_all:
        attributes = {k: hdf5_ds[k] for k in keys}
        return attributes

    for key in attributes_list:
        found = False
        keys_dict = {k.rstrip(attributes_sufix).rstrip(): k for k in keys}
        if key in keys_dict:
            data = hdf5_ds[keys_dict[key]]
            # if " " in key:
            #     key = key.split(" ")[-1]
            attributes[key] = data
        else:
            # else if part of the key is in the dataset, i.e. 'pressure hf_zone' rather than 'pressure'
            for ds_key in hdf5_ds.keys():
                if key in ds_key:
                    data = hdf5_ds[ds_key]
                    attributes[ds_key] = data
                    found = True

            if not found:
                print(f"Attribute {key} not found in dataset.\nAvailable attributes: {hdf5_ds.keys()}")
            # print(f"Attribute {key} not found in dataset.\nAvailable attributes: {hdf5_ds.keys()}")
    if _rename_dict is not None:
        for key in _rename_dict.keys():
            if key in attributes:
                attributes[_rename_dict[key]] = attributes[key]

    return attributes

def plot_cords(elm_centers: npt.NDArray[np.float64], time_step: int=-1) -> None:
    """
    Plot the coordinates of the element centers.

    Parameters:
    -----------
        elm_centers (npt.NDArray[np.float64]): An array of element center coordinates.
        time_step (int, optional): The time step to plot. Defaults to -1 (last time step).

    Returns:
    -----------
        None
    """
    xcord = elm_centers[time_step, :, 0]
    ycord = elm_centers[time_step, :, 1]
    zcord = elm_centers[time_step, :, 2]

    plt.subplot(131)
    plt.plot(xcord)
    plt.title("xcord")

    plt.subplot(132)
    plt.plot(ycord)
    plt.title("ycord")

    plt.subplot(133)
    plt.plot(zcord)
    plt.title("zcord")

    plt.show()

def get_frac_length(elm_centers: npt.NDArray[np.float64], selected_axis: str, method: int=0, get_negative: bool=False) -> npt.NDArray[ np.float64 ]:
    """
    Calculate the length of fractures based on their center coordinates.

    Parameters:
    -----------
        elm_centers (npt.NDArray[np.float64]): An array of element center coordinates.
        selected_axis (str): The axis along which to calculate the length.
        method (int, optional): The method to use for calculation [0, 1]. Defaults to 0.
        get_negative (bool, optional): Whether to include negative values in the calculation. Defaults to False.

    Returns:
    -----------
        npt.NDArray[np.float64]: An array containing the calculated lengths.
    """
    axis_dict = {"x": 0, "y": 1, "z": 2}
    length_arr_pos = []
    length_arr_neg = []

    for t in range(0, elm_centers.shape[0]):
        axis = elm_centers[t, :, axis_dict[selected_axis]]
        
        if method == 1:
            # Method 2: remove trailing zeros and use last 2 values
            axis_clean_pos = axis[axis >= 0]
            axis_clean_neg = axis[axis < 0][::-1]

        else:
            # Method 1: sort and unique
            # NOTE: np.unique sorts a copy; do NOT sort `axis` in place - it is a
            # view into elm_centers and mutating it scrambles the element-to-row
            # correspondence with aperture/pressure arrays for later consumers.
            axis_clean_pos = np.unique(axis)
            axis_clean_neg = axis_clean_pos[::-1]
            # print(axis.shape, axis_clean.shape)
        
        half_step_pos = (axis_clean_pos[-1] - axis_clean_pos[-2])/2
        frac_length_pos = axis_clean_pos[-1] + half_step_pos

    

        half_step_neg = (axis_clean_neg[-1] - axis_clean_neg[-2])/2
        frac_length_neg = axis_clean_neg[-1] + half_step_neg

        length_arr_pos.append(frac_length_pos)
        length_arr_neg.append(frac_length_neg)

    if get_negative:
        return np.array(length_arr_pos, dtype=np.float64), np.array(length_arr_neg, dtype=np.float64)

    return np.array(length_arr_pos, dtype=np.float64)



def get_frac_length_from_path(file_path: str, selected_axis: str, method: int=0, return_attributes: bool=False, get_negative: bool=False, selected_axis_negative: str=None) -> tuple[npt.NDArray[ np.float64 ], Optional[dict]]:
    """
    Get fracture length from an HDF5 file.

    Parameters:
    -----------
        file_path (str): The path to the HDF5 file.
        selected_axis (str): The axis along which to calculate the length.
        method (int, optional): The method to use for calculation [0, 1]. Defaults to 0.
        return_attributes (bool, optional): Whether to return the attributes of the dataset, only [elementCenter, time]. Defaults to False.
        get_negative (bool, optional): Whether to include negative values in the calculation. Defaults to False.
        selected_axis_negative (str, optional): The axis along which to calculate the negative length. Defaults to None.
    Returns:
    -----------
        npt.NDArray[np.float64]: An array containing the calculated lengths.
        dict: A dictionary containing the attributes of the dataset, if `return_attributes` is True.
    """
    data, _ = read_hdf5(file_path)
    atrb = get_attributes(data, attributes_list=["pressure elementCenter", "pressure Time"])
    
    if len(atrb.keys()) < 2:
        print(f"Target attributes not found in dataset.")
        return None

    frac_length = get_frac_length(atrb["elementCenter"], selected_axis, method)

    if get_negative:
        frac_pos, frac_neg = get_frac_length(atrb["elementCenter"], selected_axis_negative, method, get_negative=True)
        frac_length = (frac_length, frac_pos, frac_neg)

    if return_attributes:
        return frac_length, atrb
    return frac_length

def get_strain_volumes(strain_arr: npt.NDArray[np.float64], elm_centers: npt.NDArray[np.float64], time_indices: list[int]=None, comps=COMPONENTS, show_msgs: bool=False, as_volume: bool=False) -> dict[str, dict[int, npt.NDArray[np.float64]]]:
    """
    Get strain volumes from strain data and element centers.

    Parameters:
    -----------
        strain_arr (npt.NDArray[np.float64]): An array of strain data.
        elm_centers (npt.NDArray[np.float64]): An array of element center coordinates.
        time_indices (list[int], optional): A list of indices for which to calculate strain volumes. Defaults to all indices.
        comps (list[str]): A list of components for which to calculate strain volumes.
        show_msgs (bool, optional): Whether to display messages during calculation. Defaults to False.
        as_volume (bool, optional): Whether to return the strain data as volumes (3D arrays) instead of flattened arrays. Defaults to False.


    Returns:
    -----------
        dict[int, dict[str, npt.NDArray[np.float64]]]: A dictionary containing the calculated strain volumes for each component.
    """

    def _build_strain_volume(strain_data, t, comp, ix, iy, iz, nx, ny, nz, fill=np.nan):
        vol = np.full((nx, ny, nz), fill, dtype=float)
        vol[ix, iy, iz] = strain_data[t, :, comp]
        return vol
    
    nt, npts, ncomp = strain_arr.shape

    if time_indices is None:
        time_indices = list(range(nt))
    else:
        if max(time_indices) >= nt or min(time_indices) < 0:
            print(f"Provided time steps {time_indices} are out of bounds for strain array with {nt} time steps.")
            return None

    if ncomp < len(comps):
        print(f"Number of components in strain array ({ncomp}) is less than the number of components provided ({len(comps)}).")
        return None

    if set(comps).issubset(COMPONENTS) == False:
        print(f"Provided components {comps} are not all in the list of valid components {COMPONENTS}.")
        return None

    print("Input strain array shape:", strain_arr.shape) if show_msgs else None

    # Coordinates are usually static across time; use a reference step for indexing.
    t_ref = 0
    x = elm_centers[t_ref, :, 0]
    y = elm_centers[t_ref, :, 1]
    z = elm_centers[t_ref, :, 2]

    # Round coordinates to suppress floating-point jitter before unique/searchsorted.
    decimals = 10
    xr = np.round(x, decimals=decimals)
    yr = np.round(y, decimals=decimals)
    zr = np.round(z, decimals=decimals)

    x_vals = np.unique(xr)
    y_vals = np.unique(yr)
    z_vals = np.unique(zr)
    nx, ny, nz = len(x_vals), len(y_vals), len(z_vals)
    print("recovered grid:", (nx, ny, nz), "product:", nx * ny * nz, "npts:", npts) if show_msgs else None

    ix = np.searchsorted(x_vals, xr)
    iy = np.searchsorted(y_vals, yr)
    iz = np.searchsorted(z_vals, zr)

    # Validate one-to-one occupancy of Cartesian cells.
    flat_idx = (ix * ny + iy) * nz + iz
    unique_cells = np.unique(flat_idx).size
    print("unique mapped cells:", unique_cells, "duplicates:", npts - unique_cells) if show_msgs else None

    # Optional drift check: element centers should not move for this mapping strategy.
    drift = np.max(np.abs(elm_centers[-1] - elm_centers[0]))
    print("max coordinate drift across time:", drift) if show_msgs else None

    ret_dict = {}


    for idx in time_indices:
        ret_dict[idx] = {}
        for comp in comps:
            comp_idx = COMPONENTS.index(comp)
            ret_dict[idx][comp] = _build_strain_volume(strain_arr, idx, comp_idx, ix, iy, iz, nx, ny, nz)

    if as_volume:
        if set(comps).issubset(["xx", "yy", "zz"]):
            _vol = np.zeros((len(time_indices), nx, ny, nz, len(comps)), dtype=np.float64)
            for _idx, _comps_dict in ret_dict.items():
                for _comp, _data in _comps_dict.items():
                    _comp_idx = COMPONENTS.index(_comp)
                    _vol[_idx, :, :, :, _comp_idx] = _data
            
            return np.moveaxis(_vol, -1, 1), (x_vals, y_vals, z_vals)  # Move component axis to second position for (time, component, x, y, z)
        else:
            print(f"Volume output is only supported for components ['xx', 'yy', 'zz']. Provided components: {comps}. Returning dict of arrays instead.")               

    return ret_dict

def plot_to_axis(fig: plt.Figure, ax: plt.Axes, vol: np.ndarray, time_idx: int, axis_idx: int, slice_idx: int, cmap: str="seismic", extent: Optional[list[float]]=None, clim: Optional[np.float64]=None, title: Optional[str]=None, xlabel: Optional[str]=None, ylabel: Optional[str]=None) -> None:
    """
    Plot a 2D slice of a 3D volume to a given Matplotlib axis.

    Parameters:
    -----------
        fig (plt.Figure): The Matplotlib figure to plot on.
        ax (plt.Axes): The Matplotlib axis to plot on.
        vol (np.ndarray): The 3D volume to plot.
        time_idx (int): The index of the time step to plot.
        axis_idx (int): The index of the axis along which to to plot using slice_idx.
        slice_idx (int): The index of the slice to plot.
        cmap (str, optional): The colormap to use for plotting. Defaults to "seismic".
        extent (list[float], optional): The extent of the axes in the format [xmin, xmax, ymin, ymax]. Defaults to None.
        clim (float, optional): The color limits for the plot. Defaults to None.
        title (str, optional): The title of the plot. Defaults to None.
        xlabel (str, optional): The label for the x-axis. Defaults to None.
        ylabel (str, optional): The label for the y-axis. Defaults to None.
    Returns:
    -----------
        None
    """
    # print(f"Input volume shape: {vol.shape}, time index: {time_idx}, axis index: {axis_idx}, slice index: {slice_idx}")
    if len(vol.shape) != 5:
        print(f"Input volume must be a 5D array with shape (time, component, x, y, z). Provided shape: {vol.shape}.")
        return None
    if time_idx >= vol.shape[0] or (time_idx < 0 and time_idx != -1):
        print(f"Invalid time index {time_idx} for volume with {vol.shape[0]} time steps.")
        return None
    if axis_idx < 0 or axis_idx > 2:
        print(f"Invalid axis index {axis_idx}. Valid options are 0 (x), 1 (y), or 2 (z).")
        return None

    data_lambda = {
        0: lambda i: vol[time_idx, axis_idx, i, :, :],
        1: lambda i: vol[time_idx, axis_idx, :, i, :],
        2: lambda i: vol[time_idx, axis_idx, :, :, i]
    }
    vol_limits = {
        0: vol.shape[-3],
        1: vol.shape[-2],
        2: vol.shape[-1]
    }
    
    data = data_lambda[axis_idx](slice_idx)

    if clim is None:
        clim = np.abs(data).max()
    
    im = ax.imshow(
        data.T,
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=-clim,
        vmax=clim,
        cmap=cmap
    )
    
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    fig.colorbar(im, ax=ax)

    return data

def plot_strain_slice(vol: np.ndarray, t_idx: int, a_idx: int, s_idx: int, x_vals: np.ndarray, y_vals: np.ndarray, z_vals: np.ndarray, time_dict: dict, figsize: tuple[int, int]=(8, 6), frac_length: float=-1, fig_prev: Optional[plt.Figure]=None, ax_prev: Optional[plt.Axes]=None) -> None:
    """
    Plot a waterfall plot of strain data.

    Parameters:
    -----------
        vol (np.ndarray): The 3D volume of strain data to plot.
        t_idx (int): The index of the time step to plot.
        a_idx (int): The index of the axis along which to to plot using slice_idx.
        s_idx (int): The index of the slice to plot.
        x_vals (np.ndarray): The unique x coordinates corresponding to the volume data.
        y_vals (np.ndarray): The unique y coordinates corresponding to the volume data.
        z_vals (np.ndarray): The unique z coordinates corresponding to the volume data.
        time_dict (dict): A dictionary mapping time indices to actual time values.
        figsize (tuple[int, int], optional): The size of the figure to create. Defaults to (8, 6).
        frac_length (float, optional): If greater than 0, a vertical line will be plotted at this x-value to indicate fracture length. Defaults to -1 (no line).
        fig_prev (plt.Figure, optional): A previous Matplotlib figure to plot on. Defaults to None (create new figure).
        ax_prev (plt.Axes, optional): A previous Matplotlib axes to plot on. Defaults to None (create new axes).
    Returns:
    -----------
        None
    """
    fig_labels_dict = {
        0: ("XY", "x (m)", "y (m)", [x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()]),
        1: ("XZ", "x (m)", "z (m)", [x_vals.min(), x_vals.max(), z_vals.min(), z_vals.max()]),
        2: ("YZ", "y (m)", "z (m)", [y_vals.min(), y_vals.max(), z_vals.min(), z_vals.max()])
    }

    actual_time = float(time_dict[t_idx][0])
    actual_offset = {0: x_vals, 1: y_vals, 2: z_vals}[a_idx][s_idx]

    if fig_prev is None or ax_prev is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = fig_prev
        ax = ax_prev

    panel, x_label, y_label, extent = fig_labels_dict[a_idx]
    title = f"{panel}: {rf'$\epsilon_{{{["xx", "yy", "zz"][a_idx]}}}$'}, {actual_time:.2f} s, {actual_offset:.2f} m"

    data = plot_to_axis(
        fig=fig,
        ax=ax,
        vol=vol,
        time_idx=t_idx,
        axis_idx=a_idx,
        slice_idx=s_idx,
        extent=extent,
        title=title,
        xlabel=x_label,
        ylabel=y_label,
        clim=None
    )
    if frac_length > 0:
        ax.axvline(x=frac_length, color="k", linestyle="--", label=f"Fracture Length {frac_length:.2f} m")
        ax.legend(loc="best")

    return data, fig, ax

def get_frac_aperture(elm_centers: npt.NDArray[np.float64], aperature_arr: npt.NDArray[np.float64], axis_1: str, axis_2: str, time_indices: list[int]=None) -> npt.NDArray[np.float64]:
    """
    Get fracture aperture from inputs.

    Parameters:
    -----------
        elm_centers (npt.NDArray[np.float64]): An array containing the element centers.
        aperature_arr (npt.NDArray[np.float64]): An array containing the aperture values.
        axis_1 (str): The first axis along which to calculate the aperture.
        axis_2 (str): The second axis along which to calculate the aperture.
        time_indices (list[int], optional): A list of time indices for which to calculate the aperture. Defaults to None.
    Returns:
    -----------
        npt.NDArray[np.float64]: An array containing the calculated apertures.
    """
    nt = aperature_arr.shape[0]
    if time_indices is None:
        time_indices = list(range(nt))

    data_lambda = {
        "x": lambda centers: centers[:, 0],
        "y": lambda centers: centers[:, 1],
        "z": lambda centers: centers[:, 2]
    }

    ret_dict = {}
    
    for t in time_indices:
        d1 = data_lambda[axis_1](elm_centers[t])
        d2 = data_lambda[axis_2](elm_centers[t])
        aper = aperature_arr[t]

        idx_skips = np.argwhere(d1 == 0.0).flatten()
        
        d1_nonzero = [_ for i, _ in enumerate(d1) if i not in idx_skips]
        d2_nonzero = [_ for i, _ in enumerate(d2) if i not in idx_skips]

        aper_nonzero = aper[aper != 0.0]

        ret_dict[t] = {
            axis_1: d1_nonzero,
            axis_2: d2_nonzero,
            "aperture": aper_nonzero
        }

    return ret_dict

def get_open_frac_extent(frac_centers: npt.NDArray[np.float64], frac_aperture: npt.NDArray[np.float64], selected_axis: str, time_index: int=-1, aperture_threshold: float=1e-4) -> float:
    """
    Max coordinate along an axis of fracture elements that are actually OPEN
    (aperture above threshold), at one time frame.

    Unlike get_frac_length, which measures the extent of all split elements,
    this measures the hydraulically opened footprint.  The two can differ a
    lot: split-but-closed elements carry zero aperture, so a fiber placed by
    split length may cross closed fracture and see no jump.

    Parameters:
    -----------
        frac_centers (npt.NDArray[np.float64]): Fracture element centers, shape (nt, n_alloc, 3).
        frac_aperture (npt.NDArray[np.float64]): Apertures, shape (nt, n_frac).
        selected_axis (str): Axis to measure the extent along ("x", "y" or "z").
        time_index (int, optional): Frame to evaluate. Defaults to -1 (last).
        aperture_threshold (float, optional): Elements above this count as open. Defaults to 1e-4.

    Returns:
    -----------
        float: Max coordinate of open elements along the axis, or np.nan if none are open.
    """
    axis_dict = {"x": 0, "y": 1, "z": 2}
    n_frac = frac_aperture.shape[1]
    centers = frac_centers[time_index][:n_frac]
    open_mask = frac_aperture[time_index] > aperture_threshold
    if not open_mask.any():
        print("No open fracture elements at the requested frame.")
        return np.nan
    return float(centers[open_mask, axis_dict[selected_axis]].max())

def get_fiber_waterfall(strain_vol: npt.NDArray[np.float64], coords: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]], strain_time: npt.NDArray[np.float64], frac_centers: npt.NDArray[np.float64], frac_aperture: npt.NDArray[np.float64], frac_time: npt.NDArray[np.float64], fiber_position: tuple[float, float], normal_axis: str="y", plane_coord: float=0.0, aperture_threshold: float=1e-5, search_radius: float=1.0, gauge_length: Optional[float]=None, compute_rate: bool=True) -> dict:
    """
    Extract a DSS-style fiber strain waterfall along the fracture-normal axis,
    including the displacement jump (aperture) where the fiber crosses the
    fracture plane.

    GEOS `averageStrain` is solid-element strain only: the fracture is a
    zero-thickness interface, so the opening jump across it belongs to no
    element.  A physical fiber crossing the fracture measures
    aperture / gauge_length there (mm over ~1 m -> 1e3 microstrain), which
    dominates the waterfall.  This function adds that term to the two rows of
    elements straddling the fracture plane whenever an open fracture element
    lies within `search_radius` of the fiber location.

    Parameters:
    -----------
        strain_vol (npt.NDArray[np.float64]): 5D strain volume with shape
            (time, component, x, y, z) from get_strain_volumes(as_volume=True).
        coords (tuple): (x_vals, y_vals, z_vals) unique coordinates from
            get_strain_volumes(as_volume=True).
        strain_time (npt.NDArray[np.float64]): Times of the strain frames, shape (nt,) or (nt, 1).
        frac_centers (npt.NDArray[np.float64]): Fracture element centers from the
            timeHistory file, shape (nt_frac, n_alloc, 3).
        frac_aperture (npt.NDArray[np.float64]): Fracture apertures, shape (nt_frac, n_frac).
            (n_alloc may exceed n_frac; extra buffer rows are ignored.)
        frac_time (npt.NDArray[np.float64]): Times of the fracture frames, shape (nt_frac,) or (nt_frac, 1).
        fiber_position (tuple[float, float]): In-plane coordinates of the fiber.
            For normal_axis "y" this is (x, z); for "x" it is (y, z); for "z" it is (x, y).
        normal_axis (str, optional): Fracture-normal axis the fiber runs along
            ("x", "y" or "z"); also selects the strain component (xx, yy or zz). Defaults to "y".
        plane_coord (float, optional): Coordinate of the fracture plane on the normal axis. Defaults to 0.0.
        aperture_threshold (float, optional): Apertures above this count as open fracture. Defaults to 1e-5.
        search_radius (float, optional): Max in-plane distance (m) between the fiber and the
            nearest open fracture element for the crossing to count as intercepted. Defaults to 1.0.
        gauge_length (float, optional): Fiber gauge length (m) over which the jump is averaged.
            Defaults to the distance between the two element-center rows straddling the plane.
        compute_rate (bool, optional): Also return time derivatives (strain rate). Defaults to True.

    Returns:
    -----------
        dict: {
            "time": (nt,) strain frame times,
            "offsets": (n_axis,) fiber coordinates along the normal axis,
            "solid": (nt, n_axis) solid-only waterfall (what averageStrain gives),
            "corrected": (nt, n_axis) waterfall including the aperture jump,
            "aperture": (nt,) aperture at the fiber crossing (0 before interception),
            "crossing_rows": indices of the two rows straddling the plane,
            "gauge_length": gauge length used,
            "solid_rate", "corrected_rate": (nt, n_axis) d(eps)/dt, if compute_rate,
        }
    """
    axis_dict = {"x": 0, "y": 1, "z": 2}
    if normal_axis not in axis_dict:
        print(f"Invalid normal_axis '{normal_axis}'. Valid options: {list(axis_dict)}.")
        return None

    a_idx = axis_dict[normal_axis]
    plane_axes = [i for i in range(3) if i != a_idx]  # in-plane coordinate indices

    strain_time = np.asarray(strain_time).flatten()
    frac_time = np.asarray(frac_time).flatten()
    offsets = coords[a_idx]

    # fiber = line along the normal axis at the fixed in-plane position
    fixed_idx = [np.argmin(np.abs(coords[ax] - pos)) for ax, pos in zip(plane_axes, fiber_position)]
    actual_pos = [float(coords[ax][i]) for ax, i in zip(plane_axes, fixed_idx)]
    if any(abs(a - p) > 1e-6 for a, p in zip(actual_pos, fiber_position)):
        print(f"Fiber snapped to nearest element centers: requested {fiber_position}, using {tuple(actual_pos)}.")

    slicer = [slice(None), a_idx, slice(None), slice(None), slice(None)]
    for ax, i in zip(plane_axes, fixed_idx):
        slicer[2 + ax] = i
    solid = strain_vol[tuple(slicer)]  # (nt, n_axis)

    # rows straddling the fracture plane and the gauge over which the jump is averaged
    below = np.where(offsets < plane_coord)[0]
    above = np.where(offsets > plane_coord)[0]
    if len(below) == 0 or len(above) == 0:
        print(f"Fracture plane at {normal_axis}={plane_coord} is outside the fiber extent; returning solid strain only.")
        crossing_rows = []
    else:
        crossing_rows = [below[-1], above[0]]
    if gauge_length is None and crossing_rows:
        gauge_length = float(offsets[crossing_rows[1]] - offsets[crossing_rows[0]])

    # aperture at the fiber crossing, matched to the strain frames by nearest time
    n_frac = frac_aperture.shape[1]
    aperture = np.zeros(len(strain_time))
    for k, tk in enumerate(strain_time):
        kf = int(np.argmin(np.abs(frac_time - tk)))
        if abs(frac_time[kf] - tk) > 1.0:
            print(f"Warning: nearest fracture frame ({frac_time[kf]:.1f} s) is far from strain frame ({tk:.1f} s).")
        centers = frac_centers[kf][:n_frac]
        aper = frac_aperture[kf]
        open_mask = aper > aperture_threshold
        if not open_mask.any():
            continue
        dist = np.hypot(centers[open_mask, plane_axes[0]] - actual_pos[0],
                        centers[open_mask, plane_axes[1]] - actual_pos[1])
        if dist.min() <= search_radius:
            aperture[k] = aper[open_mask][np.argmin(dist)]

    corrected = solid.copy()
    if crossing_rows:
        for j in crossing_rows:
            corrected[:, j] += aperture / gauge_length

    ret_dict = {
        "time": strain_time,
        "offsets": offsets,
        "solid": solid,
        "corrected": corrected,
        "aperture": aperture,
        "crossing_rows": crossing_rows,
        "gauge_length": gauge_length,
        "fiber_position": tuple(actual_pos),
        "normal_axis": normal_axis
    }

    if compute_rate:
        if len(strain_time) < 2:
            print("Fewer than 2 time frames; skipping strain-rate computation.")
        else:
            ret_dict["solid_rate"] = np.gradient(solid, strain_time, axis=0)
            ret_dict["corrected_rate"] = np.gradient(corrected, strain_time, axis=0)

    return ret_dict

def get_displacement_fiber_waterfall(node_positions: npt.NDArray[np.float64], node_displacements: npt.NDArray[np.float64], disp_time: npt.NDArray[np.float64], fiber_position: tuple[float, float], normal_axis: str="y", plane_coord: float=0.0, column_tol: float=1e-3, plane_eps: float=1e-6, compute_rate: bool=True) -> dict:
    """
    Compute a DSS-style fiber strain waterfall directly from nodal
    totalDisplacement: eps = du_normal / d(offset) between consecutive nodes
    along the fiber.  Because displacement is differenced ACROSS the fracture
    plane, the opening jump is captured exactly - no aperture approximation.

    Nodes lying on the fracture plane are excluded (after the
    SurfaceGenerator splits them they are duplicated with +/- side values),
    so the segment spanning the plane runs from the last node below to the
    first node above it and contains the full jump.

    Parameters:
    -----------
        node_positions (npt.NDArray[np.float64]): Node reference positions from the
            displacement time history, shape (nt, n_alloc, 3).
        node_displacements (npt.NDArray[np.float64]): totalDisplacement, shape (nt, n_alloc, 3).
        disp_time (npt.NDArray[np.float64]): Frame times, shape (nt,) or (nt, 1).
        fiber_position (tuple[float, float]): In-plane coordinates of the fiber
            (for normal_axis "y": (x, z)).  Snapped to the nearest node column.
        normal_axis (str, optional): Fracture-normal axis the fiber runs along
            ("x", "y" or "z"); also selects the displacement component. Defaults to "y".
        plane_coord (float, optional): Coordinate of the fracture plane on the normal axis. Defaults to 0.0.
        column_tol (float, optional): Tolerance (m) for selecting nodes of the fiber column. Defaults to 1e-3.
        plane_eps (float, optional): Nodes within this distance of the plane are excluded. Defaults to 1e-6.
        compute_rate (bool, optional): Also return the time derivative. Defaults to True.

    Returns:
    -----------
        dict: {
            "time": (nt,) frame times,
            "offsets": (n_seg,) segment midpoints along the normal axis,
            "fiber": (nt, n_seg) gauge strain per segment,
            "gauge_lengths": (n_seg,) segment lengths,
            "crossing_row": index of the segment spanning the fracture plane,
            "fiber_position": snapped in-plane coordinates,
            "fiber_rate": (nt, n_seg) d(eps)/dt, if compute_rate,
        }
    """
    axis_dict = {"x": 0, "y": 1, "z": 2}
    if normal_axis not in axis_dict:
        print(f"Invalid normal_axis '{normal_axis}'. Valid options: {list(axis_dict)}.")
        return None

    a_idx = axis_dict[normal_axis]
    plane_axes = [i for i in range(3) if i != a_idx]
    disp_time = np.asarray(disp_time).flatten()
    nt = node_displacements.shape[0]

    # snap to the nearest node column (frame 0; padding rows sit at the origin
    # and only matter for a fiber requested at the exact domain corner)
    p0 = node_positions[0]
    d_inplane = np.hypot(p0[:, plane_axes[0]] - fiber_position[0], p0[:, plane_axes[1]] - fiber_position[1])
    nearest = p0[np.argmin(d_inplane)]
    actual_pos = (float(nearest[plane_axes[0]]), float(nearest[plane_axes[1]]))
    if any(abs(a - p) > 1e-6 for a, p in zip(actual_pos, fiber_position)):
        print(f"Fiber snapped to nearest node column: requested {fiber_position}, using {actual_pos}.")

    fiber = None
    offsets = None
    for k in range(nt):
        pos = node_positions[k]
        col = (np.abs(pos[:, plane_axes[0]] - actual_pos[0]) < column_tol) & \
              (np.abs(pos[:, plane_axes[1]] - actual_pos[1]) < column_tol) & \
              (np.abs(pos[:, a_idx] - plane_coord) > plane_eps)
        y = pos[col, a_idx]
        u = node_displacements[k][col, a_idx]
        order = np.argsort(y)
        y, u = y[order], u[order]
        if fiber is None:
            offsets = 0.5*(y[1:] + y[:-1])
            gauges = np.diff(y)
            fiber = np.full((nt, len(offsets)), np.nan)
        if len(y) != len(offsets) + 1:
            print(f"Frame {k}: unexpected node count along the fiber ({len(y)} vs {len(offsets)+1}); skipping frame.")
            continue
        fiber[k] = np.diff(u) / gauges

    below = np.where(offsets < plane_coord)[0]
    above = np.where(offsets > plane_coord)[0]
    crossing_row = None
    if len(below) and len(above):
        # the crossing segment is the one whose endpoints straddle the plane
        crossing_row = below[-1] if offsets[below[-1]] + 0.5*gauges[below[-1]] > plane_coord else below[-1] + 1

    ret_dict = {
        "time": disp_time,
        "offsets": offsets,
        "fiber": fiber,
        "gauge_lengths": gauges,
        "crossing_row": crossing_row,
        "fiber_position": actual_pos,
        "normal_axis": normal_axis
    }

    if compute_rate:
        if nt < 2:
            print("Fewer than 2 time frames; skipping strain-rate computation.")
        else:
            ret_dict["fiber_rate"] = np.gradient(fiber, disp_time, axis=0)

    return ret_dict

def get_displacement_fiber_waterfall_from_path(displacement_file_path: str, fiber_position: tuple[float, float], normal_axis: str="y", **kwargs) -> dict:
    """
    Convenience wrapper around get_displacement_fiber_waterfall that loads the
    nodal displacement time-history HDF5 of a run
    (heterogeneousInSitu_timeHistory_displacement.hdf5).

    Parameters:
    -----------
        displacement_file_path (str): Path to the displacement time-history file.
        fiber_position (tuple[float, float]): In-plane coordinates of the fiber.
        normal_axis (str, optional): Fracture-normal axis the fiber runs along. Defaults to "y".
        **kwargs: Forwarded to get_displacement_fiber_waterfall.

    Returns:
    -----------
        dict: See get_displacement_fiber_waterfall.
    """
    loaded = read_displacement_hdf5(displacement_file_path)
    if loaded is None:
        return None
    disp_time, node_positions, node_displacements = loaded

    return get_displacement_fiber_waterfall(
        node_positions=node_positions,
        node_displacements=node_displacements,
        disp_time=disp_time,
        fiber_position=fiber_position,
        normal_axis=normal_axis,
        **kwargs
    )

def read_displacement_hdf5(displacement_file_path: str) -> Optional[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
    """
    Load a nodal totalDisplacement TimeHistory HDF5 and return
    (time [nt], positions [nt, nn, 3], displacements [nt, nn, 3]).

    Handles both naming variants GEOS produces:
      - whole-nodeManager collections: "totalDisplacement",
        "totalDisplacement ReferencePosition", "totalDisplacement Time"
      - setNames-restricted collections (e.g. a fiber nodeset): the set name
        is appended, "totalDisplacement fiber",
        "totalDisplacement ReferencePosition fiber" - which endswith()-style
        matching misses.

    Parameters:
    -----------
        displacement_file_path (str): Path to the displacement time-history file.

    Returns:
    -----------
        tuple or None: (time, positions, displacements); None if the datasets
        could not be identified.
    """
    ds, _ = read_hdf5(displacement_file_path)
    keys = ds.keys()
    field_key = next((k for k in keys if "totalDisplacement" in k
                      and "ReferencePosition" not in k and "Time" not in k), None)
    pos_key = next((k for k in keys if "ReferencePosition" in k), None)
    time_key = next((k for k in keys if "Time" in k), None)
    if not all((field_key, time_key, pos_key)):
        print(f"Displacement datasets not found. Available keys: {keys}")
        return None
    return np.asarray(ds[time_key]).ravel(), np.asarray(ds[pos_key]), np.asarray(ds[field_key])

def get_fiber_waterfall_from_path(strain_file_path: str, history_file_path: str, fiber_position: tuple[float, float], normal_axis: str="y", **kwargs) -> dict:
    """
    Convenience wrapper around get_fiber_waterfall that loads everything from
    the two timeHistory HDF5 files of a run.

    Parameters:
    -----------
        strain_file_path (str): Path to heterogeneousInSitu_timeHistory_strain.hdf5.
        history_file_path (str): Path to heterogeneousInSitu_timeHistory.hdf5
            (source of fracture element centers and apertures).
        fiber_position (tuple[float, float]): In-plane coordinates of the fiber
            (see get_fiber_waterfall).
        normal_axis (str, optional): Fracture-normal axis the fiber runs along. Defaults to "y".
        **kwargs: Forwarded to get_fiber_waterfall (plane_coord, search_radius, gauge_length, ...).

    Returns:
    -----------
        dict: See get_fiber_waterfall.
    """
    strain_ds, _ = read_hdf5(strain_file_path)
    strain_atrb = get_attributes(strain_ds, attributes_list=["averageStrain", "averageStrain Time", "averageStrain elementCenter"])
    if len(strain_atrb.keys()) < 3:
        print("Target strain attributes not found in dataset.")
        return None

    vol, coords = get_strain_volumes(
        strain_atrb["averageStrain"],
        strain_atrb["averageStrain elementCenter"],
        comps=["xx", "yy", "zz"],
        as_volume=True
    )

    history_ds, _ = read_hdf5(history_file_path)
    hist_atrb = get_attributes(history_ds, attributes_list=["elementAperture", "elementAperture Time", "elementAperture elementCenter"], add_renamed=False)
    if len(hist_atrb.keys()) < 3:
        print("Target aperture attributes not found in dataset.")
        return None

    return get_fiber_waterfall(
        strain_vol=vol,
        coords=coords,
        strain_time=strain_atrb["averageStrain Time"],
        frac_centers=hist_atrb["elementAperture elementCenter"],
        frac_aperture=hist_atrb["elementAperture"],
        frac_time=hist_atrb["elementAperture Time"],
        fiber_position=fiber_position,
        normal_axis=normal_axis,
        **kwargs
    )








    