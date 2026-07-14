"""Plotting and loading helpers for timeSeries_viewer_compare.ipynb.

Functions here were moved out of the notebook so it holds only configuration
and calls.  Grouping:

  Attribute plots        plot_attributes, plot_pressure_vs_length
  Waterfall plots        cell_edges, strain_rate, plot_waterfall_pair,
                         plot_fiber_panels, plot_fiber_vs_displacement,
                         plot_displacement_fiber_grid
  Displacement loading   read_displacement_pvd, load_case_displacement
                         (HDF5 loading itself lives in hdf5_tools)

Conventions shared with hdf5_tools:
  - fracture time-history arrays are (nt, n_alloc); rows past the current
    element count are zero padding (the buffer grows as the fracture does)
  - waterfall arrays are (nt, n_offsets), plotted transposed with pcolormesh
    on TRUE cell edges (imshow+extent distorts graded meshes)
"""

import numpy as np
import matplotlib.pyplot as plt

import hdf5_tools as hdt

try:  # display() is only available under IPython; fall back for headless use
    from IPython.display import display
except ImportError:  # pragma: no cover
    def display(*_args, **_kwargs):
        pass


# ============================================================================
# small utilities
# ============================================================================

def cell_edges(v):
    """Cell-edge positions for pcolormesh from an array of cell centers.

    Works for non-uniform spacing (graded meshes): each interior edge is the
    midpoint between neighboring centers; the two outer edges extrapolate by
    half the first/last spacing.
    """
    v = np.asarray(v, dtype=float)
    return np.concatenate([[v[0] - (v[1] - v[0])/2],
                           (v[1:] + v[:-1])/2,
                           [v[-1] + (v[-1] - v[-2])/2]])


def strain_rate(waterfall, time, smooth_frames=1):
    """Time derivative of a (nt, n_offsets) waterfall via central differences.

    `time` may be in any unit; the rate comes back per that unit.

    Parameters:
    -----------
        waterfall (ndarray): (nt, n_offsets) strain array.
        time (ndarray): Frame times (may be non-uniformly spaced).
        smooth_frames (int, optional): Width (in frames) of a centered
            moving-average applied to the strain BEFORE differentiating.
            Suppresses the vertical striping that per-frame differencing
            produces from discrete propagation/contact events and non-uniform
            frame spacing, without touching minute-scale morphology (heart,
            lobes).  1 (default) = no smoothing; 3-5 is usually enough.
            Edge frames use a partial window (no attenuation).

    Returns:
    -----------
        ndarray: (nt, n_offsets) strain rate.
    """
    waterfall = np.asarray(waterfall, float)
    time = np.asarray(time, float).ravel()

    if smooth_frames and smooth_frames > 1:
        k = int(smooth_frames)
        kernel = np.ones(k)
        # normalize by the actual window size at the edges (partial windows)
        norm = np.convolve(np.ones(waterfall.shape[0]), kernel, mode="same")
        waterfall = np.apply_along_axis(
            lambda col: np.convolve(col, kernel, mode="same") / norm, 0, waterfall)

    return np.gradient(waterfall, time, axis=0)


# ============================================================================
# attribute plots (fracture shape / length / height / aperture / pressure)
# ============================================================================

def plot_attributes(
    atrb_dict,
    rename_dict,
    frac_mesh=None,
    prints=False,
    frac_domain=["x", "z"],
    pad=0.05,
    zoom=1.0,
    aperture_tol=1e-9,
    legends=True,
    style_dict=None
):
    """Overview plots for a group of runs from their fracture time histories.

    Produces five figures - fracture shape (final footprint), length(t),
    height(t), max aperture(t), and pressure(t) - with every run in
    `rename_dict` overlaid.

    Parameters:
    -----------
        atrb_dict (dict): {run: attributes} from hdf5_tools.get_attributes;
            each entry needs elementCenter, elementAperture, elementArea,
            pressure and Time.
        rename_dict (dict): {run: display label}; also selects WHICH runs are
            plotted (keys must match atrb_dict).
        frac_mesh (tuple, optional): (fig, ax) to draw the fracture shape on
            (e.g. from mesh_tools.plot_mesh_2d); (None, None) makes a new one.
        prints (bool, optional): Verbose shape printouts. Defaults to False.
        frac_domain (list, optional): The two in-plane axes of the fracture,
            e.g. ["x", "z"] for a y-normal fracture. Defaults to ["x", "z"].
        pad (float, optional): Relative padding of the shape-plot limits.
        zoom (float, optional): Zoom factor for the shape plot. Defaults to 1.
        aperture_tol (float, optional): Elements with aperture above this are
            counted as part of the fracture shape. Defaults to 1e-9.
        legends (bool, optional): Show legends. Defaults to True.
        style_dict (dict, optional): {run: style} per-run overrides so runs
            stay distinguishable when curves overlap.  Recognized keys:
              "color"           - curve/footprint color (default: tab10 by index)
              "linestyle"       - time-series linestyle (default "-")
              "marker"          - time-series marker (default "o")
              "markerfacecolor" - e.g. "none" for open markers
              "shape"           - footprint rendering in the shape plot:
                                  "fill" (default, translucent squares) or
                                  "outline" (boundary polygon only - use for
                                  pre-existing fractures so their large
                                  footprint does not cover the others)
            Runs not listed fall back to the defaults.  Recommended encoding:
            a unique color+marker per run, linestyle grouping a second factor
            (e.g. benchmark solid, HFTS-2 dashed).

    Returns:
    -----------
        tuple: ((fig, ax) x 5, ret_dict) where ret_dict[run] holds the
        processed series (time, pressure, fracture_length, heights,
        aperture_arr, ...) for downstream cells.
    """
    assert rename_dict.keys() == atrb_dict.keys(), "Rename dict keys must match attribute dict keys"
    ret_dict = {}

    # fracture shape plot (optionally on top of a mesh drawing)
    fig1, ax1 = plt.subplots(figsize=(10, 8)) if frac_mesh is None or frac_mesh[0] is None else frac_mesh
    ax1.set_title("Fracture shape")

    # Preserve any pre-existing legend (e.g., from mt.plot_mesh_2d)
    mesh_legend = ax1.get_legend()

    fig2, ax2 = plt.subplots(figsize=(10, 8))
    ax2.set_title("Fracture length")

    fig3, ax3 = plt.subplots(figsize=(10, 8))
    ax3.set_title("Fracture height")

    fig4, ax4 = plt.subplots(figsize=(10, 8))
    ax4.set_title("Fracture aperture")

    fig5, ax5 = plt.subplots(figsize=(10, 8))
    ax5.set_title("Fracture pressure")

    cmap = plt.get_cmap("tab10")

    fig1_xlim_min, fig1_xlim_max = np.inf, -np.inf
    fig1_ylim_min, fig1_ylim_max = np.inf, -np.inf

    shape_handles = []
    shape_labels = []

    for i, k in enumerate(rename_dict.keys()):
        name = rename_dict.get(k, k)
        _style = (style_dict or {}).get(k, {})
        color = _style.get("color", cmap(i))
        ls = _style.get("linestyle", "-")
        mk = _style.get("marker", "o")
        mfc = _style.get("markerfacecolor", None)   # "none" -> open markers
        shape_mode = _style.get("shape", "fill")    # footprint rendering

        elm_center = atrb_dict[k]["elementCenter"].copy()
        if prints:
            print("element center shape:")
            print(elm_center.shape)

        xcord = elm_center[-1, :, 0]
        ycord = elm_center[-1, :, 1]
        zcord = elm_center[-1, :, 2]

        ds = {
            "x": (xcord, "x (m)"),
            "y": (ycord, "y (m)"),
            "z": (zcord, "z (m)")
        }

        elm_aper = atrb_dict[k]["elementAperture"]

        # Build fracture-shape points from aperture bins and keep only active parts.
        aperture_dict = hdt.get_frac_aperture(elm_center, elm_aper, frac_domain[0], frac_domain[1])
        fracture_x = []
        fracture_y = []

        for val in aperture_dict.values():
            if np.max(val["aperture"]) > aperture_tol:
                fracture_x.extend(np.asarray(val[frac_domain[0]], dtype=float).tolist())
                fracture_y.extend(np.asarray(val[frac_domain[1]], dtype=float).tolist())

        if len(fracture_x) > 0:
            xvals = np.asarray(fracture_x)
            yvals = np.asarray(fracture_y)
        else:
            xvals = ds[frac_domain[0]][0]
            yvals = ds[frac_domain[1]][0]

        # Track fracture extents for final zoomed limits
        fig1_xlim_min = min(fig1_xlim_min, np.min(xvals))
        fig1_xlim_max = max(fig1_xlim_max, np.max(xvals))
        fig1_ylim_min = min(fig1_ylim_min, np.min(yvals))
        fig1_ylim_max = max(fig1_ylim_max, np.max(yvals))

        if shape_mode == "outline" and len(fracture_x) > 0:
            # Draw only the footprint BOUNDARY so this run does not cover the
            # runs drawn after it (used for the large pre-existing fractures).
            # The points are cell centers on a regular grid: for every row of
            # the vertical axis take the min/max of the horizontal axis, pad
            # by half a cell so the outline hugs the cell edges, and connect
            # right side (bottom->top) + left side (top->bottom) into a
            # closed polygon.
            uy = np.unique(yvals)
            ux = np.unique(xvals)
            hdx = 0.5 * np.median(np.diff(ux)) if len(ux) > 1 else 0.0
            hdy = 0.5 * np.median(np.diff(uy)) if len(uy) > 1 else 0.0
            xmin = np.array([xvals[yvals == v].min() for v in uy]) - hdx
            xmax = np.array([xvals[yvals == v].max() for v in uy]) + hdx
            ypad = uy.copy()
            ypad[0] -= hdy
            ypad[-1] += hdy
            bx = np.r_[xmax, xmin[::-1], xmax[:1]]
            by = np.r_[ypad, ypad[::-1], ypad[:1]]
            shape_line, = ax1.plot(bx, by, linestyle=_style.get("linestyle", "--"),
                                   linewidth=2.0, label=name, color=color)
        else:
            # Filled footprint: translucent squares at the opened cell centers.
            shape_line, = ax1.plot(xvals, yvals, "s", markersize=16, label=name,
                                   color=color, alpha=0.15)

        shape_handles.append(shape_line)
        shape_labels.append(name)
        ax1.set_xlabel(ds[frac_domain[0]][1])
        ax1.set_ylabel(ds[frac_domain[1]][1])

        time = atrb_dict[k]["Time"]
        pressure = atrb_dict[k]["pressure"]
        elm_area = atrb_dict[k]["elementArea"]

        frac_length = hdt.get_frac_length(elm_center, frac_domain[0], get_negative=False)

        ax2.plot(time, frac_length, marker=mk, markerfacecolor=mfc, linestyle=ls, label=f"{name} {frac_length[-1]:.2f} m", color=color)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Fracture length (m)")
        ax2.legend()

        frac_height_pos, frac_height_neg = hdt.get_frac_length(elm_center, frac_domain[1], get_negative=True)

        ax3.plot(time, frac_height_pos, marker=mk, markerfacecolor=mfc, linestyle=ls, label=f"{name} pos {frac_height_pos[-1]:.2f} m", color=color)
        ax3.plot(time, frac_height_neg, marker=mk, markerfacecolor=mfc, linestyle=ls, alpha=0.55, label=f"{name} neg {frac_height_neg[-1]:.2f} m", color=color)
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Fracture height (m)")
        ax3.legend()

        aperture_arr = []
        for idx, val in aperture_dict.items():
            aperture_arr.append(max(val["aperture"]) * 1e3)

        ax4.plot(time, aperture_arr, marker=mk, markerfacecolor=mfc, linestyle=ls, label=f"{name} {max(aperture_arr):.2f} mm", color=color)
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Fracture aperture (mm)")
        ax4.legend()

        pressure_MPa = pressure[:, 0] * 1e-6
        ax5.plot(time, pressure_MPa, marker=mk, markerfacecolor=mfc, linestyle=ls, label=f"{name} {max(pressure_MPa):.2f} MPa", color=color)
        ax5.set_xlabel("Time (s)")
        ax5.set_ylabel("Fracture pressure (MPa)")
        ax5.legend()

        ret_dict[k] = {
            "element_center": ds,
            "time": time,
            "pressure": pressure,
            "element_area": elm_area,
            "fracture_length": frac_length,
            "fracture_height_pos": frac_height_pos,
            "fracture_height_neg": frac_height_neg,
            "fracture_length_final": frac_length[-1],
            "aperture_arr": aperture_arr,
            "aperture_dict": aperture_dict
        }

    # Apply zoomed limits around fracture extents
    xspan = fig1_xlim_max - fig1_xlim_min
    yspan = fig1_ylim_max - fig1_ylim_min

    if xspan == 0:
        xspan = 1.0
    if yspan == 0:
        yspan = 1.0

    xpad = xspan * pad
    ypad = yspan * pad

    xcenter = 0.5 * (fig1_xlim_min + fig1_xlim_max)
    ycenter = 0.5 * (fig1_ylim_min + fig1_ylim_max)

    half_x = 0.5 * (xspan + 2.0 * xpad) / zoom
    half_y = 0.5 * (yspan + 2.0 * ypad) / zoom

    ax1.set_xlim(xcenter - half_x, xcenter + half_x)
    ax1.set_ylim(ycenter - half_y, ycenter + half_y)
    ax1.set_aspect("equal", adjustable="box")

    # Keep mesh legend and add a second legend for the attribute points
    if mesh_legend is not None:
        mesh_legend.set_loc("upper left")
        ax1.add_artist(mesh_legend)
        ax1.legend(shape_handles, shape_labels, title="Attributes", loc="upper right")
    else:
        # upper right keeps the legend clear of the fracture footprints, which
        # grow from the origin toward +x
        ax1.legend(shape_handles, shape_labels, title="Attributes", loc="upper right")

    if not legends:
        ax1.legend().set_visible(False)
        ax2.legend().set_visible(False)
        ax3.legend().set_visible(False)
        ax4.legend().set_visible(False)
        ax5.legend().set_visible(False)

    display(fig1) if frac_mesh is not None else None

    return ((fig1, ax1), (fig2, ax2), (fig3, ax3), (fig4, ax4), (fig5, ax5)), ret_dict


def plot_pressure_vs_length(proc_atrb_dict, rename_dict, figsize=(12, 8)):
    """Twin-axis pressure(t) + fracture length(t) per run.

    (Previously duplicated inline in the KGD spacing and injection sections.)
    """
    for k in rename_dict:
        print(k)

        pressure = proc_atrb_dict[k]["pressure"][:, 0] * 1e-6
        time = proc_atrb_dict[k]["time"]
        frac_length = proc_atrb_dict[k]["fracture_length"]

        fig, ax1 = plt.subplots(figsize=figsize)

        color = 'tab:red'
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Pressure (MPa)', color=color)
        ax1.plot(time, pressure, marker='o', color=color, label=f"Pressure {max(pressure):.2f} MPa")
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()   # second Axes sharing the x-axis

        color = 'tab:blue'
        ax2.set_ylabel('Fracture Length (m)', color=color)
        ax2.plot(time, frac_length, marker="o", color=color, label=f"Fracture Length {max(frac_length):.2f} m")
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()
        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")
        plt.show()


# ============================================================================
# waterfall plots (strain + strain rate)
# ============================================================================

def plot_waterfall_pair(waterfall, time, offsets, comp="yy", clim=None,
                        rate_clim=None, ylim=None, title="", offset_label="y (m)",
                        time_label="Time (s)", figsize=(20, 6), annotate_minmax=True,
                        smooth_frames=1, axes=None):
    """Side-by-side strain and strain-RATE waterfalls for one run.

    Uses pcolormesh on true cell edges so graded meshes are geometrically
    correct (imshow with extent= stretches non-uniform offsets).

    Parameters:
    -----------
        waterfall (ndarray): (nt, n_offsets) strain array.
        time (ndarray): Frame times, same unit as `time_label`.
        offsets (ndarray): Position of each channel along the fiber/line.
        comp (str, optional): Tensor component for labels ("yy", "xx", ...).
        clim (float, optional): Symmetric strain color limit; robust default.
        rate_clim (float, optional): Symmetric rate color limit; robust default.
        ylim (tuple, optional): Offset-axis limits.
        title (str, optional): Figure suptitle.
        annotate_minmax (bool, optional): Print min/max in the strain panel.
        smooth_frames (int, optional): Temporal smoothing (frames) for the
            RATE panel only; see strain_rate. Defaults to 1 (off).
        axes (sequence, optional): Two existing axes (strain, rate) to draw
            into - used to assemble multi-run grid figures.  When given, no
            figure is created, `title` prefixes the panel titles instead of
            becoming a suptitle, and plt.show() is NOT called.

    Returns:
    -----------
        (fig, (ax_strain, ax_rate))
    """
    waterfall = np.asarray(waterfall, float)
    time = np.asarray(time, float).ravel()
    rate = strain_rate(waterfall, time, smooth_frames=smooth_frames)

    if clim is None:
        clim = np.nanpercentile(np.abs(waterfall), 99) * 1.2
    if rate_clim is None:
        rate_clim = np.nanpercentile(np.abs(rate), 99) * 1.2

    te, oe = cell_edges(time), cell_edges(offsets)
    unit = time_label.split("(")[-1].rstrip(")")
    panels = [
        (waterfall, clim, rf"$\epsilon_{{{comp}}}$", "strain"),
        (rate, rate_clim, rf"$\dot{{\epsilon}}_{{{comp}}}$ (1/{unit})", "strain rate"),
    ]

    own_fig = axes is None
    if own_fig:
        fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    else:
        fig = axes[0].figure
    for ax, (arr, cl, lbl, sub) in zip(axes, panels):
        pm = ax.pcolormesh(te, oe, arr.T, cmap="seismic", vmin=-cl, vmax=cl)
        ax.set_xlabel(time_label)
        ax.set_title(f"{sub}" if own_fig else f"{title} - {sub}", fontsize=10)
        if ylim is not None:
            ax.set_ylim(*ylim)
        fig.colorbar(pm, ax=ax, label=lbl, fraction=0.045)
    axes[0].set_ylabel(offset_label)

    if annotate_minmax:
        axes[0].text(0.95, 0.95, f"Max: {np.nanmax(waterfall):.2e}\nMin: {np.nanmin(waterfall):.2e}",
                     transform=axes[0].transAxes, fontsize=10, verticalalignment="top",
                     horizontalalignment="right",
                     bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    if own_fig:
        if title:
            fig.suptitle(title)
        plt.grid(False)
        plt.show()
    return fig, axes


def plot_fiber_panels(wf, clim_solid, title="", clim_fiber=None, rate_clim=None):
    """3-panel view of an aperture-corrected fiber waterfall
    (hdf5_tools.get_fiber_waterfall output): solid-only strain, corrected
    strain (with the aperture jump), and corrected strain rate.

    Parameters:
    -----------
        wf (dict): hdf5_tools.get_fiber_waterfall output.
        clim_solid (float): Symmetric color limit for the solid-only panel.
        title (str, optional): Figure suptitle prefix.
        clim_fiber (float, optional): Symmetric color limit for the corrected
            fiber-strain panel.  None (default) -> 0.5x the corrected |max|
            (the crossing core dominates; a data max would white-out the lobes).
        rate_clim (float, optional): Symmetric color limit for the corrected
            strain-rate panel (1/s).  None -> 0.5x the rate's |max|.
    """
    cl_fib = (np.nanmax(np.abs(wf["corrected"]))*0.5 if clim_fiber is None
              else clim_fiber)
    cl_rate = (np.nanmax(np.abs(wf["corrected_rate"]))*0.5 if rate_clim is None
               else rate_clim)
    te, ye = cell_edges(wf["time"]), cell_edges(wf["offsets"])
    panels = [
        (r"solid $\epsilon_{yy}$ only (misses the jump)", wf["solid"], clim_solid, r"$\epsilon_{yy}$"),
        (r"fiber $\epsilon_{yy}$ (+ aperture jump)", wf["corrected"], cl_fib, r"$\epsilon_{yy}$"),
        (r"fiber strain rate $\dot{\epsilon}_{yy}$", wf["corrected_rate"],
         cl_rate, r"$\dot{\epsilon}_{yy}$ (1/s)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(24, 6), sharey=True)
    for ax, (ttl, arr, cl, lbl) in zip(axes, panels):
        pm = ax.pcolormesh(te, ye, arr.T, cmap="seismic", vmin=-cl, vmax=cl)
        ax.set_title(ttl)
        ax.set_xlabel("Time (s)")
        fig.colorbar(pm, ax=ax, label=lbl, fraction=0.05)
    axes[0].set_ylabel("y (m)")
    axes[0].set_ylim(-60, 60)
    fig.suptitle(f"{title} - DSS fiber along y, "
                 f"crossing rows y = {np.round(wf['offsets'][wf['crossing_rows']], 2)}")
    plt.grid(False)
    plt.show()
    return fig, axes


def plot_waterfall_grid(entries, clim=None, rate_clim=None, ylim=None, comp="xx",
                        offset_label="x (m)", time_label="Time (s)",
                        smooth_frames=1, panel_height=2.3, width=11):
    """Multi-run strain / strain-rate waterfall grid with shared decorations.

    One row per run, columns = strain and strain rate.  Designed for print
    readability: only the top row carries the column titles ("Strain",
    "Strain Rate"), the run identity is written vertically at the left of
    each row, tick labels appear only on the bottom row / left column, a
    single shared time label and offset label serve the whole grid, and two
    horizontal colorbars at the bottom replace the per-panel ones (strain
    and strain rate carry different units, so they cannot share a single
    bar).  Both columns use ONE color limit across all rows so the rows are
    directly comparable.

    Parameters:
    -----------
        entries (list): [(row_label, waterfall, time, offsets), ...] with
            waterfall shaped (nt, n_offsets).
        clim (float, optional): Shared symmetric strain limit.  None -> the
            99th-percentile x 1.2 rule, maximized over rows.
        rate_clim (float, optional): Same for the strain-rate column.
        ylim (tuple, optional): Offset-axis limits for every panel.
        comp (str, optional): Tensor component for the colorbar labels.
        smooth_frames (int, optional): Temporal smoothing for the rate
            column (see strain_rate). Defaults to 1 (off).
        panel_height, width (float, optional): Figure geometry per row / total.

    Returns:
    -----------
        (fig, axes): axes shaped (n_rows, 2).
    """
    rates = [strain_rate(np.asarray(w, float), np.asarray(t, float).ravel(),
                         smooth_frames=smooth_frames) for _, w, t, _ in entries]
    if clim is None:
        clim = max(np.nanpercentile(np.abs(np.asarray(w, float)), 99) * 1.2
                   for _, w, _, _ in entries)
    if rate_clim is None:
        rate_clim = max(np.nanpercentile(np.abs(r), 99) * 1.2 for r in rates)

    n = len(entries)
    fig, axes = plt.subplots(n, 2, figsize=(width, panel_height * n),
                             sharex=True, sharey=True, squeeze=False,
                             gridspec_kw=dict(hspace=0.12, wspace=0.06))
    for r, ((label, w, t, offs), rate) in enumerate(zip(entries, rates)):
        te, oe = cell_edges(np.asarray(t, float).ravel()), cell_edges(offs)
        pm_s = axes[r, 0].pcolormesh(te, oe, np.asarray(w, float).T,
                                     cmap="seismic", vmin=-clim, vmax=clim)
        pm_r = axes[r, 1].pcolormesh(te, oe, rate.T,
                                     cmap="seismic", vmin=-rate_clim, vmax=rate_clim)
        if ylim:
            axes[r, 0].set_ylim(*ylim)
        axes[r, 0].set_ylabel(label, fontsize=11)     # vertical row identity
    axes[0, 0].set_title("Strain", fontsize=13)
    axes[0, 1].set_title("Strain Rate", fontsize=13)
    fig.supxlabel(time_label, fontsize=12, y=0.075)
    fig.supylabel(offset_label, fontsize=12, x=0.015)

    # shared horizontal colorbars along the bottom, one per column
    fig.subplots_adjust(bottom=0.115, left=0.14, right=0.985, top=0.965)
    cax_s = fig.add_axes([0.155, 0.045, 0.38, 0.012])
    cax_r = fig.add_axes([0.595, 0.045, 0.38, 0.012])
    fig.colorbar(pm_s, cax=cax_s, orientation="horizontal",
                 label=rf"$\epsilon_{{{comp}}}$")
    fig.colorbar(pm_r, cax=cax_r, orientation="horizontal",
                 label=rf"$\dot{{\epsilon}}_{{{comp}}}$ (1/s)")
    plt.show()
    return fig, axes


def plot_fiber_vs_displacement(wf, dwf, title="", clim=None, rate_clim=None):
    """Aperture-corrected vs displacement-based fiber waterfalls side by side
    (validation of the aperture approximation) plus the displacement-based
    strain rate.

    Parameters:
    -----------
        wf (dict): hdf5_tools.get_fiber_waterfall output (aperture-corrected).
        dwf (dict): hdf5_tools.get_displacement_fiber_waterfall output.
        title (str, optional): Figure suptitle prefix.
        clim (float, optional): Symmetric color limit for BOTH strain panels.
            None (default) -> 0.5x the displacement fiber's |max| (the core
            dominates, so a data max would white-out the lobes).
        rate_clim (float, optional): Symmetric color limit for the strain-rate
            panel (1/s).  None -> 0.5x the rate's |max|.
    """
    cl = np.nanmax(np.abs(dwf["fiber"]))*0.5 if clim is None else clim
    rcl = (np.nanmax(np.abs(dwf["fiber_rate"]))*0.5 if rate_clim is None
           else rate_clim)

    panels = [
        (r"aperture-corrected $\epsilon_{yy}$ (approximation)", wf["corrected"], wf["time"], wf["offsets"], cl, r"$\epsilon_{yy}$"),
        (r"displacement-based $\epsilon_{yy}$ (exact jump)", dwf["fiber"], dwf["time"], dwf["offsets"], cl, r"$\epsilon_{yy}$"),
        (r"displacement-based strain rate $\dot{\epsilon}_{yy}$", dwf["fiber_rate"], dwf["time"], dwf["offsets"],
         rcl, r"$\dot{\epsilon}_{yy}$ (1/s)"),
    ]

    # shared decorations for print readability: per-panel titles kept (they
    # differ in content), but one shared y axis, one shared time label, and
    # horizontal colorbars along the bottom (one spanning the two strain
    # panels, one for the rate panel) instead of three vertical ones.
    fig, axes = plt.subplots(1, 3, figsize=(22, 6.5), sharey=True,
                             gridspec_kw=dict(wspace=0.06))
    pms = []
    for ax, (ttl, arr, t_ax, offs, _cl, lbl) in zip(axes, panels):
        pm = ax.pcolormesh(cell_edges(t_ax), cell_edges(offs), arr.T, cmap="seismic", vmin=-_cl, vmax=_cl)
        pms.append(pm)
        ax.set_title(ttl)
    axes[0].set_ylim(-60, 60)
    fig.supxlabel("Time (s)", fontsize=13, y=0.09)
    fig.supylabel("y (m)", fontsize=13, x=0.055)
    fig.suptitle(f"{title} - DSS fiber @ (x, z) = {dwf['fiber_position']}")
    fig.subplots_adjust(bottom=0.16, left=0.09, right=0.985, top=0.88)
    cax_s = fig.add_axes([0.11, 0.055, 0.50, 0.015])
    cax_r = fig.add_axes([0.70, 0.055, 0.27, 0.015])
    fig.colorbar(pms[0], cax=cax_s, orientation="horizontal", label=r"$\epsilon_{yy}$")
    fig.colorbar(pms[2], cax=cax_r, orientation="horizontal", label=r"$\dot{\epsilon}_{yy}$ (1/s)")
    plt.grid(False)
    plt.show()
    return fig, axes


def plot_displacement_fiber_grid(wf_dict, monitor_x, off_break=15.0, ylim=(-100, 100),
                                 smooth_frames=1, clim=None):
    """Strain + strain-rate waterfalls for several runs' displacement-based
    fiber dicts (hdf5_tools.get_displacement_fiber_waterfall), one column per
    run, time in minutes.  The rate color scale is set from OFF-core channels
    (|offset| > off_break) so the saturated core stripe cannot hide the lobes.
    `smooth_frames` (see strain_rate) suppresses per-frame striping in the
    rate row.

    clim (float, optional): Fixed symmetric strain color limit (strain, not
        microstrain) applied to EVERY run's strain row, e.g. 2e-5.  A shared
        limit keeps the panels amplitude-comparable (the amplitude contrast
        between new and re-dilated fractures is one of the discriminators)
        while low enough that the leading extension stays visible.  None
        (default) auto-scales each run to 0.5x its own |max|.
    """
    # shared decorations: run names as column titles, "Strain" / "Strain rate"
    # written vertically at the left of each row, tick labels only on the
    # outer edges, and two horizontal colorbars along the bottom.  The rate
    # scale is shared across the runs (off-core rule, maximized over runs)
    # so the amplitude contrast between the runs stays visible.
    names = list(wf_dict)
    # shared strain limit: the fixed clim if given, else 0.5x the largest |max|
    strain_cl = (clim*1e6 if clim is not None else
                 max(np.nanmax(np.abs(wf["fiber"]))*1e6*0.5 for wf in wf_dict.values()))
    rates, rate_cl = {}, 0.0
    for name, wf in wf_dict.items():
        tm = np.asarray(wf["time"]).ravel()/60.0
        off = np.abs(wf["offsets"]) > off_break
        sr = strain_rate(wf["fiber"], tm, smooth_frames=smooth_frames)*1e6   # microstrain / min
        rates[name] = sr
        rate_cl = max(rate_cl, np.percentile(np.abs(sr[:, off]), 99)*1.2)

    fig, axes = plt.subplots(2, len(wf_dict), figsize=(6.0*len(wf_dict), 9),
                             sharex=True, sharey=True, squeeze=False,
                             gridspec_kw=dict(hspace=0.08, wspace=0.06))
    for j, name in enumerate(names):
        wf = wf_dict[name]
        tm = np.asarray(wf["time"]).ravel()/60.0
        te, oe = cell_edges(tm), cell_edges(wf["offsets"])
        pm_s = axes[0][j].pcolormesh(te, oe, (wf["fiber"]*1e6).T, cmap="seismic",
                                     vmin=-strain_cl, vmax=strain_cl)
        pm_r = axes[1][j].pcolormesh(te, oe, rates[name].T, cmap="seismic",
                                     vmin=-rate_cl, vmax=rate_cl)
        axes[0][j].set_ylim(*ylim)
        axes[0][j].set_title(name, fontsize=12)
    axes[0][0].set_ylabel("Strain", fontsize=12)
    axes[1][0].set_ylabel("Strain rate", fontsize=12)
    fig.supxlabel("time (min)", fontsize=12, y=0.075)
    fig.supylabel("y along fiber (m)", fontsize=12, x=0.02)
    fig.suptitle(f"Displacement-based fiber at x = {monitor_x:.0f} m")
    fig.subplots_adjust(bottom=0.115, left=0.11, right=0.985, top=0.92)
    cax_s = fig.add_axes([0.13, 0.045, 0.37, 0.013])
    cax_r = fig.add_axes([0.60, 0.045, 0.37, 0.013])
    fig.colorbar(pm_s, cax=cax_s, orientation="horizontal", label=r"$\epsilon_{yy}$ (ue)")
    fig.colorbar(pm_r, cax=cax_r, orientation="horizontal", label=r"$\dot{\epsilon}_{yy}$ (ue/min)")
    plt.grid(False)
    plt.show()
    return fig, axes


# ============================================================================
# displacement loading (HDF5 preferred, PVD fallback)
# ============================================================================

def read_displacement_pvd(pvd_path, x_center, x_tol=2.5, z_tol=3.0, array="totalDisplacement"):
    """Fallback displacement reader: extract the fiber node column near
    (x_center, z=0) from every timestep of a GEOS PVD/VTK series.

    Returns (time [nt], positions [nt, nn, 3], disp [nt, nn, 3]) - the same
    tuple as hdf5_tools.read_displacement_hdf5, so the two are interchangeable.
    Points shared between rank pieces are deduplicated by rounded coordinates.
    GEOS VTK points are REFERENCE positions (displacement is the array).
    """
    import pyvista as pv   # lazy: only needed when the HDF5 is missing
    reader = pv.get_reader(pvd_path)
    times = np.asarray(reader.time_values)

    def _leaf_grids(block, out):
        for i in range(block.n_blocks):
            sub = block[i]
            if sub is None:
                continue
            if isinstance(sub, pv.MultiBlock):
                _leaf_grids(sub, out)
            elif array in sub.point_data:
                out.append(sub)
        return out

    ref_pos, disp_frames = None, []
    for tv in times:
        reader.set_active_time_value(tv)
        pts_list, u_list = [], []
        for g in _leaf_grids(reader.read(), []):
            p = np.asarray(g.points)
            m = (np.abs(p[:, 0] - x_center) < x_tol) & (np.abs(p[:, 2]) < z_tol)
            if m.any():
                pts_list.append(p[m])
                u_list.append(np.asarray(g.point_data[array])[m])
        pts = np.vstack(pts_list)
        u = np.vstack(u_list)
        srt = np.lexsort((pts[:, 2], pts[:, 1], pts[:, 0]))
        pts, u = pts[srt], u[srt]
        keep = np.ones(len(pts), bool)
        keep[1:] = np.any(np.abs(np.diff(pts, axis=0)) > 1e-6, axis=1)   # drop rank-duplicate points
        pts, u = pts[keep], u[keep]
        if ref_pos is None:
            ref_pos = pts
        disp_frames.append(u)
    disp = np.stack(disp_frames)
    positions = np.broadcast_to(ref_pos, (len(times), *ref_pos.shape)).copy()
    return times, positions, disp


def load_case_displacement(run_dir, case, monitor_x, prefer="hdf5"):
    """Displacement for one heart-signature run: HDF5 fiber TimeHistory when
    available (5 s cadence), PVD tree otherwise (60 s cadence, slower read).

    Parameters:
    -----------
        run_dir (str): Run output directory.
        case (str): "new" or "old" (selects the file names).
        monitor_x (float): Fiber x-offset, used for the PVD column extraction.
        prefer (str, optional): "hdf5" or "pvd". Defaults to "hdf5".

    Returns:
    -----------
        tuple: (time [s], positions [nt, nn, 3], disp [nt, nn, 3])
    """
    import os
    fiber_h5 = os.path.join(run_dir, f"{case}_fracture_fiber.hdf5")
    pvd = os.path.join(run_dir, f"{case}_fracture.pvd")
    if prefer == "hdf5" and os.path.exists(fiber_h5):
        loaded = hdt.read_displacement_hdf5(fiber_h5)
        if loaded is not None:
            print(f"{case}: displacement from HDF5 ({os.path.basename(fiber_h5)}, {loaded[0].size} frames)")
            return loaded
    print(f"{case}: displacement from PVD ({os.path.basename(pvd)}) ...")
    return read_displacement_pvd(pvd, x_center=monitor_x)
