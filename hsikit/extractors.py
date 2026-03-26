import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from matplotlib.path import Path
from matplotlib.axes import Axes

from typing import Optional, Literal


# ---------- Extract median/mean spectra per rectangle grid-based ROIs --------------
def Grid_ROI_extractor(
    cube: NDArray,
    start: int | tuple[int, int],
    roi_size: int | tuple[int, int],
    gap: int | tuple[int, int],
    n_rois: int | tuple[int, int],
    statistic: Literal['median', 'mean'] = 'median',
    ignore_nan: bool = True,
    band_for_display: Optional[int] = None,
    ax: Optional[Axes] = None,
    visualize: bool = True,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """
    Extracts ROIs column-wise.
    
    Parameters
    ----------
    cube : NDArray
        HSI 3D array, expected shape (H, W, B)
    start : int | tuple[int, int]
        Top-left corner of the first ROI
        If int -> same coordinate index
        If tuple -> (start_row, start_col)
    roi_size : int | tuple[int, int]
        If int -> square ROI
        If tuple -> (roi_height, roi_width)
    gap : int | tuple[int, int]
        Gap between ROIs (pixels)
        If int -> same gap for rows and cols
        If tuple -> (row_gap, col_gap)
    n_rois : int | tuple[int, int]
        Number of ROIs vertically and horizontally
        If int -> same number of ROIs
        If tuple -> (n_rows, n_cols)
    band_for_display : Optional[int]
        Band index to visualize; if None, mean over bands
    ax : Optional[Axes]
        Axis to draw on
    visualize : bool
        Whether to plot the base image overlayed with ROIs as red rectangles

    Returns
    -------
    spectra : ndarray (N_rois, B)
        Median spectrum per ROI
    roi_coords : list of tuples
        (r0, r1, c0, c1) for each ROI
    """

    H, W, B = cube.shape

    # --- normalize inputs ---
    start_row, start_col = (start, start) if isinstance(start, int) else start
    roi_h, roi_w = (roi_size, roi_size) if isinstance(roi_size, int) else roi_size
    row_gap, col_gap = (gap, gap) if isinstance(gap, int) else gap
    n_rows, n_cols = (n_rois, n_rois) if isinstance(n_rois, int) else n_rois

    # --- statistic ---
    if statistic.lower() == "mean":
        stat_func = np.nanmean if ignore_nan else np.mean
    elif statistic.lower() == "median":
        stat_func = np.nanmedian if ignore_nan else np.median
    else:
        raise ValueError("statistic must be 'mean' or 'median'")

    spectra = []
    roi_coords = []

    # --- visualization setup ---
    if visualize:
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))

        img = cube.mean(axis=2) if band_for_display is None else cube[:, :, band_for_display]
        ax.imshow(img, cmap="gray")
        ax.axis("off")

    # --- main loop ---
    for j in range(n_cols):
        for i in range(n_rows):
            r0 = start_row + i * (roi_h + row_gap)
            c0 = start_col + j * (roi_w + col_gap)
            r1 = r0 + roi_h
            c1 = c0 + roi_w

            if r1 > H or c1 > W:
                continue

            roi = cube[r0:r1, c0:c1, :]
            spectrum = stat_func(roi, axis=(0, 1))

            spectra.append(spectrum)
            roi_coords.append((c0, r0, c1, r1))  # unified format

            if visualize:
                rect = plt.Rectangle(
                    (c0, r0),
                    roi_w,
                    roi_h,
                    linewidth=1.5,
                    edgecolor="red",
                    facecolor="none",
                )
                ax.add_patch(rect)

    return np.array(spectra), roi_coords


# ----------------- Interactive rectangle/polygon spectra extractor --------------------
class ROIExtractor:
    """
    Unified interactive ROI extractor for hyperspectral cubes.
    - ! requires %matplotlib widget !
    - ! Usage (Jupyter): ROIExtractor.show() -> interact/draw -> press ENTER -> call get_results() in next cell !

    Methods
    -------
    - show():
        Plots bg image and lets user define ROIs -> None.
    - get_results()
        Returns the results of extraction from defined ROIs -> tuple[spectra, rois]
    
    Modes
    -----
    - "rectangle"
    - "polygon"

    Controls
    --------
    - Rectangle:
        Drag mouse → select ROI
        ENTER → finish

    - Polygon:
        Left click → add vertex
        SPACE → close polygon
        ESC → cancel polygon
        ENTER → finish
    """

    def __init__(self, cube, mode="rectangle", band_index=150,
                 statistic="median", ignore_nan=True, min_span=5):

        if cube.ndim != 3:
            raise ValueError("cube must have shape (rows, cols, bands)")

        self.cube = cube
        self.mode = mode
        self.band_index = band_index
        self.ignore_nan = ignore_nan
        self.min_span = min_span

        self.spectra = []
        self.rois = []
        self.finished = False

        # Statistic
        if statistic.lower() == "mean":
            self.stat_func = np.nanmean if ignore_nan else np.mean
        elif statistic.lower() == "median":
            self.stat_func = np.nanmedian if ignore_nan else np.median
        else:
            raise ValueError("statistic must be 'mean' or 'median'")

        # Figure
        self.fig, self.ax = plt.subplots()
        self.ax.imshow(self.cube[:, :, band_index], cmap="gray")

        # Mode-specific init
        if mode == "rectangle":
            self._init_rectangle()
        elif mode == "polygon":
            self._init_polygon()
        else:
            raise ValueError("mode must be 'rectangle' or 'polygon'")

        # Common key handler
        self.cid_key = self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    # =========================
    # RECTANGLE MODE
    # =========================
    def _init_rectangle(self):
        self.ax.set_title("Draw rectangles. Press ENTER to finish.")

        self.selector = RectangleSelector(
            self.ax,
            self._onselect_rectangle,
            useblit=False,
            button=[1],
            minspanx=self.min_span,
            minspany=self.min_span,
            spancoords='data'
        )

    def _onselect_rectangle(self, eclick, erelease):
        if self.finished:
            return
        if None in (eclick.xdata, eclick.ydata, erelease.xdata, erelease.ydata):
            return

        x0, y0 = int(eclick.xdata), int(eclick.ydata)
        x1, y1 = int(erelease.xdata), int(erelease.ydata)

        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))

        h, w, _ = self.cube.shape
        x0, x1 = np.clip([x0, x1], 0, w)
        y0, y1 = np.clip([y0, y1], 0, h)

        if x1 <= x0 or y1 <= y0:
            return

        roi = self.cube[y0:y1, x0:x1, :]
        if roi.size == 0:
            return

        spectrum = self.stat_func(roi, axis=(0, 1))

        self.spectra.append(spectrum)
        self.rois.append((x0, y0, x1, y1))

        rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                             fill=False, edgecolor='red', linewidth=1.5)
        self.ax.add_patch(rect)

        self.ax.text(x0, y0, str(len(self.spectra)),
                     color='yellow', fontsize=10)

        self.ax.set_title(f"ROIs: {len(self.spectra)} | ENTER to finish")
        self.fig.canvas.draw_idle()

    # =========================
    # POLYGON MODE
    # =========================
    def _init_polygon(self):
        self.ax.set_title("Click: vertices | SPACE: close | ENTER: finish")

        self.current_vertices = []

        ny, nx = self.cube.shape[:2]
        y, x = np.mgrid[:ny, :nx]
        self.grid_points = np.vstack((x.ravel(), y.ravel())).T
        self.shape = (ny, nx)

        self.current_line, = self.ax.plot([], [], color="red", lw=1.5)

        self.cid_click = self.fig.canvas.mpl_connect(
            "button_press_event", self._on_click_polygon
        )

    def _on_click_polygon(self, event):
        if self.finished or event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        self.current_vertices.append((event.xdata, event.ydata))

        xs, ys = zip(*self.current_vertices)
        self.current_line.set_data(xs, ys)
        self.fig.canvas.draw_idle()

    def _finalize_polygon(self):
        if len(self.current_vertices) < 3:
            print("Polygon needs at least 3 points.")
            return

        polygon = np.array(self.current_vertices)

        closed = np.vstack([polygon, polygon[0]])
        self.ax.plot(closed[:, 0], closed[:, 1], color="red", lw=1.5)

        mask = Path(polygon).contains_points(self.grid_points).reshape(self.shape)
        roi_pixels = self.cube[mask]

        if roi_pixels.size > 0:
            spectrum = self.stat_func(roi_pixels, axis=0)
            self.spectra.append(spectrum)
            self.rois.append(polygon)

            cx, cy = polygon.mean(axis=0)
            self.ax.text(cx, cy, str(len(self.spectra)),
                         color="yellow", fontsize=10,
                         ha="center", va="center")

        self.current_vertices = []
        self.current_line.set_data([], [])

        self.ax.set_title(f"ROIs: {len(self.spectra)} | ENTER to finish")
        self.fig.canvas.draw_idle()

    # =========================
    # COMMON HANDLER
    # =========================
    def _on_key(self, event):
        if self.finished:
            return

        if self.mode == "polygon":
            if event.key == " ":
                self._finalize_polygon()
            elif event.key == "escape":
                self.current_vertices = []
                self.current_line.set_data([], [])
                self.fig.canvas.draw_idle()

        if event.key == "enter":
            print("Finished ROI selection.")
            self.finished = True

            if self.mode == "rectangle":
                self.selector.set_active(False)
            elif self.mode == "polygon":
                self.fig.canvas.mpl_disconnect(self.cid_click)

            plt.close(self.fig)

    # =========================
    # PUBLIC API
    # =========================
    def show(self):
        plt.show()

    def get_results(self):
        if not self.finished:
            raise RuntimeError("Selection not finished. Press ENTER first.")
        return np.array(self.spectra), self.rois