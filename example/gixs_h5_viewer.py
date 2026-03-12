import os
import gc
import sys
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import h5py

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm, Normalize
from matplotlib.widgets import RectangleSelector
from matplotlib.patches import Rectangle


H5_IMG_PATH = "/entry_0000/data/img_gid_q"
H5_QXY_PATH = "/entry_0000/data/q_xy"
H5_QZ_PATH  = "/entry_0000/data/q_z"


def _load_h5_raw(path):
    with h5py.File(path, "r") as f:
        img = f[H5_IMG_PATH][()]
        qxy = np.ravel(f[H5_QXY_PATH][()]).astype(float)
        qz  = np.ravel(f[H5_QZ_PATH][()]).astype(float)

    img = np.asarray(img)
    if img.ndim >= 3 and img.shape[0] == 1:
        img = np.squeeze(img, axis=0)
    else:
        img = np.squeeze(img)

    if img.ndim == 2:
        img2d = img.astype(float)
        if img2d.shape != (qz.size, qxy.size):
            if img2d.shape == (qxy.size, qz.size):
                img2d = img2d.T
            else:
                raise ValueError(
                    f"2D shape mismatch: img={img2d.shape}, expected {(qz.size, qxy.size)} "
                    f"(or transposed {(qxy.size, qz.size)})."
                )
        return img2d, qxy, qz

    if img.ndim == 3:
        img3d = img.astype(float)
        n, a, b = img3d.shape
        if (a, b) != (qz.size, qxy.size):
            if (a, b) == (qxy.size, qz.size):
                img3d = np.transpose(img3d, (0, 2, 1))
            else:
                raise ValueError(
                    f"3D shape mismatch: img={img3d.shape}, expected (n,{qz.size},{qxy.size}) "
                    f"or (n,{qxy.size},{qz.size})."
                )
        return img3d, qxy, qz

    raise ValueError(f"img_gid_q must be 2D or 3D after squeeze; got ndim={img.ndim}, shape={img.shape}")


class ScrollableFrame(tk.Frame):
    def __init__(self, master, width=None, height=None, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.canvas = tk.Canvas(self, highlightthickness=0, width=width, height=height)
        self.v_scroll = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.v_scroll.set)

        self.inner = tk.Frame(self.canvas)
        self.inner_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.v_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.inner.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self._bind_mousewheel(self.canvas)

    def _on_frame_configure(self, _evt=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, evt):
        self.canvas.itemconfig(self.inner_id, width=evt.width)

    def _bind_mousewheel(self, widget):
        def _on_mousewheel(event):
            if getattr(event, "delta", 0):
                self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            elif getattr(event, "num", None) == 4:
                self.canvas.yview_scroll(-1, "units")
            elif getattr(event, "num", None) == 5:
                self.canvas.yview_scroll(1, "units")

        widget.bind_all("<MouseWheel>", _on_mousewheel)
        widget.bind_all("<Button-4>", _on_mousewheel)
        widget.bind_all("<Button-5>", _on_mousewheel)


class GIXSViewer(tk.Tk):
    def __init__(self, start_folder=None):
        super().__init__()
        self.title("GIXS HDF5 Viewer")
        self.geometry("1780x980")

        self.start_folder = start_folder if (start_folder and os.path.isdir(start_folder)) else None

        self.current_folder = None
        self.current_file = None

        self.img_raw = None
        self.qxy = None
        self.qz  = None

        self.bg_file = None
        self.bg_raw = None
        self.bg_qxy = None
        self.bg_qz  = None

        self.frame_index = tk.IntVar(value=0)
        self.sum_frames = tk.BooleanVar(value=False)

        self.im = None

        self.cmap_var = tk.StringVar(value="gist_rainbow")
        self.scale_mode = tk.StringVar(value="log")

        self.log_vmin = tk.DoubleVar(value=-1.0)
        self.log_vmax = tk.DoubleVar(value=3.7)
        self.lin_vmin = tk.DoubleVar(value=0.0)
        self.lin_vmax = tk.DoubleVar(value=5000.0)

        self.qxy_min_var = tk.DoubleVar(value=0.0)
        self.qxy_max_var = tk.DoubleVar(value=0.0)
        self.qz_min_var  = tk.DoubleVar(value=0.0)
        self.qz_max_var  = tk.DoubleVar(value=0.0)
        self._ranges_initialized = False

        self.bg_enabled = tk.BooleanVar(value=False)
        self.bg_scale = tk.DoubleVar(value=1.0)

        self._pending_redraw_id = None
        self._redraw_delay_ms = 60

        self.base_label_font = 18
        self.label_font = int(self.base_label_font)
        self.tick_font = self.label_font

        self.roi_v = {"x0": tk.DoubleVar(value=0.0), "z0": tk.DoubleVar(value=0.0),
                      "w": tk.DoubleVar(value=0.01), "h": tk.DoubleVar(value=0.01)}
        self.roi_h = {"x0": tk.DoubleVar(value=0.0), "z0": tk.DoubleVar(value=0.0),
                      "w": tk.DoubleVar(value=0.01), "h": tk.DoubleVar(value=0.01)}
        self._roi_initialized = False

        self.roi_v_patch = None
        self.roi_h_patch = None

        self._drag_roi_kind = None
        self._drag_anchor = None
        self._drag_mode = None
        self._drag_resize_corner = None
        self._roi_handle_tol = 0.02

        self._rect_selector = None

        self.lc_log_v = tk.BooleanVar(value=False)
        self.lc_log_h = tk.BooleanVar(value=False)

        self._build_ui()
        self._build_plot_panes()

        self._sync_slider_visibility()
        self._sync_entries_from_vars()
        self._sync_frame_controls_visibility()

        if self.start_folder:
            self.current_folder = self.start_folder
            self.populate_file_list(self.start_folder)

        self.bind("<Left>", self._on_key_left)
        self.bind("<Right>", self._on_key_right)
        self.bind("<Shift-Left>", self._on_key_left_fast)
        self.bind("<Shift-Right>", self._on_key_right_fast)

    # ---------------- UI (top + left) ----------------
    def _build_ui(self):
        top = tk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        tk.Button(top, text="Open folder", command=self.open_folder).pack(side=tk.LEFT)
        tk.Button(top, text="Refresh", command=self.refresh_file_list).pack(side=tk.LEFT, padx=(6, 0))
        tk.Button(top, text="Export PNG (300 DPI)", command=self.export_png).pack(side=tk.LEFT, padx=(8, 0))

        tk.Label(top, text="Colormap:").pack(side=tk.LEFT, padx=(12, 4))
        tk.OptionMenu(
            top, self.cmap_var,
            "gist_rainbow", "jet", "nipy_spectral", "viridis", "plasma",
            "inferno", "magma", "cividis"
        ).pack(side=tk.LEFT)

        tk.Label(top, text="Scale:").pack(side=tk.LEFT, padx=(12, 4))
        tk.Radiobutton(top, text="Log", variable=self.scale_mode, value="log",
                       command=self.update_plot).pack(side=tk.LEFT)
        tk.Radiobutton(top, text="Linear", variable=self.scale_mode, value="linear",
                       command=self.update_plot).pack(side=tk.LEFT)

        self.left_scroll = ScrollableFrame(self, width=520)
        self.left_scroll.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)
        left = self.left_scroll.inner

        tk.Label(left, text="Files", font=("Arial", 12, "bold")).pack(anchor="w")
        file_box = tk.Frame(left)
        file_box.pack(fill=tk.X, pady=(4, 10))

        self.file_list = tk.Listbox(file_box, width=62, height=14)
        file_scroll = tk.Scrollbar(file_box, orient="vertical", command=self.file_list.yview)
        self.file_list.configure(yscrollcommand=file_scroll.set)
        self.file_list.pack(side=tk.LEFT, fill=tk.X, expand=True)
        file_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.file_list.bind("<<ListboxSelect>>", self.on_select_file)

        tk.Label(left, text="ROI averaging + linecuts", font=("Arial", 12, "bold")).pack(anchor="w", pady=(6, 4))
        roi_box = tk.LabelFrame(left, text="Draw (new) ROI by drag, or edit numbers")
        roi_box.pack(fill=tk.X, pady=(0, 10))

        btn_row = tk.Frame(roi_box)
        btn_row.pack(fill=tk.X, padx=6, pady=(6, 6))
        tk.Button(btn_row, text="Draw Vertical ROI", command=lambda: self.start_roi_draw("v")).pack(side=tk.LEFT)
        tk.Button(btn_row, text="Draw Horizontal ROI", command=lambda: self.start_roi_draw("h")).pack(side=tk.LEFT, padx=(8, 0))
        tk.Button(btn_row, text="Update linecuts", command=self.update_linecuts).pack(side=tk.LEFT, padx=(8, 0))

        log_row = tk.Frame(roi_box)
        log_row.pack(fill=tk.X, padx=6, pady=(0, 6))
        tk.Checkbutton(log_row, text="log10 Y for I(Qz) (right plot)", variable=self.lc_log_v,
                       command=self.update_linecuts).pack(anchor="w")
        tk.Checkbutton(log_row, text="log10 Y for I(Qxy) (bottom plot)", variable=self.lc_log_h,
                       command=self.update_linecuts).pack(anchor="w")

        vbox = tk.LabelFrame(roi_box, text="Vertical ROI (solid): I vs Qz, avg over Qxy")
        vbox.pack(fill=tk.X, padx=6, pady=(0, 6))
        self._roi_fields(vbox, self.roi_v, apply_cmd=lambda: self.apply_roi_from_entries("v"))

        hbox = tk.LabelFrame(roi_box, text="Horizontal ROI (dashed): I vs Qxy, avg over Qz")
        hbox.pack(fill=tk.X, padx=6, pady=(0, 6))
        self._roi_fields(hbox, self.roi_h, apply_cmd=lambda: self.apply_roi_from_entries("h"))

        tk.Label(left, text="Background subtraction", font=("Arial", 12, "bold")).pack(anchor="w", pady=(6, 4))
        bg = tk.LabelFrame(left, text="Background settings (kept across files)")
        bg.pack(fill=tk.X, pady=(0, 10))

        rowb = tk.Frame(bg); rowb.pack(fill=tk.X, padx=6, pady=(6, 4))
        tk.Button(rowb, text="Set background file", command=self.set_background_file).pack(side=tk.LEFT)
        self.bg_label = tk.Label(rowb, text="(none)", anchor="w")
        self.bg_label.pack(side=tk.LEFT, padx=(8, 0), fill=tk.X, expand=True)

        rowb2 = tk.Frame(bg); rowb2.pack(fill=tk.X, padx=6, pady=(0, 6))
        tk.Checkbutton(rowb2, text="Enable subtraction", variable=self.bg_enabled,
                       command=self.reload_current).pack(side=tk.LEFT)
        tk.Label(rowb2, text="Scale:").pack(side=tk.LEFT, padx=(10, 4))
        tk.Entry(rowb2, width=8, textvariable=self.bg_scale).pack(side=tk.LEFT)
        tk.Button(rowb2, text="Apply", command=self.reload_current).pack(side=tk.LEFT, padx=(6, 0))

        tk.Label(left, text="Plot ranges", font=("Arial", 12, "bold")).pack(anchor="w", pady=(6, 4))
        rng = tk.LabelFrame(left, text="Q ranges (kept across files)")
        rng.pack(fill=tk.X, pady=(0, 10))

        r1 = tk.Frame(rng); r1.pack(fill=tk.X, padx=6, pady=(6, 2))
        tk.Label(r1, text="Qxy min:").pack(side=tk.LEFT)
        tk.Entry(r1, width=10, textvariable=self.qxy_min_var).pack(side=tk.LEFT, padx=(4, 10))
        tk.Label(r1, text="Qxy max:").pack(side=tk.LEFT)
        tk.Entry(r1, width=10, textvariable=self.qxy_max_var).pack(side=tk.LEFT, padx=(4, 10))

        r2 = tk.Frame(rng); r2.pack(fill=tk.X, padx=6, pady=(2, 6))
        tk.Label(r2, text="Qz min:").pack(side=tk.LEFT)
        tk.Entry(r2, width=10, textvariable=self.qz_min_var).pack(side=tk.LEFT, padx=(4, 10))
        tk.Label(r2, text="Qz max:").pack(side=tk.LEFT)
        tk.Entry(r2, width=10, textvariable=self.qz_max_var).pack(side=tk.LEFT, padx=(4, 10))

        tk.Button(rng, text="Apply ranges", command=self.update_plot).pack(anchor="w", padx=6, pady=(0, 6))

        tk.Label(left, text="Color scale", font=("Arial", 12, "bold")).pack(anchor="w", pady=(6, 6))

        self.log_frame = tk.LabelFrame(left, text="Log scale (edit log10 values)")
        row = tk.Frame(self.log_frame); row.pack(fill=tk.X, padx=6, pady=(6, 2))
        tk.Label(row, text="log10(vmin):").pack(side=tk.LEFT)
        self.log_vmin_entry = tk.Entry(row, width=8); self.log_vmin_entry.pack(side=tk.LEFT, padx=(4, 10))
        tk.Label(row, text="log10(vmax):").pack(side=tk.LEFT)
        self.log_vmax_entry = tk.Entry(row, width=8); self.log_vmax_entry.pack(side=tk.LEFT, padx=(4, 10))
        tk.Button(row, text="Set", command=self.apply_log_entries).pack(side=tk.LEFT)

        self.log_min_slider = tk.Scale(
            self.log_frame, from_=-15, to=15, resolution=0.05, orient=tk.HORIZONTAL,
            label="log10(vmin)", variable=self.log_vmin, command=lambda _=None: self.schedule_redraw()
        )
        self.log_min_slider.pack(fill=tk.X, padx=6, pady=4)

        self.log_max_slider = tk.Scale(
            self.log_frame, from_=-15, to=15, resolution=0.05, orient=tk.HORIZONTAL,
            label="log10(vmax)", variable=self.log_vmax, command=lambda _=None: self.schedule_redraw()
        )
        self.log_max_slider.pack(fill=tk.X, padx=6, pady=4)

        self.lin_frame = tk.LabelFrame(left, text="Linear scale (edit values)")
        row2 = tk.Frame(self.lin_frame); row2.pack(fill=tk.X, padx=6, pady=(6, 2))
        tk.Label(row2, text="vmin:").pack(side=tk.LEFT)
        self.lin_vmin_entry = tk.Entry(row2, width=10); self.lin_vmin_entry.pack(side=tk.LEFT, padx=(4, 10))
        tk.Label(row2, text="vmax:").pack(side=tk.LEFT)
        self.lin_vmax_entry = tk.Entry(row2, width=10); self.lin_vmax_entry.pack(side=tk.LEFT, padx=(4, 10))
        tk.Button(row2, text="Set", command=self.apply_lin_entries).pack(side=tk.LEFT)

        self.lin_min_slider = tk.Scale(
            self.lin_frame, from_=-5000, to=20000, resolution=1, orient=tk.HORIZONTAL,
            label="vmin", variable=self.lin_vmin, command=lambda _=None: self.schedule_redraw()
        )
        self.lin_min_slider.pack(fill=tk.X, padx=6, pady=4)

        self.lin_max_slider = tk.Scale(
            self.lin_frame, from_=-5000, to=20000, resolution=1, orient=tk.HORIZONTAL,
            label="vmax", variable=self.lin_vmax, command=lambda _=None: self.schedule_redraw()
        )
        self.lin_max_slider.pack(fill=tk.X, padx=6, pady=4)

        for s in (self.log_min_slider, self.log_max_slider, self.lin_min_slider, self.lin_max_slider):
            s.bind("<ButtonRelease-1>", lambda _evt: self.update_plot())

    def _roi_fields(self, parent, roi_vars, apply_cmd):
        frm = tk.Frame(parent)
        frm.pack(fill=tk.X, padx=6, pady=(4, 6))

        tk.Label(frm, text="x0:").grid(row=0, column=0, sticky="w")
        tk.Entry(frm, width=10, textvariable=roi_vars["x0"]).grid(row=0, column=1, padx=(4, 10))
        tk.Label(frm, text="z0:").grid(row=0, column=2, sticky="w")
        tk.Entry(frm, width=10, textvariable=roi_vars["z0"]).grid(row=0, column=3, padx=(4, 10))

        tk.Label(frm, text="width:").grid(row=1, column=0, sticky="w")
        tk.Entry(frm, width=10, textvariable=roi_vars["w"]).grid(row=1, column=1, padx=(4, 10))
        tk.Label(frm, text="height:").grid(row=1, column=2, sticky="w")
        tk.Entry(frm, width=10, textvariable=roi_vars["h"]).grid(row=1, column=3, padx=(4, 10))

        tk.Button(frm, text="Apply", command=apply_cmd).grid(row=0, column=4, rowspan=2, padx=(8, 0), sticky="ns")

    # ---------------- Plot panes ----------------
    def _build_plot_panes(self):
        self.plot_outer = tk.PanedWindow(self, orient=tk.VERTICAL, sashrelief=tk.RAISED, sashwidth=8)
        self.plot_outer.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        top_container = tk.Frame(self.plot_outer)
        bottom_container = tk.Frame(self.plot_outer)

        self.plot_outer.add(top_container, stretch="always")
        self.plot_outer.add(bottom_container, stretch="always")

        self.plot_top = tk.PanedWindow(top_container, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, sashwidth=8)
        self.plot_top.pack(fill=tk.BOTH, expand=True)

        img_container = tk.Frame(self.plot_top)
        vcut_container = tk.Frame(self.plot_top)

        self.plot_top.add(img_container, stretch="always")
        self.plot_top.add(vcut_container, stretch="always")

        # ---- Image figure ----
        self.fig = Figure(figsize=(6.8, 6.0), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel(r"$Q_{xy}\ (\mathrm{\AA^{-1}})$", fontsize=self.label_font)
        self.ax.set_ylabel(r"$Q_{z}\ (\mathrm{\AA^{-1}})$", fontsize=self.label_font)
        self.ax.tick_params(labelsize=self.tick_font)

        self.canvas = FigureCanvasTkAgg(self.fig, master=img_container)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        ctrl = tk.Frame(img_container)
        ctrl.pack(side=tk.BOTTOM, fill=tk.X, pady=(6, 0))

        self.sum_btn = tk.Checkbutton(ctrl, text="Sum over frames", variable=self.sum_frames,
                                      command=self.on_sum_toggle)
        self.sum_btn.pack(side=tk.LEFT)

        tk.Label(ctrl, text="Frame:").pack(side=tk.LEFT, padx=(10, 4))
        self.frame_slider = tk.Scale(
            ctrl, from_=0, to=0, resolution=1, orient=tk.HORIZONTAL,
            variable=self.frame_index, showvalue=True,
            command=lambda _=None: self.on_frame_change()
        )
        self.frame_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.status_var = tk.StringVar(value="Move mouse over image to see (Qxy, Qz, I).")
        status = tk.Label(img_container, textvariable=self.status_var, anchor="w")
        status.pack(side=tk.BOTTOM, fill=tk.X)

        # ---- Vertical linecut figure (right) ----
        self.v_fig = Figure(figsize=(4.2, 6.0), dpi=100)
        self.v_ax = self.v_fig.add_subplot(111)
        self.v_ax.set_xlabel(r"$Q_{z}\ (\mathrm{\AA^{-1}})$", fontsize=self.label_font)
        self.v_ax.set_ylabel("")  # CHANGED: remove vertical axis label
        self.v_ax.tick_params(labelsize=self.tick_font)
        self.v_ax.tick_params(axis="y", labelleft=False)
        self.v_ax.set_title("")
        self.v_fig.subplots_adjust(left=0.10, right=0.98, bottom=0.14, top=0.98)  # NEW/CHANGED

        self.v_canvas = FigureCanvasTkAgg(self.v_fig, master=vcut_container)
        self.v_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        (self._v_line,) = self.v_ax.plot([], [], marker="o", linestyle="-")

        # ---- Horizontal linecut figure (bottom) ----
        # CHANGED: make this shorter and leave more bottom margin so xlabel is visible
        self.h_fig = Figure(figsize=(11.2, 1.65), dpi=100)  # NEW/CHANGED (shorter)
        self.h_ax = self.h_fig.add_subplot(111)
        self.h_ax.set_xlabel(r"$Q_{xy}\ (\mathrm{\AA^{-1}})$", fontsize=self.label_font)
        self.h_ax.set_ylabel("")  # CHANGED: remove vertical axis label
        self.h_ax.tick_params(labelsize=self.tick_font)
        self.h_ax.tick_params(axis="y", labelleft=False)
        self.h_ax.set_title("")
        self.h_fig.subplots_adjust(left=0.06, right=0.995, bottom=0.38, top=0.98)  # NEW/CHANGED

        self.h_canvas = FigureCanvasTkAgg(self.h_fig, master=bottom_container)
        self.h_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        (self._h_line,) = self.h_ax.plot([], [], marker="s", linestyle="-")

        # Matplotlib events (hover + draggable ROI)
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self.canvas.mpl_connect("button_press_event", self._on_mouse_press)
        self.canvas.mpl_connect("button_release_event", self._on_mouse_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_drag)

        # NEW/CHANGED: set initial sash positions so bottom is shorter by default
        self.after(120, self._set_initial_sashes)

    def _set_initial_sashes(self):
        try:
            # Make bottom pane ~20% height initially
            total_h = self.plot_outer.winfo_height()
            if total_h > 100:
                self.plot_outer.sash_place(0, 0, int(total_h * 0.78))
            # Make right pane (vcut) ~20% width initially
            total_w = self.plot_top.winfo_width()
            if total_w > 100:
                self.plot_top.sash_place(0, int(total_w * 0.80), 0)
        except Exception:
            pass

    # ---------------- Hover readout ----------------
    def _on_mouse_move(self, event):
        if self.im is None or self.img_raw is None:
            return
        if event.inaxes != self.ax:
            self.status_var.set("Move mouse over image to see (Qxy, Qz, I).")
            return
        if event.xdata is None or event.ydata is None:
            return

        qxy_val = float(event.xdata)
        qz_val = float(event.ydata)
        img2d = self._current_display_image()
        if img2d is None or self.qxy is None or self.qz is None:
            return

        ix = int(np.argmin(np.abs(self.qxy - qxy_val)))
        iz = int(np.argmin(np.abs(self.qz  - qz_val)))
        if 0 <= iz < img2d.shape[0] and 0 <= ix < img2d.shape[1]:
            I = img2d[iz, ix]
            self.status_var.set(f"Qxy={self.qxy[ix]:.6g}  Qz={self.qz[iz]:.6g}  I={I:.6g}   (nearest pixel)")
        else:
            self.status_var.set(f"Qxy={qxy_val:.6g}  Qz={qz_val:.6g}")

    # ---------------- Draggable ROI ----------------
    def _roi_get(self, kind):
        vars_ = self.roi_v if kind == "v" else self.roi_h
        x0 = float(vars_["x0"].get()); z0 = float(vars_["z0"].get())
        w  = float(vars_["w"].get());  h  = float(vars_["h"].get())
        return x0, z0, w, h

    def _roi_set(self, kind, x0, z0, w, h, update_entries=True):
        if w < 0:
            x0, w = x0 + w, -w
        if h < 0:
            z0, h = z0 + h, -h
        vars_ = self.roi_v if kind == "v" else self.roi_h
        if update_entries:
            vars_["x0"].set(x0); vars_["z0"].set(z0); vars_["w"].set(w); vars_["h"].set(h)
        self._draw_roi_patch(kind)

    def _roi_hit_test(self, kind, x, y):
        if not self._roi_initialized:
            return False, None, None
        x0, z0, w, h = self._roi_get(kind)
        x1, z1 = x0 + w, z0 + h
        xmin, xmax = min(x0, x1), max(x0, x1)
        zmin, zmax = min(z0, z1), max(z0, z1)

        xspan = (self.ax.get_xlim()[1] - self.ax.get_xlim()[0]) or 1.0
        zspan = (self.ax.get_ylim()[1] - self.ax.get_ylim()[0]) or 1.0
        xtol = 0.02 * abs(xspan)
        ztol = 0.02 * abs(zspan)

        corners = {"bl": (xmin, zmin), "br": (xmax, zmin), "tl": (xmin, zmax), "tr": (xmax, zmax)}
        for name, (cx, cz) in corners.items():
            if abs(x - cx) <= xtol and abs(y - cz) <= ztol:
                return True, "resize", name

        if (xmin <= x <= xmax) and (zmin <= y <= zmax):
            return True, "move", None

        return False, None, None

    def _on_mouse_press(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        for kind in ("v", "h"):
            hit, mode, corner = self._roi_hit_test(kind, float(event.xdata), float(event.ydata))
            if hit:
                self._drag_roi_kind = kind
                self._drag_anchor = (float(event.xdata), float(event.ydata))
                self._drag_mode = mode
                self._drag_resize_corner = corner
                return

    def _on_mouse_release(self, event):
        if self._drag_roi_kind is None:
            return
        self._drag_roi_kind = None
        self._drag_anchor = None
        self._drag_mode = None
        self._drag_resize_corner = None
        self.update_linecuts()
        self.canvas.draw_idle()

    def _on_mouse_drag(self, event):
        if self._drag_roi_kind is None:
            return
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        kind = self._drag_roi_kind
        x0, z0, w, h = self._roi_get(kind)
        x1, z1 = x0 + w, z0 + h
        xmin, xmax = min(x0, x1), max(x0, x1)
        zmin, zmax = min(z0, z1), max(z0, z1)

        x, y = float(event.xdata), float(event.ydata)

        if self._drag_mode == "move":
            ax, ay = self._drag_anchor
            dx, dy = x - ax, y - ay
            self._drag_anchor = (x, y)
            xmin2, xmax2 = xmin + dx, xmax + dx
            zmin2, zmax2 = zmin + dy, zmax + dy
            self._roi_set(kind, xmin2, zmin2, xmax2 - xmin2, zmax2 - zmin2, update_entries=True)

        elif self._drag_mode == "resize":
            c = self._drag_resize_corner
            if c == "bl":
                xmin2, zmin2 = x, y; xmax2, zmax2 = xmax, zmax
            elif c == "br":
                xmin2, zmin2 = xmin, y; xmax2, zmax2 = x, zmax
            elif c == "tl":
                xmin2, zmin2 = x, zmin; xmax2, zmax2 = xmax, y
            else:  # tr
                xmin2, zmin2 = xmin, zmin; xmax2, zmax2 = x, y

            self._roi_set(kind, xmin2, zmin2, xmax2 - xmin2, zmax2 - zmin2, update_entries=True)

        self.canvas.draw_idle()

    def start_roi_draw(self, kind: str):
        if self.im is None or self.img_raw is None:
            messagebox.showinfo("No image", "Load a file first.")
            return

        if self._rect_selector is not None:
            try:
                self._rect_selector.set_active(False)
            except Exception:
                pass
            self._rect_selector = None

        def onselect(eclick, erelease):
            if eclick.xdata is None or eclick.ydata is None or erelease.xdata is None or erelease.ydata is None:
                return
            x0 = float(min(eclick.xdata, erelease.xdata))
            x1 = float(max(eclick.xdata, erelease.xdata))
            z0 = float(min(eclick.ydata, erelease.ydata))
            z1 = float(max(eclick.ydata, erelease.ydata))
            self._roi_initialized = True
            self._roi_set(kind, x0, z0, x1 - x0, z1 - z0, update_entries=True)

            if self._rect_selector is not None:
                try:
                    self._rect_selector.set_active(False)
                except Exception:
                    pass
                self._rect_selector = None

            self.update_linecuts()
            self.canvas.draw_idle()

        self._rect_selector = RectangleSelector(
            self.ax, onselect,
            useblit=True, interactive=False, button=[1],
            minspanx=1e-12, minspany=1e-12, spancoords="data"
        )
        self.status_var.set("Drag to draw ROI. Then drag inside to move, corners to resize.")

    def apply_roi_from_entries(self, kind: str):
        vars_ = self.roi_v if kind == "v" else self.roi_h
        x0 = float(vars_["x0"].get())
        z0 = float(vars_["z0"].get())
        w  = float(vars_["w"].get())
        h  = float(vars_["h"].get())
        self._roi_initialized = True
        self._roi_set(kind, x0, z0, w, h, update_entries=True)
        self.update_linecuts()
        self.canvas.draw_idle()

    def _draw_roi_patch(self, kind: str):
        vars_ = self.roi_v if kind == "v" else self.roi_h
        x0 = float(vars_["x0"].get()); z0 = float(vars_["z0"].get())
        w  = float(vars_["w"].get());  h  = float(vars_["h"].get())

        if kind == "v" and self.roi_v_patch is not None:
            try: self.roi_v_patch.remove()
            except Exception: pass
            self.roi_v_patch = None
        if kind == "h" and self.roi_h_patch is not None:
            try: self.roi_h_patch.remove()
            except Exception: pass
            self.roi_h_patch = None

        linestyle = "-" if kind == "v" else "--"
        patch = Rectangle((x0, z0), w, h, fill=False, linewidth=2.0, linestyle=linestyle)
        self.ax.add_patch(patch)
        if kind == "v":
            self.roi_v_patch = patch
        else:
            self.roi_h_patch = patch

    # ---------------- Linecuts ----------------
    def _safe_log10(self, y):
        y = np.asarray(y, float)
        out = np.full_like(y, np.nan, dtype=float)
        m = np.isfinite(y) & (y > 0)
        out[m] = np.log10(y[m])
        return out

    def update_linecuts(self):
        if self.img_raw is None or self.qxy is None or self.qz is None:
            return
        img2d = self._current_display_image()
        if img2d is None:
            return

        if not self._roi_initialized:
            x_min, x_max = float(np.min(self.qxy)), float(np.max(self.qxy))
            z_min, z_max = float(np.min(self.qz)), float(np.max(self.qz))
            x_span = x_max - x_min
            z_span = z_max - z_min

            self.roi_h["x0"].set(x_min + 0.10 * x_span)
            self.roi_h["w"].set(0.80 * x_span)
            self.roi_h["z0"].set(z_min + 0.40 * z_span)
            self.roi_h["h"].set(0.10 * z_span)

            self.roi_v["x0"].set(x_min + 0.45 * x_span)
            self.roi_v["w"].set(0.10 * x_span)
            self.roi_v["z0"].set(z_min + 0.10 * z_span)
            self.roi_v["h"].set(0.80 * z_span)

            self._roi_initialized = True
            self._draw_roi_patch("h")
            self._draw_roi_patch("v")

        # Horizontal ROI -> I(Qxy)
        xh0 = float(self.roi_h["x0"].get()); zh0 = float(self.roi_h["z0"].get())
        xh1 = xh0 + float(self.roi_h["w"].get())
        zh1 = zh0 + float(self.roi_h["h"].get())
        mask_xh = (self.qxy >= min(xh0, xh1)) & (self.qxy <= max(xh0, xh1))
        mask_zh = (self.qz  >= min(zh0, zh1)) & (self.qz  <= max(zh0, zh1))
        if np.any(mask_xh) and np.any(mask_zh):
            sub = img2d[np.ix_(mask_zh, mask_xh)]
            y_h = np.nanmean(sub, axis=0)
            x_h = self.qxy[mask_xh]
        else:
            x_h = np.array([]); y_h = np.array([])

        # Vertical ROI -> I(Qz)
        xv0 = float(self.roi_v["x0"].get()); zv0 = float(self.roi_v["z0"].get())
        xv1 = xv0 + float(self.roi_v["w"].get())
        zv1 = zv0 + float(self.roi_v["h"].get())
        mask_xv = (self.qxy >= min(xv0, xv1)) & (self.qxy <= max(xv0, xv1))
        mask_zv = (self.qz  >= min(zv0, zv1)) & (self.qz  <= max(zv0, zv1))
        if np.any(mask_xv) and np.any(mask_zv):
            sub = img2d[np.ix_(mask_zv, mask_xv)]
            y_v = np.nanmean(sub, axis=1)
            x_v = self.qz[mask_zv]
        else:
            x_v = np.array([]); y_v = np.array([])

        # Apply log10-Y toggles (y labels remain blank as requested)
        y_h_plot = self._safe_log10(y_h) if self.lc_log_h.get() else y_h
        y_v_plot = self._safe_log10(y_v) if self.lc_log_v.get() else y_v

        self._h_line.set_data(x_h, y_h_plot)
        self._v_line.set_data(x_v, y_v_plot)

        self.h_ax.relim(); self.h_ax.autoscale_view()
        self.v_ax.relim(); self.v_ax.autoscale_view()

        self.h_ax.set_title("")
        self.v_ax.set_title("")
        self.h_ax.set_ylabel("")  # ensure blank
        self.v_ax.set_ylabel("")  # ensure blank

        self.h_canvas.draw_idle()
        self.v_canvas.draw_idle()

    # ---------------- Background & frames ----------------
    def set_background_file(self):
        initial = self.current_folder or self.start_folder or None
        path = filedialog.askopenfilename(
            title="Select background .h5 file",
            initialdir=initial,
            filetypes=[("HDF5 files", "*.h5 *.hdf5"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            bg_raw, bg_qxy, bg_qz = _load_h5_raw(path)
            self.bg_file = path
            self.bg_raw = bg_raw
            self.bg_qxy = bg_qxy
            self.bg_qz = bg_qz
            self.bg_label.configure(text=os.path.splitext(os.path.basename(path))[0])
            self.reload_current()
        except Exception as e:
            messagebox.showerror("Background load failed", str(e))

    def _subtract_background(self, raw, qxy, qz):
        if not self.bg_enabled.get() or self.bg_raw is None:
            return raw
        if self.bg_qxy is None or self.bg_qz is None:
            raise ValueError("Background axes missing.")
        if self.bg_qxy.size != qxy.size or self.bg_qz.size != qz.size:
            raise ValueError("Background axes lengths do not match current file axes.")
        if not (np.allclose(self.bg_qxy, qxy, rtol=1e-6, atol=1e-12) and np.allclose(self.bg_qz, qz, rtol=1e-6, atol=1e-12)):
            raise ValueError("Background Q grids differ from current file.")

        scale = float(self.bg_scale.get())
        bg = self.bg_raw

        if raw.ndim == 2 and bg.ndim == 2:
            if raw.shape != bg.shape:
                raise ValueError("Background shape != image shape (2D).")
            return raw - scale * bg
        if raw.ndim == 3 and bg.ndim == 3:
            if raw.shape != bg.shape:
                raise ValueError("Background shape != image shape (3D).")
            return raw - scale * bg
        if raw.ndim == 3 and bg.ndim == 2:
            if raw.shape[1:] != bg.shape:
                raise ValueError("2D background shape != each frame shape.")
            return raw - scale * bg[None, :, :]
        raise ValueError("Incompatible background dimensionality for subtraction.")

    def _sync_frame_controls_visibility(self):
        is3d = (self.img_raw is not None and self.img_raw.ndim == 3)
        if not is3d:
            self.frame_slider.configure(state="disabled")
            self.sum_btn.configure(state="disabled")
            return
        self.sum_btn.configure(state="normal")
        self.frame_slider.configure(state="disabled" if self.sum_frames.get() else "normal")

    def on_frame_change(self):
        if self.img_raw is None or self.sum_frames.get():
            return
        self.update_plot()

    def on_sum_toggle(self):
        self._sync_frame_controls_visibility()
        self.update_plot()

    def _frame_navigation_allowed(self) -> bool:
        return (
            self.img_raw is not None and self.img_raw.ndim == 3
            and not self.sum_frames.get()
            and str(self.frame_slider.cget("state")) != "disabled"
        )

    def _step_frame(self, step: int):
        if not self._frame_navigation_allowed():
            return
        n = self.img_raw.shape[0]
        new_idx = int(self.frame_index.get()) + step
        new_idx = max(0, min(n - 1, new_idx))
        if new_idx != int(self.frame_index.get()):
            self.frame_index.set(new_idx)
            self.update_plot()

    def _on_key_left(self, _evt=None): self._step_frame(-1)
    def _on_key_right(self, _evt=None): self._step_frame(+1)
    def _on_key_left_fast(self, _evt=None): self._step_frame(-10)
    def _on_key_right_fast(self, _evt=None): self._step_frame(+10)

    # ---------------- File list ----------------
    def open_folder(self):
        initial = self.current_folder or self.start_folder or os.getcwd()

        folder = filedialog.askdirectory(
            title="Select folder containing .h5 files",
            initialdir=initial
        )
        if not folder:
            return

        self.current_folder = folder
        self.populate_file_list(folder)
    
    def refresh_file_list(self):
        """Reload file list from current or startup folder."""
        folder = self.current_folder or self.start_folder
        if folder is None:
            messagebox.showinfo("No folder", "No folder selected yet.")
            return

        self.populate_file_list(folder)

    def populate_file_list(self, folder):
        self.file_list.delete(0, tk.END)
        files = sorted([f for f in os.listdir(folder) if f.lower().endswith((".h5", ".hdf5"))])
        for f in files:
            self.file_list.insert(tk.END, f)
        if not files:
            messagebox.showinfo("No files", "No .h5/.hdf5 files found in that folder.")

    def on_select_file(self, _evt=None):
        folder = self.current_folder or self.start_folder
        if folder is None:
            return
        sel = self.file_list.curselection()
        if not sel:
            return
        fname = self.file_list.get(sel[0])
        self.load_file(os.path.join(folder, fname), keep_state=False)

    # ---------------- Load/display ----------------
    def reload_current(self):
        if self.current_file is None:
            return
        self.load_file(self.current_file, keep_state=True)

    def _drop_current_data(self):
        self.img_raw = None
        self.qxy = None
        self.qz = None
        if self.im is not None:
            try: self.im.remove()
            except Exception: pass
            self.im = None
        for p in (self.roi_v_patch, self.roi_h_patch):
            if p is not None:
                try: p.remove()
                except Exception: pass
        self.roi_v_patch = None
        self.roi_h_patch = None

        self.ax.clear()
        self.ax.set_xlabel(r"$Q_{xy}\ (\mathrm{\AA^{-1}})$", fontsize=self.label_font)
        self.ax.set_ylabel(r"$Q_{z}\ (\mathrm{\AA^{-1}})$", fontsize=self.label_font)
        self.ax.tick_params(labelsize=self.tick_font)
        gc.collect()

    def load_file(self, path, keep_state=False):
        self._drop_current_data()

        raw, qxy, qz = _load_h5_raw(path)
        raw = self._subtract_background(raw, qxy, qz)

        self.current_file = path
        self.img_raw, self.qxy, self.qz = raw, qxy, qz

        if not keep_state:
            self.frame_index.set(0)

        if self.img_raw.ndim == 3:
            n = self.img_raw.shape[0]
            self.frame_slider.configure(from_=0, to=max(0, n - 1))
        else:
            self.frame_slider.configure(from_=0, to=0)

        if not self._ranges_initialized:
            self.qxy_min_var.set(float(np.nanmin(qxy)))
            self.qxy_max_var.set(float(np.nanmax(qxy)))
            self.qz_min_var.set(float(np.nanmin(qz)))
            self.qz_max_var.set(float(np.nanmax(qz)))
            self._ranges_initialized = True

        self._sync_frame_controls_visibility()
        self._sync_slider_visibility()
        self._sync_entries_from_vars()

        self.redraw_full()
        self.update_linecuts()

    def _current_display_image(self):
        if self.img_raw is None:
            return None
        if self.img_raw.ndim == 2:
            return self.img_raw
        if self.sum_frames.get():
            return np.nansum(self.img_raw, axis=0)
        idx = int(self.frame_index.get())
        idx = max(0, min(idx, self.img_raw.shape[0] - 1))
        return self.img_raw[idx]

    # ---------------- scales ----------------
    def _sync_entries_from_vars(self):
        self.log_vmin_entry.delete(0, tk.END)
        self.log_vmin_entry.insert(0, f"{self.log_vmin.get():.4g}")
        self.log_vmax_entry.delete(0, tk.END)
        self.log_vmax_entry.insert(0, f"{self.log_vmax.get():.4g}")
        self.lin_vmin_entry.delete(0, tk.END)
        self.lin_vmin_entry.insert(0, f"{self.lin_vmin.get():.6g}")
        self.lin_vmax_entry.delete(0, tk.END)
        self.lin_vmax_entry.insert(0, f"{self.lin_vmax.get():.6g}")

    def _sync_slider_visibility(self):
        if self.scale_mode.get() == "log":
            if self.lin_frame.winfo_ismapped():
                self.lin_frame.pack_forget()
            if not self.log_frame.winfo_ismapped():
                self.log_frame.pack(fill=tk.X, pady=(0, 10))
        else:
            if self.log_frame.winfo_ismapped():
                self.log_frame.pack_forget()
            if not self.lin_frame.winfo_ismapped():
                self.lin_frame.pack(fill=tk.X, pady=(0, 10))

    def apply_log_entries(self):
        try:
            vmin = float(self.log_vmin_entry.get())
            vmax = float(self.log_vmax_entry.get())
            if vmax <= vmin:
                raise ValueError("log10(vmax) must be > log10(vmin)")
            self.log_vmin.set(vmin)
            self.log_vmax.set(vmax)
            self.update_plot()
        except Exception as e:
            messagebox.showerror("Invalid log values", str(e))

    def apply_lin_entries(self):
        try:
            vmin = float(self.lin_vmin_entry.get())
            vmax = float(self.lin_vmax_entry.get())
            if vmax <= vmin:
                raise ValueError("vmax must be > vmin")
            self.lin_vmin.set(vmin)
            self.lin_vmax.set(vmax)
            self.update_plot()
        except Exception as e:
            messagebox.showerror("Invalid linear values", str(e))

    def schedule_redraw(self):
        self._sync_entries_from_vars()
        if self._pending_redraw_id is not None:
            try:
                self.after_cancel(self._pending_redraw_id)
            except Exception:
                pass
        self._pending_redraw_id = self.after(self._redraw_delay_ms, self.update_plot)

    def _get_norm(self):
        if self.scale_mode.get() == "log":
            vmin = 10 ** float(self.log_vmin.get())
            vmax = 10 ** float(self.log_vmax.get())
            vmin = max(vmin, 1e-12)
            vmax = max(vmax, vmin * 1.001)
            return LogNorm(vmin=vmin, vmax=vmax)
        vmin = float(self.lin_vmin.get())
        vmax = float(self.lin_vmax.get())
        vmax = max(vmax, vmin + 1e-12)
        return Normalize(vmin=vmin, vmax=vmax)

    def _apply_ranges_to_axes(self):
        try:
            xmin = float(self.qxy_min_var.get())
            xmax = float(self.qxy_max_var.get())
            zmin = float(self.qz_min_var.get())
            zmax = float(self.qz_max_var.get())
            if xmax > xmin:
                self.ax.set_xlim(xmin, xmax)
            if zmax > zmin:
                self.ax.set_ylim(zmin, zmax)
        except Exception:
            pass

    def redraw_full(self):
        img2d = self._current_display_image()
        if img2d is None:
            return

        self.ax.clear()
        self.ax.set_xlabel(r"$Q_{xy}\ (\mathrm{\AA^{-1}})$", fontsize=self.label_font)
        self.ax.set_ylabel(r"$Q_{z}\ (\mathrm{\AA^{-1}})$", fontsize=self.label_font)
        self.ax.tick_params(labelsize=self.tick_font)

        if self.current_file:
            self.ax.set_title(os.path.splitext(os.path.basename(self.current_file))[0], fontsize=self.tick_font)
        else:
            self.ax.set_title("", fontsize=self.tick_font)

        extent = [float(self.qxy.min()), float(self.qxy.max()), float(self.qz.min()), float(self.qz.max())]
        self.im = self.ax.imshow(
            img2d,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap=self.cmap_var.get(),
            norm=self._get_norm(),
            interpolation="nearest",
        )

        self.ax.axvline(0, linestyle="--", linewidth=1.5, color="black")
        self.ax.axhline(0, linestyle="--", linewidth=1.5, color="black")

        if self._roi_initialized:
            self._draw_roi_patch("v")
            self._draw_roi_patch("h")

        self._apply_ranges_to_axes()
        self.canvas.draw_idle()

    def update_plot(self):
        self._pending_redraw_id = None
        self._sync_slider_visibility()
        self._sync_entries_from_vars()

        if self.im is None or self.img_raw is None:
            return

        img2d = self._current_display_image()
        self.im.set_data(img2d)
        self.im.set_cmap(self.cmap_var.get())
        self.im.set_norm(self._get_norm())

        if self.current_file:
            self.ax.set_title(os.path.splitext(os.path.basename(self.current_file))[0], fontsize=self.tick_font)
        else:
            self.ax.set_title("", fontsize=self.tick_font)

        self._apply_ranges_to_axes()
        self._sync_frame_controls_visibility()

        self.canvas.draw_idle()
        self.update_linecuts()

    def export_png(self):
        if self.im is None:
            messagebox.showinfo("Nothing to export", "Load a file first.")
            return

        default_name = "export.png"
        if self.current_file:
            default_name = os.path.splitext(os.path.basename(self.current_file))[0] + ".png"

        initial_dir = os.path.dirname(self.start_folder) if self.start_folder else os.getcwd()

        out = filedialog.asksaveasfilename(
            title="Export PNG",
            initialdir=initial_dir,
            defaultextension=".png",
            initialfile=default_name,
            filetypes=[("PNG", "*.png")]
        )
        if not out:
            return

        try:
            # ===== NEW PART: temporarily hide ROI patches =====
            hidden = []
            for p in (self.roi_v_patch, self.roi_h_patch):
                if p is not None:
                    hidden.append((p, p.get_visible()))
                    p.set_visible(False)

            self.canvas.draw_idle()

            self.fig.savefig(out, dpi=300, bbox_inches="tight")

            # restore
            for p, vis in hidden:
                p.set_visible(vis)

            self.canvas.draw_idle()
            # ================================================

            messagebox.showinfo("Exported", f"Saved:\n{out}")

        except Exception as e:
            messagebox.showerror("Export failed", str(e))



if __name__ == "__main__":
    start_folder = sys.argv[1] if len(sys.argv) > 1 else None
    app = GIXSViewer(start_folder=start_folder)
    app.mainloop()
