from __future__ import annotations

import numpy as np
import pyqtgraph
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets


pyqtgraph.setConfigOptions(imageAxisOrder='row-major')
pg.setConfigOptions(antialias=True)


class LiveCameraWindow(QtWidgets.QMainWindow):
	frame_received = QtCore.pyqtSignal(object)
	fft_received = QtCore.pyqtSignal(object)
	phase_plot_received = QtCore.pyqtSignal(object, object, object, object)
	intensity_plot_received = QtCore.pyqtSignal(object, object)

	def __init__(self, parent=None):
		super().__init__(parent)
		self.setWindowTitle("Allied Vision Live Camera")
		self.resize(1400, 800)
		self._camera_view_initialized = False
		self._fft_view_initialized = False
		self._camera_frame_count = 0
		self._fft_frame_count = 0
		self._recovered_initialized = False

		self._setup_ui()
		self._connect_internal_slots()
		self._set_default_states()

	def _build_perceptual_rainbow_colormap(self) -> pg.ColorMap:
		for name in ("CET-R4", "CET-L17", "viridis"):
			try:
				return pg.colormap.get(name)
			except Exception:
				pass
		positions = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=float)
		colors = np.array(
			[
				[68, 1, 84, 255],
				[59, 82, 139, 255],
				[33, 145, 140, 255],
				[94, 201, 98, 255],
				[253, 231, 37, 255],
				[255, 94, 77, 255],
			],
			dtype=np.ubyte,
		)
		return pg.ColorMap(pos=positions, color=colors)

	def _new_image_view(self) -> pg.ImageView:
		view = pg.ImageView(view=pg.PlotItem())
		view.ui.histogram.hide()
		view.ui.roiBtn.hide()
		view.ui.menuBtn.hide()
		view.getView().setAspectLocked(False)
		view.getView().invertY(False)
		view.getView().setMouseEnabled(x=False, y=False)
		view.getView().setMenuEnabled(False)
		return view

	def _setup_ui(self) -> None:
		self.start_button = QtWidgets.QPushButton("Start Live")
		self.stop_button = QtWidgets.QPushButton("Stop")
		self.auto_exposure_checkbox = QtWidgets.QCheckBox("Auto exposure")
		self.exposure_spinbox = QtWidgets.QDoubleSpinBox()
		self.exposure_spinbox.setDecimals(1)
		self.exposure_spinbox.setRange(10.0, 1_000_000.0)
		self.exposure_spinbox.setValue(1000.0)
		self.exposure_spinbox.setSuffix(" us")
		self.trigger_mode_combo = QtWidgets.QComboBox()
		self.trigger_mode_combo.addItem("FreeRun", "freerun")
		self.trigger_mode_combo.addItem("Line1", "line1")
		self.trigger_delay_edit = QtWidgets.QLineEdit("0.000")
		self.drop_frames_combo = QtWidgets.QComboBox()
		for n in [0, 1, 2, 3, 5, 10]:
			self.drop_frames_combo.addItem(str(n), n)
		self.pattern_counter_label = QtWidgets.QLabel("Pattern: - / -")
		self.pattern_counter_label.setMinimumWidth(180)

		self.status_label = QtWidgets.QLabel("Ready")
		self.live_roi_status_label = QtWidgets.QLabel("Live ROI: not set")
		self.live_roi_intensity_label = QtWidgets.QLabel("Live ROI sum intensity: n/a")
		self.fft_status_label = QtWidgets.QLabel("Waiting for frames...")
		self.recovered_field_status_label = QtWidgets.QLabel("Recovered field not available.")

		self.camera_view = self._new_image_view()
		self.fft_view = self._new_image_view()
		self.recovered_field_view = self._new_image_view()
		self.perceptual_rainbow_colormap = self._build_perceptual_rainbow_colormap()
		self.live_image_view = self.camera_view
		self.fft_image_view = self.fft_view
		self.recovered_field_view.setColorMap(self.perceptual_rainbow_colormap)

		self.live_roi_item = pg.RectROI([40, 40], [120, 120], pen=pg.mkPen(255, 255, 0, width=2), movable=True)
		self.live_roi_item.addScaleHandle([1, 1], [0, 0])
		self.camera_view.getView().addItem(self.live_roi_item)

		self.fft_roi_item = pg.RectROI([80, 80], [140, 140], pen=pg.mkPen(0, 255, 255, width=2), movable=True)
		self.fft_roi_item.addScaleHandle([1, 1], [0, 0])
		self.fft_view.getView().addItem(self.fft_roi_item)

		self.grating_x_edit = QtWidgets.QLineEdit("0")
		self.grating_x_value_label = QtWidgets.QLabel("Off")
		self.grating_y_edit = QtWidgets.QLineEdit("0")
		self.grating_y_value_label = QtWidgets.QLabel("Off")

		self.phase_plot_widget = pg.PlotWidget()
		self.phase_plot_widget.setTitle("Live Point Phases")
		self.phase_plot_widget.setLabel("left", "Phase (rad)")
		self.phase_plot_widget.setLabel("bottom", "Time (s)")
		self.phase_plot_widget.showGrid(x=True, y=True, alpha=0.25)
		self.phase_curve_phi1 = self.phase_plot_widget.plot(pen=pg.mkPen("g", width=2), name="phi1")
		self.phase_curve_phi2 = self.phase_plot_widget.plot(pen=pg.mkPen("m", width=2), name="phi2")
		self.phase_curve_dphi = self.phase_plot_widget.plot(pen=pg.mkPen("c", width=2), name="dphi_0_2pi")

		self.intensity_plot_widget = pg.PlotWidget()
		self.intensity_plot_widget.setTitle("Live ROI Intensity")
		self.intensity_plot_widget.setLabel("left", "Summed intensity")
		self.intensity_plot_widget.setLabel("bottom", "Time (s)")
		self.intensity_plot_widget.showGrid(x=True, y=True, alpha=0.25)
		self.intensity_curve = self.intensity_plot_widget.plot(pen=pg.mkPen("r", width=2), name="roi_sum")

		self.zernike_max_degree_combo = QtWidgets.QComboBox()
		for degree in range(0, 6):
			self.zernike_max_degree_combo.addItem(str(degree), degree)
		self.zernike_opt_iterations_spin = QtWidgets.QSpinBox()
		self.zernike_opt_iterations_spin.setRange(1, 1000)
		self.zernike_opt_iterations_spin.setValue(25)
		self.zernike_opt_step_spin = QtWidgets.QDoubleSpinBox()
		self.zernike_opt_step_spin.setRange(0.0001, 10.0)
		self.zernike_opt_step_spin.setValue(0.10)
		self.zernike_opt_start_button = QtWidgets.QPushButton("Start Optimizer")
		self.zernike_opt_stop_button = QtWidgets.QPushButton("Stop Optimizer")
		self.zernike_status_label = QtWidgets.QLabel("Active Zernike terms: 0")
		self.zernike_auto_optimize_checkbox = QtWidgets.QCheckBox("Continuous optimize (update every PLM pattern change)")
		self.zernike_opt_status_label = QtWidgets.QLabel("Optimizer: idle")

		self.pattern_path_edit = QtWidgets.QLineEdit()
		self.pattern_browse_button = QtWidgets.QPushButton("Browse")
		self.pattern_load_button = QtWidgets.QPushButton("Load .npy")
		self.pattern_start_button = QtWidgets.QPushButton("Start PLM Loop")
		self.pattern_stop_button = QtWidgets.QPushButton("Stop PLM Loop")
		self.pattern_frame_ms_spin = QtWidgets.QDoubleSpinBox()
		self.pattern_frame_ms_spin.setRange(1.0, 5000.0)
		self.pattern_frame_ms_spin.setDecimals(1)
		self.pattern_frame_ms_spin.setValue(100.0)
		self.pattern_frame_ms_spin.setSuffix(" ms")
		self.pattern_status_label = QtWidgets.QLabel("Load a .npy pattern stack to start PLM playback.")

		self.channel_matrix_status_label = QtWidgets.QLabel("Channel matrix: waiting for target modes and recovered modes.")
		self.channel_matrix_xt_label = QtWidgets.QLabel("XT (dB): n/a")
		self.channel_matrix_widget = pg.PlotWidget()
		self.channel_matrix_widget.setTitle("Channel Matrix")
		self.channel_matrix_widget.setLabel("left", "Target mode index")
		self.channel_matrix_widget.setLabel("bottom", "Recovered mode index")
		self.channel_matrix_image_item = pg.ImageItem()
		self.channel_matrix_widget.addItem(self.channel_matrix_image_item)

		self.pattern_gallery_status_label = QtWidgets.QLabel("Load PLM patterns to start gallery capture.")
		self.pattern_gallery_scroll = QtWidgets.QScrollArea()
		self.pattern_gallery_scroll.setWidgetResizable(True)
		self.pattern_gallery_container = QtWidgets.QWidget()
		self.pattern_gallery_grid = QtWidgets.QGridLayout(self.pattern_gallery_container)
		self.pattern_gallery_scroll.setWidget(self.pattern_gallery_container)

		live_panel = QtWidgets.QVBoxLayout()
		live_panel.addWidget(QtWidgets.QLabel("Live Camera"))
		live_panel.addWidget(self.camera_view)
		live_panel.addWidget(self.status_label)
		live_panel.addWidget(self.live_roi_status_label)
		live_panel.addWidget(self.live_roi_intensity_label)

		grating_x_row = QtWidgets.QHBoxLayout()
		grating_x_row.addWidget(QtWidgets.QLabel("X grating period (px):"))
		grating_x_row.addWidget(self.grating_x_edit)
		grating_x_row.addWidget(self.grating_x_value_label)
		grating_x_row.addStretch(1)
		live_panel.addLayout(grating_x_row)

		grating_y_row = QtWidgets.QHBoxLayout()
		grating_y_row.addWidget(QtWidgets.QLabel("Y grating period (px):"))
		grating_y_row.addWidget(self.grating_y_edit)
		grating_y_row.addWidget(self.grating_y_value_label)
		grating_y_row.addStretch(1)
		live_panel.addLayout(grating_y_row)

		fft_panel = QtWidgets.QVBoxLayout()
		fft_panel.addWidget(QtWidgets.QLabel("2D FFT Magnitude"))
		fft_panel.addWidget(self.fft_view)
		fft_panel.addWidget(self.fft_status_label)

		display_row = QtWidgets.QHBoxLayout()
		display_row.addLayout(live_panel, 1)
		display_row.addLayout(fft_panel, 1)

		live_fft_tab = QtWidgets.QWidget()
		live_fft_layout = QtWidgets.QVBoxLayout(live_fft_tab)
		live_fft_layout.addLayout(display_row)

		phase_plot_tab = QtWidgets.QWidget()
		phase_plot_layout = QtWidgets.QVBoxLayout(phase_plot_tab)
		phase_plot_layout.addWidget(self.phase_plot_widget)

		intensity_plot_tab = QtWidgets.QWidget()
		intensity_plot_layout = QtWidgets.QVBoxLayout(intensity_plot_tab)
		intensity_plot_layout.addWidget(self.intensity_plot_widget)

		zernike_tab = QtWidgets.QWidget()
		zernike_layout = QtWidgets.QVBoxLayout(zernike_tab)
		zernike_layout.addWidget(QtWidgets.QLabel("Zernike polynomial coefficients (radians), radial degree n ≤ 5"))
		zernike_degree_row = QtWidgets.QHBoxLayout()
		zernike_degree_row.addWidget(QtWidgets.QLabel("Show terms up to radial degree n:"))
		zernike_degree_row.addWidget(self.zernike_max_degree_combo)
		zernike_degree_row.addStretch(1)
		zernike_layout.addLayout(zernike_degree_row)
		zernike_layout.addWidget(self.zernike_status_label)
		zernike_opt_row = QtWidgets.QHBoxLayout()
		zernike_opt_row.addWidget(QtWidgets.QLabel("Iterations:"))
		zernike_opt_row.addWidget(self.zernike_opt_iterations_spin)
		zernike_opt_row.addWidget(QtWidgets.QLabel("Step (rad):"))
		zernike_opt_row.addWidget(self.zernike_opt_step_spin)
		zernike_opt_row.addWidget(self.zernike_opt_start_button)
		zernike_opt_row.addWidget(self.zernike_opt_stop_button)
		zernike_opt_row.addStretch(1)
		zernike_layout.addLayout(zernike_opt_row)
		zernike_layout.addWidget(self.zernike_auto_optimize_checkbox)
		zernike_layout.addWidget(self.zernike_opt_status_label)
		zernike_layout.addStretch(1)

		pattern_tab = QtWidgets.QWidget()
		pattern_layout = QtWidgets.QVBoxLayout(pattern_tab)
		file_row = QtWidgets.QHBoxLayout()
		file_row.addWidget(QtWidgets.QLabel("Pattern file:"))
		file_row.addWidget(self.pattern_path_edit, 1)
		file_row.addWidget(self.pattern_browse_button)
		file_row.addWidget(self.pattern_load_button)
		pattern_layout.addLayout(file_row)
		ctrl_row = QtWidgets.QHBoxLayout()
		ctrl_row.addWidget(self.pattern_start_button)
		ctrl_row.addWidget(self.pattern_stop_button)
		ctrl_row.addSpacing(12)
		ctrl_row.addWidget(QtWidgets.QLabel("Frame interval:"))
		ctrl_row.addWidget(self.pattern_frame_ms_spin)
		ctrl_row.addStretch(1)
		pattern_layout.addLayout(ctrl_row)
		pattern_layout.addWidget(self.pattern_status_label)
		pattern_layout.addStretch(1)

		recovered_tab = QtWidgets.QWidget()
		recovered_layout = QtWidgets.QVBoxLayout(recovered_tab)
		recovered_layout.addWidget(QtWidgets.QLabel("Recovered Field (Off-Axis Holography)"))
		recovered_layout.addWidget(self.recovered_field_view)
		recovered_layout.addWidget(self.recovered_field_status_label)

		pattern_gallery_tab = QtWidgets.QWidget()
		pattern_gallery_layout = QtWidgets.QVBoxLayout(pattern_gallery_tab)
		pattern_gallery_layout.addWidget(self.pattern_gallery_status_label)
		pattern_gallery_layout.addWidget(self.pattern_gallery_scroll, 1)

		channel_matrix_tab = QtWidgets.QWidget()
		channel_matrix_layout = QtWidgets.QVBoxLayout(channel_matrix_tab)
		channel_matrix_layout.addWidget(self.channel_matrix_status_label)
		channel_matrix_layout.addWidget(self.channel_matrix_xt_label)
		channel_matrix_layout.addWidget(self.channel_matrix_widget, 1)

		self.tab_widget = QtWidgets.QTabWidget()
		self.tab_widget.addTab(live_fft_tab, "Live + FFT")
		self.tab_widget.addTab(phase_plot_tab, "Phase Plot")
		self.tab_widget.addTab(intensity_plot_tab, "Intensity Plot")
		self.tab_widget.addTab(zernike_tab, "Zernike Control")
		self.tab_widget.addTab(pattern_tab, "PLM Patterns")

		self.right_tab_widget = QtWidgets.QTabWidget()
		self.right_tab_widget.addTab(recovered_tab, "Recovered Field")
		self.right_tab_widget.addTab(pattern_gallery_tab, "Pattern Gallery")
		self.right_tab_widget.addTab(channel_matrix_tab, "Channel Matrix")

		controls = QtWidgets.QHBoxLayout()
		controls.setContentsMargins(0, 0, 0, 0)
		controls.setSpacing(8)
		controls.addWidget(self.start_button)
		controls.addWidget(self.stop_button)
		controls.addSpacing(20)
		controls.addWidget(self.auto_exposure_checkbox)
		controls.addWidget(QtWidgets.QLabel("Manual exposure:"))
		controls.addWidget(self.exposure_spinbox)
		controls.addSpacing(20)
		controls.addWidget(QtWidgets.QLabel("Trigger:"))
		controls.addWidget(self.trigger_mode_combo)
		controls.addWidget(QtWidgets.QLabel("Delay (ms):"))
		controls.addWidget(self.trigger_delay_edit)
		controls.addWidget(QtWidgets.QLabel("Drop frames after pattern:"))
		controls.addWidget(self.drop_frames_combo)
		controls.addStretch(1)
		controls.addWidget(self.pattern_counter_label)

		root = QtWidgets.QWidget()
		root_layout = QtWidgets.QVBoxLayout(root)
		root_layout.setContentsMargins(8, 6, 8, 8)
		root_layout.setSpacing(6)
		root_layout.addLayout(controls)

		self.main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
		self.main_splitter.addWidget(self.tab_widget)
		self.main_splitter.addWidget(self.right_tab_widget)
		self.main_splitter.setChildrenCollapsible(False)
		self.main_splitter.setStretchFactor(0, 1)
		self.main_splitter.setStretchFactor(1, 1)
		self.main_splitter.setSizes([700, 700])
		root_layout.addWidget(self.main_splitter, 1)

		self.setCentralWidget(root)

	def _set_default_states(self) -> None:
		self.stop_button.setEnabled(False)
		self.pattern_stop_button.setEnabled(False)

	def _connect_internal_slots(self) -> None:
		self.frame_received.connect(self._on_frame_received)
		self.fft_received.connect(self._on_fft_received)
		self.phase_plot_received.connect(self._on_phase_plot_received)
		self.intensity_plot_received.connect(self._on_intensity_plot_received)
		self.live_roi_item.sigRegionChanged.connect(self._on_live_roi_changed)

	def connect_worker(self, worker) -> None:
		worker.frame_ready.connect(self.frame_received.emit, QtCore.Qt.QueuedConnection)
		worker.fft_ready.connect(self.fft_received.emit, QtCore.Qt.QueuedConnection)
		worker.phase_plot_ready.connect(self.phase_plot_received.emit, QtCore.Qt.QueuedConnection)
		worker.intensity_plot_ready.connect(self.intensity_plot_received.emit, QtCore.Qt.QueuedConnection)

	@QtCore.pyqtSlot(object)
	def _on_frame_received(self, frame_np: np.ndarray) -> None:
		frame = np.ascontiguousarray(frame_np, dtype=np.float32)
		if frame.ndim == 3:
			frame = frame[..., 0]
		self.camera_view.setImage(frame, autoLevels=False, autoRange=False, autoHistogramRange=False)
		self._camera_frame_count += 1
		if not self._camera_view_initialized:
			h, w = frame.shape[:2]
			vmin = float(np.min(frame))
			vmax = float(np.max(frame))
			if vmax <= vmin:
				vmax = vmin + 1.0
			self.camera_view.getImageItem().setLevels((vmin, vmax))
			self.camera_view.getView().setRange(xRange=(0, w), yRange=(0, h), padding=0.0)
			self.live_roi_item.setPos((max(0.0, w * 0.35), max(0.0, h * 0.35)))
			self.live_roi_item.setSize((max(20.0, w * 0.2), max(20.0, h * 0.2)))
			self._camera_view_initialized = True
		elif self._camera_frame_count % 30 == 0:
			vmin = float(np.min(frame))
			vmax = float(np.max(frame))
			if vmax <= vmin:
				vmax = vmin + 1.0
			self.camera_view.getImageItem().setLevels((vmin, vmax))
		self.status_label.setText("Live frame updated")

	@QtCore.pyqtSlot(object)
	def _on_fft_received(self, fft_np: np.ndarray) -> None:
		fft_img = np.ascontiguousarray(fft_np, dtype=np.float32)
		if fft_img.ndim == 3:
			fft_img = fft_img[..., 0]
		self.fft_view.setImage(fft_img, autoLevels=False, autoRange=False, autoHistogramRange=False)
		self._fft_frame_count += 1
		if not self._fft_view_initialized:
			h, w = fft_img.shape[:2]
			vmin = float(np.min(fft_img))
			vmax = float(np.max(fft_img))
			if vmax <= vmin:
				vmax = vmin + 1.0
			self.fft_view.getImageItem().setLevels((vmin, vmax))
			self.fft_view.getView().setRange(xRange=(0, w), yRange=(0, h), padding=0.0)
			self.fft_roi_item.setPos((max(0.0, w * 0.55), max(0.0, h * 0.30)))
			self.fft_roi_item.setSize((max(20.0, w * 0.18), max(20.0, h * 0.18)))
			self._fft_view_initialized = True
		elif self._fft_frame_count % 30 == 0:
			vmin = float(np.min(fft_img))
			vmax = float(np.max(fft_img))
			if vmax <= vmin:
				vmax = vmin + 1.0
			self.fft_view.getImageItem().setLevels((vmin, vmax))
		self.fft_status_label.setText("FFT updated")

	def set_recovered_field_image(self, image_np: np.ndarray, status_text: str = "Recovered field updated") -> None:
		if image_np.ndim == 3:
			if image_np.shape[2] in (3, 4):
				image = np.ascontiguousarray(image_np)
			else:
				image = np.ascontiguousarray(image_np[..., 0], dtype=np.float32)
		else:
			image = np.ascontiguousarray(image_np, dtype=np.float32)
		if not self._recovered_initialized:
			self.recovered_field_view.setImage(image, autoLevels=True, autoRange=True, autoHistogramRange=True)
			self._recovered_initialized = True
		else:
			self.recovered_field_view.setImage(image, autoLevels=False, autoRange=False, autoHistogramRange=False)
		self.recovered_field_status_label.setText(status_text)

	def _on_live_roi_changed(self) -> None:
		x0, y0, x1, y1 = self.get_live_roi_bounds()
		self.live_roi_status_label.setText(f"Live ROI: x[{x0}:{x1}] y[{y0}:{y1}]")

	def get_live_roi_bounds(self) -> tuple[int, int, int, int]:
		pos = self.live_roi_item.pos()
		size = self.live_roi_item.size()
		x0 = int(round(float(pos.x())))
		y0 = int(round(float(pos.y())))
		x1 = int(round(float(pos.x() + size.x())))
		y1 = int(round(float(pos.y() + size.y())))
		return x0, y0, x1, y1

	def get_fft_roi_bounds(self) -> tuple[int, int, int, int]:
		pos = self.fft_roi_item.pos()
		size = self.fft_roi_item.size()
		x0 = int(round(float(pos.x())))
		y0 = int(round(float(pos.y())))
		x1 = int(round(float(pos.x() + size.x())))
		y1 = int(round(float(pos.y() + size.y())))
		return x0, y0, x1, y1

	@QtCore.pyqtSlot(object, object, object, object)
	def _on_phase_plot_received(
		self,
		time_s_np: np.ndarray,
		phi1_np: np.ndarray,
		phi2_np: np.ndarray,
		dphi_np: np.ndarray,
	) -> None:
		time_s = np.asarray(time_s_np, dtype=np.float32)
		phi1 = np.asarray(phi1_np, dtype=np.float32)
		phi2 = np.asarray(phi2_np, dtype=np.float32)
		dphi = np.asarray(dphi_np, dtype=np.float32)
		self.phase_curve_phi1.setData(time_s, phi1)
		self.phase_curve_phi2.setData(time_s, phi2)
		self.phase_curve_dphi.setData(time_s, dphi)

	@QtCore.pyqtSlot(object, object)
	def _on_intensity_plot_received(self, time_s_np: np.ndarray, intensity_np: np.ndarray) -> None:
		time_s = np.asarray(time_s_np, dtype=np.float32)
		intensity = np.asarray(intensity_np, dtype=np.float32)
		self.intensity_curve.setData(time_s, intensity)

