import sys
import importlib
import threading
from pathlib import Path
from time import perf_counter
from typing import Optional

import numpy as np
from PIL import Image
from vmbpy import PixelFormat, VmbSystem

# Ensure local package imports work when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

try:
	QtCore = importlib.import_module("PyQt5.QtCore")
	QtGui = importlib.import_module("PyQt5.QtGui")
	QtWidgets = importlib.import_module("PyQt5.QtWidgets")

	Qt = QtCore.Qt
	QRect = QtCore.QRect
	QTimer = QtCore.QTimer
	QImage = QtGui.QImage
	QPainter = QtGui.QPainter
	QPen = QtGui.QPen
	QPixmap = QtGui.QPixmap
	QApplication = QtWidgets.QApplication
	QCheckBox = QtWidgets.QCheckBox
	QDoubleSpinBox = QtWidgets.QDoubleSpinBox
	QFileDialog = QtWidgets.QFileDialog
	QHBoxLayout = QtWidgets.QHBoxLayout
	QLabel = QtWidgets.QLabel
	QLineEdit = QtWidgets.QLineEdit
	QMainWindow = QtWidgets.QMainWindow
	QPushButton = QtWidgets.QPushButton
	QRadioButton = QtWidgets.QRadioButton
	QSpinBox = QtWidgets.QSpinBox
	QTabWidget = QtWidgets.QTabWidget
	QButtonGroup = QtWidgets.QButtonGroup
	QVBoxLayout = QtWidgets.QVBoxLayout
	QWidget = QtWidgets.QWidget
except ImportError as exc:
	raise ImportError(
		"PyQt5 is required. Install with: pip install PyQt5"
	) from exc

from ti_plm import PLM
from ti_plm.display import EventLoopExit, ImageWindow


class RoiSelectableImageLabel(QLabel):
	def __init__(self, text: str = "", parent=None):
		super().__init__(text, parent)
		self.setMouseTracking(True)
		self._image_shape: Optional[tuple[int, int]] = None
		self._display_qimage: Optional[QImage] = None
		self._dragging = False
		self._drag_start = None
		self._drag_end = None
		self._roi_rect: Optional[tuple[int, int, int, int]] = None
		self.on_roi_changed = None

	def set_image_array(self, image: np.ndarray) -> None:
		self._image_shape = (int(image.shape[0]), int(image.shape[1]))
		self._display_qimage = QImage(
			image.data,
			int(image.shape[1]),
			int(image.shape[0]),
			int(image.shape[1]),
			QImage.Format_Grayscale8,
		).copy()
		self._refresh_display()

	def get_roi_rect(self) -> Optional[tuple[int, int, int, int]]:
		return self._roi_rect

	def clear_roi(self) -> None:
		self._roi_rect = None
		self._drag_start = None
		self._drag_end = None
		self._refresh_display()
		if self.on_roi_changed is not None:
			self.on_roi_changed(self._roi_rect)

	def resizeEvent(self, event) -> None:
		super().resizeEvent(event)
		self._refresh_display()

	def mousePressEvent(self, event) -> None:
		if event.button() != Qt.LeftButton:
			super().mousePressEvent(event)
			return
		mapped = self._label_pos_to_image_pos(event.pos().x(), event.pos().y())
		if mapped is None:
			return
		self._dragging = True
		self._drag_start = mapped
		self._drag_end = mapped
		self._refresh_display()

	def mouseMoveEvent(self, event) -> None:
		if not self._dragging:
			super().mouseMoveEvent(event)
			return
		mapped = self._label_pos_to_image_pos(event.pos().x(), event.pos().y())
		if mapped is None:
			return
		self._drag_end = mapped
		self._refresh_display()

	def mouseReleaseEvent(self, event) -> None:
		if event.button() != Qt.LeftButton:
			super().mouseReleaseEvent(event)
			return
		if not self._dragging:
			return
		self._dragging = False
		mapped = self._label_pos_to_image_pos(event.pos().x(), event.pos().y())
		if mapped is not None:
			self._drag_end = mapped
		self._roi_rect = self._normalized_roi_rect(self._drag_start, self._drag_end)
		self._refresh_display()
		if self.on_roi_changed is not None:
			self.on_roi_changed(self._roi_rect)

	def _refresh_display(self) -> None:
		if self._display_qimage is None:
			return
		base_pixmap = QPixmap.fromImage(self._display_qimage)
		scaled = base_pixmap.scaled(
			self.size(),
			Qt.KeepAspectRatio,
			Qt.SmoothTransformation,
		)
		display = QPixmap(self.size())
		display.fill(Qt.black)
		painter = QPainter(display)
		x_offset = int((self.width() - scaled.width()) / 2)
		y_offset = int((self.height() - scaled.height()) / 2)
		painter.drawPixmap(x_offset, y_offset, scaled)

		rect_to_draw = None
		if self._dragging:
			rect_to_draw = self._normalized_roi_rect(self._drag_start, self._drag_end)
		elif self._roi_rect is not None:
			rect_to_draw = self._roi_rect

		if rect_to_draw is not None and self._image_shape is not None:
			label_rect = self._image_rect_to_label_rect(rect_to_draw)
			pen = QPen(Qt.red)
			pen.setWidth(2)
			painter.setPen(pen)
			painter.drawRect(label_rect)

		painter.end()
		self.setPixmap(display)

	def _image_display_geometry(self):
		if self._image_shape is None:
			return None
		image_h, image_w = self._image_shape
		if image_h <= 0 or image_w <= 0 or self.width() <= 0 or self.height() <= 0:
			return None
		scale = min(self.width() / image_w, self.height() / image_h)
		pix_w = image_w * scale
		pix_h = image_h * scale
		x_offset = (self.width() - pix_w) / 2.0
		y_offset = (self.height() - pix_h) / 2.0
		return image_w, image_h, scale, x_offset, y_offset, pix_w, pix_h

	def _label_pos_to_image_pos(self, x: int, y: int):
		geom = self._image_display_geometry()
		if geom is None:
			return None
		image_w, image_h, scale, x_offset, y_offset, pix_w, pix_h = geom
		if x < x_offset or y < y_offset or x >= x_offset + pix_w or y >= y_offset + pix_h:
			return None
		ix = int((x - x_offset) / scale)
		iy = int((y - y_offset) / scale)
		ix = max(0, min(image_w - 1, ix))
		iy = max(0, min(image_h - 1, iy))
		return ix, iy

	def _image_rect_to_label_rect(self, rect: tuple[int, int, int, int]) -> QRect:
		geom = self._image_display_geometry()
		if geom is None:
			return QRect()
		_, _, scale, x_offset, y_offset, _, _ = geom
		x0, y0, x1, y1 = rect
		left = int(round(x_offset + x0 * scale))
		top = int(round(y_offset + y0 * scale))
		right = int(round(x_offset + x1 * scale))
		bottom = int(round(y_offset + y1 * scale))
		return QRect(left, top, max(1, right - left), max(1, bottom - top))

	def _normalized_roi_rect(self, start, end) -> Optional[tuple[int, int, int, int]]:
		if start is None or end is None or self._image_shape is None:
			return None
		x0, y0 = start
		x1, y1 = end
		left = min(x0, x1)
		right = max(x0, x1)
		top = min(y0, y1)
		bottom = max(y0, y1)
		if right - left < 2 or bottom - top < 2:
			return None
		image_h, image_w = self._image_shape
		left = max(0, min(image_w - 1, left))
		right = max(1, min(image_w, right + 1))
		top = max(0, min(image_h - 1, top))
		bottom = max(1, min(image_h, bottom + 1))
		if right <= left or bottom <= top:
			return None
		return left, top, right, bottom


class LiveCameraWindow(QMainWindow):
	plm_counter_updated = QtCore.pyqtSignal(int, int)
	plm_loop_finished = QtCore.pyqtSignal(str)

	def __init__(self, cam):
		super().__init__()
		self.cam = cam
		self.setWindowTitle("Allied Vision Live Camera")
		self.resize(1400, 800)

		self.image_label = QLabel("No frame yet")
		self.image_label.setAlignment(Qt.AlignCenter)
		self.image_label.setMinimumSize(640, 480)

		self.start_button = QPushButton("Start Live")
		self.stop_button = QPushButton("Stop")
		self.stop_button.setEnabled(False)

		self.auto_exposure_checkbox = QCheckBox("Auto exposure")
		self.exposure_spinbox = QDoubleSpinBox()
		self.exposure_spinbox.setDecimals(1)
		self.exposure_spinbox.setRange(10.0, 1_000_000.0)
		self.exposure_spinbox.setSingleStep(100.0)
		self.exposure_spinbox.setSuffix(" us")

		self.status_label = QLabel()
		self.status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

		self.fft_image_label = RoiSelectableImageLabel("No FFT yet")
		self.fft_image_label.setAlignment(Qt.AlignCenter)
		self.fft_image_label.setMinimumSize(640, 480)
		self.fft_image_label.on_roi_changed = self._on_fft_roi_changed

		self.fft_status_label = QLabel("Waiting for frames...")

		self.recovered_field_label = QLabel("Select FFT ROI to recover field")
		self.recovered_field_label.setAlignment(Qt.AlignCenter)
		self.recovered_field_label.setMinimumSize(960, 640)
		self.recovered_field_status_label = QLabel("Recovered field not available.")

		self.basic_recovery_radio = QRadioButton("Basic")
		self.stabilized_recovery_radio = QRadioButton("Stabilized")
		self.stabilized_recovery_radio.setChecked(True)
		self.recovery_mode_group = QButtonGroup(self)
		self.recovery_mode_group.addButton(self.basic_recovery_radio)
		self.recovery_mode_group.addButton(self.stabilized_recovery_radio)

		self.pattern_path_edit = QLineEdit()
		self.pattern_path_edit.setPlaceholderText("Select .npy phase pattern file")
		self.pattern_browse_button = QPushButton("Browse")
		self.pattern_load_button = QPushButton("Load .npy")

		self.plm_catalog_edit = QLineEdit("p67nirtemp")
		self.plm_monitor_spin = QSpinBox()
		self.plm_monitor_spin.setRange(-8, 8)
		self.plm_monitor_spin.setValue(1)

		self.pattern_interval_spin = QDoubleSpinBox()
		self.pattern_interval_spin.setDecimals(0)
		self.pattern_interval_spin.setRange(10.0, 5000.0)
		self.pattern_interval_spin.setSingleStep(10.0)
		self.pattern_interval_spin.setValue(300.0)
		self.pattern_interval_spin.setSuffix(" ms")

		self.pattern_start_button = QPushButton("Start PLM Loop")
		self.pattern_stop_button = QPushButton("Stop PLM Loop")
		self.pattern_stop_button.setEnabled(False)

		self.pattern_status_label = QLabel("Load a .npy pattern stack to start PLM playback.")
		self.pattern_info_label = QLabel("")
		self.pattern_info_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

		self.pattern_counter_label = QLabel("Pattern: - / -")
		self.pattern_counter_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
		self.pattern_counter_label.setMinimumWidth(180)

		live_panel = QVBoxLayout()
		live_panel.addWidget(QLabel("Live Camera"))
		live_panel.addWidget(self.image_label)
		live_panel.addWidget(self.status_label)

		fft_panel = QVBoxLayout()
		fft_panel.addWidget(QLabel("2D FFT Magnitude"))
		fft_panel.addWidget(self.fft_image_label)
		fft_panel.addWidget(self.fft_status_label)

		display_row = QHBoxLayout()
		display_row.addLayout(live_panel, 1)
		display_row.addLayout(fft_panel, 1)

		live_fft_tab = QWidget()
		live_fft_layout = QVBoxLayout()
		live_fft_layout.addLayout(display_row)
		live_fft_tab.setLayout(live_fft_layout)

		recovered_tab = QWidget()
		recovered_layout = QVBoxLayout()
		recovered_layout.addWidget(QLabel("Recovered Field (Off-Axis Holography)"))
		recovery_mode_row = QHBoxLayout()
		recovery_mode_row.addWidget(QLabel("Recovery mode:"))
		recovery_mode_row.addWidget(self.basic_recovery_radio)
		recovery_mode_row.addWidget(self.stabilized_recovery_radio)
		recovery_mode_row.addStretch(1)
		recovered_layout.addLayout(recovery_mode_row)
		recovered_layout.addWidget(QLabel("Phase (RGB), Brightness Weighted by Amplitude"))
		recovered_layout.addWidget(self.recovered_field_label)
		recovered_layout.addWidget(self.recovered_field_status_label)
		recovered_tab.setLayout(recovered_layout)

		pattern_tab = QWidget()
		pattern_layout = QVBoxLayout()

		file_row = QHBoxLayout()
		file_row.addWidget(QLabel("Pattern file:"))
		file_row.addWidget(self.pattern_path_edit, 1)
		file_row.addWidget(self.pattern_browse_button)
		file_row.addWidget(self.pattern_load_button)

		config_row = QHBoxLayout()
		config_row.addWidget(QLabel("PLM catalog:"))
		config_row.addWidget(self.plm_catalog_edit)
		config_row.addSpacing(12)
		config_row.addWidget(QLabel("Monitor:"))
		config_row.addWidget(self.plm_monitor_spin)
		config_row.addSpacing(12)
		config_row.addWidget(QLabel("Frame interval:"))
		config_row.addWidget(self.pattern_interval_spin)
		config_row.addStretch(1)

		control_row = QHBoxLayout()
		control_row.addWidget(self.pattern_start_button)
		control_row.addWidget(self.pattern_stop_button)
		control_row.addStretch(1)

		pattern_layout.addLayout(file_row)
		pattern_layout.addLayout(config_row)
		pattern_layout.addLayout(control_row)
		pattern_layout.addWidget(self.pattern_info_label)
		pattern_layout.addWidget(self.pattern_status_label)
		pattern_layout.addStretch(1)
		pattern_tab.setLayout(pattern_layout)

		self.tab_widget = QTabWidget()
		self.tab_widget.addTab(live_fft_tab, "Live + FFT")
		self.tab_widget.addTab(recovered_tab, "Recovered Field")
		self.tab_widget.addTab(pattern_tab, "PLM Patterns")

		controls = QHBoxLayout()
		controls.addWidget(self.start_button)
		controls.addWidget(self.stop_button)
		controls.addSpacing(20)
		controls.addWidget(self.auto_exposure_checkbox)
		controls.addWidget(QLabel("Manual exposure:"))
		controls.addWidget(self.exposure_spinbox)
		controls.addStretch(1)
		controls.addWidget(self.pattern_counter_label)

		root = QVBoxLayout()
		root.addLayout(controls)
		root.addWidget(self.tab_widget)

		container = QWidget()
		container.setLayout(root)
		self.setCentralWidget(container)

		self.timer = QTimer(self)
		self.timer.setInterval(30)
		self.timer.timeout.connect(self._update_frame)
		self._streaming = False
		self._frame_lock = threading.Lock()
		self._latest_frame = None
		self._frame_count = 0
		self._last_frame_error: Optional[str] = None
		self._fft_roi: Optional[tuple[int, int, int, int]] = None
		self._plm_phase_patterns: Optional[np.ndarray] = None
		self._plm_pattern_path: Optional[str] = None
		self._plm_loop_thread: Optional[threading.Thread] = None
		self._plm_stop_event = threading.Event()

		self.start_button.clicked.connect(self._start_live)
		self.stop_button.clicked.connect(self._stop_live)
		self.auto_exposure_checkbox.toggled.connect(self._on_auto_exposure_toggled)
		self.exposure_spinbox.editingFinished.connect(self._on_manual_exposure_changed)
		self.pattern_browse_button.clicked.connect(self._on_browse_pattern_file)
		self.pattern_load_button.clicked.connect(self._on_load_pattern_file)
		self.pattern_start_button.clicked.connect(self._start_plm_pattern_loop)
		self.pattern_stop_button.clicked.connect(self._stop_plm_pattern_loop)
		self.plm_counter_updated.connect(self._set_pattern_counter)
		self.plm_loop_finished.connect(self._on_plm_loop_finished)

		self._exp_auto_feature_name: Optional[str] = self._detect_exposure_auto_feature_name()
		self._exp_time_feature_name: Optional[str] = self._detect_exposure_time_feature_name()

		self._initialize_exposure_controls()

	def _set_status(self, message: str) -> None:
		self.status_label.setText(message)

	def _set_pattern_counter(self, current_index: int, total_count: int) -> None:
		if total_count <= 0:
			self.pattern_counter_label.setText("Pattern: - / -")
			return
		self.pattern_counter_label.setText(f"Pattern: {current_index} / {total_count}")

	def _detect_exposure_auto_feature_name(self) -> Optional[str]:
		if hasattr(self.cam, "ExposureAuto"):
			return "ExposureAuto"
		return None

	def _detect_exposure_time_feature_name(self) -> Optional[str]:
		if hasattr(self.cam, "ExposureTime"):
			return "ExposureTime"
		if hasattr(self.cam, "ExposureTimeAbs"):
			return "ExposureTimeAbs"
		return None

	def _get_feature(self, name: str):
		return getattr(self.cam, name)

	def _initialize_exposure_controls(self) -> None:
		has_auto = self._exp_auto_feature_name is not None
		has_time = self._exp_time_feature_name is not None

		if has_time:
			try:
				current_us = float(self._get_feature(self._exp_time_feature_name).get())
			except Exception:
				current_us = 2000.0
			self.exposure_spinbox.setValue(current_us)
		else:
			self.exposure_spinbox.setEnabled(False)

		if has_auto:
			try:
				auto_state = str(self._get_feature(self._exp_auto_feature_name).get())
				is_auto = auto_state.lower() != "off"
			except Exception:
				is_auto = False
			self.auto_exposure_checkbox.setChecked(is_auto)
		else:
			self.auto_exposure_checkbox.setChecked(False)
			self.auto_exposure_checkbox.setEnabled(False)

		self.exposure_spinbox.setEnabled(has_time and not self.auto_exposure_checkbox.isChecked())

		if not has_auto and not has_time:
			self._set_status("Exposure controls are not available on this camera.")
		elif not has_auto and has_time:
			self._set_status(f"Using manual exposure via {self._exp_time_feature_name}.")
		else:
			self._set_status("Exposure control ready.")

	def _start_live(self) -> None:
		if self._streaming:
			return
		try:
			with self._frame_lock:
				self._latest_frame = None
				self._frame_count = 0
				self._last_frame_error = None
			if hasattr(self.cam, "TriggerMode"):
				self.cam.TriggerMode.set("Off")
			if hasattr(self.cam, "AcquisitionMode"):
				self.cam.AcquisitionMode.set("Continuous")
			self.cam.start_streaming(handler=self._frame_handler, buffer_count=5)
			self._streaming = True
			self.timer.start()
			self.start_button.setEnabled(False)
			self.stop_button.setEnabled(True)
			self._set_status("Live view started.")
		except Exception as exc:
			self._set_status(f"Failed to start streaming: {exc}")

	def _stop_live(self) -> None:
		self.timer.stop()
		if self._streaming:
			try:
				self.cam.stop_streaming()
			except Exception as exc:
				self._set_status(f"Failed to stop streaming cleanly: {exc}")
			self._streaming = False
		self.start_button.setEnabled(True)
		self.stop_button.setEnabled(False)
		self._set_status("Live view stopped.")

	def _frame_handler(self, cam, stream, frame) -> None:
		try:
			frame_mono8 = frame.convert_pixel_format(PixelFormat.Mono8)
			image = frame_mono8.as_opencv_image().copy()
			with self._frame_lock:
				self._latest_frame = image
				self._frame_count += 1
				self._last_frame_error = None
		except Exception as exc:
			with self._frame_lock:
				self._last_frame_error = f"Callback error: {exc}"
		finally:
			cam.queue_frame(frame)

	def _on_auto_exposure_toggled(self, checked: bool) -> None:
		self.exposure_spinbox.setEnabled(
			self._exp_time_feature_name is not None and not checked
		)

		if self._exp_auto_feature_name is None:
			if checked:
				self._set_status("Auto exposure feature is not available on this camera.")
			return

		try:
			auto_feature = self._get_feature(self._exp_auto_feature_name)
			auto_feature.set("Continuous" if checked else "Off")
			mode = "Auto" if checked else "Manual"
			self._set_status(f"Exposure mode: {mode}.")
		except Exception as exc:
			self._set_status(f"Failed to set auto exposure: {exc}")

	def _on_manual_exposure_changed(self) -> None:
		if self._exp_time_feature_name is None:
			return
		if self.auto_exposure_checkbox.isChecked():
			return

		desired_us = float(self.exposure_spinbox.value())
		try:
			self._get_feature(self._exp_time_feature_name).set(desired_us)
			actual_us = float(self._get_feature(self._exp_time_feature_name).get())
			self._set_status(
				f"Manual exposure set to {actual_us:.1f} us ({self._exp_time_feature_name})."
			)
		except Exception as exc:
			self._set_status(f"Failed to set exposure: {exc}")

	def _on_fft_roi_changed(self, roi: Optional[tuple[int, int, int, int]]) -> None:
		self._fft_roi = roi
		if roi is None:
			self.fft_status_label.setText("ROI cleared. Draw rectangle over first order in FFT.")
			self.recovered_field_status_label.setText("Recovered field not available. Select FFT ROI.")
			return
		x0, y0, x1, y1 = roi
		self.fft_status_label.setText(
			f"FFT ROI selected: x={x0}:{x1}, y={y0}:{y1}."
		)
		self.recovered_field_status_label.setText("Recovering field from selected first order...")

	def _on_browse_pattern_file(self) -> None:
		file_path, _ = QFileDialog.getOpenFileName(
			self,
			"Select pattern .npy file",
			str(Path.cwd()),
			"NumPy files (*.npy)",
		)
		if not file_path:
			return
		self.pattern_path_edit.setText(file_path)

	def _on_load_pattern_file(self) -> None:
		if self._plm_loop_thread is not None and self._plm_loop_thread.is_alive():
			self.pattern_status_label.setText("Stop PLM loop before loading another file.")
			return

		path = self.pattern_path_edit.text().strip()
		if not path:
			self.pattern_status_label.setText("Select a .npy file first.")
			return

		try:
			arr = np.load(path)
			if arr.ndim != 3:
				raise ValueError(
					f"Pattern array must be 3D [n_frames, y, x]. Got shape {arr.shape}."
				)
			if arr.shape[0] < 1:
				raise ValueError("Pattern array has no frames.")

			self._plm_phase_patterns = arr.astype(np.float32)
			self._plm_pattern_path = path
			self._set_pattern_counter(1, int(arr.shape[0]))
			self.pattern_info_label.setText(
				f"Loaded {arr.shape[0]} frames, resolution {arr.shape[2]}x{arr.shape[1]}."
			)
			self.pattern_status_label.setText("Pattern file loaded. Ready to start PLM loop.")
		except Exception as exc:
			self._plm_phase_patterns = None
			self._plm_pattern_path = None
			self._set_pattern_counter(0, 0)
			self.pattern_info_label.setText("")
			self.pattern_status_label.setText(f"Failed to load pattern file: {exc}")

	def _start_plm_pattern_loop(self) -> None:
		if self._plm_loop_thread is not None and self._plm_loop_thread.is_alive():
			self.pattern_status_label.setText("PLM loop is already running.")
			return

		if self._plm_phase_patterns is None:
			self.pattern_status_label.setText("Load a pattern .npy file first.")
			return

		plm_catalog = self.plm_catalog_edit.text().strip()
		if not plm_catalog:
			self.pattern_status_label.setText("Enter a PLM catalog (e.g., p67nirtemp).")
			return

		self._plm_stop_event.clear()
		interval_ms = float(self.pattern_interval_spin.value())
		monitor = int(self.plm_monitor_spin.value())

		self._plm_loop_thread = threading.Thread(
			target=self._plm_pattern_worker,
			args=(plm_catalog, monitor, interval_ms),
			daemon=True,
		)
		self._plm_loop_thread.start()
		self._set_pattern_counter(1, int(self._plm_phase_patterns.shape[0]))
		self.pattern_start_button.setEnabled(False)
		self.pattern_stop_button.setEnabled(True)
		self.pattern_status_label.setText("Starting PLM pattern loop...")

	def _stop_plm_pattern_loop(self) -> None:
		if self._plm_loop_thread is None or not self._plm_loop_thread.is_alive():
			self.pattern_start_button.setEnabled(True)
			self.pattern_stop_button.setEnabled(False)
			self.pattern_status_label.setText("PLM loop is not running.")
			return

		self._plm_stop_event.set()
		self.pattern_status_label.setText("Stopping PLM pattern loop...")

	def _plm_pattern_worker(self, plm_catalog: str, monitor: int, interval_ms: float) -> None:
		message = "PLM loop stopped."
		try:
			phase_patterns = self._plm_phase_patterns
			if phase_patterns is None:
				raise RuntimeError("No phase patterns loaded.")

			plm = PLM.from_db(plm_catalog)
			expected_h, expected_w = int(plm.shape[0]), int(plm.shape[1])
			if phase_patterns.shape[1] != expected_h or phase_patterns.shape[2] != expected_w:
				raise ValueError(
					f"Pattern shape {phase_patterns.shape[1:]} does not match PLM shape {(expected_h, expected_w)} for catalog '{plm_catalog}'."
				)

			images = [
				Image.fromarray(plm.process_phase_map(phase_patterns[idx]))
				for idx in range(phase_patterns.shape[0])
			]

			if not images:
				raise RuntimeError("No valid pattern images generated.")

			interval_s = max(0.01, interval_ms / 1000.0)
			with ImageWindow(fullscreen=True, monitor=monitor) as win:
				frame_idx = 0
				last_update = perf_counter() - interval_s
				total_frames = len(images)

				def loop_callback() -> None:
					nonlocal frame_idx, last_update
					if self._plm_stop_event.is_set():
						raise EventLoopExit

					now = perf_counter()
					if now - last_update >= interval_s:
						display_idx = frame_idx
						win.clear()
						win.load(images[display_idx])
						self.plm_counter_updated.emit(display_idx + 1, total_frames)
						frame_idx = (frame_idx + 1) % total_frames
						last_update = now

				win.loop_callback = loop_callback
				win.load(images[0])
				self.plm_counter_updated.emit(1, total_frames)
				win.run()

			if self._plm_stop_event.is_set():
				message = "PLM loop stopped."
			else:
				message = "PLM loop exited."
		except Exception as exc:
			message = f"PLM loop error: {exc}"
		finally:
			self.plm_loop_finished.emit(message)

	def _on_plm_loop_finished(self, message: str) -> None:
		self.pattern_start_button.setEnabled(True)
		self.pattern_stop_button.setEnabled(False)
		self.pattern_status_label.setText(message)
		self._plm_stop_event.clear()
		self._plm_loop_thread = None

	def _update_frame(self) -> None:
		try:
			with self._frame_lock:
				if self._latest_frame is None:
					if self._last_frame_error:
						self._set_status(self._last_frame_error)
						self.fft_status_label.setText(self._last_frame_error)
						self.recovered_field_status_label.setText(self._last_frame_error)
					elif self._streaming:
						self._set_status("Waiting for first frame...")
						self.fft_status_label.setText("Waiting for first frame...")
						self.recovered_field_status_label.setText("Waiting for first frame...")
					return
				image = self._latest_frame
				frame_count = self._frame_count

			if image.ndim == 3:
				image = image[..., 0]

			height, width = image.shape[:2]
			bytes_per_line = width
			qimage = QImage(
				image.data,
				width,
				height,
				bytes_per_line,
				QImage.Format_Grayscale8,
			)
			pixmap = QPixmap.fromImage(qimage)
			self.image_label.setPixmap(
				pixmap.scaled(
					self.image_label.size(),
					Qt.KeepAspectRatio,
					Qt.SmoothTransformation,
				)
			)

			fft_shifted, fft_image = self._compute_fft_shifted_and_magnitude_image(image)
			self.fft_image_label.set_image_array(fft_image)

			recovered_field = self._recover_off_axis_field(fft_shifted)
			if recovered_field is not None:
				amplitude = np.abs(recovered_field)
				amplitude_norm = amplitude / (float(np.max(amplitude)) + 1e-12)
				recovered_phase = np.angle(recovered_field)
				phase_rgb = phase_to_rgb_uint8(recovered_phase, value=amplitude_norm)
				phase_height, phase_width = phase_rgb.shape[:2]
				phase_rgb_contiguous = np.ascontiguousarray(phase_rgb)
				phase_qimage = QImage(
					phase_rgb_contiguous.data,
					phase_width,
					phase_height,
					phase_width * 3,
					QImage.Format_RGB888,
				).copy()
				phase_pixmap = QPixmap.fromImage(phase_qimage)
				self.recovered_field_label.setPixmap(
					phase_pixmap.scaled(
						self.recovered_field_label.size(),
						Qt.KeepAspectRatio,
						Qt.SmoothTransformation,
					)
				)

				x0, y0, x1, y1 = self._fft_roi if self._fft_roi is not None else (0, 0, 0, 0)
				self.recovered_field_status_label.setText(
					f"Recovered phase (amplitude-weighted RGB) updated from ROI x={x0}:{x1}, y={y0}:{y1}."
				)
			elif self._fft_roi is None:
				self.recovered_field_status_label.setText("Select FFT ROI to recover amplitude-weighted phase.")
			else:
				self.recovered_field_status_label.setText("ROI too small or invalid for recovery.")

			self._set_status(f"Live view started. Frames received: {frame_count}")
			if self._fft_roi is None:
				self.fft_status_label.setText(
					f"2D FFT updated. Frames processed: {frame_count}. Draw ROI around first order."
				)
		except Exception as exc:
			self._set_status(f"Render error: {exc}")
			self.fft_status_label.setText(f"FFT render error: {exc}")
			self.recovered_field_status_label.setText(f"Recovery error: {exc}")

	def _compute_fft_shifted_and_magnitude_image(
		self, image: np.ndarray
	) -> tuple[np.ndarray, np.ndarray]:
		float_image = image.astype(np.float32)
		fft_complex = np.fft.fft2(float_image)
		fft_shifted = np.fft.fftshift(fft_complex)
		magnitude = np.abs(fft_shifted)
		magnitude_log = np.log1p(magnitude)
		magnitude_norm = cv_normalize_to_uint8(magnitude_log)
		return fft_shifted, magnitude_norm

	def _recover_off_axis_field(self, fft_shifted: np.ndarray) -> Optional[np.ndarray]:
		roi = self._fft_roi
		if roi is None:
			return None
		x0, y0, x1, y1 = roi
		if x1 - x0 < 3 or y1 - y0 < 3:
			return None

		filtered = np.zeros_like(fft_shifted)
		filtered[y0:y1, x0:x1] = fft_shifted[y0:y1, x0:x1]

		if self.basic_recovery_radio.isChecked():
			height, width = fft_shifted.shape[:2]
			roi_center_x = (x0 + x1) // 2
			roi_center_y = (y0 + y1) // 2
			target_center_x = width // 2
			target_center_y = height // 2
			shift_x = target_center_x - roi_center_x
			shift_y = target_center_y - roi_center_y
			centered = np.roll(filtered, shift=(shift_y, shift_x), axis=(0, 1))
			return np.fft.ifft2(np.fft.ifftshift(centered))

		peak_y_f, peak_x_f = estimate_subpixel_peak_in_roi(np.abs(fft_shifted), roi)
		recovered_field = demodulate_to_center_no_ref(filtered, peak_y_f, peak_x_f)
		recovered_field = remove_linear_phase_tilt_blind(recovered_field)
		recovered_field = remove_global_phase_offset_blind(recovered_field)
		return recovered_field

	def closeEvent(self, event):
		self.timer.stop()
		if self._plm_loop_thread is not None and self._plm_loop_thread.is_alive():
			self._plm_stop_event.set()
		if self._streaming:
			try:
				self.cam.stop_streaming()
			except Exception:
				pass
			self._streaming = False
		super().closeEvent(event)


def cv_normalize_to_uint8(image: np.ndarray) -> np.ndarray:
	min_val = float(np.min(image))
	max_val = float(np.max(image))
	if max_val <= min_val:
		return np.zeros_like(image, dtype=np.uint8)
	normalized = (image - min_val) / (max_val - min_val)
	return (normalized * 255.0).astype(np.uint8)


def phase_to_rgb_uint8(phase: np.ndarray, value: Optional[np.ndarray] = None) -> np.ndarray:
	h = (phase + np.pi) / (2.0 * np.pi)
	h = np.mod(h, 1.0)
	s = np.ones_like(h, dtype=np.float32)
	if value is None:
		v = np.ones_like(h, dtype=np.float32)
	else:
		v = np.clip(value.astype(np.float32), 0.0, 1.0)

	i = np.floor(h * 6.0).astype(np.int32)
	f = h * 6.0 - i
	p = v * (1.0 - s)
	q = v * (1.0 - f * s)
	t = v * (1.0 - (1.0 - f) * s)

	r = np.zeros_like(h, dtype=np.float32)
	g = np.zeros_like(h, dtype=np.float32)
	b = np.zeros_like(h, dtype=np.float32)

	idx = i % 6
	mask = idx == 0
	r[mask], g[mask], b[mask] = v[mask], t[mask], p[mask]
	mask = idx == 1
	r[mask], g[mask], b[mask] = q[mask], v[mask], p[mask]
	mask = idx == 2
	r[mask], g[mask], b[mask] = p[mask], v[mask], t[mask]
	mask = idx == 3
	r[mask], g[mask], b[mask] = p[mask], q[mask], v[mask]
	mask = idx == 4
	r[mask], g[mask], b[mask] = t[mask], p[mask], v[mask]
	mask = idx == 5
	r[mask], g[mask], b[mask] = v[mask], p[mask], q[mask]

	rgb = np.stack([r, g, b], axis=-1)
	return (rgb * 255.0).astype(np.uint8)


def estimate_subpixel_peak_in_roi(
	magnitude: np.ndarray,
	roi: tuple[int, int, int, int],
	eps: float = 1e-12,
) -> tuple[float, float]:
	x0, y0, x1, y1 = roi
	patch = np.maximum(magnitude[y0:y1, x0:x1], 0.0)
	if patch.size == 0:
		return float((y0 + y1) * 0.5), float((x0 + x1) * 0.5)

	yy, xx = np.indices(patch.shape, dtype=np.float64)
	weights = patch.astype(np.float64)
	weight_sum = float(np.sum(weights))
	if weight_sum <= eps:
		return float((y0 + y1) * 0.5), float((x0 + x1) * 0.5)

	py = float(np.sum((yy + y0) * weights) / weight_sum)
	px = float(np.sum((xx + x0) * weights) / weight_sum)
	return py, px


def demodulate_to_center_no_ref(
	fft_filtered: np.ndarray,
	peak_y_f: float,
	peak_x_f: float,
) -> np.ndarray:
	height, width = fft_filtered.shape[:2]
	center_y = (height - 1) / 2.0
	center_x = (width - 1) / 2.0

	dy = center_y - float(peak_y_f)
	dx = center_x - float(peak_x_f)
	dy_i = int(round(dy))
	dx_i = int(round(dx))
	fft_integer_shifted = np.roll(fft_filtered, shift=(dy_i, dx_i), axis=(0, 1))

	dy_f = dy - dy_i
	dx_f = dx - dx_i
	recovered = np.fft.ifft2(np.fft.ifftshift(fft_integer_shifted))

	yy, xx = np.indices((height, width), dtype=np.float32)
	frac_ramp = np.exp(-1j * 2.0 * np.pi * ((dx_f * xx / width) + (dy_f * yy / height)))
	return recovered * frac_ramp


def remove_linear_phase_tilt_blind(
	field: np.ndarray,
	amp_thresh: float = 0.2,
	eps: float = 1e-12,
) -> np.ndarray:
	amp = np.abs(field)
	amp_max = float(np.max(amp))
	if amp_max <= eps:
		return field

	mask = amp > (amp_thresh * amp_max)
	gx_num = np.sum(np.conj(field[:, :-1]) * field[:, 1:] * mask[:, :-1])
	gy_num = np.sum(np.conj(field[:-1, :]) * field[1:, :] * mask[:-1, :])
	gx = np.angle(gx_num) if np.abs(gx_num) > eps else 0.0
	gy = np.angle(gy_num) if np.abs(gy_num) > eps else 0.0

	height, width = field.shape[:2]
	yy, xx = np.indices((height, width), dtype=np.float32)
	ramp = np.exp(-1j * (gx * xx + gy * yy))
	return field * ramp


def remove_global_phase_offset_blind(
	field: np.ndarray,
	amp_thresh: float = 0.2,
	eps: float = 1e-12,
) -> np.ndarray:
	amp = np.abs(field)
	amp_max = float(np.max(amp))
	if amp_max <= eps:
		return field

	mask = amp > (amp_thresh * amp_max)
	if not np.any(mask):
		return field

	phase_ref_complex = np.sum(field[mask])
	if np.abs(phase_ref_complex) <= eps:
		return field
	phase_ref = np.angle(phase_ref_complex)
	return field * np.exp(-1j * phase_ref)


def main() -> int:
	app = QApplication(sys.argv)

	with VmbSystem.get_instance() as vmb:
		cameras = vmb.get_all_cameras()
		if not cameras:
			print("No cameras found.")
			return 1

		with cameras[0] as cam:
			print("Using camera:", cam.get_name())
			window = LiveCameraWindow(cam)
			window.show()
			return app.exec_()


if __name__ == "__main__":
	raise SystemExit(main())
