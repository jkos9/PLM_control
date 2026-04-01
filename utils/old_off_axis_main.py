import sys
import importlib
import threading
import csv
from collections import deque
from pathlib import Path
from datetime import datetime
from time import perf_counter
from typing import Optional

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
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
	QIntValidator = QtGui.QIntValidator
	QPainter = QtGui.QPainter
	QPen = QtGui.QPen
	QPixmap = QtGui.QPixmap
	QApplication = QtWidgets.QApplication
	QCheckBox = QtWidgets.QCheckBox
	QComboBox = QtWidgets.QComboBox
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
		y_offset = 0
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
		y_offset = 0.0
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


class PointSelectableImageLabel(QLabel):
	def __init__(self, text: str = "", parent=None):
		super().__init__(text, parent)
		self.setMouseTracking(True)
		self._image_shape: Optional[tuple[int, int]] = None
		self._display_qimage: Optional[QImage] = None
		self._points: list[Optional[tuple[int, int]]] = [None, None]
		self._next_point_index = 0
		self.on_points_changed = None

	def set_image_rgb_array(self, image: np.ndarray) -> None:
		self._image_shape = (int(image.shape[0]), int(image.shape[1]))
		contiguous = np.ascontiguousarray(image)
		self._display_qimage = QImage(
			contiguous.data,
			int(image.shape[1]),
			int(image.shape[0]),
			int(image.shape[1]) * 3,
			QImage.Format_RGB888,
		).copy()
		self._refresh_display()                                                                                                                                                                                                                                                                                                                                                                   
          
	def get_points(self) -> tuple[Optional[tuple[int, int]], Optional[tuple[int, int]]]:
		return self._points[0], self._points[1]

	def clear_points(self) -> None:
		self._points = [None, None]
		self._next_point_index = 0
		self._refresh_display()
		if self.on_points_changed is not None:
			self.on_points_changed(self._points[0], self._points[1])

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
		self._points[self._next_point_index] = mapped
		self._next_point_index = 1 - self._next_point_index
		self._refresh_display()
		if self.on_points_changed is not None:
			self.on_points_changed(self._points[0], self._points[1])

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

		colors = [Qt.yellow, Qt.cyan]
		for idx, point in enumerate(self._points):
			if point is None:
				continue
			px, py = point
			label_rect = self._image_rect_to_label_rect((px, py, px + 1, py + 1))
			cx = label_rect.x()
			cy = label_rect.y()
			pen = QPen(colors[idx])
			pen.setWidth(2)
			painter.setPen(pen)
			painter.drawEllipse(cx - 6, cy - 6, 12, 12)
			painter.drawText(cx + 8, cy - 8, f"P{idx + 1}")

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
		x0, y0, _, _ = rect
		left = int(round(x_offset + x0 * scale))
		top = int(round(y_offset + y0 * scale))
		return QRect(left, top, 1, 1)


class LiveCameraWindow(QMainWindow):
	plm_counter_updated = QtCore.pyqtSignal(int, int)
	plm_loop_finished = QtCore.pyqtSignal(str)

	def __init__(self, cam):
		super().__init__()
		self.cam = cam
		self.setWindowTitle("Allied Vision Live Camera")
		self.resize(1400, 800)

		self.image_label = RoiSelectableImageLabel("No frame yet")
		self.image_label.setAlignment(Qt.AlignCenter)
		self.image_label.setMinimumSize(640, 480)
		self.image_label.on_roi_changed = self._on_live_roi_changed

		self.start_button = QPushButton("Start Live")
		self.stop_button = QPushButton("Stop")
		self.stop_button.setEnabled(False)

		self.auto_exposure_checkbox = QCheckBox("Auto exposure")
		self.exposure_spinbox = QDoubleSpinBox()
		self.exposure_spinbox.setDecimals(1)
		self.exposure_spinbox.setRange(10.0, 1_000_000.0)
		self.exposure_spinbox.setSingleStep(100.0)
		self.exposure_spinbox.setSuffix(" us")
		self.trigger_mode_combo = QComboBox()
		self.trigger_mode_combo.addItem("FreeRun", "freerun")
		self.trigger_mode_combo.addItem("Line1", "line1")
		self.trigger_mode_combo.setCurrentIndex(1)
		self.trigger_delay_edit = QLineEdit("0.000")
		self.trigger_delay_edit.setMaximumWidth(90)
		self.drop_frames_combo = QComboBox()
		self.drop_frames_combo.addItem("0", 0)
		self.drop_frames_combo.addItem("1", 1)
		self.drop_frames_combo.addItem("2", 2)
		self.drop_frames_combo.addItem("3", 3)
		self.drop_frames_combo.addItem("5", 5)
		self.drop_frames_combo.addItem("10", 10)
		self.drop_frames_combo.setCurrentIndex(1)

		self.status_label = QLabel()
		self.status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
		self.live_roi_status_label = QLabel("Live ROI: not set")
		self.live_roi_intensity_label = QLabel("Live ROI sum intensity: n/a")
		self.grating_x_edit = QLineEdit("0")
		self.grating_x_edit.setMaximumWidth(100)
		self.grating_x_edit.setValidator(QIntValidator(-100000, 100000, self))
		self.grating_x_value_label = QLabel("Off")
		self.grating_y_edit = QLineEdit("0")
		self.grating_y_edit.setMaximumWidth(100)
		self.grating_y_edit.setValidator(QIntValidator(-100000, 100000, self))
		self.grating_y_value_label = QLabel("Off")

		self.fft_image_label = RoiSelectableImageLabel("No FFT yet")
		self.fft_image_label.setAlignment(Qt.AlignCenter)
		self.fft_image_label.setMinimumSize(640, 480)
		self.fft_image_label.on_roi_changed = self._on_fft_roi_changed

		self.fft_status_label = QLabel("Waiting for frames...")

		self.recovered_field_label = PointSelectableImageLabel("Select FFT ROI to recover field")
		self.recovered_field_label.setAlignment(Qt.AlignCenter)
		self.recovered_field_label.setMinimumSize(960, 640)
		self.recovered_field_label.on_points_changed = self._on_recovered_points_changed
		self.recovered_field_status_label = QLabel("Recovered field not available.")
		self.point1_label = QLabel("P1: not set")
		self.point2_label = QLabel("P2: not set")
		self.phase_diff_label = QLabel("Δφ(P1-P2): n/a")
		self.phase_avg_window_combo = QComboBox()
		self.phase_avg_window_combo.addItem("1 x 1", 1)
		self.phase_avg_window_combo.addItem("3 x 3", 3)
		self.phase_avg_window_combo.addItem("5 x 5", 5)
		self.phase_avg_window_combo.addItem("7 x 7", 7)
		self.phase_avg_window_combo.addItem("9 x 9", 9)
		self.phase_avg_window_combo.setCurrentIndex(0)
		self.phase_plot_canvas = FigureCanvas(Figure(figsize=(6.0, 2.5)))
		self.phase_plot_ax = self.phase_plot_canvas.figure.add_subplot(111)
		self.phase_plot_ax.set_title("Live Point Phases")
		self.phase_plot_ax.set_xlabel("Time (s)")
		self.phase_plot_ax.set_ylabel("Phase (rad)")
		self.phase_plot_ax.set_ylim(-np.pi, 2.0 * np.pi)
		self.phase_plot_ax.grid(True, alpha=0.3)
		self.phase_plot_line_phi1, = self.phase_plot_ax.plot([], [], color="tab:green", linewidth=1.2, label="phi1")
		self.phase_plot_line_phi2, = self.phase_plot_ax.plot([], [], color="tab:purple", linewidth=1.2, label="phi2")
		self.phase_plot_line_dphi, = self.phase_plot_ax.plot([], [], color="tab:blue", linewidth=1.2, label="dphi_0_2pi")
		self.phase_plot_ax.legend(loc="upper right")
		self.intensity_plot_canvas = FigureCanvas(Figure(figsize=(6.0, 2.5)))
		self.intensity_plot_ax = self.intensity_plot_canvas.figure.add_subplot(111)
		self.intensity_plot_ax.set_title("Live ROI Intensity")
		self.intensity_plot_ax.set_xlabel("Time (s)")
		self.intensity_plot_ax.set_ylabel("Summed intensity")
		self.intensity_plot_ax.grid(True, alpha=0.3)
		self.intensity_plot_line, = self.intensity_plot_ax.plot([], [], color="tab:red", linewidth=1.2, label="roi_sum")
		self.intensity_plot_ax.legend(loc="upper right")
		self.logging_path_edit = QLineEdit(str(Path.cwd() / "phase_measurements.csv"))
		self.logging_browse_button = QPushButton("Browse Log")
		self.logging_start_button = QPushButton("Start Logging")
		self.logging_stop_button = QPushButton("Stop Logging")
		self.logging_stop_button.setEnabled(False)
		self.logging_n_times_combo = QComboBox()
		self.logging_n_times_combo.addItems(["1", "2", "3", "5", "10"])
		self.logging_n_times_combo.setCurrentText("1")
		self.logging_status_label = QLabel("Logging: inactive")
		self.phase_colorbar_label = QLabel()
		self.phase_colorbar_label.setFixedWidth(40)
		self.phase_colorbar_label.setMinimumHeight(280)

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
		self.pattern_interval_spin.setValue(1000.0)
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
		live_panel.addWidget(self.live_roi_status_label)
		live_panel.addWidget(self.live_roi_intensity_label)
		grating_x_row = QHBoxLayout()
		grating_x_row.addWidget(QLabel("X grating period (px):"))
		grating_x_row.addWidget(self.grating_x_edit)
		grating_x_row.addWidget(self.grating_x_value_label)
		grating_x_row.addStretch(1)
		live_panel.addLayout(grating_x_row)
		grating_y_row = QHBoxLayout()
		grating_y_row.addWidget(QLabel("Y grating period (px):"))
		grating_y_row.addWidget(self.grating_y_edit)
		grating_y_row.addWidget(self.grating_y_value_label)
		grating_y_row.addStretch(1)
		live_panel.addLayout(grating_y_row)

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

		recovered_plot_row = QHBoxLayout()
		recovered_plot_row.addWidget(self.recovered_field_label, 1)

		colorbar_col = QVBoxLayout()
		colorbar_col.addWidget(QLabel("Phase"), alignment=Qt.AlignHCenter)
		colorbar_row = QHBoxLayout()
		colorbar_row.addWidget(self.phase_colorbar_label)

		ticks_col = QVBoxLayout()
		ticks_col.addWidget(QLabel("+π"))
		ticks_col.addStretch(1)
		ticks_col.addWidget(QLabel("0"))
		ticks_col.addStretch(1)
		ticks_col.addWidget(QLabel("-π"))
		colorbar_row.addLayout(ticks_col)

		colorbar_col.addLayout(colorbar_row)
		colorbar_col.addStretch(1)
		recovered_plot_row.addLayout(colorbar_col)

		recovered_layout.addLayout(recovered_plot_row)
		phase_avg_row = QHBoxLayout()
		phase_avg_row.addWidget(QLabel("Point phase average window:"))
		phase_avg_row.addWidget(self.phase_avg_window_combo)
		phase_avg_row.addStretch(1)
		recovered_layout.addLayout(phase_avg_row)
		recovered_layout.addWidget(self.point1_label)
		recovered_layout.addWidget(self.point2_label)
		recovered_layout.addWidget(self.phase_diff_label)
		logging_row = QHBoxLayout()
		logging_row.addWidget(QLabel("Log file:"))
		logging_row.addWidget(self.logging_path_edit, 1)
		logging_row.addWidget(QLabel("Samples/pattern:"))
		logging_row.addWidget(self.logging_n_times_combo)
		logging_row.addWidget(self.logging_browse_button)
		logging_row.addWidget(self.logging_start_button)
		logging_row.addWidget(self.logging_stop_button)
		recovered_layout.addLayout(logging_row)
		recovered_layout.addWidget(self.logging_status_label)
		recovered_layout.addWidget(self.recovered_field_status_label)
		recovered_tab.setLayout(recovered_layout)

		phase_plot_tab = QWidget()
		phase_plot_layout = QVBoxLayout()
		phase_plot_layout.addWidget(QLabel("Live Phase Plot"))
		phase_plot_layout.addWidget(self.phase_plot_canvas)
		phase_plot_tab.setLayout(phase_plot_layout)

		intensity_plot_tab = QWidget()
		intensity_plot_layout = QVBoxLayout()
		intensity_plot_layout.addWidget(QLabel("Live Intensity Plot"))
		intensity_plot_layout.addWidget(self.intensity_plot_canvas)
		intensity_plot_tab.setLayout(intensity_plot_layout)

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
		self.tab_widget.addTab(phase_plot_tab, "Phase Plot")
		self.tab_widget.addTab(intensity_plot_tab, "Intensity Plot")
		self.tab_widget.addTab(pattern_tab, "PLM Patterns")

		controls = QHBoxLayout()
		controls.addWidget(self.start_button)
		controls.addWidget(self.stop_button)
		controls.addSpacing(20)
		controls.addWidget(self.auto_exposure_checkbox)
		controls.addWidget(QLabel("Manual exposure:"))
		controls.addWidget(self.exposure_spinbox)
		controls.addSpacing(20)
		controls.addWidget(QLabel("Trigger:"))
		controls.addWidget(self.trigger_mode_combo)
		controls.addWidget(QLabel("Delay (ms):"))
		controls.addWidget(self.trigger_delay_edit)
		controls.addWidget(QLabel("Drop frames after pattern:"))
		controls.addWidget(self.drop_frames_combo)
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
		self._live_roi: Optional[tuple[int, int, int, int]] = None
		self._fft_roi: Optional[tuple[int, int, int, int]] = None
		self._latest_recovered_field: Optional[np.ndarray] = None
		self._current_plm_pattern_index: Optional[int] = None
		self._current_plm_total_patterns: Optional[int] = None
		self._logging_enabled = False
		self._log_file_path: Optional[Path] = None
		self._pattern_upload_sequence = 0
		self._active_pattern_upload_sequence = 0
		self._logs_taken_for_active_pattern = 0
		self._last_log_time_s = 0.0
		self._last_logged_frame_count = -1
		self._last_logged_dphi_wrapped: Optional[float] = None
		self._dphi_unwrapped_accum: Optional[float] = None
		self._plm_phase_patterns: Optional[np.ndarray] = None
		self._plm_pattern_path: Optional[str] = None
		self._plm_loop_thread: Optional[threading.Thread] = None
		self._plm_stop_event = threading.Event()
		self._phase_plot_t0 = perf_counter()
		self._phase_plot_time_s = deque(maxlen=600)
		self._phase_plot_phi1 = deque(maxlen=600)
		self._phase_plot_phi2 = deque(maxlen=600)
		self._phase_plot_dphi = deque(maxlen=600)
		self._intensity_plot_t0 = perf_counter()
		self._intensity_plot_time_s = deque(maxlen=600)
		self._intensity_plot_sum = deque(maxlen=600)
		self._grating_period_x_px = 0
		self._grating_period_y_px = 0
		self._drop_frame_target_count = 0

		self.start_button.clicked.connect(self._start_live)
		self.stop_button.clicked.connect(self._stop_live)
		self.auto_exposure_checkbox.toggled.connect(self._on_auto_exposure_toggled)
		self.exposure_spinbox.editingFinished.connect(self._on_manual_exposure_changed)
		self.pattern_browse_button.clicked.connect(self._on_browse_pattern_file)
		self.pattern_load_button.clicked.connect(self._on_load_pattern_file)
		self.pattern_start_button.clicked.connect(self._start_plm_pattern_loop)
		self.pattern_stop_button.clicked.connect(self._stop_plm_pattern_loop)
		self.plm_counter_updated.connect(self._on_plm_counter_updated)
		self.plm_loop_finished.connect(self._on_plm_loop_finished)
		self.logging_browse_button.clicked.connect(self._on_browse_log_file)
		self.logging_start_button.clicked.connect(self._start_logging)
		self.logging_stop_button.clicked.connect(self._stop_logging)
		self.trigger_delay_edit.editingFinished.connect(self._on_trigger_delay_editing_finished)
		self.grating_x_edit.editingFinished.connect(self._on_grating_input_changed)
		self.grating_y_edit.editingFinished.connect(self._on_grating_input_changed)

		self._exp_auto_feature_name: Optional[str] = self._detect_exposure_auto_feature_name()
		self._exp_time_feature_name: Optional[str] = self._detect_exposure_time_feature_name()
		self._trigger_delay_feature_name: Optional[str] = self._detect_trigger_delay_feature_name()

		self._initialize_exposure_controls()
		self._initialize_trigger_delay_controls()
		self._on_grating_input_changed()
		self._update_phase_colorbar()

	def _on_grating_input_changed(self) -> None:
		def parse_period(edit: QLineEdit) -> int:
			text = edit.text().strip()
			if text == "":
				edit.setText("0")
				return 0
			try:
				value = int(text)
			except Exception:
				value = 0
			value = max(-100000, min(100000, value))
			edit.setText(str(value))
			return value

		self._grating_period_x_px = parse_period(self.grating_x_edit)
		self._grating_period_y_px = parse_period(self.grating_y_edit)
		self.grating_x_value_label.setText(
			"Off" if self._grating_period_x_px == 0 else str(self._grating_period_x_px)
		)
		self.grating_y_value_label.setText(
			"Off" if self._grating_period_y_px == 0 else str(self._grating_period_y_px)
		)

	def _build_grating_phase_ramp(
		self,
		shape: tuple[int, int],
		period_x_px: int,
		period_y_px: int,
	) -> np.ndarray:
		height, width = int(shape[0]), int(shape[1])
		ramp = np.zeros((height, width), dtype=np.float32)
		if period_x_px != 0:
			x = np.arange(width, dtype=np.float32)
			ramp += (2.0 * np.pi * x / float(period_x_px))[None, :]
		if period_y_px != 0:
			y = np.arange(height, dtype=np.float32)
			ramp += (2.0 * np.pi * y / float(period_y_px))[:, None]
		return np.mod(ramp, 2.0 * np.pi).astype(np.float32, copy=False)

	def _set_status(self, message: str) -> None:
		self.status_label.setText(message)

	def _set_pattern_counter(self, current_index: int, total_count: int) -> None:
		if total_count <= 0:
			self.pattern_counter_label.setText("Pattern: - / -")
			return
		self.pattern_counter_label.setText(f"Pattern: {current_index} / {total_count}")

	def _on_plm_counter_updated(self, current_index: int, total_count: int) -> None:
		self._current_plm_pattern_index = current_index
		self._current_plm_total_patterns = total_count
		self._pattern_upload_sequence += 1
		self._active_pattern_upload_sequence = self._pattern_upload_sequence
		self._logs_taken_for_active_pattern = 0
		self._last_log_time_s = 0.0
		drop_n = int(self.drop_frames_combo.currentData() or 0)
		with self._frame_lock:
			self._drop_frame_target_count = self._frame_count + max(0, drop_n)
		self._set_pattern_counter(current_index, total_count)

	def _on_browse_log_file(self) -> None:
		file_path, _ = QFileDialog.getSaveFileName(
			self,
			"Select output CSV file",
			self.logging_path_edit.text().strip() or str(Path.cwd() / "phase_measurements.csv"),
			"CSV files (*.csv)",
		)
		if not file_path:
			return
		self.logging_path_edit.setText(file_path)

	def _start_logging(self) -> None:
		if self._logging_enabled:
			self.logging_status_label.setText("Logging: already active")
			return

		path_text = self.logging_path_edit.text().strip()
		if not path_text:
			self.logging_status_label.setText("Logging: choose an output CSV file")
			return

		log_path = Path(path_text)
		try:
			log_path.parent.mkdir(parents=True, exist_ok=True)
			file_exists = log_path.exists() and log_path.stat().st_size > 0
			n_max = int(self.logging_n_times_combo.currentText())
			with log_path.open("a", newline="", encoding="utf-8") as handle:
				writer = csv.writer(handle)
				if not file_exists:
					writer.writerow([
						"timestamp_iso",
						"frame_count",
						"exposure_us",
						"pattern_upload_seq",
						"sample_in_pattern",
						"samples_per_pattern_max",
						"pattern_index",
						"total_patterns",
						"pattern_file",
						"recovery_mode",
						"roi_x0",
						"roi_y0",
						"roi_x1",
						"roi_y1",
						"p1_x",
						"p1_y",
						"p2_x",
						"p2_y",
						"i1",
						"i2",
						"phi1_rad",
						"phi2_rad",
						"dphi_rad_wrapped",
						"dphi_rad_0_2pi",
						"dphi_rad_unwrapped_continuous",
					])
			self._log_file_path = log_path
			self._logging_enabled = True
			self._last_log_time_s = 0.0
			self._last_logged_frame_count = -1
			self._last_logged_dphi_wrapped = None
			self._dphi_unwrapped_accum = None
			self._active_pattern_upload_sequence = self._pattern_upload_sequence
			self._logs_taken_for_active_pattern = 0
			self.logging_start_button.setEnabled(False)
			self.logging_stop_button.setEnabled(True)
			self.logging_status_label.setText(
				f"Logging: active ({log_path}) | up to {n_max} sample(s) per pattern"
			)
		except Exception as exc:
			self.logging_status_label.setText(f"Logging start failed: {exc}")

	def _stop_logging(self) -> None:
		self._logging_enabled = False
		self._logs_taken_for_active_pattern = 0
		self._last_logged_dphi_wrapped = None
		self._dphi_unwrapped_accum = None
		self.logging_start_button.setEnabled(True)
		self.logging_stop_button.setEnabled(False)
		self.logging_status_label.setText("Logging: inactive")

	def _get_current_exposure_us(self) -> float:
		for feature_name in ("ExposureTime", "ExposureTimeAbs"):
			if hasattr(self.cam, feature_name):
				try:
					return float(self._get_feature(feature_name).get())
				except Exception:
					continue
		return float("nan")

	def _maybe_log_current_measurement(self, frame_count: int) -> None:
		if not self._logging_enabled or self._log_file_path is None:
			return

		field = self._latest_recovered_field
		if field is None:
			return

		p1, p2 = self.recovered_field_label.get_points()
		if p1 is None or p2 is None:
			return

		if frame_count == self._last_logged_frame_count:
			return

		n_max = int(self.logging_n_times_combo.currentText())
		has_plm_pattern = self._current_plm_pattern_index is not None
		if has_plm_pattern:
			if self._active_pattern_upload_sequence != self._pattern_upload_sequence:
				self._active_pattern_upload_sequence = self._pattern_upload_sequence
				self._logs_taken_for_active_pattern = 0
				self._last_log_time_s = 0.0
			if self._logs_taken_for_active_pattern >= n_max:
				return
		else:
			if self._pattern_upload_sequence <= 0:
				self._pattern_upload_sequence = 1
			if self._active_pattern_upload_sequence <= 0:
				self._active_pattern_upload_sequence = self._pattern_upload_sequence
			if self._logs_taken_for_active_pattern >= n_max:
				self._pattern_upload_sequence += 1
				self._active_pattern_upload_sequence = self._pattern_upload_sequence
				self._logs_taken_for_active_pattern = 0
				self._last_log_time_s = 0.0

		exposure_us = self._get_current_exposure_us()
		now_s = perf_counter()
		if np.isfinite(exposure_us) and self._logs_taken_for_active_pattern > 0:
			min_dt = max(1e-4, exposure_us * 1e-6)
			if now_s - self._last_log_time_s < min_dt:
				return

		x1, y1 = p1
		x2, y2 = p2
		i1, phi1 = self._compute_point_metrics(field, p1)
		i2, phi2 = self._compute_point_metrics(field, p2)
		dphi = wrap_phase_to_pi(phi1 - phi2)
		dphi_0_2pi = wrap_phase_to_2pi(dphi)
		if self._dphi_unwrapped_accum is None or self._last_logged_dphi_wrapped is None:
			dphi_unwrapped = dphi
		else:
			delta_unwrapped = wrap_phase_to_pi(dphi - self._last_logged_dphi_wrapped)
			dphi_unwrapped = self._dphi_unwrapped_accum + delta_unwrapped

		roi = self._fft_roi if self._fft_roi is not None else (None, None, None, None)
		pattern_idx = self._current_plm_pattern_index if has_plm_pattern else -1
		total_patterns = self._current_plm_total_patterns if has_plm_pattern else -1
		recovery_mode = "basic" if self.basic_recovery_radio.isChecked() else "stabilized"

		try:
			with self._log_file_path.open("a", newline="", encoding="utf-8") as handle:
				writer = csv.writer(handle)
				writer.writerow([
					datetime.now().isoformat(timespec="milliseconds"),
					frame_count,
					exposure_us,
					self._pattern_upload_sequence,
					self._logs_taken_for_active_pattern + 1,
					n_max,
					pattern_idx,
					total_patterns,
					self._plm_pattern_path,
					recovery_mode,
					roi[0],
					roi[1],
					roi[2],
					roi[3],
					x1,
					y1,
					x2,
					y2,
					i1,
					i2,
					phi1,
					phi2,
					dphi,
					dphi_0_2pi,
					dphi_unwrapped,
				])
			self._last_log_time_s = now_s
			self._last_logged_frame_count = frame_count
			self._last_logged_dphi_wrapped = dphi
			self._dphi_unwrapped_accum = dphi_unwrapped
			self._logs_taken_for_active_pattern += 1
			mode_text = "plm" if has_plm_pattern else "free-run"
			self.logging_status_label.setText(
				f"Logging: active ({mode_text}) | seq={self._active_pattern_upload_sequence}, sample {self._logs_taken_for_active_pattern}/{n_max}, frame={frame_count}"
			)
		except Exception as exc:
			self.logging_status_label.setText(f"Logging error: {exc}")
			self._stop_logging()

	def _update_phase_colorbar(self) -> None:
		bar_height = 256
		phase_values = np.linspace(np.pi, -np.pi, bar_height, dtype=np.float32)
		phase_column = phase_values[:, None]
		rgb_column = phase_to_rgb_uint8(phase_column)
		rgb_bar = np.repeat(rgb_column, 24, axis=1)
		rgb_bar_contiguous = np.ascontiguousarray(rgb_bar)
		bar_qimage = QImage(
			rgb_bar_contiguous.data,
			rgb_bar.shape[1],
			rgb_bar.shape[0],
			rgb_bar.shape[1] * 3,
			QImage.Format_RGB888,
		).copy()
		bar_pixmap = QPixmap.fromImage(bar_qimage)
		self.phase_colorbar_label.setPixmap(
			bar_pixmap.scaled(
				self.phase_colorbar_label.size(),
				Qt.IgnoreAspectRatio,
				Qt.SmoothTransformation,
			)
		)

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

	def _detect_trigger_delay_feature_name(self) -> Optional[str]:
		if hasattr(self.cam, "TriggerDelay"):
			return "TriggerDelay"
		if hasattr(self.cam, "TriggerDelayAbs"):
			return "TriggerDelayAbs"
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

	def _initialize_trigger_delay_controls(self) -> None:
		has_trigger_delay = self._trigger_delay_feature_name is not None
		self.trigger_delay_edit.setEnabled(has_trigger_delay)
		if not has_trigger_delay:
			self.trigger_delay_edit.setText("n/a")
			return

		try:
			current_delay_us = float(self._get_feature(self._trigger_delay_feature_name).get())
		except Exception:
			current_delay_us = 0.0

		self.trigger_delay_edit.setText(f"{max(0.0, current_delay_us) / 1000.0:.3f}")

	def _get_trigger_delay_us_from_ui(self) -> float:
		try:
			value_ms = float(self.trigger_delay_edit.text().strip())
		except Exception:
			value_ms = 0.0
		if value_ms < 0.0:
			value_ms = 0.0
		self.trigger_delay_edit.setText(f"{value_ms:.3f}")
		return value_ms * 1000.0

	def _on_trigger_delay_editing_finished(self) -> None:
		trigger_delay_us = self._get_trigger_delay_us_from_ui()
		if self._trigger_delay_feature_name is None:
			return
		if not self._streaming:
			return
		if str(self.trigger_mode_combo.currentData() or "freerun") != "line1":
			return
		try:
			self._get_feature(self._trigger_delay_feature_name).set(trigger_delay_us)
		except Exception as exc:
			self._set_status(f"Failed to set trigger delay: {exc}")

	def _start_live(self) -> None:
		if self._streaming:
			return
		try:
			with self._frame_lock:
				self._latest_frame = None
				self._frame_count = 0
				self._last_frame_error = None
			trigger_mode = str(self.trigger_mode_combo.currentData() or "freerun")
			if trigger_mode == "line1":
				if hasattr(self.cam, "TriggerSelector"):
					self.cam.TriggerSelector.set("FrameStart")
				if hasattr(self.cam, "TriggerSource"):
					self.cam.TriggerSource.set("Line1")
				if hasattr(self.cam, "TriggerActivation"):
					self.cam.TriggerActivation.set("RisingEdge")
				if hasattr(self.cam, "TriggerMode"):
					self.cam.TriggerMode.set("On")
				if self._trigger_delay_feature_name is not None:
					self._get_feature(self._trigger_delay_feature_name).set(
						self._get_trigger_delay_us_from_ui()
					)
			else:
				if hasattr(self.cam, "TriggerMode"):
					self.cam.TriggerMode.set("Off")
			if hasattr(self.cam, "AcquisitionMode"):
				self.cam.AcquisitionMode.set("Continuous")
			self.cam.start_streaming(handler=self._frame_handler, buffer_count=5)
			self._streaming = True
			self.timer.start()
			self.start_button.setEnabled(False)
			self.stop_button.setEnabled(True)
			if trigger_mode == "line1":
				self._set_status("Live view started (hardware trigger: Line1).")
			else:
				self._set_status("Live view started (trigger: FreeRun).")
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

	def _on_live_roi_changed(self, roi: Optional[tuple[int, int, int, int]]) -> None:
		self._live_roi = roi
		if roi is None:
			self.live_roi_status_label.setText("Live ROI: not set")
			self.live_roi_intensity_label.setText("Live ROI sum intensity: n/a")
			return
		x0, y0, x1, y1 = roi
		self.live_roi_status_label.setText(f"Live ROI selected: x={x0}:{x1}, y={y0}:{y1}")

	def _on_recovered_points_changed(
		self,
		point1: Optional[tuple[int, int]],
		point2: Optional[tuple[int, int]],
	) -> None:
		self._update_recovered_point_readout(point1, point2)

	def _get_phase_average_window_size(self) -> int:
		value = self.phase_avg_window_combo.currentData()
		try:
			window = int(value)
		except Exception:
			window = 1
		return max(1, window)

	def _compute_point_metrics(
		self,
		field: np.ndarray,
		point: tuple[int, int],
	) -> tuple[float, float]:
		x, y = point
		height, width = field.shape[:2]
		window = self._get_phase_average_window_size()
		half = window // 2

		x0 = max(0, x - half)
		x1 = min(width, x + half + 1)
		y0 = max(0, y - half)
		y1 = min(height, y + half + 1)

		patch = field[y0:y1, x0:x1]
		if patch.size == 0:
			patch = field[y : y + 1, x : x + 1]

		intensity = float(np.mean(np.abs(patch) ** 2))
		phase_unit = np.exp(1j * np.angle(patch))
		phase_vec = np.mean(phase_unit)
		if np.abs(phase_vec) <= 1e-12:
			phase = float(np.angle(np.mean(patch)))
		else:
			phase = float(np.angle(phase_vec))
		return intensity, phase

	def _update_recovered_point_readout(
		self,
		point1: Optional[tuple[int, int]],
		point2: Optional[tuple[int, int]],
	) -> None:
		field = self._latest_recovered_field
		if field is None:
			self.point1_label.setText("P1: not available")
			self.point2_label.setText("P2: not available")
			self.phase_diff_label.setText("Δφ(P1-P2): n/a")
			return

		phase1 = None
		phase2 = None
		window = self._get_phase_average_window_size()

		if point1 is not None:
			x1, y1 = point1
			intensity1, phase1 = self._compute_point_metrics(field, point1)
			self.point1_label.setText(
				f"P1 ({x1}, {y1}) [{window}x{window}]: Intensity={intensity1:.4g}, Phase={phase1:.4f} rad"
			)
		else:
			self.point1_label.setText("P1: not set")

		if point2 is not None:
			x2, y2 = point2
			intensity2, phase2 = self._compute_point_metrics(field, point2)
			self.point2_label.setText(
				f"P2 ({x2}, {y2}) [{window}x{window}]: Intensity={intensity2:.4g}, Phase={phase2:.4f} rad"
			)
		else:
			self.point2_label.setText("P2: not set")

		if phase1 is None or phase2 is None:
			self.phase_diff_label.setText("Δφ(P1-P2): n/a")
			return

		delta_phase = wrap_phase_to_pi(phase1 - phase2)
		self.phase_diff_label.setText(f"Δφ(P1-P2): {delta_phase:.4f} rad")
		self._append_live_phase_sample(phase1, phase2, wrap_phase_to_2pi(delta_phase))

	def _append_live_phase_sample(self, phi1: float, phi2: float, dphi_0_2pi: float) -> None:
		t_now = perf_counter() - self._phase_plot_t0
		self._phase_plot_time_s.append(t_now)
		self._phase_plot_phi1.append(float(phi1))
		self._phase_plot_phi2.append(float(phi2))
		self._phase_plot_dphi.append(float(dphi_0_2pi))

		times = list(self._phase_plot_time_s)
		phi1_vals = list(self._phase_plot_phi1)
		phi2_vals = list(self._phase_plot_phi2)
		dphi_vals = list(self._phase_plot_dphi)

		self.phase_plot_line_phi1.set_data(times, phi1_vals)
		self.phase_plot_line_phi2.set_data(times, phi2_vals)
		self.phase_plot_line_dphi.set_data(times, dphi_vals)

		if len(times) == 1:
			self.phase_plot_ax.set_xlim(0.0, max(5.0, times[0] + 1.0))
		else:
			t_max = times[-1]
			t_min = max(0.0, t_max - 20.0)
			self.phase_plot_ax.set_xlim(t_min, t_max + 0.2)

		self.phase_plot_canvas.draw_idle()

	def _append_live_intensity_sample(self, roi_sum_intensity: float) -> None:
		t_now = perf_counter() - self._intensity_plot_t0
		self._intensity_plot_time_s.append(t_now)
		self._intensity_plot_sum.append(float(roi_sum_intensity))

		times = list(self._intensity_plot_time_s)
		sums = list(self._intensity_plot_sum)
		self.intensity_plot_line.set_data(times, sums)

		if len(times) == 1:
			self.intensity_plot_ax.set_xlim(0.0, max(5.0, times[0] + 1.0))
			self.intensity_plot_ax.set_ylim(0.0, max(1.0, sums[0] * 1.2))
		else:
			t_max = times[-1]
			t_min = max(0.0, t_max - 20.0)
			self.intensity_plot_ax.set_xlim(t_min, t_max + 0.2)
			y_max = max(sums) if len(sums) > 0 else 1.0
			self.intensity_plot_ax.set_ylim(0.0, max(1.0, y_max * 1.1))

		self.intensity_plot_canvas.draw_idle()

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
			self._current_plm_pattern_index = None
			self._current_plm_total_patterns = None
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

			phase_levels, buckets, phase_values = get_phase_values_from_calibration(plm)
			n_states = int(len(phase_values))
			if n_states <= 0 or len(phase_levels) != n_states:
				raise RuntimeError("Invalid PLM calibration LUT.")

			interval_s = max(0.01, interval_ms / 1000.0)
			with ImageWindow(fullscreen=True, monitor=monitor) as win:
				frame_idx = 0
				last_update = perf_counter() - interval_s
				total_frames = int(phase_patterns.shape[0])
				cached_periods = (-1, -1)
				cached_ramp = np.zeros((expected_h, expected_w), dtype=np.float32)

				def loop_callback() -> None:
					nonlocal frame_idx, last_update, cached_periods, cached_ramp
					if self._plm_stop_event.is_set():
						raise EventLoopExit

					now = perf_counter()
					if now - last_update >= interval_s:
						display_idx = frame_idx
						periods = (int(self._grating_period_x_px), int(self._grating_period_y_px))
						if periods != cached_periods:
							cached_ramp = self._build_grating_phase_ramp(
								(expected_h, expected_w), periods[0], periods[1]
							)
							cached_periods = periods

						base_phase = np.mod(phase_patterns[display_idx], 2.0 * np.pi)
						overlaid_phase = np.mod(base_phase + cached_ramp, 2.0 * np.pi)
						state_indices = np.digitize(overlaid_phase, buckets, right=False).astype(np.int32)
						state_indices = np.clip(state_indices, 0, n_states - 1)
						phase_for_lut = phase_values[state_indices]
						bmp = plm.process_phase_map(phase_for_lut)
						bmp_image = Image.fromarray(bmp)

						win.clear()
						win.load(bmp_image)
						self.plm_counter_updated.emit(display_idx + 1, total_frames)
						frame_idx = (frame_idx + 1) % total_frames
						last_update = now

				win.loop_callback = loop_callback
				initial_state_indices = np.digitize(np.mod(phase_patterns[0], 2.0 * np.pi), buckets, right=False).astype(np.int32)
				initial_state_indices = np.clip(initial_state_indices, 0, n_states - 1)
				initial_phase = phase_values[initial_state_indices]
				win.load(Image.fromarray(plm.process_phase_map(initial_phase)))
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
		if not self._logging_enabled:
			self._current_plm_pattern_index = None
			self._current_plm_total_patterns = None
			self._set_pattern_counter(0, 0)

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

			if self._drop_frame_target_count > 0 and frame_count <= self._drop_frame_target_count:
				if self._current_plm_pattern_index is not None and self._current_plm_total_patterns is not None:
					remaining = self._drop_frame_target_count - frame_count + 1
					self._set_status(
						f"Dropping frame after pattern {self._current_plm_pattern_index}/{self._current_plm_total_patterns} (remaining: {remaining})."
					)
				return
			if self._drop_frame_target_count > 0 and frame_count > self._drop_frame_target_count:
				self._drop_frame_target_count = 0

			if image.ndim == 3:
				image = image[..., 0]

			self.image_label.set_image_array(image)

			if self._live_roi is not None:
				x0, y0, x1, y1 = self._live_roi
				roi_patch = image[y0:y1, x0:x1]
				if roi_patch.size > 0:
					roi_sum = float(np.sum(roi_patch, dtype=np.float64))
					roi_mean = float(np.mean(roi_patch))
					self.live_roi_intensity_label.setText(
						f"Live ROI sum intensity: {roi_sum:.3f} | mean={roi_mean:.3f}"
					)
					self._append_live_intensity_sample(roi_sum)

			fft_shifted, fft_image = self._compute_fft_shifted_and_magnitude_image(image)
			self.fft_image_label.set_image_array(fft_image)

			recovered_field = self._recover_off_axis_field(fft_shifted)
			if recovered_field is not None:
				self._latest_recovered_field = recovered_field
				amplitude = np.abs(recovered_field)
				amplitude_norm = amplitude / (float(np.max(amplitude)) + 1e-12)
				recovered_phase = np.angle(recovered_field)
				phase_rgb = phase_to_rgb_uint8(recovered_phase, value=amplitude_norm)
				self.recovered_field_label.set_image_rgb_array(phase_rgb)
				p1, p2 = self.recovered_field_label.get_points()
				self._update_recovered_point_readout(p1, p2)

				x0, y0, x1, y1 = self._fft_roi if self._fft_roi is not None else (0, 0, 0, 0)
				self.recovered_field_status_label.setText(
					f"Recovered phase (amplitude-weighted RGB) updated from ROI x={x0}:{x1}, y={y0}:{y1}."
				)
				self._maybe_log_current_measurement(frame_count)
			elif self._fft_roi is None:
				self._latest_recovered_field = None
				self.recovered_field_status_label.setText("Select FFT ROI to recover amplitude-weighted phase.")
				self.point1_label.setText("P1: not available")
				self.point2_label.setText("P2: not available")
				self.phase_diff_label.setText("Δφ(P1-P2): n/a")
			else:
				self._latest_recovered_field = None
				self.recovered_field_status_label.setText("ROI too small or invalid for recovery.")
				self.point1_label.setText("P1: not available")
				self.point2_label.setText("P2: not available")
				self.phase_diff_label.setText("Δφ(P1-P2): n/a")

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


def get_phase_values_from_calibration(plm: PLM) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	phase_min, phase_max = plm.phase_range
	n_states = len(plm.displacement_ratios)

	ratio_scale = (
		plm.max_displacement_ratio
		if plm.max_displacement_ratio is not None
		else (n_states - 1) / n_states
	)

	phase_levels = phase_min + plm.displacement_ratios * ratio_scale * (phase_max - phase_min)
	phase_disp = np.hstack([phase_levels, phase_max])
	buckets = (phase_disp[:-1] + phase_disp[1:]) / 2.0

	representative_values = np.empty(n_states, dtype=np.float64)
	representative_values[0] = phase_min + 1e-9
	if n_states > 1:
		representative_values[1:] = 0.5 * (buckets[:-1] + buckets[1:])

	return phase_levels, buckets, representative_values


def principal_phase_diff(value1: complex, value2: complex) -> float:
	return float(np.angle(value1 * np.conj(value2)))
	#return float(np.angle(value1) - np.angle(value2))

def wrap_phase_to_pi(phase_rad: float) -> float:
	return float((phase_rad + np.pi) % (2.0 * np.pi) - np.pi)


def wrap_phase_to_2pi(phase_rad: float) -> float:
	return float((phase_rad + 2.0 * np.pi) % (2.0 * np.pi))


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
