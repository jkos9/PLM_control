import sys

import importlib

import threading

import csv

import math

from collections import deque

from pathlib import Path

from datetime import datetime

from time import perf_counter

from typing import Optional


import numpy as np

import matplotlib as mpl

from matplotlib.figure import Figure

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PIL import Image

from vmbpy import PixelFormat, VmbSystem


try:

    import torch

    import torch.nn.functional as F

except Exception:

    torch = None

    F = None


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

    QGridLayout = QtWidgets.QGridLayout

    QHBoxLayout = QtWidgets.QHBoxLayout

    QLabel = QtWidgets.QLabel

    QLineEdit = QtWidgets.QLineEdit

    QMainWindow = QtWidgets.QMainWindow

    QPushButton = QtWidgets.QPushButton

    QRadioButton = QtWidgets.QRadioButton

    QScrollArea = QtWidgets.QScrollArea

    QSplitter = QtWidgets.QSplitter

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

    zernike_optimizer_status_updated = QtCore.pyqtSignal(str)

    zernike_optimizer_finished = QtCore.pyqtSignal(bool, str)

    zernike_coefficients_apply_requested = QtCore.pyqtSignal(object)


    def __init__(self, cam):

        super().__init__()

        self.cam = cam

        self.setWindowTitle("Allied Vision Live Camera")

        self.resize(1400, 800)


        self.image_label = RoiSelectableImageLabel("No frame yet")

        self.image_label.setAlignment(Qt.AlignCenter)

        self.image_label.setMinimumSize(320, 240)

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

        self._zernike_terms = self._enumerate_zernike_terms(5)

        self._zernike_coeff_spins: dict[tuple[int, int], QDoubleSpinBox] = {}

        self._zernike_row_widgets: dict[tuple[int, int], tuple[QLabel, QDoubleSpinBox]] = {}

        self.zernike_status_label = QLabel("Active Zernike terms: 0")

        self.zernike_reset_button = QPushButton("Reset all to 0")

        self.zernike_max_degree_combo = QComboBox()

        for degree in range(0, 6):

            self.zernike_max_degree_combo.addItem(str(degree), degree)

        self.zernike_max_degree_combo.setCurrentIndex(5)

        self.zernike_opt_iterations_spin = QSpinBox()

        self.zernike_opt_iterations_spin.setRange(1, 1000)

        self.zernike_opt_iterations_spin.setValue(25)

        self.zernike_opt_step_spin = QDoubleSpinBox()

        self.zernike_opt_step_spin.setDecimals(4)

        self.zernike_opt_step_spin.setRange(0.0001, 10.0)

        self.zernike_opt_step_spin.setSingleStep(0.01)

        self.zernike_opt_step_spin.setValue(0.10)

        self.zernike_opt_start_button = QPushButton("Start Optimizer")

        self.zernike_opt_stop_button = QPushButton("Stop Optimizer")

        self.zernike_opt_stop_button.setEnabled(False)

        self.zernike_auto_optimize_checkbox = QCheckBox("Continuous optimize (update every PLM pattern change)")

        self.zernike_opt_results_path_edit = QLineEdit(str(Path.cwd() / "zernike_optimization_coefficients.txt"))

        self.zernike_opt_results_browse_button = QPushButton("Browse Save")

        self.zernike_opt_status_label = QLabel("Optimizer: idle")


        self.fft_image_label = RoiSelectableImageLabel("No FFT yet")

        self.fft_image_label.setAlignment(Qt.AlignCenter)

        self.fft_image_label.setMinimumSize(320, 240)

        self.fft_image_label.on_roi_changed = self._on_fft_roi_changed


        self.fft_status_label = QLabel("Waiting for frames...")


        self.recovered_field_label = PointSelectableImageLabel("Select FFT ROI to recover field")

        self.recovered_field_label.setAlignment(Qt.AlignCenter)

        self.recovered_field_label.setMinimumSize(480, 320)

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

        self.colormap_viridis_radio = QRadioButton("Viridis")

        self.colormap_plasma_radio = QRadioButton("Plasma")

        self.colormap_viridis_radio.setChecked(True)

        self.colormap_group = QButtonGroup(self)

        self.colormap_group.addButton(self.colormap_viridis_radio)

        self.colormap_group.addButton(self.colormap_plasma_radio)


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

        recovery_mode_row.addSpacing(16)

        recovery_mode_row.addWidget(QLabel("Colormap:"))

        recovery_mode_row.addWidget(self.colormap_viridis_radio)

        recovery_mode_row.addWidget(self.colormap_plasma_radio)

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


        pattern_gallery_tab = QWidget()

        pattern_gallery_layout = QVBoxLayout()

        self.pattern_gallery_status_label = QLabel("Load PLM patterns to start gallery capture.")

        self.pattern_fields_save_checkbox = QCheckBox("Save recovered fields each completed loop")

        self.pattern_fields_save_path_edit = QLineEdit(str(Path.cwd() / "pattern_fields_history.pt"))

        self.pattern_fields_save_browse_button = QPushButton("Browse Save")

        self.pattern_fields_save_now_button = QPushButton("Save Fields Now")

        pattern_fields_save_row = QHBoxLayout()

        pattern_fields_save_row.addWidget(self.pattern_fields_save_checkbox)

        pattern_fields_save_row.addWidget(self.pattern_fields_save_path_edit, 1)

        pattern_fields_save_row.addWidget(self.pattern_fields_save_browse_button)

        pattern_fields_save_row.addWidget(self.pattern_fields_save_now_button)

        self.pattern_gallery_scroll = QScrollArea()

        self.pattern_gallery_scroll.setWidgetResizable(True)

        self.pattern_gallery_container = QWidget()

        self.pattern_gallery_grid = QGridLayout()

        self.pattern_gallery_grid.setContentsMargins(6, 6, 6, 6)

        self.pattern_gallery_grid.setHorizontalSpacing(10)

        self.pattern_gallery_grid.setVerticalSpacing(10)

        self.pattern_gallery_container.setLayout(self.pattern_gallery_grid)

        self.pattern_gallery_scroll.setWidget(self.pattern_gallery_container)

        pattern_gallery_layout.addWidget(self.pattern_gallery_status_label)

        pattern_gallery_layout.addLayout(pattern_fields_save_row)

        pattern_gallery_layout.addWidget(self.pattern_gallery_scroll, 1)

        pattern_gallery_tab.setLayout(pattern_gallery_layout)


        channel_matrix_tab = QWidget()

        channel_matrix_layout = QVBoxLayout()

        self.channel_matrix_status_label = QLabel("Channel matrix: waiting for target modes and recovered modes.")

        self.channel_matrix_xt_label = QLabel("XT (dB): n/a")

        self.channel_matrix_save_checkbox = QCheckBox("Save channel matrix each completed loop")

        self.channel_matrix_save_path_edit = QLineEdit(str(Path.cwd() / "channel_matrix_history.npz"))

        self.channel_matrix_save_browse_button = QPushButton("Browse Save")

        self.channel_matrix_save_now_button = QPushButton("Save Now")

        self.channel_matrix_canvas = FigureCanvas(Figure(figsize=(5.0, 5.0)))

        channel_matrix_layout.addWidget(self.channel_matrix_status_label)

        channel_matrix_layout.addWidget(self.channel_matrix_xt_label)

        save_row = QHBoxLayout()

        save_row.addWidget(self.channel_matrix_save_checkbox)

        save_row.addWidget(self.channel_matrix_save_path_edit, 1)

        save_row.addWidget(self.channel_matrix_save_browse_button)

        save_row.addWidget(self.channel_matrix_save_now_button)

        channel_matrix_layout.addLayout(save_row)

        channel_matrix_layout.addWidget(self.channel_matrix_canvas, 1)

        channel_matrix_tab.setLayout(channel_matrix_layout)


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


        zernike_tab = QWidget()

        zernike_layout = QVBoxLayout()

        zernike_layout.addWidget(QLabel("Zernike polynomial coefficients (radians), radial degree n ≤ 5"))

        zernike_degree_row = QHBoxLayout()

        zernike_degree_row.addWidget(QLabel("Show terms up to radial degree n:"))

        zernike_degree_row.addWidget(self.zernike_max_degree_combo)

        zernike_degree_row.addStretch(1)

        zernike_layout.addLayout(zernike_degree_row)

        zernike_layout.addWidget(self.zernike_status_label)

        zernike_controls_scroll = QScrollArea()

        zernike_controls_scroll.setWidgetResizable(True)

        zernike_controls_container = QWidget()

        zernike_controls_layout = QGridLayout()

        zernike_controls_layout.setContentsMargins(6, 6, 6, 6)

        zernike_controls_layout.setHorizontalSpacing(10)

        zernike_controls_layout.setVerticalSpacing(6)

        for row_idx, (n, m) in enumerate(self._zernike_terms):

            alias = self._zernike_alias(n, m)

            label_text = f"Z({n},{m})" if alias == "" else f"Z({n},{m}) - {alias}"

            label = QLabel(label_text)

            spin = QDoubleSpinBox()

            spin.setDecimals(4)

            spin.setRange(-20.0, 20.0)

            spin.setSingleStep(0.01)

            spin.setValue(0.0)

            spin.setSuffix(" rad")

            spin.valueChanged.connect(self._on_zernike_coeff_changed)

            self._zernike_coeff_spins[(n, m)] = spin

            self._zernike_row_widgets[(n, m)] = (label, spin)

            zernike_controls_layout.addWidget(label, row_idx, 0)

            zernike_controls_layout.addWidget(spin, row_idx, 1)

        zernike_controls_container.setLayout(zernike_controls_layout)

        zernike_controls_scroll.setWidget(zernike_controls_container)

        zernike_layout.addWidget(zernike_controls_scroll, 1)

        zernike_opt_row = QHBoxLayout()

        zernike_opt_row.addWidget(QLabel("Iterations:"))

        zernike_opt_row.addWidget(self.zernike_opt_iterations_spin)

        zernike_opt_row.addSpacing(12)

        zernike_opt_row.addWidget(QLabel("Step (rad):"))

        zernike_opt_row.addWidget(self.zernike_opt_step_spin)

        zernike_opt_row.addSpacing(12)

        zernike_opt_row.addWidget(self.zernike_opt_start_button)

        zernike_opt_row.addWidget(self.zernike_opt_stop_button)

        zernike_opt_row.addStretch(1)

        zernike_layout.addLayout(zernike_opt_row)

        zernike_save_row = QHBoxLayout()

        zernike_save_row.addWidget(QLabel("Optimizer result file:"))

        zernike_save_row.addWidget(self.zernike_opt_results_path_edit, 1)

        zernike_save_row.addWidget(self.zernike_opt_results_browse_button)

        zernike_layout.addLayout(zernike_save_row)

        zernike_layout.addWidget(self.zernike_auto_optimize_checkbox)

        zernike_layout.addWidget(self.zernike_opt_status_label)

        zernike_layout.addWidget(self.zernike_reset_button, alignment=Qt.AlignLeft)

        zernike_tab.setLayout(zernike_layout)


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

        self.tab_widget.addTab(phase_plot_tab, "Phase Plot")

        self.tab_widget.addTab(intensity_plot_tab, "Intensity Plot")

        self.tab_widget.addTab(zernike_tab, "Zernike Control")

        self.tab_widget.addTab(pattern_tab, "PLM Patterns")

        recovered_tab.setMinimumWidth(420)

        self.right_tab_widget = QTabWidget()

        self.right_tab_widget.addTab(recovered_tab, "Recovered Field")

        self.right_tab_widget.addTab(pattern_gallery_tab, "Pattern Gallery")

        self.right_tab_widget.addTab(channel_matrix_tab, "Channel Matrix")


        controls = QHBoxLayout()

        controls.setContentsMargins(0, 0, 0, 0)

        controls.setSpacing(8)

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

        root.setContentsMargins(8, 6, 8, 8)

        root.setSpacing(6)

        root.addLayout(controls)

        self.main_splitter = QSplitter(Qt.Horizontal)

        self.main_splitter.addWidget(self.tab_widget)

        self.main_splitter.addWidget(self.right_tab_widget)

        self.main_splitter.setChildrenCollapsible(False)

        self.main_splitter.setStretchFactor(0, 1)

        self.main_splitter.setStretchFactor(1, 1)

        self.main_splitter.setSizes([700, 700])

        root.addWidget(self.main_splitter, 1)


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

        self._zernike_phase_cache_shape: Optional[tuple[int, int]] = None

        self._zernike_phase_cache_coeffs: Optional[tuple[float, ...]] = None

        self._zernike_phase_cache_map: Optional[np.ndarray] = None

        self._zernike_optimizer_thread: Optional[threading.Thread] = None

        self._zernike_optimizer_stop_event = threading.Event()

        self._zernike_optimizer_metric_condition = threading.Condition()

        self._continuous_opt_best_xt_db: Optional[float] = None

        self._continuous_opt_best_coeffs: Optional[dict[tuple[int, int], float]] = None

        self._continuous_opt_last_candidate_coeffs: Optional[dict[tuple[int, int], float]] = None

        self._continuous_opt_last_step_sequence: int = -1

        self._continuous_opt_step_count: int = 0

        self._last_processed_frame_count = -1

        self._drop_frames_remaining = 0

        self._drop_frames_start_count = -1

        self._last_pattern_switch_time_s = 0.0

        self._pattern_preview_labels: dict[int, QLabel] = {}

        self._pattern_preview_pixmaps: dict[int, QPixmap] = {}

        self._pattern_preview_pending_index: Optional[int] = None

        self._pattern_preview_pending_sequence = 0

        self._pattern_fields_by_pattern: dict[int, np.ndarray] = {}

        self._current_pattern_fields_loop: Optional[np.ndarray] = None

        self._target_modes_path = Path.cwd() / "data" / "lp_modes_camera.npy"

        self._target_modes: Optional[np.ndarray] = None

        self._recovered_modes_by_pattern: dict[int, np.ndarray] = {}

        self._channel_matrix: Optional[np.ndarray] = None

        self._channel_matrix_xt_db: Optional[float] = None

        self._channel_matrix_dirty = False

        self._channel_matrix_loop_history: list[np.ndarray] = []

        self._channel_matrix_xt_history_db: list[float] = []

        self._completed_pattern_loops = 0

        self._last_plm_counter_index: Optional[int] = None

        self._phase_colormap_name = "viridis"

        self._phase_plot_dirty = False

        self._intensity_plot_dirty = False

        self._plot_refresh_timer = QTimer(self)

        self._plot_refresh_timer.setInterval(100)

        self._plot_refresh_timer.timeout.connect(self._refresh_plots_if_dirty)

        self._plot_refresh_timer.start()


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

        self.zernike_reset_button.clicked.connect(self._reset_zernike_coefficients)

        self.zernike_max_degree_combo.currentIndexChanged.connect(self._on_zernike_degree_limit_changed)

        self.zernike_opt_start_button.clicked.connect(self._start_zernike_optimizer)

        self.zernike_opt_stop_button.clicked.connect(self._stop_zernike_optimizer)

        self.zernike_opt_results_browse_button.clicked.connect(self._on_browse_zernike_opt_results_file)

        self.zernike_auto_optimize_checkbox.toggled.connect(self._on_zernike_continuous_opt_toggled)

        self.zernike_optimizer_status_updated.connect(self._set_zernike_optimizer_status)

        self.zernike_optimizer_finished.connect(self._on_zernike_optimizer_finished)

        self.zernike_coefficients_apply_requested.connect(self._apply_zernike_coefficients)

        self.colormap_viridis_radio.toggled.connect(self._on_colormap_toggled)

        self.colormap_plasma_radio.toggled.connect(self._on_colormap_toggled)

        self.tab_widget.currentChanged.connect(self._on_left_tab_changed)

        self.right_tab_widget.currentChanged.connect(self._on_right_tab_changed)

        self.channel_matrix_save_browse_button.clicked.connect(self._on_browse_channel_matrix_save_file)

        self.channel_matrix_save_now_button.clicked.connect(self._on_save_channel_matrix_now)

        self.pattern_fields_save_browse_button.clicked.connect(self._on_browse_pattern_fields_save_file)

        self.pattern_fields_save_now_button.clicked.connect(self._on_save_pattern_fields_now)


        self._exp_auto_feature_name: Optional[str] = self._detect_exposure_auto_feature_name()

        self._exp_time_feature_name: Optional[str] = self._detect_exposure_time_feature_name()

        self._trigger_delay_feature_name: Optional[str] = self._detect_trigger_delay_feature_name()


        self._initialize_exposure_controls()

        self._initialize_trigger_delay_controls()

        self._on_grating_input_changed()

        self._update_zernike_visibility()

        self._on_zernike_coeff_changed(0.0)

        self._update_phase_colorbar()

        self._load_default_target_modes()

        self._refresh_channel_matrix_plot(force=True)


    def _on_left_tab_changed(self, _index: int) -> None:

        self._refresh_plots_if_dirty(force=True)


    def _on_right_tab_changed(self, _index: int) -> None:

        self._refresh_plots_if_dirty(force=True)


    def _on_colormap_toggled(self, _checked: bool) -> None:

        self._phase_colormap_name = "plasma" if self.colormap_plasma_radio.isChecked() else "viridis"

        self._update_phase_colorbar()

        if self._latest_recovered_field is not None:

            amplitude = np.abs(self._latest_recovered_field)

            amplitude_norm = amplitude / (float(np.max(amplitude)) + 1e-12)

            recovered_phase = np.angle(self._latest_recovered_field)

            phase_rgb = phase_to_rgb_uint8(

                recovered_phase,

                value=amplitude_norm,

                cmap_name=self._phase_colormap_name,

            )

            self.recovered_field_label.set_image_rgb_array(phase_rgb)

        self._channel_matrix_dirty = True

        self._refresh_plots_if_dirty(force=True)


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


    def _on_zernike_coeff_changed(self, _value: float) -> None:

        enabled_terms = self._get_enabled_zernike_terms()

        active_terms = 0

        for term in enabled_terms:

            spin = self._zernike_coeff_spins[term]

            if abs(float(spin.value())) > 1e-9:

                active_terms += 1

        self.zernike_status_label.setText(

            f"Active Zernike terms: {active_terms} / {len(enabled_terms)} (shown)"

        )

        self._zernike_phase_cache_shape = None

        self._zernike_phase_cache_coeffs = None

        self._zernike_phase_cache_map = None


    def _on_zernike_degree_limit_changed(self, _index: int) -> None:

        self._update_zernike_visibility()

        self._on_zernike_coeff_changed(0.0)


    def _zernike_alias(self, n: int, m: int) -> str:

        alias_map = {

            (0, 0): "Piston",

            (1, -1): "Tilt Y",

            (1, 1): "Tilt X",

            (2, -2): "Astigmatism 45°",

            (2, 0): "Defocus",

            (2, 2): "Astigmatism 0°",

            (3, -3): "Trefoil Y",

            (3, -1): "Coma Y",

            (3, 1): "Coma X",

            (3, 3): "Trefoil X",

            (4, -4): "Quadrafoil 45°",

            (4, -2): "Secondary Astigmatism 45°",

            (4, 0): "Primary Spherical",

            (4, 2): "Secondary Astigmatism 0°",

            (4, 4): "Quadrafoil 0°",

            (5, -5): "Pentafoil Y",

            (5, -3): "Secondary Trefoil Y",

            (5, -1): "Secondary Coma Y",

            (5, 1): "Secondary Coma X",

            (5, 3): "Secondary Trefoil X",

            (5, 5): "Pentafoil X",

        }

        return str(alias_map.get((int(n), int(m)), ""))


    def _get_selected_zernike_max_degree(self) -> int:

        value = self.zernike_max_degree_combo.currentData()

        try:

            return max(0, min(5, int(value)))

        except Exception:

            return 5


    def _get_enabled_zernike_terms(self) -> list[tuple[int, int]]:

        max_degree = self._get_selected_zernike_max_degree()

        return [term for term in self._zernike_terms if term[0] <= max_degree]


    def _update_zernike_visibility(self) -> None:

        max_degree = self._get_selected_zernike_max_degree()

        for (n, _m), widgets in self._zernike_row_widgets.items():

            visible = int(n) <= max_degree

            widgets[0].setVisible(visible)

            widgets[1].setVisible(visible)


    def _reset_zernike_coefficients(self) -> None:

        for spin in self._zernike_coeff_spins.values():

            spin.blockSignals(True)

            spin.setValue(0.0)

            spin.blockSignals(False)

        self._on_zernike_coeff_changed(0.0)


    def _set_zernike_optimizer_status(self, message: str) -> None:

        self.zernike_opt_status_label.setText(message)


    def _reset_continuous_optimizer_state(self) -> None:

        self._continuous_opt_best_xt_db = None

        self._continuous_opt_best_coeffs = None

        self._continuous_opt_last_candidate_coeffs = None

        self._continuous_opt_last_step_sequence = -1

        self._continuous_opt_step_count = 0


    def _on_zernike_continuous_opt_toggled(self, checked: bool) -> None:

        self._reset_continuous_optimizer_state()

        if checked:

            self._set_zernike_optimizer_status("Optimizer: continuous mode enabled")

        else:

            self._set_zernike_optimizer_status("Optimizer: continuous mode disabled")


    def _capture_zernike_coefficients_snapshot(self) -> dict[tuple[int, int], float]:

        coeffs: dict[tuple[int, int], float] = {}

        for term, spin in self._zernike_coeff_spins.items():

            coeffs[term] = float(spin.value())

        return coeffs


    def _on_browse_zernike_opt_results_file(self) -> None:

        file_path, _ = QFileDialog.getSaveFileName(

            self,

            "Select optimizer coefficient output file",

            self.zernike_opt_results_path_edit.text().strip() or str(Path.cwd() / "zernike_optimization_coefficients.txt"),

            "Text files (*.txt);;All files (*)",

        )

        if not file_path:

            return

        self.zernike_opt_results_path_edit.setText(file_path)


    def _save_zernike_coefficients_to_text(

        self,

        coeffs: dict[tuple[int, int], float],

        best_xt_db: float,

        enabled_terms: list[tuple[int, int]],

        max_degree: int,

        output_path_text: str,

    ) -> tuple[bool, str]:

        path_text = str(output_path_text).strip()

        if not path_text:

            return False, "Optimizer save failed: no output file path"

        output_path = Path(path_text)

        try:

            output_path.parent.mkdir(parents=True, exist_ok=True)

            lines: list[str] = []

            lines.append(f"timestamp_iso={datetime.now().isoformat()}")

            lines.append(f"best_xt_db={best_xt_db:.6f}")

            lines.append(f"max_radial_degree={int(max_degree)}")

            lines.append(f"enabled_terms_count={len(enabled_terms)}")

            lines.append("")

            for n, m in sorted(coeffs.keys(), key=lambda item: (item[0], item[1])):

                alias = self._zernike_alias(n, m)

                alias_text = f" # {alias}" if alias else ""

                lines.append(f"Z({n},{m})={float(coeffs[(n, m)]):+.6f}{alias_text}")

            output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            return True, str(output_path)

        except Exception as exc:

            return False, f"{exc}"


    def _apply_zernike_coefficients(self, coeffs: object) -> None:

        if not isinstance(coeffs, dict):

            return

        for term, spin in self._zernike_coeff_spins.items():

            value = float(coeffs.get(term, spin.value()))

            spin.blockSignals(True)

            spin.setValue(max(-20.0, min(20.0, value)))

            spin.blockSignals(False)

        self._on_zernike_coeff_changed(0.0)


    def _wait_for_next_loop_xt_metric(self, last_seen_loop: int, timeout_s: float = 60.0) -> Optional[tuple[int, float]]:

        deadline = perf_counter() + max(0.1, float(timeout_s))

        while not self._zernike_optimizer_stop_event.is_set():

            with self._zernike_optimizer_metric_condition:

                loop_now = int(self._completed_pattern_loops)

                xt_now = self._channel_matrix_xt_db

                if loop_now > int(last_seen_loop) and xt_now is not None and np.isfinite(float(xt_now)):

                    return loop_now, float(xt_now)

                remaining = deadline - perf_counter()

                if remaining <= 0.0:

                    return None

                self._zernike_optimizer_metric_condition.wait(timeout=min(0.5, remaining))

        return None


    def _start_zernike_optimizer(self) -> None:

        if self.zernike_auto_optimize_checkbox.isChecked():

            self._set_zernike_optimizer_status(

                "Optimizer: disable continuous checkbox to run finite-iteration optimizer"

            )

            return

        if self._zernike_optimizer_thread is not None and self._zernike_optimizer_thread.is_alive():

            self._set_zernike_optimizer_status("Optimizer: already running")

            return

        if self._plm_loop_thread is None or not self._plm_loop_thread.is_alive():

            self._set_zernike_optimizer_status("Optimizer: start PLM loop first")

            return

        enabled_terms = self._get_enabled_zernike_terms()

        if len(enabled_terms) <= 0:

            self._set_zernike_optimizer_status("Optimizer: no enabled Zernike terms")

            return


        iterations = int(self.zernike_opt_iterations_spin.value())

        step = float(self.zernike_opt_step_spin.value())

        max_degree = self._get_selected_zernike_max_degree()

        results_path = self.zernike_opt_results_path_edit.text().strip()

        initial_coeffs = self._capture_zernike_coefficients_snapshot()

        self._zernike_optimizer_stop_event.clear()

        self.zernike_opt_start_button.setEnabled(False)

        self.zernike_opt_stop_button.setEnabled(True)

        self._set_zernike_optimizer_status("Optimizer: running")


        self._zernike_optimizer_thread = threading.Thread(

            target=self._run_zernike_optimizer,

            args=(enabled_terms, initial_coeffs, iterations, step, max_degree, results_path),

            daemon=True,

        )

        self._zernike_optimizer_thread.start()


    def _maybe_step_continuous_zernike_optimizer(self) -> None:

        if not self.zernike_auto_optimize_checkbox.isChecked():

            return

        if self._zernike_optimizer_thread is not None and self._zernike_optimizer_thread.is_alive():

            return

        if self._active_pattern_upload_sequence == self._continuous_opt_last_step_sequence:

            return

        if self._channel_matrix_xt_db is None or not np.isfinite(float(self._channel_matrix_xt_db)):

            return


        enabled_terms = self._get_enabled_zernike_terms()

        if len(enabled_terms) <= 0:

            return


        current_xt_db = float(self._channel_matrix_xt_db)

        if self._continuous_opt_best_coeffs is None:

            self._continuous_opt_best_coeffs = self._capture_zernike_coefficients_snapshot()

            self._continuous_opt_best_xt_db = current_xt_db

        elif self._continuous_opt_last_candidate_coeffs is not None:

            if self._continuous_opt_best_xt_db is None or current_xt_db < float(self._continuous_opt_best_xt_db):

                self._continuous_opt_best_xt_db = current_xt_db

                self._continuous_opt_best_coeffs = dict(self._continuous_opt_last_candidate_coeffs)

            else:

                self._apply_zernike_coefficients(self._continuous_opt_best_coeffs)


        step = max(1e-4, float(self.zernike_opt_step_spin.value()))

        rng = np.random.default_rng()

        candidate = dict(self._continuous_opt_best_coeffs)

        for term in enabled_terms:

            candidate[term] = float(np.clip(candidate[term] + rng.normal(0.0, step), -20.0, 20.0))


        self._apply_zernike_coefficients(candidate)

        self._continuous_opt_last_candidate_coeffs = dict(candidate)

        self._continuous_opt_last_step_sequence = int(self._active_pattern_upload_sequence)

        self._continuous_opt_step_count += 1

        best_text = "n/a" if self._continuous_opt_best_xt_db is None else f"{self._continuous_opt_best_xt_db:.3f} dB"

        self._set_zernike_optimizer_status(


            f"Optimizer: continuous step{self._continuous_opt_step_count}, current XT={current_xt_db:.3f} dB,best XT={best_text}")


    def _stop_zernike_optimizer(self) -> None:

        if self._zernike_optimizer_thread is None or not self._zernike_optimizer_thread.is_alive():

            self._set_zernike_optimizer_status("Optimizer: not running")

            return

        self._zernike_optimizer_stop_event.set()

        self._set_zernike_optimizer_status("Optimizer: stopping...")


    def _on_zernike_optimizer_finished(self, _success: bool, message: str) -> None:

        self.zernike_opt_start_button.setEnabled(True)

        self.zernike_opt_stop_button.setEnabled(False)

        self._set_zernike_optimizer_status(message)


    def _run_zernike_optimizer(

        self,

        enabled_terms: list[tuple[int, int]],

        initial_coeffs: dict[tuple[int, int], float],

        iterations: int,

        step: float,

        max_degree: int,

        results_path: str,

    ) -> None:

        try:

            local_rng = np.random.default_rng()

            best_coeffs = dict(initial_coeffs)

            last_loop = int(self._completed_pattern_loops)


            self.zernike_optimizer_status_updated.emit("Optimizer: baseline evaluation")

            baseline_metric = self._wait_for_next_loop_xt_metric(last_loop, timeout_s=90.0)

            if baseline_metric is None:

                self.zernike_optimizer_finished.emit(False, "Optimizer stopped: no loop metric")

                return

            last_loop, best_xt_db = baseline_metric


            step_now = max(1e-4, float(step))

            for it in range(1, int(iterations) + 1):

                if self._zernike_optimizer_stop_event.is_set():

                    self.zernike_optimizer_finished.emit(False, "Optimizer stopped by user")

                    return


                candidate = dict(best_coeffs)

                for term in enabled_terms:

                    candidate[term] = float(np.clip(

                        candidate[term] + local_rng.normal(0.0, step_now),

                        -20.0,

                        20.0,

                    ))


                self.zernike_coefficients_apply_requested.emit(candidate)

                metric = self._wait_for_next_loop_xt_metric(last_loop, timeout_s=90.0)

                if metric is None:

                    self.zernike_optimizer_finished.emit(False, "Optimizer stopped: timed out waiting for XT")

                    return

                last_loop, candidate_xt_db = metric


                if candidate_xt_db < best_xt_db:

                    best_xt_db = candidate_xt_db

                    best_coeffs = candidate

                    step_now = max(1e-4, step_now * 1.03)

                    self.zernike_optimizer_status_updated.emit(

                        f"Optimizer: iter {it}/{iterations}, improved XT={best_xt_db:.3f} dB"

                    )

                else:

                    step_now = max(1e-4, step_now * 0.97)

                    self.zernike_optimizer_status_updated.emit(

                        f"Optimizer: iter {it}/{iterations}, XT={candidate_xt_db:.3f} dB (best {best_xt_db:.3f} dB)"

                    )


            self.zernike_coefficients_apply_requested.emit(best_coeffs)

            saved_ok, save_info = self._save_zernike_coefficients_to_text(

                best_coeffs,

                float(best_xt_db),

                enabled_terms,

                int(max_degree),

                str(results_path),

            )

            if saved_ok:

                self.zernike_optimizer_finished.emit(

                    True,

                    f"Optimizer finished: best XT={best_xt_db:.3f} dB | saved {save_info}",

                )

            else:

                self.zernike_optimizer_finished.emit(

                    True,

                    f"Optimizer finished: best XT={best_xt_db:.3f} dB | save failed: {save_info}",

                )

        except Exception as exc:

            self.zernike_optimizer_finished.emit(False, f"Optimizer error: {exc}")


    def _enumerate_zernike_terms(self, max_radial_degree: int) -> list[tuple[int, int]]:

        terms: list[tuple[int, int]] = []

        for n in range(max(0, int(max_radial_degree)) + 1):

            for m in range(-n, n + 1, 2):

                terms.append((n, m))

        return terms


    def _zernike_radial_polynomial(self, n: int, m_abs: int, r: np.ndarray) -> np.ndarray:

        if (n - m_abs) % 2 != 0:

            return np.zeros_like(r, dtype=np.float32)

        radial = np.zeros_like(r, dtype=np.float64)

        k_max = (n - m_abs) // 2

        for k in range(k_max + 1):

            numerator = math.factorial(n - k)

            denominator = (

                math.factorial(k)

                * math.factorial((n + m_abs) // 2 - k)

                * math.factorial((n - m_abs) // 2 - k)

            )

            coefficient = ((-1) ** k) * numerator / float(denominator)

            radial += coefficient * np.power(r, n - 2 * k)

        return radial.astype(np.float32)


    def _build_zernike_phase_map(self, shape: tuple[int, int]) -> np.ndarray:

        height, width = int(shape[0]), int(shape[1])

        enabled_terms = set(self._get_enabled_zernike_terms())

        max_degree = self._get_selected_zernike_max_degree()

        coeff_values = tuple(

            round(float(self._zernike_coeff_spins[(n, m)].value()), 6) if (n, m) in enabled_terms else 0.0

            for (n, m) in self._zernike_terms

        )

        if (

            self._zernike_phase_cache_map is not None

            and self._zernike_phase_cache_shape == (height, width)

            and self._zernike_phase_cache_coeffs == (max_degree, *coeff_values)

        ):

            return self._zernike_phase_cache_map


        y = np.linspace(-1.0, 1.0, height, dtype=np.float32)

        x = np.linspace(-1.0, 1.0, width, dtype=np.float32)

        yy, xx = np.meshgrid(y, x, indexing="ij")

        r = np.sqrt(xx * xx + yy * yy)

        theta = np.arctan2(yy, xx)

        aperture = r <= 1.0

        phase = np.zeros((height, width), dtype=np.float32)


        for coeff, (n, m) in zip(coeff_values, self._zernike_terms):

            if abs(coeff) <= 1e-9:

                continue

            radial = self._zernike_radial_polynomial(n, abs(m), r)

            if m == 0:

                mode = radial

            elif m > 0:

                mode = radial * np.cos(float(m) * theta)

            else:

                mode = radial * np.sin(float(abs(m)) * theta)

            mode = np.where(aperture, mode, 0.0)

            rms = float(np.sqrt(np.mean(np.square(mode[aperture])))) if np.any(aperture) else 0.0

            if rms > 1e-12:

                mode = mode / rms

            phase += float(coeff) * mode.astype(np.float32, copy=False)


        phase = np.where(aperture, phase, 0.0).astype(np.float32, copy=False)

        self._zernike_phase_cache_shape = (height, width)

        self._zernike_phase_cache_coeffs = (max_degree, *coeff_values)

        self._zernike_phase_cache_map = phase

        return phase


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


    def _clear_pattern_gallery(self) -> None:

        while self.pattern_gallery_grid.count() > 0:

            item = self.pattern_gallery_grid.takeAt(0)

            widget = item.widget()

            if widget is not None:

                widget.deleteLater()

        self._pattern_preview_labels = {}

        self._pattern_preview_pixmaps = {}

        self._pattern_preview_pending_index = None

        self._pattern_preview_pending_sequence = 0


    def _triangular_row_col(self, index_1_based: int) -> tuple[int, int]:

        row = 1

        remaining = int(index_1_based)

        while remaining > row:

            remaining -= row

            row += 1

        return row - 1, remaining - 1


    def _initialize_pattern_gallery(self, total_patterns: int) -> None:

        self._clear_pattern_gallery()

        if total_patterns <= 0:

            self.pattern_gallery_status_label.setText("Load PLM patterns to start gallery capture.")

            return


        for index in range(1, total_patterns + 1):

            row, col = self._triangular_row_col(index)

            cell = QWidget()

            cell_layout = QVBoxLayout()

            cell_layout.setContentsMargins(0, 0, 0, 0)

            cell_layout.setSpacing(4)

            index_label = QLabel(f"Pattern {index}")

            thumb_label = QLabel("pending")

            thumb_label.setAlignment(Qt.AlignCenter)

            thumb_label.setFixedSize(180, 120)

            cell_layout.addWidget(index_label, alignment=Qt.AlignHCenter)

            cell_layout.addWidget(thumb_label)

            cell.setLayout(cell_layout)

            self.pattern_gallery_grid.addWidget(cell, row, col)

            self._pattern_preview_labels[index] = thumb_label


        self.pattern_gallery_status_label.setText(

            f"Pattern gallery ready: {total_patterns} pattern slots in triangular layout."

        )


    def _capture_pattern_preview_if_ready(self, phase_rgb: np.ndarray) -> None:

        pattern_index = self._pattern_preview_pending_index

        if pattern_index is None:

            return

        if self._pattern_preview_pending_sequence != self._pattern_upload_sequence:

            return

        if pattern_index not in self._pattern_preview_labels:

            return


        height, width = int(phase_rgb.shape[0]), int(phase_rgb.shape[1])

        contiguous = np.ascontiguousarray(phase_rgb)

        image = QImage(contiguous.data, width, height, width * 3, QImage.Format_RGB888).copy()

        pixmap = QPixmap.fromImage(image)

        label = self._pattern_preview_labels[pattern_index]

        scaled = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.FastTransformation)

        self._pattern_preview_pixmaps[pattern_index] = scaled

        label.setPixmap(scaled)

        label.setText("")

        self._pattern_preview_pending_index = None

        self.pattern_gallery_status_label.setText(

            f"Updated gallery for pattern {pattern_index}/{len(self._pattern_preview_labels)}"

        )


    def _store_pattern_field_if_ready(self, recovered_field: np.ndarray) -> None:

        pattern_index = self._pattern_preview_pending_index

        if pattern_index is None:

            return

        if self._pattern_preview_pending_sequence != self._pattern_upload_sequence:

            return

        self._pattern_fields_by_pattern[int(pattern_index)] = np.asarray(recovered_field, dtype=np.complex64).copy()


    def _clear_pattern_fields_data(self) -> None:

        self._pattern_fields_by_pattern = {}

        self._current_pattern_fields_loop = None


    def _build_current_pattern_fields_loop(self) -> Optional[np.ndarray]:

        if not self._is_current_pattern_field_loop_complete():

            return None

        if self._current_plm_total_patterns is not None and int(self._current_plm_total_patterns) > 0:

            expected_modes = int(self._current_plm_total_patterns)

        else:

            expected_modes = len(self._pattern_preview_labels)

        try:

            return np.stack(


                [np.asarray(self._pattern_fields_by_pattern[idx],

dtype=np.complex64) for idx in range(1, expected_modes + 1)],

                axis=0,

            )

        except Exception:

            return None


    def _is_current_pattern_field_loop_complete(self) -> bool:

        if self._current_plm_total_patterns is not None and int(self._current_plm_total_patterns) > 0:

            expected_modes = int(self._current_plm_total_patterns)

        else:

            expected_modes = len(self._pattern_preview_labels)

        if expected_modes <= 0:

            return False

        for pattern_idx in range(1, expected_modes + 1):

            if pattern_idx not in self._pattern_fields_by_pattern:

                return False

        return True


    def _record_completed_pattern_field_loop_if_ready(self) -> None:

        loop_fields = self._build_current_pattern_fields_loop()

        if loop_fields is None:

            return

        self._current_pattern_fields_loop = loop_fields

        if self.pattern_fields_save_checkbox.isChecked():

            self._save_pattern_fields_history()


    def _on_browse_pattern_fields_save_file(self) -> None:

        file_path, _ = QFileDialog.getSaveFileName(

            self,

            "Select recovered fields output file",

            self.pattern_fields_save_path_edit.text().strip() or str(Path.cwd() / "pattern_fields_history.pt"),

            "PyTorch tensor (*.pt);;NumPy archive (*.npz)",

        )

        if not file_path:

            return

        self.pattern_fields_save_path_edit.setText(file_path)


    def _save_pattern_fields_history(self) -> bool:

        if self._current_pattern_fields_loop is None:

            self.pattern_gallery_status_label.setText("Pattern fields save skipped: no completed loop available.")

            return False

        path_text = self.pattern_fields_save_path_edit.text().strip()

        if not path_text:

            self.pattern_gallery_status_label.setText("Pattern fields save failed: output path is empty.")

            return False

        output_path = Path(path_text)

        try:

            output_path.parent.mkdir(parents=True, exist_ok=True)

            fields_complex = np.asarray(self._current_pattern_fields_loop, dtype=np.complex64)

            current_loop_index = int(self._completed_pattern_loops)

            if torch is not None:

                payload = {

                    "fields_complex": torch.from_numpy(fields_complex),

                    "loop_index": torch.tensor(current_loop_index, dtype=torch.int32),

                }

                torch.save(payload, str(output_path))

            else:

                np.savez(

                    str(output_path),

                    fields_complex=fields_complex,

                    loop_index=np.int32(current_loop_index),

                )

            self.pattern_gallery_status_label.setText(

                f"Pattern fields saved: shape {fields_complex.shape} (loop {current_loop_index}) -> {output_path}"

            )

            return True

        except Exception as exc:

            self.pattern_gallery_status_label.setText(f"Pattern fields save failed: {exc}")

            return False


    def _on_save_pattern_fields_now(self) -> None:

        loop_fields = self._build_current_pattern_fields_loop()

        if loop_fields is not None:

            self._current_pattern_fields_loop = loop_fields

        self._save_pattern_fields_history()


    def _prepare_mode_for_target_shape(self, mode: np.ndarray) -> np.ndarray:

        if self._target_modes is None:

            return np.ascontiguousarray(mode.astype(np.complex64))

        target_h = int(self._target_modes.shape[1])

        target_w = int(self._target_modes.shape[2])

        mode_h = int(mode.shape[0])

        mode_w = int(mode.shape[1])

        if mode_h == target_h and mode_w == target_w:

            return np.ascontiguousarray(mode.astype(np.complex64))

        result = np.zeros((target_h, target_w), dtype=np.complex64)

        y_overlap = min(target_h, mode_h)

        x_overlap = min(target_w, mode_w)

        target_y0 = (target_h - y_overlap) // 2

        target_x0 = (target_w - x_overlap) // 2

        mode_y0 = (mode_h - y_overlap) // 2

        mode_x0 = (mode_w - x_overlap) // 2

        result[

            target_y0 : target_y0 + y_overlap,

            target_x0 : target_x0 + x_overlap,

        ] = mode[

            mode_y0 : mode_y0 + y_overlap,

            mode_x0 : mode_x0 + x_overlap,

        ]

        return np.ascontiguousarray(result)


    def _load_default_target_modes(self) -> None:

        if not self._target_modes_path.exists():

            self.channel_matrix_status_label.setText(

                f"Channel matrix: target modes file not found at {self._target_modes_path}."

            )

            return

        try:

            modes = np.load(str(self._target_modes_path))

            if modes.ndim != 3:

                raise ValueError(f"target modes must be 3D [n, y, x], got {modes.shape}")

            if not np.iscomplexobj(modes):

                modes = modes.astype(np.float32).astype(np.complex64)

            else:

                modes = modes.astype(np.complex64)

            self._target_modes = np.ascontiguousarray(modes)

            self._channel_matrix_xt_db = None

            self.channel_matrix_xt_label.setText("XT (dB): n/a")

            self.channel_matrix_status_label.setText(

                f"Channel matrix: loaded target modes {modes.shape[0]} x {modes.shape[1]} x {modes.shape[2]}."

            )

        except Exception as exc:

            self._target_modes = None

            self.channel_matrix_status_label.setText(f"Channel matrix: failed to load target modes ({exc})")


    def _recompute_channel_matrix(self) -> None:

        if self._target_modes is None:

            self._channel_matrix = None

            self._channel_matrix_xt_db = None

            return

        if len(self._recovered_modes_by_pattern) <= 0:

            self._channel_matrix = None

            self._channel_matrix_xt_db = None

            return


        target_modes = np.ascontiguousarray(self._target_modes)

        expected_modes = 0

        if self._current_plm_total_patterns is not None and int(self._current_plm_total_patterns) > 0:

            expected_modes = int(self._current_plm_total_patterns)

        else:

            expected_modes = int(target_modes.shape[0])


        mode_template = np.zeros((target_modes.shape[1], target_modes.shape[2]), dtype=np.complex64)

        recovered_modes_list: list[np.ndarray] = []

        captured_modes = 0

        for pattern_idx in range(1, expected_modes + 1):

            mode = self._recovered_modes_by_pattern.get(pattern_idx)

            if mode is None:

                recovered_modes_list.append(mode_template)

            else:

                recovered_modes_list.append(np.asarray(mode, dtype=np.complex64))

                captured_modes += 1

        recovered_modes = np.stack(recovered_modes_list, axis=0)


        try:

            if torch is not None and F is not None:

                target_t = torch.from_numpy(target_modes.astype(np.complex64))

                recovered_t = torch.from_numpy(recovered_modes.astype(np.complex64))

                recovered_t = auto_align_modes_torch(recovered_t, target_t)

                target_t = normalize_modes_torch(target_t)

                recovered_t = normalize_modes_torch(recovered_t)

                T_flat = target_t.reshape(target_t.shape[0], -1)

                R_flat = recovered_t.reshape(recovered_t.shape[0], -1)

                channel_matrix = torch.matmul(torch.conj(T_flat), R_flat.T)

                self._channel_matrix = channel_matrix.detach().cpu().numpy()

            else:

                target_norm = normalize_modes_np(target_modes)

                recovered_norm = normalize_modes_np(recovered_modes)

                T_flat = target_norm.reshape(target_norm.shape[0], -1)

                R_flat = recovered_norm.reshape(recovered_norm.shape[0], -1)

                self._channel_matrix = np.matmul(np.conj(T_flat), R_flat.T)

            self._channel_matrix_xt_db = measure_xt_db(self._channel_matrix)

            if self._channel_matrix_xt_db is not None:

                self.channel_matrix_xt_label.setText(f"XT (dB): {self._channel_matrix_xt_db:.3f}")

            else:

                self.channel_matrix_xt_label.setText("XT (dB): n/a")

            self.channel_matrix_status_label.setText(


                f"Channel matrix: updated({captured_modes}/{expected_modes} captured this loop), matrix shape{self._channel_matrix.shape}.")

        except Exception as exc:

            self._channel_matrix = None

            self._channel_matrix_xt_db = None

            self.channel_matrix_xt_label.setText("XT (dB): n/a")

            self.channel_matrix_status_label.setText(f"Channel matrix update failed: {exc}")


    def _is_current_channel_matrix_complete_square(self) -> bool:

        if self._channel_matrix is None:

            return False

        matrix_now = np.asarray(self._channel_matrix)

        if matrix_now.ndim != 2:

            return False

        if self._current_plm_total_patterns is not None and int(self._current_plm_total_patterns) > 0:

            expected_modes = int(self._current_plm_total_patterns)

        else:

            expected_modes = int(self._target_modes.shape[0]) if self._target_modes is not None else 0

        if expected_modes <= 0:

            return False

        if matrix_now.shape[0] != expected_modes or matrix_now.shape[1] != expected_modes:

            return False

        for pattern_idx in range(1, expected_modes + 1):

            if pattern_idx not in self._recovered_modes_by_pattern:

                return False

        return True


    def _store_recovered_mode_if_ready(self, recovered_field: np.ndarray) -> None:

        pattern_index = self._pattern_preview_pending_index

        if pattern_index is None:

            return

        if self._pattern_preview_pending_sequence != self._pattern_upload_sequence:

            return

        mode = self._prepare_mode_for_target_shape(recovered_field)

        self._recovered_modes_by_pattern[int(pattern_index)] = mode

        self._recompute_channel_matrix()

        self._channel_matrix_dirty = True


    def _clear_channel_matrix_data(self) -> None:

        self._recovered_modes_by_pattern = {}

        self._channel_matrix = None

        self._channel_matrix_xt_db = None

        self._channel_matrix_dirty = True

        self._channel_matrix_loop_history = []

        self._channel_matrix_xt_history_db = []

        self._completed_pattern_loops = 0

        self._last_plm_counter_index = None

        self.channel_matrix_xt_label.setText("XT (dB): n/a")

        self.channel_matrix_status_label.setText(

            "Channel matrix: waiting for recovered modes during PLM loop."

        )


    def _on_browse_channel_matrix_save_file(self) -> None:

        file_path, _ = QFileDialog.getSaveFileName(

            self,

            "Select channel matrix output file",

            self.channel_matrix_save_path_edit.text().strip() or str(Path.cwd() / "channel_matrix_history.npz"),

            "NumPy archive (*.npz)",

        )

        if not file_path:

            return

        self.channel_matrix_save_path_edit.setText(file_path)


    def _save_channel_matrix_history(self) -> bool:

        if len(self._channel_matrix_loop_history) <= 0:

            self.channel_matrix_status_label.setText("Channel matrix save skipped: no matrices available.")

            return False

        path_text = self.channel_matrix_save_path_edit.text().strip()

        if not path_text:

            self.channel_matrix_status_label.setText("Channel matrix save failed: output path is empty.")

            return False

        output_path = Path(path_text)

        try:

            output_path.parent.mkdir(parents=True, exist_ok=True)

            mats_list = [np.asarray(m, dtype=np.complex64) for m in self._channel_matrix_loop_history]

            shapes = [tuple(m.shape) for m in mats_list]

            same_shape = all(shape == shapes[0] for shape in shapes)

            if not same_shape:

                self.channel_matrix_status_label.setText(

                    f"Channel matrix save failed: history has mixed shapes {shapes}. Clear and save full loops only."

                )

                return False

            mats = np.stack(mats_list, axis=0)

            xt_vals = np.asarray(self._channel_matrix_xt_history_db, dtype=np.float32)

            loop_ids = np.arange(1, len(mats_list) + 1, dtype=np.int32)

            np.savez(

                str(output_path),

                channel_matrices_complex=mats,

                xt_db=xt_vals,

                loop_index=loop_ids,

            )

            self.channel_matrix_status_label.setText(

                f"Channel matrix saved: {len(mats_list)} loop(s) -> {output_path}"

            )

            return True

        except Exception as exc:

            self.channel_matrix_status_label.setText(f"Channel matrix save failed: {exc}")

            return False


    def _on_save_channel_matrix_now(self) -> None:

        if self._channel_matrix is None and len(self._channel_matrix_loop_history) <= 0:

            self.channel_matrix_status_label.setText("Channel matrix save skipped: no current matrix yet.")

            return


        if self._channel_matrix is not None:

            if not self._is_current_channel_matrix_complete_square():

                shape_now = np.asarray(self._channel_matrix).shape

                self.channel_matrix_status_label.setText(


                    f"Channel matrix save skipped: current matrix notcomplete square [modes,modes]. Current shape={shape_now}.")

                return

            matrix_now = np.asarray(self._channel_matrix)

            if matrix_now.ndim != 2:

                self.channel_matrix_status_label.setText(

                    f"Channel matrix save skipped: expected 2D matrix, got shape {matrix_now.shape}."

                )

                return

            if len(self._channel_matrix_loop_history) <= 0 or not np.array_equal(

                self._channel_matrix_loop_history[-1], matrix_now

            ):

                self._channel_matrix_loop_history.append(np.asarray(matrix_now, dtype=np.complex64).copy())

                self._channel_matrix_xt_history_db.append(

                    float(self._channel_matrix_xt_db) if self._channel_matrix_xt_db is not None else float("nan")

                )

        self._save_channel_matrix_history()


    def _record_completed_loop_if_ready(self) -> None:

        if self._channel_matrix is None:

            return

        if not self._is_current_channel_matrix_complete_square():

            shape_now = np.asarray(self._channel_matrix).shape

            self.channel_matrix_status_label.setText(

                f"Loop capture skipped: matrix not complete square yet (shape={shape_now})."

            )

            return

        matrix_now = np.asarray(self._channel_matrix)

        if matrix_now.ndim != 2:

            self.channel_matrix_status_label.setText(

                f"Channel matrix loop capture skipped: expected 2D matrix, got shape {matrix_now.shape}."

            )

            return

        self._completed_pattern_loops += 1

        self._channel_matrix_loop_history.append(np.asarray(matrix_now, dtype=np.complex64).copy())

        self._channel_matrix_xt_history_db.append(

            float(self._channel_matrix_xt_db) if self._channel_matrix_xt_db is not None else float("nan")

        )

        with self._zernike_optimizer_metric_condition:

            self._zernike_optimizer_metric_condition.notify_all()

        if self.channel_matrix_save_checkbox.isChecked():

            self._save_channel_matrix_history()


    def _refresh_channel_matrix_plot(self, force: bool = False) -> None:

        if not (force or self._channel_matrix_dirty):

            return

        if not (force or self.right_tab_widget.currentIndex() == 2):

            return


        fig = self.channel_matrix_canvas.figure

        fig.clear()

        if self._channel_matrix is None:

            ax = fig.add_subplot(111)

            ax.set_title("Channel Matrix")

            ax.text(0.5, 0.5, "Waiting for channel matrix data", ha="center", va="center")

            ax.set_box_aspect(1)

            ax.set_axis_off()

            self.channel_matrix_canvas.draw_idle()

            self._channel_matrix_dirty = False

            return


        power_linear = np.abs(self._channel_matrix) ** 2

        power_mw_rel = power_linear / (float(np.max(power_linear)) + 1e-12)

        power_dbm_rel = 10.0 * np.log10(np.maximum(power_mw_rel, 1e-12))

        power_floor_dbm = -40.0


        ax_amp = fig.add_subplot(111)

        im_amp = ax_amp.imshow(

            power_dbm_rel,

            cmap=self._phase_colormap_name,

            vmin=power_floor_dbm,

            vmax=0.0,

            aspect="equal",

        )

        ax_amp.set_box_aspect(1)

        xt_text = "n/a" if self._channel_matrix_xt_db is None else f"{self._channel_matrix_xt_db:.3f} dB"

        ax_amp.set_title(f"Channel Matrix Intensity (dBm rel) | XT={xt_text}")

        ax_amp.set_xlabel("Recovered mode index")

        ax_amp.set_ylabel("Target mode index")


        norm = mpl.colors.Normalize(vmin=power_floor_dbm, vmax=0.0)

        sm = mpl.cm.ScalarMappable(norm=norm, cmap=self._phase_colormap_name)

        sm.set_array([])

        cbar = fig.colorbar(sm, ax=ax_amp, fraction=0.046, pad=0.04)

        cbar.set_label("Power dBm (relative, clipped at -40 dBm)")

        fig.tight_layout()

        self.channel_matrix_canvas.draw_idle()

        self._channel_matrix_dirty = False


    def _on_plm_counter_updated(self, current_index: int, total_count: int) -> None:

        previous_index = self._last_plm_counter_index

        self._last_plm_counter_index = current_index

        if (

            previous_index is not None

            and total_count > 0

            and previous_index == total_count

            and current_index == 1

        ):

            self._record_completed_pattern_field_loop_if_ready()

            self._record_completed_loop_if_ready()


        self._current_plm_pattern_index = current_index

        self._current_plm_total_patterns = total_count

        self._last_pattern_switch_time_s = perf_counter()

        self._pattern_upload_sequence += 1

        self._pattern_preview_pending_index = current_index

        self._pattern_preview_pending_sequence = self._pattern_upload_sequence

        self._active_pattern_upload_sequence = self._pattern_upload_sequence

        self._logs_taken_for_active_pattern = 0

        self._last_log_time_s = 0.0

        drop_n = int(self.drop_frames_combo.currentData() or 0)

        with self._frame_lock:

            self._drop_frames_start_count = self._frame_count

        self._drop_frames_remaining = max(0, drop_n)

        self._set_pattern_counter(current_index, total_count)

        if self.zernike_auto_optimize_checkbox.isChecked():

            self._set_zernike_optimizer_status(

                f"Optimizer: waiting metric for pattern {current_index}/{total_count}"

            )


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


                f"Logging: active ({mode_text}) |seq={self._active_pattern_upload_sequence}, sample{self._logs_taken_for_active_pattern}/{n_max}, frame={frame_count}")

        except Exception as exc:

            self.logging_status_label.setText(f"Logging error: {exc}")

            self._stop_logging()


    def _update_phase_colorbar(self) -> None:

        bar_height = 256

        phase_values = np.linspace(np.pi, -np.pi, bar_height, dtype=np.float32)

        phase_column = phase_values[:, None]

        rgb_column = phase_to_rgb_uint8(phase_column, cmap_name=self._phase_colormap_name)

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


    def _get_trigger_delay_ms_from_ui_raw(self) -> float:

        try:

            value_ms = float(self.trigger_delay_edit.text().strip())

        except Exception:

            return 0.0

        return max(0.0, value_ms)


    def _on_trigger_delay_editing_finished(self) -> None:

        trigger_delay_us = self._get_trigger_delay_us_from_ui()

        if self._trigger_delay_feature_name is None:

            return

        if not self._streaming:

            return

        if str(self.trigger_mode_combo.currentData() or "freerun") != "line1":

            return

        try:

            delay_feature = self._get_feature(self._trigger_delay_feature_name)

            delay_feature.set(trigger_delay_us)

            actual_us = float(delay_feature.get())

            self._set_status(

                f"Trigger delay set: requested={trigger_delay_us / 1000.0:.3f} ms, applied={actual_us / 1000.0:.3f} ms"

            )

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

            self._last_processed_frame_count = -1

            self._drop_frames_remaining = 0

            self._drop_frames_start_count = -1

            self._last_pattern_switch_time_s = 0.0

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

        self._phase_plot_dirty = True


    def _append_live_intensity_sample(self, roi_sum_intensity: float) -> None:

        t_now = perf_counter() - self._intensity_plot_t0

        self._intensity_plot_time_s.append(t_now)

        self._intensity_plot_sum.append(float(roi_sum_intensity))

        self._intensity_plot_dirty = True


    def _refresh_plots_if_dirty(self, force: bool = False) -> None:

        active_index = self.tab_widget.currentIndex()


        if self._phase_plot_dirty and (force or active_index == 1):

            times = list(self._phase_plot_time_s)

            phi1_vals = list(self._phase_plot_phi1)

            phi2_vals = list(self._phase_plot_phi2)

            dphi_vals = list(self._phase_plot_dphi)

            self.phase_plot_line_phi1.set_data(times, phi1_vals)

            self.phase_plot_line_phi2.set_data(times, phi2_vals)

            self.phase_plot_line_dphi.set_data(times, dphi_vals)

            if len(times) == 1:

                self.phase_plot_ax.set_xlim(0.0, max(5.0, times[0] + 1.0))

            elif len(times) > 1:

                t_max = times[-1]

                t_min = max(0.0, t_max - 20.0)

                self.phase_plot_ax.set_xlim(t_min, t_max + 0.2)

            self.phase_plot_canvas.draw_idle()

            self._phase_plot_dirty = False


        if self._intensity_plot_dirty and (force or active_index == 2):

            times = list(self._intensity_plot_time_s)

            sums = list(self._intensity_plot_sum)

            self.intensity_plot_line.set_data(times, sums)

            if len(times) == 1:

                self.intensity_plot_ax.set_xlim(0.0, max(5.0, times[0] + 1.0))

                self.intensity_plot_ax.set_ylim(0.0, max(1.0, sums[0] * 1.2))

            elif len(times) > 1:

                t_max = times[-1]

                t_min = max(0.0, t_max - 20.0)

                self.intensity_plot_ax.set_xlim(t_min, t_max + 0.2)

                y_max = max(sums) if len(sums) > 0 else 1.0

                self.intensity_plot_ax.set_ylim(0.0, max(1.0, y_max * 1.1))

            self.intensity_plot_canvas.draw_idle()

            self._intensity_plot_dirty = False


        self._refresh_channel_matrix_plot(force=force)


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

            self._clear_pattern_fields_data()

            self._clear_channel_matrix_data()

            self._initialize_pattern_gallery(int(arr.shape[0]))

            self._set_pattern_counter(1, int(arr.shape[0]))

            self.pattern_info_label.setText(

                f"Loaded {arr.shape[0]} frames, resolution {arr.shape[2]}x{arr.shape[1]}."

            )

            self.pattern_status_label.setText("Pattern file loaded. Ready to start PLM loop.")

        except Exception as exc:

            self._plm_phase_patterns = None

            self._plm_pattern_path = None

            self._clear_pattern_fields_data()

            self._clear_channel_matrix_data()

            self._clear_pattern_gallery()

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

        self._reset_continuous_optimizer_state()

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


                    f"Pattern shape {phase_patterns.shape[1:]} does not match PLM shape {(expected_h, expected_w)} for catalog'{plm_catalog}'.")


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

                        periods = (

                            int(self._grating_period_x_px),

                            int(self._grating_period_y_px),

                        )

                        if periods != cached_periods:

                            cached_ramp = self._build_grating_phase_ramp(

                                (expected_h, expected_w), periods[0], periods[1]

                            )

                            cached_periods = periods


                        zernike_phase = self._build_zernike_phase_map((expected_h, expected_w))

                        base_phase = np.mod(phase_patterns[display_idx], 2.0 * np.pi)

                        overlaid_phase = np.mod(base_phase + cached_ramp + zernike_phase, 2.0 * np.pi)

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


                initial_state_indices =np.digitize(np.mod(phase_patterns[0], 2.0 * np.pi), buckets,

right=False).astype(np.int32)

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


            if frame_count == self._last_processed_frame_count:

                return


            trigger_mode = str(self.trigger_mode_combo.currentData() or "freerun")

            if trigger_mode == "line1" and self._last_pattern_switch_time_s > 0.0:

                settle_s = self._get_trigger_delay_ms_from_ui_raw() / 1000.0

                if settle_s > 0.0:

                    elapsed_s = perf_counter() - self._last_pattern_switch_time_s

                    if elapsed_s < settle_s:

                        self._last_processed_frame_count = frame_count

                        if self._current_plm_pattern_index is not None and self._current_plm_total_patterns is not None:

                            remaining_ms = max(0.0, (settle_s - elapsed_s) * 1000.0)

                            self._set_status(


                                f"Waiting settle after pattern{self._current_plm_pattern_index}/{self._current_plm_total_patterns}:{remaining_ms:.1f} ms"

                            )

                        return


            if self._drop_frames_remaining > 0 and frame_count > self._drop_frames_start_count:

                self._drop_frames_remaining -= 1

                self._last_processed_frame_count = frame_count

                if self._current_plm_pattern_index is not None and self._current_plm_total_patterns is not None:

                    self._set_status(


                        f"Dropping frame after pattern{self._current_plm_pattern_index}/{self._current_plm_total_patterns}(remaining: {self._drop_frames_remaining})."

                    )

                return


            if image.ndim == 3:

                image = image[..., 0]

            self._last_processed_frame_count = frame_count


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

                phase_rgb = phase_to_rgb_uint8(

                    recovered_phase,

                    value=amplitude_norm,

                    cmap_name=self._phase_colormap_name,

                )

                self.recovered_field_label.set_image_rgb_array(phase_rgb)

                self._store_pattern_field_if_ready(recovered_field)

                self._store_recovered_mode_if_ready(recovered_field)

                self._maybe_step_continuous_zernike_optimizer()

                self._capture_pattern_preview_if_ready(phase_rgb)

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

        self._zernike_optimizer_stop_event.set()

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



def phase_to_rgb_uint8(

    phase: np.ndarray,

    value: Optional[np.ndarray] = None,

    cmap_name: str = "viridis",

) -> np.ndarray:

    h = np.mod((phase + np.pi) / (2.0 * np.pi), 1.0)

    rgb = mpl.colormaps[cmap_name](h)[..., :3].astype(np.float32)

    if value is not None:

        v = np.clip(value.astype(np.float32), 0.0, 1.0)

        rgb *= v[..., None]

    return np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)



def normalize_modes_np(modes: np.ndarray) -> np.ndarray:

    norm = np.sqrt(np.sum(np.abs(modes) ** 2, axis=(-2, -1), keepdims=True))

    norm = np.maximum(norm, 1e-12)

    return modes / norm



def normalize_modes_torch(modes):

    norm = torch.sqrt(torch.sum(torch.abs(modes) ** 2, dim=(-2, -1), keepdim=True))

    norm = torch.clamp(norm, min=1e-12)

    return modes / norm



def measure_xt_db(matrix):

    if matrix is None:

        return None

    if torch is not None:

        matrix_t = matrix if torch.is_tensor(matrix) else torch.as_tensor(matrix)

        p = torch.abs(matrix_t) ** 2

        max_col, _ = torch.max(p, dim=0)

        xt = (torch.sum(p, dim=0) - max_col) / torch.clamp(max_col, min=1e-12)

        return float(10.0 * torch.log10(torch.clamp(torch.mean(xt), min=1e-12)).item())


    matrix_np = np.asarray(matrix)

    p = np.abs(matrix_np) ** 2

    max_col = np.max(p, axis=0)

    xt = (np.sum(p, axis=0) - max_col) / np.maximum(max_col, 1e-12)

    return float(10.0 * np.log10(max(np.mean(xt), 1e-12)))



def get_spatial_moments_torch(modes, threshold_ratio: float = 0.05, power: float = 1.0):

    _, height, width = modes.shape

    intensity = torch.abs(modes) ** 2

    max_intensity = intensity.amax(dim=(-2, -1), keepdim=True)

    intensity = torch.where(

        intensity > float(threshold_ratio) * max_intensity,

        intensity,

        torch.zeros_like(intensity),

    )

    if float(power) != 1.0:

        intensity = intensity ** float(power)

    p = intensity / (intensity.sum(dim=(-2, -1), keepdim=True) + 1e-12)

    y_grid = torch.linspace(-1, 1, height, device=modes.device)

    x_grid = torch.linspace(-1, 1, width, device=modes.device)

    yy, xx = torch.meshgrid(y_grid, x_grid, indexing="ij")

    cx = (p * xx).sum(dim=(-2, -1), keepdim=True)

    cy = (p * yy).sum(dim=(-2, -1), keepdim=True)

    var_x = (p * (xx - cx) ** 2).sum(dim=(-2, -1))

    var_y = (p * (yy - cy) ** 2).sum(dim=(-2, -1))

    std_x = torch.sqrt(var_x + 1e-12)

    std_y = torch.sqrt(var_y + 1e-12)

    return cx.squeeze(-1).squeeze(-1), cy.squeeze(-1).squeeze(-1), std_x, std_y



def auto_align_modes_torch(recovered_modes, target_modes):

    n_rec, _, _ = recovered_modes.shape

    rec_fund = recovered_modes[0:1]

    tar_fund = target_modes[0:1]

    r_cx, r_cy, r_std_x, r_std_y = get_spatial_moments_torch(rec_fund)

    t_cx, t_cy, t_std_x, t_std_y = get_spatial_moments_torch(tar_fund)

    scale_x = (r_std_x / (t_std_x + 1e-12))[0]

    scale_y = (r_std_y / (t_std_y + 1e-12))[0]

    shift_x = r_cx[0] - scale_x * t_cx[0]

    shift_y = r_cy[0] - scale_y * t_cy[0]

    theta = torch.zeros((n_rec, 2, 3), device=recovered_modes.device)

    theta[:, 0, 0] = scale_x

    theta[:, 1, 1] = scale_y

    theta[:, 0, 2] = shift_x

    theta[:, 1, 2] = shift_y

    is_complex = torch.is_complex(recovered_modes)

    if is_complex:

        modes_formatted = torch.view_as_real(recovered_modes).permute(0, 3, 1, 2)

    else:

        modes_formatted = recovered_modes.unsqueeze(1)

    grid = F.affine_grid(theta, modes_formatted.size(), align_corners=True)

    aligned = F.grid_sample(

        modes_formatted,

        grid,

        mode="bilinear",

        padding_mode="zeros",

        align_corners=True,

    )

    if is_complex:

        return torch.view_as_complex(aligned.permute(0, 2, 3, 1).contiguous())

    return aligned.squeeze(1)



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

