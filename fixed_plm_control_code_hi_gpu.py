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
from matplotlib import image
import numpy as np
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PIL import Image
#from pygame import image
from vmbpy import PixelFormat, VmbSystem

from utils.math_utils import *
from utils.field_utils import *
#from utils.plm_utils import *


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
    QSizePolicy = QtWidgets.QSizePolicy

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
    raise ImportError("PyQt5 is required. Install with: pip install PyQt5") from exc


from ti_plm import PLM
from ti_plm.display import EventLoopExit, ImageWindow


class FrameProcessorThread(QtCore.QThread):
    # Payload: frame_count, image, fft_image, recovered_field, phase_rgb, roi_sum, roi_mean
    #frame_processed = QtCore.pyqtSignal(int, np.ndarray, np.ndarray, object, object, object, object)
    #error_occurred = QtCore.pyqtSignal(str)

    frame_processed = QtCore.pyqtSignal(dict)
    error_occurred = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._is_running = False
        self._lock = threading.Lock()
        
        # Shared state written by Main Thread, read by Worker Thread
        self.latest_frame = None
        self.frame_count = 0
        self.master_dark_frame = None
        self.fft_roi = None
        self.live_roi = None
        self.basic_recovery_mode = True
        self.colormap_name = "viridis"

        self.pattern_seq = -1
        self.pattern_idx = -1
        self.device = torch.device("cuda" if torch and torch.cuda.is_available() else "cpu")

        self._dark_frame_gpu = None
        self._last_dark_frame_ref = None

    def update_state(self, frame, count, dark_frame, fft_roi, live_roi, basic_recovery, colormap_name, pattern_seq, pattern_idx):
        """Called by the main thread to push new data to the worker."""
        with self._lock:
            self.latest_frame = frame
            self.frame_count = count
            self.master_dark_frame = dark_frame
            self.fft_roi = fft_roi
            self.live_roi = live_roi
            self.basic_recovery_mode = basic_recovery
            self.colormap_name = colormap_name
            self.pattern_seq = pattern_seq
            self.pattern_idx = pattern_idx

    def stop(self):
        self._is_running = False
        self.wait()

    def run(self):
        self._is_running = True
        last_processed_count = -1

        while self._is_running:
            # 1. Safely grab the latest inputs
            with self._lock:
                frame = self.latest_frame
                frame_count = self.frame_count
                dark_frame = self.master_dark_frame
                fft_roi = self.fft_roi
                live_roi = self.live_roi
                basic_recovery = self.basic_recovery_mode
                cmap_name = self.colormap_name
                # Capture to local variables!
                thread_seq = self.pattern_seq 
                thread_idx = self.pattern_idx

            # 2. Check if we actually have a new frame to process
            if frame is None or frame_count == last_processed_count:
                self.msleep(10) # Prevent CPU pegging
                continue

            last_processed_count = frame_count

            try:
                # 3. Process Dark Frame
                image_np = frame.copy()
                if image_np.ndim == 3:
                    image_np = image_np[..., 0]

                image_t = torch.from_numpy(image_np).to(self.device, dtype=torch.float32)

                # 3. Process Dark Frame on GPU
                if dark_frame is not None and dark_frame.shape == image_np.shape:
                    # Only upload dark frame to GPU if it changed
                    if dark_frame is not self._last_dark_frame_ref:
                        self._dark_frame_gpu = torch.from_numpy(dark_frame).to(self.device, dtype=torch.float32)
                        self._last_dark_frame_ref = dark_frame
                    
                    image_t = torch.clamp(image_t - self._dark_frame_gpu, 0.0, 255.0)
                
                # We need a CPU uint8 copy for ROI intensity math and final payload
                image_corrected_np = image_t.detach().cpu().to(torch.uint8).numpy()
                image = image_corrected_np
                
                # 4. Calculate Live ROI Intensity
                roi_sum = None
                roi_mean = None
                if live_roi is not None:
                    x0, y0, x1, y1 = live_roi
                    roi_patch = image_np[y0:y1, x0:x1]
                    if roi_patch.size > 0:
                        roi_sum = float(np.sum(roi_patch, dtype=np.float64))
                        roi_mean = float(np.mean(roi_patch))
             
                # 5. Heavy Math: PyTorch FFT section
                fft_complex = torch.fft.fft2(image_t)
                fft_shifted = torch.fft.fftshift(fft_complex)
                
                magnitude = torch.abs(fft_shifted)
                magnitude_log = torch.log1p(magnitude)
                
                # Bring log magnitude back to CPU for OpenCV normalization util
                fft_image = cv_normalize_to_uint8(magnitude_log.detach().cpu().numpy())

                # 6. Heavy Math: Field Recovery & Colormap
                recovered_field = None
                phase_rgb = None
                
                if fft_roi is not None:
                    x0, y0, x1, y1 = fft_roi
                    if x1 - x0 >= 3 and y1 - y0 >= 3:
                        filtered = torch.zeros_like(fft_shifted)
                        filtered[y0:y1, x0:x1] = fft_shifted[y0:y1, x0:x1]
                        
                        if basic_recovery:
                            # Do Basic Recovery entirely on the GPU
                            filtered_t = torch.zeros_like(fft_shifted)
                            filtered_t[y0:y1, x0:x1] = fft_shifted[y0:y1, x0:x1]
                            
                            height, width = fft_shifted.shape[:2]
                            shift_x = (width // 2) - ((x0 + x1) // 2)
                            shift_y = (height // 2) - ((y0 + y1) // 2)
                            
                            centered_t = torch.roll(filtered_t, shifts=(shift_y, shift_x), dims=(0, 1))
                            recovered_field_t = torch.fft.ifft2(torch.fft.ifftshift(centered_t))
                            
                            # Pull the final recovered field back to CPU
                            recovered_field = recovered_field_t.detach().cpu().numpy()
                        else:

                            peak_y_f, peak_x_f = estimate_subpixel_peak_in_roi_torch(torch.abs(fft_shifted), fft_roi)
                            recovered_field = demodulate_to_center_no_ref_torch(filtered, peak_y_f, peak_x_f)
                            recovered_field = remove_linear_phase_tilt_blind_torch(recovered_field)
                            recovered_field = remove_global_phase_offset_blind_torch(recovered_field)    
                            
                            #recovered_field = recovered_field.detach().cpu().numpy()


                        # Generate the RGB colormap representation entirely in the background thread
                        if recovered_field is not None:
                            amplitude = torch.abs(recovered_field)
                            amplitude_norm = amplitude / (float(torch.max(amplitude)) + 1e-12)
                            recovered_phase = torch.angle(recovered_field)
                            phase_rgb = phase_to_rgb_uint8_torch(
                                recovered_phase, value=amplitude_norm, cmap_name=cmap_name
                            )

                            recovered_field = recovered_field.detach().cpu().numpy()
                            phase_rgb = phase_rgb.detach().cpu().numpy()
                            recovered_phase = recovered_phase.detach().cpu().numpy()


                # 7. Ship it back to the main thread in a stamped dictionary
                payload = {
                    "frame_count": frame_count,
                    "image": image,
                    "fft_image": fft_image,
                    "recovered_field": recovered_field,
                    "phase_rgb": phase_rgb,
                    "roi_sum": roi_sum,
                    "roi_mean": roi_mean,
                    "pattern_seq": thread_seq, # Stamped Sequence
                    "pattern_idx": thread_idx  # Stamped Index
                }
                self.frame_processed.emit(payload)

            except Exception as e:
                self.error_occurred.emit(f"Processor Thread Error: {e}")


                # 7. Ship it back to the main thread!
                #self.frame_processed.emit(frame_count, image, fft_image, recovered_field, phase_rgb, roi_sum, roi_mean)

            except Exception as e:
                self.error_occurred.emit(f"Processor Thread Error: {e}")


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
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return max(1, int(width))

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

        if (
            x < x_offset
            or y < y_offset
            or x >= x_offset + pix_w
            or y >= y_offset + pix_h
        ):

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

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def hasHeightForWidth(self) -> bool:

        return True

    def heightForWidth(self, width: int) -> int:

        return max(1, int(width))

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

        if (
            x < x_offset
            or y < y_offset
            or x >= x_offset + pix_w
            or y >= y_offset + pix_h
        ):

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
    plm_phase_preview_updated = QtCore.pyqtSignal(object)
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
        self.image_label.setMinimumSize(120, 120)
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
        self.dark_frame_avg_combo = QComboBox()
        self.dark_frame_avg_combo.addItem("1", 1)
        self.dark_frame_avg_combo.addItem("2", 2)
        self.dark_frame_avg_combo.addItem("4", 4)
        self.dark_frame_avg_combo.addItem("8", 8)
        self.dark_frame_avg_combo.addItem("16", 16)
        self.dark_frame_avg_combo.addItem("32", 32)
        self.dark_frame_avg_combo.setCurrentIndex(3)
        self.dark_frame_capture_button = QPushButton("Capture Dark Frame")
        self.dark_frame_clear_button = QPushButton("Clear Dark")
        self.dark_frame_status_label = QLabel("Dark frame: off")
        self.status_label = QLabel()
        self.status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.live_roi_status_label = QLabel("Live ROI: not set")
        self.live_roi_intensity_label = QLabel("Live ROI sum intensity: n/a")
        self.gaussian_mfd_checkbox = QCheckBox("Measure Gaussian MFD")
        self.gaussian_mfd_label = QLabel("Gaussian MFD (pixel pitch 15 um): disabled")
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
        self._zernike_row_widgets: dict[
            tuple[int, int], tuple[QLabel, QDoubleSpinBox]
        ] = {}

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
        self.zernike_opt_perturb_spin = QDoubleSpinBox()
        self.zernike_opt_perturb_spin.setDecimals(4)
        self.zernike_opt_perturb_spin.setRange(0.0001, 10.0)
        self.zernike_opt_perturb_spin.setSingleStep(0.01)
        self.zernike_opt_perturb_spin.setValue(0.05)
        self.zernike_opt_algorithm_combo = QComboBox()
        self.zernike_opt_algorithm_combo.addItem("ASPGD", "aspgd")
        self.zernike_opt_algorithm_combo.addItem("Nelder-Mead", "nelder_mead")
        self.zernike_opt_start_button = QPushButton("Start Optimizer")
        self.zernike_opt_stop_button = QPushButton("Stop Optimizer")
        self.zernike_opt_stop_button.setEnabled(False)

        self.zernike_opt_results_path_edit = QLineEdit(
            str(Path.cwd() / "zernike_optimization_coefficients.txt")
        )

        self.zernike_opt_results_browse_button = QPushButton("Browse Save")
        self.zernike_opt_history_path_edit = QLineEdit(
            str(Path.cwd() / "zernike_optimization_history.csv")
        )
        self.zernike_opt_history_browse_button = QPushButton("Browse CSV")
        self.zernike_opt_status_label = QLabel("Optimizer: idle")
        self.fft_image_label = RoiSelectableImageLabel("No FFT yet")
        self.fft_image_label.setAlignment(Qt.AlignCenter)
        self.fft_image_label.setMinimumSize(120, 120)
        self.fft_image_label.on_roi_changed = self._on_fft_roi_changed
        self.fft_status_label = QLabel("Waiting for frames...")

        self.recovered_field_label = PointSelectableImageLabel(
            "Select FFT ROI to recover field"
        )

        self.recovered_field_label.setAlignment(Qt.AlignCenter)
        self.recovered_field_label.setMinimumSize(140, 140)
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

        (self.phase_plot_line_phi1,) = self.phase_plot_ax.plot(
            [], [], color="tab:green", linewidth=1.2, label="phi1"
        )

        (self.phase_plot_line_phi2,) = self.phase_plot_ax.plot(
            [], [], color="tab:purple", linewidth=1.2, label="phi2"
        )

        (self.phase_plot_line_dphi,) = self.phase_plot_ax.plot(
            [], [], color="tab:blue", linewidth=1.2, label="dphi_0_2pi"
        )

        self.phase_plot_ax.legend(loc="upper right")
        self.intensity_plot_canvas = FigureCanvas(Figure(figsize=(6.0, 2.5)))
        self.intensity_plot_ax = self.intensity_plot_canvas.figure.add_subplot(111)
        self.intensity_plot_ax.set_title("Live ROI Intensity")
        self.intensity_plot_ax.set_xlabel("Time (s)")
        self.intensity_plot_ax.set_ylabel("Summed intensity")
        self.intensity_plot_ax.grid(True, alpha=0.3)

        (self.intensity_plot_line,) = self.intensity_plot_ax.plot(
            [], [], color="tab:red", linewidth=1.2, label="roi_sum"
        )

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

        self.pattern_status_label = QLabel(
            "Load a .npy pattern stack to start PLM playback."
        )

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

        gaussian_row = QHBoxLayout()

        gaussian_row.addWidget(self.gaussian_mfd_checkbox)

        gaussian_row.addWidget(self.gaussian_mfd_label, 1)

        live_panel.addLayout(gaussian_row)

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

        recovered_layout.addWidget(
            QLabel("Phase (RGB), Brightness Weighted by Amplitude")
        )

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

        self.pattern_gallery_status_label = QLabel(
            "Load PLM patterns to start gallery capture."
        )

        self.pattern_fields_save_checkbox = QCheckBox(
            "Save recovered fields each completed loop"
        )

        self.pattern_fields_save_path_edit = QLineEdit(
            str(Path.cwd() / "pattern_fields_history.pt")
        )

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

        self.channel_matrix_status_label = QLabel(
            "Channel matrix: waiting for target modes and recovered modes."
        )

        self.channel_matrix_xt_label = QLabel("XT (dB): n/a")

        self.channel_matrix_save_checkbox = QCheckBox(
            "Save channel matrix each completed loop"
        )

        self.channel_matrix_save_path_edit = QLineEdit(
            str(Path.cwd() / "channel_matrix_history.npz")
        )

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

        zernike_layout.addWidget(
            QLabel("Zernike polynomial coefficients (radians), radial degree n ≤ 5")
        )

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

        zernike_opt_row.addWidget(QLabel("SPGD iterations:"))

        zernike_opt_row.addWidget(self.zernike_opt_iterations_spin)

        zernike_opt_row.addSpacing(12)

        zernike_opt_row.addWidget(QLabel("Learning rate a0 (rad):"))

        zernike_opt_row.addWidget(self.zernike_opt_step_spin)

        zernike_opt_row.addSpacing(12)

        zernike_opt_row.addWidget(QLabel("Perturbation c0 (rad):"))

        zernike_opt_row.addWidget(self.zernike_opt_perturb_spin)

        zernike_opt_row.addSpacing(12)

        zernike_opt_row.addWidget(QLabel("Algorithm:"))

        zernike_opt_row.addWidget(self.zernike_opt_algorithm_combo)

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

        zernike_history_row = QHBoxLayout()

        zernike_history_row.addWidget(QLabel("Per-iteration CSV log:"))

        zernike_history_row.addWidget(self.zernike_opt_history_path_edit, 1)

        zernike_history_row.addWidget(self.zernike_opt_history_browse_button)

        zernike_layout.addLayout(zernike_history_row)

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

        phase_cal_tab = QWidget()

        phase_cal_layout = QVBoxLayout()

        phase_cal_grid_row = QHBoxLayout()

        self.phase_cal_spx_spin = QSpinBox()

        self.phase_cal_spx_spin.setRange(6, 510)

        self.phase_cal_spx_spin.setSingleStep(6)

        self.phase_cal_spx_spin.setValue(6)

        self.phase_cal_spy_spin = QSpinBox()

        self.phase_cal_spy_spin.setRange(9, 504)

        self.phase_cal_spy_spin.setSingleStep(9)

        self.phase_cal_spy_spin.setValue(9)

        self.phase_cal_ref_row_spin = QSpinBox()

        self.phase_cal_ref_row_spin.setRange(0, 4096)

        self.phase_cal_ref_row_spin.setValue(0)

        self.phase_cal_ref_col_spin = QSpinBox()

        self.phase_cal_ref_col_spin.setRange(0, 4096)

        self.phase_cal_ref_col_spin.setValue(0)

        self.phase_cal_ref_row_spin.setEnabled(False)

        self.phase_cal_ref_col_spin.setEnabled(False)

        self.phase_cal_sec_row_start_spin = QSpinBox()

        self.phase_cal_sec_row_start_spin.setRange(0, 4096)

        self.phase_cal_sec_row_start_spin.setValue(0)

        self.phase_cal_sec_row_end_spin = QSpinBox()

        self.phase_cal_sec_row_end_spin.setRange(0, 4096)

        self.phase_cal_sec_row_end_spin.setValue(0)

        self.phase_cal_sec_col_start_spin = QSpinBox()

        self.phase_cal_sec_col_start_spin.setRange(0, 4096)

        self.phase_cal_sec_col_start_spin.setValue(0)

        self.phase_cal_sec_col_end_spin = QSpinBox()

        self.phase_cal_sec_col_end_spin.setRange(0, 4096)

        self.phase_cal_sec_col_end_spin.setValue(0)

        phase_cal_grid_row.addWidget(QLabel("Super-pixel X:"))

        phase_cal_grid_row.addWidget(self.phase_cal_spx_spin)

        phase_cal_grid_row.addSpacing(10)

        phase_cal_grid_row.addWidget(QLabel("Super-pixel Y:"))

        phase_cal_grid_row.addWidget(self.phase_cal_spy_spin)

        phase_cal_grid_row.addSpacing(10)

        phase_cal_grid_row.addWidget(QLabel("Reference row:"))

        phase_cal_grid_row.addWidget(self.phase_cal_ref_row_spin)

        phase_cal_grid_row.addSpacing(10)

        phase_cal_grid_row.addWidget(QLabel("Reference col:"))

        phase_cal_grid_row.addWidget(self.phase_cal_ref_col_spin)

        phase_cal_grid_row.addSpacing(10)

        phase_cal_grid_row.addWidget(QLabel("Section rows:"))

        phase_cal_grid_row.addWidget(self.phase_cal_sec_row_start_spin)

        phase_cal_grid_row.addWidget(QLabel("to"))

        phase_cal_grid_row.addWidget(self.phase_cal_sec_row_end_spin)

        phase_cal_grid_row.addSpacing(10)

        phase_cal_grid_row.addWidget(QLabel("Section cols:"))

        phase_cal_grid_row.addWidget(self.phase_cal_sec_col_start_spin)

        phase_cal_grid_row.addWidget(QLabel("to"))

        phase_cal_grid_row.addWidget(self.phase_cal_sec_col_end_spin)

        phase_cal_grid_row.addStretch(1)

        phase_cal_avg_row = QHBoxLayout()

        self.phase_cal_drop_frames_spin = QSpinBox()

        self.phase_cal_drop_frames_spin.setRange(0, 50)

        self.phase_cal_drop_frames_spin.setValue(2)

        self.phase_cal_avg_frames_spin = QSpinBox()

        self.phase_cal_avg_frames_spin.setRange(1, 200)

        self.phase_cal_avg_frames_spin.setValue(8)

        self.phase_cal_use_live_roi_checkbox = QCheckBox(
            "Use Live ROI for intensity (recommended)"
        )

        self.phase_cal_use_live_roi_checkbox.setChecked(True)

        self.phase_cal_center_window_spin = QSpinBox()

        self.phase_cal_center_window_spin.setRange(1, 51)

        self.phase_cal_center_window_spin.setSingleStep(2)

        self.phase_cal_center_window_spin.setValue(3)

        self.phase_cal_enable_grating_checkbox = QCheckBox(
            "Apply horizontal PSI grating"
        )

        self.phase_cal_enable_grating_checkbox.setChecked(True)

        self.phase_cal_grating_period_spin = QSpinBox()

        self.phase_cal_grating_period_spin.setRange(1, 4096)

        self.phase_cal_grating_period_spin.setValue(6)

        phase_cal_avg_row.addWidget(QLabel("Drop frames/step:"))

        phase_cal_avg_row.addWidget(self.phase_cal_drop_frames_spin)

        phase_cal_avg_row.addSpacing(10)

        phase_cal_avg_row.addWidget(QLabel("Average frames/step:"))

        phase_cal_avg_row.addWidget(self.phase_cal_avg_frames_spin)

        phase_cal_avg_row.addSpacing(10)

        phase_cal_avg_row.addWidget(self.phase_cal_use_live_roi_checkbox)

        phase_cal_avg_row.addSpacing(10)

        phase_cal_avg_row.addWidget(QLabel("Center window:"))

        phase_cal_avg_row.addWidget(self.phase_cal_center_window_spin)

        phase_cal_avg_row.addSpacing(10)

        phase_cal_avg_row.addWidget(self.phase_cal_enable_grating_checkbox)

        phase_cal_avg_row.addWidget(QLabel("Period X:"))

        phase_cal_avg_row.addWidget(self.phase_cal_grating_period_spin)

        phase_cal_avg_row.addStretch(1)

        phase_cal_control_row = QHBoxLayout()

        self.phase_cal_generate_button = QPushButton("Generate Sweep Stack")

        self.phase_cal_start_button = QPushButton("Start Calibration")

        self.phase_cal_stop_button = QPushButton("Stop Calibration")

        self.phase_cal_stop_button.setEnabled(False)

        self.phase_cal_apply_checkbox = QCheckBox(
            "Apply aberration correction during display"
        )

        self.phase_cal_apply_checkbox.setChecked(False)

        self.phase_cal_visibility_button = QPushButton("Show Section Visibility")

        self.phase_cal_visibility_button.setCheckable(True)

        self.phase_cal_clear_button = QPushButton("Clear Map")

        phase_cal_control_row.addWidget(self.phase_cal_generate_button)

        phase_cal_control_row.addWidget(self.phase_cal_start_button)

        phase_cal_control_row.addWidget(self.phase_cal_stop_button)

        phase_cal_control_row.addSpacing(12)

        phase_cal_control_row.addWidget(self.phase_cal_apply_checkbox)

        phase_cal_control_row.addWidget(self.phase_cal_visibility_button)

        phase_cal_control_row.addWidget(self.phase_cal_clear_button)

        phase_cal_control_row.addStretch(1)

        phase_cal_save_row = QHBoxLayout()

        self.phase_cal_map_save_path_edit = QLineEdit(
            str(Path.cwd() / "phase_patterns" / "plm_phase_aberration_map.npz")
        )

        self.phase_cal_map_save_browse_button = QPushButton("Browse Save")

        self.phase_cal_map_save_button = QPushButton("Save Map")

        phase_cal_save_row.addWidget(QLabel("Save map path:"))

        phase_cal_save_row.addWidget(self.phase_cal_map_save_path_edit, 1)

        phase_cal_save_row.addWidget(self.phase_cal_map_save_browse_button)

        phase_cal_save_row.addWidget(self.phase_cal_map_save_button)

        phase_cal_load_row = QHBoxLayout()

        self.phase_cal_map_load_path_edit = QLineEdit(
            str(Path.cwd() / "phase_patterns" / "plm_phase_aberration_map.npz")
        )

        self.phase_cal_map_load_browse_button = QPushButton("Browse Load")

        self.phase_cal_map_load_button = QPushButton("Load Map")

        phase_cal_load_row.addWidget(QLabel("Load map path:"))

        phase_cal_load_row.addWidget(self.phase_cal_map_load_path_edit, 1)

        phase_cal_load_row.addWidget(self.phase_cal_map_load_browse_button)

        phase_cal_load_row.addWidget(self.phase_cal_map_load_button)

        self.phase_cal_info_label = QLabel("PSI: no sweep stack generated.")

        self.phase_cal_progress_label = QLabel("PSI progress: idle")

        self.phase_cal_status_label = QLabel("PSI status: waiting")

        self.phase_cal_map_stats_label = QLabel("Map: not available")

        self.phase_cal_loaded_map_label = QLabel("Loaded map: none")

        self.phase_cal_map_shape_label = QLabel("Map grid: n/a")

        self.phase_cal_map_canvas = FigureCanvas(Figure(figsize=(6.2, 3.8)))

        self.phase_cal_live_pattern_label = QLabel("Current PLM pattern: idle")

        self.phase_cal_live_pattern_canvas = FigureCanvas(Figure(figsize=(6.2, 3.0)))

        phase_cal_layout.addLayout(phase_cal_grid_row)

        phase_cal_layout.addLayout(phase_cal_avg_row)

        phase_cal_layout.addLayout(phase_cal_control_row)

        phase_cal_layout.addLayout(phase_cal_save_row)

        phase_cal_layout.addLayout(phase_cal_load_row)

        phase_cal_layout.addWidget(self.phase_cal_info_label)

        phase_cal_layout.addWidget(self.phase_cal_progress_label)

        phase_cal_layout.addWidget(self.phase_cal_status_label)

        phase_cal_layout.addWidget(self.phase_cal_map_stats_label)

        phase_cal_layout.addWidget(self.phase_cal_loaded_map_label)

        phase_cal_layout.addWidget(self.phase_cal_map_shape_label)

        phase_cal_layout.addWidget(self.phase_cal_map_canvas, 1)

        phase_cal_layout.addWidget(self.phase_cal_live_pattern_label)

        phase_cal_layout.addWidget(self.phase_cal_live_pattern_canvas, 1)

        phase_cal_layout.addStretch(1)

        phase_cal_tab.setLayout(phase_cal_layout)

        self.tab_widget = QTabWidget()

        self.tab_widget.addTab(live_fft_tab, "Live + FFT")

        self.tab_widget.addTab(phase_plot_tab, "Phase Plot")

        self.tab_widget.addTab(intensity_plot_tab, "Intensity Plot")

        self.tab_widget.addTab(zernike_tab, "Zernike Control")

        self.tab_widget.addTab(phase_cal_tab, "Phase Calibration")

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

        controls.addSpacing(12)

        controls.addWidget(QLabel("Dark avg:"))

        controls.addWidget(self.dark_frame_avg_combo)

        controls.addWidget(self.dark_frame_capture_button)

        controls.addWidget(self.dark_frame_clear_button)

        controls.addWidget(self.dark_frame_status_label)

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
        #self.timer = QTimer(self)
        #self.timer.setInterval(30)
        #self.timer.timeout.connect(self._update_frame)
        self._streaming = False

        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self._frame_count = 0

        self._last_ui_update_time = 0.0

        self._last_frame_error: Optional[str] = None
        self._master_dark_frame: Optional[np.ndarray] = None
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

        self._phase_cal_generated_patterns: Optional[np.ndarray] = None

        self._phase_cal_total_patterns = 0

        self._phase_cal_pattern_meta: list[tuple[int, int]] = []

        self._phase_cal_pattern_index_by_signal_step: dict[tuple[int, int], int] = {}

        self._phase_cal_signal_indices: list[int] = []

        self._phase_cal_grid_rows = 0

        self._phase_cal_grid_cols = 0

        self._phase_cal_spx = 0

        self._phase_cal_spy = 0

        self._phase_cal_ref_flat = 0

        self._phase_cal_section_row_start = 0

        self._phase_cal_section_row_end = 0

        self._phase_cal_section_col_start = 0

        self._phase_cal_section_col_end = 0

        self._phase_cal_grating_period_x = 6

        self._phase_cal_visibility_running = False

        self._phase_cal_visibility_restore_pending = False

        self._phase_cal_visibility_previous_patterns: Optional[np.ndarray] = None

        self._phase_cal_visibility_previous_pattern_path: Optional[str] = None

        self._phase_cal_running = False

        self._phase_cal_completed_this_run = False

        self._phase_cal_restore_pending = False

        self._phase_cal_previous_patterns: Optional[np.ndarray] = None

        self._phase_cal_previous_pattern_path: Optional[str] = None

        self._phase_cal_step_sum: Optional[np.ndarray] = None

        self._phase_cal_step_count: Optional[np.ndarray] = None

        self._phase_cal_step_mean: Optional[np.ndarray] = None

        self._phase_cal_prev_drop_combo_index: Optional[int] = None

        self._phase_cal_last_pattern_seq = -1

        self._phase_cal_drop_remaining_for_seq = 0

        self._phase_cal_superpixel_map: Optional[np.ndarray] = None

        self._phase_cal_fullres_map: Optional[np.ndarray] = None

        self._phase_cal_active_correction_map: Optional[np.ndarray] = None

        self._phase_cal_modulation_superpixel: Optional[np.ndarray] = None

        self._phase_cal_fullres_shape: Optional[tuple[int, int]] = None

        self._phase_cal_active_correction_shape: Optional[tuple[int, int]] = None

        self._phase_cal_loaded_map_path: Optional[str] = None

        self._phase_cal_last_live_plot_t = 0.0

        self._phase_cal_latest_plm_phase_pattern: Optional[np.ndarray] = None

        self._grating_period_x_px = 0

        self._grating_period_y_px = 0

        self._zernike_phase_cache_shape: Optional[tuple[int, int]] = None

        self._zernike_phase_cache_coeffs: Optional[tuple[float, ...]] = None

        self._zernike_phase_cache_map: Optional[np.ndarray] = None

        self._zernike_optimizer_thread: Optional[threading.Thread] = None

        self._zernike_optimizer_stop_event = threading.Event()

        self._zernike_optimizer_metric_condition = threading.Condition()

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

        self._zernike_optimizer_algorithm = "aspgd"

        self._completed_pattern_loops = 0

        self._last_plm_counter_index: Optional[int] = None

        self._phase_colormap_name = "viridis"

        self._gaussian_pixel_pitch_um = 15.0

        self._gaussian_mfd_enabled = False

        self._phase_plot_dirty = False

        self._intensity_plot_dirty = False

        self._plot_refresh_timer = QTimer(self)

        self.processor_thread = FrameProcessorThread(self)
        self.processor_thread.frame_processed.connect(self._on_frame_processed)
        self.processor_thread.error_occurred.connect(self._on_processor_error)

        self._plot_refresh_timer.setInterval(100)

        self._plot_refresh_timer.timeout.connect(self._refresh_plots_if_dirty)

        self._plot_refresh_timer.start()

        self.start_button.clicked.connect(self._start_live)

        self.stop_button.clicked.connect(self._stop_live)

        self.dark_frame_capture_button.clicked.connect(self._capture_dark_frame_snapshot)

        self.dark_frame_clear_button.clicked.connect(self._clear_dark_frame)

        self.auto_exposure_checkbox.toggled.connect(self._on_auto_exposure_toggled)

        self.exposure_spinbox.editingFinished.connect(self._on_manual_exposure_changed)

        self.pattern_browse_button.clicked.connect(self._on_browse_pattern_file)

        self.pattern_load_button.clicked.connect(self._on_load_pattern_file)

        self.pattern_start_button.clicked.connect(self._start_plm_pattern_loop)

        self.pattern_stop_button.clicked.connect(self._stop_plm_pattern_loop)

        self.plm_counter_updated.connect(self._on_plm_counter_updated)

        self.plm_phase_preview_updated.connect(self._on_plm_phase_preview_updated)

        self.plm_loop_finished.connect(self._on_plm_loop_finished)

        self.logging_browse_button.clicked.connect(self._on_browse_log_file)

        self.logging_start_button.clicked.connect(self._start_logging)

        self.logging_stop_button.clicked.connect(self._stop_logging)

        self.trigger_delay_edit.editingFinished.connect(
            self._on_trigger_delay_editing_finished
        )

        self.grating_x_edit.editingFinished.connect(self._on_grating_input_changed)

        self.grating_y_edit.editingFinished.connect(self._on_grating_input_changed)

        self.gaussian_mfd_checkbox.toggled.connect(self._on_gaussian_mfd_toggled)

        self.zernike_reset_button.clicked.connect(self._reset_zernike_coefficients)

        self.zernike_max_degree_combo.currentIndexChanged.connect(
            self._on_zernike_degree_limit_changed
        )

        self.zernike_opt_start_button.clicked.connect(self._start_zernike_optimizer)

        self.zernike_opt_stop_button.clicked.connect(self._stop_zernike_optimizer)

        self.zernike_opt_results_browse_button.clicked.connect(
            self._on_browse_zernike_opt_results_file
        )

        self.zernike_opt_history_browse_button.clicked.connect(
            self._on_browse_zernike_opt_history_file
        )

        self.zernike_optimizer_status_updated.connect(
            self._set_zernike_optimizer_status
        )

        self.zernike_optimizer_finished.connect(self._on_zernike_optimizer_finished)

        self.zernike_coefficients_apply_requested.connect(
            self._apply_zernike_coefficients
        )

        self.colormap_viridis_radio.toggled.connect(self._on_colormap_toggled)

        self.colormap_plasma_radio.toggled.connect(self._on_colormap_toggled)

        self.tab_widget.currentChanged.connect(self._on_left_tab_changed)

        self.right_tab_widget.currentChanged.connect(self._on_right_tab_changed)

        self.channel_matrix_save_browse_button.clicked.connect(
            self._on_browse_channel_matrix_save_file
        )

        self.channel_matrix_save_now_button.clicked.connect(
            self._on_save_channel_matrix_now
        )

        self.pattern_fields_save_browse_button.clicked.connect(
            self._on_browse_pattern_fields_save_file
        )

        self.pattern_fields_save_now_button.clicked.connect(
            self._on_save_pattern_fields_now
        )

        self.phase_cal_generate_button.clicked.connect(
            self._on_generate_phase_calibration_patterns
        )

        self.phase_cal_start_button.clicked.connect(self._start_phase_calibration)

        self.phase_cal_stop_button.clicked.connect(self._stop_phase_calibration)

        self.phase_cal_map_save_browse_button.clicked.connect(
            self._on_browse_phase_calibration_map_save_file
        )

        self.phase_cal_map_load_browse_button.clicked.connect(
            self._on_browse_phase_calibration_map_load_file
        )

        self.phase_cal_map_save_button.clicked.connect(self._save_phase_calibration_map)

        self.phase_cal_map_load_button.clicked.connect(self._load_phase_calibration_map)

        self.phase_cal_clear_button.clicked.connect(self._clear_phase_calibration_map)

        self.phase_cal_visibility_button.toggled.connect(
            self._toggle_phase_calibration_visibility
        )

        self.phase_cal_spx_spin.valueChanged.connect(
            self._on_phase_calibration_superpixel_changed
        )

        self.phase_cal_spy_spin.valueChanged.connect(
            self._on_phase_calibration_superpixel_changed
        )

        self.phase_cal_sec_row_start_spin.valueChanged.connect(
            self._on_phase_calibration_section_changed
        )

        self.phase_cal_sec_row_end_spin.valueChanged.connect(
            self._on_phase_calibration_section_changed
        )

        self.phase_cal_sec_col_start_spin.valueChanged.connect(
            self._on_phase_calibration_section_changed
        )

        self.phase_cal_sec_col_end_spin.valueChanged.connect(
            self._on_phase_calibration_section_changed
        )

        self._exp_auto_feature_name: Optional[str] = (
            self._detect_exposure_auto_feature_name()
        )

        self._exp_time_feature_name: Optional[str] = (
            self._detect_exposure_time_feature_name()
        )

        self._trigger_delay_feature_name: Optional[str] = (
            self._detect_trigger_delay_feature_name()
        )

        self._initialize_exposure_controls()
        self._initialize_trigger_delay_controls()
        self._on_grating_input_changed()
        self._update_zernike_visibility()
        self._on_zernike_coeff_changed(0.0)
        self._update_phase_colorbar()

        self.phase_cal_ref_row_spin.valueChanged.connect(
            self._on_phase_calibration_ref_changed
        )

        self.phase_cal_ref_col_spin.valueChanged.connect(
            self._on_phase_calibration_ref_changed
        )

        self.phase_cal_apply_checkbox.toggled.connect(
            self._on_phase_calibration_apply_toggled
        )

        self.phase_cal_center_window_spin.valueChanged.connect(
            self._on_phase_calibration_center_window_changed
        )

        self._on_phase_calibration_superpixel_changed()
        self._on_phase_calibration_center_window_changed(
            int(self.phase_cal_center_window_spin.value())
        )

        self._on_phase_calibration_ref_changed()
        self._refresh_phase_calibration_map_plot()
        self._refresh_phase_calibration_live_pattern_plot(None)
        self._load_default_target_modes()
        self._refresh_channel_matrix_plot(force=True)

    def _on_left_tab_changed(self, _index: int) -> None:
        self._refresh_plots_if_dirty(force=True)

    def _on_right_tab_changed(self, _index: int) -> None:
        self._refresh_plots_if_dirty(force=True)

    def _on_colormap_toggled(self, _checked: bool) -> None:

        self._phase_colormap_name = (
            "plasma" if self.colormap_plasma_radio.isChecked() else "viridis"
        )

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

    def _capture_zernike_coefficients_snapshot(self) -> dict[tuple[int, int], float]:

        coeffs: dict[tuple[int, int], float] = {}

        for term, spin in self._zernike_coeff_spins.items():

            coeffs[term] = float(spin.value())

        return coeffs

    def _on_browse_zernike_opt_results_file(self) -> None:

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Select optimizer coefficient output file",
            self.zernike_opt_results_path_edit.text().strip()
            or str(Path.cwd() / "zernike_optimization_coefficients.txt"),
            "Text files (*.txt);;All files (*)",
        )

        if not file_path:

            return

        self.zernike_opt_results_path_edit.setText(file_path)

    def _on_browse_zernike_opt_history_file(self) -> None:

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Select optimizer per-iteration CSV file",
            self.zernike_opt_history_path_edit.text().strip()
            or str(Path.cwd() / "zernike_optimization_history.csv"),
            "CSV files (*.csv)",
        )

        if not file_path:

            return

        self.zernike_opt_history_path_edit.setText(file_path)

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

    def _wait_for_next_loop_xt_metric(
        self, last_seen_loop: int, timeout_s: float = 60.0
    ) -> Optional[tuple[int, float]]:

        deadline = perf_counter() + max(0.1, float(timeout_s))

        while not self._zernike_optimizer_stop_event.is_set():

            with self._zernike_optimizer_metric_condition:

                loop_now = int(self._completed_pattern_loops)

                xt_now = self._channel_matrix_xt_db

                if (
                    loop_now > int(last_seen_loop)
                    and xt_now is not None
                    and np.isfinite(float(xt_now))
                ):

                    return loop_now, float(xt_now)

                remaining = deadline - perf_counter()

                if remaining <= 0.0:

                    return None

                self._zernike_optimizer_metric_condition.wait(
                    timeout=min(0.5, remaining)
                )

        return None

    def _coeff_vector_to_dict(
        self, enabled_terms: list[tuple[int, int]], coeff_vector: np.ndarray
    ) -> dict[tuple[int, int], float]:

        coeff_dict = self._capture_zernike_coefficients_snapshot()

        for idx, term in enumerate(enabled_terms):

            coeff_dict[term] = float(np.clip(coeff_vector[idx], -20.0, 20.0))

        return coeff_dict

    def _evaluate_zernike_optimizer_candidate(
        self,
        enabled_terms: list[tuple[int, int]],
        coeff_vector: np.ndarray,
        last_loop: int,
        timeout_s: float = 120.0,
    ) -> Optional[tuple[int, float]]:

        self.zernike_coefficients_apply_requested.emit(
            self._coeff_vector_to_dict(enabled_terms, coeff_vector)
        )

        return self._wait_for_next_loop_xt_metric(last_loop, timeout_s=timeout_s)

    def _start_zernike_optimizer(self) -> None:

        if (
            self._zernike_optimizer_thread is not None
            and self._zernike_optimizer_thread.is_alive()
        ):

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

        learning_rate = float(self.zernike_opt_step_spin.value())

        perturbation = float(self.zernike_opt_perturb_spin.value())

        max_degree = self._get_selected_zernike_max_degree()

        optimizer_algorithm = str(
            self.zernike_opt_algorithm_combo.currentData()
            or self._zernike_optimizer_algorithm
            or "aspgd"
        )

        if optimizer_algorithm not in ("aspgd", "nelder_mead"):

            optimizer_algorithm = "aspgd"

        self._zernike_optimizer_algorithm = optimizer_algorithm

        results_path = self.zernike_opt_results_path_edit.text().strip()

        history_path = self.zernike_opt_history_path_edit.text().strip()

        initial_coeffs = self._capture_zernike_coefficients_snapshot()

        self._zernike_optimizer_stop_event.clear()

        self.zernike_opt_start_button.setEnabled(False)

        self.zernike_opt_stop_button.setEnabled(True)

        self._set_zernike_optimizer_status(
            f"Optimizer: running ({optimizer_algorithm})"
        )

        self._zernike_optimizer_thread = threading.Thread(
            target=self._run_zernike_optimizer,
            args=(
                enabled_terms,
                initial_coeffs,
                iterations,
                learning_rate,
                perturbation,
                max_degree,
                optimizer_algorithm,
                results_path,
                history_path,
            ),
            daemon=True,
        )

        self._zernike_optimizer_thread.start()

    def _stop_zernike_optimizer(self) -> None:

        if (
            self._zernike_optimizer_thread is None
            or not self._zernike_optimizer_thread.is_alive()
        ):

            self._set_zernike_optimizer_status("Optimizer: not running")

            return

        self._zernike_optimizer_stop_event.set()

        self._set_zernike_optimizer_status("Optimizer: stopping...")

    def _on_zernike_optimizer_finished(self, _success: bool, message: str) -> None:

        self.zernike_opt_start_button.setEnabled(True)

        self.zernike_opt_stop_button.setEnabled(False)

        self._zernike_optimizer_thread = None

        self._set_zernike_optimizer_status(message)

    def _run_zernike_optimizer(
        self,
        enabled_terms: list[tuple[int, int]],
        initial_coeffs: dict[tuple[int, int], float],
        iterations: int,
        learning_rate: float,
        perturbation: float,
        max_degree: int,
        optimizer_algorithm: str,
        results_path: str,
        history_path: str,
    ) -> None:

        try:

            history_path_text = str(history_path).strip()

            if history_path_text == "":

                self.zernike_optimizer_finished.emit(
                    False, "Optimizer stopped: missing per-iteration CSV path"
                )

                return

            history_file_path = Path(history_path_text)

            history_file_path.parent.mkdir(parents=True, exist_ok=True)

            optimizer_algorithm = str(optimizer_algorithm or "aspgd")

            if optimizer_algorithm not in ("aspgd", "nelder_mead"):

                optimizer_algorithm = "aspgd"

            local_rng = np.random.default_rng()

            term_columns = [f"Z({n},{m})" for (n, m) in enabled_terms]

            csv_columns = [
                "timestamp_iso",
                "iteration",
                "phase",
                "loop_index",
                "xt_db",
                "best_xt_db",
                "xt_plus_db",
                "xt_minus_db",
                "learning_rate_ak",
                "perturbation_ck",
                "gradient_norm",
                "optimizer_algorithm",
            ] + term_columns

            theta = np.array(
                [float(initial_coeffs.get(term, 0.0)) for term in enabled_terms],
                dtype=np.float64,
            )

            m = np.zeros_like(theta)

            v = np.zeros_like(theta)

            beta1 = 0.9

            beta2 = 0.99

            eps = 1e-8

            a0 = max(1e-5, float(learning_rate))

            c0 = max(1e-5, float(perturbation))

            best_theta = np.array(theta, copy=True)

            best_xt_db: Optional[float] = None

            with history_file_path.open("w", newline="", encoding="utf-8") as csv_fp:

                writer = csv.DictWriter(csv_fp, fieldnames=csv_columns)

                writer.writeheader()

                def _write_row(
                    iteration_idx: int,
                    phase_label: str,
                    loop_idx: int,
                    xt_value: float,
                    best_xt_value: float,
                    xt_plus_value: Optional[float],
                    xt_minus_value: Optional[float],
                    ak_value: float,
                    ck_value: float,
                    grad_norm_value: float,
                    coeff_vector_value: np.ndarray,
                ) -> None:

                    row: dict[str, object] = {
                        "timestamp_iso": datetime.now().isoformat(),
                        "iteration": int(iteration_idx),
                        "phase": str(phase_label),
                        "loop_index": int(loop_idx),
                        "xt_db": float(xt_value),
                        "best_xt_db": float(best_xt_value),
                        "xt_plus_db": None
                        if xt_plus_value is None
                        else float(xt_plus_value),
                        "xt_minus_db": None
                        if xt_minus_value is None
                        else float(xt_minus_value),
                        "learning_rate_ak": float(ak_value),
                        "perturbation_ck": float(ck_value),
                        "gradient_norm": float(grad_norm_value),
                        "optimizer_algorithm": str(optimizer_algorithm),
                    }

                    for idx, column in enumerate(term_columns):

                        row[column] = float(coeff_vector_value[idx])

                    writer.writerow(row)

                    csv_fp.flush()

                last_loop = int(self._completed_pattern_loops)

                self.zernike_optimizer_status_updated.emit(
                    f"Optimizer: baseline evaluation ({optimizer_algorithm})"
                )

                baseline_metric = self._evaluate_zernike_optimizer_candidate(
                    enabled_terms,
                    theta,
                    last_loop,
                    timeout_s=120.0,
                )

                if baseline_metric is None:

                    self.zernike_optimizer_finished.emit(
                        False, "Optimizer stopped: no loop metric"
                    )

                    return

                last_loop, baseline_xt_db = baseline_metric

                best_xt_db = float(baseline_xt_db)

                _write_row(
                    0,
                    "baseline",
                    int(last_loop),
                    float(baseline_xt_db),
                    float(best_xt_db),
                    None,
                    None,
                    0.0,
                    0.0,
                    0.0,
                    theta,
                )

                if optimizer_algorithm == "nelder_mead":

                    n_dims = int(theta.shape[0])

                    simplex = np.tile(theta[None, :], (n_dims + 1, 1)).astype(
                        np.float64, copy=False
                    )

                    nm_step = max(1e-5, float(c0))

                    for dim in range(n_dims):

                        simplex[dim + 1, dim] = float(
                            np.clip(simplex[dim + 1, dim] + nm_step, -20.0, 20.0)
                        )

                    values = np.full(n_dims + 1, np.inf, dtype=np.float64)

                    values[0] = float(baseline_xt_db)

                    for vertex_idx in range(1, n_dims + 1):

                        metric_vertex = self._evaluate_zernike_optimizer_candidate(
                            enabled_terms,
                            simplex[vertex_idx],
                            last_loop,
                            timeout_s=120.0,
                        )

                        if metric_vertex is None:

                            self.zernike_optimizer_finished.emit(
                                False,
                                "Optimizer stopped: timed out initializing Nelder-Mead simplex",
                            )

                            return

                        last_loop, vertex_xt_db = metric_vertex

                        values[vertex_idx] = float(vertex_xt_db)

                    order = np.argsort(values)

                    simplex = simplex[order]

                    values = values[order]

                    best_xt_db = float(values[0])

                    best_theta = np.array(simplex[0], copy=True)

                    alpha = 1.0

                    gamma = 2.0

                    rho = 0.5

                    sigma = 0.5

                    for it in range(1, int(iterations) + 1):

                        if self._zernike_optimizer_stop_event.is_set():

                            self.zernike_optimizer_finished.emit(
                                False, "Optimizer stopped by user"
                            )

                            return

                        order = np.argsort(values)

                        simplex = simplex[order]

                        values = values[order]

                        centroid = np.mean(simplex[:-1], axis=0)

                        worst = simplex[-1]

                        worst_value = float(values[-1])

                        second_worst_value = float(values[-2])

                        best_value = float(values[0])

                        phase_label = "reflect"

                        xr = np.clip(centroid + alpha * (centroid - worst), -20.0, 20.0)

                        metric_reflect = self._evaluate_zernike_optimizer_candidate(
                            enabled_terms,
                            xr,
                            last_loop,
                            timeout_s=120.0,
                        )

                        if metric_reflect is None:

                            self.zernike_optimizer_finished.emit(
                                False,
                                "Optimizer stopped: timed out evaluating Nelder-Mead reflection",
                            )

                            return

                        last_loop, fr = metric_reflect

                        if float(fr) < best_value:

                            xe = np.clip(
                                centroid + gamma * (xr - centroid), -20.0, 20.0
                            )

                            metric_expand = self._evaluate_zernike_optimizer_candidate(
                                enabled_terms,
                                xe,
                                last_loop,
                                timeout_s=120.0,
                            )

                            if metric_expand is None:

                                self.zernike_optimizer_finished.emit(
                                    False,
                                    "Optimizer stopped: timed out evaluating Nelder-Mead expansion",
                                )

                                return

                            last_loop, fe = metric_expand

                            if float(fe) < float(fr):

                                simplex[-1] = xe

                                values[-1] = float(fe)

                                phase_label = "expand"

                            else:

                                simplex[-1] = xr

                                values[-1] = float(fr)

                                phase_label = "reflect-best"

                        elif float(fr) < second_worst_value:

                            simplex[-1] = xr

                            values[-1] = float(fr)

                            phase_label = "reflect-accept"

                        else:

                            do_shrink = False

                            if float(fr) < worst_value:

                                xc = np.clip(
                                    centroid + rho * (xr - centroid), -20.0, 20.0
                                )

                                metric_contract_out = self._evaluate_zernike_optimizer_candidate(
                                    enabled_terms,
                                    xc,
                                    last_loop,
                                    timeout_s=120.0,
                                )

                                if metric_contract_out is None:

                                    self.zernike_optimizer_finished.emit(
                                        False,
                                        "Optimizer stopped: timed out evaluating Nelder-Mead outside contraction",
                                    )

                                    return

                                last_loop, fc = metric_contract_out

                                if float(fc) <= float(fr):

                                    simplex[-1] = xc

                                    values[-1] = float(fc)

                                    phase_label = "contract-out"

                                else:

                                    do_shrink = True

                            else:

                                xc = np.clip(
                                    centroid - rho * (centroid - worst), -20.0, 20.0
                                )

                                metric_contract_in = self._evaluate_zernike_optimizer_candidate(
                                    enabled_terms,
                                    xc,
                                    last_loop,
                                    timeout_s=120.0,
                                )

                                if metric_contract_in is None:

                                    self.zernike_optimizer_finished.emit(
                                        False,
                                        "Optimizer stopped: timed out evaluating Nelder-Mead inside contraction",
                                    )

                                    return

                                last_loop, fc = metric_contract_in

                                if float(fc) < worst_value:

                                    simplex[-1] = xc

                                    values[-1] = float(fc)

                                    phase_label = "contract-in"

                                else:

                                    do_shrink = True

                            if do_shrink:

                                best_vertex = np.array(simplex[0], copy=True)

                                for vertex_idx in range(1, n_dims + 1):

                                    simplex[vertex_idx] = np.clip(
                                        best_vertex
                                        + sigma * (simplex[vertex_idx] - best_vertex),
                                        -20.0,
                                        20.0,
                                    )

                                    metric_shrink = self._evaluate_zernike_optimizer_candidate(
                                        enabled_terms,
                                        simplex[vertex_idx],
                                        last_loop,
                                        timeout_s=120.0,
                                    )

                                    if metric_shrink is None:

                                        self.zernike_optimizer_finished.emit(
                                            False,
                                            "Optimizer stopped: timed out evaluating Nelder-Mead shrink",
                                        )

                                        return

                                    last_loop, fs = metric_shrink

                                    values[vertex_idx] = float(fs)

                                phase_label = "shrink"

                        order = np.argsort(values)

                        simplex = simplex[order]

                        values = values[order]

                        current_best_xt_db = float(values[0])

                        current_best_theta = np.array(simplex[0], copy=True)

                        if current_best_xt_db < float(best_xt_db):

                            best_xt_db = float(current_best_xt_db)

                            best_theta = np.array(current_best_theta, copy=True)

                        _write_row(
                            it,
                            phase_label,
                            int(last_loop),
                            float(current_best_xt_db),
                            float(best_xt_db),
                            None,
                            None,
                            0.0,
                            float(nm_step),
                            0.0,
                            current_best_theta,
                        )

                        self.zernike_optimizer_status_updated.emit(
                            f"Optimizer(Nelder-Mead): iter {it}/{iterations}, XT={current_best_xt_db:.3f} dB (best {float(best_xt_db):.3f} dB)"
                        )

                else:

                    m = np.zeros_like(theta)

                    v = np.zeros_like(theta)

                    beta1 = 0.9

                    beta2 = 0.99

                    eps = 1e-8

                    for it in range(1, int(iterations) + 1):

                        if self._zernike_optimizer_stop_event.is_set():

                            self.zernike_optimizer_finished.emit(
                                False, "Optimizer stopped by user"
                            )

                            return

                        ak = a0 / math.pow(float(it) + 9.0, 0.602)

                        ck = c0 / math.pow(float(it), 0.101)

                        delta = local_rng.choice([-1.0, 1.0], size=theta.shape[0])

                        theta_plus = np.clip(theta + ck * delta, -20.0, 20.0)

                        metric_plus = self._evaluate_zernike_optimizer_candidate(
                            enabled_terms,
                            theta_plus,
                            last_loop,
                            timeout_s=120.0,
                        )

                        if metric_plus is None:

                            self.zernike_optimizer_finished.emit(
                                False, "Optimizer stopped: timed out waiting for XT(+)"
                            )

                            return

                        last_loop, xt_plus_db = metric_plus

                        theta_minus = np.clip(theta - ck * delta, -20.0, 20.0)

                        metric_minus = self._evaluate_zernike_optimizer_candidate(
                            enabled_terms,
                            theta_minus,
                            last_loop,
                            timeout_s=120.0,
                        )

                        if metric_minus is None:

                            self.zernike_optimizer_finished.emit(
                                False, "Optimizer stopped: timed out waiting for XT(-)"
                            )

                            return

                        last_loop, xt_minus_db = metric_minus

                        grad_scale = (float(xt_plus_db) - float(xt_minus_db)) / (
                            2.0 * max(1e-8, float(ck))
                        )

                        grad = grad_scale * delta

                        m = beta1 * m + (1.0 - beta1) * grad

                        v = beta2 * v + (1.0 - beta2) * (grad * grad)

                        m_hat = m / (1.0 - math.pow(beta1, float(it)))

                        v_hat = v / (1.0 - math.pow(beta2, float(it)))

                        theta = np.clip(
                            theta - float(ak) * (m_hat / (np.sqrt(v_hat) + eps)),
                            -20.0,
                            20.0,
                        )

                        metric_current = self._evaluate_zernike_optimizer_candidate(
                            enabled_terms,
                            theta,
                            last_loop,
                            timeout_s=120.0,
                        )

                        if metric_current is None:

                            self.zernike_optimizer_finished.emit(
                                False,
                                "Optimizer stopped: timed out waiting for XT(updated)",
                            )

                            return

                        last_loop, current_xt_db = metric_current

                        if float(current_xt_db) < float(best_xt_db):

                            best_xt_db = float(current_xt_db)

                            best_theta = np.array(theta, copy=True)

                        grad_norm = float(np.linalg.norm(grad))

                        _write_row(
                            it,
                            "update",
                            int(last_loop),
                            float(current_xt_db),
                            float(best_xt_db),
                            float(xt_plus_db),
                            float(xt_minus_db),
                            float(ak),
                            float(ck),
                            float(grad_norm),
                            theta,
                        )

                        self.zernike_optimizer_status_updated.emit(
                            f"Optimizer(ASPGD): iter {it}/{iterations}, XT={float(current_xt_db):.3f} dB (best {float(best_xt_db):.3f} dB)"
                        )

            best_coeffs = self._coeff_vector_to_dict(enabled_terms, best_theta)

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
                    f"Optimizer finished ({optimizer_algorithm}): best XT={best_xt_db:.3f} dB | coeffs saved {save_info} | csv {history_file_path}",
                )

            else:

                self.zernike_optimizer_finished.emit(
                    True,
                    f"Optimizer finished ({optimizer_algorithm}): best XT={best_xt_db:.3f} dB | csv {history_file_path} | coeff save failed: {save_info}",
                )

        except Exception as exc:

            self.zernike_optimizer_finished.emit(False, f"Optimizer error: {exc}")

    def _enumerate_zernike_terms(self, max_radial_degree: int) -> list[tuple[int, int]]:

        terms: list[tuple[int, int]] = []

        for n in range(max(0, int(max_radial_degree)) + 1):

            for m in range(-n, n + 1, 2):

                terms.append((n, m))

        return terms

    def _zernike_radial_polynomial(
        self, n: int, m_abs: int, r: np.ndarray
    ) -> np.ndarray:

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
            (
                round(float(self._zernike_coeff_spins[(n, m)].value()), 6)
                if (n, m) in enabled_terms
                else 0.0
            )
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

            rms = (
                float(np.sqrt(np.mean(np.square(mode[aperture]))))
                if np.any(aperture)
                else 0.0
            )

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

            self.pattern_gallery_status_label.setText(
                "Load PLM patterns to start gallery capture."
            )

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

        image = QImage(
            contiguous.data, width, height, width * 3, QImage.Format_RGB888
        ).copy()

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

        self._pattern_fields_by_pattern[int(pattern_index)] = np.asarray(
            recovered_field, dtype=np.complex64
        ).copy()

    def _clear_pattern_fields_data(self) -> None:

        self._pattern_fields_by_pattern = {}

        self._current_pattern_fields_loop = None

    def _build_current_pattern_fields_loop(self) -> Optional[np.ndarray]:

        if not self._is_current_pattern_field_loop_complete():

            return None

        if (
            self._current_plm_total_patterns is not None
            and int(self._current_plm_total_patterns) > 0
        ):

            expected_modes = int(self._current_plm_total_patterns)

        else:

            expected_modes = len(self._pattern_preview_labels)

        try:

            return np.stack(
                [
                    np.asarray(self._pattern_fields_by_pattern[idx], dtype=np.complex64)
                    for idx in range(1, expected_modes + 1)
                ],
                axis=0,
            )

        except Exception:

            return None

    def _is_current_pattern_field_loop_complete(self) -> bool:

        if (
            self._current_plm_total_patterns is not None
            and int(self._current_plm_total_patterns) > 0
        ):

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
            self.pattern_fields_save_path_edit.text().strip()
            or str(Path.cwd() / "pattern_fields_history.pt"),
            "PyTorch tensor (*.pt);;NumPy archive (*.npz)",
        )

        if not file_path:

            return

        self.pattern_fields_save_path_edit.setText(file_path)

    def _save_pattern_fields_history(self) -> bool:

        if self._current_pattern_fields_loop is None:

            self.pattern_gallery_status_label.setText(
                "Pattern fields save skipped: no completed loop available."
            )

            return False

        path_text = self.pattern_fields_save_path_edit.text().strip()

        if not path_text:

            self.pattern_gallery_status_label.setText(
                "Pattern fields save failed: output path is empty."
            )

            return False

        output_path = Path(path_text)

        try:

            output_path.parent.mkdir(parents=True, exist_ok=True)

            fields_complex = np.asarray(
                self._current_pattern_fields_loop, dtype=np.complex64
            )

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

            self.pattern_gallery_status_label.setText(
                f"Pattern fields save failed: {exc}"
            )

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

                raise ValueError(
                    f"target modes must be 3D [n, y, x], got {modes.shape}"
                )

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

            self.channel_matrix_status_label.setText(
                f"Channel matrix: failed to load target modes ({exc})"
            )

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

        if (
            self._current_plm_total_patterns is not None
            and int(self._current_plm_total_patterns) > 0
        ):

            expected_modes = int(self._current_plm_total_patterns)

        else:

            expected_modes = int(target_modes.shape[0])

        mode_template = np.zeros(
            (target_modes.shape[1], target_modes.shape[2]), dtype=np.complex64
        )

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

                self.channel_matrix_xt_label.setText(
                    f"XT (dB): {self._channel_matrix_xt_db:.3f}"
                )

            else:

                self.channel_matrix_xt_label.setText("XT (dB): n/a")

            self.channel_matrix_status_label.setText(
                f"Channel matrix: updated({captured_modes}/{expected_modes} captured this loop), matrix shape{self._channel_matrix.shape}."
            )

        except Exception as exc:

            self._channel_matrix = None

            self._channel_matrix_xt_db = None

            self.channel_matrix_xt_label.setText("XT (dB): n/a")

            self.channel_matrix_status_label.setText(
                f"Channel matrix update failed: {exc}"
            )

    def _is_current_channel_matrix_complete_square(self) -> bool:

        if self._channel_matrix is None:

            return False

        matrix_now = np.asarray(self._channel_matrix)

        if matrix_now.ndim != 2:

            return False

        if (
            self._current_plm_total_patterns is not None
            and int(self._current_plm_total_patterns) > 0
        ):

            expected_modes = int(self._current_plm_total_patterns)

        else:

            expected_modes = (
                int(self._target_modes.shape[0])
                if self._target_modes is not None
                else 0
            )

        if expected_modes <= 0:

            return False

        if (
            matrix_now.shape[0] != expected_modes
            or matrix_now.shape[1] != expected_modes
        ):

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
            self.channel_matrix_save_path_edit.text().strip()
            or str(Path.cwd() / "channel_matrix_history.npz"),
            "NumPy archive (*.npz)",
        )

        if not file_path:

            return

        self.channel_matrix_save_path_edit.setText(file_path)

    def _save_channel_matrix_history(self) -> bool:

        if len(self._channel_matrix_loop_history) <= 0:

            self.channel_matrix_status_label.setText(
                "Channel matrix save skipped: no matrices available."
            )

            return False

        path_text = self.channel_matrix_save_path_edit.text().strip()

        if not path_text:

            self.channel_matrix_status_label.setText(
                "Channel matrix save failed: output path is empty."
            )

            return False

        output_path = Path(path_text)

        try:

            output_path.parent.mkdir(parents=True, exist_ok=True)

            mats_list = [
                np.asarray(m, dtype=np.complex64)
                for m in self._channel_matrix_loop_history
            ]

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

            self.channel_matrix_status_label.setText(
                f"Channel matrix save failed: {exc}"
            )

            return False

    def _on_save_channel_matrix_now(self) -> None:

        if self._channel_matrix is None and len(self._channel_matrix_loop_history) <= 0:

            self.channel_matrix_status_label.setText(
                "Channel matrix save skipped: no current matrix yet."
            )

            return

        if self._channel_matrix is not None:

            if not self._is_current_channel_matrix_complete_square():

                shape_now = np.asarray(self._channel_matrix).shape

                self.channel_matrix_status_label.setText(
                    f"Channel matrix save skipped: current matrix notcomplete square [modes,modes]. Current shape={shape_now}."
                )

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

                self._channel_matrix_loop_history.append(
                    np.asarray(matrix_now, dtype=np.complex64).copy()
                )

                self._channel_matrix_xt_history_db.append(
                    float(self._channel_matrix_xt_db)
                    if self._channel_matrix_xt_db is not None
                    else float("nan")
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

        self._channel_matrix_loop_history.append(
            np.asarray(matrix_now, dtype=np.complex64).copy()
        )

        self._channel_matrix_xt_history_db.append(
            float(self._channel_matrix_xt_db)
            if self._channel_matrix_xt_db is not None
            else float("nan")
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

            ax.text(
                0.5, 0.5, "Waiting for channel matrix data", ha="center", va="center"
            )

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

        xt_text = (
            "n/a"
            if self._channel_matrix_xt_db is None
            else f"{self._channel_matrix_xt_db:.3f} dB"
        )

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

    def _on_browse_log_file(self) -> None:

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Select output CSV file",
            self.logging_path_edit.text().strip()
            or str(Path.cwd() / "phase_measurements.csv"),
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

                    writer.writerow(
                        [
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
                        ]
                    )

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

        recovery_mode = (
            "basic" if self.basic_recovery_radio.isChecked() else "stabilized"
        )

        try:

            with self._log_file_path.open("a", newline="", encoding="utf-8") as handle:

                writer = csv.writer(handle)

                writer.writerow(
                    [
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
                    ]
                )

            self._last_log_time_s = now_s

            self._last_logged_frame_count = frame_count

            self._last_logged_dphi_wrapped = dphi

            self._dphi_unwrapped_accum = dphi_unwrapped

            self._logs_taken_for_active_pattern += 1

            mode_text = "plm" if has_plm_pattern else "free-run"

            self.logging_status_label.setText(
                f"Logging: active ({mode_text}) |seq={self._active_pattern_upload_sequence}, sample{self._logs_taken_for_active_pattern}/{n_max}, frame={frame_count}"
            )

        except Exception as exc:

            self.logging_status_label.setText(f"Logging error: {exc}")

            self._stop_logging()

    def _update_phase_colorbar(self) -> None:

        bar_height = 256

        phase_values = np.linspace(np.pi, -np.pi, bar_height, dtype=np.float32)

        phase_column = phase_values[:, None]

        rgb_column = phase_to_rgb_uint8(
            phase_column, cmap_name=self._phase_colormap_name
        )

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

        self.exposure_spinbox.setEnabled(
            has_time and not self.auto_exposure_checkbox.isChecked()
        )

        if not has_auto and not has_time:

            self._set_status("Exposure controls are not available on this camera.")

        elif not has_auto and has_time:

            self._set_status(
                f"Using manual exposure via {self._exp_time_feature_name}."
            )

        else:

            self._set_status("Exposure control ready.")

    def _initialize_trigger_delay_controls(self) -> None:

        has_trigger_delay = self._trigger_delay_feature_name is not None

        self.trigger_delay_edit.setEnabled(has_trigger_delay)

        if not has_trigger_delay:

            self.trigger_delay_edit.setText("n/a")

            return

        try:

            current_delay_us = float(
                self._get_feature(self._trigger_delay_feature_name).get()
            )

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
                
            self.processor_thread.start()
            self.cam.start_streaming(handler=self._frame_handler, buffer_count=5)
            
            self._streaming = True
            #self.timer.start()
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            
            if trigger_mode == "line1":
                self._set_status("Live view started (hardware trigger: Line1).")

            else:
                self._set_status("Live view started (trigger: FreeRun).")

        except Exception as exc:
            self._set_status(f"Failed to start streaming: {exc}")

    def _stop_live(self) -> None:
        # self.timer.stop()
        if self._streaming:
            try:
                self.cam.stop_streaming()

            except Exception as exc:
                self._set_status(f"Failed to stop streaming cleanly: {exc}")
        
            self.processor_thread.stop()
            self._streaming = False

        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self._set_status("Live view stopped.")

    def _clear_dark_frame(self) -> None:

        self._master_dark_frame = None

        self.dark_frame_status_label.setText("Dark frame: off")

        self._set_status("Dark frame cleared.")

    def _capture_dark_frame_snapshot(self) -> None:

        if not self._streaming:

            self._set_status("Start live view before capturing a dark frame.")

            return

        n_frames = int(self.dark_frame_avg_combo.currentData() or 1)

        if n_frames <= 0:

            self._set_status("Invalid dark-frame averaging count.")

            return

        self.dark_frame_capture_button.setEnabled(False)

        self.dark_frame_clear_button.setEnabled(False)

        self.dark_frame_status_label.setText(f"Dark frame: capturing {n_frames} frame(s)...")

        accum: Optional[np.ndarray] = None

        captured = 0

        last_seen_frame_count = -1

        timeout_s = max(2.0, 0.6 * float(n_frames))

        t0 = perf_counter()

        try:

            while captured < n_frames:

                with self._frame_lock:

                    latest_frame = self._latest_frame

                    frame_count = self._frame_count

                if latest_frame is None:

                    if perf_counter() - t0 > timeout_s:

                        raise RuntimeError("Timed out waiting for camera frames.")

                    QApplication.processEvents()

                    continue

                if frame_count == last_seen_frame_count:

                    if perf_counter() - t0 > timeout_s:

                        raise RuntimeError("Timed out waiting for fresh camera frames.")

                    QApplication.processEvents()

                    continue

                frame_to_add = latest_frame

                if frame_to_add.ndim == 3:

                    frame_to_add = frame_to_add[..., 0]

                frame_float = frame_to_add.astype(np.float32, copy=False)

                if accum is None:

                    accum = np.zeros_like(frame_float, dtype=np.float64)

                elif accum.shape != frame_float.shape:

                    raise RuntimeError(
                        "Frame shape changed while capturing dark frame. Try again."
                    )

                accum += frame_float

                captured += 1

                last_seen_frame_count = frame_count

                self.dark_frame_status_label.setText(
                    f"Dark frame: capturing {captured}/{n_frames}"
                )

                QApplication.processEvents()

            if accum is None:

                raise RuntimeError("No frames captured for dark frame.")

            self._master_dark_frame = (accum / float(n_frames)).astype(np.float32)

            h, w = self._master_dark_frame.shape

            self.dark_frame_status_label.setText(f"Dark frame: ON ({n_frames} avg, {w}x{h})")

            self._set_status(f"Captured dark frame from {n_frames} frame(s).")

        except Exception as exc:

            self.dark_frame_status_label.setText("Dark frame: capture failed")

            self._set_status(f"Dark frame capture failed: {exc}")

        finally:

            self.dark_frame_capture_button.setEnabled(True)

            self.dark_frame_clear_button.setEnabled(True)


    def _frame_handler(self, cam, stream, frame):
        try:
            frame_mono8 = frame.convert_pixel_format(PixelFormat.Mono8)
            image = frame_mono8.as_opencv_image().copy()

            with self._frame_lock:
                self._latest_frame = image
                self._frame_count += 1
                
                # Check if the liquid crystals are still settling
                # Only stamp the frame if we have surpassed the drop count!
                drop_target = self._drop_frames_start_count + getattr(self, '_drop_frames_remaining', 0)
                
                if self._frame_count > drop_target:
                    active_seq = self._active_pattern_upload_sequence
                    active_idx = self._current_plm_pattern_index
                else:
                    active_seq = -1  # Transient frame, ignore for gallery
                    active_idx = -1
                
            # Push the latest state to the worker thread safely
            self.processor_thread.update_state(
                frame=image,
                count=self._frame_count,
                dark_frame=self._master_dark_frame,
                fft_roi=self._fft_roi,
                live_roi=self._live_roi,
                basic_recovery=self.basic_recovery_radio.isChecked(),
                colormap_name=self._phase_colormap_name,
                pattern_seq=active_seq, # Stamped ONLY if stable
                pattern_idx=active_idx  # Stamped ONLY if stable
            )
        finally:
            cam.queue_frame(frame)


    def _store_pattern_field_direct(self, recovered_field: np.ndarray, pattern_idx: int) -> None:
        self._pattern_fields_by_pattern[int(pattern_idx)] = np.asarray(
            recovered_field, dtype=np.complex64
        ).copy()

    def _store_recovered_mode_direct(self, recovered_field: np.ndarray, pattern_idx: int) -> None:
        mode = self._prepare_mode_for_target_shape(recovered_field)
        self._recovered_modes_by_pattern[int(pattern_idx)] = mode
        self._recompute_channel_matrix()
        self._channel_matrix_dirty = True

    def _capture_pattern_preview_direct(self, phase_rgb: np.ndarray, pattern_idx: int) -> None:
        if pattern_idx not in self._pattern_preview_labels:
            return
        
        height, width = int(phase_rgb.shape[0]), int(phase_rgb.shape[1])
        contiguous = np.ascontiguousarray(phase_rgb)
        image = QImage(contiguous.data, width, height, width * 3, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(image)
        
        label = self._pattern_preview_labels[pattern_idx]
        scaled = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.FastTransformation)
        
        self._pattern_preview_pixmaps[pattern_idx] = scaled
        label.setPixmap(scaled)
        label.setText("")
        
        self.pattern_gallery_status_label.setText(
            f"Updated gallery for pattern {pattern_idx}/{len(self._pattern_preview_labels)}"
        )


    def _on_auto_exposure_toggled(self, checked: bool) -> None:

        self.exposure_spinbox.setEnabled(
            self._exp_time_feature_name is not None and not checked
        )

        if self._exp_auto_feature_name is None:

            if checked:

                self._set_status(
                    "Auto exposure feature is not available on this camera."
                )

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

            self.fft_status_label.setText(
                "ROI cleared. Draw rectangle over first order in FFT."
            )

            self.recovered_field_status_label.setText(
                "Recovered field not available. Select FFT ROI."
            )

            return

        x0, y0, x1, y1 = roi

        self.fft_status_label.setText(f"FFT ROI selected: x={x0}:{x1}, y={y0}:{y1}.")

        self.recovered_field_status_label.setText(
            "Recovering field from selected first order..."
        )

    def _on_live_roi_changed(self, roi: Optional[tuple[int, int, int, int]]) -> None:

        self._live_roi = roi

        if roi is None:

            self.live_roi_status_label.setText("Live ROI: not set")

            self.live_roi_intensity_label.setText("Live ROI sum intensity: n/a")

            return

        x0, y0, x1, y1 = roi

        self.live_roi_status_label.setText(
            f"Live ROI selected: x={x0}:{x1}, y={y0}:{y1}"
        )

    def _on_gaussian_mfd_toggled(self, checked: bool) -> None:

        self._gaussian_mfd_enabled = bool(checked)

        if not self._gaussian_mfd_enabled:

            self.gaussian_mfd_label.setText("Gaussian MFD (pixel pitch 15 um): disabled")

            return

        self.gaussian_mfd_label.setText("Gaussian MFD (pixel pitch 15 um): waiting for frames...")

    def _on_recovered_points_changed(
        self,
        point1: Optional[tuple[int, int]],
        point2: Optional[tuple[int, int]],
    ) -> None:

        self._update_recovered_point_readout(point1, point2)

    def _fit_gaussian_profile_mfd_um(self, profile: np.ndarray) -> Optional[float]:

        values = np.asarray(profile, dtype=np.float64).ravel()

        if values.size < 7:

            return None

        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

        background = float(np.percentile(values, 10.0))

        signal = values - background

        signal = np.clip(signal, 0.0, None)

        peak_signal = float(np.max(signal))

        if peak_signal <= 0.0:

            return None

        peak_index = int(np.argmax(signal))

        x = np.arange(signal.size, dtype=np.float64) - float(peak_index)

        mask = signal >= (0.1 * peak_signal)

        if int(np.count_nonzero(mask)) < 7:

            mask = signal > 0.0

        if int(np.count_nonzero(mask)) < 7:

            return None

        y_log = np.log(np.clip(signal[mask], 1e-12, None))

        x_fit = x[mask]

        try:

            a, _b, _c = np.polyfit(x_fit, y_log, 2)

        except Exception:

            return None

        if not np.isfinite(a) or a >= -1e-12:

            return None

        waist_px = math.sqrt(-2.0 / float(a))

        mfd_um = 2.0 * waist_px * float(self._gaussian_pixel_pitch_um)

        if not np.isfinite(mfd_um) or mfd_um <= 0.0:

            return None

        max_reasonable_mfd_um = 2.0 * float(signal.size) * float(self._gaussian_pixel_pitch_um)

        if mfd_um > max_reasonable_mfd_um:

            return None

        return float(mfd_um)

    def _measure_gaussian_mfd(self, image: np.ndarray) -> tuple[Optional[float], Optional[float], Optional[float], str]:

        if image.ndim != 2:

            return None, None, None, "invalid"

        roi = self._live_roi

        if roi is not None:

            x0, y0, x1, y1 = roi

            patch = np.asarray(image[y0:y1, x0:x1], dtype=np.float64)

            source_name = "ROI"

        else:

            patch = np.asarray(image, dtype=np.float64)

            source_name = "full frame"

        if patch.size == 0 or patch.shape[0] < 7 or patch.shape[1] < 7:

            return None, None, None, source_name

        peak_y, peak_x = np.unravel_index(np.argmax(patch), patch.shape)

        profile_x = patch[peak_y, :]

        profile_y = patch[:, peak_x]

        mfd_x_um = self._fit_gaussian_profile_mfd_um(profile_x)

        mfd_y_um = self._fit_gaussian_profile_mfd_um(profile_y)

        if mfd_x_um is None or mfd_y_um is None:

            return mfd_x_um, mfd_y_um, None, source_name

        mfd_eq_um = math.sqrt(max(0.0, mfd_x_um * mfd_y_um))

        return mfd_x_um, mfd_y_um, mfd_eq_um, source_name

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

    def _append_live_phase_sample(
        self, phi1: float, phi2: float, dphi_0_2pi: float
    ) -> None:

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

    def _on_phase_calibration_center_window_changed(self, value: int) -> None:

        adjusted = max(1, int(value))

        if adjusted % 2 == 0:

            adjusted += 1

        if adjusted != int(self.phase_cal_center_window_spin.value()):

            self.phase_cal_center_window_spin.blockSignals(True)

            self.phase_cal_center_window_spin.setValue(adjusted)

            self.phase_cal_center_window_spin.blockSignals(False)

    def _coerce_to_multiple(
        self, value: int, base: int, minimum: int, maximum: int
    ) -> int:

        clamped = max(minimum, min(maximum, int(value)))

        coerced = int(math.ceil(clamped / float(base)) * base)

        if coerced > maximum:

            coerced = int((maximum // base) * base)

        return max(minimum, coerced)

    def _update_phase_calibration_grid_from_catalog(self) -> bool:

        plm_catalog = self.plm_catalog_edit.text().strip()

        if not plm_catalog:

            return False

        try:

            first_init = (
                self._phase_cal_grid_rows <= 0 or self._phase_cal_grid_cols <= 0
            )

            plm = PLM.from_db(plm_catalog)

            height, width = int(plm.shape[0]), int(plm.shape[1])

            spx = self._coerce_to_multiple(
                int(self.phase_cal_spx_spin.value()), base=6, minimum=6, maximum=510
            )

            spy = self._coerce_to_multiple(
                int(self.phase_cal_spy_spin.value()), base=9, minimum=9, maximum=504
            )

            self._phase_cal_grid_cols = int(math.ceil(width / float(spx)))

            self._phase_cal_grid_rows = int(math.ceil(height / float(spy)))

            self._phase_cal_spx = spx

            self._phase_cal_spy = spy

            if first_init:

                self.phase_cal_sec_row_start_spin.setValue(0)

                self.phase_cal_sec_col_start_spin.setValue(0)

                self.phase_cal_sec_row_end_spin.setValue(
                    max(0, self._phase_cal_grid_rows - 1)
                )

                self.phase_cal_sec_col_end_spin.setValue(
                    max(0, self._phase_cal_grid_cols - 1)
                )

            return True

        except Exception:

            return False

    def _on_phase_calibration_superpixel_changed(self, *_args) -> None:

        x_new = self._coerce_to_multiple(
            int(self.phase_cal_spx_spin.value()), base=6, minimum=6, maximum=510
        )

        y_new = self._coerce_to_multiple(
            int(self.phase_cal_spy_spin.value()), base=9, minimum=9, maximum=504
        )

        if x_new != int(self.phase_cal_spx_spin.value()):

            self.phase_cal_spx_spin.blockSignals(True)

            self.phase_cal_spx_spin.setValue(x_new)

            self.phase_cal_spx_spin.blockSignals(False)

        if y_new != int(self.phase_cal_spy_spin.value()):

            self.phase_cal_spy_spin.blockSignals(True)

            self.phase_cal_spy_spin.setValue(y_new)

            self.phase_cal_spy_spin.blockSignals(False)

        if self._update_phase_calibration_grid_from_catalog():

            self._on_phase_calibration_section_changed()

    def _on_phase_calibration_section_changed(self, *_args) -> None:

        if self._phase_cal_grid_rows <= 0 or self._phase_cal_grid_cols <= 0:

            if not self._update_phase_calibration_grid_from_catalog():

                return

        max_row = self._phase_cal_grid_rows - 1

        max_col = self._phase_cal_grid_cols - 1

        controls = [
            self.phase_cal_sec_row_start_spin,
            self.phase_cal_sec_row_end_spin,
            self.phase_cal_sec_col_start_spin,
            self.phase_cal_sec_col_end_spin,
        ]

        for control in controls:

            control.blockSignals(True)

        self.phase_cal_sec_row_start_spin.setMaximum(max_row)

        self.phase_cal_sec_row_end_spin.setMaximum(max_row)

        self.phase_cal_sec_col_start_spin.setMaximum(max_col)

        self.phase_cal_sec_col_end_spin.setMaximum(max_col)

        row_start = max(0, min(max_row, int(self.phase_cal_sec_row_start_spin.value())))

        row_end = max(0, min(max_row, int(self.phase_cal_sec_row_end_spin.value())))

        col_start = max(0, min(max_col, int(self.phase_cal_sec_col_start_spin.value())))

        col_end = max(0, min(max_col, int(self.phase_cal_sec_col_end_spin.value())))

        if row_start > row_end:

            row_start, row_end = row_end, row_start

        if col_start > col_end:

            col_start, col_end = col_end, col_start

        self.phase_cal_sec_row_start_spin.setValue(row_start)

        self.phase_cal_sec_row_end_spin.setValue(row_end)

        self.phase_cal_sec_col_start_spin.setValue(col_start)

        self.phase_cal_sec_col_end_spin.setValue(col_end)

        for control in controls:

            control.blockSignals(False)

        self._on_phase_calibration_ref_changed()

    def _on_phase_calibration_ref_changed(self) -> None:

        if self._phase_cal_grid_rows <= 0 or self._phase_cal_grid_cols <= 0:

            return

        max_row = self._phase_cal_grid_rows - 1

        max_col = self._phase_cal_grid_cols - 1

        row_start = max(0, min(max_row, int(self.phase_cal_sec_row_start_spin.value())))

        row_end = max(0, min(max_row, int(self.phase_cal_sec_row_end_spin.value())))

        col_start = max(0, min(max_col, int(self.phase_cal_sec_col_start_spin.value())))

        col_end = max(0, min(max_col, int(self.phase_cal_sec_col_end_spin.value())))

        if row_start > row_end:

            row_start, row_end = row_end, row_start

        if col_start > col_end:

            col_start, col_end = col_end, col_start

        self._phase_cal_section_row_start = int(row_start)

        self._phase_cal_section_row_end = int(row_end)

        self._phase_cal_section_col_start = int(col_start)

        self._phase_cal_section_col_end = int(col_end)

        self.phase_cal_sec_row_start_spin.blockSignals(True)

        self.phase_cal_sec_row_end_spin.blockSignals(True)

        self.phase_cal_sec_col_start_spin.blockSignals(True)

        self.phase_cal_sec_col_end_spin.blockSignals(True)

        self.phase_cal_sec_row_start_spin.setValue(row_start)

        self.phase_cal_sec_row_end_spin.setValue(row_end)

        self.phase_cal_sec_col_start_spin.setValue(col_start)

        self.phase_cal_sec_col_end_spin.setValue(col_end)

        self.phase_cal_sec_row_start_spin.blockSignals(False)

        self.phase_cal_sec_row_end_spin.blockSignals(False)

        self.phase_cal_sec_col_start_spin.blockSignals(False)

        self.phase_cal_sec_col_end_spin.blockSignals(False)

        center_row = int((row_start + row_end) // 2)

        center_col = int((col_start + col_end) // 2)

        self.phase_cal_ref_row_spin.blockSignals(True)

        self.phase_cal_ref_col_spin.blockSignals(True)

        self.phase_cal_ref_row_spin.setMaximum(self._phase_cal_grid_rows - 1)

        self.phase_cal_ref_col_spin.setMaximum(self._phase_cal_grid_cols - 1)

        self.phase_cal_ref_row_spin.setValue(center_row)

        self.phase_cal_ref_col_spin.setValue(center_col)

        self.phase_cal_ref_row_spin.blockSignals(False)

        self.phase_cal_ref_col_spin.blockSignals(False)

    def _generate_phase_calibration_pattern_stack(self) -> bool:
        try:

            plm_catalog = self.plm_catalog_edit.text().strip()

            if not plm_catalog:

                self.phase_cal_status_label.setText(
                    "PSI status: enter a PLM catalog first"
                )

                return False

            plm = PLM.from_db(plm_catalog)

            height, width = int(plm.shape[0]), int(plm.shape[1])

            spx = self._coerce_to_multiple(
                int(self.phase_cal_spx_spin.value()), base=6, minimum=6, maximum=510
            )

            spy = self._coerce_to_multiple(
                int(self.phase_cal_spy_spin.value()), base=9, minimum=9, maximum=504
            )

            if spx != int(self.phase_cal_spx_spin.value()):

                self.phase_cal_spx_spin.setValue(spx)

            if spy != int(self.phase_cal_spy_spin.value()):

                self.phase_cal_spy_spin.setValue(spy)

            grid_cols = int(math.ceil(width / float(spx)))

            grid_rows = int(math.ceil(height / float(spy)))

            self._phase_cal_grid_rows = grid_rows

            self._phase_cal_grid_cols = grid_cols

            self._phase_cal_spx = spx

            self._phase_cal_spy = spy

            self._on_phase_calibration_section_changed()

            ref_row = int(self.phase_cal_ref_row_spin.value())

            ref_col = int(self.phase_cal_ref_col_spin.value())

            ref_flat = ref_row * grid_cols + ref_col

            if ref_flat < 0 or ref_flat >= grid_rows * grid_cols:

                self.phase_cal_status_label.setText(
                    "PSI status: invalid reference super-pixel"
                )

                return False

            self._phase_cal_ref_flat = int(ref_flat)

            self._phase_cal_grating_period_x = int(
                self.phase_cal_grating_period_spin.value()
            )

            phase_steps = [0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi]

            row_start = int(self._phase_cal_section_row_start)

            row_end = int(self._phase_cal_section_row_end)

            col_start = int(self._phase_cal_section_col_start)

            col_end = int(self._phase_cal_section_col_end)

            section_indices = [
                row * grid_cols + col
                for row in range(row_start, row_end + 1)
                for col in range(col_start, col_end + 1)
            ]

            signal_indices = [idx for idx in section_indices if idx != ref_flat]

            if len(signal_indices) == 0:

                self.phase_cal_status_label.setText(
                    "PSI status: no signal pixels to sweep. Reduce super-pixel size."
                )

                return False

            pattern_meta: list[tuple[int, int]] = []

            pattern_index_by_signal_step: dict[tuple[int, int], int] = {}

            for signal_flat in signal_indices:
                for step_idx, _phase_step in enumerate(phase_steps):
                    pattern_idx = int(len(pattern_meta))
                    pattern_meta.append((int(signal_flat), int(step_idx)))
                    pattern_index_by_signal_step[(int(signal_flat), int(step_idx))] = (
                        pattern_idx
                    )

            self._phase_cal_generated_patterns = None

            self._phase_cal_pattern_meta = pattern_meta

            self._phase_cal_pattern_index_by_signal_step = pattern_index_by_signal_step

            self._phase_cal_signal_indices = signal_indices

            self._phase_cal_total_patterns = int(len(pattern_meta))

            total_patterns = int(self._phase_cal_total_patterns)

            self.phase_cal_info_label.setText(
                f"PSI sweep ready (dynamic): {total_patterns} frames ({len(signal_indices)} positions x 4 steps), section rows {row_start}:{row_end}, cols {col_start}:{col_end}, ref=({ref_row},{ref_col}), grid {grid_cols}x{grid_rows}."
            )

            self.phase_cal_progress_label.setText(
                "PSI progress: sweep metadata generated"
            )

            self.phase_cal_status_label.setText("PSI status: ready to start")

            return True

        except Exception as exc:

            self._phase_cal_generated_patterns = None

            self._phase_cal_total_patterns = 0

            self._phase_cal_pattern_meta = []

            self._phase_cal_pattern_index_by_signal_step = {}

            self._phase_cal_signal_indices = []

            self.phase_cal_status_label.setText(
                f"PSI status: failed to generate stack: {exc}"
            )

            return False

    def _on_generate_phase_calibration_patterns(self) -> None:

        self._generate_phase_calibration_pattern_stack()

    def _start_phase_calibration(self) -> None:
        if self._phase_cal_visibility_running:
            self.phase_cal_visibility_button.setChecked(False)

        if not self._streaming:
            self.phase_cal_status_label.setText(
                "PSI status: start camera live view before calibration"
            )

            return

        if self._plm_loop_thread is not None and self._plm_loop_thread.is_alive():
            self.phase_cal_status_label.setText(
                "PSI status: stop current PLM loop before starting calibration"
            )

            return

        if not self._generate_phase_calibration_pattern_stack():
            return

        if (
            self._phase_cal_total_patterns <= 0
            or len(self._phase_cal_pattern_meta) <= 0
        ):

            self.phase_cal_status_label.setText(
                "PSI status: calibration sweep metadata is empty"
            )

            return

        self._phase_cal_previous_patterns = self._plm_phase_patterns
        self._phase_cal_previous_pattern_path = self._plm_pattern_path

        self._phase_cal_prev_drop_combo_index = int(
            self.drop_frames_combo.currentIndex()
        )

        self.drop_frames_combo.setCurrentIndex(0)
        self._phase_cal_restore_pending = True
        self._plm_phase_patterns = None
        self._plm_pattern_path = "<phase_calibration_internal>"
        n_patterns = int(self._phase_cal_total_patterns)
        self._phase_cal_step_sum = np.zeros(n_patterns, dtype=np.float64)
        self._phase_cal_step_count = np.zeros(n_patterns, dtype=np.int32)
        self._phase_cal_step_mean = np.full(n_patterns, np.nan, dtype=np.float64)
        self._phase_cal_last_pattern_seq = -1
        self._phase_cal_drop_remaining_for_seq = 0
        map_rows = max(0, int(self._phase_cal_grid_rows))
        map_cols = max(0, int(self._phase_cal_grid_cols))

        loaded_map = self._phase_cal_superpixel_map
        loaded_shape_ok = (
            loaded_map is not None
            and tuple(np.asarray(loaded_map).shape) == (map_rows, map_cols)
        )
        preserve_loaded_superpixel_map = bool(
            self._phase_cal_loaded_map_path
            and self.phase_cal_apply_checkbox.isChecked()
            and loaded_shape_ok
        )

        if preserve_loaded_superpixel_map:
            self._phase_cal_superpixel_map = wrap_phase_array_to_pi(
                np.asarray(loaded_map, dtype=np.float32)
            ).copy()

            if (
                self._phase_cal_modulation_superpixel is None
                or tuple(np.asarray(self._phase_cal_modulation_superpixel).shape)
                != (map_rows, map_cols)
            ):
                self._phase_cal_modulation_superpixel = np.full(
                    (map_rows, map_cols), np.nan, dtype=np.float32
                )
            else:
                self._phase_cal_modulation_superpixel = np.asarray(
                    self._phase_cal_modulation_superpixel, dtype=np.float32
                ).copy()

        else:
            # Start with a fresh recovered map unless explicitly preserving a loaded one.
            self._phase_cal_loaded_map_path = None
            self._update_loaded_phase_map_label()

            self._phase_cal_superpixel_map = np.full(
                (map_rows, map_cols), np.nan, dtype=np.float32
            )

            self._phase_cal_modulation_superpixel = np.full(
                (map_rows, map_cols), np.nan, dtype=np.float32
            )

            if map_rows > 0 and map_cols > 0:
                ref_r, ref_c = divmod(int(self._phase_cal_ref_flat), map_cols)
                if 0 <= ref_r < map_rows and 0 <= ref_c < map_cols:
                    self._phase_cal_superpixel_map[ref_r, ref_c] = 0.0

        plm_catalog = self.plm_catalog_edit.text().strip()
        plm = PLM.from_db(plm_catalog)
        height, width = int(plm.shape[0]), int(plm.shape[1])
        self._phase_cal_fullres_shape = (height, width)
        self._phase_cal_fullres_map = self._expand_superpixel_map_to_fullres(
            self._phase_cal_superpixel_map, width=width, height=height
        )
        self._phase_cal_last_live_plot_t = 0.0
        self._refresh_phase_calibration_map_plot()
        self._phase_cal_running = True
        self._phase_cal_completed_this_run = False
        self.phase_cal_start_button.setEnabled(False)
        self.phase_cal_stop_button.setEnabled(True)
        self.phase_cal_progress_label.setText(
            f"PSI progress: 0/{n_patterns} steps measured"
        )
        self.phase_cal_status_label.setText("PSI status: starting PLM sweep")
        self._start_plm_pattern_loop()
        if self._plm_loop_thread is None or not self._plm_loop_thread.is_alive():
            self._phase_cal_running = False
            self._phase_cal_completed_this_run = False
            self.phase_cal_start_button.setEnabled(True)
            self.phase_cal_stop_button.setEnabled(False)
            self._restore_patterns_after_phase_calibration()
            self.phase_cal_status_label.setText("PSI status: failed to start PLM sweep")


    def _get_phase_calibration_section_bounds_pixels(
        self,
        shape: tuple[int, int],
    ) -> tuple[int, int, int, int, int, int, int, int]:

        height, width = int(shape[0]), int(shape[1])

        if self._phase_cal_grid_rows <= 0 or self._phase_cal_grid_cols <= 0:
            raise RuntimeError("PSI grid is not initialized")

        max_row = self._phase_cal_grid_rows - 1
        max_col = self._phase_cal_grid_cols - 1
        row_start = max(0, min(max_row, int(self.phase_cal_sec_row_start_spin.value())))
        row_end = max(0, min(max_row, int(self.phase_cal_sec_row_end_spin.value())))
        col_start = max(0, min(max_col, int(self.phase_cal_sec_col_start_spin.value())))
        col_end = max(0, min(max_col, int(self.phase_cal_sec_col_end_spin.value())))

        if row_start > row_end:
            row_start, row_end = row_end, row_start

        if col_start > col_end:
            col_start, col_end = col_end, col_start

        y0 = max(0, min(height, row_start * int(self._phase_cal_spy)))
        y1 = max(y0, min(height, (row_end + 1) * int(self._phase_cal_spy)))
        x0 = max(0, min(width, col_start * int(self._phase_cal_spx)))
        x1 = max(x0, min(width, (col_end + 1) * int(self._phase_cal_spx)))

        return row_start, row_end, col_start, col_end, x0, x1, y0, y1

    def _build_phase_calibration_visibility_pattern(
        self, shape: tuple[int, int]
    ) -> np.ndarray:

        height, width = int(shape[0]), int(shape[1])

        _, _, _, _, x0, x1, y0, y1 = self._get_phase_calibration_section_bounds_pixels(
            (height, width)
        )

        pattern = np.zeros((height, width), dtype=np.float32)

        period_x = max(1, int(self.phase_cal_grating_period_spin.value()))

        self._phase_cal_grating_period_x = period_x

        x = np.arange(width, dtype=np.float32)

        ramp_1d = np.mod(2.0 * np.pi * x / float(period_x), 2.0 * np.pi)

        section_h = max(0, y1 - y0)

        section_w = max(0, x1 - x0)

        if section_h <= 0 or section_w <= 0:

            raise RuntimeError("Selected section is empty")

        border_h = max(1, min(int(self._phase_cal_spy), section_h))

        border_w = max(1, min(int(self._phase_cal_spx), section_w))

        section_ramp = np.tile(ramp_1d[x0:x1][None, :], (section_h, 1))

        # Top border
        pattern[y0 : y0 + border_h, x0:x1] = section_ramp[:border_h, :]

        # Bottom border
        pattern[y1 - border_h : y1, x0:x1] = section_ramp[section_h - border_h :, :]

        # Left border
        pattern[y0:y1, x0 : x0 + border_w] = section_ramp[:, :border_w]

        # Right border
        pattern[y0:y1, x1 - border_w : x1] = section_ramp[:, section_w - border_w :]

        return pattern

    def _restore_patterns_after_phase_calibration_visibility(self) -> None:

        if not self._phase_cal_visibility_restore_pending:

            return

        self._plm_phase_patterns = self._phase_cal_visibility_previous_patterns

        self._plm_pattern_path = self._phase_cal_visibility_previous_pattern_path

        self._phase_cal_visibility_previous_patterns = None

        self._phase_cal_visibility_previous_pattern_path = None

        self._phase_cal_visibility_restore_pending = False

    def _toggle_phase_calibration_visibility(self, checked: bool) -> None:

        if checked:

            if self._phase_cal_running:

                self.phase_cal_status_label.setText(
                    "PSI status: stop calibration before visibility test"
                )

                self.phase_cal_visibility_button.blockSignals(True)

                self.phase_cal_visibility_button.setChecked(False)

                self.phase_cal_visibility_button.blockSignals(False)

                return

            if self._plm_loop_thread is not None and self._plm_loop_thread.is_alive():

                self.phase_cal_status_label.setText(
                    "PSI status: stop PLM loop before visibility test"
                )

                self.phase_cal_visibility_button.blockSignals(True)

                self.phase_cal_visibility_button.setChecked(False)

                self.phase_cal_visibility_button.blockSignals(False)

                return

            if not self._update_phase_calibration_grid_from_catalog():

                self.phase_cal_status_label.setText(
                    "PSI status: enter valid PLM catalog before visibility test"
                )

                self.phase_cal_visibility_button.blockSignals(True)

                self.phase_cal_visibility_button.setChecked(False)

                self.phase_cal_visibility_button.blockSignals(False)

                return

            self._on_phase_calibration_section_changed()

            try:

                plm_catalog = self.plm_catalog_edit.text().strip()

                plm = PLM.from_db(plm_catalog)

                pattern = self._build_phase_calibration_visibility_pattern(
                    (int(plm.shape[0]), int(plm.shape[1]))
                )

                row_start, row_end, col_start, col_end, x0, x1, y0, y1 = (
                    self._get_phase_calibration_section_bounds_pixels(
                        (int(plm.shape[0]), int(plm.shape[1]))
                    )
                )

            except Exception as exc:

                self.phase_cal_status_label.setText(
                    f"PSI status: visibility pattern failed: {exc}"
                )

                self.phase_cal_visibility_button.blockSignals(True)

                self.phase_cal_visibility_button.setChecked(False)

                self.phase_cal_visibility_button.blockSignals(False)

                return

            self._phase_cal_visibility_previous_patterns = self._plm_phase_patterns

            self._phase_cal_visibility_previous_pattern_path = self._plm_pattern_path

            self._phase_cal_visibility_restore_pending = True

            self._plm_phase_patterns = pattern[None, ...].astype(np.float32, copy=False)

            self._plm_pattern_path = "<phase_cal_visibility_internal>"

            self._phase_cal_visibility_running = True

            self.phase_cal_visibility_button.setText("Hide Section Visibility")

            self.phase_cal_status_label.setText(
                f"PSI status: visibility ON rows {row_start}:{row_end}, cols {col_start}:{col_end}, pixels x={x0}:{x1}, y={y0}:{y1}"
            )

            self._start_plm_pattern_loop()

            if self._plm_loop_thread is None or not self._plm_loop_thread.is_alive():

                self._phase_cal_visibility_running = False

                self.phase_cal_visibility_button.setText("Show Section Visibility")

                self._restore_patterns_after_phase_calibration_visibility()

                self.phase_cal_visibility_button.blockSignals(True)

                self.phase_cal_visibility_button.setChecked(False)

                self.phase_cal_visibility_button.blockSignals(False)

                self.phase_cal_status_label.setText(
                    "PSI status: failed to start visibility test"
                )

            return

        if self._phase_cal_visibility_running:

            self._phase_cal_visibility_running = False

            self.phase_cal_visibility_button.setText("Show Section Visibility")

            if self._plm_loop_thread is not None and self._plm_loop_thread.is_alive():

                self._stop_plm_pattern_loop()

                self.phase_cal_status_label.setText(
                    "PSI status: stopping section visibility"
                )

            else:

                self._restore_patterns_after_phase_calibration_visibility()

                self.phase_cal_status_label.setText(
                    "PSI status: section visibility stopped"
                )

    def _stop_phase_calibration(self) -> None:

        was_running = bool(self._phase_cal_running)

        self._phase_cal_running = False

        self._phase_cal_completed_this_run = False

        self.phase_cal_start_button.setEnabled(True)

        self.phase_cal_stop_button.setEnabled(False)

        if self._plm_loop_thread is not None and self._plm_loop_thread.is_alive():

            self._stop_plm_pattern_loop()

            self.phase_cal_status_label.setText(
                "PSI status: stopping calibration sweep"
            )

        else:

            self._restore_patterns_after_phase_calibration()

            self.phase_cal_status_label.setText("PSI status: calibration stopped")

        if not was_running and not self._phase_cal_restore_pending:

            self.phase_cal_status_label.setText(
                "PSI status: calibration is not running"
            )

    def _restore_patterns_after_phase_calibration(self) -> None:

        if not self._phase_cal_restore_pending:

            return

        self._plm_phase_patterns = self._phase_cal_previous_patterns

        self._plm_pattern_path = self._phase_cal_previous_pattern_path

        self._phase_cal_previous_patterns = None

        self._phase_cal_previous_pattern_path = None

        if self._phase_cal_prev_drop_combo_index is not None:

            self.drop_frames_combo.setCurrentIndex(
                int(self._phase_cal_prev_drop_combo_index)
            )

        self._phase_cal_prev_drop_combo_index = None

        self._phase_cal_restore_pending = False

    def _get_phase_calibration_roi_intensity(
        self, image: np.ndarray
    ) -> Optional[float]:

        if (
            self.phase_cal_use_live_roi_checkbox.isChecked()
            and self._live_roi is not None
        ):

            x0, y0, x1, y1 = self._live_roi

            if x1 > x0 and y1 > y0:

                cx = int((x0 + x1 - 1) // 2)

                cy = int((y0 + y1 - 1) // 2)

                height, width = image.shape[:2]

                cx = max(0, min(width - 1, cx))

                cy = max(0, min(height - 1, cy))

                return float(image[cy, cx])

        window = max(1, int(self.phase_cal_center_window_spin.value()))

        if window % 2 == 0:

            window += 1

        height, width = image.shape[:2]

        cx = width // 2

        cy = height // 2

        half = window // 2

        x0 = max(0, cx - half)

        x1 = min(width, cx + half + 1)

        y0 = max(0, cy - half)

        y1 = min(height, cy + half + 1)

        patch = image[y0:y1, x0:x1]

        if patch.size == 0:

            return None

        return float(np.sum(patch, dtype=np.float64))

    def _handle_phase_calibration_frame(
        self, image: np.ndarray, _frame_count: int
    ) -> None:

        if not self._phase_cal_running:

            return

        if (
            self._phase_cal_step_mean is None
            or self._phase_cal_step_sum is None
            or self._phase_cal_step_count is None
        ):

            return

        pattern_idx_1_based = self._current_plm_pattern_index

        total_patterns = self._current_plm_total_patterns

        if pattern_idx_1_based is None or total_patterns is None:

            return

        pattern_idx = int(pattern_idx_1_based - 1)

        if pattern_idx < 0 or pattern_idx >= int(self._phase_cal_step_mean.size):

            return

        if self._pattern_upload_sequence != self._phase_cal_last_pattern_seq:

            self._phase_cal_last_pattern_seq = int(self._pattern_upload_sequence)

            self._phase_cal_drop_remaining_for_seq = max(
                0, int(self.phase_cal_drop_frames_spin.value())
            )

        if self._phase_cal_drop_remaining_for_seq > 0:

            self._phase_cal_drop_remaining_for_seq -= 1

            return

        if np.isfinite(self._phase_cal_step_mean[pattern_idx]):

            return

        intensity = self._get_phase_calibration_roi_intensity(image)

        if intensity is None:

            return

        self._phase_cal_step_sum[pattern_idx] += float(intensity)

        self._phase_cal_step_count[pattern_idx] += 1

        avg_target = max(1, int(self.phase_cal_avg_frames_spin.value()))

        if int(self._phase_cal_step_count[pattern_idx]) >= avg_target:

            self._phase_cal_step_mean[pattern_idx] = self._phase_cal_step_sum[
                pattern_idx
            ] / float(self._phase_cal_step_count[pattern_idx])

            signal_flat, _step_idx = self._phase_cal_pattern_meta[pattern_idx]

            self._update_phase_calibration_partial_map_for_signal(int(signal_flat))

            measured = int(np.sum(np.isfinite(self._phase_cal_step_mean)))

            total = int(self._phase_cal_step_mean.size)

            self.phase_cal_progress_label.setText(
                f"PSI progress: {measured}/{total} steps measured"
            )

            now = perf_counter()

            if (
                now - float(self._phase_cal_last_live_plot_t)
            ) >= 0.2 or measured >= total:

                self._phase_cal_last_live_plot_t = now

                self._refresh_phase_calibration_map_plot()

            if measured >= total:

                self._finalize_phase_calibration_from_measurements()

    def _expand_superpixel_map_to_fullres(
        self, superpixel_map: np.ndarray, width: int, height: int
    ) -> np.ndarray:

        expanded = np.repeat(
            np.repeat(superpixel_map, self._phase_cal_spy, axis=0),
            self._phase_cal_spx,
            axis=1,
        )

        return np.asarray(expanded[:height, :width], dtype=np.float32)

    def _resample_phase_map_nearest(
        self, phase_map: np.ndarray, target_shape: tuple[int, int]
    ) -> np.ndarray:

        target_h, target_w = int(target_shape[0]), int(target_shape[1])

        src_h, src_w = phase_map.shape[:2]

        if src_h <= 0 or src_w <= 0 or target_h <= 0 or target_w <= 0:

            return np.zeros((target_h, target_w), dtype=np.float32)

        y_idx = np.minimum(
            src_h - 1, np.floor(np.linspace(0.0, src_h - 1, target_h)).astype(np.int32)
        )

        x_idx = np.minimum(
            src_w - 1, np.floor(np.linspace(0.0, src_w - 1, target_w)).astype(np.int32)
        )

        return np.asarray(phase_map[y_idx][:, x_idx], dtype=np.float32)

    def _refresh_phase_calibration_map_plot(self) -> None:

        fig = self.phase_cal_map_canvas.figure

        fig.clear()

        ax = fig.add_subplot(111)

        if self._phase_cal_superpixel_map is None:

            ax.set_title("Recovered PLM Phase Map")

            ax.text(0.5, 0.5, "No calibration map available", ha="center", va="center")

            ax.set_axis_off()

            self.phase_cal_map_shape_label.setText("Map grid: n/a")

            self.phase_cal_map_canvas.draw_idle()

            return

        phase_grid = np.asarray(self._phase_cal_superpixel_map, dtype=np.float32)

        valid_mask = np.isfinite(phase_grid)

        cmap = mpl.colormaps[self._phase_colormap_name].copy()

        cmap.set_bad(color="#ececec")

        phase_display = np.ma.masked_invalid(phase_grid)

        im = ax.imshow(phase_display, cmap=cmap, vmin=-np.pi, vmax=np.pi, aspect="auto")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        cbar.set_label("Recovered phase (rad)")

        total_count = int(phase_grid.size)

        valid_count = int(np.sum(valid_mask))

        if np.any(valid_mask):

            ax.set_title(f"Recovered PLM Phase Map | valid {valid_count}/{total_count}")

            try:

                min_val = float(np.nanmin(phase_display))

                max_val = float(np.nanmax(phase_display))

                self.phase_cal_map_stats_label.setText(
                    f"Map: valid {valid_count}/{total_count} super-pixels | phase range [{min_val:.3f}, {max_val:.3f}] rad"
                )

            except Exception:

                self.phase_cal_map_stats_label.setText(
                    f"Map: valid {valid_count}/{total_count} super-pixels"
                )

        else:

            ax.set_title(f"Recovered PLM Phase Map | valid 0/{total_count}")

            self.phase_cal_map_stats_label.setText(
                f"Map: valid 0/{total_count} super-pixels"
            )

        ax.set_xlabel("Super-pixel col index")

        ax.set_ylabel("Super-pixel row index")

        self.phase_cal_map_shape_label.setText(
            f"Map grid: {phase_grid.shape[1]} cols x {phase_grid.shape[0]} rows | PLM 904 x 800 / super-pixel {self._phase_cal_spx} x {self._phase_cal_spy}"
        )

        fig.tight_layout()

        self.phase_cal_map_canvas.draw_idle()

    def _refresh_phase_calibration_live_pattern_plot(
        self, phase_pattern: Optional[np.ndarray]
    ) -> None:

        fig = self.phase_cal_live_pattern_canvas.figure

        fig.clear()

        ax = fig.add_subplot(111)

        if phase_pattern is None:

            ax.set_title("Current PLM Output Phase")

            ax.text(0.5, 0.5, "PLM loop idle", ha="center", va="center")

            ax.set_axis_off()

            self.phase_cal_live_pattern_label.setText("Current PLM pattern: idle")

            self.phase_cal_live_pattern_canvas.draw_idle()

            return

        phase_img = np.asarray(phase_pattern, dtype=np.float32)

        im = ax.imshow(
            phase_img,
            cmap=self._phase_colormap_name,
            vmin=0.0,
            vmax=2.0 * np.pi,
            aspect="auto",
        )

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Phase (rad)")
        cbar.set_ticks([0.0, np.pi, 2.0 * np.pi])
        cbar.set_ticklabels(["0", "pi", "2pi"])

        ax.set_title("Current PLM Output Phase (wrapped)")

        ax.set_xlabel("X (px)")

        ax.set_ylabel("Y (px)")

        self.phase_cal_live_pattern_label.setText(
            f"Current PLM pattern: shape {phase_img.shape[1]}x{phase_img.shape[0]} | phase range [{float(np.nanmin(phase_img)):.3f}, {float(np.nanmax(phase_img)):.3f}] rad"
        )

        fig.tight_layout()

        self.phase_cal_live_pattern_canvas.draw_idle()

    def _on_plm_phase_preview_updated(self, phase_pattern: object) -> None:

        try:

            pattern = np.asarray(phase_pattern, dtype=np.float32)

            if pattern.ndim != 2:

                return

            self._phase_cal_latest_plm_phase_pattern = pattern

            self._refresh_phase_calibration_live_pattern_plot(pattern)

        except Exception:

            return

    def _update_phase_calibration_partial_map_for_signal(
        self, signal_flat: int
    ) -> None:

        if self._phase_cal_step_mean is None:

            return

        if (
            self._phase_cal_superpixel_map is None
            or self._phase_cal_modulation_superpixel is None
        ):

            return

        if self._phase_cal_grid_cols <= 0 or self._phase_cal_grid_rows <= 0:

            return

        pattern_indices = [
            self._phase_cal_pattern_index_by_signal_step.get(
                (int(signal_flat), step_idx), -1
            )
            for step_idx in range(4)
        ]

        if any(int(idx) < 0 for idx in pattern_indices):
            return

        vals = [float(self._phase_cal_step_mean[int(idx)]) for idx in pattern_indices]

        if not all(np.isfinite(v) for v in vals):
            return

        i1, i2, i3, i4 = vals

        #theta = (float(np.arctan2(i4 - i2, i1 - i3)))

        theta = wrap_phase_to_pi(float(np.arctan2(i4 - i2, i1 - i3)))
        #theta = wrap_phase_to_pi(float(np.arctan2(i2 - i4, i1 - i3)))
        modulation = float(0.5 * np.hypot(i4 - i2, i1 - i3))
        row, col = divmod(int(signal_flat), int(self._phase_cal_grid_cols))

        if 0 <= row < int(self._phase_cal_grid_rows) and 0 <= col < int(
            self._phase_cal_grid_cols
        ):
            self._phase_cal_superpixel_map[row, col] = theta
            self._phase_cal_modulation_superpixel[row, col] = modulation

            if self._phase_cal_fullres_shape is not None:

                full_h, full_w = int(self._phase_cal_fullres_shape[0]), int(
                    self._phase_cal_fullres_shape[1]
                )

                self._phase_cal_fullres_map = self._expand_superpixel_map_to_fullres(
                    self._phase_cal_superpixel_map,
                    width=full_w,
                    height=full_h,
                )

    def _finalize_phase_calibration_from_measurements(self) -> None:

        if self._phase_cal_step_mean is None:

            return

        n_cells = int(self._phase_cal_grid_rows * self._phase_cal_grid_cols)

        signal_steps = np.full((n_cells, 4), np.nan, dtype=np.float64)

        for pattern_idx, (signal_flat, step_idx) in enumerate(
            self._phase_cal_pattern_meta
        ):

            signal_steps[int(signal_flat), int(step_idx)] = float(
                self._phase_cal_step_mean[pattern_idx]
            )

        superpixel_map = np.full(
            (self._phase_cal_grid_rows, self._phase_cal_grid_cols),
            np.nan,
            dtype=np.float32,
        )

        modulation_map = np.full(
            (self._phase_cal_grid_rows, self._phase_cal_grid_cols),
            np.nan,
            dtype=np.float32,
        )

        ref_r, ref_c = divmod(
            int(self._phase_cal_ref_flat), int(self._phase_cal_grid_cols)
        )

        superpixel_map[ref_r, ref_c] = 0.0

        for signal_flat in self._phase_cal_signal_indices:
            vals = signal_steps[int(signal_flat)]
            if not np.all(np.isfinite(vals)):
                continue

            i1, i2, i3, i4 = [float(v) for v in vals]
            theta = wrap_phase_to_pi(float(np.arctan2(i4 - i2, i1 - i3)))
            modulation = float(0.5 * np.hypot(i4 - i2, i1 - i3))
            row, col = divmod(int(signal_flat), int(self._phase_cal_grid_cols))

            superpixel_map[row, col] = theta

            modulation_map[row, col] = modulation

        valid_mask = np.isfinite(superpixel_map)

        superpixel_stored = np.asarray(superpixel_map, dtype=np.float32)

        superpixel_for_fullres = np.where(valid_mask, superpixel_map, np.nan).astype(
            np.float32
        )

        plm_catalog = self.plm_catalog_edit.text().strip()

        plm = PLM.from_db(plm_catalog)

        height, width = int(plm.shape[0]), int(plm.shape[1])

        fullres_map = self._expand_superpixel_map_to_fullres(
            superpixel_for_fullres, width=width, height=height
        )

        self._phase_cal_superpixel_map = superpixel_stored

        self._phase_cal_modulation_superpixel = np.where(
            np.isfinite(modulation_map), modulation_map, 0.0
        ).astype(np.float32)

        self._phase_cal_fullres_map = fullres_map

        self._phase_cal_fullres_shape = (height, width)

        self._phase_cal_active_correction_map = np.asarray(
            fullres_map, dtype=np.float32
        ).copy()

        self._phase_cal_active_correction_shape = (height, width)

        valid_count = int(np.sum(valid_mask))

        total_count = int(superpixel_map.size)

        self.phase_cal_map_stats_label.setText(
            f"Map: valid {valid_count}/{total_count} super-pixels"
        )

        self._refresh_phase_calibration_map_plot()

        self._update_loaded_phase_map_label()

        self.phase_cal_apply_checkbox.setChecked(True)

        self._phase_cal_running = False

        self._phase_cal_completed_this_run = True

        self.phase_cal_start_button.setEnabled(True)

        self.phase_cal_stop_button.setEnabled(False)

        self.phase_cal_status_label.setText(
            "PSI status: calibration finished, correction map ready"
        )

        if self._plm_loop_thread is not None and self._plm_loop_thread.is_alive():

            self._stop_plm_pattern_loop()

    def _on_browse_phase_calibration_map_save_file(self) -> None:

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Select PSI map file",
            str(Path.cwd()),
            "NumPy archive (*.npz)",
        )

        if file_path:

            self.phase_cal_map_save_path_edit.setText(file_path)

    def _on_browse_phase_calibration_map_load_file(self) -> None:

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select PSI map file to load",
            str(Path.cwd()),
            "NumPy archive (*.npz)",
        )

        if file_path:

            self.phase_cal_map_load_path_edit.setText(file_path)

    def _update_loaded_phase_map_label(self) -> None:

        if not self._phase_cal_loaded_map_path:

            self.phase_cal_loaded_map_label.setText("Loaded map: none")

            return

        self.phase_cal_loaded_map_label.setText(
            f"Loaded map: {self._phase_cal_loaded_map_path}"
        )

    def _save_phase_calibration_map(self) -> None:

        if (
            self._phase_cal_fullres_map is None
            or self._phase_cal_superpixel_map is None
        ):

            self.phase_cal_status_label.setText("PSI status: no map to save")

            return

        path_text = self.phase_cal_map_save_path_edit.text().strip()

        if not path_text:

            self.phase_cal_status_label.setText(
                "PSI status: choose an output .npz path"
            )

            return

        path = Path(path_text)

        try:

            path.parent.mkdir(parents=True, exist_ok=True)

            map_superpixel_to_save = wrap_phase_array_to_pi(
                np.asarray(self._phase_cal_superpixel_map, dtype=np.float32)
            )

            map_fullres_to_save = wrap_phase_array_to_pi(
                np.asarray(self._phase_cal_fullres_map, dtype=np.float32)
            )

            np.savez_compressed(
                path,
                map_superpixel=map_superpixel_to_save,
                map_fullres=map_fullres_to_save,
                modulation_superpixel=(
                    self._phase_cal_modulation_superpixel.astype(np.float32)
                    if self._phase_cal_modulation_superpixel is not None
                    else np.zeros_like(self._phase_cal_superpixel_map, dtype=np.float32)
                ),
                grid_rows=np.int32(self._phase_cal_grid_rows),
                grid_cols=np.int32(self._phase_cal_grid_cols),
                superpixel_x=np.int32(self._phase_cal_spx),
                superpixel_y=np.int32(self._phase_cal_spy),
                section_row_start=np.int32(self._phase_cal_section_row_start),
                section_row_end=np.int32(self._phase_cal_section_row_end),
                section_col_start=np.int32(self._phase_cal_section_col_start),
                section_col_end=np.int32(self._phase_cal_section_col_end),
                grating_enabled=np.int32(
                    1 if self.phase_cal_enable_grating_checkbox.isChecked() else 0
                ),
                grating_period_x=np.int32(
                    int(self.phase_cal_grating_period_spin.value())
                ),
                reference_flat=np.int32(self._phase_cal_ref_flat),
                fullres_height=np.int32(self._phase_cal_fullres_map.shape[0]),
                fullres_width=np.int32(self._phase_cal_fullres_map.shape[1]),
                phase_steps=np.asarray(
                    [0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi], dtype=np.float32
                ),
                created_iso=np.asarray(datetime.now().isoformat(timespec="seconds")),
            )

            self.phase_cal_status_label.setText(f"PSI status: map saved to {path}")

        except Exception as exc:

            self.phase_cal_status_label.setText(
                f"PSI status: failed to save map: {exc}"
            )

    def _load_phase_calibration_map(self) -> None:
        path_text = self.phase_cal_map_load_path_edit.text().strip()
        if not path_text:
            self.phase_cal_status_label.setText("PSI status: choose a .npz map path")
            return

        path = Path(path_text)
        if not path.exists():

            self.phase_cal_status_label.setText(
                f"PSI status: map file not found: {path}"
            )
            return

        try:

            with np.load(path) as data:
                if "map_fullres" in data:
                    fullres_map = np.asarray(data["map_fullres"], dtype=np.float32)

                elif (
                    "map_superpixel" in data
                    and "fullres_height" in data
                    and "fullres_width" in data
                ):

                    superpixel_tmp = np.asarray(
                        data["map_superpixel"], dtype=np.float32
                    )
                    h = int(data["fullres_height"])
                    w = int(data["fullres_width"])

                    spx = (
                        int(data["superpixel_x"])
                        if "superpixel_x" in data
                        else int(self.phase_cal_spx_spin.value())
                    )

                    spy = (
                        int(data["superpixel_y"])
                        if "superpixel_y" in data
                        else int(self.phase_cal_spy_spin.value())
                    )

                    self._phase_cal_spx = self._coerce_to_multiple(
                        spx, base=6, minimum=6, maximum=510
                    )

                    self._phase_cal_spy = self._coerce_to_multiple(
                        spy, base=9, minimum=9, maximum=504
                    )

                    fullres_map = np.repeat(
                        np.repeat(superpixel_tmp, self._phase_cal_spy, axis=0),
                        self._phase_cal_spx,
                        axis=1,
                    )[:h, :w]

                else:
                    raise ValueError("Missing map_fullres in archive")

                self._phase_cal_fullres_map = wrap_phase_array_to_pi(
                    np.asarray(fullres_map, dtype=np.float32)
                )

                self._phase_cal_fullres_shape = (
                    int(self._phase_cal_fullres_map.shape[0]),
                    int(self._phase_cal_fullres_map.shape[1]),
                )

                self._phase_cal_active_correction_map = np.asarray(
                    self._phase_cal_fullres_map, dtype=np.float32
                ).copy()

                self._phase_cal_active_correction_shape = (
                    int(self._phase_cal_fullres_map.shape[0]),
                    int(self._phase_cal_fullres_map.shape[1]),
                )

                if "map_superpixel" in data:

                    self._phase_cal_superpixel_map = wrap_phase_array_to_pi(
                        np.asarray(data["map_superpixel"], dtype=np.float32)
                    )

                else:

                    self._phase_cal_superpixel_map = None

                if "modulation_superpixel" in data:

                    self._phase_cal_modulation_superpixel = np.asarray(
                        data["modulation_superpixel"], dtype=np.float32
                    )

                else:

                    self._phase_cal_modulation_superpixel = None

                if "grid_rows" in data:

                    self._phase_cal_grid_rows = int(data["grid_rows"])

                if "grid_cols" in data:

                    self._phase_cal_grid_cols = int(data["grid_cols"])

                if "superpixel_x" in data:

                    self._phase_cal_spx = self._coerce_to_multiple(
                        int(data["superpixel_x"]), base=6, minimum=6, maximum=510
                    )

                if "superpixel_y" in data:

                    self._phase_cal_spy = self._coerce_to_multiple(
                        int(data["superpixel_y"]), base=9, minimum=9, maximum=504
                    )

                if "section_row_start" in data:

                    self._phase_cal_section_row_start = int(data["section_row_start"])

                if "section_row_end" in data:

                    self._phase_cal_section_row_end = int(data["section_row_end"])

                if "section_col_start" in data:

                    self._phase_cal_section_col_start = int(data["section_col_start"])

                if "section_col_end" in data:

                    self._phase_cal_section_col_end = int(data["section_col_end"])

                if "grating_period_x" in data:

                    self._phase_cal_grating_period_x = int(data["grating_period_x"])

                if "grating_enabled" in data:

                    self.phase_cal_enable_grating_checkbox.setChecked(
                        int(data["grating_enabled"]) != 0
                    )

                if "reference_flat" in data:

                    self._phase_cal_ref_flat = int(data["reference_flat"])

            self._phase_cal_loaded_map_path = str(path)

            self._update_loaded_phase_map_label()

            self.phase_cal_spx_spin.setValue(self._phase_cal_spx)

            self.phase_cal_spy_spin.setValue(self._phase_cal_spy)

            self.phase_cal_grating_period_spin.setValue(
                max(1, int(self._phase_cal_grating_period_x))
            )

            self.phase_cal_sec_row_start_spin.setValue(
                max(0, int(self._phase_cal_section_row_start))
            )

            self.phase_cal_sec_row_end_spin.setValue(
                max(0, int(self._phase_cal_section_row_end))
            )

            self.phase_cal_sec_col_start_spin.setValue(
                max(0, int(self._phase_cal_section_col_start))
            )

            self.phase_cal_sec_col_end_spin.setValue(
                max(0, int(self._phase_cal_section_col_end))
            )

            self._on_phase_calibration_section_changed()

            self.phase_cal_map_stats_label.setText(
                f"Map: loaded {self._phase_cal_fullres_map.shape[1]}x{self._phase_cal_fullres_map.shape[0]} | phase range [{float(np.nanmin(self._phase_cal_fullres_map)):.3f}, {float(np.nanmax(self._phase_cal_fullres_map)):.3f}] rad"
            )

            self.phase_cal_status_label.setText(f"PSI status: map loaded from {path}")

            self.phase_cal_apply_checkbox.setChecked(True)

            correction_preview = np.mod(
                -wrap_phase_array_to_pi(
                    np.asarray(self._phase_cal_active_correction_map, dtype=np.float32)
                ),
                2.0 * np.pi,
            ).astype(np.float32)

            self._phase_cal_latest_plm_phase_pattern = correction_preview

            self._refresh_phase_calibration_live_pattern_plot(
                self._phase_cal_latest_plm_phase_pattern
            )

            self._refresh_phase_calibration_map_plot()

        except Exception as exc:

            self.phase_cal_status_label.setText(
                f"PSI status: failed to load map: {exc}"
            )

    def _clear_phase_calibration_map(self) -> None:
        self._phase_cal_superpixel_map = None
        self._phase_cal_modulation_superpixel = None
        self._phase_cal_fullres_map = None
        self._phase_cal_fullres_shape = None
        self._phase_cal_active_correction_map = None
        self._phase_cal_active_correction_shape = None
        self._phase_cal_loaded_map_path = None
        self.phase_cal_apply_checkbox.setChecked(False)
        self.phase_cal_map_stats_label.setText("Map: not available")
        self._update_loaded_phase_map_label()
        self.phase_cal_map_shape_label.setText("Map grid: n/a")
        self.phase_cal_status_label.setText("PSI status: map cleared")
        self._refresh_phase_calibration_map_plot()

    def _on_phase_calibration_apply_toggled(self, checked: bool) -> None:

        if checked and self._phase_cal_fullres_map is None:

            self.phase_cal_apply_checkbox.blockSignals(True)

            self.phase_cal_apply_checkbox.setChecked(False)

            self.phase_cal_apply_checkbox.blockSignals(False)

            self.phase_cal_status_label.setText(
                "PSI status: load or run calibration first"
            )

            return

        state_text = "enabled" if checked else "disabled"

        self.phase_cal_status_label.setText(f"PSI status: correction {state_text}")

    def _get_active_phase_correction_map(
        self, shape: tuple[int, int]
    ) -> Optional[np.ndarray]:

        if not self.phase_cal_apply_checkbox.isChecked():
            return None
        correction_src = self._phase_cal_active_correction_map
        correction_shape = self._phase_cal_active_correction_shape

        if correction_src is None:
            correction_src = self._phase_cal_fullres_map
            correction_shape = self._phase_cal_fullres_shape

        if correction_src is None:
            return None

        target_h, target_w = int(shape[0]), int(shape[1])

        if (
            correction_shape is not None
            and int(correction_shape[0]) == target_h
            and int(correction_shape[1]) == target_w
        ):

            correction = np.asarray(correction_src, dtype=np.float32)

        else:

            correction = self._resample_phase_map_nearest(
                np.asarray(correction_src, dtype=np.float32), (target_h, target_w)
            )

        correction_principal = wrap_phase_array_to_pi(
            np.asarray(correction, dtype=np.float32)
        )

        # Do not negate as conjugation is already applied, then rescale to PLM phase domain [0, 2pi).
        correction_rescaled = np.mod(correction_principal, 2.0 * np.pi).astype(
            np.float32,
            copy=False,
        )

        return np.nan_to_num(correction_rescaled, nan=0.0).astype(
            np.float32, copy=False
        )

    def _build_phase_calibration_pattern_for_index(
        self,
        pattern_index: int,
        shape: tuple[int, int],
    ) -> np.ndarray:

        height, width = int(shape[0]), int(shape[1])
        if pattern_index < 0 or pattern_index >= len(self._phase_cal_pattern_meta):
            raise IndexError(f"PSI pattern index out of range: {pattern_index}")

        if self._phase_cal_grid_cols <= 0 or self._phase_cal_grid_rows <= 0:
            raise RuntimeError("PSI grid is not initialized")

        signal_flat, step_idx = self._phase_cal_pattern_meta[pattern_index]
        phase_steps = (0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi)
        step_idx = int(step_idx)

        if step_idx < 0 or step_idx >= len(phase_steps):
            raise RuntimeError(f"Invalid PSI step index: {step_idx}")

        pattern = np.zeros((height, width), dtype=np.float32)
        ref_r, ref_c = divmod(
            int(self._phase_cal_ref_flat), int(self._phase_cal_grid_cols)
        )
        ref_y0 = int(ref_r * self._phase_cal_spy)
        ref_y1 = min(height, ref_y0 + int(self._phase_cal_spy))
        ref_x0 = int(ref_c * self._phase_cal_spx)
        ref_x1 = min(width, ref_x0 + int(self._phase_cal_spx))

        pattern[ref_y0:ref_y1, ref_x0:ref_x1] = float(phase_steps[step_idx])
        sig_r, sig_c = divmod(int(signal_flat), int(self._phase_cal_grid_cols))

        sig_y0 = int(sig_r * self._phase_cal_spy)

        sig_y1 = min(height, sig_y0 + int(self._phase_cal_spy))

        sig_x0 = int(sig_c * self._phase_cal_spx)

        sig_x1 = min(width, sig_x0 + int(self._phase_cal_spx))

        if self.phase_cal_enable_grating_checkbox.isChecked():

            period_x = max(1, int(self.phase_cal_grating_period_spin.value()))

            self._phase_cal_grating_period_x = period_x

            x = np.arange(width, dtype=np.float32)

            ramp_1d = np.mod(2.0 * np.pi * x / float(period_x), 2.0 * np.pi)

            ref_patch = np.mod(
                float(phase_steps[step_idx]) + ramp_1d[ref_x0:ref_x1][None, :],
                2.0 * np.pi,
            )

            sig_patch = np.mod(ramp_1d[sig_x0:sig_x1][None, :], 2.0 * np.pi)

            pattern[ref_y0:ref_y1, ref_x0:ref_x1] = ref_patch

            pattern[sig_y0:sig_y1, sig_x0:sig_x1] = sig_patch

        else:

            pattern[sig_y0:sig_y1, sig_x0:sig_x1] = 0.0

        return pattern

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

            self.pattern_status_label.setText(
                "Stop PLM loop before loading another file."
            )

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

            self.pattern_status_label.setText(
                "Pattern file loaded. Ready to start PLM loop."
            )

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

        dynamic_phase_cal = bool(
            self._phase_cal_running
            and self._plm_phase_patterns is None
            and self._phase_cal_total_patterns > 0
            and len(self._phase_cal_pattern_meta) == self._phase_cal_total_patterns
        )

        if self._plm_phase_patterns is None and not dynamic_phase_cal:

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

        total_for_counter = (
            int(self._phase_cal_total_patterns)
            if dynamic_phase_cal
            else int(self._plm_phase_patterns.shape[0])
        )

        self._set_pattern_counter(1, total_for_counter)

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

    def _plm_pattern_worker(
        self, plm_catalog: str, monitor: int, interval_ms: float
    ) -> None:

        message = "PLM loop stopped."

        try:

            phase_patterns = self._plm_phase_patterns

            dynamic_phase_cal = bool(
                phase_patterns is None
                and self._phase_cal_running
                and self._phase_cal_total_patterns > 0
                and len(self._phase_cal_pattern_meta) == self._phase_cal_total_patterns
            )

            if phase_patterns is None and not dynamic_phase_cal:

                raise RuntimeError("No phase patterns loaded.")

            plm = PLM.from_db(plm_catalog)

            expected_h, expected_w = int(plm.shape[0]), int(plm.shape[1])

            if (not dynamic_phase_cal) and (
                phase_patterns.shape[1] != expected_h
                or phase_patterns.shape[2] != expected_w
            ):

                raise ValueError(
                    f"Pattern shape {phase_patterns.shape[1:]} does not match PLM shape {(expected_h, expected_w)} for catalog'{plm_catalog}'."
                )

            phase_levels, buckets, phase_values = get_phase_values_from_calibration(plm)

            n_states = int(len(phase_values))

            if n_states <= 0 or len(phase_levels) != n_states:

                raise RuntimeError("Invalid PLM calibration LUT.")

            interval_s = max(0.01, interval_ms / 1000.0)

            with ImageWindow(fullscreen=True, monitor=monitor) as win:

                frame_idx = 0

                last_update = perf_counter() - interval_s

                total_frames = (
                    int(self._phase_cal_total_patterns)
                    if dynamic_phase_cal
                    else int(phase_patterns.shape[0])
                )

                cached_periods = (-1, -1)

                cached_ramp = np.zeros((expected_h, expected_w), dtype=np.float32)

                def _compose_overlaid_phase(display_idx: int) -> np.ndarray:
                    nonlocal cached_periods, cached_ramp
                    if dynamic_phase_cal:
                        base_phase = self._build_phase_calibration_pattern_for_index(
                            display_idx, (expected_h, expected_w)
                        )
                        # For PSI acquisition, keep non-target super-pixels at background phase.
                        global_grating_phase = np.zeros(
                            (expected_h, expected_w), dtype=np.float32
                        )
                        zernike_phase = np.zeros(
                            (expected_h, expected_w), dtype=np.float32
                        )

                    else:

                        base_phase = np.mod(phase_patterns[display_idx], 2.0 * np.pi)
                        periods = (
                            int(self._grating_period_x_px),
                            int(self._grating_period_y_px),
                        )
                        if periods != cached_periods:
                            cached_ramp = self._build_grating_phase_ramp(
                                (expected_h, expected_w), periods[0], periods[1]
                            )
                            cached_periods = periods
                        global_grating_phase = cached_ramp
                        zernike_phase = self._build_zernike_phase_map(
                            (expected_h, expected_w)
                        )

                    phase_correction = self._get_active_phase_correction_map(
                        (expected_h, expected_w)
                    )

                    if phase_correction is None:
                        return np.mod(
                            base_phase + global_grating_phase + zernike_phase,
                            2.0 * np.pi,
                        )

                    return np.mod(
                        base_phase
                        + global_grating_phase
                        + zernike_phase
                        + phase_correction,
                        2.0 * np.pi,
                    )

                def loop_callback() -> None:
                    nonlocal frame_idx, last_update, cached_periods, cached_ramp
                    if self._plm_stop_event.is_set():
                        raise EventLoopExit
                    
                    now = perf_counter()

                    if now - last_update >= interval_s:
                        display_idx = frame_idx
                        overlaid_phase = _compose_overlaid_phase(display_idx)
                        self.plm_phase_preview_updated.emit(
                            np.asarray(overlaid_phase, dtype=np.float32)
                        )

                        state_indices = np.digitize(
                            overlaid_phase, buckets, right=False
                        ).astype(np.int32)

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
                initial_phase_overlaid = _compose_overlaid_phase(0)
                self.plm_phase_preview_updated.emit(
                    np.asarray(initial_phase_overlaid, dtype=np.float32)
                )

                initial_state_indices = np.digitize(
                    initial_phase_overlaid, buckets, right=False
                ).astype(np.int32)

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

        if self._phase_cal_running:
            self._phase_cal_running = False
            self._phase_cal_completed_this_run = False
            self.phase_cal_start_button.setEnabled(True)
            self.phase_cal_stop_button.setEnabled(False)
            self.phase_cal_status_label.setText(
                "PSI status: calibration stopped before completion"
            )

        elif self._phase_cal_completed_this_run:
            self.phase_cal_status_label.setText("PSI status: calibration completed")
            self._phase_cal_completed_this_run = False
        self._restore_patterns_after_phase_calibration()

        if self._phase_cal_visibility_running:
            self._phase_cal_visibility_running = False
            self.phase_cal_visibility_button.blockSignals(True)
            self.phase_cal_visibility_button.setChecked(False)
            self.phase_cal_visibility_button.blockSignals(False)
            self.phase_cal_visibility_button.setText("Show Section Visibility")
        self._restore_patterns_after_phase_calibration_visibility()

        if not self._logging_enabled:
            self._current_plm_pattern_index = None
            self._current_plm_total_patterns = None
            self._set_pattern_counter(0, 0)

        self._phase_cal_latest_plm_phase_pattern = None
        self._refresh_phase_calibration_live_pattern_plot(None)
    
    
    @QtCore.pyqtSlot(str)
    def _on_processor_error(self, err_msg):
        self._set_status(err_msg)
        self.fft_status_label.setText("Thread error, check status bar.")

    @QtCore.pyqtSlot(dict)
    def _on_frame_processed(self, payload):
        # 1. Unpack the payload
        frame_count = payload["frame_count"]
        image = payload["image"]
        fft_image = payload["fft_image"]
        recovered_field = payload["recovered_field"]
        phase_rgb = payload["phase_rgb"]
        roi_sum = payload["roi_sum"]
        roi_mean = payload["roi_mean"]
        pattern_seq = payload["pattern_seq"] # This guarantees sync!
        pattern_idx = payload["pattern_idx"] # This guarantees sync!
        
        self._last_processed_frame_count = frame_count

        # --- DATA PROCESSING (Always execute to prevent missing data) ---
        
        self._handle_phase_calibration_frame(image, frame_count)
        self._latest_recovered_field = recovered_field

        if roi_sum is not None and roi_mean is not None:
            self._append_live_intensity_sample(roi_sum)

        if recovered_field is not None and phase_rgb is not None:
            # Check if this is a stable frame tied to a valid pattern
            if pattern_idx is not None and pattern_idx > 0 and pattern_seq > 0:
                
                # Initialize sequence tracker if it doesn't exist
                if not hasattr(self, "_last_captured_gallery_seq"):
                    self._last_captured_gallery_seq = -1
                
                # ONLY capture if we haven't already captured this exact PLM sequence!
                if pattern_seq > self._last_captured_gallery_seq:
                    self._last_captured_gallery_seq = pattern_seq
                    
                    # Force the capture functions to use the STAMPED index!
                    self._store_pattern_field_direct(recovered_field, pattern_idx)
                    self._store_recovered_mode_direct(recovered_field, pattern_idx)
                    self._capture_pattern_preview_direct(phase_rgb, pattern_idx)
            
            self._maybe_log_current_measurement(frame_count)
            
            p1, p2 = self.recovered_field_label.get_points()
            self._update_recovered_point_readout(p1, p2)

        # --- UI DRAWING THROTTLE (Max ~30 FPS to stop mouse lag) ---
        now = perf_counter()
        if now - self._last_ui_update_time > 0.033: 
            self._last_ui_update_time = now
            
            # Only update the heavy image labels here
            self.image_label.set_image_array(image)
            self.fft_image_label.set_image_array(fft_image)
            
            if phase_rgb is not None:
                self.recovered_field_label.set_image_rgb_array(phase_rgb)

            # Update text labels
            if roi_sum is not None:
                self.live_roi_intensity_label.setText(
                    f"Live ROI sum intensity: {roi_sum:.3f} | mean={roi_mean:.3f}"
                )

            if self._gaussian_mfd_enabled:
                mfd_x_um, mfd_y_um, mfd_eq_um, source_name = self._measure_gaussian_mfd(image)
                if mfd_x_um is None or mfd_y_um is None or mfd_eq_um is None:
                    self.gaussian_mfd_label.setText(
                        f"Gaussian MFD (pixel pitch 15 um, {source_name}): fit failed"
                    )
                else:
                    self.gaussian_mfd_label.setText(
                        f"Gaussian MFD (pixel pitch 15 um, {source_name}): X={mfd_x_um:.1f} um, Y={mfd_y_um:.1f} um, eq={mfd_eq_um:.1f} um"
                    )
            
            if self._fft_roi is None:
                self.fft_status_label.setText(
                    f"2D FFT updated. Frames processed: {frame_count}. Draw ROI around first order."
                )
                self.recovered_field_status_label.setText("Select FFT ROI to recover field.")
            elif recovered_field is not None:
                x0, y0, x1, y1 = self._fft_roi
                self.recovered_field_status_label.setText(
                    f"Recovered phase updated from ROI x={x0}:{x1}, y={y0}:{y1}."
                )
            
            self._set_status(f"Live view started. Frames received: {frame_count}")


    def closeEvent(self, event):
        # self.timer.stop()
        self._phase_cal_running = False
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

def get_phase_values_from_calibration(
    plm: PLM,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    phase_min, phase_max = plm.phase_range
    n_states = len(plm.displacement_ratios)
    ratio_scale = (
        plm.max_displacement_ratio
        if plm.max_displacement_ratio is not None
        else (n_states - 1) / n_states
    )

    phase_levels = phase_min + plm.displacement_ratios * ratio_scale * (
        phase_max - phase_min
    )

    phase_disp = np.hstack([phase_levels, phase_max])
    buckets = (phase_disp[:-1] + phase_disp[1:]) / 2.0
    representative_values = np.empty(n_states, dtype=np.float64)
    representative_values[0] = phase_min + 1e-9
    if n_states > 1:

        representative_values[1:] = 0.5 * (buckets[:-1] + buckets[1:])
    return phase_levels, buckets, representative_values






if __name__ == "__main__":

    raise SystemExit(main())
