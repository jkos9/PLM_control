import sys
import importlib
import threading
from typing import Optional

from vmbpy import PixelFormat, VmbSystem

try:
	QtCore = importlib.import_module("PyQt5.QtCore")
	QtGui = importlib.import_module("PyQt5.QtGui")
	QtWidgets = importlib.import_module("PyQt5.QtWidgets")

	Qt = QtCore.Qt
	QTimer = QtCore.QTimer
	QImage = QtGui.QImage
	QPixmap = QtGui.QPixmap
	QApplication = QtWidgets.QApplication
	QCheckBox = QtWidgets.QCheckBox
	QDoubleSpinBox = QtWidgets.QDoubleSpinBox
	QHBoxLayout = QtWidgets.QHBoxLayout
	QLabel = QtWidgets.QLabel
	QMainWindow = QtWidgets.QMainWindow
	QPushButton = QtWidgets.QPushButton
	QVBoxLayout = QtWidgets.QVBoxLayout
	QWidget = QtWidgets.QWidget
except ImportError as exc:
	raise ImportError(
		"PyQt5 is required. Install with: pip install PyQt5"
	) from exc


class LiveCameraWindow(QMainWindow):
	def __init__(self, cam):
		super().__init__()
		self.cam = cam
		self.setWindowTitle("Allied Vision Live Camera")
		self.resize(1100, 800)

		self.image_label = QLabel("No frame yet")
		self.image_label.setAlignment(Qt.AlignCenter)
		self.image_label.setMinimumSize(960, 640)

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

		controls = QHBoxLayout()
		controls.addWidget(self.start_button)
		controls.addWidget(self.stop_button)
		controls.addSpacing(20)
		controls.addWidget(self.auto_exposure_checkbox)
		controls.addWidget(QLabel("Manual exposure:"))
		controls.addWidget(self.exposure_spinbox)
		controls.addStretch(1)

		root = QVBoxLayout()
		root.addLayout(controls)
		root.addWidget(self.image_label)
		root.addWidget(self.status_label)

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

		self.start_button.clicked.connect(self._start_live)
		self.stop_button.clicked.connect(self._stop_live)
		self.auto_exposure_checkbox.toggled.connect(self._on_auto_exposure_toggled)
		self.exposure_spinbox.editingFinished.connect(self._on_manual_exposure_changed)

		self._exp_auto_feature_name: Optional[str] = self._detect_exposure_auto_feature_name()
		self._exp_time_feature_name: Optional[str] = self._detect_exposure_time_feature_name()

		self._initialize_exposure_controls()

	def _set_status(self, message: str) -> None:
		self.status_label.setText(message)

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

	def _update_frame(self) -> None:
		try:
			with self._frame_lock:
				if self._latest_frame is None:
					if self._last_frame_error:
						self._set_status(self._last_frame_error)
					elif self._streaming:
						self._set_status("Waiting for first frame...")
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
			self._set_status(f"Live view started. Frames received: {frame_count}")
		except Exception as exc:
			self._set_status(f"Render error: {exc}")

	def closeEvent(self, event):
		self.timer.stop()
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


if __name__ == "__main__":
	raise SystemExit(main())
