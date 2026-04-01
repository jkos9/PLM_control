from __future__ import annotations

import queue
import time
import threading
from typing import Optional

import numpy as np

from PyQt5.QtCore import QThread, pyqtSignal
from vmbpy import PixelFormat, VmbSystem


class CameraWorker(QThread):
	error = pyqtSignal(str)
	started_stream = pyqtSignal(str)
	stopped_stream = pyqtSignal()

	def __init__(
		self,
		camera_id: Optional[str] = None,
		queue_maxsize: int = 5,
		parent=None,
	):
		super().__init__(parent)
		self.camera_id = camera_id
		self.frame_queue: queue.Queue = queue.Queue(maxsize=max(1, int(queue_maxsize)))
		self._stop_event = threading.Event()
		self._config_lock = threading.Lock()
		self._config_dirty = True
		self._auto_exposure_enabled = False
		self._manual_exposure_us = 1000.0
		self._trigger_mode = "freerun"
		self._trigger_delay_ms = 0.0

	def set_auto_exposure(self, enabled: bool) -> None:
		with self._config_lock:
			self._auto_exposure_enabled = bool(enabled)
			self._config_dirty = True

	def set_manual_exposure_us(self, exposure_us: float) -> None:
		exposure = max(10.0, float(exposure_us))
		with self._config_lock:
			self._manual_exposure_us = exposure
			self._config_dirty = True

	def set_trigger_mode(self, mode: str) -> None:
		mode_value = str(mode).strip().lower()
		if mode_value not in {"freerun", "line1"}:
			mode_value = "freerun"
		with self._config_lock:
			self._trigger_mode = mode_value
			self._config_dirty = True

	def set_trigger_delay_ms(self, delay_ms: float) -> None:
		delay = max(0.0, float(delay_ms))
		with self._config_lock:
			self._trigger_delay_ms = delay
			self._config_dirty = True

	def _try_set_feature(self, camera, feature_name: str, value) -> bool:
		if not hasattr(camera, feature_name):
			return False
		try:
			getattr(camera, feature_name).set(value)
			return True
		except Exception:
			return False

	def _apply_camera_configuration(self, camera) -> None:
		with self._config_lock:
			if not self._config_dirty:
				return
			auto_exposure_enabled = self._auto_exposure_enabled
			manual_exposure_us = self._manual_exposure_us
			trigger_mode = self._trigger_mode
			trigger_delay_ms = self._trigger_delay_ms
			self._config_dirty = False

		if auto_exposure_enabled:
			self._try_set_feature(camera, "ExposureAuto", "Continuous")
		else:
			self._try_set_feature(camera, "ExposureAuto", "Off")
			if not self._try_set_feature(camera, "ExposureTime", manual_exposure_us):
				self._try_set_feature(camera, "ExposureTimeAbs", manual_exposure_us)

		if trigger_mode == "line1":
			self._try_set_feature(camera, "TriggerSelector", "FrameStart")
			self._try_set_feature(camera, "TriggerSource", "Line1")
			self._try_set_feature(camera, "TriggerActivation", "RisingEdge")
			self._try_set_feature(camera, "TriggerMode", "On")
			self._try_set_feature(camera, "TriggerDelay", trigger_delay_ms * 1000.0)
			self._try_set_feature(camera, "TriggerDelayAbs", trigger_delay_ms * 1000.0)
		else:
			self._try_set_feature(camera, "TriggerMode", "Off")
			self._try_set_feature(camera, "AcquisitionMode", "Continuous")

	def stop(self, timeout_ms: int = 3000) -> None:
		self._stop_event.set()
		if self.isRunning():
			self.wait(max(0, int(timeout_ms)))

	def _push_frame(self, image) -> None:
		try:
			self.frame_queue.put_nowait(image)
		except queue.Full:
			try:
				self.frame_queue.get_nowait()
			except queue.Empty:
				pass
			try:
				self.frame_queue.put_nowait(image)
			except queue.Full:
				pass

	def _frame_handler(self, cam, stream, frame) -> None:
		try:
			image = self._extract_image_from_frame(frame)
			self._push_frame(image)
		except Exception as exc:
			self.error.emit(f"Camera callback error: {exc}")
		finally:
			cam.queue_frame(frame)

	def _extract_image_from_frame(self, frame):
		try:
			image = None
			try:
				image = frame.as_opencv_image().copy()
			except Exception:
				image = None

			if image is None:
				try:
					frame_mono8 = frame.convert_pixel_format(PixelFormat.Mono8)
					image = frame_mono8.as_opencv_image().copy()
				except Exception:
					image = None

			if image is None:
				raise RuntimeError("Unable to extract image from camera frame.")

			if image.ndim == 3:
				image = image[..., 0]
			image = np.ascontiguousarray(image)
			return image
		except Exception as exc:
			raise RuntimeError(f"Unable to decode frame: {exc}")

	def run(self) -> None:
		self._stop_event.clear()
		try:
			with VmbSystem.get_instance() as vmb:
				cameras = vmb.get_all_cameras()
				if not cameras:
					self.error.emit("No cameras found.")
					return

				camera = None
				if self.camera_id:
					for candidate in cameras:
						if candidate.get_id() == self.camera_id:
							camera = candidate
							break
					if camera is None:
						self.error.emit(f"Camera with id '{self.camera_id}' not found.")
						return
				else:
					camera = cameras[0]

				with camera:
					try:
						camera.set_pixel_format(PixelFormat.Mono8)
					except Exception:
						pass
					self._apply_camera_configuration(camera)
					self.started_stream.emit(camera.get_name())
					last_frame_ts = time.monotonic()
					while not self._stop_event.is_set():
						self._apply_camera_configuration(camera)
						try:
							frame = camera.get_frame(timeout_ms=250)
						except Exception:
							if (time.monotonic() - last_frame_ts) > 2.0 and self._trigger_mode == "line1":
								self.error.emit("No frames received: waiting for Line1 trigger pulses.")
							elif (time.monotonic() - last_frame_ts) > 2.0 and self._trigger_mode != "line1":
								self.error.emit("No frames received from camera. Check exposure/trigger configuration.")
							time.sleep(0.01)
							continue

						try:
							image = self._extract_image_from_frame(frame)
							self._push_frame(image)
							last_frame_ts = time.monotonic()
						except Exception as exc:
							self.error.emit(f"Frame decode error: {exc}")
		except Exception as exc:
			self.error.emit(f"Camera worker error: {exc}")
		finally:
			self.stopped_stream.emit()

