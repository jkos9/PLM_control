from __future__ import annotations

import sys
import time
import threading
from collections import deque
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

from utils.ui_main import LiveCameraWindow
from utils.camera_worker import CameraWorker
from utils.math_utils import compute_fft_shifted_and_magnitude_image, recover_off_axis_field


class AppController(QtCore.QObject):
	plm_counter_updated = QtCore.pyqtSignal(int, int)
	plm_loop_finished = QtCore.pyqtSignal(str)

	def __init__(self, window: LiveCameraWindow, camera_worker: CameraWorker):
		super().__init__()
		self.window = window
		self.camera_worker = camera_worker
		self._is_running = False
		self._phase_t = deque(maxlen=600)
		self._phi1 = deque(maxlen=600)
		self._phi2 = deque(maxlen=600)
		self._dphi = deque(maxlen=600)
		self._intensity_t = deque(maxlen=600)
		self._intensity = deque(maxlen=600)
		self._plot_time = 0.0
		self._last_heavy_update_ts = 0.0
		self._heavy_update_interval_s = 0.12
		self._drop_frames_after_pattern = 0
		self._drop_frames_remaining = 0

		self._pattern_stack: np.ndarray | None = None
		self._plm_bmp_stack: list[np.ndarray] = []
		self._plm_worker_thread: threading.Thread | None = None
		self._plm_stop_event = threading.Event()
		self._plm_frame_interval_s = 0.1
		self._plm_monitor = -1
		self._pattern_index = 0
		self._pattern_loop_running = False

		self._zernike_terms: dict[tuple[int, int], float] = {}
		self._zernike_optimizer_running = False
		self._zernike_timer = QtCore.QTimer(self)
		self._zernike_timer.setInterval(120)
		self._zernike_timer.timeout.connect(self._zernike_optimizer_step)
		self._zernike_iterations_done = 0
		self._zernike_target_iterations = 0
		self._zernike_step_size = 0.1

		self._channel_matrix_history: list[np.ndarray] = []

		self._poll_timer = QtCore.QTimer(self)
		self._poll_timer.setInterval(5)
		self._poll_timer.timeout.connect(self._poll_camera_queue)

		self.camera_worker.error.connect(self._on_camera_error, QtCore.Qt.QueuedConnection)
		self.camera_worker.started_stream.connect(self._on_camera_started, QtCore.Qt.QueuedConnection)
		self.camera_worker.stopped_stream.connect(self._on_camera_stopped, QtCore.Qt.QueuedConnection)
		self.window.start_button.clicked.connect(self.start)
		self.window.stop_button.clicked.connect(self.stop)
		self.window.auto_exposure_checkbox.toggled.connect(self._on_auto_exposure_toggled)
		self.window.exposure_spinbox.valueChanged.connect(self._on_exposure_changed)
		self.window.trigger_mode_combo.currentIndexChanged.connect(self._on_trigger_mode_changed)
		self.window.trigger_delay_edit.editingFinished.connect(self._on_trigger_delay_changed)
		self.window.drop_frames_combo.currentIndexChanged.connect(self._on_drop_frames_changed)
		self.window.pattern_browse_button.clicked.connect(self._on_pattern_browse)
		self.window.pattern_load_button.clicked.connect(self._on_load_pattern_file)
		self.window.pattern_start_button.clicked.connect(self._start_plm_pattern_loop)
		self.window.pattern_stop_button.clicked.connect(self._stop_plm_pattern_loop)
		self.window.zernike_max_degree_combo.currentIndexChanged.connect(self._on_zernike_max_degree_changed)
		self.window.zernike_opt_start_button.clicked.connect(self._start_zernike_optimizer)
		self.window.zernike_opt_stop_button.clicked.connect(self._stop_zernike_optimizer)
		self.plm_counter_updated.connect(self._on_plm_counter_updated, QtCore.Qt.QueuedConnection)
		self.plm_loop_finished.connect(self._on_plm_loop_finished, QtCore.Qt.QueuedConnection)

		self._on_auto_exposure_toggled(self.window.auto_exposure_checkbox.isChecked())
		self._on_drop_frames_changed(self.window.drop_frames_combo.currentIndex())
		self._on_trigger_mode_changed(self.window.trigger_mode_combo.currentIndex())
		self._on_trigger_delay_changed()
		self._on_zernike_max_degree_changed(self.window.zernike_max_degree_combo.currentIndex())

	def start(self) -> None:
		if self._is_running:
			return
		self._is_running = True
		self.camera_worker.start()
		self._poll_timer.start()
		self.window.start_button.setEnabled(False)
		self.window.stop_button.setEnabled(True)
		self.window.status_label.setText("Starting camera stream...")

	def stop(self) -> None:
		if not self._is_running:
			return
		self._is_running = False
		self._poll_timer.stop()
		self.camera_worker.stop()
		self._stop_plm_pattern_loop()
		self._stop_zernike_optimizer()
		self.window.start_button.setEnabled(True)
		self.window.stop_button.setEnabled(False)
		self.window.status_label.setText("Camera stopped")

	@QtCore.pyqtSlot(str)
	def _on_camera_error(self, msg: str) -> None:
		print(msg)
		self.window.status_label.setText(msg)

	@QtCore.pyqtSlot(str)
	def _on_camera_started(self, camera_name: str) -> None:
		print(f"Using camera: {camera_name}")
		self.window.status_label.setText(f"Using camera: {camera_name}")

	@QtCore.pyqtSlot()
	def _on_camera_stopped(self) -> None:
		print("Camera stream stopped.")
		self.window.status_label.setText("Camera stream stopped.")

	@QtCore.pyqtSlot(bool)
	def _on_auto_exposure_toggled(self, checked: bool) -> None:
		self.camera_worker.set_auto_exposure(checked)
		self.window.exposure_spinbox.setEnabled(not checked)
		mode = "Auto" if checked else "Manual"
		self.window.status_label.setText(f"Exposure mode: {mode}")

	@QtCore.pyqtSlot(float)
	def _on_exposure_changed(self, value_us: float) -> None:
		self.camera_worker.set_manual_exposure_us(value_us)
		if self.window.auto_exposure_checkbox.isChecked():
			return
		self.window.status_label.setText(f"Manual exposure: {value_us:.1f} us")

	@QtCore.pyqtSlot(int)
	def _on_trigger_mode_changed(self, _: int) -> None:
		mode = self.window.trigger_mode_combo.currentData() or "freerun"
		self.camera_worker.set_trigger_mode(str(mode))
		self.window.status_label.setText(f"Trigger mode: {mode}")

	def _on_trigger_delay_changed(self) -> None:
		text = self.window.trigger_delay_edit.text().strip()
		try:
			delay_ms = float(text)
		except ValueError:
			delay_ms = 0.0
			self.window.trigger_delay_edit.setText("0.000")
		self.camera_worker.set_trigger_delay_ms(delay_ms)
		self.window.status_label.setText(f"Trigger delay: {delay_ms:.3f} ms")

	@QtCore.pyqtSlot(int)
	def _on_drop_frames_changed(self, _: int) -> None:
		self._drop_frames_after_pattern = int(self.window.drop_frames_combo.currentData() or 0)
		self.window.status_label.setText(f"Drop frames after pattern: {self._drop_frames_after_pattern}")

	def _on_pattern_browse(self) -> None:
		path, _ = QtWidgets.QFileDialog.getOpenFileName(
			self.window,
			"Select PLM Pattern Stack",
			str(Path.cwd()),
			"NumPy files (*.npy)",
		)
		if path:
			self.window.pattern_path_edit.setText(path)

	def _on_load_pattern_file(self) -> None:
		path = self.window.pattern_path_edit.text().strip()
		if not path:
			self.window.pattern_status_label.setText("Select a .npy file first.")
			return
		try:
			stack = np.load(path)
		except Exception as exc:
			self.window.pattern_status_label.setText(f"Load failed: {exc}")
			return
		if stack.ndim < 3:
			self.window.pattern_status_label.setText("Pattern stack must be at least 3D: [N, H, W].")
			return
		self._pattern_stack = np.asarray(stack, dtype=np.float32)
		self._plm_bmp_stack = []
		try:
			sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
			from ti_plm import PLM
			plm = PLM.from_db("p67nirtemp")
			for idx in range(int(self._pattern_stack.shape[0])):
				phase = np.asarray(self._pattern_stack[idx], dtype=np.float32)
				if phase.ndim > 2:
					phase = phase[..., 0]
				bmp = np.asarray(plm.process_phase_map(phase))
				if bmp.ndim == 3:
					bmp = bmp[..., 0]
				self._plm_bmp_stack.append(np.ascontiguousarray(bmp))
		except Exception as exc:
			self.window.pattern_status_label.setText(f"PLM conversion failed: {exc}")
			return
		self._pattern_index = 0
		total = int(self._pattern_stack.shape[0])
		self.window.pattern_counter_label.setText(f"Pattern: 1 / {total}")
		self.window.pattern_status_label.setText(f"Loaded {total} patterns from {Path(path).name}")
		self._refresh_pattern_gallery()
		self.window.pattern_start_button.setEnabled(True)

	def _on_pattern_load(self) -> None:
		self._on_load_pattern_file()

	def _start_plm_pattern_loop(self) -> None:
		if self._pattern_stack is None or not self._plm_bmp_stack:
			self.window.pattern_status_label.setText("Load a .npy pattern stack before starting.")
			return
		if self._pattern_loop_running:
			return
		if hasattr(self.window, "pattern_frame_ms_spin"):
			self._plm_frame_interval_s = max(0.001, float(self.window.pattern_frame_ms_spin.value()) / 1000.0)
		self._plm_stop_event.clear()
		self._pattern_loop_running = True
		self._plm_worker_thread = threading.Thread(target=self._plm_pattern_worker, daemon=True)
		self._plm_worker_thread.start()
		self.window.pattern_start_button.setEnabled(False)
		self.window.pattern_stop_button.setEnabled(True)
		self.window.pattern_status_label.setText("PLM loop running")

	def _stop_plm_pattern_loop(self) -> None:
		self._plm_stop_event.set()
		if not self._pattern_loop_running:
			self.window.pattern_start_button.setEnabled(self._pattern_stack is not None)
			self.window.pattern_stop_button.setEnabled(False)
			return

	def _stop_pattern_loop(self) -> None:
		self._stop_plm_pattern_loop()

	def _start_pattern_loop(self) -> None:
		self._start_plm_pattern_loop()

	def _advance_pattern(self) -> None:
		if self._pattern_stack is None or self._pattern_stack.shape[0] == 0:
			return
		total = int(self._pattern_stack.shape[0])
		self._pattern_index = (self._pattern_index + 1) % total
		self._drop_frames_remaining = self._drop_frames_after_pattern
		self.window.pattern_counter_label.setText(f"Pattern: {self._pattern_index + 1} / {total}")
		self.window.pattern_status_label.setText("PLM loop running")
		if self.window.zernike_auto_optimize_checkbox.isChecked() and not self._zernike_optimizer_running:
			self._start_zernike_optimizer()

	def _on_plm_counter_updated(self, index: int, total: int) -> None:
		self._pattern_index = max(0, int(index) - 1)
		self._drop_frames_remaining = self._drop_frames_after_pattern
		self.window.pattern_counter_label.setText(f"Pattern: {index} / {total}")
		if self.window.zernike_auto_optimize_checkbox.isChecked() and not self._zernike_optimizer_running:
			self._start_zernike_optimizer()

	def _on_plm_loop_finished(self, status_text: str) -> None:
		self._pattern_loop_running = False
		self.window.pattern_start_button.setEnabled(self._pattern_stack is not None)
		self.window.pattern_stop_button.setEnabled(False)
		self.window.pattern_status_label.setText(status_text)

	def _plm_pattern_worker(self) -> None:
		status_text = "PLM loop stopped"
		try:
			from PIL import Image
			from ti_plm.display import ImageWindow, EventLoopExit

			if not self._plm_bmp_stack:
				status_text = "No PLM patterns loaded"
				return

			total = len(self._plm_bmp_stack)
			index = 0
			next_tick = time.perf_counter()

			with ImageWindow(fullscreen=True, monitor=self._plm_monitor) as win:
				def _loop_callback() -> None:
					nonlocal index, next_tick
					if self._plm_stop_event.is_set():
						raise EventLoopExit
					now = time.perf_counter()
					if now < next_tick:
						return

					bmp = self._plm_bmp_stack[index]
					if bmp.dtype != np.uint8:
						bmp_u8 = np.clip(bmp, 0, 255).astype(np.uint8)
					else:
						bmp_u8 = bmp
					win.load(Image.fromarray(bmp_u8))
					self.plm_counter_updated.emit(index + 1, total)
					index = (index + 1) % total

					next_tick += self._plm_frame_interval_s
					if now - next_tick > self._plm_frame_interval_s:
						next_tick = now + self._plm_frame_interval_s

				win.loop_callback = _loop_callback
				win.run()
		except Exception as exc:
			status_text = f"PLM loop error: {exc}"
		finally:
			self.plm_loop_finished.emit(status_text)

	@QtCore.pyqtSlot(int)
	def _on_zernike_max_degree_changed(self, _: int) -> None:
		max_degree = int(self.window.zernike_max_degree_combo.currentData() or 0)
		terms = [(n, m) for n in range(max_degree + 1) for m in range(-n, n + 1, 2)]
		self._zernike_terms = {k: self._zernike_terms.get(k, 0.0) for k in terms}
		self.window.zernike_status_label.setText(f"Active Zernike terms: {len(self._zernike_terms)}")

	def _start_zernike_optimizer(self) -> None:
		if self._zernike_optimizer_running:
			return
		self._zernike_target_iterations = int(self.window.zernike_opt_iterations_spin.value())
		self._zernike_step_size = float(self.window.zernike_opt_step_spin.value())
		self._zernike_iterations_done = 0
		self._zernike_optimizer_running = True
		self._zernike_timer.start()
		self.window.zernike_opt_start_button.setEnabled(False)
		self.window.zernike_opt_stop_button.setEnabled(True)
		self.window.zernike_opt_status_label.setText("Optimizer: running")

	def _stop_zernike_optimizer(self) -> None:
		if not self._zernike_optimizer_running:
			self.window.zernike_opt_start_button.setEnabled(True)
			self.window.zernike_opt_stop_button.setEnabled(False)
			return
		self._zernike_optimizer_running = False
		self._zernike_timer.stop()
		self.window.zernike_opt_start_button.setEnabled(True)
		self.window.zernike_opt_stop_button.setEnabled(False)
		self.window.zernike_opt_status_label.setText("Optimizer: stopped")

	def _zernike_optimizer_step(self) -> None:
		if not self._zernike_optimizer_running:
			return
		if self._zernike_iterations_done >= self._zernike_target_iterations:
			self._stop_zernike_optimizer()
			self.window.zernike_opt_status_label.setText("Optimizer: complete")
			return
		if self._zernike_terms:
			phase = self._plot_time
			for idx, key in enumerate(self._zernike_terms.keys()):
				delta = self._zernike_step_size * np.sin(phase + 0.25 * idx)
				self._zernike_terms[key] = float(np.clip(self._zernike_terms[key] + 0.01 * delta, -np.pi, np.pi))
		self._zernike_iterations_done += 1
		self.window.zernike_opt_status_label.setText(
			f"Optimizer: running ({self._zernike_iterations_done}/{self._zernike_target_iterations})"
		)

	def _refresh_pattern_gallery(self) -> None:
		while self.window.pattern_gallery_grid.count():
			item = self.window.pattern_gallery_grid.takeAt(0)
			widget = item.widget()
			if widget is not None:
				widget.deleteLater()
		if self._pattern_stack is None:
			self.window.pattern_gallery_status_label.setText("Load PLM patterns to start gallery capture.")
			return
		total = int(self._pattern_stack.shape[0])
		preview_count = min(12, total)
		for idx in range(preview_count):
			img = np.asarray(self._pattern_stack[idx], dtype=np.float32)
			if img.ndim > 2:
				img = img[..., 0]
			mini = pg.ImageView(view=pg.PlotItem())
			mini.ui.histogram.hide()
			mini.ui.roiBtn.hide()
			mini.ui.menuBtn.hide()
			mini.setColorMap(self.window.perceptual_rainbow_colormap)
			mini.setMinimumSize(180, 130)
			mini.setImage(img, autoLevels=True)
			cell = QtWidgets.QWidget()
			cell_layout = QtWidgets.QVBoxLayout(cell)
			cell_layout.setContentsMargins(2, 2, 2, 2)
			cell_layout.addWidget(QtWidgets.QLabel(f"Pattern {idx + 1}"))
			cell_layout.addWidget(mini)
			self.window.pattern_gallery_grid.addWidget(cell, idx // 3, idx % 3)
		self.window.pattern_gallery_status_label.setText(f"Showing {preview_count} / {total} patterns")

	def _make_phase_intensity_rgb(self, complex_field: np.ndarray) -> np.ndarray:
		phase = np.angle(complex_field)
		phase_norm = ((phase + np.pi) / (2.0 * np.pi)).astype(np.float32)
		rgb_phase = self.window.perceptual_rainbow_colormap.map(phase_norm, mode="float")[:, :, :3]

		intensity = np.abs(complex_field).astype(np.float32)
		p99 = float(np.percentile(intensity, 99.0)) if intensity.size else 0.0
		if p99 > 0.0:
			intensity_norm = np.clip(intensity / p99, 0.0, 1.0)
		else:
			intensity_norm = np.zeros_like(intensity, dtype=np.float32)
		brightness = np.sqrt(intensity_norm)
		rgb = rgb_phase * brightness[:, :, None]
		return (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)

	def _update_recovered_and_channel_matrix(self, fft_shifted: np.ndarray) -> None:
		if fft_shifted.ndim != 2:
			return
		h, w = int(fft_shifted.shape[0]), int(fft_shifted.shape[1])
		x0, y0, x1, y1 = self.window.get_fft_roi_bounds()
		x0 = int(np.clip(x0, 0, max(0, w - 2)))
		y0 = int(np.clip(y0, 0, max(0, h - 2)))
		x1 = int(np.clip(x1, x0 + 2, w))
		y1 = int(np.clip(y1, y0 + 2, h))
		recovered = recover_off_axis_field(fft_shifted, (x0, y0, x1, y1), basic_mode=False)
		if recovered is None:
			self.window.recovered_field_status_label.setText("Recovered field not available.")
			return
		rgb = self._make_phase_intensity_rgb(recovered)
		self.window.set_recovered_field_image(
			rgb,
			status_text="Recovered field updated (phase: hue, intensity: brightness)",
		)
		amp = np.abs(recovered).astype(np.float32)

		metrics = np.array(
			[
				float(np.mean(amp)),
				float(np.std(amp)),
				float(np.max(amp)),
			],
			dtype=np.float32,
		)
		if metrics[2] > 0:
			metrics = metrics / metrics[2]
		self._channel_matrix_history.append(metrics)
		if len(self._channel_matrix_history) > 12:
			self._channel_matrix_history = self._channel_matrix_history[-12:]
		m = np.stack(self._channel_matrix_history, axis=0)
		matrix = m @ m.T
		matrix = np.clip(matrix, 1e-6, None)
		matrix_db = 20.0 * np.log10(matrix / np.max(matrix))
		self.window.channel_matrix_image_item.setImage(matrix_db)
		off_diag = matrix_db[~np.eye(matrix_db.shape[0], dtype=bool)]
		xt_db = float(np.max(off_diag)) if off_diag.size else 0.0
		self.window.channel_matrix_xt_label.setText(f"XT (dB): {xt_db:.2f}")
		self.window.channel_matrix_status_label.setText(f"Channel matrix updated ({matrix_db.shape[0]}x{matrix_db.shape[1]})")

	def _poll_camera_queue(self) -> None:
		latest_frame = None
		while True:
			try:
				latest_frame = self.camera_worker.frame_queue.get_nowait()
			except Exception:
				break

		if latest_frame is None:
			return
		if self._drop_frames_remaining > 0:
			self._drop_frames_remaining -= 1
			self.window.status_label.setText(f"Dropping frame ({self._drop_frames_remaining} remaining)")
			return

		frame = np.ascontiguousarray(latest_frame, dtype=np.float32)
		self.window.frame_received.emit(frame)

		h, w = frame.shape[:2]
		lx0, ly0, lx1, ly1 = self.window.get_live_roi_bounds()
		lx0 = int(np.clip(lx0, 0, max(0, w - 2)))
		ly0 = int(np.clip(ly0, 0, max(0, h - 2)))
		lx1 = int(np.clip(lx1, lx0 + 2, w))
		ly1 = int(np.clip(ly1, ly0 + 2, h))
		live_roi = frame[ly0:ly1, lx0:lx1]
		if live_roi.size:
			self.window.live_roi_intensity_label.setText(f"Live ROI sum intensity: {float(np.sum(live_roi)):.2f}")

		left_tab_idx = int(self.window.tab_widget.currentIndex())
		right_tab_idx = int(self.window.right_tab_widget.currentIndex())
		need_fft = (left_tab_idx == 0) or (right_tab_idx in (0, 2))
		need_heavy = right_tab_idx in (0, 2)

		if need_fft:
			fft_shifted, fft_img_u8 = compute_fft_shifted_and_magnitude_image(frame)
			fft_img = fft_img_u8.astype(np.float32)
			if left_tab_idx == 0:
				self.window.fft_received.emit(fft_img)
			if need_heavy:
				now = time.perf_counter()
				if (now - self._last_heavy_update_ts) >= self._heavy_update_interval_s:
					self._update_recovered_and_channel_matrix(fft_shifted)
					self._last_heavy_update_ts = now

		self._plot_time += 0.005
		mean_intensity = float(np.mean(live_roi)) if live_roi.size else float(np.mean(frame))
		phi1 = float(np.sin(self._plot_time))
		phi2 = float(np.cos(self._plot_time))
		dphi = float((phi1 - phi2) % (2.0 * np.pi))

		self._phase_t.append(self._plot_time)
		self._phi1.append(phi1)
		self._phi2.append(phi2)
		self._dphi.append(dphi)
		self._intensity_t.append(self._plot_time)
		self._intensity.append(mean_intensity)

		if left_tab_idx == 1:
			self.window.phase_plot_received.emit(
				np.asarray(self._phase_t, dtype=np.float32),
				np.asarray(self._phi1, dtype=np.float32),
				np.asarray(self._phi2, dtype=np.float32),
				np.asarray(self._dphi, dtype=np.float32),
			)
		if left_tab_idx == 2:
			self.window.intensity_plot_received.emit(
				np.asarray(self._intensity_t, dtype=np.float32),
				np.asarray(self._intensity, dtype=np.float32),
			)


def main() -> int:
	app = QtWidgets.QApplication(sys.argv)
	window = LiveCameraWindow()
	camera_worker = CameraWorker(queue_maxsize=5)
	controller = AppController(window=window, camera_worker=camera_worker)

	def _cleanup() -> None:
		controller.stop()

	app.aboutToQuit.connect(_cleanup)
	window.show()
	return app.exec_()


if __name__ == "__main__":
	raise SystemExit(main())
