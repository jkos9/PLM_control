from __future__ import annotations

import math
from typing import Optional

import numpy as np


def _to_numpy(array, dtype: Optional[np.dtype] = None) -> np.ndarray:
	arr = np.asarray(array)
	if dtype is None:
		return arr
	return np.asarray(arr, dtype=dtype)


def compute_fft_shifted_and_magnitude_image(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	image_np = _to_numpy(image, dtype=np.float32)
	fft_complex = np.fft.fft2(image_np)
	fft_shifted = np.fft.fftshift(fft_complex)
	magnitude_log = np.log1p(np.abs(fft_shifted))
	min_val = float(np.min(magnitude_log))
	max_val = float(np.max(magnitude_log))
	den = max_val - min_val
	if den <= 0.0:
		magnitude_uint8 = np.zeros_like(magnitude_log, dtype=np.uint8)
	else:
		normalized = (magnitude_log - min_val) / den
		magnitude_uint8 = np.clip(normalized * 255.0, 0.0, 255.0).astype(np.uint8)
	return fft_shifted.astype(np.complex64, copy=False), magnitude_uint8


def estimate_subpixel_peak_in_roi(
	magnitude: np.ndarray,
	roi: tuple[int, int, int, int],
	eps: float = 1e-12,
) -> tuple[float, float]:
	magnitude_np = _to_numpy(magnitude, dtype=np.float32)
	x0, y0, x1, y1 = [int(v) for v in roi]
	patch = np.clip(magnitude_np[y0:y1, x0:x1], 0.0, None)
	if patch.size == 0:
		return float((y0 + y1) * 0.5), float((x0 + x1) * 0.5)

	weights = patch
	weight_sum = float(np.sum(weights))
	if weight_sum <= float(eps):
		return float((y0 + y1) * 0.5), float((x0 + x1) * 0.5)

	y_coords = np.arange(y0, y1, dtype=np.float32).reshape(-1, 1)
	x_coords = np.arange(x0, x1, dtype=np.float32).reshape(1, -1)
	py = float(np.sum(y_coords * weights) / weight_sum)
	px = float(np.sum(x_coords * weights) / weight_sum)
	return py, px


def demodulate_to_center_no_ref(
	fft_filtered: np.ndarray,
	peak_y_f: float,
	peak_x_f: float,
) -> np.ndarray:
	fft_filtered_np = _to_numpy(fft_filtered)
	if not np.iscomplexobj(fft_filtered_np):
		fft_filtered_np = fft_filtered_np.astype(np.complex64)

	height, width = fft_filtered_np.shape[-2], fft_filtered_np.shape[-1]
	center_y = (float(height) - 1.0) / 2.0
	center_x = (float(width) - 1.0) / 2.0

	dy = center_y - float(peak_y_f)
	dx = center_x - float(peak_x_f)
	dy_i = int(round(dy))
	dx_i = int(round(dx))
	fft_integer_shifted = np.roll(fft_filtered_np, shift=(dy_i, dx_i), axis=(-2, -1))

	dy_f = float(dy - dy_i)
	dx_f = float(dx - dx_i)
	recovered = np.fft.ifft2(np.fft.ifftshift(fft_integer_shifted, axes=(-2, -1)))

	yy = np.arange(height, dtype=np.float32).reshape(-1, 1)
	xx = np.arange(width, dtype=np.float32).reshape(1, -1)
	phase = -2.0 * math.pi * ((dx_f * xx / float(width)) + (dy_f * yy / float(height)))
	frac_ramp = np.exp(1j * phase).astype(np.complex64)
	return recovered * frac_ramp


def remove_linear_phase_tilt_blind(
	field: np.ndarray,
	amp_thresh: float = 0.2,
	eps: float = 1e-12,
) -> np.ndarray:
	field_np = _to_numpy(field)
	if not np.iscomplexobj(field_np):
		field_np = field_np.astype(np.complex64)

	amp = np.abs(field_np)
	amp_max = float(np.max(amp))
	if amp_max <= float(eps):
		return field_np

	mask = amp > (float(amp_thresh) * amp_max)
	mask_complex_x = mask[:, :-1].astype(field_np.dtype)
	mask_complex_y = mask[:-1, :].astype(field_np.dtype)
	gx_num = np.sum(np.conj(field_np[:, :-1]) * field_np[:, 1:] * mask_complex_x)
	gy_num = np.sum(np.conj(field_np[:-1, :]) * field_np[1:, :] * mask_complex_y)

	gx = float(np.angle(gx_num)) if float(np.abs(gx_num)) > float(eps) else 0.0
	gy = float(np.angle(gy_num)) if float(np.abs(gy_num)) > float(eps) else 0.0

	height, width = field_np.shape[-2], field_np.shape[-1]
	yy = np.arange(height, dtype=np.float32).reshape(-1, 1)
	xx = np.arange(width, dtype=np.float32).reshape(1, -1)
	ramp = np.exp(-1j * (gx * xx + gy * yy)).astype(np.complex64)
	return field_np * ramp


def remove_global_phase_offset_blind(
	field: np.ndarray,
	amp_thresh: float = 0.2,
	eps: float = 1e-12,
) -> np.ndarray:
	field_np = _to_numpy(field)
	if not np.iscomplexobj(field_np):
		field_np = field_np.astype(np.complex64)

	amp = np.abs(field_np)
	amp_max = float(np.max(amp))
	if amp_max <= float(eps):
		return field_np

	mask = amp > (float(amp_thresh) * amp_max)
	if not np.any(mask):
		return field_np

	phase_ref_complex = np.sum(field_np[mask])
	if float(np.abs(phase_ref_complex)) <= float(eps):
		return field_np
	phase_ref = float(np.angle(phase_ref_complex))
	return field_np * np.exp(-1j * phase_ref).astype(np.complex64)


def recover_off_axis_field(
	fft_shifted: np.ndarray,
	roi: tuple[int, int, int, int],
	basic_mode: bool = False,
	amp_thresh: float = 0.2,
	eps: float = 1e-12,
) -> Optional[np.ndarray]:
	fft_shifted_np = _to_numpy(fft_shifted)
	if not np.iscomplexobj(fft_shifted_np):
		fft_shifted_np = fft_shifted_np.astype(np.complex64)

	x0, y0, x1, y1 = [int(v) for v in roi]
	if x1 - x0 < 3 or y1 - y0 < 3:
		return None

	filtered = np.zeros_like(fft_shifted_np)
	filtered[y0:y1, x0:x1] = fft_shifted_np[y0:y1, x0:x1]

	if bool(basic_mode):
		height, width = fft_shifted_np.shape[-2], fft_shifted_np.shape[-1]
		roi_center_x = (x0 + x1) // 2
		roi_center_y = (y0 + y1) // 2
		target_center_x = width // 2
		target_center_y = height // 2
		shift_x = int(target_center_x - roi_center_x)
		shift_y = int(target_center_y - roi_center_y)
		centered = np.roll(filtered, shift=(shift_y, shift_x), axis=(-2, -1))
		return np.fft.ifft2(np.fft.ifftshift(centered, axes=(-2, -1)))

	peak_y_f, peak_x_f = estimate_subpixel_peak_in_roi(np.abs(fft_shifted_np), (x0, y0, x1, y1), eps=eps)
	recovered_field = demodulate_to_center_no_ref(filtered, peak_y_f, peak_x_f)
	recovered_field = remove_linear_phase_tilt_blind(recovered_field, amp_thresh=amp_thresh, eps=eps)
	recovered_field = remove_global_phase_offset_blind(recovered_field, amp_thresh=amp_thresh, eps=eps)
	return recovered_field

