import numpy as np
import torch
from torch.nn import functional as F


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


