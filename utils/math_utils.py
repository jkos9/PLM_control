import numpy as np
import torch
from typing import Optional
import matplotlib as mpl


def principal_phase_diff(value1: complex, value2: complex) -> float:
    return float(np.angle(value1 * np.conj(value2)))
    # return float(np.angle(value1) - np.angle(value2))


def wrap_phase_to_pi(phase_rad: float) -> float:
    return float((phase_rad + np.pi) % (2.0 * np.pi) - np.pi)


def wrap_phase_to_2pi(phase_rad: float) -> float:
    return float((phase_rad + 2.0 * np.pi) % (2.0 * np.pi))

def wrap_phase_array_to_pi(phase: np.ndarray) -> np.ndarray:
    return np.asarray((phase + np.pi) % (2.0 * np.pi) - np.pi, dtype=np.float32)

def normalize_modes_np(modes: np.ndarray) -> np.ndarray:
    norm = np.sqrt(np.sum(np.abs(modes) ** 2, axis=(-2, -1), keepdims=True))
    norm = np.maximum(norm, 1e-12)
    return modes / norm

def normalize_modes_torch(modes):
    norm = torch.sqrt(torch.sum(torch.abs(modes) ** 2, dim=(-2, -1), keepdim=True))
    norm = torch.clamp(norm, min=1e-12)
    return modes / norm


def count_complete_groups(total_elements):
    total = int(max(0, total_elements))
    complete_groups = 0
    group_size_needed = 1

    while total >= group_size_needed:
        complete_groups += 1
        total -= group_size_needed
        group_size_needed += 1

    return complete_groups


def generate_number_triangle(n):
    result = []
    current_num = 0

    for row_length in range(1, int(max(0, n)) + 1):
        row = []
        for _ in range(row_length):
            row.append(current_num)
            current_num += 1
        result.append(row)

    return result


def measure_xt_db_diagonal(matrix):
    if matrix is None:
        return None
    matrix_t = matrix if torch.is_tensor(matrix) else torch.as_tensor(matrix)

    if matrix_t.ndim != 2 or matrix_t.shape[0] <= 0 or matrix_t.shape[1] <= 0:
        return None

    num_modes = int(min(matrix_t.shape[0], matrix_t.shape[1]))

    if num_modes <= 0:
        return None

    matrix_t = matrix_t[:num_modes, :num_modes]

    # Use power-domain coupling for a scalar XT metric.
    power = torch.abs(matrix_t) ** 2
    row_power = torch.sum(power, dim=1)
    diag_power = torch.diagonal(power)
    off_diag_power = torch.clamp(row_power - diag_power, min=0.0)

    eps = 1e-12
    xt_linear_per_row = off_diag_power / torch.clamp(diag_power, min=eps)
    xt_linear = torch.mean(xt_linear_per_row)

    xt_db = 10.0 * torch.log10(torch.clamp(xt_linear, min=eps))

    return float(xt_db.item())


def measure_xt_db_mode_group(matrix, mode_group_indices, one_based=False):
    if matrix is None:
        return None

    matrix_t = matrix if torch.is_tensor(matrix) else torch.as_tensor(matrix)
    p = torch.abs(matrix_t) ** 2

    if p.ndim != 2 or p.shape[0] <= 0 or p.shape[1] <= 0:
        return None

    num_modes = int(min(p.shape[0], p.shape[1]))

    if num_modes <= 0:
        return None

    p = p[:num_modes, :num_modes]

    mg_xt_values = []

    for group in mode_group_indices:
        if group is None or len(group) <= 0:
            continue

        group_indices = [int(idx) - 1 if one_based else int(idx) for idx in group]
        group_indices = [idx for idx in group_indices if 0 <= idx < num_modes]

        if len(group_indices) <= 0:
            continue

        group_tensor = torch.tensor(group_indices, dtype=torch.long, device=p.device)

        signal_power = torch.sum(p[group_tensor][:, group_tensor])

        mask = torch.ones(num_modes, dtype=torch.bool, device=p.device)
        mask[group_tensor] = False
        outside_indices = torch.nonzero(mask, as_tuple=False).flatten()

        if outside_indices.numel() <= 0:
            xt_power = torch.zeros((), dtype=p.real.dtype, device=p.device)
        else:
            xt_power = torch.sum(p[outside_indices][:, group_tensor])

        xt_group = xt_power / torch.clamp(signal_power, min=1e-12)
        mg_xt_values.append(xt_group)

    if len(mg_xt_values) <= 0:
        return None

    mg_xt_mean = torch.mean(torch.stack(mg_xt_values))
    return float(10.0 * torch.log10(torch.clamp(mg_xt_mean, min=1e-12)).item())


def measure_in_group_power(matrix, mode_group_indices, one_based=False, normalize=True, return_as_loss=True):
    if matrix is None:
        return None

    matrix_t = matrix if torch.is_tensor(matrix) else torch.as_tensor(matrix)
    
    # Calculate power matrix
    p = torch.abs(matrix_t) ** 2

    if p.ndim != 2 or p.shape[0] <= 0 or p.shape[1] <= 0:
        return None

    num_modes = int(min(p.shape[0], p.shape[1]))

    if num_modes <= 0:
        return None

    p = p[:num_modes, :num_modes]

    total_signal_power = torch.zeros((), dtype=p.real.dtype, device=p.device)

    for group in mode_group_indices:
        if group is None or len(group) <= 0:
            continue

        group_indices = [int(idx) - 1 if one_based else int(idx) for idx in group]
        group_indices = [idx for idx in group_indices if 0 <= idx < num_modes]

        if len(group_indices) <= 0:
            continue

        group_tensor = torch.tensor(group_indices, dtype=torch.long, device=p.device)

        # Sum the power within this specific mode group block 
        # (rows = launched modes in group, cols = received modes in group)
        signal_power = torch.sum(p[group_tensor][:, group_tensor])
        total_signal_power += signal_power

    # Normalize against the total power in the matrix to get efficiency
    if normalize:
        total_matrix_power = torch.sum(p)
        power_metric = total_signal_power / torch.clamp(total_matrix_power, min=1e-12)
    else:
        power_metric = total_signal_power

    # Optimizers minimize the loss metric. 
    # If normalizing, we minimize (1.0 - ratio). If raw power, we minimize (-power).
    if return_as_loss:
        if normalize:
            return float((1.0 - power_metric).item())
        else:
            return -float(power_metric.item())

    return float(power_metric.item())


def measure_xt_db(matrix):
    # Backward-compatible default metric.
    return measure_xt_db_diagonal(matrix)


def cv_normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    min_val = float(np.min(image))
    max_val = float(np.max(image))
    if max_val <= min_val:
        return np.zeros_like(image, dtype=np.uint8)
    normalized = (image - min_val) / (max_val - min_val)
    return (normalized * 255.0).astype(np.uint8)


_CMAP_LUTS = {}

def phase_to_rgb_uint8(
    phase: np.ndarray,
    value: Optional[np.ndarray] = None,
    cmap_name: str = "viridis",
) -> np.ndarray:
    
    # 1. Fetch or build the LUT once
    if cmap_name not in _CMAP_LUTS:
        # Precompute 256 RGB values
        cmap = mpl.colormaps[cmap_name]
        lut = cmap(np.linspace(0, 1, 256))[:, :3] * 255.0
        _CMAP_LUTS[cmap_name] = lut.astype(np.uint8)
        
    lut = _CMAP_LUTS[cmap_name]
    
    # 2. Fast mapping: Normalize phase to 0-255 integer indices
    # (phase + pi) / 2pi -> mapped to 0-255
    phase_norm = np.clip((phase + np.pi) / (2.0 * np.pi) * 255.0, 0, 255).astype(np.uint8)
    
    # 3. Fast indexing (Replaces the slow Matplotlib call)
    rgb = lut[phase_norm]
    
    # 4. Apply amplitude weighting
    if value is not None:
        v = np.clip(value.astype(np.float32), 0.0, 1.0)
        # Fast broadcasting
        rgb = (rgb.astype(np.float32) * v[..., None]).astype(np.uint8)
        
    return rgb


_CMAP_LUTS_TORCH = {}

def phase_to_rgb_uint8_torch(
    phase: torch.Tensor,
    value: Optional[torch.Tensor] = None,
    cmap_name: str = "viridis",
) -> torch.Tensor:
    
    device = phase.device
    lut_key = (cmap_name, device)
    
    # 1. Fetch or build the LUT once per colormap/device combination
    if lut_key not in _CMAP_LUTS_TORCH:
        # We still use matplotlib/numpy to generate the raw color data
        cmap = mpl.colormaps[cmap_name]
        lut_np = cmap(np.linspace(0, 1, 256))[:, :3] * 255.0
        
        # Convert to a PyTorch tensor and push it to the correct device immediately
        _CMAP_LUTS_TORCH[lut_key] = torch.from_numpy(lut_np).to(
            dtype=torch.uint8, device=device
        )
        
    lut = _CMAP_LUTS_TORCH[lut_key]
    
    # 2. Fast mapping: Normalize phase to 0-255 integer indices
    # (phase + pi) / 2pi -> mapped to 0-255
    phase_norm_float = (phase + torch.pi) / (2.0 * torch.pi) * 255.0
    
    # PyTorch requires indices to be long (int64) for array indexing
    phase_norm = torch.clamp(phase_norm_float, min=0.0, max=255.0).to(torch.long)
    
    # 3. Fast indexing
    rgb = lut[phase_norm]
    
    # 4. Apply amplitude weighting
    if value is not None:
        v = torch.clamp(value.to(torch.float32), min=0.0, max=1.0)
        
        # .unsqueeze(-1) replaces NumPy's [..., None] for broadcasting
        rgb = (rgb.to(torch.float32) * v.unsqueeze(-1)).to(torch.uint8)
        
    return rgb

