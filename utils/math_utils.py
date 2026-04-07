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

