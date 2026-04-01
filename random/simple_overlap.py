


target_modes = np.load('data\lp_modes_camera.npy')

    channel_matrix = torch.zeros((target_modes.shape[0], recovered_output_modes.shape[0]), dtype=target_modes.dtype, device=target_modes.device)
    recovered_output_modes = recovered_output_modes / torch.clamp(torch.sqrt(torch.sum(torch.abs(recovered_output_modes) ** 2, dim=(-2, -1), keepdim=True)), min=1e-12)
    target_modes = target_modes / torch.clamp(torch.sqrt(torch.sum(torch.abs(target_modes) ** 2, dim=(-2, -1), keepdim=True)), min=1e-12)

    # 1. Automatically align and scale based on target statistics
    recovered_output_modes = auto_align_modes(recovered_output_modes, target_modes)

    # 2. Normalize 
    recovered_output_modes = recovered_output_modes / torch.clamp(torch.sqrt(torch.sum(torch.abs(recovered_output_modes) ** 2, dim=(-2, -1), keepdim=True)), min=1e-12)
    target_modes = target_modes / torch.clamp(torch.sqrt(torch.sum(torch.abs(target_modes) ** 2, dim=(-2, -1), keepdim=True)), min=1e-12)

    # 3. Fast Overlap Calculation (Replaces nested loops!)
    T_flat = target_modes.view(target_modes.shape[0], -1)
    R_flat = recovered_output_modes.view(recovered_output_modes.shape[0], -1)
    channel_matrix = torch.matmul(torch.conj(T_flat), R_flat.T)

import torch
import torch.nn.functional as F
def get_spatial_moments(modes):
    """Calculates the center of mass (cx, cy) and standard deviations (std_x, std_y)."""
    N, H, W = modes.shape
    
    # 1. Intensity distribution
    intensity = torch.abs(modes)**2
    p = intensity / (intensity.sum(dim=(-2, -1), keepdim=True) + 1e-12)
    
    # 2. Coordinate grids
    y_grid = torch.linspace(-1, 1, H, device=modes.device)
    x_grid = torch.linspace(-1, 1, W, device=modes.device)
    Y, X = torch.meshgrid(y_grid, x_grid, indexing='ij')
    
    # 3. First moments (Center of Mass)
    cx = (p * X).sum(dim=(-2, -1), keepdim=True)
    cy = (p * Y).sum(dim=(-2, -1), keepdim=True)
    
    # 4. Second central moments (Variance -> Standard Deviation)
    var_x = (p * (X - cx)**2).sum(dim=(-2, -1))
    var_y = (p * (Y - cy)**2).sum(dim=(-2, -1))
    
    std_x = torch.sqrt(var_x + 1e-12)
    std_y = torch.sqrt(var_y + 1e-12)
    
    return cx.squeeze(-1).squeeze(-1), cy.squeeze(-1).squeeze(-1), std_x, std_y

def auto_align_modes(recovered_modes, target_modes):
    """
    Centers and scales ALL recovered_modes globally, based entirely on 
    the spatial statistics of the fundamental mode (index 0).
    """
    N_rec, H, W = recovered_modes.shape
    
    # 1. Isolate ONLY the fundamental mode (index 0) for calculations
    # Using [0:1] keeps the batch dimension so get_spatial_moments works
    rec_fund = recovered_modes[0:1] 
    tar_fund = target_modes[0:1]
    
    # 2. Get spatial stats for the fundamental modes only
    r_cx, r_cy, r_std_x, r_std_y = get_spatial_moments(rec_fund)
    t_cx, t_cy, t_std_x, t_std_y = get_spatial_moments(tar_fund)
    
    # 3. Calculate Global Scale Factors (extracting the scalar value from index 0)
    scale_x = (r_std_x / t_std_x)[0]
    scale_y = (r_std_y / t_std_y)[0]

    # 4. Calculate Global Shifts
    # grid_sample maps [Destination] -> [Source] using: Source = Scale * Dest + Shift
    # We want the target's center (Dest) to map exactly to the recovered center (Source)
    shift_x = r_cx[0] - scale_x * t_cx[0]
    shift_y = r_cy[0] - scale_y * t_cy[0]

    # 5. Build a UNIFORM Affine Transformation Matrix for ALL modes
    theta = torch.zeros((N_rec, 2, 3), device=recovered_modes.device)
    
    # Broadcast the exact same scale and shift across the entire N_rec batch
    theta[:, 0, 0] = scale_x      # Global Auto-Scale X
    theta[:, 1, 1] = scale_y      # Global Auto-Scale Y
    theta[:, 0, 2] = shift_x      # Global Translate X
    theta[:, 1, 2] = shift_y      # Global Translate Y

    # Handle Complex vs Real Tensors
    is_complex = torch.is_complex(recovered_modes)
    if is_complex:
        modes_formatted = torch.view_as_real(recovered_modes).permute(0, 3, 1, 2)
    else:
        modes_formatted = recovered_modes.unsqueeze(1)

    # Apply Transformation globally
    grid = F.affine_grid(theta, modes_formatted.size(), align_corners=False)
    aligned = F.grid_sample(modes_formatted, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

    # Revert to original format
    if is_complex:
        return torch.view_as_complex(aligned.permute(0, 2, 3, 1).contiguous())
    else:
        return aligned.squeeze(1)









