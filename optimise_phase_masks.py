import torch
import math
from utils.channel_utils import gen_coupling_matrices, interpolate_xt_tensor, principal_mode_tensors
from utils.init_mode_utils import generate_basis_hg_torch, generate_basis_lg_torch, transfer_function_of_free_space_torch_angle, single_gaussian_profile
from utils.channel_utils import modewise_coupling_matrix
from utils.sim_config_utils import get_sim_config
import matplotlib.pyplot as plt
from pathlib import Path
from utils.launcher_utils import mode_launcher
import torch.optim as optim
import torch.nn.functional as F
import sys
from tqdm import tqdm
import numpy as np



def get_config():
    return get_sim_config()


def setup_coordinates(config):
    nx, ny = config["nx"], config["ny"]
    dx = config["pixel_size_x"]
    dy = config["pixel_size_y"]
    device = config["device"]

    x_vec = (np.arange(1, ny + 1) - (ny / 2 + 0.5)) * dx
    y_vec = (np.arange(1, nx + 1) - (nx / 2 + 0.5)) * dy
    x_grid, y_grid = np.meshgrid(x_vec, y_vec)

    theta = np.arctan2(y_grid, x_grid)
    radius = np.hypot(x_grid, y_grid)
    x_rot = x_grid
    y_rot = y_grid

    x_rot_t = torch.tensor(x_rot, dtype=torch.float32, device=device)
    y_rot_t = torch.tensor(y_rot, dtype=torch.float32, device=device)
    theta_t = torch.tensor(theta, dtype=torch.float32, device=device)
    radius_t = torch.tensor(radius, dtype=torch.float32, device=device)
    return x_rot_t, y_rot_t, theta_t, radius_t






def compute_mode_correlation_vs_wavelength(output_modes, lp_modes, eps=1e-12):
    if output_modes.ndim != 4:
        raise ValueError("output_modes must be [wavelength, mode, x, y]")
    if lp_modes.ndim != 3:
        raise ValueError("lp_modes must be [mode, x, y]")
    if output_modes.shape[1:] != lp_modes.shape:
        raise ValueError("output_modes shape [mode, x, y] must match lp_modes shape")

    w_count, mode_count, _, _ = output_modes.shape
    out_flat = output_modes.reshape(w_count, mode_count, -1)
    lp_flat = lp_modes.reshape(mode_count, -1)

    numer = torch.abs(torch.sum(torch.conj(lp_flat)[None, :, :] * out_flat, dim=-1)) ** 2
    lp_pow = torch.sum(torch.abs(lp_flat) ** 2, dim=-1)
    out_pow = torch.sum(torch.abs(out_flat) ** 2, dim=-1)
    corr = numer / torch.clamp(lp_pow[None, :] * out_pow, min=eps)

    corr_db = 10.0 * torch.log10(torch.clamp(corr, min=eps))
    return corr, corr_db


def phase_smoothness_loss(phase):
    """
    Compute a smoothing loss over a complex phase parameter.
    Penalizes large spatial gradients.
    Args:        phase (torch.Tensor): shape (Nx, Ny), dtype complex
    Returns:
        torch.Tensor: scalar loss
    """
    phase_angle = torch.angle(phase)  # Extract phase
    dx = torch.abs(phase_angle[:, 1:] - phase_angle[:, :-1])
    dy = torch.abs(phase_angle[1:, :] - phase_angle[:-1, :])
    return torch.mean(dx**2) + torch.mean(dy**2)


def gaussian_kernel(kernel_size, sigma):
    """
    Create a 2D Gaussian kernel.
    """
    k = kernel_size // 2
    x = torch.arange(-k, k + 1, dtype=torch.float32)
    y = torch.arange(-k, k + 1, dtype=torch.float32)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / torch.sum(kernel)
    # Reshape kernel for convolution: [out_channels, in_channels, H, W]
    return kernel.unsqueeze(0).unsqueeze(0)


def smooth_complex_mask(mask, kernel):
    """
    Smooth a complex-valued mask by convolving the phase and preserving magnitude.
    """
    pad = kernel.shape[-1] // 2

    if mask.ndim == 2:
        phase = torch.angle(mask).unsqueeze(0).unsqueeze(0)
        smoothed_phase = F.conv2d(phase, kernel, padding=pad).squeeze(0).squeeze(0)
        return torch.exp(1j * smoothed_phase)

    if mask.ndim == 3:
        phase = torch.angle(mask).unsqueeze(1)  # [modes, 1, Nx, Ny]
        smoothed_phase = F.conv2d(phase, kernel, padding=pad).squeeze(1)
        return torch.exp(1j * smoothed_phase)

    raise ValueError("mask must be [Nx, Ny] or [modes, Nx, Ny]")


if __name__ == "__main__":
    config = get_config()
    if config["max_mg"] < 1:
        raise ValueError("config['max_mg'] must be >= 1")

    x_rot_t, y_rot_t, theta_t, radius_t = setup_coordinates(config)
    H0 = transfer_function_of_free_space_torch_angle(
        x_rot_t,
        y_rot_t,
        config['plane_spacing'],
        config['lambda_'],
        theta_x=config['prop_tilt_x_deg'] * (np.pi / 180.0),
        theta_y=config['prop_tilt_y_deg'] * (np.pi / 180.0),
        recenter=config['prop_recenter'],
    )
    maxR = torch.max(radius_t)
    H = H0 * (radius_t < (config['k_space_filter'] * maxR))
    
    H = H.to(dtype=torch.complex64)
    H0 = H0.to(dtype=torch.complex64)


    device = config["device"]
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Config requests CUDA but no CUDA device was found.")
        device_index = 0 if device.index is None else device.index
        if device_index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Config requests cuda:{device_index} but only {torch.cuda.device_count()} CUDA device(s) are available."
            )
        device = torch.device(f"cuda:{device_index}")
    print(f"Using device: {device}")

    
    pics_dir = Path("pics/computational")
    pics_dir.mkdir(parents=True, exist_ok=True)

    wavelengths = torch.arange(1530e-9, 1575e-9, 1e-9, device=device)  # Wavelengths across the C-band
    target_wavelength = 1550e-9
    
    lp_modes = torch.load("data/lp_modes.pt", map_location="cpu").to(device)
    if lp_modes.ndim != 3:
        raise ValueError("Expected lp_modes.pt tensor with shape [modes, nx, ny]")

    modes = int(lp_modes.shape[0])
    launcher = mode_launcher(layers=1, Nx=lp_modes.shape[1], Ny=lp_modes.shape[2], n_modes=modes, H=H).to(device)
    iterations = 2000
    learning_rate = 0.01
    input_field = torch.load("data/input_gaussian_profile.pt", map_location="cpu").to(device).to(torch.cfloat)
    target_fields = lp_modes.to(torch.cfloat)

    print(target_fields.shape)  # Should be [modes, nx, ny]
    print(input_field.shape)  # Should be [nx, ny] or [modes, nx, ny]

    if input_field.ndim == 2:
        input_fields = input_field.unsqueeze(0).repeat(modes, 1, 1)
    elif input_field.ndim == 3 and input_field.shape[0] == modes:
        input_fields = input_field
    else:
        raise ValueError("input_gaussian_profile.pt must be [Nx, Ny] or [modes, Nx, Ny]")

    optimizer = optim.Adam(launcher.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=100,
        threshold=1e-5,
        min_lr=1e-6,
    )
    smooth_kernel = gaussian_kernel(
        kernel_size=config["smoothing_kernel_size"],
        sigma=config["smoothing_sigma"],
    ).to(device)

    smooth_weight = 0.01

    # apply roll in x direction of target fields to shift away from 0 diffraction order
    #target_fields = torch.roll(target_fields, shifts=50, dims=2)
    '''
    plt.figure()
    plt.imshow(torch.abs(target_fields[0]).cpu().numpy(), cmap="viridis")
    plt.title("Example target LP mode (rolled to shift away from 0 order)") 
    plt.colorbar()
    plt.show()
    sys.exit()
    '''


    losses = []
    for iter in tqdm(range(iterations)):
        optimizer.zero_grad()
        output_fields = launcher(input_fields)

        smooth_loss = torch.tensor(0.0, device=device)
        if config["apply_smoothing"]:
            current_mask = torch.exp(1j * launcher.phase_layer_1.phase)
            smoothed_mask = smooth_complex_mask(current_mask, smooth_kernel)
            smooth_loss = torch.mean(torch.abs(current_mask - smoothed_mask) ** 2)


        coupling_amp, coupling_power = modewise_coupling_matrix(output_fields, target_fields, return_power=True)
        diag_power = torch.diagonal(coupling_power, dim1=-2, dim2=-1)
        diag_mean = torch.mean(diag_power)
        offdiag_mask = 1.0 - torch.eye(coupling_power.shape[-1], device=device, dtype=coupling_power.dtype)
        offdiag_sum = torch.sum(coupling_power * offdiag_mask)
        offdiag_count = torch.clamp(torch.sum(offdiag_mask), min=1.0)
        offdiag_mean = offdiag_sum / offdiag_count
        coupling_loss = (1.0 - diag_mean) + offdiag_mean

        loss = coupling_loss + smooth_weight * smooth_loss 

        loss.backward()
        optimizer.step()
        #scheduler.step(loss.item())
        losses.append(loss.item())

    # plot the output modes after training
    output_fields = launcher(input_fields).detach().cpu()

    phase_masks = torch.remainder(launcher.phase_layer_1.phase.detach(), 2 * math.pi).to(torch.float32).cpu()
    torch.save(phase_masks, "data/mode_launcher_phase_masks.pt")
    torch.save(launcher.state_dict(), "data/mode_launcher_state_dict.pt")
    torch.save(output_fields, "data/mode_launcher_output_fields.pt")


    print("Saved trained phase masks to data/mode_launcher_phase_masks.pt (real, wrapped to [0, 2pi))")

    plt.figure(figsize=(8, 5))
    plt.plot(losses, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("Training Loss Curve for Mode Launcher")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(pics_dir / "mode_launcher_training_loss.png", dpi=300)
    plt.close()



    n_cols = int(math.ceil(math.sqrt(modes)))
    n_rows = int(math.ceil(modes / n_cols))
    plt.figure(figsize=(16, 16))
    for mode_i in range(modes):
        plt.subplot(n_rows, n_cols, mode_i + 1)
        plt.imshow((torch.abs(output_fields[mode_i])**2).detach().cpu().numpy(), cmap="viridis", origin="lower", aspect="equal")
        plt.colorbar()
        plt.title(f"Output mode {mode_i} after training")
        plt.xlabel("x (pixel)")
        plt.ylabel("y (pixel)")

    plt.tight_layout()
    plt.savefig(pics_dir / "mode_launcher_output_modes_after_training.png", dpi=300)  

    plt.figure(figsize=(12, 6))
    for mode_i in range(10):
        plt.subplot(2, 5, mode_i + 1)
        plt.imshow((torch.abs(target_fields[mode_i])**2).detach().cpu().numpy(), cmap="viridis", origin="lower", aspect="equal")
        plt.colorbar()
        plt.title(f"Target LP mode {mode_i}")
        plt.xlabel("x (pixel)")
        plt.ylabel("y (pixel)")

    plt.tight_layout()
    plt.savefig(pics_dir / "lp_modes.png")



