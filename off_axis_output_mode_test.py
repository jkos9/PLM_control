import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch.nn.functional as F
from pathlib import Path

from utils.channel_utils import _infer_mode_group_sizes, gen_coupling_matrices, interpolate_xt_tensor, principal_mode_tensors
from utils.channel_utils import modewise_coupling_matrix
from utils.optics_utils import propagate_t
from utils.init_mode_utils import generate_basis_hg_torch, generate_basis_lg_torch, transfer_function_of_free_space_torch_angle, single_gaussian_profile
from utils.channel_utils import _infer_mode_group_sizes, _build_mode_group_mask

from utils.off_axis_utils import shift_complex_fields_subpixel, estimate_shift_to_center_from_intensity, simulate_channel_measurement

import torch
import torch.fft

import sys


def get_config():
    config = {
        "device": torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
        "max_mg": 5,
        "lambda_": 1550e-9,
        "downsample": 1,
        "plane_spacing": 0.1,
        "array_dist_to_first_plane": 5e-3,
        "plane_count": 8,
        "mf_din": 65e-6,
        "mf_dout": 300e-6,
        "k_space_filter": 0.1,
        "symmetric_masks": False,
        "iteration_count": 500,
        "boost": 1,
        "apply_smoothing": True,
        "smoothing_kernel_size": 3,
        "smoothing_sigma": 0.55,
        "learning_rate": 3e-1,
        "prop_tilt_x_deg": 0.0,
        "prop_tilt_y_deg": 0.0,
        "prop_recenter": True,
    }

    config["pixel_size_x"] = config["downsample"] * 4e-6
    config["pixel_size_y"] = config["downsample"] * 4e-6
    config["mode_count"] = int(np.sum(np.arange(1, config["max_mg"] + 1)))


    desired_size_m_x = 2e-3
    desired_size_m_y = 2e-3
    config["nx"] = int(desired_size_m_x / config["pixel_size_x"])
    config["ny"] = int(desired_size_m_y / config["pixel_size_y"])

    config["mask_offset"] = 1e-9
    return config


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
    x_rot = radius * np.cos(theta - np.pi / 4)
    y_rot = radius * np.sin(theta - np.pi / 4)

    x_rot_t = torch.tensor(x_rot, dtype=torch.float32, device=device)
    y_rot_t = torch.tensor(y_rot, dtype=torch.float32, device=device)
    theta_t = torch.tensor(theta, dtype=torch.float32, device=device)
    radius_t = torch.tensor(radius, dtype=torch.float32, device=device)
    return x_rot_t, y_rot_t, theta_t, radius_t


def normalize_modes(m, eps=1e-12):
    # m: [modes,x,y] or [x,y]
    if m.ndim == 2:
        n = torch.sqrt(torch.clamp(torch.sum(torch.abs(m) ** 2), min=eps))
        return m / n
    n = torch.sqrt(torch.clamp(torch.sum(torch.abs(m) ** 2, dim=(1, 2), keepdim=True), min=eps))
    return m / n


def amp_phase_to_rgb(field, amp_gamma=0.7, eps=1e-12):
    amp = torch.abs(field)
    amp = amp / torch.clamp(torch.max(amp), min=eps)
    amp = torch.clamp(amp, 0.0, 1.0) ** amp_gamma

    phase = torch.angle(field)
    hue = (phase + np.pi) / (2 * np.pi)

    hsv = np.stack(
        [
            hue.detach().cpu().numpy(),
            np.ones_like(hue.detach().cpu().numpy()),
            amp.detach().cpu().numpy(),
        ],
        axis=-1,
    )
    return mcolors.hsv_to_rgb(hsv)



if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. No CUDA device found.")
    if torch.cuda.device_count() < 2:
        raise RuntimeError("Requested cuda:1 but fewer than 2 CUDA devices are available.")
    device = torch.device("cuda:1")
    print(f"Using device: {device}")

    launched_modes = torch.load("data/mode_launcher_output_fields.pt", map_location=device).to(device)
    lp_modes = torch.load("data/lp_modes.pt", map_location=device).to(device)

    coupling_amp, coupling_power = modewise_coupling_matrix(launched_modes, lp_modes, return_power=True)
    coupling_db = 10.0 * torch.log10(torch.clamp(coupling_power, min=1e-12))

    pics_dir = Path("pics/computational")
    pics_dir.mkdir(parents=True, exist_ok=True)


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


    H = H.to(device=device, dtype=torch.complex64)
    H0 = H0.to(device=device, dtype=torch.complex64)

    
    pics_dir = Path("pics/computational")
    pics_dir.mkdir(parents=True, exist_ok=True)

    wavelengths = torch.arange(1530e-9, 1575e-9, 1e-9, device=device)  # Wavelengths across the C-band
    target_wavelength = 1550e-9

    '''
    complex_coupling = modewise_coupling_matrix(launched_modes, lp_modes, return_power=False)
    modes = int(lp_modes.shape[0])  # Number of modes in the system
    xt_value = 10

    xt_matrix_1550 = gen_coupling_matrices(modes, xt_value, mode_group_only=True)
    '''


    eps = 1e-12
    modes = int(lp_modes.shape[0])  # Number of modes in the system
    xt_value = 5


    lp_modes = normalize_modes(lp_modes, eps).to(device)
    launched_modes = normalize_modes(launched_modes, eps).to(device)

    # 1) Input coupling coefficients a_kj = <LP_k, launched_j>
    #lp_flat = lp_modes.reshape(lp_modes.shape[0], -1)               # [n_lp, p]
    #in_flat = launched_modes.reshape(launched_modes.shape[0], -1)   # [n_launch, p]
    #a = torch.einsum("kp,jp->kj", torch.conj(lp_flat), in_flat)     # [n_lp, n_launch]


    a = torch.zeros((lp_modes.shape[0], lp_modes.shape[0]), dtype=lp_modes.dtype, device=lp_modes.device)
    lp_modes = lp_modes / torch.clamp(torch.sqrt(torch.sum(torch.abs(lp_modes) ** 2, dim=(-2, -1), keepdim=True)), min=1e-12)
    launched_modes = launched_modes / torch.clamp(torch.sqrt(torch.sum(torch.abs(launched_modes) ** 2, dim=(-2, -1), keepdim=True)), min=1e-12)

    for i in range(lp_modes.shape[0]):
        for j in range(lp_modes.shape[0]):
            a[i, j] = torch.sum(torch.conj(lp_modes[i]) * launched_modes[j])

    xt_matrix_1550 = gen_coupling_matrices(modes, xt_value, mode_group_only=True).to(device)
    
    # 2) Propagate coefficients through MMF
    b = xt_matrix_1550 #@ a  # [n_lp, n_launch]
    # we are getting some residual phases from from the input coupling.
    # This is expected since the launched modes are not perfectly orthogonal to the LP modes and have some phase structure, so the coupling coefficients a_kj are complex and can introduce phase variations across the launched modes.
    # so to compare the measurement we just use the ideal channel matrix as the "ground truth" for the output field phases, since we are not trying to test the input coupling in this experiment, just the output mode profiles and their measurement.

    E_out_channel = torch.einsum("ij, ixy -> jxy", a, lp_modes)
    E_out_fiber_channel = torch.einsum("ij, ixy -> jxy", b, lp_modes)


    #E_out_channel = propagate_t(E_out_channel, H0)
    #E_out_fiber_channel = propagate_t(E_out_fiber_channel, H0)
    #E_out_fiber_channel = propagate_t(E_out_fiber_channel, H0)

    '''
    simulated_shift_y_px = 40.0
    simulated_shift_x_px = -45.0
    E_out_fiber_channel = shift_complex_fields_subpixel(
        E_out_fiber_channel,
        shift_y_px=simulated_shift_y_px,
        shift_x_px=simulated_shift_x_px,
        mode="bilinear",
        padding_mode="zeros",
    )

    # Optional: Additive noise to simulate measurement imperfections
    noise_amp = 0.0 * torch.max(torch.abs(E_out_fiber_channel))
    noise = noise_amp * (torch.randn_like(E_out_fiber_channel) + 1j * torch.randn_like(E_out_fiber_channel))
    E_out_fiber_channel_noisy = E_out_fiber_channel + noise

    # shift back the noisy field to simulate misalignment correction errors
    est_shift_y, est_shift_x = estimate_shift_to_center_from_intensity(
        E_out_fiber_channel_noisy,
        percentile=90.0,
    )

    correction_error_std_px = 0.75
    residual_shift_y = correction_error_std_px * torch.randn((), device=device)
    residual_shift_x = correction_error_std_px * torch.randn((), device=device)

    applied_shift_y = est_shift_y + residual_shift_y
    applied_shift_x = est_shift_x + residual_shift_x

    E_out_fiber_channel_corrected = shift_complex_fields_subpixel(
        E_out_fiber_channel_noisy,
        shift_y_px=float(applied_shift_y.item()),
        shift_x_px=float(applied_shift_x.item()),
        mode="bilinear",
        padding_mode="zeros",
    )

    print(
        "Blind recenter estimate "
        f"(dy={float(est_shift_y.item()):.2f}, dx={float(est_shift_x.item()):.2f}) px, "
        f"applied with residual "
        f"(dy={float(residual_shift_y.item()):.2f}, dx={float(residual_shift_x.item()):.2f}) px"
    )

    # 3) Reconstruct output fields
    #E_out_launch = torch.einsum("wij,ixy->wjxy", a, lp_modes)              # [n_w, n_launch, x, y]


    print(reference_wave.shape)
    print(E_out_fiber_channel_corrected.shape)


    # plot combined amplitude+phase (HSV: hue=phase, value=amplitude) for first 5 output modes after fiber
    plt.figure(figsize=(16, 4))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        rgb = amp_phase_to_rgb(E_out_fiber_channel_corrected[i])
        plt.imshow(rgb)
        plt.title(f"Output mode {i}\n(hue=phase, value=amp)")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(pics_dir / "output_modes_after_fiber.png", dpi=300)
    plt.close()
    '''

    reference_wave = torch.load("data/reference_wave.pt", map_location=device).to(device)
    intensity, recovered_output_modes, channel_matrix = simulate_channel_measurement(E_out_fiber_channel, reference_wave, lp_modes)


    # compare the phase of the recovered channel matrix to the original
    # weighted complex alignment: high-amplitude elements dominate the phase convention
    align_weights = torch.abs(xt_matrix_1550) ** 2
    numerator = torch.sum(align_weights * xt_matrix_1550 * torch.conj(channel_matrix))
    denominator = torch.clamp(torch.sum(align_weights * torch.abs(channel_matrix) ** 2), min=1e-12)
    align_factor = numerator / denominator
    align_phase = align_factor / torch.clamp(torch.abs(align_factor), min=1e-12)
    channel_matrix_aligned = channel_matrix * align_phase

    # print the mean absolute error in coupling amplitude and phase between the recovered and original channel matrices
    amp_error = torch.abs(torch.abs(channel_matrix) - torch.abs(xt_matrix_1550))
    phase_delta = torch.angle(torch.exp(1j * (torch.angle(channel_matrix_aligned) - torch.angle(xt_matrix_1550))))
    phase_error = torch.abs(phase_delta)

    # Prioritize high-amplitude couplings with a normalized weighted mean.
    phase_weights = torch.abs(xt_matrix_1550) ** 2
    weighted_phase_error = phase_error * phase_weights
    mean_phase_error_weighted = weighted_phase_error.sum() / torch.clamp(phase_weights.sum(), min=1e-12)


    #n = xt_matrix_1550.shape[0]
    #sizes =  _infer_mode_group_sizes(n)
    #group_mask = _build_mode_group_mask(n, sizes, device=phase_error.device).to(torch.float32)
    #phase_error = phase_error * group_mask

    print(f"Mean absolute amplitude error: {amp_error.mean().item():.4f}")
    print(f"Mean absolute phase error (amplitude-weighted): {mean_phase_error_weighted.item():.4f} radians")


    # plot combined amplitude+phase (HSV: hue=phase, value=amplitude) for the 5 recovered output modes
    plt.figure(figsize=(16, 4))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        rgb = amp_phase_to_rgb(E_out_fiber_channel[i])
        plt.imshow(rgb)
        plt.title(f"Output mode {i}\n(hue=phase, value=amp)")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(pics_dir / "e_out_fiber_channel.png", dpi=300)
    plt.close()


    # plot combined amplitude+phase (HSV: hue=phase, value=amplitude) for the 5 recovered output modes
    plt.figure(figsize=(16, 4))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        rgb = amp_phase_to_rgb(lp_modes[i])
        plt.imshow(rgb)
        plt.title(f"LP mode {i}\n(hue=phase, value=amp)")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(pics_dir / "lp_modes.png", dpi=300)
    plt.close()


    # plot combined amplitude+phase (HSV: hue=phase, value=amplitude) for the 5 recovered output modes
    plt.figure(figsize=(16, 4))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        rgb = amp_phase_to_rgb(recovered_output_modes[i])
        plt.imshow(rgb)
        plt.title(f"Recovered mode {i}\n(hue=phase, value=amp)")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(pics_dir / "recovered_output_modes.png", dpi=300)
    plt.close()




    '''
    # plot the recovered channel matrix with the original channel matrix for comparison in dB
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(20 * torch.log10(torch.abs(channel_matrix) + 1e-6).detach().cpu().numpy(), cmap="viridis", aspect="equal")
    plt.colorbar(label="Recovered coupling (dB)")
    plt.xlabel("Launched mode index")
    plt.ylabel("LP reference mode index")
    plt.title("Recovered channel matrix (dB)")  
    plt.subplot(1, 2, 2)
    plt.imshow(20 * torch.log10(torch.abs(xt_matrix_1550) + 1e-6).detach().cpu().numpy(), cmap="viridis", aspect="equal")
    plt.colorbar(label="Original coupling (dB)")
    plt.xlabel("Launched mode index")
    plt.ylabel("LP reference mode index")
    plt.title("Original channel matrix (dB)")  
    plt.tight_layout()
    plt.savefig(pics_dir / "channel_matrix_comparison.png", dpi=300)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(torch.angle(channel_matrix_aligned).detach().cpu().numpy(), cmap="twilight", aspect="equal")
    plt.colorbar(label="Recovered coupling phase (radians)")
    plt.xlabel("Launched mode index")
    plt.ylabel("LP reference mode index")
    plt.title("Recovered channel matrix phase (aligned)")
    plt.subplot(1, 2, 2)
    plt.imshow(torch.angle(xt_matrix_1550).detach().cpu().numpy(), cmap="twilight", aspect="equal")
    plt.colorbar(label="Original coupling phase (radians)")
    plt.xlabel("Launched mode index")
    plt.ylabel("LP reference mode index")
    plt.title("Original channel matrix phase")
    plt.tight_layout()
    plt.savefig(pics_dir / "channel_matrix_phase_comparison.png", dpi=300)
    plt.close()

    '''

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(amp_error.detach().cpu().numpy(), cmap="inferno", aspect="equal")
    plt.colorbar(label="Amplitude error")
    plt.xlabel("Launched mode index")
    plt.ylabel("LP reference mode index")
    plt.title("Amplitude error between recovered and original channel")
    plt.subplot(1, 2, 2)
    plt.imshow(weighted_phase_error.detach().cpu().numpy(), cmap="inferno", aspect="equal")
    plt.colorbar(label="Phase error (radians)")
    plt.xlabel("Launched mode index")
    plt.ylabel("LP reference mode index")
    plt.title("Phase error between recovered and original channel")
    plt.tight_layout()
    plt.savefig(pics_dir / "channel_matrix_error_maps.png", dpi=300)


