import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path



def shift_complex_fields_subpixel(fields, shift_y_px=0.0, shift_x_px=0.0, mode="bilinear", padding_mode="zeros"):
    squeeze_back = False
    if fields.ndim == 2:
        fields = fields.unsqueeze(0)
        squeeze_back = True

    n_modes, h, w = fields.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h, device=fields.device, dtype=torch.float32),
        torch.linspace(-1, 1, w, device=fields.device, dtype=torch.float32),
        indexing="ij",
    )

    tx = 2.0 * float(shift_x_px) / max(w - 1, 1)
    ty = 2.0 * float(shift_y_px) / max(h - 1, 1)
    x_src = xx - tx
    y_src = yy - ty
    grid = torch.stack([x_src, y_src], dim=-1).unsqueeze(0).repeat(n_modes, 1, 1, 1)

    real = F.grid_sample(
        fields.real.unsqueeze(1),
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=True,
    ).squeeze(1)
    imag = F.grid_sample(
        fields.imag.unsqueeze(1),
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=True,
    ).squeeze(1)

    shifted = real + 1j * imag
    return shifted[0] if squeeze_back else shifted


def estimate_shift_to_center_from_intensity(fields, percentile=90.0, eps=1e-12):
    if fields.ndim == 2:
        intensity = torch.abs(fields) ** 2
    else:
        intensity = torch.mean(torch.abs(fields) ** 2, dim=0)

    h, w = intensity.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=intensity.device, dtype=torch.float32),
        torch.arange(w, device=intensity.device, dtype=torch.float32),
        indexing="ij",
    )

    thr = torch.quantile(intensity.reshape(-1), percentile / 100.0)
    weights = torch.clamp(intensity - thr, min=0.0)
    s = torch.clamp(torch.sum(weights), min=eps)

    cy = torch.sum(yy * weights) / s
    cx = torch.sum(xx * weights) / s

    center_y = (h - 1) / 2.0
    center_x = (w - 1) / 2.0

    shift_y = center_y - cy
    shift_x = center_x - cx
    return shift_y, shift_x


def _mode_centroid_and_rms_radius(field, eps=1e-12):
    intensity = torch.abs(field) ** 2
    h, w = intensity.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=field.device, dtype=torch.float32),
        torch.arange(w, device=field.device, dtype=torch.float32),
        indexing="ij",
    )

    s = torch.clamp(torch.sum(intensity), min=eps)
    cy = torch.sum(yy * intensity) / s
    cx = torch.sum(xx * intensity) / s
    rr2 = (yy - cy) ** 2 + (xx - cx) ** 2
    rms_radius = torch.sqrt(torch.clamp(torch.sum(rr2 * intensity) / s, min=eps))
    return cy, cx, rms_radius


def _warp_complex_field(field, scale=1.0, shift_y_px=0.0, shift_x_px=0.0, mode="bilinear", padding_mode="zeros"):
    h, w = field.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h, device=field.device, dtype=torch.float32),
        torch.linspace(-1, 1, w, device=field.device, dtype=torch.float32),
        indexing="ij",
    )

    tx = 2.0 * float(shift_x_px) / max(w - 1, 1)
    ty = 2.0 * float(shift_y_px) / max(h - 1, 1)
    x_src = (xx - tx) / float(scale)
    y_src = (yy - ty) / float(scale)
    grid = torch.stack([x_src, y_src], dim=-1).unsqueeze(0)

    real = F.grid_sample(
        field.real.unsqueeze(0).unsqueeze(0),
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=True,
    ).squeeze(0).squeeze(0)
    imag = F.grid_sample(
        field.imag.unsqueeze(0).unsqueeze(0),
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=True,
    ).squeeze(0).squeeze(0)
    return real + 1j * imag


def robust_channel_matrix(target_modes, recovered_modes, eps=1e-12):
    n_t = target_modes.shape[0]
    n_r = recovered_modes.shape[0]
    channel = torch.zeros((n_t, n_r), dtype=target_modes.dtype, device=target_modes.device)

    for i in range(n_t):
        t = target_modes[i]
        cy_t, cx_t, r_t = _mode_centroid_and_rms_radius(t, eps=eps)
        t_vec = t.reshape(-1)
        t_norm = torch.sqrt(torch.clamp(torch.sum(torch.abs(t_vec) ** 2), min=eps))

        for j in range(n_r):
            r = recovered_modes[j]
            cy_r, cx_r, r_r = _mode_centroid_and_rms_radius(r, eps=eps)

            shift_y = float((cy_t - cy_r).item())
            shift_x = float((cx_t - cx_r).item())
            scale = float(torch.clamp(r_t / torch.clamp(r_r, min=eps), min=0.6, max=1.6).item())

            r_reg = _warp_complex_field(r, scale=scale, shift_y_px=shift_y, shift_x_px=shift_x)
            r_vec = r_reg.reshape(-1)
            r_norm = torch.sqrt(torch.clamp(torch.sum(torch.abs(r_vec) ** 2), min=eps))

            channel[i, j] = torch.sum(torch.conj(t_vec) * r_vec) / torch.clamp(t_norm * r_norm, min=eps)

    return channel


def _demodulate_to_center_no_ref(F_filtered, peak_y_f, peak_x_f):
    _, h, w = F_filtered.shape
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0

    dy = cy - float(peak_y_f)
    dx = cx - float(peak_x_f)
    dy_i, dx_i = int(round(dy)), int(round(dx))
    F_i = torch.roll(F_filtered, shifts=(dy_i, dx_i), dims=(-2, -1))

    dy_f = dy - dy_i
    dx_f = dx - dx_i

    g = torch.fft.ifft2(torch.fft.ifftshift(F_i, dim=(-2, -1)))
    yy, xx = torch.meshgrid(
        torch.arange(h, device=F_i.device, dtype=torch.float32),
        torch.arange(w, device=F_i.device, dtype=torch.float32),
        indexing="ij",
    )
    frac_ramp = torch.exp(-1j * 2.0 * torch.pi * ((dx_f * xx / w) + (dy_f * yy / h)))
    return g * frac_ramp.unsqueeze(0)


def _subpixel_peak_2d(mag2d, py, px, win=5, eps=1e-12):
    H, W = mag2d.shape
    y0, y1 = max(0, py - win), min(H, py + win + 1)
    x0, x1 = max(0, px - win), min(W, px + win + 1)
    patch = mag2d[y0:y1, x0:x1]
    yy, xx = torch.meshgrid(
        torch.arange(y0, y1, device=mag2d.device, dtype=torch.float32),
        torch.arange(x0, x1, device=mag2d.device, dtype=torch.float32),
        indexing="ij",
    )
    w = torch.clamp(patch, min=0.0)
    s = torch.clamp(torch.sum(w), min=eps)
    py_f = torch.sum(yy * w) / s
    px_f = torch.sum(xx * w) / s
    return py_f, px_f


def _remove_linear_phase_tilt_blind(field, amp_thresh=0.2, eps=1e-12):
    # field: [H,W] complex
    H, W = field.shape
    amp = torch.abs(field)
    mask = amp > (amp_thresh * torch.max(amp))

    # robust phase-gradient estimate from complex neighbor products
    gx = torch.angle(torch.sum(torch.conj(field[:, :-1]) * field[:, 1:] * mask[:, :-1]))
    gy = torch.angle(torch.sum(torch.conj(field[:-1, :]) * field[1:, :] * mask[:-1, :]))

    yy, xx = torch.meshgrid(
        torch.arange(H, device=field.device, dtype=torch.float32),
        torch.arange(W, device=field.device, dtype=torch.float32),
        indexing="ij",
    )
    ramp = torch.exp(-1j * (gx * xx + gy * yy))
    return field * ramp


def simulate_channel_measurement(output_modes, reference_wave, target_modes, kx=30.0, ky=30.0):
    """
    Simulates off-axis holography to measure the channel transmission matrix.
    
    output_modes: [15, 500, 500] (Complex)
    reference_wave: [500, 500] (Complex)
    target_modes: [15, 500, 500] (Complex)
    kx, ky: Spatial carrier frequencies for the off-axis tilt
    """
    num_modes, H, W = output_modes.shape
    device = output_modes.device
    
    # ---------------------------------------------------------
    # 1. Create the Off-Axis Reference Wave
    # ---------------------------------------------------------
    # Create normalized spatial coordinates from -1 to 1
    y, x = torch.meshgrid(torch.linspace(-1, 1, H, device=device), 
                          torch.linspace(-1, 1, W, device=device), indexing='ij')
    
    # Apply linear phase ramp (tilt) to the reference wave
    phase_tilt = torch.exp(1j * 2 * torch.pi * (kx * x + ky * y))
    ref_tilted = reference_wave * phase_tilt

    pics_dir = Path("pics/computational")
    pics_dir.mkdir(parents=True, exist_ok=True)
    
    # ---------------------------------------------------------
    # 2. Simulate the Camera Measurement (Intensity)
    # ---------------------------------------------------------
    # Broadcast the tilted reference across all 15 output modes
    total_field = output_modes + ref_tilted.unsqueeze(0)
    
    # Camera captures purely real intensity
    intensity = torch.abs(total_field) ** 2 

    # plot the intensity pattern to visualize the off-axis interference
    plt.figure(figsize=(6, 5))
    plt.imshow(intensity[4].detach().cpu().numpy(), cmap='gray')
    plt.colorbar(label='Intensity')
    plt.title('Off-Axis Interference Pattern')
    plt.savefig(pics_dir / "off_axis_interference.png", dpi=300)
    plt.close()

    # plot the reference wave amplitude and phase
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(torch.abs(reference_wave).detach().cpu().numpy(), cmap='viridis')
    plt.colorbar(label='Amplitude')
    plt.title('Tilted Reference Wave Amplitude')
    plt.subplot(1, 2, 2)
    plt.imshow(torch.angle(reference_wave).detach().cpu().numpy(), cmap='twilight')
    plt.colorbar(label='Phase (radians)')
    plt.title('Tilted Reference Wave Phase')
    plt.tight_layout()
    plt.savefig(pics_dir / "reference_wave_amplitude_phase.png", dpi=300)
    plt.close()



    # Move to spatial frequency (k-space) domain
    F_intensity = torch.fft.fftshift(torch.fft.fft2(intensity), dim=(-2, -1))

    # Automatically detect the strongest 1st-order sideband
    center_y, center_x = H // 2, W // 2
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij',
    )

    spectrum_mag = torch.mean(torch.abs(F_intensity), dim=0)
    
    # Mask DC neighborhood
    dc_exclusion_radius = max(10, int(min(H, W) * 0.04))
    dc_mask = ((yy - center_y) ** 2 + (xx - center_x) ** 2) > (dc_exclusion_radius ** 2)
    searchable = spectrum_mag * dc_mask
    
    # IMPORTANT: Force selection of Sideband A (E_out * E_ref*)
    # Since your kx and ky are positive, Sideband A is shifted to negative frequencies
    # (top-left quadrant in an fftshift-ed array). Zero out the right and bottom halves.

    searchable[:, center_x:] = 0  
    searchable[center_y:, :] = 0  

    peak_flat_idx = torch.argmax(searchable)
    peak_y = int((peak_flat_idx // W).item())
    peak_x = int((peak_flat_idx % W).item())

    # Adaptive filter radius
    peak_dist = torch.sqrt(torch.tensor((peak_y - center_y) ** 2 + (peak_x - center_x) ** 2, device=device, dtype=torch.float32))
    radius = float(torch.clamp(0.25 * peak_dist, min=8.0, max=min(H, W) / 6).item())
    filter_mask = (((yy - peak_y) ** 2 + (xx - peak_x) ** 2) <= radius ** 2).float()

    # plot the F_intensity magnitude with the detected peak and filter region
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(torch.log10(spectrum_mag + 1e-12).detach().cpu().numpy(), cmap='inferno')
    plt.colorbar(label='Log Intensity')
    plt.title('Spatial Frequency Spectrum ')
    plt.subplot(1, 2, 2)
    plt.imshow(filter_mask.detach().cpu().numpy(), cmap='gray')
    plt.colorbar(label='Filter Mask')
    plt.scatter([peak_x], [peak_y], color='cyan', marker='x', s=100, label='Detected Peak')
    plt.title(f'Detected Peak and Filter (radius={radius:.1f} px)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(pics_dir / "off_axis_peak_detection.png", dpi=300)
    plt.close()


    F_filtered = F_intensity * filter_mask.unsqueeze(0)

    # ...existing code...
    peak_flat_idx = torch.argmax(searchable)
    peak_y = int((peak_flat_idx // W).item())
    peak_x = int((peak_flat_idx % W).item())

    # sub-pixel peak
    peak_y_f, peak_x_f = _subpixel_peak_2d(searchable, peak_y, peak_x, win=6)

    # precise centering (integer + fractional)
    recovered_field = _demodulate_to_center_no_ref(F_filtered, peak_y_f, peak_x_f)

    # no reference access: keep up-to-scale field, remove residual linear phase blindly
    recovered_output_modes = torch.stack(
        [_remove_linear_phase_tilt_blind(recovered_field[m]) for m in range(num_modes)],
        dim=0,
    )


    print(f"Detected 1st-order peak at (y={peak_y}, x={peak_x}), filter radius={radius:.2f} px")

    # ---------------------------------------------------------
    # 4. Channel Matrix Calculation (robust to size/centering mismatch)
    # ---------------------------------------------------------
    #channel_matrix = robust_channel_matrix(target_modes, recovered_output_modes)

    channel_matrix = torch.zeros((target_modes.shape[0], recovered_output_modes.shape[0]), dtype=target_modes.dtype, device=target_modes.device)
    recovered_output_modes = recovered_output_modes / torch.clamp(torch.sqrt(torch.sum(torch.abs(recovered_output_modes) ** 2, dim=(-2, -1), keepdim=True)), min=1e-12)
    target_modes = target_modes / torch.clamp(torch.sqrt(torch.sum(torch.abs(target_modes) ** 2, dim=(-2, -1), keepdim=True)), min=1e-12)

    for i in range(target_modes.shape[0]):
        for j in range(recovered_output_modes.shape[0]):
            channel_matrix[i, j] = torch.sum(torch.conj(target_modes[i]) * recovered_output_modes[j])

    return intensity, recovered_output_modes, channel_matrix







