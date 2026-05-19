import os
import sys
import time
import math
import numpy as np
import pygame
from OpenGL.GL import (
    glBegin,
    glBindTexture,
    glClear,
    glEnd,
    glEnable,
    glGenTextures,
    glLoadIdentity,
    glMatrixMode,
    glOrtho,
    glPixelStorei,
    glTexCoord2f,
    glTexImage2D,
    glTexParameteri,
    glTexSubImage2D,
    glVertex2f,
    glViewport,
    GL_COLOR_BUFFER_BIT,
    GL_LUMINANCE,
    GL_MODELVIEW,
    GL_NEAREST,
    GL_PROJECTION,
    GL_TEXTURE_2D,
    GL_RGB,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_UNPACK_ALIGNMENT,
    GL_UNSIGNED_BYTE,
    GL_QUADS,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_CLAMP_TO_EDGE,
)

# Ensure local imports work when running directly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ti_plm import PLM


PATTERN_PATH = r"phase_patterns\mode_launcher_phase_masks_mfdin_3300um_mfdout_1449um_tiltx_0p00deg_arrdist_0p000mm_modegroups_5_smooth_loss_0.005.npy"


#PATTERN_PATH = r"phase_patterns\test_patterns\line_phase_sweep_max_20_line_spacing.npy"
PLM_CATALOG = "p67nirtemp"

DISPLAY_INDEX = 1
TARGET_FPS = 10 # max 30 fps
FLIP_Y = True
AS_RGB_TEXTURE = True
USE_PLM_RESOLUTION = False
CENTER_IN_WINDOW = True

GRATING_PERIOD_X_PX = 4
GRATING_PERIOD_Y_PX = 0

FOCAL_LENGTH_M = 0.0
FOCAL_WAVELENGTH_M = 1.55e-6

# Zernike settings: coefficients in radians, keyed by (n, m)
ZERN_MAX_DEGREE = 2
ZERN_COEFFS = {
    # (2, 0): 0.5,
}


def get_phase_values_from_calibration(plm: PLM) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phase_min, phase_max = plm.phase_range
    n_states = len(plm.displacement_ratios)
    ratio_scale = (
        plm.max_displacement_ratio
        if plm.max_displacement_ratio is not None
        else (n_states - 1) / n_states
    )

    phase_levels = phase_min + plm.displacement_ratios * ratio_scale * (
        phase_max - phase_min
    )

    phase_disp = np.hstack([phase_levels, phase_max])
    buckets = (phase_disp[:-1] + phase_disp[1:]) / 2.0
    representative_values = np.empty(n_states, dtype=np.float64)
    representative_values[0] = phase_min + 1e-9
    if n_states > 1:
        representative_values[1:] = (phase_levels[1:] + phase_levels[:-1]) / 2.0
    return phase_levels, buckets, representative_values


def enumerate_zernike_terms(max_radial_degree: int) -> list[tuple[int, int]]:

    terms: list[tuple[int, int]] = []

    for n in range(max(0, int(max_radial_degree)) + 1):

        for m in range(-n, n + 1, 2):

            terms.append((n, m))

    return terms

def zernike_radial_polynomial(n: int, m_abs: int, r: np.ndarray) -> np.ndarray:

    if (n - m_abs) % 2 != 0:

        return np.zeros_like(r, dtype=np.float32)

    radial = np.zeros_like(r, dtype=np.float64)

    k_max = (n - m_abs) // 2

    for k in range(k_max + 1):

        numerator = math.factorial(n - k)

        denominator = (
            math.factorial(k)
            * math.factorial((n + m_abs) // 2 - k)
            * math.factorial((n - m_abs) // 2 - k)
        )

        coefficient = ((-1) ** k) * numerator / float(denominator)

        radial += coefficient * np.power(r, n - 2 * k)

    return radial.astype(np.float32)

def build_zernike_phase_map(
    shape: tuple[int, int],
    coeffs: dict[tuple[int, int], float],
    max_degree: int,
    pitch_x_m: float,
    pitch_y_m: float,
) -> np.ndarray:

    height, width = int(shape[0]), int(shape[1])

    terms = enumerate_zernike_terms(max_degree)
    coeff_values = tuple(float(coeffs.get((n, m), 0.0)) for (n, m) in terms)

    if pitch_x_m <= 0.0:
        pitch_x_m = 16.2e-6

    if pitch_y_m <= 0.0:
        pitch_y_m = 10.8e-6

    x_m = (np.arange(width, dtype=np.float32) - 0.5 * float(width - 1)) * float(
        pitch_x_m
    )

    y_m = (np.arange(height, dtype=np.float32) - 0.5 * float(height - 1)) * float(
        pitch_y_m
    )

    yy_m, xx_m = np.meshgrid(y_m, x_m, indexing="ij")

    half_extent_x = float(np.max(np.abs(x_m))) if width > 0 else 0.0

    half_extent_y = float(np.max(np.abs(y_m))) if height > 0 else 0.0

    aperture_radius_m = max(1e-12, min(half_extent_x, half_extent_y))

    xx = xx_m / float(aperture_radius_m)

    yy = yy_m / float(aperture_radius_m)

    r = np.sqrt(xx * xx + yy * yy)

    theta = np.arctan2(yy, xx)

    aperture = r <= 1.0

    phase = np.zeros((height, width), dtype=np.float32)

    for coeff, (n, m) in zip(coeff_values, terms):

        if abs(coeff) <= 1e-9:

            continue

        radial = zernike_radial_polynomial(n, abs(m), r)

        if m == 0:

            mode = radial

        elif m > 0:

            mode = radial * np.cos(float(m) * theta)

        else:

            mode = radial * np.sin(float(abs(m)) * theta)

        mode = np.where(aperture, mode, 0.0)

        rms = (
            float(np.sqrt(np.mean(np.square(mode[aperture]))))
            if np.any(aperture)
            else 0.0
        )

        if rms > 1e-12:

            mode = mode / rms

        phase += float(coeff) * mode.astype(np.float32, copy=False)

    phase = np.where(aperture, phase, 0.0).astype(np.float32, copy=False)

    return phase


def build_grating_phase_ramp(
    shape: tuple[int, int],
    period_x_px: int,
    period_y_px: int,
    ) -> np.ndarray:

    height, width = int(shape[0]), int(shape[1])
    ramp = np.zeros((height, width), dtype=np.float32)
    if period_x_px != 0:

        x = np.arange(width, dtype=np.float32)
        ramp += (2.0 * np.pi * x / float(period_x_px))[None, :]

    if period_y_px != 0:
        y = np.arange(height, dtype=np.float32)
        ramp += (2.0 * np.pi * y / float(period_y_px))[:, None]

    return np.mod(ramp, 2.0 * np.pi).astype(np.float32, copy=False)


def build_focal_lens_phase_map(
    shape: tuple[int, int],
    focal_length_m: float,
    pitch_x_m: float,
    pitch_y_m: float,
    wavelength_m: float,
) -> np.ndarray:

    height, width = int(shape[0]), int(shape[1])

    if (
        height <= 0
        or width <= 0
        or abs(float(focal_length_m)) <= 1e-15
        or wavelength_m <= 0.0
        or pitch_x_m <= 0.0
        or pitch_y_m <= 0.0
    ):

        return np.zeros((max(0, height), max(0, width)), dtype=np.float32)

    x = (np.arange(width, dtype=np.float32) - 0.5 * float(width - 1)) * float(
        pitch_x_m
    )

    y = (np.arange(height, dtype=np.float32) - 0.5 * float(height - 1)) * float(
        pitch_y_m
    )

    yy, xx = np.meshgrid(y, x, indexing="ij")

    # Thin-lens quadratic phase; this is equivalent to Zernike defocus up to a piston term.
    phase = (np.pi / (float(wavelength_m) * float(focal_length_m))) * (
        xx * xx + yy * yy
    )

    phase -= float(np.mean(phase))

    return np.mod(phase, 2.0 * np.pi).astype(np.float32, copy=False)


def load_patterns(path: str) -> np.ndarray:
    patterns = np.load(path)
    if patterns.ndim == 2:
        patterns = patterns[None, :, :]
    if patterns.ndim != 3:
        raise ValueError(f"Expected 2D or 3D pattern array, got shape {patterns.shape}")
    return np.asarray(patterns, dtype=np.float32)


def compose_overlaid_phase(
    base_phase: np.ndarray,
    grating: np.ndarray,
    focal: np.ndarray,
    zernike: np.ndarray,
) -> np.ndarray:
    return np.mod(base_phase + grating + focal + zernike, 2.0 * np.pi)


def pad_bitmap_if_needed(bmp: np.ndarray) -> np.ndarray:
    if bmp.ndim != 2:
        return bmp
    height, width = int(bmp.shape[0]), int(bmp.shape[1])
    if (height, width) == (1600, 2712):
        padded = np.zeros((1600, 2716), dtype=bmp.dtype)
        padded[:, 2:2 + 2712] = bmp
        return padded
    return bmp


def normalize_plm_bitmap(bmp: np.ndarray, as_rgb: bool) -> np.ndarray:
    bmp = pad_bitmap_if_needed(bmp)
    if as_rgb:
        bmp = np.repeat(bmp[:, :, None], 3, axis=2)
    return bmp


def precompute_bmp_stack(
    patterns: np.ndarray,
    plm: PLM,
    grating: np.ndarray,
    focal: np.ndarray,
    zernike: np.ndarray,
) -> np.ndarray:
    phase_levels, buckets, phase_values = get_phase_values_from_calibration(plm)
    n_states = int(len(phase_values))
    if n_states <= 0 or len(phase_levels) != n_states:
        raise RuntimeError("Invalid PLM calibration LUT.")

    total_frames = int(patterns.shape[0])
    if total_frames <= 0:
        return np.empty((0, 0, 0), dtype=np.uint8)

    base_phase = np.mod(patterns[0], 2.0 * np.pi)
    overlaid = compose_overlaid_phase(base_phase, grating, focal, zernike)
    state_indices = np.digitize(overlaid, buckets, right=False).astype(np.int32)
    state_indices = np.clip(state_indices, 0, n_states - 1)
    phase_for_lut = phase_values[state_indices]
    bmp0 = np.asarray(plm.process_phase_map(phase_for_lut), dtype=np.uint8)

    if bmp0.ndim != 2:
        raise ValueError(f"PLM bitmap must be 2D, got shape {bmp0.shape}")

    bmp0 = normalize_plm_bitmap(bmp0, AS_RGB_TEXTURE)
    if bmp0.ndim == 3:
        bmp_h, bmp_w = int(bmp0.shape[0]), int(bmp0.shape[1])
        bmp_stack = np.empty((total_frames, bmp_h, bmp_w, 3), dtype=np.uint8)
    else:
        bmp_h, bmp_w = int(bmp0.shape[0]), int(bmp0.shape[1])
        bmp_stack = np.empty((total_frames, bmp_h, bmp_w), dtype=np.uint8)

    bmp_stack[0] = bmp0

    for idx in range(1, total_frames):
        base_phase = np.mod(patterns[idx], 2.0 * np.pi)
        overlaid = compose_overlaid_phase(base_phase, grating, focal, zernike)
        state_indices = np.digitize(overlaid, buckets, right=False).astype(np.int32)
        state_indices = np.clip(state_indices, 0, n_states - 1)
        phase_for_lut = phase_values[state_indices]
        bmp = np.asarray(plm.process_phase_map(phase_for_lut), dtype=np.uint8)
        if bmp.ndim != 2:
            raise ValueError(f"PLM bitmap must be 2D, got shape {bmp.shape}")
        bmp = normalize_plm_bitmap(bmp, AS_RGB_TEXTURE)
        if bmp.shape != bmp_stack.shape[1:]:
            raise ValueError(
                f"Inconsistent PLM bitmap shape: {bmp.shape} vs {bmp_stack.shape[1:]}"
            )
        bmp_stack[idx] = bmp

    return bmp_stack



def main() -> int:
    patterns = load_patterns(PATTERN_PATH)

    plm = PLM.from_db(PLM_CATALOG)
    expected_h, expected_w = int(plm.shape[0]), int(plm.shape[1])
    if int(patterns.shape[1]) != expected_h or int(patterns.shape[2]) != expected_w:
        raise ValueError(
            f"Pattern shape {patterns.shape[1:]} does not match PLM shape {(expected_h, expected_w)}"
        )

    pitch_y_m = float(plm.pitch[0]) if plm.pitch is not None else 0.0
    pitch_x_m = float(plm.pitch[1]) if plm.pitch is not None else 0.0

    grating = build_grating_phase_ramp(
        (expected_h, expected_w), int(GRATING_PERIOD_X_PX), int(GRATING_PERIOD_Y_PX)
    )
    focal = build_focal_lens_phase_map(
        (expected_h, expected_w),
        float(FOCAL_LENGTH_M),
        float(pitch_x_m),
        float(pitch_y_m),
        float(FOCAL_WAVELENGTH_M),
    )
    zernike = build_zernike_phase_map(
        (expected_h, expected_w),
        ZERN_COEFFS,
        int(ZERN_MAX_DEGREE),
        float(pitch_x_m),
        float(pitch_y_m),
    )

    bmp_stack = precompute_bmp_stack(patterns, plm, grating, focal, zernike)
    total_frames = int(bmp_stack.shape[0])
    if total_frames <= 0:
        raise RuntimeError("No frames to display.")

    os.environ["SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS"] = "0"
    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_SWAP_CONTROL, 0)

    try:
        screen_size = pygame.display.get_desktop_sizes()[DISPLAY_INDEX]
    except IndexError:
        screen_size = pygame.display.get_desktop_sizes()[0]

    tex_h, tex_w = int(bmp_stack.shape[1]), int(bmp_stack.shape[2])
    window_size = (tex_w, tex_h) if USE_PLM_RESOLUTION else screen_size

    pygame.display.set_mode(
        window_size,
        pygame.OPENGL | pygame.FULLSCREEN | pygame.DOUBLEBUF,
        display=DISPLAY_INDEX,
    )

    window_w, window_h = pygame.display.get_window_size()

    glViewport(0, 0, int(window_w), int(window_h))
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0, int(window_w), int(window_h), 0, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glEnable(GL_TEXTURE_2D)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

    frame_idx = 0
    running = True
    period = 1.0 / float(TARGET_FPS)
    next_time = time.perf_counter()
    tex_initialized = False
    tex_format = GL_RGB if AS_RGB_TEXTURE else GL_LUMINANCE


    while running:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        if not running:
            break

        frame = bmp_stack[frame_idx]
        #print(frame.shape, frame.dtype)
        #sys.exit()

        if FLIP_Y:
            frame = np.flipud(frame)
        frame = np.ascontiguousarray(frame, dtype=np.uint8)

        glBindTexture(GL_TEXTURE_2D, tex)
        if not tex_initialized:
            glTexImage2D(
                GL_TEXTURE_2D,
                0,
                tex_format,
                tex_w,
                tex_h,
                0,
                tex_format,
                GL_UNSIGNED_BYTE,
                frame,
            )
            tex_initialized = True
        else:
            glTexSubImage2D(
                GL_TEXTURE_2D,
                0,
                0,
                0,
                tex_w,
                tex_h,
                tex_format,
                GL_UNSIGNED_BYTE,
                frame,
            )
        glClear(GL_COLOR_BUFFER_BIT)


        if CENTER_IN_WINDOW:
            x = 0.5 * (float(window_w) - float(tex_w))
            y = 0.5 * (float(window_h) - float(tex_h))
        else:
            x = 0.0
            y = 0.0

        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 1.0); glVertex2f(x, y)
        glTexCoord2f(1.0, 1.0); glVertex2f(x + tex_w, y)
        glTexCoord2f(1.0, 0.0); glVertex2f(x + tex_w, y + tex_h)
        glTexCoord2f(0.0, 0.0); glVertex2f(x, y + tex_h)
        glEnd()

        pygame.display.flip()

        next_time += period
        now = time.perf_counter()
        sleep_time = next_time - now
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            next_time = now

        frame_idx = (frame_idx + 1) % total_frames

    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())