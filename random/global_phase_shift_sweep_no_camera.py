import numpy as np
from vmbpy import VmbSystem
from vmbpy import *

from pathlib import Path
import sys

# Force local workspace package first so DB comes from PLM_NIR_control/src/ti_plm/db
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ti_plm import PLM
from ti_plm.display import ImageWindow, EventLoopExit
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from time import perf_counter, sleep
from tqdm import tqdm
from utils.holo_utils import generate_checkerboard, generate_line_pattern, plot_phase_map, generate_global_pattern


def extract_unique_electrode_patterns(bmp: np.ndarray, phase_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    phase_h, phase_w = phase_shape
    if bmp.shape[0] % phase_h != 0 or bmp.shape[1] % phase_w != 0:
        raise ValueError(
            f"Bitmap shape {bmp.shape} is not an integer expansion of phase shape {phase_shape}."
        )

    tile_h = bmp.shape[0] // phase_h
    tile_w = bmp.shape[1] // phase_w

    bmp_bits = (bmp > 0).astype(np.uint8)
    tiles = bmp_bits.reshape(phase_h, tile_h, phase_w, tile_w).transpose(0, 2, 1, 3).reshape(-1, tile_h, tile_w)
    unique_tiles, counts = np.unique(tiles, axis=0, return_counts=True)
    return unique_tiles, counts, (tile_h, tile_w)


def plot_unique_pattern_gallery(pattern_tiles: list[np.ndarray], tile_shape: tuple[int, int]) -> None:
    n_patterns = len(pattern_tiles)
    if n_patterns == 0:
        print("No unique electrode patterns found.")
        return

    cols = min(8, n_patterns)
    rows = int(np.ceil(n_patterns / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.6 * rows))
    axes = np.atleast_1d(axes).ravel()

    for idx, ax in enumerate(axes):
        if idx >= n_patterns:
            ax.axis("off")
            continue

        tile = pattern_tiles[idx]
        ax.imshow(tile, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"Pattern {idx + 1}\\nones={int(tile.sum())}", fontsize=9)
        ax.set_xticks(range(tile_shape[1]))
        ax.set_yticks(range(tile_shape[0]))
        ax.grid(color="red", linestyle="-", linewidth=0.7)

    fig.suptitle(f"Unique electrode micro-patterns ({tile_shape[0]}x{tile_shape[1]})", fontsize=12)
    fig.tight_layout()
    plt.show()


def get_phase_values_from_calibration(plm: PLM) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phase_min, phase_max = plm.phase_range
    n_states = len(plm.displacement_ratios)

    ratio_scale = (
        plm.max_displacement_ratio
        if plm.max_displacement_ratio is not None
        else (n_states - 1) / n_states
    )

    # Calibrated phase level for each device state (length = n_states)
    phase_levels = phase_min + plm.displacement_ratios * ratio_scale * (phase_max - phase_min)

    # Same bucket boundaries used by quantize()
    phase_disp = np.hstack([phase_levels, phase_max])
    buckets = (phase_disp[:-1] + phase_disp[1:]) / 2.0

    # Representative phase per state (guaranteed inside each state interval)
    reps = np.empty(n_states, dtype=np.float64)
    reps[0] = phase_min + 1e-9
    if n_states > 1:
        reps[1:] = 0.5 * (buckets[:-1] + buckets[1:])

    return phase_levels, buckets, reps



def main() -> None:
    DEFAULT_MONITOR = 1
    DISPLAY_DURATION_S = 0.2
    ROI_CENTER_X = 247
    ROI_CENTER_Y = 254
    ROI_HALF_WIDTH = 12
    ROI_HALF_HEIGHT = 12
    AVERAGING_FRAME_COUNT = 1
    AVERAGING_FRAME_INTERVAL_S = 0.025

    x_dim = 904
    y_dim = 800

    # Generate a checkerboard phase pattern with values in [0, 2pi)
    tile_size = 32*2
    #phase = generate_checkerboard(x_dim, y_dim, tile_size)
    #line_spacing = 32
    line_spacing = 50

    #phase_values = np.linspace(0, 2* np.pi, 33)[:-1]


    plm = PLM.from_db('p67nirtemp')


    phase_levels, buckets, phase_values = get_phase_values_from_calibration(plm)

    print("State  idx | phase_level(rad) | bucket(rad) | rep_phase(rad)")
    for idx in range(len(phase_values)):
        print(f"{idx:9d} | {phase_levels[idx]:16.9f} | {buckets[idx]:11.9f} | {phase_values[idx]:13.9f}")

    # Optional sanity check: each representative maps to unique state 0..N-1
    states = plm.quantize(phase_values[:, None, None]).reshape(-1)
    print("quantized states from representative phases:", states)
    
    print(phase_values)
    #sys.exit()

    x_vals = np.linspace(0, 2*np.pi, len(phase_values))   
    plt.figure()
    plt.plot(x_vals, phase_levels, label="Calibrated Phase Levels")
    plt.plot(x_vals, np.linspace(0, 2*np.pi, len(phase_values)), label="Ideal Linear Phase")
    #plt.show()
    #sys.exit()


    #phase_values = buckets

    bmp_max_list = []
    unique_pattern_dict: dict[tuple[int, ...], np.ndarray] = {}
    tile_shape: tuple[int, int] | None = None
    n_phase = len(phase_values)
    subplot_rows = 4
    subplot_cols = int(np.ceil(n_phase / subplot_rows))



    #plt.figure(figsize=(2.0 * subplot_cols, 1 * subplot_rows))

    for i, phase_val in tqdm(enumerate(phase_values), total=len(phase_values)):

        #phase_val = 0.5 * np.pi

        #phase = generate_line_pattern(x_dim, y_dim, line_spacing, orientation="vertical", max_phase=phase_val)
        phase = generate_global_pattern(x_dim, y_dim, max_phase=phase_val)
        # Process phase data into bitmap specific to the .67 PLM
        # This handles all quantization to appropriate phase displacement levels and mapping to the correct 2x2 electrode locations
        bmp = plm.process_phase_map(phase)
        bmp_img = Image.fromarray(bmp)



        print(bmp.shape, phase.shape)
        a = 1
        #plt.figure(figsize=(8, 6))
        #plt.subplot(subplot_rows, subplot_cols, i + 1)
        #plt.imshow(bmp[a:a+2, 0:3], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        #plt.title(f"Phase: {phase_val:.6f} rad")
        #plt.show()
        #plt.close()
        #'''
        unique_tiles, tile_counts, this_tile_shape = extract_unique_electrode_patterns(bmp, phase.shape)
        tile_shape = this_tile_shape
        for tile in unique_tiles:
            key = tuple(tile.flatten().tolist())
            if key not in unique_pattern_dict:
                unique_pattern_dict[key] = tile.copy()

        print(
            f"phase={phase_val:.6f} rad | bmp_shape={bmp.shape} | "
            f"tile_shape={this_tile_shape} | unique_tiles_in_frame={len(unique_tiles)}"
        )


        #print(bmp_img)
        #sys.exit()

        on_mask = phase > 0
        off_mask = ~on_mask
        row_scale = bmp.shape[0] // phase.shape[0]
        col_scale = bmp.shape[1] // phase.shape[1]
        on_mask_bmp = np.repeat(np.repeat(on_mask, row_scale, axis=0), col_scale, axis=1)
        off_mask_bmp = ~on_mask_bmp
        bmp_mean = float(np.mean(bmp))
        bmp_min = int(np.min(bmp))
        bmp_max = int(np.max(bmp))
        bmp_on_mean = float(np.mean(bmp[on_mask_bmp])) if np.any(on_mask_bmp) else float("nan")
        bmp_off_mean = float(np.mean(bmp[off_mask_bmp])) if np.any(off_mask_bmp) else float("nan")
        bmp_on_fraction = float(np.mean(bmp > 0))

        print(
            f"phase={phase_val:.6f} rad | "
            f"bmp_mean={bmp_mean:.3f}, bmp_min={bmp_min}, bmp_max={bmp_max}, "
            f"on_band_mean={bmp_on_mean:.3f}, off_band_mean={bmp_off_mean:.3f}, "
            f"on_pixel_fraction={bmp_on_fraction:.4f}"
        )

        roi_frame_sums = []

        with ImageWindow(fullscreen=True, monitor=DEFAULT_MONITOR) as win:
            win.load(bmp_img)
            start_time = perf_counter()

            def stop_after_duration() -> None:
                #nonlocal img_artist, roi_rect
                if perf_counter() - start_time >= DISPLAY_DURATION_S:
                    plt.pause(0.001)
                    raise EventLoopExit

            win.loop_callback = stop_after_duration
            win.run()
            

            bmp_max_list.append(bmp_max)
        #'''

    #plt.show()
    sys.exit()

    if tile_shape is not None:
        pattern_tiles = list(unique_pattern_dict.values())
        print(f"Total unique electrode patterns across sweep: {len(pattern_tiles)}")
        plot_unique_pattern_gallery(pattern_tiles, tile_shape)


if __name__ == "__main__":
    main()


