"""Simple end-to-end example for showing a checkerboard on a TI PLM p67 NIR display."""

from pathlib import Path
import sys

# Force local workspace package first so DB comes from PLM_NIR_control/src/ti_plm/db
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ti_plm import PLM
from ti_plm.display import ImageWindow, EventLoopExit
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from time import perf_counter
from tqdm import tqdm
from utils.holo_utils import generate_checkerboard, generate_line_pattern, plot_phase_map


def main() -> None:
    DEFAULT_MONITOR = 1
    DISPLAY_DURATION_S = 2

    x_dim = 904
    y_dim = 800

    # Generate a checkerboard phase pattern with values in [0, 2pi)
    tile_size = 32*2
    #phase = generate_checkerboard(x_dim, y_dim, tile_size)
    #line_spacing = 32
    line_spacing = 15

    phase_values = np.linspace(0, 1 * np.pi, 33)[:-1]
    print(phase_values)

    plm = PLM.from_db('p67nirtemp', phase_range = (0, 1 * np.pi))

    for phase_val in tqdm(phase_values, desc="Generating and displaying patterns"):  
        phase = generate_line_pattern(x_dim, y_dim, line_spacing, orientation="vertical", max_phase=phase_val)
        # Process phase data into bitmap specific to the .67 PLM
        # This handles all quantization to appropriate phase displacement levels and mapping to the correct 2x2 electrode locations
        bmp = plm.process_phase_map(phase)
        bmp_img = Image.fromarray(bmp)

        with ImageWindow(fullscreen=True, monitor=DEFAULT_MONITOR) as win:
            win.load(bmp_img)
            start_time = perf_counter()

            def stop_after_duration() -> None:
                if perf_counter() - start_time >= DISPLAY_DURATION_S:
                    raise EventLoopExit

            win.loop_callback = stop_after_duration
            win.run()

if __name__ == "__main__":
    main()


