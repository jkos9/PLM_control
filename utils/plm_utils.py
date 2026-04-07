import numpy as np
from vmbpy import VmbSystem
from vmbpy import *

from pathlib import Path
import sys

# Force local workspace package first so DB comes from PLM_NIR_control/src/ti_plm/db
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

#from ti_plm import PLM
#from ti_plm.display import ImageWindow, EventLoopExit
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from time import perf_counter, sleep
from tqdm import tqdm
#from utils.holo_utils import generate_checkerboard, generate_line_pattern, plot_phase_map, generate_global_pattern


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

    



