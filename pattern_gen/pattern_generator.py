from tqdm import tqdm
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from utils.holo_utils import generate_checkerboard, generate_line_pattern, plot_phase_map, generate_global_pattern    
import numpy as np
from ti_plm import PLM
from utils.plm_utils import get_phase_values_from_calibration


x_dim = 904
y_dim = 800

#phase_values = np.linspace(0, 2 * np.pi, 33)[:-1]


plm = PLM.from_db('p67nirtemp')
phase_levels, buckets, phase_values = get_phase_values_from_calibration(plm)


line_spacing = 20 #80 * 3

final_phase_array = np.zeros((len(phase_values), y_dim, x_dim))

for i, phase_val in enumerate(phase_values):  
    phase_val = 0.5 * np.pi
    
    
    phase = generate_line_pattern(x_dim, y_dim, line_spacing, orientation="vertical", max_phase=phase_val)
    #phase = generate_global_pattern(x_dim, y_dim, max_phase=phase_val)
    final_phase_array[i] = phase

#np.save(f"phase_patterns/line_phase_sweep_max_20_line_spacing.npy", final_phase_array)
#np.save(f"phase_patterns/line_phase_sweep_thicker.npy", final_phase_array)
np.save(f"phase_patterns/global_phase_sweep.npy", final_phase_array)


print(final_phase_array.shape)

