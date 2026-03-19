from tqdm import tqdm

from utils.holo_utils import generate_checkerboard, generate_line_pattern, plot_phase_map, generate_global_pattern    
import numpy as np



x_dim = 904
y_dim = 800

phase_values = np.linspace(0, 2 * np.pi, 33)[:-1]
line_spacing = 14

final_phase_array = np.zeros((len(phase_values), y_dim, x_dim))

for i, phase_val in enumerate(phase_values):  
    #phase_val = 0.5 * np.pi
    
    phase = generate_line_pattern(x_dim, y_dim, line_spacing, orientation="vertical", max_phase=phase_val)
    #phase = generate_global_pattern(x_dim, y_dim, max_phase=phase_val)
    final_phase_array[i] = phase

#np.save(f"phase_patterns/global_phase_sweep.npy", final_phase_array)
np.save(f"phase_patterns/line_phase_sweep.npy", final_phase_array)

print(final_phase_array.shape)

