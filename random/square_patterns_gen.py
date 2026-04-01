from tqdm import tqdm
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from utils.holo_utils import generate_checkerboard, generate_line_pattern, generate_square_test_patterns, plot_phase_map, generate_global_pattern    
import numpy as np
from ti_plm import PLM
from utils.plm_utils import get_phase_values_from_calibration
import matplotlib.pyplot as plt

x_dim = 904
y_dim = 800

#phase_values = np.linspace(0, 2 * np.pi, 33)[:-1]


plm = PLM.from_db('p67nirtemp')
phase_levels, buckets, phase_values = get_phase_values_from_calibration(plm)


line_spacing = 20 #80 * 3

final_phase_array = np.zeros((len(phase_values), y_dim, x_dim))
'''

for i, phase_val in enumerate(phase_values):  
    phase_val = 0.5 * np.pi
    
    
    phase = generate_square_test_patterns(x_dim, y_dim)
    #phase = generate_global_pattern(x_dim, y_dim, max_phase=phase_val)
    final_phase_array[i] = phase
'''

final_phase_array = np.array(generate_square_test_patterns(x_dim, y_dim))
plt.figure()   
for i in range(final_phase_array.shape[0]):
    plt.subplot(2, 2, i + 1)
    plt.imshow(final_phase_array[i], cmap='viridis')
    plt.title(f'Square Pattern {i + 1}')
    plt.colorbar(label='Phase (radians)')

plt.tight_layout()
#plt.show()
#sys.exit()
#plt.savefig('square_test_patterns.png', dpi=300)


np.save(f"phase_patterns/square_test_patterns.npy", final_phase_array)

print(final_phase_array.shape)
