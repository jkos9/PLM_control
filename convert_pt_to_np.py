import sys

import torch
import numpy as np  
from matplotlib import pyplot as plt

lp_phase_masks = torch.load("phase_patterns\mode_launcher_phase_masks_in_2200_out_700_shift_50.pt")#['phase_layer_1.phase']


#lp_phase_masks = lp_phase_masks.transpose(-1, -2)
print(lp_phase_masks.shape)
lp_phase_masks_np = lp_phase_masks.cpu().detach().numpy()
print(lp_phase_masks_np.shape)


plt.figure()
for i in range(lp_phase_masks_np.shape[0]):
    plt.subplot(lp_phase_masks_np.shape[0]//3, 3, i + 1)
    plt.imshow(lp_phase_masks_np[i], cmap='viridis')
    plt.colorbar()  
    plt.title(f'Phase Mask {i+1}')
plt.colorbar()  
plt.title('Phase Masks (Mode Launcher)')
#plt.show()

np.save("phase_patterns\mode_launcher_phase_masks_in_2200_out_700_shift_50.npy", lp_phase_masks_np)


