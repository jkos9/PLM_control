import sys

import torch
import numpy as np  
from matplotlib import pyplot as plt

lp_phase_masks = torch.load("data\mode_launcher_phase_masks_mfdin_3300um_mfdout_500um_tiltx_0p00deg_arrdist_280p000mm_smooth_loss_0.0005.pt")#['phase_layer_1.phase']


#lp_phase_masks = lp_phase_masks.transpose(-1, -2)
print(lp_phase_masks.shape)
lp_phase_masks_np = lp_phase_masks.cpu().detach().numpy()
print(lp_phase_masks_np.shape)

# optional flag to flip the phase mask values from [0, 2pi] to [2pi, 0]

lp_phase_masks_np_flipped = 2 * np.pi - lp_phase_masks_np

plt.figure()
for i in range(lp_phase_masks_np.shape[0]):
    plt.subplot(lp_phase_masks_np.shape[0]//3, 3, i + 1)
    plt.imshow(lp_phase_masks_np[i], cmap='viridis')
    plt.colorbar()  
    plt.title(f'Phase Mask {i+1}')
plt.colorbar()  
plt.title('Phase Masks (Mode Launcher)')

plt.figure()
for i in range(lp_phase_masks_np_flipped.shape[0]):
    plt.subplot(lp_phase_masks_np_flipped.shape[0]//3, 3, i + 1)
    plt.imshow(lp_phase_masks_np_flipped[i], cmap='viridis')
    plt.colorbar()  
    plt.title(f'Phase Mask {i+1}')
plt.colorbar()  
plt.title('Phase Masks (Mode Launcher) Flipped')
#plt.show()

# take first 3 phase masks and save as numpy array
lp_phase_masks_np = lp_phase_masks_np[:3]

np.save("data/mode_launcher_phase_masks_mfdin_3300um_mfdout_500um_tiltx_0p00deg_arrdist_280p000mm_smooth_loss_0.0005_first3.npy", lp_phase_masks_np)


