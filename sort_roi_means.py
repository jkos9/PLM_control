import numpy as np
from matplotlib import pyplot as plt



loaded_roi_values = np.load('summed_roi_values_sorted.npy')
original_lut = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

sort_idx = np.argsort(loaded_roi_values)
sorted_roi_values = loaded_roi_values[sort_idx]
sorted_lut = np.array(original_lut)[sort_idx]
print(loaded_roi_values)



print("Sorted ROI values (smallest to largest):")
print(sorted_roi_values)
print("LUT reordered with same sorting:")
print(sorted_lut.tolist())




