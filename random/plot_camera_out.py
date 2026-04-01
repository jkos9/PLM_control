import numpy as np
from vmbpy import *
from matplotlib import pyplot as plt

def main():
    # Open VmbPy system
    with VmbSystem.get_instance() as vmb:
        cams = vmb.get_all_cameras()
        if not cams:
            print("No cameras found.")
            return

        # Use first detected camera
        with cams[0] as cam:
            print("Using camera:", cam.get_name())

            # Acquire single frame
            frame = cam.get_frame()

            # Convert and RE-ASSIGN
            # This creates a new 8-bit frame from the 14-bit original
            frame_mono8 = frame.convert_pixel_format(PixelFormat.Mono8)

            # Convert to OpenCV/Numpy format
            img = frame_mono8.as_opencv_image()
            
            plt.figure()
            plt.imshow(img)
            plt.title("Captured Frame")
            plt.show()
            
if __name__ == "__main__":
    main()

