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
from holo_utils import generate_checkerboard, generate_line_pattern, plot_phase_map, generate_global_pattern


def main() -> None:
    DEFAULT_MONITOR = 1
    DISPLAY_DURATION_S = 2
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
    line_spacing = 10

    #phase_values = np.linspace(0, 2* np.pi, 33)[:-1]

    phase_values = np.linspace(0, 2 * np.pi, 33)[:-1]
    print(phase_values)

    # print(np.linspace(0,0.5, 32).tolist())
    #sys.exit(0)

    plm = PLM.from_db('p67nirtemp')


    with VmbSystem.get_instance() as vmb:
        cams = vmb.get_all_cameras()
        if not cams:
            print("No cameras found.")
            return

        with cams[0] as cam:
            print("Using camera:", cam.get_name())

            desired_exposure_us = 200.0 
            if hasattr(cam, 'ExposureAuto'):
                cam.ExposureAuto.set('Off')
                
            if hasattr(cam, 'ExposureTime'):
                cam.ExposureTime.set(desired_exposure_us)
                print(f"Exposure successfully set to {cam.ExposureTime.get()} us")


            for phase_val in tqdm(phase_values, desc="Generating, displaying, and capturing"):  
                #phase_val = 0.5 * np.pi
                #phase = generate_line_pattern(x_dim, y_dim, line_spacing, orientation="vertical", max_phase=phase_val)
                phase = generate_global_pattern(x_dim, y_dim, max_phase=phase_val)
                # Process phase data into bitmap specific to the .67 PLM
                # This handles all quantization to appropriate phase displacement levels and mapping to the correct 2x2 electrode locations
                bmp = plm.process_phase_map(phase)
                bmp_img = Image.fromarray(bmp)
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

                    frame = cam.get_frame()
                    frame_mono8 = frame.convert_pixel_format(PixelFormat.Mono8)
                    img = frame_mono8.as_opencv_image()
                    latest_img = img

                    img_f = latest_img.astype(np.float32)
                    img_f = img_f - img_f.mean()  # remove DC bias (helps a lot)


                    fft_magnitude = np.abs(np.fft.fftshift(np.fft.fft2(img_f), axes=(1,)))

                    plt.figure(figsize=(12, 6))
                    plt.subplot(1, 2, 1)
                    plt.imshow(latest_img, cmap='gray')
                    plt.colorbar(label='Pixel Intensity')
                    plt.title(f'Captured Image at Phase {phase_val:.2f} rad')
                    plt.subplot(1, 2, 2)
                    plt.title('Fourier Transform Magnitude')
                    plt.imshow((fft_magnitude), cmap='inferno')
                    plt.colorbar(label='Magnitude')

                    plt.savefig(f"pics/captured_image_phase_{phase_val:.2f}_rad.png", dpi=300)

                    plt.show()
                    sys.exit()




            
if __name__ == "__main__":
    main()


