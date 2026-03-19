import numpy as np
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
from utils.holo_utils import generate_checkerboard, generate_line_pattern, plot_phase_map


def main() -> None:
    DEFAULT_MONITOR = 1
    DISPLAY_DURATION_S = 0.5
    ROI_CENTER_X = 247
    ROI_CENTER_Y = 254
    ROI_HALF_WIDTH = 12
    ROI_HALF_HEIGHT = 12
    AVERAGING_FRAME_COUNT = 15
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


    summed_roi_values = []



    with VmbSystem.get_instance() as vmb:
        cams = vmb.get_all_cameras()
        if not cams:
            print("No cameras found.")
            return

        with cams[0] as cam:
            print("Using camera:", cam.get_name())

            plt.ion()
            fig, ax = plt.subplots()
            img_artist = None
            roi_rect = None

            for phase_val in tqdm(phase_values, desc="Generating, displaying, and capturing"):  
                #phase_val = 0.5 * np.pi
                phase = generate_line_pattern(x_dim, y_dim, line_spacing, orientation="vertical", max_phase=phase_val)
                # Process phase data into bitmap specific to the .67 PLM
                # This handles all quantization to appropriate phase displacement levels and mapping to the correct 2x2 electrode locations
                bmp = plm.process_phase_map(phase)
                bmp_img = Image.fromarray(bmp)
                roi_frame_sums = []

                with ImageWindow(fullscreen=True, monitor=DEFAULT_MONITOR) as win:
                    win.load(bmp_img)
                    start_time = perf_counter()

                    def stop_after_duration() -> None:
                        nonlocal img_artist, roi_rect
                        if perf_counter() - start_time >= DISPLAY_DURATION_S:
                            latest_img = None
                            x0 = x1 = y0 = y1 = None

                            for _ in range(AVERAGING_FRAME_COUNT):
                                frame = cam.get_frame()
                                frame_mono8 = frame.convert_pixel_format(PixelFormat.Mono8)
                                img = frame_mono8.as_opencv_image()
                                latest_img = img

                                x0 = max(0, ROI_CENTER_X - ROI_HALF_WIDTH)
                                x1 = min(img.shape[1], ROI_CENTER_X + ROI_HALF_WIDTH + 1)
                                y0 = max(0, ROI_CENTER_Y - ROI_HALF_HEIGHT)
                                y1 = min(img.shape[0], ROI_CENTER_Y + ROI_HALF_HEIGHT + 1)
                                roi_sum = np.sum(img[y0:y1, x0:x1], dtype=np.float64)
                                roi_frame_sums.append(roi_sum)
                                sleep(AVERAGING_FRAME_INTERVAL_S)

                            mean_roi_sum = float(np.mean(roi_frame_sums)) if roi_frame_sums else float('nan')
                            summed_roi_values.append(mean_roi_sum)
                            print(
                                f"Phase {phase_val:.4f} rad | ROI mean over {AVERAGING_FRAME_COUNT} frames: {mean_roi_sum:.2f}"
                            )

                            if latest_img is not None:
                                if img_artist is None:
                                    img_artist = ax.imshow(latest_img, cmap='gray')
                                    roi_rect = Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor='red', linewidth=2)
                                    ax.add_patch(roi_rect)
                                else:
                                    img_artist.set_data(latest_img)
                                    if roi_rect is not None:
                                        roi_rect.set_xy((x0, y0))
                                        roi_rect.set_width(x1 - x0)
                                        roi_rect.set_height(y1 - y0)

                            ax.set_title(f"Camera output during hologram display | ROI mean: {mean_roi_sum:.1f}")

                            fig.canvas.draw_idle()
                            plt.pause(0.001)
                            raise EventLoopExit

                    win.loop_callback = stop_after_duration
                    win.run()

            plt.ioff()
            plt.show()

            print("Summed ROI means per phase value:")
            print(summed_roi_values)
            np.save("summed_roi_values.npy", np.array(summed_roi_values))

            plt.figure()
            plt.plot(phase_values, summed_roi_values, marker='o')
            plt.xlabel('Phase value (rad)')
            plt.ylabel('Mean ROI sum')
            plt.title('Mean ROI sum vs phase value')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("roi_sum_vs_phase_1pi.png")
            plt.show()
            
if __name__ == "__main__":
    main()

