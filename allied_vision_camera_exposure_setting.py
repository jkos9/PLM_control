import sys
import matplotlib.pyplot as plt
from vmbpy import VmbSystem

def main():
    with VmbSystem.get_instance() as vmb:
        cams = vmb.get_all_cameras()
        if not cams:
            print("No cameras found.")
            return

        with cams[0] as cam:
            print("Using camera:", cam.get_name())

            # --- NEW CODE: Set Camera Exposure ---
            try:
                # 1. Turn off Auto Exposure so we can set it manually
                if hasattr(cam, 'ExposureAuto'):
                    cam.ExposureAuto.set('Off')
                
                # 2. Define your desired exposure time in microseconds (us)
                # Example: 10000.0 us = 10 ms
                desired_exposure_us = 200.0 
                
                # 3. Apply the exposure time, checking the exact feature name your camera supports
                if hasattr(cam, 'ExposureTime'):
                    cam.ExposureTime.set(desired_exposure_us)
                    print(f"Exposure successfully set to {cam.ExposureTime.get()} us")
                elif hasattr(cam, 'ExposureTimeAbs'): # Fallback for older camera models
                    cam.ExposureTimeAbs.set(desired_exposure_us)
                    print(f"Exposure successfully set to {cam.ExposureTimeAbs.get()} us")
                else:
                    print("Warning: Could not identify the exposure time feature on this camera.")
                    
            except Exception as e:
                print(f"Failed to set exposure: {e}")
            # ---------------------------------------

            
            # (Your frame acquisition loop would go here)

    sys.exit()

if __name__ == '__main__':
    main()