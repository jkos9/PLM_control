import ctypes
import numpy as np
import os

# 1. Point to your Xeneth DLL
# Usually found in C:\Program Files\Xeneth\SDK\Lib\x64\XCamera.dll
dll_path = r"C:\Program Files (x86)\Common Files\XenICs\Runtime\Drivers\libusb0_x64.dll"

if not os.path.exists(dll_path):
    raise FileNotFoundError(f"Could not find XCamera.dll at {dll_path}")

# Load the library
xeneth = ctypes.cdll.LoadLibrary(dll_path)

def capture_frame():
    handle = None
    try:
        # 2. Open the camera (cam://0 is the first camera found)
        # Xeneth function: XCameraOpen(const char * url, int flags, int reserved)
        print("Opening camera...")
        handle = xeneth.XCameraOpen(b"cam://0", 0, 0)
        
        if handle == 0 or handle == -1:
            print("Failed to open camera. Is it plugged in?")
            return

        # 3. Start Acquisition
        xeneth.XCameraStart(handle)

        # 4. Get Camera Dimensions
        width = xeneth.XCameraGetWidth(handle)
        height = xeneth.XCameraGetHeight(handle)
        print(f"Camera resolution: {width} x {height}")

        # 5. Capture a frame
        # We create a buffer to hold the image (usually 16-bit for SWIR)
        frame_buffer = np.zeros((height, width), dtype=np.uint16)
        
        # XCameraCapture(handle, buffer, size, timeout)
        # 1 indicates we want the latest frame
        status = xeneth.XCameraGetFrame(handle, 1, frame_buffer.ctypes.data, frame_buffer.nbytes, 1000)

        if status == 0: # Success code for Xeneth is 0
            print("Frame captured successfully!")
            print("Mean pixel value:", np.mean(frame_buffer))
            return frame_buffer
        else:
            print(f"Capture failed with status: {status}")

    finally:
        # 6. Cleanup
        if handle and handle != -1:
            xeneth.XCameraStop(handle)
            xeneth.XCameraClose(handle)
            print("Camera connection closed.")

if __name__ == "__main__":
    image = capture_frame()
    
    # If you want to see it (requires matplotlib)
    if image is not None:
        try:
            import matplotlib.pyplot as plt
            plt.imshow(image, cmap='gray')
            plt.colorbar()
            plt.show()
        except ImportError:
            print("Install matplotlib to visualize the image.")


    