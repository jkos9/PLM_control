"""Simple end-to-end example for showing an image on a TI PLM p67 NIR display.

Usage:
    python examples/p67nir_show_image.py

Optional overrides:
    python examples/p67nir_show_image.py --image path/to/image.png --monitor -1
"""

import argparse
from pathlib import Path
import sys
import pathlib

try:
    import numpy as np
    from PIL import Image
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: "
        f"{exc.name}. Install requirements with `python -m pip install -e .[display]`."
    ) from exc


def _import_ti_plm():
    """Import ti_plm from site-packages or local src checkout."""
    try:
        from ti_plm import PLM
        from ti_plm.display import ImageWindow
        return PLM, ImageWindow
    except ModuleNotFoundError:
        # Allow running this example directly from a source checkout without installation.
        repo_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repo_root / "src"))
        try:
            from ti_plm import PLM
            from ti_plm.display import ImageWindow
            return PLM, ImageWindow
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "Missing dependency: "
                f"{exc.name}. From repo root run: `python -m pip install -e .[display]`"
            ) from exc


PLM, ImageWindow = _import_ti_plm()

HERE = Path(__file__).resolve().parent
DEFAULT_IMAGE_PATH = HERE / "dlp_logo_8bit.png"
#DEFAULT_IMAGE_PATH = HERE / "bird_p67_7cm.png"


DEFAULT_MONITOR = 1




def main() -> None:
    here = pathlib.Path(__file__).parent  # change to `pathlib.Path('.')` if running in a notebook
    out = here / 'out'
    out.mkdir(exist_ok=True)

    # Initialize PLM object pre-configured with .67 parameters
    plm = PLM.from_db('p67nir')

    # Read image file containing phase info encoded as 8-bit values and scale it between 0 and 2pi
    img = Image.open(DEFAULT_IMAGE_PATH)
    phase = np.asarray(img, dtype=float) / 255 * 2 * np.pi

    x_dim = 904
    y_dim = 800
    # reshape phase map to match PLM dimensions (if needed)
    phase = phase[:y_dim, :x_dim]

    # PIL loads images with channel in last dimension slot, but we need it in the first
    # This only applies to RGB images (e.g. if separate phase patterns were encoded into each RGB channel)
    if len(phase.shape) > 2:
        phase = np.moveaxis(phase, 2, 0)

    # Process phase data into bitmap specific to the .67 PLM
    # This handles all quantization to appropriate phase displacement levels and mapping to the correct 2x2 electrode locations
    bmp = plm.process_phase_map(phase)
    bmp_img = Image.fromarray(bmp)

    with ImageWindow(fullscreen=True, monitor=DEFAULT_MONITOR) as win:
        win.load(bmp_img)
        win.run()


if __name__ == "__main__":
    main()


