import numpy as np
import matplotlib.pyplot as plt


def generate_checkerboard(x_dim: int, y_dim: int, tile_size: int) -> np.ndarray:
    """Generate a checkerboard phase pattern with values in [0, 2pi)."""
    yy, xx = np.indices((y_dim, x_dim))
    checkerboard = ((xx // tile_size + yy // tile_size) % 2).astype(float)
    return checkerboard * 2 *np.pi


def generate_line_pattern(
    x_dim: int,
    y_dim: int,
    spacing: int,
    orientation: str = "vertical",
    max_phase: float = 2 * np.pi
) -> np.ndarray:
    """Generate alternating equal-width 0/1 line bands with values in [0, 2pi]."""
    if spacing <= 0:
        raise ValueError("spacing must be a positive integer")
    if orientation not in {"vertical", "horizontal"}:
        raise ValueError("orientation must be either 'vertical' or 'horizontal'")

    yy, xx = np.indices((y_dim, x_dim))
    bands = ((xx // spacing) % 2).astype(float) if orientation == "vertical" else ((yy // spacing) % 2).astype(float)
    phase = bands * max_phase
    return phase

def generate_global_pattern(x_dim: int, y_dim: int, max_phase: float = 2 * np.pi) -> np.ndarray:
    """Generate a global pattern where all pixels have the same phase value."""
    return np.full((y_dim, x_dim), max_phase)

def plot_phase_map(phase: np.ndarray) -> None:
    """Utility function to visualize the generated phase map."""
    plt.imshow(phase, cmap='viridis')
    plt.colorbar(label='Phase (radians)')
    plt.title('Generated Phase Map')
    plt.xlabel('X-axis (pixels)')
    plt.ylabel('Y-axis (pixels)')
    plt.savefig('phase_map.png', dpi=300)
    #plt.show()


def generate_square_test_patterns(x_dim: int, y_dim: int) -> list[np.ndarray]:
    """Generate an array of test patterns with different squares in the centre."""
    
    patterns = []
    for i in range(1, 5):
        base_pattern = np.zeros((y_dim, x_dim), dtype=float)
        
        # The size of the square increases quadratically with i
        sq_size = i**2 
        
        # Calculate the starting and ending indices for the centered square.
        # Boundary checks prevent indexing errors if sq_size > array dimensions.
        start_y = max(0, (y_dim - sq_size) // 2)
        end_y = min(y_dim, start_y + sq_size)
        
        start_x = max(0, (x_dim - sq_size) // 2)
        end_x = min(x_dim, start_x + sq_size)
        
        # Set the center region to Pi
        base_pattern[start_y:end_y, start_x:end_x] = np.pi
        
        patterns.append(base_pattern)

    return patterns


def generate_square_test_patterns(x_dim: int, y_dim: int) -> list[np.ndarray]:
    """Generate an array of test patterns with different squares in the centre."""
    
    patterns = []
    for i in range(1, 5):
        base_pattern = np.zeros((y_dim, x_dim), dtype=float)
        
        # The size of the square increases quadratically with i
        sq_size = i**2 
        
        # Calculate the starting and ending indices for the centered square.
        # Boundary checks prevent indexing errors if sq_size > array dimensions.
        start_y = max(0, (y_dim - sq_size) // 2)
        end_y = min(y_dim, start_y + sq_size)
        
        start_x = max(0, (x_dim - sq_size) // 2)
        end_x = min(x_dim, start_x + sq_size)
        
        # Set the center region to Pi
        base_pattern[start_y:end_y, start_x:end_x] = np.pi
        
        patterns.append(base_pattern)

    return patterns



