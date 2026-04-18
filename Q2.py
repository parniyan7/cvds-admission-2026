import numpy as np


def swap(coords: np.ndarray) -> np.ndarray:
    """
    Swaps the x and y coordinates of each bounding box.
    
    Input format:  [x1, y1, x2, y2, class_id]
    Output format: [y1, x1, y2, x2, class_id]
    
    What was wrong in the original code:
    - The tuple assignment was malformed and overwrote x values with y values.
    - It modified the original array in place (bad practice for this kind of function).
    - Coordinates were not correctly swapped for both corners.
    
    How I fixed it:
    - Created a copy of the input array so the original remains unchanged.
    - Properly swapped each pair of coordinates (x1↔y1 and x2↔y2).
    - Kept the class_id untouched.
    """
    # Create a copy to avoid modifying the original array (important for safety)
    result = coords.copy()
    
    # Swap x1 with y1 for every row
    result[:, 0], result[:, 1] = coords[:, 1], coords[:, 0]
    
    # Swap x2 with y2 for every row
    result[:, 2], result[:, 3] = coords[:, 3], coords[:, 2]
    
    return result


# ========================
# Test
# ========================
if __name__ == "__main__":
    coords = np.array([
        [10, 5, 15, 6, 0],
        [11, 3, 13, 6, 0],
        [5, 3, 13, 6, 1],
        [4, 4, 13, 6, 1],
        [6, 5, 13, 16, 1]
    ])
    
    swapped_coords = swap(coords)
    print(swapped_coords)