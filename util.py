import numpy as np
from typing import Tuple, Dict

# --------------------------
# 3. Helper functions for local coordinates per limb
# --------------------------
def get_limb_bounds(uv_img: np.ndarray, limb_id: int, limb_colors: Dict[Tuple[int, int, int], int]) -> Tuple[int, int, int, int]:
	"""
	Get the bounding box for a given limb ID in a specific UV image.
	
	Args:
		uv_img: UV image array (H, W, 3)
		limb_id: The limb ID to find bounds for
		
	Returns:
		Tuple of (min_x, min_y, max_x, max_y)
	"""
	uv_height, uv_width, _ = uv_img.shape
	limb_pixels = []
	for y in range(uv_height):
		for x in range(uv_width):
			if limb_colors.get(tuple(uv_img[y, x]), None) == limb_id:
				limb_pixels.append((x, y))
	
	if not limb_pixels:
		return (0, 0, 1, 1)  # Return minimal valid bounds if limb not found
		
	xs, ys = zip(*limb_pixels)
	return (min(xs), min(ys), max(xs), max(ys))

def get_local_coordinates(x: int, y: int, limb_id: int, limb_colors: Dict[Tuple[int, int, int], int], uv_img: np.ndarray) -> Tuple[float, float]:
	"""
	Convert global coordinates to local limb coordinates.
	
	Args:
		x, y: Global pixel coordinates
		limb_id: ID of the limb
		uv_img: UV image to get limb bounds from
		
	Returns:
		Tuple of (local_x, local_y) normalized to [0, 1]
	"""
	min_x, min_y, max_x, max_y = get_limb_bounds(uv_img, limb_id, limb_colors)
	
	# Avoid division by zero
	width = max(max_x - min_x, 1)
	height = max(max_y - min_y, 1)
	
	local_x = (x - min_x) / width
	local_y = (y - min_y) / height
	
	# Clamp to [0, 1] just in case
	local_x = max(0.0, min(1.0, local_x))
	local_y = max(0.0, min(1.0, local_y))
	
	return (local_x, local_y)
