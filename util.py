import numpy as np
from typing import List, Tuple, Dict
from PIL import Image

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

def get_limb_direction_features(x: int, y: int, limb_id: int, limb_colors: Dict[Tuple[int, int, int], int], uv_img: np.ndarray) -> Tuple[float, float, float, float]:
	"""
	Calculate directional features for a limb at a given position.
	
	Returns:
		- angle_from_center: Angle from limb center to current pixel
		- principal_angle: Main orientation angle of the limb (PCA)
		- distance_ratio: Normalized distance from center (0=center, 1=edge)
		- aspect_ratio: Width/height ratio of the limb
	"""
	uv_height, uv_width, _ = uv_img.shape
	
	# Get all pixels belonging to this limb
	limb_pixels = []
	for ly in range(uv_height):
		for lx in range(uv_width):
			if limb_colors.get(tuple(uv_img[ly, lx]), None) == limb_id:
				limb_pixels.append((lx, ly))
	
	if len(limb_pixels) < 3:  # Need at least 3 points for meaningful analysis
		return (0.0, 0.0, 0.0, 1.0)
	
	# Convert to numpy array for easier computation
	pixels = np.array(limb_pixels)
	
	# Calculate center of mass
	center_x = np.mean(pixels[:, 0])
	center_y = np.mean(pixels[:, 1])
	
	# Angle from center to current pixel
	angle_from_center = np.arctan2(y - center_y, x - center_x)
	# Normalize to [0, 1]
	angle_from_center = (angle_from_center + np.pi) / (2 * np.pi)
	
	# Principal Component Analysis for main limb direction
	centered_pixels = pixels - np.array([center_x, center_y])
	if len(centered_pixels) > 1:
		cov_matrix = np.cov(centered_pixels.T)
		eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
		principal_vector = eigenvectors[:, np.argmax(eigenvalues)]
		principal_angle = np.arctan2(principal_vector[1], principal_vector[0])
		# Normalize to [0, 1]
		principal_angle = (principal_angle + np.pi) / (2 * np.pi)
	else:
		principal_angle = 0.0
	
	# Distance from center (normalized by max distance in limb)
	current_dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
	max_dist = np.max(np.sqrt(np.sum(centered_pixels**2, axis=1))) if len(centered_pixels) > 0 else 1.0
	distance_ratio = min(current_dist / max_dist, 1.0) if max_dist > 0 else 0.0
	
	# Aspect ratio (width/height of bounding box)
	min_x, min_y, max_x, max_y = get_limb_bounds(uv_img, limb_id, limb_colors)
	width = max_x - min_x + 1
	height = max_y - min_y + 1
	aspect_ratio = width / height if height > 0 else 1.0
	# Normalize aspect ratio to [0, 1] range (assuming max aspect ratio of 5)
	aspect_ratio = min(aspect_ratio / 5.0, 1.0)
	
	return (angle_from_center, principal_angle, distance_ratio, aspect_ratio)

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

def from_spritesheet(image: Image.Image, cell_width: int, cell_height: int, desired_size: Tuple[int, int]) -> List[Image.Image]:
	"""
	Extracts images from a spritesheet.
	
	Args:
		Image: The spritesheet image.
		cell_width: The width of each cell in the spritesheet.
		cell_height: The height of each cell in the spritesheet.

	Returns:
		List of extracted images.
	"""
	width, height = image.size
	num_cols = width // cell_width
	num_rows = height // cell_height

	cells = []
	for row in range(num_rows):
		for col in range(num_cols):
			img = image.crop((
				col * cell_width, 
				row * cell_height, 
				(col + 1) * cell_width, 
				(row + 1) * cell_height))
			# Ensure the image is not empty
			if img.size == 0 or img.size[0] <= 0 or img.size[1] <= 0:
				print(f"Warning: Empty image at row {row}, col {col} in spritesheet")
			
			if not img.getbbox(alpha_only=True):
				print(f"Warning: Image at row {row}, col {col} in spritesheet is empty")
			
   
   
			# Paste the cropped image onto a blank 32x32 canvas, centered
			image_resized = Image.new("RGBA", desired_size, (0, 0, 0, 0))  # Transparent background
			offset_x = (desired_size[0] - img.width) // 2
			offset_y = (desired_size[1] - img.height) // 2
			image_resized.paste(img, (offset_x, offset_y))
   
			cells.append(image_resized)
   
	return cells

def to_spritesheet(images: List[Image.Image], cell_width: int, cell_height: int) -> Image.Image:
	"""
	Creates a spritesheet from a list of images.
	
	Args:
		images: List of images to combine into a spritesheet.
		cell_width: Width of each cell in the spritesheet.
		cell_height: Height of each cell in the spritesheet.
		image_width: Width of the final spritesheet image.
		image_height: Height of the final spritesheet image.

	Returns:
		A single Image object containing the spritesheet.
	"""
	spritesheet = Image.new("RGBA", (cell_width * len(images), cell_height), (0, 0, 0, 0))  # Transparent background
 
	for idx, img in enumerate(images):
		image_resized = Image.new("RGBA", (cell_width, cell_height), (0, 0, 0, 0))
		offset_x = (cell_width - img.width) // 2
		offset_y = (cell_height - img.height) // 2
		image_resized.paste(img, (offset_x, offset_y))
		spritesheet.paste(image_resized, (idx * cell_width, 0))

	return spritesheet
