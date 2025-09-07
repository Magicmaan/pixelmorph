import os
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Union
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, join

from constants import LIMB_COLORS, NUM_LIMBS
from model import Model
from util import get_local_coordinates, get_limb_direction_features, to_spritesheet

# --------------------------
# 1. Define multiple training UV + texture file paths
# --------------------------

IS_SPRITESHEET: bool = True  # Set to True if using a spritesheet

def from_spritesheet(image: Image.Image, cell_width: int, cell_height: int) -> List[Image.Image]:
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
				continue
			
			if not img.getbbox(alpha_only=True):
				print(f"Warning: Image at row {row}, col {col} in spritesheet is empty")
				continue
			
			# Paste the cropped image onto a blank 32x32 canvas, centered
			canvas = Image.new("RGBA", (32, 32), (0, 0, 0, 0))  # Transparent background
			offset_x = (32 - img.width) // 2
			offset_y = (32 - img.height) // 2
			canvas.paste(img, (offset_x, offset_y))
			img = canvas
   
			cells.append(img)
   
	return cells



def load_files(duplicate: bool = False) -> Tuple[List[str], List[str], List[str]]:
	"""
	Load UV and texture files for training.
	"""
	images_path = Path("images")
 
	uv_files: list[str] = []
	tex_files: list[str] = []
	target_files: list[str] = []
 
	for file in listdir(images_path / "uv"):
		if isfile(join(images_path / "uv", file)) and file.endswith(".png"):
			uv_files.append((images_path / "uv" / file).resolve().as_posix())
	uv_files.sort()  # Ensure consistent order
 
 
	for file in listdir(images_path / "texture"):
		if isfile(join(images_path / "texture", file)) and file.endswith(".png"):
			tex_files.append((images_path / "texture" / file).resolve().as_posix())
	tex_files.sort()  # Ensure consistent order
	
	assert len(uv_files) == len(tex_files), "UV and texture file counts must match"
 
	for file in listdir(images_path / "target"):
		if isfile(join(images_path / "target", file)) and file.endswith(".png"):
			target_files.append((images_path / "target" / file).resolve().as_posix())
	
	if duplicate:
    #  add duplicate files for training
		uv_files += uv_files[:len(uv_files)//2]
		tex_files += tex_files[:len(tex_files)//2]
		# target_files += target_files[:len(target_files)//2]
	print(f"uv files: {uv_files}")
	print(f"texture files: {tex_files}")
	print(f"target files: {target_files}")
	return uv_files, tex_files, target_files

uv_files, tex_files, target_files = load_files(duplicate=True)  # Use data duplication
# Load first UV to get image size
uv_sample = np.array(Image.open(uv_files[0]).convert("RGB"))
UV_HEIGHT: int
UV_WIDTH: int
UV_HEIGHT, UV_WIDTH, _ = uv_sample.shape

target_images = [Image.open(path).convert("RGB") for path in target_files]
# split the image into individual sprites if using a spritesheet
# also resizes the cells to 32x32 to match the model input
if IS_SPRITESHEET:
	new_target_images = []
	for img in target_images:
		# Extract images from spritesheet
		img_sprites = from_spritesheet(img, 16, 32)
		if img_sprites:
			new_target_images.extend(img_sprites)
	target_images = new_target_images


# 4. Build palette from all textures combined
# --------------------------
all_colors: Set[Tuple[int, int, int]] = set()
for tex_path in tex_files:
	tex_img: np.ndarray = np.array(Image.open(tex_path).convert("RGB"))
	unique_colors: Set[Tuple[int, int, int]] = {tuple(c) for row in tex_img for c in row}
	all_colors.update(unique_colors)

palette: List[Tuple[int, int, int]] = sorted(list(all_colors))
color_to_index: Dict[Tuple[int, int, int], int] = {c: i for i, c in enumerate(palette)}
num_colors = len(palette)
print(f"Palette size: {num_colors} colors (combined from all training textures)")

# --------------------------
# 5. Prepare training data from all pairs using local coordinates per limb
# --------------------------
inputs_list: List[np.ndarray] = []
targets_list: List[int] = []

for idx, (uv_path, tex_path) in enumerate(zip(uv_files, tex_files)):
	uv_img = np.array(Image.open(uv_path).convert("RGB"))
	tex_img = np.array(Image.open(tex_path).convert("RGB"))

	# Process each limb separately
	for limb_id in range(NUM_LIMBS):
		# Get all pixels for this specific limb
		for y in range(UV_HEIGHT):
			for x in range(UV_WIDTH):
				pixel_limb_id = LIMB_COLORS.get(tuple(uv_img[y, x]), None)
				
				# Only process pixels that belong to the current limb
				if pixel_limb_id != limb_id:
					continue

				one_hot = np.zeros(NUM_LIMBS)
				one_hot[limb_id] = 1.0

				# Use local coordinates relative to this limb's bounding box
				local_x, local_y = get_local_coordinates(x, y, limb_id, LIMB_COLORS, uv_img)
				
				# Get directional features for this limb
				angle_from_center, principal_angle, distance_ratio, aspect_ratio = get_limb_direction_features(
					x, y, limb_id, LIMB_COLORS, uv_img
				)

				inputs_list.append(np.concatenate([
					one_hot, 
					[local_x, local_y], 
					[angle_from_center, principal_angle, distance_ratio, aspect_ratio]
				]))
				targets_list.append(color_to_index[tuple(tex_img[y, x])])

# Check for CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

inputs = torch.tensor(np.array(inputs_list), dtype=torch.float32, device=device)
targets = torch.tensor(np.array(targets_list), dtype=torch.long, device=device)

# --------------------------
# 8. Main execution
# --------------------------
def main() -> None:
	print(f"Training data size: {len(inputs)} samples")
	if torch.cuda.is_available():
		print(f"CUDA memory allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
  
	# os.remove("weights/weights.pth") if Path("weights/weights.pth").exists() else None

	model: Model = Model(palette, num_colors, inputs, targets)
	print("Model created")
	
	# load the existing model weights if exists
	if not model.load():
		print("Starting training from scratch...")
		# Try better training parameters
		model.train(20000, 0.001)  # More epochs, lower learning rate
		model.save()

	if IS_SPRITESHEET:
		new_sprites = [model.apply_to_new_pose(t) for t in target_images]
		if new_sprites:
			spritesheet = to_spritesheet(new_sprites, 16, 32)
			spritesheet.save("output/spritesheet.png")
	else:
		for idx,t in enumerate(target_images):
			new_pose = model.apply_to_new_pose(t)

			if new_pose is not None:
				new_pose.save(Path("output") / Path(f"new_pose_{idx}.png"))
				print(f"New pose textured image saved as 'new_pose_{idx}.png'")
			else:
				print("Failed to apply model to new pose.")

	# Clean up CUDA memory
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
		print("CUDA cache cleared")

if __name__ == "__main__":
	main()
