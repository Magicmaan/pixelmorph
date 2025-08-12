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
from util import get_local_coordinates

# --------------------------
# 1. Define multiple training UV + texture file paths
# --------------------------

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
		target_files += target_files[:len(target_files)//2]
	print(f"uv files: {uv_files}")
	print(f"texture files: {tex_files}")
	print(f"target files: {target_files}")
	return uv_files, tex_files, target_files

uv_files, tex_files, target_files = load_files()
# Load first UV to get image size
uv_sample = np.array(Image.open(uv_files[0]).convert("RGB"))
UV_HEIGHT: int
UV_WIDTH: int
UV_HEIGHT, UV_WIDTH, _ = uv_sample.shape




# --------------------------
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

				inputs_list.append(np.concatenate([one_hot, [local_x, local_y]]))
				targets_list.append(color_to_index[tuple(tex_img[y, x])])

# Check for CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

inputs = torch.tensor(np.array(inputs_list), dtype=torch.float32, device=device)
targets = torch.tensor(np.array(targets_list), dtype=torch.long, device=device)


# # --------------------------
# # 7. Apply to new pose
# # --------------------------
# def apply_to_new_pose(model: Model, uv_newpose_path: str, output_path: str = "newpose_textured.png") -> None:
# 	uv_newpose: np.ndarray = np.array(Image.open(uv_newpose_path).convert("RGB"))
# 	output: np.ndarray = np.zeros((UV_HEIGHT, UV_WIDTH, 3), dtype=np.uint8)

# 	with torch.no_grad():  # Disable gradient computation for inference
# 		for y in range(UV_HEIGHT):
# 			for x in range(UV_WIDTH):
# 				limb_id: Optional[int] = limb_colors.get(tuple(uv_newpose[y, x]), None)
# 				if limb_id is None:
# 					continue
# 				one_hot: np.ndarray = np.zeros(num_limbs)
# 				one_hot[limb_id] = 1.0
# 				x_norm: float = x / UV_WIDTH
# 				y_norm: float = y / UV_HEIGHT
# 				inp: torch.Tensor = torch.tensor(np.concatenate([one_hot, [x_norm, y_norm]]), dtype=torch.float32, device=device)
# 				logits: torch.Tensor = model(inp.unsqueeze(0))
				
# 				pred_idx: int = int(torch.argmax(logits, dim=1).cpu().item())  # Move to CPU before getting item
# 				output[y, x] = palette[pred_idx]

# 	Image.fromarray(output).save(output_path)
# 	print(f"Saved textured pose to {output_path}")

# def load_model_weights(model: Model, weights_path: str) -> bool:
# 	"""
# 	Loads model weights from a specified path.
# 	"""
# 	if not Path(weights_path).exists():
# 		print(f"Model weights not found at {weights_path}")
# 		return False
# 	model.load_state_dict(torch.load(weights_path, map_location=device))
# 	print(f"Model weights loaded from {weights_path}")
# 	return True

# --------------------------
# 8. Main execution
# --------------------------
def main() -> None:
	print(f"Training data size: {len(inputs)} samples")
	if torch.cuda.is_available():
		print(f"CUDA memory allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
  
	os.remove("weights/weights.pth") if Path("weights/weights.pth").exists() else None

	model: Model = Model(palette, num_colors, inputs, targets)
	print("Model created")
	
	# load the existing model weights if exists
	if not model.load():
		print("Starting training from scratch...")
		model.train(20000, 0.004)
		model.save()

	new_pose = model.apply_to_new_pose(target_files[0])  # Use the first target file as the new pose
 
	if new_pose is not None:
		new_pose.save(Path("output") / Path(target_files[0]).name)
		print(f"New pose textured image saved as '{target_files[0]}'")
	else:
		print("Failed to apply model to new pose.")

	# Clean up CUDA memory
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
		print("CUDA cache cleared")

if __name__ == "__main__":
	main()
