from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Union
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, join

# --------------------------
# 1. Define multiple training UV + texture file paths
# --------------------------

def load_files():
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
# 2. Limb color definitions
# --------------------------
limb_colors: Dict[Tuple[int, int, int], int] = {
	(0, 255,255): 0,    # Head
	(0, 255, 0): 1,     # Torso
	(255, 0, 0): 2,     # Left Arm
	(0, 0, 255): 3,     # Right Arm
	(255, 255, 0): 4,   # Left Leg
	(255, 0, 255): 5,   # Right Leg
}
num_limbs = len(limb_colors)

# --------------------------
# 3. Build palette from all textures combined
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
# 4. Prepare training data from all pairs
# --------------------------
inputs_list: List[np.ndarray] = []
targets_list: List[int] = []

for uv_path, tex_path in zip(uv_files, tex_files):
	uv_img = np.array(Image.open(uv_path).convert("RGB"))
	tex_img = np.array(Image.open(tex_path).convert("RGB"))

	for y in range(UV_HEIGHT):
		for x in range(UV_WIDTH):
			limb_id = limb_colors.get(tuple(uv_img[y, x]), None)
			if limb_id is None:
				continue

			one_hot = np.zeros(num_limbs)
			one_hot[limb_id] = 1.0

			# You can switch here to local limb coords if you want, currently global normalized:
			x_norm = x / UV_WIDTH
			y_norm = y / UV_HEIGHT

			inputs_list.append(np.concatenate([one_hot, [x_norm, y_norm]]))
			targets_list.append(color_to_index[tuple(tex_img[y, x])])

# Check for CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

inputs = torch.tensor(np.array(inputs_list), dtype=torch.float32, device=device)
targets = torch.tensor(np.array(targets_list), dtype=torch.long, device=device)


class Model(nn.Module):
	def __init__(self, num_limbs: int, num_colors: int) -> None:
		super().__init__()

		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		match torch.cuda.is_available():
			case True:
				print("Using CUDA for training")
			case False:
				print("Using CPU for training")
	
		self.model = nn.Sequential(
			nn.Linear(num_limbs + 2, 64),
			nn.ReLU(),
			nn.Linear(64, 64),
			nn.ReLU(),
			nn.Linear(64, num_colors)
		)
		self.to(device)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.model(x)

	def save(self, path: Path = Path("weights/weights.pth")) -> None:
		"""
  		Saves the model weights to a specified path.

		Args:
			path (str): The file path to save the model weights to.
		"""
		torch.save(self.state_dict(), path)

	def load(self, path: Path = Path("weights/weights.pth")) -> bool:
		"""
 		Loads the model weights from a specified path.

		Args:
			path (str): The file path to load the model weights from.

		Returns:
			bool: True if loading was successful, False otherwise.
		"""
		if not Path(path).exists():
			print(f"Model weights not found at {path}")
			return False
		self.load_state_dict(torch.load(path, map_location=self.device))
		print(f"Model weights loaded from {path}")
		return True

	def train(self, epochs: int = 20000, lr: float = 0.002, betas: Tuple[float, float] = (0.7, 0.999)):
		"""
  		Train the model.

		Args:
			epochs (int, optional): Number of training epochs. Defaults to 20000.
			lr (float, optional): Learning rate. Defaults to 0.002.
			betas (tuple, optional): Betas for the Adam optimizer. Defaults to (0.7, 0.999).
		"""
		optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=betas)
		loss_fn = nn.CrossEntropyLoss()

		print(f"Starting training with {len(inputs)} samples...")

		for epoch in range(epochs):
			optimizer.zero_grad(set_to_none=True)
			logits: torch.Tensor = self(inputs)
			loss: torch.Tensor = loss_fn(logits, targets)
			loss.backward()
			optimizer.step()
			
			if epoch % 50 == 0:
				print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
				
			# Clear CUDA cache periodically to prevent memory issues
			if epoch % 1000 == 0 and torch.cuda.is_available():
				torch.cuda.empty_cache()

	def apply_to_new_pose(self, new_pose_path: str) -> Image.Image:
		"""
		Applies the trained model to a new pose UV map and saves the textured image.

		Args:
			new_pose_path (str): Path to the new pose UV map.
		"""
		uv_newpose = np.array(Image.open(new_pose_path).convert("RGB"))
		output = np.zeros((UV_HEIGHT, UV_WIDTH, 3), dtype=np.uint8)

		with torch.no_grad():  # Disable gradient computation for inference
			for y in range(UV_HEIGHT):
				for x in range(UV_WIDTH):
					limb_id = limb_colors.get(tuple(uv_newpose[y, x]), None)
					if limb_id is None:
						continue
					one_hot = np.zeros(num_limbs)
					one_hot[limb_id] = 1.0
					x_norm = x / UV_WIDTH
					y_norm = y / UV_HEIGHT
					inp = torch.tensor(np.concatenate([one_hot, [x_norm, y_norm]]), dtype=torch.float32, device=device)
					logits = self(inp.unsqueeze(0))

					pred_idx = int(torch.argmax(logits, dim=1).detach().item())  # Move to CPU before getting item
					output[y, x] = palette[pred_idx]

		# Convert output to RGBA format
		output_rgba = np.zeros((UV_HEIGHT, UV_WIDTH, 4), dtype=np.uint8)
  
		# Make background transparent: pixels not belonging to any limb remain transparent
		for y in range(UV_HEIGHT):
			for x in range(UV_WIDTH):
				if not np.any(output[y, x]):  # If pixel is still black (no limb assigned)
					output_rgba[y, x, 3] = 0  # Alpha = 0 (transparent)
				else:
					output_rgba[y, x, :3] = output[y, x]
					output_rgba[y, x, 3] = 255  # Alpha = 255 (opaque)
		return Image.fromarray(output_rgba, mode="RGBA")


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

	model: Model = Model(num_limbs, num_colors)
	print("Model created")
	
	# load the existing model weights if exists
	if not model.load():
		print("Starting training from scratch...")
		model.train(20000, 0.001)
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
