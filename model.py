import os
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Union
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, join

from constants import LIMB_COLORS, NUM_LIMBS, UV_HEIGHT, UV_WIDTH
from util import get_local_coordinates

class Model(nn.Module):
	def __init__(self, palette, num_colors: int, inputs, targets) -> None:
		super().__init__()

		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		match torch.cuda.is_available():
			case True:
				print("Using CUDA for training")
			case False:
				print("Using CPU for training")
	
		self.model = nn.Sequential(
			nn.Linear(NUM_LIMBS + 2, 128),
			nn.ReLU(),
			nn.Linear(128, 128),
			nn.ReLU(),
			nn.Linear(128, num_colors)
		)
		self.to(self.device)
  
		self.num_colors = num_colors
		self.inputs = inputs
		self.targets = targets
		self.palette = palette

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

		print(f"Starting training with {len(self.inputs)} samples...")

		for epoch in range(epochs):
			optimizer.zero_grad(set_to_none=True)
			logits: torch.Tensor = self(self.inputs)
			loss: torch.Tensor = loss_fn(logits, self.targets)
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
		Uses local coordinates per limb for inference.

		Args:
			new_pose_path (str): Path to the new pose UV map.
		"""
		uv_newpose = np.array(Image.open(new_pose_path).convert("RGB"))
		output = np.zeros((UV_HEIGHT, UV_WIDTH, 3), dtype=np.uint8)

		with torch.no_grad():  # Disable gradient computation for inference
			# Process each limb separately
			for limb_id in range(NUM_LIMBS):
				for y in range(UV_HEIGHT):
					for x in range(UV_WIDTH):
						pixel_limb_id = LIMB_COLORS.get(tuple(uv_newpose[y, x]), None)

						# Only process pixels that belong to the current limb
						if pixel_limb_id != limb_id:
							continue
							
						one_hot = np.zeros(NUM_LIMBS)
						one_hot[limb_id] = 1.0
						
						# Use local coordinates for this limb
						local_x, local_y = get_local_coordinates(x, y, limb_id, LIMB_COLORS, uv_newpose)

						inp = torch.tensor(np.concatenate([one_hot, [local_x, local_y]]), dtype=torch.float32, device=self.device)
						logits = self(inp.unsqueeze(0))

						pred_idx = int(torch.argmax(logits, dim=1).detach().item())
						output[y, x] = self.palette[pred_idx]

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
