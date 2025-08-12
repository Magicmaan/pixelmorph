# --------------------------
# 2. Limb color definitions
# --------------------------
import os
from PIL import Image
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


LIMB_COLORS: Dict[Tuple[int, int, int], int] = {
	(0, 255,255): 0,    # Head
	(0, 255, 0): 1,     # Torso
	(255, 0, 0): 2,     # Left Arm
	(0, 0, 255): 3,     # Right Arm
	(255, 255, 0): 4,   # Left Leg
	(255, 0, 255): 5,   # Right Leg
}
NUM_LIMBS = len(LIMB_COLORS)

UV_DIRECTORY = (Path("images") / "uv").resolve()
TEXTURE_DIRECTORY = (Path("images") / "texture").resolve().absolute()
TARGET_DIRECTORY = (Path("images") / "target").resolve().absolute()

sample_image = os.listdir(str(UV_DIRECTORY.resolve()))[0]
print(f"Using sample image: {sample_image}")
__sample_uv__ = np.array(Image.open(UV_DIRECTORY / sample_image).convert("RGB"))
UV_HEIGHT: int
UV_WIDTH: int
UV_HEIGHT, UV_WIDTH, _ = __sample_uv__.shape
