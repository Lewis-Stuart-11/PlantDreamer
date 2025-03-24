import random
from dataclasses import dataclass, field

import threestudio
import torch
import torch.nn as nn
import torch.nn.functional as F
from threestudio.models.background.base import BaseBackground
from threestudio.models.networks import get_encoding, get_mlp
from threestudio.utils.ops import get_activation
from threestudio.utils.typing import *


@threestudio.register("plantdreamer-background")
class StandardBackground(BaseBackground):
    @dataclass
    class Config(BaseBackground.Config):
        n_output_dims: int = 3
        background_color: Optional[Tuple[float, float, float]] = (0.75, 0.75, 0.75)

    cfg: Config

    def forward(self, dirs: Float[Tensor, "B H W 3"]) -> Float[Tensor, "B H W Nc"]:
        colour = torch.ones(*dirs.shape[:-1], self.cfg.n_output_dims).to(
                dirs
            ) * torch.as_tensor(self.cfg.background_color).to(dirs)    

        return colour
