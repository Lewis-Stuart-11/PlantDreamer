import threestudio
from packaging.version import Version

if hasattr(threestudio, "__version__") and Version(threestudio.__version__) >= Version(
    "0.2.0"
):
    pass
else:
    if hasattr(threestudio, "__version__"):
        print(f"[INFO] threestudio version: {threestudio.__version__}")
    raise ValueError(
        "threestudio version must be >= 0.2.0, please update threestudio by pulling the latest version from github"
    )

from .background import gaussian_background
from .geometry import exporter, gaussian_base, gaussian_io
from .guidance import stable_diffusion_lora_and_depth_controlnet_guidance
from .renderer import diff_gaussian_rasterizer
from .system import gaussian_splatting
