import math
import os
import random
import sys
import copy
from dataclasses import dataclass, field
from datetime import datetime
from typing import NamedTuple

import numpy as np
import threestudio
import torch
import torch.nn as nn
import torch.nn.functional as F
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from threestudio.models.geometry.base import BaseGeometry
from threestudio.utils.misc import C
from threestudio.utils.typing import *

from .gaussian_io import GaussianIO
import open3d as o3d

from .plant_handler import get_plant_type

C0 = 0.28209479177387814


def RGB2SH(rgb):
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    return sh * C0 + 0.5


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def gaussian_3d_coeff(xyzs, covs):
    # xyzs: [N, 3]
    # covs: [N, 6]
    x, y, z = xyzs[:, 0], xyzs[:, 1], xyzs[:, 2]
    a, b, c, d, e, f = (
        covs[:, 0],
        covs[:, 1],
        covs[:, 2],
        covs[:, 3],
        covs[:, 4],
        covs[:, 5],
    )

    # eps must be small enough !!!
    inv_det = 1 / (
        a * d * f + 2 * e * c * b - e**2 * a - c**2 * d - b**2 * f + 1e-24
    )
    inv_a = (d * f - e**2) * inv_det
    inv_b = (e * c - b * f) * inv_det
    inv_c = (e * b - c * d) * inv_det
    inv_d = (a * f - c**2) * inv_det
    inv_e = (b * c - e * a) * inv_det
    inv_f = (a * d - b**2) * inv_det

    power = (
        -0.5 * (x**2 * inv_a + y**2 * inv_d + z**2 * inv_f)
        - x * y * inv_b
        - x * z * inv_c
        - y * z * inv_e
    )

    power[power > 0] = -1e10  # abnormal values... make weights 0

    return torch.exp(power)


def build_rotation(r):
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device="cuda")

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


def safe_state(silent):
    old_f = sys.stdout

    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(
                        x.replace(
                            "\n",
                            " [{}]\n".format(
                                str(datetime.now().strftime("%d/%m %H:%M:%S"))
                            ),
                        )
                    )
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))

def transform_point_cloud(xyz, dirs):
    """
    Creates a transformation matrix based on directions and applies it to the point cloud.
    xyz: NumPy array of shape (N, 3)
    dirs: String, directions for transformation (e.g., '+y,+x,+z')
    """
    valid_dirs = ["+x", "+y", "+z", "-x", "-y", "-z"]
    dir2vec = {
        "+x": np.array([1, 0, 0]),
        "+y": np.array([0, 1, 0]),
        "+z": np.array([0, 0, 1]),
        "-x": np.array([-1, 0, 0]),
        "-y": np.array([0, -1, 0]),
        "-z": np.array([0, 0, -1]),
    }
    # Initialize transformation matrix

    T = np.zeros((3, 3))

    # Map directions to transformation matrix
    for i, dir in enumerate(dirs.split(',')):
        if dir in valid_dirs:
            T[:, i] = dir2vec[dir]
        else:
            raise ValueError(f"Invalid direction: {dir}")

    # Apply transformation
    transformed_xyz = np.dot(xyz, T)
    return transformed_xyz, T

class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array


class Camera(NamedTuple):
    FoVx: torch.Tensor
    FoVy: torch.Tensor
    camera_center: torch.Tensor
    image_width: int
    image_height: int
    world_view_transform: torch.Tensor
    full_proj_transform: torch.Tensor


@threestudio.register("plantdreamer-base")
class GaussianBaseModel(BaseGeometry, GaussianIO):
    @dataclass
    class Config(BaseGeometry.Config):
        max_num: int = 500000
        sh_degree: int = 0
        position_lr: Any = 0.001
        scale_lr: Any = 0.003
        feature_lr: Any = 0.01
        opacity_lr: Any = 0.05
        scaling_lr: Any = 0.005
        rotation_lr: Any = 0.005
        pred_normal: bool = False
        normal_lr: Any = 0.001

        densification_interval: int = 50
        prune_interval: int = 50
        opacity_reset_interval: int = 100000
        densify_from_iter: int = 100
        prune_from_iter: int = 100
        densify_until_iter: int = 2000
        prune_until_iter: int = 2000
        densify_grad_threshold: Any = 0.01
        min_opac_prune: Any = 0.005
        split_thresh: Any = 0.02
        radii2d_thresh: Any = 1000

        sphere: bool = False
        prune_big_points: bool = False
        color_clip: Any = 2.0

        geometry_convert_from: str = ""
        initialisation_type: str = ""
        init_num_pts: int = 100
        pc_init_radius: float = 0.8
        opacity_init: float = 0.1

        num_downsample_points: int = 300000
        point_cloud_colour_type: int = 0
        
        use_original_depth: bool = True

        cull_gaussian_start_epoch: int = 500
        cull_gaussian_rate: int = 100
        cull_gaussian_std_factor: float = 6

        plant_type: str = ""

    cfg: Config

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def gaussian_volumes(self):
        return torch.sqrt(torch.sum(torch.pow(torch.exp(self.get_scaling.clone()), 2), axis=1))

    def cull_large_gaussians(self, global_step):
        
        if (global_step > self.cfg.cull_gaussian_start_epoch) and ((global_step + 1) % self.cfg.cull_gaussian_rate == 0):
            # Calculate Gaussian Volumes
            gaussian_sizes = self.gaussian_volumes()
    
            # Get the mean and std of the gaussian sizes
            mean_and_std = torch.std_mean(gaussian_sizes)

            outlier_range = mean_and_std[1] + (mean_and_std[0] * self.cfg.cull_gaussian_std_factor)

            # Cull Gaussians with a size larger than the outlier range
            valid_gaussians = ~(gaussian_sizes < outlier_range)
    
            print(f"Culling {(valid_gaussians).sum()} Gaussians")
    
            self.prune_points(valid_gaussians)

    def scale_and_centre_pointcloud(self, point_cloud, target_range: int=(-1.0, 1.0)):
        coords = np.asarray(point_cloud.points)

        # Centre the point cloud to the origin based on the average position of all points
        centre = (np.max(coords, axis=0) + np.min(coords, axis=0))/2
        coords -= centre

        # Calculate the range of each dimension
        ranges = np.max(coords, axis=0) - np.min(coords, axis=0)
 
        # Find the index of the longest dimension
        longest_dim_idx = np.argmax(ranges)
 
        # Calculate the scaling factor for the longest dimension
        scaling_factor = (target_range[1] - target_range[0]) / ranges[longest_dim_idx]
        scaling_factor *= 2
 
        # Scale all dimensions proportionally
        scaled_coords = coords * scaling_factor

        point_cloud.points = o3d.utility.Vector3dVector(scaled_coords)

        return point_cloud, scaling_factor, torch.from_numpy(centre).to("cuda:0")

    def downsample_pointcloud(self, point_cloud, num_downsample_points: int=300000, acceptable_range: Tuple=(0.75, 1.25)):
        coords = np.asarray(point_cloud.points)

        subsampled_point_cloud = copy.deepcopy(point_cloud)

        ranges = np.max(coords, axis=0) - np.min(coords, axis=0)
        volume = ranges[0] * ranges[1] * ranges[2]
 
        volume_scale = 100
        voxel_size = (volume*volume_scale)/num_downsample_points
        
        # If the number of coords is larger than the acceptable range, then downsample this point cloud
        if coords.shape[0] > num_downsample_points * acceptable_range[1]:

            num_subsample_attempts = 50

            # Keep attempting to downsample until either the point cloud has the correct number of points, or the number
            # of attemprs expires
            i = 0
            while i < num_subsample_attempts:
                
                # If the number of points is lower than the acceptable range, then decrease the volume scale 
                if np.asarray(subsampled_point_cloud.points).shape[0] < num_downsample_points * acceptable_range[0]:
                    volume_scale -= 10 * (num_subsample_attempts-i)/num_subsample_attempts
                
                # If the number of points is larger than the acceptable range, then increase the volume scale  
                elif np.asarray(subsampled_point_cloud.points).shape[0]  > num_downsample_points * acceptable_range[1]:
                    volume_scale += 10 * (num_subsample_attempts-i)/num_subsample_attempts
                else:
                    break

                subsampled_point_cloud = copy.deepcopy(point_cloud)

                # Downsample point cloud
                voxel_size = (volume*volume_scale)/num_downsample_points
                subsampled_point_cloud = subsampled_point_cloud.voxel_down_sample(voxel_size)

                i += 1

        # Remove outliers
        subsampled_point_cloud, _ = subsampled_point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=3)

        return subsampled_point_cloud

    def rescale_gaussian_splat_to_original(self):
        # Rescale and translate positions back to the original coordinate frame
        rescaled_positions = self._xyz.clone()
        rescaled_positions /= self.scaling_factor
        rescaled_positions += self.centre_translation
        self._xyz = nn.Parameter((rescaled_positions).requires_grad_(True))
        
        # Rescale Gaussians to fit original coordinate frame
        rescaled_scales = self._scaling.clone()
        self._scaling =  nn.Parameter((rescaled_scales - np.log(self.scaling_factor)).requires_grad_(True)) 

    def rescale_gaussian_splat_to_training(self):
        # Rescale and translate positions back to the original coordinate frame
        rescaled_positions = self._xyz.clone()
        rescaled_positions -= self.centre_translation
        rescaled_positions *= self.scaling_factor
        self._xyz = nn.Parameter((rescaled_positions).requires_grad_(True))
        
        # Rescale Gaussians to fit original coordinate frame
        rescaled_scales = self._scaling.clone()
        self._scaling =  nn.Parameter((rescaled_scales + np.log(self.scaling_factor)).requires_grad_(True)) 

    def save_and_rescale_ply(self, save_path):
        self.rescale_gaussian_splat_to_original()
        self.save_ply(save_path)
        self.rescale_gaussian_splat_to_training()

    def configure(self) -> None:
        super().configure()
        self.active_sh_degree = 0
        self.max_sh_degree = self.cfg.sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        if self.cfg.pred_normal:
            self._normal = torch.empty(0)
        self.optimizer = None
        self.setup_functions()

        # Get the plant type from the defined list of plant types 
        self.plant_type = get_plant_type(self.cfg.plant_type)

        self.scaling_factor = 1
        self.centre_translation = np.array([0,0,0])

        if self.cfg.initialisation_type is None or self.cfg.initialisation_type == "":
            raise Exception("Initialisation type must be provided")
        
        # Initialise using L-Systems
        if self.cfg.initialisation_type == "l-system":

            l_systems_mesh_path = self.plant_type.generate_l_system_mesh()

            self.cfg.geometry_convert_from = l_systems_mesh_path

        # Initialise using a defined checkpoint
        if self.cfg.initialisation_type == "checkpoint":
            if self.cfg.geometry_convert_from is None or not os.path.exists(self.cfg.geometry_convert_from) or not self.cfg.geometry_convert_from.endswith(".ckpt"):
                raise Exception("Must provide valid ckpt file to initialise the scene")

            ckpt_dict = torch.load(self.cfg.geometry_convert_from)
            num_pts = ckpt_dict["state_dict"]["geometry._xyz"].shape[0]
            pcd = BasicPointCloud(
                points=np.zeros((num_pts, 3)),
                colors=np.zeros((num_pts, 3)),
                normals=np.zeros((num_pts, 3)),
            )
            self.create_from_pcd(pcd, 10)
            self.training_setup()
            new_ckpt_dict = {}
            for key in self.state_dict():
                if ckpt_dict["state_dict"].__contains__("geometry." + key):
                    new_ckpt_dict[key] = ckpt_dict["state_dict"]["geometry." + key]
                else:
                    new_ckpt_dict[key] = self.state_dict()[key]
            self.load_state_dict(new_ckpt_dict)
        
        # Initialise using .ply file
        elif self.cfg.initialisation_type == "ply" or (self.cfg.geometry_convert_from is not None and self.cfg.geometry_convert_from.endswith(".ply")):
            if not os.path.exists(self.cfg.geometry_convert_from):
                raise Exception("Must provide valid ply file to initialise the scene")

            threestudio.info(
                "Loading point cloud/mesh from %s" % self.cfg.geometry_convert_from
            )

            plydata = PlyData.read(self.cfg.geometry_convert_from)

            # If features are included in the meta data, then it must be a 3DGS file
            if "f_dc_0" in plydata['vertex'].data.dtype.names:
                print("Loading in 3DGS Scene")

                # Load in 3DGS scene
                self.load_ply(self.cfg.geometry_convert_from)
            else:

                # If face is included in the metadata, then assume it is a mesh
                if 'face' in str(plydata.elements):
                    print("Loading in Mesh")

                    mesh = o3d.io.read_triangle_mesh(self.cfg.geometry_convert_from)

                    # Set higher mesh surface density
                    self.cfg.opacity_init = 0.7

                    # Sample point cloud from mesh
                    point_cloud = mesh.sample_points_uniformly(number_of_points=self.cfg.num_downsample_points)

                # Otherwise, load in point cloud
                else:
                    print("Loading in Point Cloud")

                    point_cloud = o3d.io.read_point_cloud(self.cfg.geometry_convert_from)

                # Scale and centre point cloud 
                point_cloud, scaling_factor, centre_translation = self.scale_and_centre_pointcloud(point_cloud)

                self.scaling_factor = scaling_factor
                self.centre_translation = centre_translation

                # Downsample point cloud 
                point_cloud = self.downsample_pointcloud(point_cloud, num_downsample_points=self.cfg.num_downsample_points)
            
                points = np.asarray(point_cloud.points)

                # Coloured Point Cloud Colour
                if point_cloud.has_colors() and self.cfg.point_cloud_colour_type == 0:
                    rgb = np.asarray(point_cloud.colors)
                
                # White Point Cloud Colour
                elif self.cfg.point_cloud_colour_type == 1:
                    rgb = np.ones((coords.shape[0], 3))
                
                # Black Point Cloud Colour
                elif self.cfg.point_cloud_colour_type == 2:
                    rgb = np.zeros((coords.shape[0], 3))
                
                # Randomised Point Cloud Colour
                elif self.cfg.point_cloud_colour_type == 3:
                    rgb = np.stack((np.random.rand(coords.shape[0]), np.random.rand(coords.shape[0]), np.random.rand(coords.shape[0])), axis=-1)
                
                else:
                    raise Exception("Invalid point cloud colour type/ Point cloud does not have colour")
                
                pcd = BasicPointCloud(points=points, colors=rgb, normals=np.zeros((points.shape[0], 3)))

                self.create_from_pcd(pcd, 10)

            self.training_setup()
        else:
            raise Exception("Invalid initialisation type")

    @property
    def get_scaling(self):
        if self.cfg.sphere:
            return self.scaling_activation(
                torch.mean(self._scaling, dim=-1).unsqueeze(-1).repeat(1, 3)
            )
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_normal(self):
        if self.cfg.pred_normal:
            return self._normal
        else:
            raise ValueError("Normal is not predicted")

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
            .float()
            .cuda()
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        threestudio.info(
            f"Number of points at initialisation:{fused_point_cloud.shape[0]}"
        )

        dist2 = torch.clamp_min(
            distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
            0.0000001,
        )
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(
            self.cfg.opacity_init
            * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )

        self._xyz = nn.Parameter((fused_point_cloud).requires_grad_(True))
        self._features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        if self.cfg.pred_normal:
            normals = torch.zeros((fused_point_cloud.shape[0], 3), device="cuda")
            self._normal = nn.Parameter(normals.requires_grad_(True))
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")

        self.fused_point_cloud = fused_point_cloud.cpu().clone().detach()
        self.features = features.cpu().clone().detach()
        self.scales = scales.cpu().clone().detach()
        self.rots = rots.cpu().clone().detach()
        self.opacities = opacities.cpu().clone().detach()

    def training_setup(self):
        training_args = self.cfg
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {
                "params": [self._xyz],
                "lr": C(training_args.position_lr, 0, 0),
                "name": "xyz",
            },
            {
                "params": [self._features_dc],
                "lr": C(training_args.feature_lr, 0, 0),
                "name": "f_dc",
            },
            {
                "params": [self._features_rest],
                "lr": C(training_args.feature_lr, 0, 0) / 20.0,
                "name": "f_rest",
            },
            {
                "params": [self._opacity],
                "lr": C(training_args.opacity_lr, 0, 0),
                "name": "opacity",
            },
            {
                "params": [self._scaling],
                "lr": C(training_args.scaling_lr, 0, 0),
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": C(training_args.rotation_lr, 0, 0),
                "name": "rotation",
            },
        ]
        if self.cfg.pred_normal:
            l.append(
                {
                    "params": [self._normal],
                    "lr": C(training_args.normal_lr, 0, 0),
                    "name": "normal",
                },
            )

        self.optimize_params = [
            "xyz",
            "f_dc",
            "f_rest",
            "opacity",
            "scaling",
            "rotation",
        ]
        self.optimize_list = l
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def merge_optimizer(self, net_optimizer):
        l = self.optimize_list
        for param in net_optimizer.param_groups:
            l.append(
                {
                    "params": param["params"],
                    "lr": param["lr"],
                }
            )
        self.optimizer = torch.optim.Adam(l, lr=0.0)
        return self.optimizer

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if not ("name" in param_group):
                continue
            if param_group["name"] == "xyz":
                param_group["lr"] = C(
                    self.cfg.position_lr, 0, iteration, interpolation="linear"
                )
            if param_group["name"] == "scaling":
                param_group["lr"] = C(
                    self.cfg.scaling_lr, 0, iteration, interpolation="linear"
                )
            if param_group["name"] == "f_dc":
                param_group["lr"] = C(
                    self.cfg.feature_lr, 0, iteration, interpolation="linear"
                )
            if param_group["name"] == "f_rest":
                param_group["lr"] = (
                    C(self.cfg.feature_lr, 0, iteration, interpolation="linear") / 20.0
                )
            if param_group["name"] == "opacity":
                param_group["lr"] = C(
                    self.cfg.opacity_lr, 0, iteration, interpolation="linear"
                )
            if param_group["name"] == "rotation":
                param_group["lr"] = C(
                    self.cfg.rotation_lr, 0, iteration, interpolation="linear"
                )
            if param_group["name"] == "normal":
                param_group["lr"] = C(
                    self.cfg.normal_lr, 0, iteration, interpolation="linear"
                )
        self.color_clip = C(self.cfg.color_clip, 0, iteration)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(self.get_opacity * 0.9)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def to(self, device="cpu"):
        self._xyz = self._xyz.to(device)
        self._features_dc = self._features_dc.to(device)
        self._features_rest = self._features_rest.to(device)
        self._opacity = self._opacity.to(device)
        self._scaling = self._scaling.to(device)
        self._rotation = self._rotation.to(device)
        self._normal = self._normal.to(device)

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if ("name" in group) and group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if ("name" in group) and (group["name"] in self.optimize_params):
                stored_state = self.optimizer.state.get(group["params"][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del self.optimizer.state[group["params"][0]]
                    group["params"][0] = nn.Parameter(
                        (group["params"][0][mask].requires_grad_(True))
                    )
                    self.optimizer.state[group["params"][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(
                        group["params"][0][mask].requires_grad_(True)
                    )
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if self.cfg.pred_normal:
            self._normal = optimizable_tensors["normal"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if ("name" in group) and (group["name"] in self.optimize_params):
                extension_tensor = tensors_dict[group["name"]]
                stored_state = self.optimizer.state.get(group["params"][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.cat(
                        (stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                        dim=0,
                    )
                    stored_state["exp_avg_sq"] = torch.cat(
                        (
                            stored_state["exp_avg_sq"],
                            torch.zeros_like(extension_tensor),
                        ),
                        dim=0,
                    )

                    del self.optimizer.state[group["params"][0]]
                    group["params"][0] = nn.Parameter(
                        torch.cat(
                            (group["params"][0], extension_tensor), dim=0
                        ).requires_grad_(True)
                    )
                    self.optimizer.state[group["params"][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(
                        torch.cat(
                            (group["params"][0], extension_tensor), dim=0
                        ).requires_grad_(True)
                    )
                    optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
        new_normal=None,
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }
        if self.cfg.pred_normal:
            d.update({"normal": new_normal})

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if self.cfg.pred_normal:
            self._normal = optimizable_tensors["normal"]

        self.xyz_gradient_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, N=2):
        n_init_points = self._xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > self.cfg.split_thresh,
        )

        # divide N to enhance robustness
        stds = self.get_scaling[selected_pts_mask].repeat(N, 1) / N
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self._xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        if self.cfg.pred_normal:
            new_normal = self._normal[selected_pts_mask].repeat(N, 1)
        else:
            new_normal = None

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_normal,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= self.cfg.split_thresh,
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        if self.cfg.pred_normal:
            new_normal = self._normal[selected_pts_mask]
        else:
            new_normal = None

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_normal,
        )

    def densify(self, max_grad):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad)
        self.densify_and_split(grads, max_grad)

    def prune(self, min_opacity, max_screen_size, iteration):
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            prune_mask = torch.logical_or(prune_mask, big_points_vs)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1

    @torch.no_grad()
    def update_states(
        self,
        iteration,
        visibility_filter,
        radii,
        viewspace_point_tensor,
    ):
        if self._xyz.shape[0] >= self.cfg.max_num + 100:
            prune_mask = torch.randperm(self._xyz.shape[0]).to(self._xyz.device)
            prune_mask = prune_mask > self.cfg.max_num
            self.prune_points(prune_mask)
            return
        # Keep track of max radii in image-space for pruning
        # loop over batch
        bs = len(viewspace_point_tensor)
        for i in range(bs):
            radii_i = radii[i]
            visibility_filter_i = visibility_filter[i]
            viewspace_point_tensor_i = viewspace_point_tensor[i]
            self.max_radii2D = torch.max(self.max_radii2D, radii_i.float())

            self.add_densification_stats(viewspace_point_tensor_i, visibility_filter_i)

        if (
            iteration > self.cfg.prune_from_iter
            and iteration < self.cfg.prune_until_iter
            and iteration % self.cfg.prune_interval == 0
        ):
            self.prune(self.cfg.min_opac_prune, self.cfg.radii2d_thresh, iteration)
            if iteration % self.cfg.opacity_reset_interval == 0:
                self.reset_opacity()

        if (
            iteration > self.cfg.densify_from_iter
            and iteration < self.cfg.densify_until_iter
            and iteration % self.cfg.densification_interval == 0
        ):
            self.densify(self.cfg.densify_grad_threshold)
