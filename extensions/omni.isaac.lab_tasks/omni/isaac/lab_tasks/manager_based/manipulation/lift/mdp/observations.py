# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import subtract_frame_transforms
from omni.isaac.lab.managers.manager_base import ManagerTermBase
from omni.isaac.lab.managers.manager_term_cfg import ObservationTermCfg
import numpy as np
import torch
from typing import Tuple
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt




if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

from omni.isaac.core.utils.prims import get_prim_at_path

def image(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    data_type: str = "rgb",
    convert_perspective_to_orthogonal: bool = False,
    normalize: bool = True,
) -> torch.Tensor:
    """Images of a specific datatype from the camera sensor.

    If the flag :attr:`normalize` is True, post-processing of the images are performed based on their
    data-types:

    - "rgb": Scales the image to (0, 1) and subtracts with the mean of the current image batch.
    - "depth" or "distance_to_camera" or "distance_to_plane": Replaces infinity values with zero.

    Args:
        env: The environment the cameras are placed within.
        sensor_cfg: The desired sensor to read from. Defaults to SceneEntityCfg("tiled_camera").
        data_type: The data type to pull from the desired camera. Defaults to "rgb".
        convert_perspective_to_orthogonal: Whether to orthogonalize perspective depth images.
            This is used only when the data type is "distance_to_camera". Defaults to False.
        normalize: Whether to normalize the images. This depends on the selected data type.
            Defaults to True.

    Returns:
        The images produced at the last time-step
    """
    # extract the used quantities (to enable type-hinting)
    sensor: TiledCamera | Camera | RayCasterCamera = env.scene.sensors[sensor_cfg.name]

    # obtain the input image
    images = sensor.data.output[data_type]

    # depth image conversion
    if (data_type == "distance_to_camera") and convert_perspective_to_orthogonal:
        images = math_utils.orthogonalize_perspective_depth(images, sensor.data.intrinsic_matrices)

    # rgb/depth image normalization
    if normalize:
        if data_type == "rgb":
            images = images.float() / 255.0
            mean_tensor = torch.mean(images, dim=(1, 2), keepdim=True)
            images -= mean_tensor
        elif "distance_to" in data_type or "depth" in data_type:
            images[images == float("inf")] = 0

    return images.clone()

def joint_pos_rel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]

def joint_vel_rel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """The joint velocities of the asset w.r.t. the default joint velocities.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, asset_cfg.joint_ids] - asset.data.default_joint_vel[:, asset_cfg.joint_ids]




class image_features(ManagerTermBase):
    """Extracted image features from a pre-trained frozen encoder.

    This method calls the :meth:`image` function to retrieve images, and then performs
    inference on those images.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        from torchvision import models
        from transformers import AutoModel

        def create_theia_model(model_name):
            return {
                "model": (
                    lambda: AutoModel.from_pretrained(f"theaiinstitute/{model_name}", trust_remote_code=True)
                    .eval()
                    .to("cuda:0")
                ),
                "preprocess": lambda img: (img - torch.amin(img, dim=(1, 2), keepdim=True)) / (
                    torch.amax(img, dim=(1, 2), keepdim=True) - torch.amin(img, dim=(1, 2), keepdim=True)
                ),
                "inference": lambda model, images: model.forward_feature(
                    images, do_rescale=False, interpolate_pos_encoding=True
                ),
            }

        def create_resnet_model(resnet_name):
            return {
                "model": lambda: getattr(models, resnet_name)(pretrained=True).eval().to("cuda:0"),
                "preprocess": lambda img: (
                    img.permute(0, 3, 1, 2)  # Convert [batch, height, width, 3] -> [batch, 3, height, width]
                    - torch.tensor([0.485, 0.456, 0.406], device=img.device).view(1, 3, 1, 1)
                ) / torch.tensor([0.229, 0.224, 0.225], device=img.device).view(1, 3, 1, 1),
                "inference": lambda model, images: model(images),
            }

        def create_resnet_model(resnet_name):
            model = getattr(models, resnet_name)(pretrained=True)

            # 修改 ResNet 第一层，使其支持 4 通道输入
            old_conv = model.conv1
            model.conv1 = nn.Conv2d(
                in_channels=4,  # 从 3 改成 4
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )

            # 复制原有的权重，并随机初始化新通道的权重
            with torch.no_grad():
                model.conv1.weight[:, :3] = old_conv.weight  # 复制 RGB 权重
                model.conv1.weight[:, 3:] = torch.mean(old_conv.weight, dim=1, keepdim=True)  # 用 RGB 平均值填充 Depth 通道

            return {
                "model": lambda: model.eval().to("cuda:0"),
                "preprocess": lambda img: (
                    img.permute(0, 3, 1, 2)  # Convert [batch, height, width, 4] -> [batch, 4, height, width]
                    - torch.tensor([0.485, 0.456, 0.406, 0.0], device=img.device).view(1, 4, 1, 1)
                ) / torch.tensor([0.229, 0.224, 0.225, 1.0], device=img.device).view(1, 4, 1, 1),
                "inference": lambda model, images: model(images),
            }
        # List of Theia models
        theia_models = [
            "theia-tiny-patch16-224-cddsv",
            "theia-tiny-patch16-224-cdiv",
            "theia-small-patch16-224-cdiv",
            "theia-base-patch16-224-cdiv",
            "theia-small-patch16-224-cddsv",
            "theia-base-patch16-224-cddsv",
        ]

        # List of ResNet models
        resnet_models = ["resnet18", "resnet50", "resnet101"]

        self.default_model_zoo_cfg = {}

        # Add Theia models to the zoo
        for model_name in theia_models:
            self.default_model_zoo_cfg[model_name] = create_theia_model(model_name)

        # Add ResNet models to the zoo
        for resnet_name in resnet_models:
            self.default_model_zoo_cfg[resnet_name] = create_resnet_model(resnet_name)

        self.model_zoo_cfg = self.default_model_zoo_cfg
        self.model_zoo = {}


    def __call__(
        self,
        env: ManagerBasedEnv,
        sensor_cfg_rgb: SceneEntityCfg = SceneEntityCfg("RGB_Camera"),
        sensor_cfg_depth: SceneEntityCfg = SceneEntityCfg("depth_Camera"),
        data_type: str = "rgb",
        convert_perspective_to_orthogonal: bool = False,
        model_zoo_cfg: dict | None = None,
        model_name: str = "resnet18",
        model_device: str | None = "cuda:0",
        reset_model: bool = False,
    ) -> torch.Tensor:
        """Extract 4-channel (RGBD) image features from a pre-trained model."""

        # 获取 RGB 和 Depth 图像
        images_rgb = image(env=env, sensor_cfg=sensor_cfg_rgb, data_type="rgb", normalize=True)  # [B, 244, 244, 3]
        images_depth = image(env=env, sensor_cfg=sensor_cfg_depth, data_type="distance_to_camera", normalize=True)  # [B, 244, 244, 1]

        # 确保 Depth 的形状是 [B, H, W, 1]
        if images_depth.ndim == 3:  # [B, H, W] -> [B, H, W, 1]
            images_depth = images_depth.unsqueeze(-1)
        elif images_depth.shape[-1] != 1:  # 如果 Depth 图的通道数异常
            images_depth = images_depth[..., :1]  # 只取第一通道

        # 确保 Depth 和 RGB 形状一致（调整 H, W）
        if images_rgb.shape[1:3] != images_depth.shape[1:3]:
            from torchvision.transforms.functional import resize
            images_depth = resize(images_depth.permute(0, 3, 1, 2), size=images_rgb.shape[1:3])  # 先变成 [B, 1, H, W]
            images_depth = images_depth.permute(0, 2, 3, 1)  # 变回 [B, H, W, 1]



        # 拼接 RGB 和 Depth 为 4 通道
        images_combined = torch.cat([images_rgb, images_depth], dim=-1)  # 变成 [B, 244, 244, 4]


        # 选择并加载模型
        if model_name not in self.model_zoo or reset_model:
            print(f"[INFO]: Adding {model_name} to the model zoo")
            self.model_zoo[model_name] = self.model_zoo_cfg[model_name]["model"]()

        # 迁移到计算设备
        if model_device is not None:
            images_combined = images_combined.to(model_device)

        # 预处理
        proc_images = self.model_zoo_cfg[model_name]["preprocess"](images_combined)

        # 进行特征提取
        features = self.model_zoo_cfg[model_name]["inference"](self.model_zoo[model_name], proc_images)
        # print(f"[INFO] Original RGB Shape: {images_rgb.shape}")    # 预期: [1, 244, 244, 3]
        # print(f"[INFO] Original Depth Shape: {images_depth.shape}")  # 预期: [1, 244, 244, 1]
        # print(f"[INFO] Fixed Depth Shape: {images_depth.shape}")  # 确保最终是 [1, 244, 244, 1]
        # print(f"[INFO] Combined Input Shape: {images_combined.shape}, Channels: {images_combined.shape[-1]}") 

        # mean_feature = features[0].mean(dim=0).cpu().detach().numpy()
        # plt.figure(figsize=(10,5))
        # plt.plot(mean_feature)
        # plt.title("Feature Vector Distribution")
        # plt.show()

        return features.to(images_rgb.device).clone()



