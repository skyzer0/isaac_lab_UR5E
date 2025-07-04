# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import FrameTransformer
from omni.isaac.lab.utils.math import combine_frame_transforms






if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

def view_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """目标是否在 RGBD 观察范围内"""
    observations = env.observation_manager.compute_group("policy")
    start_idx = 11 + 12 + 12  # actions(11) + joint_pos(12) + joint_vel(12)
    end_idx = start_idx + 1000  # `camera_rgbd_features` 长度是 1000
    # **提取 `camera_rgbd_features`**
    rgbd_features = observations[:, start_idx:end_idx]

    # print(f"[DEBUG] Observations shape: {observations.shape}")
    # print(f"[DEBUG] Observation Manager Terms: {env.observation_manager}")
    # print(f"[DEBUG] camera_rgbd_features shape: {rgbd_features.shape}")  


    # **计算目标是否在视野内**
    target_in_view = torch.mean(rgbd_features, dim=1) > 0.5  
    # reward = target_in_view.float() * 2.0 
    # print(f"[DEBUG] RGBD Features Mean: {torch.mean(rgbd_features, dim=1)}")
    # print(f"[DEBUG] View Reward: {reward}")
    return target_in_view.float() * 2.0


def distance_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """机械臂靠近目标的奖励"""
    
    # 获取机械臂末端位置 (`gripper`) 和目标 (`target`) 的世界坐标
    gripper_pos = env.scene["ee_frame"].data.target_pos_w.squeeze(1)  # [batch, 3]
    target_pos = env.scene["object"].data.root_pos_w  # [batch, 3]

    # 计算欧几里得距离 (L2 范数)
    distance = torch.norm(gripper_pos - target_pos, dim=1)

    # 归一化距离奖励：距离越近，奖励越高
    reward = 1.0 / (distance + 1e-3)  

    return reward


def above_target_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """机械臂到达目标上方的奖励"""

    gripper_pos = env.scene["ee_frame"].data.target_pos_w.squeeze(1)  # [batch, 3]
    target_pos = env.scene["object"].data.root_pos_w  # [batch, 3]

    # 计算水平方向 (x, y) 的距离
    horizontal_distance = torch.norm(gripper_pos[:, :2] - target_pos[:, :2], dim=1)

    # 归一化：越靠近目标上方，奖励越大
    reward = 3.0 / (horizontal_distance + 1e-3)  

    return reward


def descent_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """夹爪下降到目标的奖励"""

    gripper_pos = env.scene["ee_frame"].data.target_pos_w.squeeze(1)  # [batch, 3]
    target_pos = env.scene["object"].data.root_pos_w  # [batch, 3]

    # 计算 `z` 方向高度差
    vertical_distance = torch.abs(gripper_pos[:, 2] - target_pos[:, 2])

    # 归一化：机械臂下降到接近目标，奖励增加
    reward = 3.0 / (vertical_distance + 1e-3)  

    return reward


# **(6) 抓取稳定性**
def grasp_stability_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """ 成功抓取奖励 """
    contact_sensor = env.scene.sensors.get("gripper_contact_sensor")
    gripper_actuator = env.scene.articulations.get("ur5e_hand")

    if contact_sensor is None or gripper_actuator is None:
        return torch.zeros(env.num_envs, device=env.device)  # 无抓取条件时无奖励

    is_contact = torch.any(contact_sensor.data.contact_detected, dim=1)  
    gripper_effort = gripper_actuator.effort_limit  # 夹爪力
    is_grasping = gripper_effort > 30.0  # 夹爪是否真正闭合抓取

    return (is_contact.float() & is_grasping.float()) * 10.0  # 仅在抓取成功时奖励


def lift_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """ 物体被成功提起的奖励 """
    target_pos = env.scene["object"].data.root_pos_w
    lift_height = 0.1
    is_lifted = target_pos[:, 2] > lift_height

    return is_lifted.float() * 10.0  # 抬起目标时给高奖励

