# # Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause

# """视觉推抓任务的观察函数。"""

# from typing import Dict, List, Tuple, Optional
# import torch
# import numpy as np

# def camera_rgb(env, camera_name: str) -> torch.Tensor:
#     """从指定相机获取RGB图像。
    
#     Args:
#         env: 环境实例
#         camera_name: 相机名称
        
#     Returns:
#         torch.Tensor: RGB图像张量，形状为[N, 3, H, W]，值范围[0, 1]
#     """
#     # 从环境获取相机数据
#     camera_sensor = env.scene[camera_name]
#     rgb_data = camera_sensor.get_rgb_data()
    
#     # 归一化到[0, 1]
#     rgb_data = rgb_data / 255.0
    
#     # 确保数据格式正确 (N, 3, H, W)
#     if len(rgb_data.shape) == 3:
#         rgb_data = rgb_data.unsqueeze(0)
    
#     # 转置为模型所需的格式
#     # 原始格式通常是 (N, H, W, 3)，需要转换为 (N, 3, H, W)
#     if rgb_data.shape[-1] == 3:
#         rgb_data = rgb_data.permute(0, 3, 1, 2)
    
#     return rgb_data

# def camera_depth(env, camera_name: str) -> torch.Tensor:
#     """从指定相机获取深度图像。
    
#     Args:
#         env: 环境实例
#         camera_name: 相机名称
        
#     Returns:
#         torch.Tensor: 深度图像张量，形状为[N, 1, H, W]
#     """
#     # 从环境获取相机数据
#     camera_sensor = env.scene[camera_name]
#     depth_data = camera_sensor.get_distance_to_camera_data()
    
#     # 归一化深度数据（截断到10米）
#     depth_data = torch.clamp(depth_data, 0.0, 10.0) / 10.0
    
#     # 确保数据格式正确 (N, 1, H, W)
#     if len(depth_data.shape) == 2:
#         depth_data = depth_data.unsqueeze(0).unsqueeze(0)
#     elif len(depth_data.shape) == 3:
#         if depth_data.shape[-1] == 1:
#             # 如果格式是 (N, H, W, 1)
#             depth_data = depth_data.permute(0, 3, 1, 2)
#         else:
#             # 如果格式是 (N, H, W)
#             depth_data = depth_data.unsqueeze(1)
    
#     return depth_data

# def end_effector_pose(env, asset_name: str, body_name: str) -> torch.Tensor:
#     """获取末端执行器的位姿。
    
#     Args:
#         env: 环境实例
#         asset_name: 机器人资产名称
#         body_name: 末端执行器Body名称
        
#     Returns:
#         torch.Tensor: 末端执行器位姿张量，形状为[N, 7]，包含位置(3)和四元数旋转(4)
#     """
#     # 获取机器人
#     robot = env.scene[asset_name]
    
#     # 获取末端执行器Pose
#     ee_pos, ee_rot = robot.get_body_pose(body_name)
    
#     # 组合成位姿向量 [x, y, z, qw, qx, qy, qz]
#     pose = torch.cat([ee_pos, ee_rot], dim=-1)
    
#     return pose

# def gripper_joints_pos(env, asset_name: str, joint_names: List[str]) -> torch.Tensor:
#     """获取抓手关节的位置。
    
#     Args:
#         env: 环境实例
#         asset_name: 机器人资产名称
#         joint_names: 抓手关节名称列表
        
#     Returns:
#         torch.Tensor: 抓手关节位置张量，形状为[N, len(joint_names)]
#     """
#     # 获取机器人
#     robot = env.scene[asset_name]
    
#     # 存储所有关节位置
#     all_joint_positions = []
    
#     # 获取每个关节的位置
#     for joint_name in joint_names:
#         joint_pos = robot.get_joint_positions(joint_names=[joint_name])
#         all_joint_positions.append(joint_pos)
    
#     # 组合所有关节位置
#     if len(all_joint_positions) > 0:
#         joint_positions = torch.cat(all_joint_positions, dim=-1)
#     else:
#         # 如果没有关节，返回空张量
#         joint_positions = torch.zeros((env.num_envs, 0), device=env.device)
    
#     return joint_positions

# def last_action(env) -> torch.Tensor:
#     """获取上一步执行的动作。
    
#     Args:
#         env: 环境实例
        
#     Returns:
#         torch.Tensor: 上一步动作张量
#     """
#     # 如果环境中存储了上一步动作，则返回它
#     if hasattr(env, "last_actions"):
#         return env.last_actions
#     else:
#         # 否则返回全零张量
#         # 假设动作空间维度是8
#         return torch.zeros((env.num_envs, 8), device=env.device) 