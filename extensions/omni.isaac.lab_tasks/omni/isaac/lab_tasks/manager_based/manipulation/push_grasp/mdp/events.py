# # Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause

# """视觉推抓任务的事件函数。"""

# from typing import Dict, List, Tuple, Optional, Any
# import torch
# import numpy as np

# def reset_scene_to_default(env) -> None:
#     """重置场景到默认状态。
    
#     Args:
#         env: 环境实例
#     """
#     # 重置场景中的所有对象
#     for asset_name in env.scene.get_asset_names():
#         # 不重置机器人本身
#         if asset_name != "robot" and not "camera" in asset_name:
#             asset = env.scene[asset_name]
#             # 检查是否存在默认状态
#             if hasattr(asset, "default_root_state"):
#                 # 重置到默认状态
#                 asset.set_root_state(asset.default_root_state.clone())
    
#     # 重置机器人关节到初始状态
#     if "robot" in env.scene.get_asset_names():
#         robot = env.scene["robot"]
#         if hasattr(robot, "default_joint_pos"):
#             robot.set_joint_positions(robot.default_joint_pos.clone())

# def reset_multiple_objects_uniform(
#     env,
#     num_objects: int,
#     object_prefix: str,
#     pose_range: Dict[str, Tuple[float, float]],
#     rotation_range: Dict[str, Tuple[float, float]],
#     min_distance: float = 0.05,
# ) -> None:
#     """随机重置多个物体的位置和方向。
    
#     Args:
#         env: 环境实例
#         num_objects: 要重置的物体数量
#         object_prefix: 物体名称前缀
#         pose_range: 位置范围，格式为{'x': (min, max), 'y': (min, max), 'z': (min, max)}
#         rotation_range: 旋转范围，格式为{'roll': (min, max), 'pitch': (min, max), 'yaw': (min, max)}
#         min_distance: 物体之间的最小距离
#     """
#     # 获取设备
#     device = env.device
#     num_envs = env.num_envs
    
#     # 定义工作区范围
#     x_range = pose_range["x"]
#     y_range = pose_range["y"]
#     z_range = pose_range["z"]
    
#     # 为每个物体生成随机位置
#     for i in range(num_objects):
#         object_name = f"{object_prefix}{i}"
#         if object_name in env.scene.get_asset_names():
#             obj = env.scene[object_name]
            
#             # 为每个环境中的物体生成随机位置
#             # 注意：这里我们简化了位置生成逻辑，实际应用中可能需要更复杂的碰撞检测
#             positions = torch.zeros((num_envs, 3), device=device)
            
#             # 随机X位置
#             positions[:, 0] = torch.rand(num_envs, device=device) * (x_range[1] - x_range[0]) + x_range[0]
#             # 随机Y位置
#             positions[:, 1] = torch.rand(num_envs, device=device) * (y_range[1] - y_range[0]) + y_range[0]
#             # Z位置（可能是固定的高度）
#             positions[:, 2] = torch.rand(num_envs, device=device) * (z_range[1] - z_range[0]) + z_range[0]
            
#             # 生成随机旋转
#             rotations = torch.zeros((num_envs, 4), device=device)  # 四元数表示
            
#             # 随机偏航角（绕Z轴旋转）
#             yaw = torch.rand(num_envs, device=device) * (rotation_range["yaw"][1] - rotation_range["yaw"][0]) + rotation_range["yaw"][0]
            
#             # 随机俯仰角（绕Y轴旋转）
#             pitch = torch.rand(num_envs, device=device) * (rotation_range["pitch"][1] - rotation_range["pitch"][0]) + rotation_range["pitch"][0]
            
#             # 随机翻滚角（绕X轴旋转）
#             roll = torch.rand(num_envs, device=device) * (rotation_range["roll"][1] - rotation_range["roll"][0]) + rotation_range["roll"][0]
            
#             # 计算四元数（简化版，实际应用中可能需要更准确的欧拉角到四元数转换）
#             cy = torch.cos(yaw * 0.5)
#             sy = torch.sin(yaw * 0.5)
#             cp = torch.cos(pitch * 0.5)
#             sp = torch.sin(pitch * 0.5)
#             cr = torch.cos(roll * 0.5)
#             sr = torch.sin(roll * 0.5)
            
#             rotations[:, 0] = cr * cp * cy + sr * sp * sy  # w
#             rotations[:, 1] = sr * cp * cy - cr * sp * sy  # x
#             rotations[:, 2] = cr * sp * cy + sr * cp * sy  # y
#             rotations[:, 3] = cr * cp * sy - sr * sp * cy  # z
            
#             # 设置物体的位置和旋转
#             root_state = torch.zeros((num_envs, 13), device=device)
#             root_state[:, 0:3] = positions
#             root_state[:, 3:7] = rotations
            
#             # 设置物体状态
#             obj.set_root_state(root_state)
            
#             # 存储初始高度用于后续奖励计算
#             if not hasattr(env, "initial_object_heights"):
#                 env.initial_object_heights = torch.zeros((num_envs, num_objects), device=device)
#             env.initial_object_heights[:, i] = positions[:, 2]
    
#     # 更新工作区物体数量
#     env.objects_in_workspace_count = torch.ones(num_envs, device=device) * num_objects
#     env.previous_objects_in_workspace_count = env.objects_in_workspace_count.clone() 