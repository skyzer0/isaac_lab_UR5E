# # Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause

# """视觉推抓任务的奖励函数。"""

# from typing import Dict, List, Tuple, Optional
# import torch
# import numpy as np

# def push_success_reward(
#     env,
#     scene_change_threshold: float = 0.02,
#     min_pixels_changed: int = 100,
# ) -> torch.Tensor:
#     """推动成功奖励，基于场景变化检测。
    
#     Args:
#         env: 环境实例
#         scene_change_threshold: 场景变化检测阈值
#         min_pixels_changed: 最少变化像素数
        
#     Returns:
#         torch.Tensor: 推动奖励张量，形状为[N]
#     """
#     # 初始化奖励为零
#     reward = torch.zeros(env.num_envs, device=env.device)
    
#     # 检查是否有当前和上一帧的深度图
#     if hasattr(env, "current_depth_map") and hasattr(env, "previous_depth_map"):
#         # 计算深度图差异
#         depth_diff = torch.abs(env.current_depth_map - env.previous_depth_map)
        
#         # 应用阈值，获取变化显著的像素
#         changed_pixels = depth_diff > scene_change_threshold
        
#         # 计算每个环境中变化像素的数量
#         num_changed_pixels = torch.sum(changed_pixels.float(), dim=(1, 2, 3))
        
#         # 如果变化像素数量超过阈值，则认为推动成功
#         push_success = num_changed_pixels > min_pixels_changed
        
#         # 设置奖励
#         reward = push_success.float()
    
#     return reward

# def grasp_success_reward(
#     env,
#     height_threshold: float = 0.05,
#     gripper_close_threshold: float = 0.5,
# ) -> torch.Tensor:
#     """抓取成功奖励，基于物体是否被成功抓起。
    
#     Args:
#         env: 环境实例
#         height_threshold: 物体被抓起的高度阈值
#         gripper_close_threshold: 判断爪子闭合的阈值
        
#     Returns:
#         torch.Tensor: 抓取奖励张量，形状为[N]
#     """
#     # 初始化奖励为零
#     reward = torch.zeros(env.num_envs, device=env.device)
    
#     # 检查是否有物体位置和抓手状态信息
#     if hasattr(env, "object_heights") and hasattr(env, "gripper_positions"):
#         # 获取物体高度和初始高度
#         current_object_heights = env.object_heights
#         initial_object_heights = env.initial_object_heights if hasattr(env, "initial_object_heights") else torch.zeros_like(current_object_heights)
        
#         # 计算物体升起的高度差
#         height_diff = current_object_heights - initial_object_heights
        
#         # 检查抓手是否闭合
#         gripper_closed = env.gripper_positions > gripper_close_threshold
        
#         # 判断抓取是否成功（物体高度增加且抓手闭合）
#         grasp_success = (height_diff > height_threshold) & gripper_closed
        
#         # 设置奖励
#         reward = grasp_success.float()
    
#     return reward

# def workspace_clearance_reward(
#     env,
#     workspace_bounds: Dict,
# ) -> torch.Tensor:
#     """工作区域清理奖励，奖励机器人从工作区移除物体。
    
#     Args:
#         env: 环境实例
#         workspace_bounds: 工作区边界，格式为{'x': (min, max), 'y': (min, max), 'z': (min, max)}
        
#     Returns:
#         torch.Tensor: 工作区清理奖励张量，形状为[N]
#     """
#     # 初始化奖励为零
#     reward = torch.zeros(env.num_envs, device=env.device)
    
#     # 检查是否有当前和之前的工作区物体数量
#     if hasattr(env, "objects_in_workspace_count") and hasattr(env, "previous_objects_in_workspace_count"):
#         # 计算物体数量减少的数量
#         objects_removed = env.previous_objects_in_workspace_count - env.objects_in_workspace_count
        
#         # 只有当物体数量减少时，才给予奖励
#         reward = torch.clamp(objects_removed.float(), min=0)
    
#     return reward

# def empty_workspace_penalty(
#     env,
#     min_object_count: int = 3,
# ) -> torch.Tensor:
#     """工作区空置惩罚，惩罚工作区内物体太少的情况。
    
#     Args:
#         env: 环境实例
#         min_object_count: 工作区内最少物体数
        
#     Returns:
#         torch.Tensor: 惩罚张量，形状为[N]
#     """
#     # 初始化惩罚为零
#     penalty = torch.zeros(env.num_envs, device=env.device)
    
#     # 检查是否有工作区物体数量
#     if hasattr(env, "objects_in_workspace_count"):
#         # 如果物体数量少于最低要求，给予惩罚
#         too_few_objects = env.objects_in_workspace_count < min_object_count
        
#         # 设置惩罚
#         penalty = too_few_objects.float()
    
#     return penalty 