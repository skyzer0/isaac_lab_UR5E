# # Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause

# """视觉推抓任务的课程学习函数。"""

# from typing import Dict, List, Tuple, Optional, Any
# import torch
# import numpy as np

# def adjust_object_count(
#     env,
#     start_count: int,
#     target_count: int,
#     num_steps: int,
# ) -> None:
#     """随着训练进行逐渐增加物体数量。
    
#     Args:
#         env: 环境实例
#         start_count: 起始物体数量
#         target_count: 目标物体数量
#         num_steps: 达到目标数量的步骤数
#     """
#     # 获取当前迭代次数
#     current_iter = env.unwrapped.curriculum_iteration if hasattr(env.unwrapped, "curriculum_iteration") else 0
    
#     # 计算当前应该的物体数量
#     if current_iter >= num_steps:
#         # 如果达到指定步数，使用目标数量
#         current_count = target_count
#     else:
#         # 线性插值计算当前数量
#         progress = current_iter / num_steps
#         current_count = int(start_count + progress * (target_count - start_count))
    
#     # 保存当前物体数量，供重置函数使用
#     env.num_objects_to_reset = current_count

# def modify_reward_weight(
#     env,
#     term_name: str,
#     weight: float,
#     target_weight: Optional[float] = None,
#     num_steps: Optional[int] = None,
# ) -> None:
#     """修改特定奖励项的权重。
    
#     Args:
#         env: 环境实例
#         term_name: 奖励项名称
#         weight: 起始权重值
#         target_weight: 目标权重值
#         num_steps: 达到目标权重的步骤数
#     """
#     # 如果没有提供目标权重或步骤数，直接设置为固定权重
#     if target_weight is None or num_steps is None:
#         if hasattr(env, "task"):
#             if hasattr(env.task, "reward_manager"):
#                 # 直接设置权重
#                 env.task.reward_manager.set_term_weight(term_name, weight)
#         return
    
#     # 获取当前迭代次数
#     current_iter = env.unwrapped.curriculum_iteration if hasattr(env.unwrapped, "curriculum_iteration") else 0
    
#     # 计算当前的权重
#     if current_iter >= num_steps:
#         # 如果达到指定步数，使用目标权重
#         current_weight = target_weight
#     else:
#         # 线性插值计算当前权重
#         progress = current_iter / num_steps
#         current_weight = weight + progress * (target_weight - weight)
    
#     # 设置奖励权重
#     if hasattr(env, "task"):
#         if hasattr(env.task, "reward_manager"):
#             env.task.reward_manager.set_term_weight(term_name, current_weight)

# def modify_exploration_rate(
#     env,
#     initial_rate: float,
#     final_rate: float,
#     num_steps: int,
# ) -> None:
#     """随着训练进行逐渐减少探索率。
    
#     Args:
#         env: 环境实例
#         initial_rate: 初始探索率
#         final_rate: 最终探索率
#         num_steps: 达到最终探索率的步骤数
#     """
#     # 获取当前迭代次数
#     current_iter = env.unwrapped.curriculum_iteration if hasattr(env.unwrapped, "curriculum_iteration") else 0
    
#     # 计算当前的探索率
#     if current_iter >= num_steps:
#         # 如果达到指定步数，使用最终探索率
#         current_rate = final_rate
#     else:
#         # 线性或指数衰减计算当前探索率
#         progress = current_iter / num_steps
#         # 线性衰减
#         # current_rate = initial_rate + progress * (final_rate - initial_rate)
#         # 指数衰减
#         current_rate = initial_rate * (final_rate / initial_rate) ** progress
    
#     # 设置当前探索率
#     env.exploration_rate = current_rate 