# # Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause

# """视觉推抓任务的终止条件。"""

# from typing import Dict, List, Tuple, Optional, Any
# import torch
# import numpy as np

# def time_out(env) -> torch.Tensor:
#     """检查是否达到时间限制。
    
#     Args:
#         env: 环境实例
        
#     Returns:
#         torch.Tensor: 布尔张量，指示每个环境是否超时
#     """
#     # 检查当前步骤是否达到最大步骤数
#     # 最大步数通常在环境初始化时设置
#     return env.progress_buf >= env.max_episode_length

# def workspace_cleared(
#     env,
#     max_objects_left: int = 0,
#     workspace_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
# ) -> torch.Tensor:
#     """检查工作区是否已清空（任务完成）。
    
#     Args:
#         env: 环境实例
#         max_objects_left: 工作区内最多剩余物体数量，清空条件
#         workspace_bounds: 工作区边界，格式为{'x': (min, max), 'y': (min, max), 'z': (min, max)}
        
#     Returns:
#         torch.Tensor: 布尔张量，指示每个环境是否已清空工作区
#     """
#     # 初始化为假
#     cleared = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
#     # 检查是否有工作区物体数量
#     if hasattr(env, "objects_in_workspace_count"):
#         # 工作区已清空的条件：物体数量少于或等于阈值
#         cleared = env.objects_in_workspace_count <= max_objects_left
    
#     return cleared 