# # Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause

# """视觉推抓任务的动作定义。"""

# from typing import Dict, List, Tuple, Optional, Union
# import torch
# import numpy as np

# from omni.isaac.lab.managers import ActionTermCfg
# from omni.isaac.lab.utils import configclass
# from omni.isaac.lab.utils.dict_utils import class_to_dict, recursive_update_dict

# @configclass
# class ActionSelectorCfg(ActionTermCfg):
#     """用于选择推动或抓取的动作选择器配置。
    
#     动作向量格式为[selector, x, y, z, angle, pad1, pad2, pad3]
#     - selector: 0-0.5为推动，0.5-1为抓取
#     - x, y, z: 目标位置（归一化到工作区间）
#     - angle: 动作角度（0-1对应0-2π）
#     - pad1, pad2, pad3: 预留
#     """
    
#     action_space_dim: int  # 动作空间维度
    
#     def __post_init__(self):
#         """初始化后处理。"""
#         self.func = action_selector
        
#         # 设置范围字典
#         ranges_dict = {
#             "selector": (0.0, 1.0),  # 0-0.5为推动，0.5-1为抓取
#             "position": {
#                 "x": (0.0, 1.0),  # 归一化坐标
#                 "y": (0.0, 1.0),
#                 "z": (0.0, 1.0),
#             },
#             "angle": (0.0, 1.0),  # 归一化角度
#         }
        
#         self.params = {"ranges": ranges_dict, "action_space_dim": self.action_space_dim}

# def action_selector(action_tensor: torch.Tensor, ranges: Dict, action_space_dim: int, 
#                    push_action: Optional[Dict] = None, grasp_action: Optional[Dict] = None) -> Dict:
#     """实现动作选择器逻辑。
    
#     Args:
#         action_tensor: 原始动作张量
#         ranges: 动作范围配置
#         action_space_dim: 动作空间维度
#         push_action: 推动动作（如果有）
#         grasp_action: 抓取动作（如果有）
        
#     Returns:
#         Dict: 处理后的动作字典
#     """
#     # 解析动作
#     batch_size = action_tensor.shape[0]
#     action_dict = {}
    
#     # 提取选择器值
#     selector = action_tensor[:, 0].clone()  # 第一个值是选择器
    
#     # 归一化位置和角度
#     position_x = action_tensor[:, 1].clone()
#     position_y = action_tensor[:, 2].clone()
#     position_z = action_tensor[:, 3].clone()
#     angle = action_tensor[:, 4].clone()
    
#     # 创建掩码区分推动和抓取
#     push_mask = (selector < 0.5).float()
#     grasp_mask = (selector >= 0.5).float()
    
#     # 设置动作字典，包含选择器和原始动作
#     action_dict["selector"] = selector
#     action_dict["raw_action"] = action_tensor.clone()
#     action_dict["push_mask"] = push_mask
#     action_dict["grasp_mask"] = grasp_mask
    
#     # 填充位置和角度信息
#     action_dict["position"] = {
#         "x": position_x,
#         "y": position_y,
#         "z": position_z,
#     }
#     action_dict["angle"] = angle
    
#     return action_dict

# @configclass
# class PushActionCfg(ActionTermCfg):
#     """推动动作配置。"""
    
#     asset_name: str  # 机器人资产名称
#     ee_body_name: str  # 末端执行器Body名称
#     max_push_distance: float = 0.15  # 最大推动距离（米）
#     push_height_offset: float = 0.01  # 推动高度偏移（距离物体表面）
#     z_offset_before_pushing: float = 0.05  # 开始推动前的高度偏移
#     push_speed: float = 0.1  # 推动速度（米/秒）
    
#     def __post_init__(self):
#         """初始化后处理。"""
#         self.func = push_action
        
#         # 设置参数字典
#         params_dict = {
#             "asset_name": self.asset_name,
#             "ee_body_name": self.ee_body_name,
#             "max_push_distance": self.max_push_distance,
#             "push_height_offset": self.push_height_offset,
#             "z_offset_before_pushing": self.z_offset_before_pushing,
#             "push_speed": self.push_speed,
#         }
        
#         self.params = params_dict

# def push_action(action_dict: Dict, asset_name: str, ee_body_name: str, 
#                 max_push_distance: float, push_height_offset: float,
#                 z_offset_before_pushing: float, push_speed: float) -> Dict:
#     """实现推动动作逻辑。
    
#     Args:
#         action_dict: 动作字典（从action_selector）
#         asset_name: 机器人资产名称
#         ee_body_name: 末端执行器Body名称
#         max_push_distance: 最大推动距离
#         push_height_offset: 推动高度偏移
#         z_offset_before_pushing: 开始推动前的高度偏移
#         push_speed: 推动速度
        
#     Returns:
#         Dict: 处理后的推动动作字典
#     """
#     # 这里实现将选择器输出转换为具体的推动动作
#     # 实际动作根据IsaacLab期望的格式输出
#     push_dict = {}
    
#     # 仅当存在推动掩码为1的环境时处理
#     if torch.any(action_dict["push_mask"] > 0):
#         # 获取推动起点和方向
#         start_pos_x = action_dict["position"]["x"]
#         start_pos_y = action_dict["position"]["y"]
#         start_pos_z = action_dict["position"]["z"]
#         push_angle = action_dict["angle"] * 2.0 * np.pi  # 转换为弧度
        
#         # 计算推动方向的单位向量
#         push_dir_x = torch.cos(push_angle)
#         push_dir_y = torch.sin(push_angle)
        
#         # 计算推动终点
#         end_pos_x = start_pos_x + push_dir_x * max_push_distance
#         end_pos_y = start_pos_y + push_dir_y * max_push_distance
        
#         # 设置机器人属性
#         push_dict["asset_name"] = asset_name
#         push_dict["ee_body_name"] = ee_body_name
        
#         # 设置推动轨迹
#         push_dict["trajectory"] = {
#             "start": {
#                 "x": start_pos_x,
#                 "y": start_pos_y,
#                 "z": start_pos_z + z_offset_before_pushing,
#             },
#             "push_start": {
#                 "x": start_pos_x,
#                 "y": start_pos_y,
#                 "z": start_pos_z + push_height_offset,
#             },
#             "push_end": {
#                 "x": end_pos_x,
#                 "y": end_pos_y,
#                 "z": start_pos_z + push_height_offset,
#             },
#             "end": {
#                 "x": end_pos_x,
#                 "y": end_pos_y,
#                 "z": start_pos_z + z_offset_before_pushing,
#             },
#         }
        
#         # 设置速度和掩码
#         push_dict["speed"] = push_speed
#         push_dict["mask"] = action_dict["push_mask"]
    
#     return push_dict

# @configclass
# class GraspActionCfg(ActionTermCfg):
#     """抓取动作配置。"""
    
#     asset_name: str  # 机器人资产名称
#     ee_body_name: str  # 末端执行器Body名称
#     gripper_joint_names: List[str]  # 抓手关节名称
#     pre_grasp_height: float = 0.1  # 抓取前的高度
#     grasp_height: float = 0.02  # 抓取高度（相对于物体）
#     post_grasp_height: float = 0.2  # 抓取后提升高度
    
#     def __post_init__(self):
#         """初始化后处理。"""
#         self.func = grasp_action
        
#         # 设置参数字典
#         params_dict = {
#             "asset_name": self.asset_name,
#             "ee_body_name": self.ee_body_name,
#             "gripper_joint_names": self.gripper_joint_names,
#             "pre_grasp_height": self.pre_grasp_height,
#             "grasp_height": self.grasp_height,
#             "post_grasp_height": self.post_grasp_height,
#         }
        
#         self.params = params_dict

# def grasp_action(action_dict: Dict, asset_name: str, ee_body_name: str,
#                 gripper_joint_names: List[str], pre_grasp_height: float,
#                 grasp_height: float, post_grasp_height: float) -> Dict:
#     """实现抓取动作逻辑。
    
#     Args:
#         action_dict: 动作字典（从action_selector）
#         asset_name: 机器人资产名称
#         ee_body_name: 末端执行器Body名称
#         gripper_joint_names: 抓手关节名称
#         pre_grasp_height: 抓取前的高度
#         grasp_height: 抓取高度
#         post_grasp_height: 抓取后提升高度
        
#     Returns:
#         Dict: 处理后的抓取动作字典
#     """
#     # 这里实现将选择器输出转换为具体的抓取动作
#     grasp_dict = {}
    
#     # 仅当存在抓取掩码为1的环境时处理
#     if torch.any(action_dict["grasp_mask"] > 0):
#         # 获取抓取位置和角度
#         grasp_pos_x = action_dict["position"]["x"]
#         grasp_pos_y = action_dict["position"]["y"]
#         grasp_pos_z = action_dict["position"]["z"]
#         grasp_angle = action_dict["angle"] * 2.0 * np.pi  # 转换为弧度
        
#         # 设置机器人属性
#         grasp_dict["asset_name"] = asset_name
#         grasp_dict["ee_body_name"] = ee_body_name
#         grasp_dict["gripper_joint_names"] = gripper_joint_names
        
#         # 设置抓取轨迹
#         grasp_dict["trajectory"] = {
#             "pre_grasp": {
#                 "x": grasp_pos_x,
#                 "y": grasp_pos_y,
#                 "z": grasp_pos_z + pre_grasp_height,
#                 "angle": grasp_angle,
#             },
#             "grasp": {
#                 "x": grasp_pos_x,
#                 "y": grasp_pos_y,
#                 "z": grasp_pos_z + grasp_height,
#                 "angle": grasp_angle,
#             },
#             "post_grasp": {
#                 "x": grasp_pos_x,
#                 "y": grasp_pos_y,
#                 "z": grasp_pos_z + post_grasp_height,
#                 "angle": grasp_angle,
#             },
#         }
        
#         # 设置抓取控制
#         grasp_dict["gripper_control"] = {
#             "pre_grasp": 0.0,  # 完全打开
#             "grasp": 1.0,  # 完全闭合
#             "post_grasp": 1.0,  # 保持闭合
#         }
        
#         # 设置掩码
#         grasp_dict["mask"] = action_dict["grasp_mask"]
    
#     return grasp_dict 