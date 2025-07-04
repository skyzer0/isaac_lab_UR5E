#!/usr/bin/env python

# ./isaaclab.sh -p source/standalone/sky/fyp_sky/visual_pushing_grasping.py

# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to run a visual pushing and grasping (VPG) environment in Isaac Lab.

This implementation is inspired by the original VPG paper:
"Learning Synergies between Pushing and Grasping with Self-supervised Deep Reinforcement Learning"
by Zeng et al. (IROS 2018)

The code integrates the VPG approach into the Isaac Lab environment,
implementing both pushing and grasping actions based on RGB-D observations.
"""

"""Launch Omniverse Toolkit first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Visual Pushing and Grasping for Isaac Lab")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--save",
    action="store_true",
    default=False,
    help="Save the data from camera at index specified by ``--camera_id``.",
)
parser.add_argument(
    "--save_ply",
    action="store_true",
    default=False,
    help="Save pointcloud data in PLY format.",
)
parser.add_argument(
    "--rule_based",
    action="store_true",
    default=True,
    help="Use rule-based policy instead of neural network.",
)
parser.add_argument(
    "--camera_id",
    type=int,
    choices={0, 1},
    default=0,
    help=(
        "The camera ID to use for displaying points or saving the camera data. Default is 0 (top_camera)."
        " Camera 1 is side_camera. The viewport will always initialize with the perspective of camera 0."
    ),
)
parser.add_argument(
    "--workspace_limits",
    type=float,
    nargs='+',
    default=[0.3, 0.7, -0.2, 0.2, 0.0, 0.3],
    help="Workspace limits [x_min, x_max, y_min, y_max, z_min, z_max]",
)
parser.add_argument(
    "--heightmap_resolution",
    type=float,
    default=0.002,
    help="Meters per pixel of heightmap.",
)
parser.add_argument(
    "--debug",
    action="store_true",
    default=False,
    help="Enable debug output (additional visualizations and logging).",
)

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# Parse the arguments
args_cli = parser.parse_args()
args_cli.enable_cameras = True
# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

import gymnasium as gym
import omni.isaac.lab.sim as sim_utils
from collections.abc import Sequence
import warp as wp
from omni.isaac.lab.assets.rigid_object.rigid_object_data import RigidObjectData
import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg
from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg
from omni.isaac.lab.sensors import CameraData
import omni.replicator.core as rep
from omni.isaac.lab.utils import convert_dict_to_backend
from omni.isaac.lab.sensors import TiledCamera, TiledCameraCfg, save_images_to_file
import os
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from collections import OrderedDict

# 导入自定义工具函数
from vpg_utils import generate_heightmaps, get_affordance_vis, get_action_visualization


# Initialize warp
wp.init()

# Set random seed
np.random.seed(42)
torch.manual_seed(42)

# Convert workspace limits to the format expected by the environment
workspace_limits = np.array([
    [args_cli.workspace_limits[0], args_cli.workspace_limits[1]],  # x
    [args_cli.workspace_limits[2], args_cli.workspace_limits[3]],  # y
    [args_cli.workspace_limits[4], args_cli.workspace_limits[5]]   # z
])
heightmap_resolution = args_cli.heightmap_resolution

# Define action modes
class ActionMode:
    """Action modes for VPG."""
    PUSH = 0
    GRASP = 1

# Define states for robot actions
class RobotState:
    """机器人状态机的所有可能状态。

    状态转换流程：
        IDLE -> APPROACH_TOP (当设置新动作时)
        APPROACH_TOP -> PUSHING (如果是推动动作)
        APPROACH_TOP -> APPROACH_OBJECT (如果是抓取动作)
        APPROACH_OBJECT -> GRASPING
        GRASPING -> LIFTING
        LIFTING -> RETURNING_HOME
        PUSHING -> RETREATING
        RETREATING -> RETURNING_HOME
        RETURNING_HOME -> IDLE
    """
    IDLE = 0             # 空闲状态，机器人在家位置等待
    APPROACH_TOP = 1     # 从上方接近目标位置，保持安全高度
    APPROACH_OBJECT = 2  # 从上方开始下降，接近目标物体
    PUSHING = 3          # 推动物体中，先下降后水平推动
    GRASPING = 4         # 抓取物体中，关闭夹爪
    LIFTING = 5          # 提起物体中，将物体提升到安全高度
    RETREATING = 6       # 后退中，推动后从接触位置后退到安全高度
    RETURNING_HOME = 7   # 返回初始位置中，完成动作后的最后阶段

class GripperState:
    """States for the gripper."""
    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)

# Rule-based policy for pushing and grasping
class RuleBasedPolicy:
    """A simple rule-based policy for pushing and grasping."""
    
    def __init__(self, workspace_limits, heightmap_resolution):
        """Initialize the policy.
        
        Args:
            workspace_limits: Workspace limits [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
            heightmap_resolution: Meters per pixel of heightmap
        """
        self.workspace_limits = workspace_limits
        self.heightmap_resolution = heightmap_resolution
        self.push_step_size = 0.05  # meters
        
    def get_action(self, color_heightmap, depth_heightmap):
        """Get the next action based on current heightmaps.
        
        Args:
            color_heightmap: Color heightmap (H, W, 3)
            depth_heightmap: Depth heightmap (H, W)
            
        Returns:
            action_mode: Push (0) or grasp (1)
            best_pix_ind: Best pixel index (y, x) for the action
            best_rotation_angle: Best rotation angle for the action (in radians)
        """
        # Create copy of depth heightmap
        depth_heightmap_copy = depth_heightmap.copy()
        
        # Filter out nan values
        depth_heightmap_copy[np.isnan(depth_heightmap_copy)] = 0
        
        # Compute gradient of the depth image
        Gx = cv2.Sobel(depth_heightmap_copy, cv2.CV_64F, 1, 0, ksize=5)
        Gy = cv2.Sobel(depth_heightmap_copy, cv2.CV_64F, 0, 1, ksize=5)
        
        # Compute edge magnitude
        edge_magnitude = np.sqrt(Gx**2 + Gy**2)
        
        # Compute edge orientation
        edge_orientation = np.arctan2(Gy, Gx)
        
        # Apply thresholding to find edges
        edge_mask = edge_magnitude > 0.01  # Threshold can be tuned
        
        # Look for grasping points along edges
        grasp_candidates = np.zeros_like(depth_heightmap_copy)
        
        # Simple rule: try to grasp at points with high edge magnitude
        grasp_candidates[edge_mask] = edge_magnitude[edge_mask]
        
        # Apply a distance transform to find points away from edges for pushing
        dist_transform = cv2.distanceTransform((~edge_mask).astype(np.uint8), cv2.DIST_L2, 5)
        push_candidates = dist_transform.copy()
        
        # Normalize candidates
        if np.max(grasp_candidates) > 0:
            grasp_candidates = grasp_candidates / np.max(grasp_candidates)
        if np.max(push_candidates) > 0:
            push_candidates = push_candidates / np.max(push_candidates)
        
        # Decide whether to push or grasp
        best_grasp_score = np.max(grasp_candidates) if np.max(grasp_candidates) > 0 else 0
        best_push_score = np.max(push_candidates) if np.max(push_candidates) > 0 else 0
        
        # Simple heuristic: if there are good grasp points, grasp, otherwise push
        if best_grasp_score > 0.5 and best_grasp_score > best_push_score:
            # Grasp
            action_mode = ActionMode.GRASP
            best_pix_ind = np.unravel_index(np.argmax(grasp_candidates), grasp_candidates.shape)
            best_rotation_angle = edge_orientation[best_pix_ind]
        else:
            # Push
            action_mode = ActionMode.PUSH
            best_pix_ind = np.unravel_index(np.argmax(push_candidates), push_candidates.shape)
            
            # Choose random direction for pushing
            best_rotation_angle = np.random.uniform(0, np.pi)
        
        # Visualize affordances if debug mode is on
        if args_cli.debug:
            affordance_vis = get_affordance_vis(grasp_candidates, push_candidates)
            action_vis = get_action_visualization(
                color_heightmap, depth_heightmap, 
                action_mode, best_pix_ind[1], best_pix_ind[0], 
                best_rotation_angle
            )
            
            # Save visualizations
            os.makedirs(os.path.join("output", "VPG", "viz"), exist_ok=True)
            timestamp = int(time.time())
            cv2.imwrite(os.path.join("output", "VPG", "viz", f"affordance_{timestamp}.png"), affordance_vis)
            cv2.imwrite(os.path.join("output", "VPG", "viz", f"action_{timestamp}.png"), action_vis)
        
        return action_mode, best_pix_ind, best_rotation_angle

# Class for implementing pushing and grasping actions
class PushingGraspingController:
    """Controller for pushing and grasping actions."""
    
    def __init__(self, dt, num_envs, device, workspace_limits, heightmap_resolution):
        """Initialize the controller.
        
        Args:
            dt: Time step
            num_envs: Number of environments
            device: Computation device
            workspace_limits: Workspace limits
            heightmap_resolution: Heightmap resolution
        """
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        self.workspace_limits = workspace_limits
        self.heightmap_resolution = heightmap_resolution
        
        # 初始化狀態機
        self.state = torch.full((self.num_envs,), RobotState.IDLE, dtype=torch.int32, device=self.device)
        self.wait_time = torch.zeros((self.num_envs,), device=self.device)
        
        # 初始化動作參數
        self.action_mode = ActionMode.GRASP  # 默認抓取
        self.target_position = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_rotation = torch.zeros((self.num_envs,), device=self.device)
        self.gripper_state = torch.full((self.num_envs,), float(GripperState.OPEN), device=self.device)
        
        # 初始化機器人的家位置（工作空間中心上方的安全位置）
        x_center = (workspace_limits[0][0] + workspace_limits[0][1]) / 2
        y_center = (workspace_limits[1][0] + workspace_limits[1][1]) / 2
        z_safe = workspace_limits[2][1] + 0.15  # 工作空間上方15厘米
        self.home_position = torch.tensor([x_center, y_center, z_safe], device=self.device).repeat(num_envs, 1)
        
        # 推動參數
        self.push_start_position = torch.zeros((self.num_envs, 3), device=self.device)
        self.push_end_position = torch.zeros((self.num_envs, 3), device=self.device)
        self.push_progress = torch.zeros((self.num_envs,), device=self.device)
        self.push_step_size = 0.05  # 米
        
        # 所需狀態
        self.des_ee_pose = torch.zeros((self.num_envs, 7), device=self.device)  # 位置 + 四元數
        self.des_gripper_state = torch.full((self.num_envs,), 0.0, device=self.device)
        
        # 接近高度偏移
        self.approach_height = 0.1  # 目標上方10厘米
        
        # 不同狀態的等待時間
        self.wait_time_idle = 0.2
        self.wait_time_approach = 0.5
        self.wait_time_push = 0.5
        self.wait_time_grasp = 0.5
        self.wait_time_lift = 0.8
        self.wait_time_retreat = 0.5
        
        # 動作完成標誌
        self.action_complete = torch.full((self.num_envs,), True, dtype=torch.bool, device=self.device)
    
    def reset_idx(self, env_ids=None):
        """重置控制器。"""
        if env_ids is None:
            env_ids = slice(None)
        self.state[env_ids] = RobotState.IDLE
        self.wait_time[env_ids] = 0.0
        self.gripper_state[env_ids] = float(GripperState.OPEN)
        self.action_complete[env_ids] = True
    
    def set_action(self, action_mode, target_position, target_rotation):
        """設置動作參數。
        
        Args:
            action_mode: 推動或抓取
            target_position: 目標位置 (x, y, z)
            target_rotation: 目標旋轉角度
        """
        self.action_mode = action_mode
        self.target_position = torch.tensor(target_position, device=self.device).reshape(self.num_envs, 3)
        self.target_rotation = torch.tensor(target_rotation, device=self.device)
        
        # 重置狀態機
        self.state = torch.full((self.num_envs,), RobotState.APPROACH_TOP, dtype=torch.int32, device=self.device)
        self.wait_time = torch.zeros((self.num_envs,), device=self.device)
        self.action_complete = torch.full((self.num_envs,), False, dtype=torch.bool, device=self.device)
        
        # 計算推動的起點和終點
        if action_mode == ActionMode.PUSH:
            # 起點位置在目標上方
            self.push_start_position[:, 0] = self.target_position[:, 0] - self.push_step_size * np.cos(target_rotation)
            self.push_start_position[:, 1] = self.target_position[:, 1] - self.push_step_size * np.sin(target_rotation)
            self.push_start_position[:, 2] = self.target_position[:, 2] + self.approach_height
            
            # 終點位置在目標另一側
            self.push_end_position[:, 0] = self.target_position[:, 0] + self.push_step_size * np.cos(target_rotation)
            self.push_end_position[:, 1] = self.target_position[:, 1] + self.push_step_size * np.sin(target_rotation)
            self.push_end_position[:, 2] = self.target_position[:, 2] + 0.005  # 稍微高於表面以避免碰撞
            
            self.push_progress = torch.zeros((self.num_envs,), device=self.device)
    
    def compute(self, ee_pose: torch.Tensor, step_dt: float):
        """計算所需的末端效應器姿態和夾爪狀態。
        
        Args:
            ee_pose: 當前末端效應器姿態 (位置 + 四元數)
            step_dt: 時間步長
            
        Returns:
            actions: 機器人的動作(所需的ee姿態 + 夾爪狀態)
        """
        # 更新等待時間
        self.wait_time += step_dt
        
        # 状态机执行逻辑
        # 每个状态包含两部分：
        # 1. 在状态内部的行为（主要是位置插值和夹爪控制）
        # 2. 状态的转换条件（通常基于等待时间）
        
        if self.state[0] == RobotState.IDLE:
            # IDLE状态：机器人保持在家位置，夹爪打开，等待新任务
            # 此状态会一直持续直到收到新的set_action调用
            self.des_ee_pose[0, :3] = self.home_position[0]
            self.des_ee_pose[0, 3:] = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device)  # 示例四元數
            self.gripper_state[0] = float(GripperState.OPEN)
            self.action_complete[0] = True
        
        elif self.state[0] == RobotState.APPROACH_TOP:
            # APPROACH_TOP状态：机器人移动到目标位置正上方的安全高度
            # 目的是在进入实际操作前先到达目标区域上方
            if self.wait_time[0] < self.wait_time_approach:
                # 计算在目标上方的位置
                top_position = self.target_position[0].clone()
                top_position[2] = self.target_position[0, 2] + self.approach_height
                
                # 设置所需的末端效应器姿态
                self.des_ee_pose[0, :3] = top_position
                self.des_ee_pose[0, 3:] = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device)
                
                # 保持夹爪打开
                self.gripper_state[0] = float(GripperState.OPEN)
            else:
                # 状态转换：等待时间结束后，根据动作类型选择下一个状态
                if self.action_mode == ActionMode.PUSH:
                    # 如果是推动动作，直接进入推动状态
                    self.state[0] = RobotState.PUSHING
                else:  # GRASP
                    # 如果是抓取动作，先接近物体
                    self.state[0] = RobotState.APPROACH_OBJECT
                
                # 重置等待时间以便下一个状态使用
                self.wait_time[0] = 0.0
        
        elif self.state[0] == RobotState.APPROACH_OBJECT:
            # APPROACH_OBJECT状态：机器人从安全高度下降到物体位置
            # 这是抓取操作的准备阶段
            if self.wait_time[0] < self.wait_time_approach:
                # 计算接近物体的进度（0.0到1.0之间的值）
                approach_progress = min(1.0, float(self.wait_time[0] / self.wait_time_approach))
                
                # 基于进度计算位置：保持XY不变，Z轴随进度下降
                pos_x = float(self.target_position[0, 0])
                pos_y = float(self.target_position[0, 1])
                pos_z = float(self.target_position[0, 2]) + self.approach_height * (1.0 - approach_progress)
                
                # 设置所需的末端效应器姿态
                self.des_ee_pose[0, :3] = torch.tensor([pos_x, pos_y, pos_z], device=self.device)
                
                # 计算抓取方向的四元数（可以根据目标旋转角度调整）
                rotation_rad = float(self.target_rotation[0])
                
                # 设置姿态，在实际应用中应根据抓取方向调整
                self.des_ee_pose[0, 3:] = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device)
                
                # 保持夹爪打开准备抓取
                self.gripper_state[0] = float(GripperState.OPEN)
            else:
                # 状态转换：等待时间结束后进入抓取状态
                self.state[0] = RobotState.GRASPING
                self.wait_time[0] = 0.0
        
        elif self.state[0] == RobotState.PUSHING:
            # PUSHING状态：执行推动操作，先下降到物体高度，然后水平推动
            if self.wait_time[0] < self.wait_time_push:
                # 计算推动进度
                self.push_progress[0] = min(1.0, float(self.wait_time[0] / self.wait_time_push))
                progress_float = float(self.push_progress[0])
                
                # 分阶段执行推动：
                # 1. 前30%时间用于下降到目标高度
                # 2. 后70%时间用于水平推动
                down_progress = min(0.3, progress_float) / 0.3
                push_progress = (progress_float - 0.3) / 0.7 if progress_float > 0.3 else 0.0
                
                # 计算位置插值
                pos_x = float(self.push_start_position[0, 0]) * (1.0 - self.push_progress[0]) + float(self.push_end_position[0, 0]) * self.push_progress[0]
                pos_y = float(self.push_start_position[0, 1]) * (1.0 - self.push_progress[0]) + float(self.push_end_position[0, 1]) * self.push_progress[0]
                
                # Z轴位置根据阶段计算
                if progress_float <= 0.3:
                    # 下降阶段
                    pos_z = float(self.push_start_position[0, 2]) * (1.0 - down_progress) + float(self.push_end_position[0, 2]) * down_progress
                else:
                    # 推动阶段
                    pos_z = float(self.push_end_position[0, 2])
                
                # 设置所需的末端效应器姿态
                self.des_ee_pose[0, :3] = torch.tensor([pos_x, pos_y, pos_z], device=self.device)
                self.des_ee_pose[0, 3:] = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device)
                
                # 推动时保持夹爪打开
                self.gripper_state[0] = float(GripperState.OPEN)
            else:
                # 状态转换：等待时间结束后进入后退状态
                self.state[0] = RobotState.RETREATING
                self.wait_time[0] = 0.0
        
        elif self.state[0] == RobotState.GRASPING:
            # GRASPING状态：到达物体位置并关闭夹爪抓取物体
            if self.wait_time[0] < self.wait_time_grasp:
                # 保持在目标位置
                pos_x = float(self.target_position[0, 0])
                pos_y = float(self.target_position[0, 1])
                pos_z = float(self.target_position[0, 2])
                
                # 设置所需的末端效应器姿态
                self.des_ee_pose[0, :3] = torch.tensor([pos_x, pos_y, pos_z], device=self.device)
                self.des_ee_pose[0, 3:] = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device)
                
                # 关闭夹爪抓取物体
                self.gripper_state[0] = float(GripperState.CLOSE)
            else:
                # 状态转换：等待时间结束后进入提升状态
                self.state[0] = RobotState.LIFTING
                self.wait_time[0] = 0.0
        
        elif self.state[0] == RobotState.LIFTING:
            # LIFTING状态：抓取物体后将其提升到安全高度
            if self.wait_time[0] < self.wait_time_lift:
                # 计算提升进度
                lift_progress = min(1.0, float(self.wait_time[0] / self.wait_time_lift))
                
                # 基于进度计算位置：保持XY不变，Z轴随进度上升
                pos_x = float(self.target_position[0, 0])
                pos_y = float(self.target_position[0, 1])
                pos_z = float(self.target_position[0, 2]) + self.approach_height * lift_progress
                
                # 设置所需的末端效应器姿态
                self.des_ee_pose[0, :3] = torch.tensor([pos_x, pos_y, pos_z], device=self.device)
                self.des_ee_pose[0, 3:] = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device)
                
                # 保持夹爪关闭以继续抓住物体
                self.gripper_state[0] = float(GripperState.CLOSE)
            else:
                # 状态转换：等待时间结束后进入返回家位置状态
                self.state[0] = RobotState.RETURNING_HOME
                self.wait_time[0] = 0.0
        
        elif self.state[0] == RobotState.RETREATING:
            # RETREATING状态：推动操作后从推动位置后退到安全高度
            if self.wait_time[0] < self.wait_time_retreat:
                # 计算后退进度
                retreat_progress = min(1.0, float(self.wait_time[0] / self.wait_time_retreat))
                
                # 保持XY不变，Z轴随进度上升
                pos_x = float(self.push_end_position[0, 0])
                pos_y = float(self.push_end_position[0, 1])
                pos_z = float(self.push_end_position[0, 2]) + self.approach_height * retreat_progress
                
                # 设置所需的末端效应器姿态
                self.des_ee_pose[0, :3] = torch.tensor([pos_x, pos_y, pos_z], device=self.device)
                self.des_ee_pose[0, 3:] = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device)
                
                # 保持夹爪打开
                self.gripper_state[0] = float(GripperState.OPEN)
            else:
                # 状态转换：等待时间结束后进入返回家位置状态
                self.state[0] = RobotState.RETURNING_HOME
                self.wait_time[0] = 0.0
        
        elif self.state[0] == RobotState.RETURNING_HOME:
            # RETURNING_HOME状态：动作完成后返回初始安全位置
            if self.wait_time[0] < self.wait_time_approach:
                # 计算返回进度
                return_progress = min(1.0, float(self.wait_time[0] / self.wait_time_approach))
                
                # 从当前位置平滑过渡到家位置
                current_pos = self.des_ee_pose[0, :3].clone()
                pos_x = float(current_pos[0]) * (1.0 - return_progress) + float(self.home_position[0, 0]) * return_progress
                pos_y = float(current_pos[1]) * (1.0 - return_progress) + float(self.home_position[0, 1]) * return_progress
                pos_z = float(current_pos[2]) * (1.0 - return_progress) + float(self.home_position[0, 2]) * return_progress
                
                # 设置所需的末端效应器姿态
                self.des_ee_pose[0, :3] = torch.tensor([pos_x, pos_y, pos_z], device=self.device)
                self.des_ee_pose[0, 3:] = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device)
                
                # 根据动作类型决定夹爪状态
                # - 推动：夹爪打开
                # - 抓取：夹爪关闭（保持抓住物体）
                if self.action_mode == ActionMode.GRASP:
                    self.gripper_state[0] = float(GripperState.CLOSE)
                else:
                    self.gripper_state[0] = float(GripperState.OPEN)
            else:
                # 状态转换：等待时间结束后回到空闲状态
                # 标记动作完成，准备接受新任务
                self.state[0] = RobotState.IDLE
                self.wait_time[0] = 0.0
                self.action_complete[0] = True
                print(f"動作{'抓取' if self.action_mode == ActionMode.GRASP else '推動'}完成")
        
        # 返回動作
        return torch.cat([self.des_ee_pose, self.gripper_state.unsqueeze(-1)], dim=-1)

def main():
    # Parse configuration - we won't use type annotation to avoid the lint error
    env_cfg = parse_env_cfg(
        "Isaac-Lift-Cube-UR5e-IK-Abs-v0",
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    
    # 打印环境配置中的相机设置信息
    if args_cli.debug:
        print("\n==== 相机配置信息 ====")
        print("顶部相机位置:", env_cfg.scene.top_camera.offset.pos)
        print("顶部相机朝向:", env_cfg.scene.top_camera.offset.rot)
        print("侧面相机位置:", env_cfg.scene.side_camera.offset.pos)
        print("侧面相机朝向:", env_cfg.scene.side_camera.offset.rot)
        print("========================\n")
    
    # Create environment
    env = gym.make("Isaac-Lift-Cube-UR5e-IK-Abs-v0", cfg=env_cfg)
    
    # Reset environment at start
    obs_dict = env.reset()
    
    # Get device and action shape from environment
    device = env.unwrapped.device
    action_shape = env.unwrapped.action_space.shape
    
    # Create action buffers (position + quaternion + gripper)
    actions = torch.zeros(action_shape, device=device)
    actions[:, 3] = 1.0  # quaternion w component
    
    # Create controller
    controller = PushingGraspingController(
        env_cfg.sim.dt * env_cfg.decimation,
        env.unwrapped.num_envs,
        device,
        workspace_limits,
        heightmap_resolution
    )
    
    # Create rule-based policy
    policy = RuleBasedPolicy(workspace_limits, heightmap_resolution)
    
    # 输出相机ID说明
    print(f"使用相机ID: {args_cli.camera_id} ({'顶部相机' if args_cli.camera_id == 0 else '侧面相机'})")
    
    # Camera data - 确保使用正确的相机名称
    camera_names = ["top_camera", "side_camera"]
    camera_name = camera_names[args_cli.camera_id]
    camera = env.unwrapped.scene[camera_name]
    camera_index = 0  # 每个相机单独用索引0
    
    if args_cli.debug:
        print(f"选择的相机: {camera_name}")
        print(f"可用的场景物体: {list(env.unwrapped.scene.keys())}")
    
    # Output directory for camera data
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "VPG")
    os.makedirs(output_dir, exist_ok=True)
    
    # Replicator writer for saving images
    rep_writer = rep.BasicWriter(
        output_dir=output_dir,
        frame_padding=0,
        colorize_instance_id_segmentation=False, 
        colorize_instance_segmentation=False,
        colorize_semantic_segmentation=False
    )
    
    # Action execution flag
    executing_action = False
    
    # Heightmaps
    color_heightmap = None
    depth_heightmap = None
    
    # Main simulation loop
    while simulation_app.is_running():
        # Run everything in inference mode
        with torch.inference_mode():
            # Step environment
            obs_dict, reward, terminated, truncated, info = env.step(actions)
            dones = terminated
            
            # Get observations
            # -- end-effector frame
            ee_frame_sensor = env.unwrapped.scene["ee_frame"]
            tcp_position = ee_frame_sensor.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
            tcp_orientation = ee_frame_sensor.data.target_quat_w[..., 0, :].clone()
            ee_pose = torch.cat([tcp_position, tcp_orientation], dim=-1)
            
            # -- object frame (for visualization and debugging)
            object_data = env.unwrapped.scene["object1"].data
            object_position = object_data.root_pos_w - env.unwrapped.scene.env_origins
            
            # Get RGB-D images if we're not currently executing an action
            if not executing_action:
                # Get camera data
                single_cam_data = convert_dict_to_backend(camera.data.output[camera_index], backend="numpy")
                color_img = single_cam_data["rgb"]
                depth_img = single_cam_data["distance_to_camera"][:, :, 0]
                
                # 打印相机数据信息
                if args_cli.debug:
                    print(f"\n==== {camera_name} 相机数据 ====")
                    print(f"色彩图像尺寸: {color_img.shape}")
                    print(f"深度图像尺寸: {depth_img.shape}")
                    print(f"深度范围: {np.nanmin(depth_img):.3f} 到 {np.nanmax(depth_img):.3f} 米")
                    if np.isnan(depth_img).any():
                        print(f"注意: 深度图包含 {np.sum(np.isnan(depth_img))} 个NaN值")
                
                # Get camera parameters
                camera_intrinsics = camera.data.intrinsic_matrices[camera_index].cpu().numpy()
                
                # 构建相机位姿矩阵（4x4）
                # 从相机数据中获取位置和方向
                camera_pos = camera.data.pos_w[camera_index].cpu().numpy()
                camera_quat = camera.data.quat_w_world[camera_index].cpu().numpy()
                
                # 从四元数创建旋转矩阵
                from scipy.spatial.transform import Rotation as R
                rot_matrix = R.from_quat([camera_quat[1], camera_quat[2], camera_quat[3], camera_quat[0]]).as_matrix()
                
                # 构建4x4位姿矩阵
                camera_pose = np.eye(4)
                camera_pose[:3, :3] = rot_matrix
                camera_pose[:3, 3] = camera_pos
                
                # 打印相机位姿信息
                if args_cli.debug:
                    print(f"\n相机位置: {camera_pos}")
                    print(f"相机四元数: {camera_quat}")
                    print(f"相机内参矩阵:\n{camera_intrinsics}")
                
                # 当前时间戳（用于保存文件名）
                timestamp = int(time.time())
                
                # Generate heightmaps
                color_heightmap, depth_heightmap = generate_heightmaps(
                    color_img, 
                    depth_img, 
                    camera_intrinsics, 
                    camera_pose, 
                    workspace_limits, 
                    heightmap_resolution,
                    device=str(device),
                    save_ply=args_cli.save_ply,
                    output_dir=output_dir if args_cli.save or args_cli.save_ply else None,
                    timestamp=timestamp
                )
                
                # Save heightmaps
                if args_cli.save:
                    cv2.imwrite(os.path.join(output_dir, f"color_heightmap_{timestamp}.png"), color_heightmap)
                    np.save(os.path.join(output_dir, f"depth_heightmap_{timestamp}.npy"), depth_heightmap)
                
                # Get action from policy
                action_mode, best_pix_ind, best_rotation_angle = policy.get_action(color_heightmap, depth_heightmap)
                
                # Convert pixel indices to 3D world coordinates
                pixel_x = best_pix_ind[1]
                pixel_y = best_pix_ind[0]
                
                # Calculate 3D position
                position_x = pixel_x * heightmap_resolution + workspace_limits[0][0]
                position_y = pixel_y * heightmap_resolution + workspace_limits[1][0]
                
                # Get z position from depth heightmap
                if np.isnan(depth_heightmap[pixel_y, pixel_x]):
                    position_z = workspace_limits[2][0]
                else:
                    position_z = depth_heightmap[pixel_y, pixel_x] + workspace_limits[2][0]
                
                target_position = np.array([position_x, position_y, position_z])
                
                # Set the action
                controller.set_action(action_mode, target_position, best_rotation_angle)
                executing_action = True
                print(f"执行{'推动' if action_mode == ActionMode.PUSH else '抓取'}动作，位置：{target_position}")
            
            # Compute actions from controller
            actions = controller.compute(ee_pose, env_cfg.sim.dt * env_cfg.decimation)
            
            # Check if action is complete
            if controller.state[0] == RobotState.IDLE and executing_action:
                executing_action = False
                print("动作执行完成")
            
            # Save camera data if requested
            if args_cli.save:
                # Get camera data
                single_cam_data = convert_dict_to_backend(camera.data.output[camera_index], backend="numpy")
                single_cam_info = camera.data.info[camera_index]
                
                # Pack data for saving
                rep_output = {"annotators": {}}
                for key, data, info in zip(single_cam_data.keys(), single_cam_data.values(), single_cam_info.values()):
                    if info is not None:
                        rep_output["annotators"][key] = {"render_product": {"data": data, **info}}
                    else:
                        rep_output["annotators"][key] = {"render_product": {"data": data}}
                
                # Add on-time data for Replicator
                rep_output["trigger_outputs"] = {"on_time": camera.frame[camera_index]}
                
                # Save images
                rep_writer.write(rep_output)
                
                # 另外保存单独的RGB和深度图像，便于查看
                os.makedirs(os.path.join(output_dir, "rgb"), exist_ok=True)
                os.makedirs(os.path.join(output_dir, "depth"), exist_ok=True)
                timestamp = int(time.time())
                cv2.imwrite(os.path.join(output_dir, "rgb", f"rgb_{timestamp}.png"), 
                            single_cam_data["rgb"])
                
                # 保存深度图(归一化为可视化)
                depth_for_vis = single_cam_data["distance_to_camera"][:, :, 0].copy()
                depth_valid = depth_for_vis[~np.isnan(depth_for_vis)]
                if len(depth_valid) > 0:
                    depth_min = np.min(depth_valid)
                    depth_max = np.max(depth_valid)
                    depth_range = depth_max - depth_min
                    if depth_range > 0:
                        depth_normalized = (depth_for_vis - depth_min) / depth_range * 255
                        depth_normalized[np.isnan(depth_for_vis)] = 0
                        cv2.imwrite(os.path.join(output_dir, "depth", f"depth_{timestamp}.png"), 
                                    depth_normalized.astype(np.uint8))
            
            # Reset controller if environment is done
            if any(dones):
                # Find which environments are done
                done_env_ids = torch.nonzero(torch.tensor(dones, device=device), as_tuple=False).squeeze(-1)
                controller.reset_idx(done_env_ids)
    
    # Close the environment
    env.close()

if __name__ == "__main__":
    # Run the main function
    main()
    # Close sim app
    simulation_app.close() 