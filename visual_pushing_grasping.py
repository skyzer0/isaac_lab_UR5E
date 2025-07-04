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
    default=False,
    help="Use rule-based policy instead of neural network.",
)
parser.add_argument(
    "--method",
    type=str,
    choices=["reactive", "reinforcement"],
    default="reinforcement",
    help="Learning method to use: reactive (supervised) or reinforcement.",
)
parser.add_argument(
    "--is_testing",
    action="store_true",
    default=False,
    help="Run in testing mode (no backprop).",
)
parser.add_argument(
    "--load_snapshot",
    action="store_true", 
    default=False,
    help="Load pre-trained model.",
)
parser.add_argument(
    "--snapshot_file",
    type=str,
    default=None,
    help="Path to pre-trained model snapshot.",
)
parser.add_argument(
    "--force_cpu",
    action="store_true",
    default=False,
    help="Force CPU use even when GPU is available",
)
parser.add_argument(
    "--experience_replay",
    action="store_true",
    default=False,
    help="Use experience replay to train model",
)
parser.add_argument(
    "--continue_logging",
    action="store_true",
    default=False,
    help="Continue logging from previous session",
)
parser.add_argument(
    "--logging_directory",
    type=str,
    default="logs",
    help="Directory for logging data",
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
    default=[0.35, 0.65, -0.15, 0.15, 0.0, 0.2],  # 调整工作空间范围，更贴近相机视野
    help="Workspace limits [x_min, x_max, y_min, y_max, z_min, z_max]",
)
parser.add_argument(
    "--heightmap_resolution",
    type=float,
    default=0.002,
    help="Meters per pixel of heightmap.",
)
parser.add_argument(
    "--num_rotations",
    type=int,
    default=16,
    help="Number of rotations to discretize for grasping",
)
parser.add_argument(
    "--debug",
    action="store_true",
    default=False,
    help="Enable debug output (additional visualizations and logging).",
)
parser.add_argument(
    "--save_camera_data",
    action="store_true",
    default=False,
    help="Save camera data in addition to RGB and depth images.",
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

# 導入訓練器和日誌記錄器
import sys
# 將當前目錄添加到模塊搜索路徑
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from sky.fyp.trainer import Trainer
from sky.fyp.logger import Logger

# 導入自定義工具函數
from vpg_utils import generate_heightmaps, get_affordance_vis, get_action_visualization, RuleBasedPolicy

# 添加后处理函数用于确保高度图有有效数据
def process_heightmaps(color_heightmap, depth_heightmap, workspace_limits):
    """处理高度图，确保有有效数据。
    
    Args:
        color_heightmap: 原始颜色高度图
        depth_heightmap: 原始深度高度图
        workspace_limits: 工作空间范围
        
    Returns:
        color_heightmap: 处理后的颜色高度图
        depth_heightmap: 处理后的深度高度图
    """
    # 检查深度高度图是否全为0
    if np.max(depth_heightmap) <= 0.0001:
        print("警告: 深度高度图全为0，正在生成模拟数据以进行测试...")
        
        # 生成一个简单的正方形物体在高度图中央
        h, w = depth_heightmap.shape
        center_h, center_w = h // 2, w // 2
        object_size = min(h, w) // 8  # 高度图尺寸的1/8
        
        # 在高度图中心创建一个模拟物体
        depth_heightmap[center_h-object_size:center_h+object_size, 
                        center_w-object_size:center_w+object_size] = 0.02  # 2厘米高的物体
        
        # 在颜色高度图中给物体上色
        color_heightmap[center_h-object_size:center_h+object_size, 
                        center_w-object_size:center_w+object_size, :] = [200, 0, 0]  # 红色
        
        print(f"已生成测试物体，位置为高度图中心，大小为{object_size*2}x{object_size*2}像素")
    
    return color_heightmap, depth_heightmap

# Initialize warp
wp.init()

# 检查CUDA是否可用
if torch.cuda.is_available():
    print("=================================================")
    print("CUDA可用，检测到以下GPU设备:")
    print(f"GPU设备: {torch.cuda.get_device_name(0)}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    print(f"当前GPU: {torch.cuda.current_device()}")
    print(f"当前CUDA版本: {torch.version.cuda}")
    print("=================================================")
else:
    print("=================================================")
    print("CUDA不可用，将使用CPU运行")
    print("=================================================")

# 重置随机种子，确保训练的可重复性
random_seed = 42
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # 如果使用多GPU
np.random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print(f"所有随机种子已设置为: {random_seed}")

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

# 用於場景變化檢測的閾值參數
DEPTH_DIFF_THRESH = 0.01  # 深度變化閾值 (米)
MIN_PIXEL_DIFF_THRESH = 50  # 最小像素變化數
REWARD_SCALING = 1.0  # 獎勵縮放係數

# 定義其他常數參數
PUSH_DISTANCE = 0.05  # 推動距離（米）
Z_TARGET = 0.001  # 目標Z高度 (米)

# Define action modes
class ActionMode:
    """Action modes for VPG."""
    PUSH = 0
    GRASP = 1
    
# 設置學習日誌和經驗緩衝區結構
TRANSITIONS_DIRECTORY = os.path.join('logs', 'transitions')

# 定義數據結構來跟蹤實驗過程
class Experiment:
    """跟踪实验状态和历史."""
    
    def __init__(self):
        """初始化实验跟踪器."""
        self.iteration = 0            # 实验迭代次数
        self.primitive_action = None  # 当前原始动作类型 (推动或抓取)
        self.push_success = False     # 推动是否成功
        self.grasp_success = False    # 抓取是否成功
        self.change_detected = False  # 是否检测到场景变化
        
        # 最佳操作点
        self.best_pix_ind = None      # 最佳像素索引 (旋转, y, x)
        self.best_pix_ind_coords = [0, 0]  # 最佳像素坐标 (y, x)，初始化为[0, 0]而不是None
        self.best_rotation_angle = 0  # 最佳旋转角度
        
        # 预测
        self.push_predictions = None  # 推动预测
        self.grasp_predictions = None  # 抓取预测
        
        # 场景变化检测相关
        self.prev_color_img = None    # 上一张彩色图像
        self.prev_depth_img = None    # 上一张深度图像
        self.prev_color_heightmap = None  # 上一张彩色高度图
        self.prev_depth_heightmap = None  # 上一张深度高度图
        self.prev_object_position = None  # 上一帧物体位置
        
        # 动作历史记录，用于检测重复动作
        self.action_history = {
            ActionMode.PUSH: [],  # 记录推动位置
            ActionMode.GRASP: []  # 记录抓取位置
        }
        self.location_similarity_threshold = 7  # 像素距离阈值
        self.history_max_size = 5  # 历史记录最大长度
        
        # 连续失败计数
        self.consecutive_failures = 0
        self.max_failures_before_boost = 2
        
        # 增强：区域探索记录，确保能够探索多个区域
        # 将高度图划分为3x3=9个区域
        self.explored_regions = []  # 记录已探索区域
        self.region_grid_size = 3   # 区域划分网格大小
    
    def set_primitive_action(self, action_mode):
        """设置动作类型."""
        self.primitive_action = action_mode
        # 重置动作成功标志
        if action_mode == ActionMode.PUSH:
            self.push_success = False
        elif action_mode == ActionMode.GRASP:
            self.grasp_success = False
        self.change_detected = False
    
    def set_predictions(self, push_preds, grasp_preds):
        """设置预测."""
        self.push_predictions = push_preds
        self.grasp_predictions = grasp_preds
    
    def set_best_pix_ind(self, pix_ind):
        """设置最佳像素索引."""
        self.best_pix_ind = pix_ind
    
    def set_coords(self, coords):
        """设置最佳坐标."""
        self.best_pix_ind_coords = coords
    
    def set_rotation_angle(self, angle):
        """设置旋转角度."""
        self.best_rotation_angle = angle
    
    def reset(self):
        """重置实验状态."""
        self.primitive_action = None
        self.push_success = False
        self.grasp_success = False
        self.change_detected = False
        self.best_pix_ind = None
        self.best_pix_ind_coords = [0, 0]  # 重置为[0, 0]而不是None
        self.best_rotation_angle = 0
        self.push_predictions = None
        self.grasp_predictions = None
        self.prev_color_img = None
        self.prev_depth_img = None
        self.prev_color_heightmap = None
        self.prev_depth_heightmap = None
        self.action_history = {ActionMode.PUSH: [], ActionMode.GRASP: []}
        self.consecutive_failures = 0
        self.explored_regions = []
    
    def add_action_to_memory(self, action_type, pixel_location):
        """添加动作到历史记录."""
        self.action_history[action_type].append(pixel_location)
        
        # 如果历史记录超过最大长度，移除最旧的记录
        if len(self.action_history[action_type]) > self.history_max_size:
            self.action_history[action_type].pop(0)
    
    def is_action_repeated(self, action_type, pixel_location):
        """检查动作是否重复."""
        if len(self.action_history[action_type]) == 0:
            return False
        
        # 计算与历史动作的距离
        for hist_loc in self.action_history[action_type]:
            # 计算欧几里得距离
            distance = np.sqrt((pixel_location[0] - hist_loc[0])**2 + 
                              (pixel_location[1] - hist_loc[1])**2)
            
            # 如果在阈值范围内，则认为是重复的
            if distance < self.location_similarity_threshold:
                return True
        
        return False
    
    def update_failure_count(self, success):
        """更新连续失败次数."""
        if success:
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1
    
    def get_exploration_boost(self):
        """根据连续失败次数获取探索提升率."""
        if self.consecutive_failures <= self.max_failures_before_boost:
            return 0.0  # 如果连续失败次数较少，不提升探索
        
        # 连续失败越多，探索率越高，但最高不超过0.9
        exploration_boost = min(0.9, 0.3 + (self.consecutive_failures - self.max_failures_before_boost) * 0.2)
        
        # 新增：如果已探索的区域数量较少，进一步提高探索率
        if len(self.explored_regions) < 4:  # 如果探索的区域少于一半
            exploration_boost = min(0.95, exploration_boost + 0.1)
            
        return exploration_boost
        
    def get_region_id(self, coords, heightmap_shape):
        """根据坐标获取区域ID."""
        y, x = coords
        h, w = heightmap_shape
        region_h = h // self.region_grid_size
        region_w = w // self.region_grid_size
        
        region_y = min(self.region_grid_size - 1, y // region_h)
        region_x = min(self.region_grid_size - 1, x // region_w)
        
        return region_y * self.region_grid_size + region_x
    
    def record_explored_region(self, coords, heightmap_shape):
        """记录已探索的区域."""
        region_id = self.get_region_id(coords, heightmap_shape)
        if region_id not in self.explored_regions:
            self.explored_regions.append(region_id)
            print(f"探索了新区域 {region_id}，已探索区域数：{len(self.explored_regions)}")

# 定义机器人状态
class RobotState:
    """機器人狀態機的所有可能狀態。

    狀態轉換流程：
        IDLE -> APPROACH_TOP (當設置新動作時)
        APPROACH_TOP -> PUSHING (如果是推動動作)
        APPROACH_TOP -> APPROACH_OBJECT (如果是抓取動作)
        APPROACH_OBJECT -> GRASPING
        GRASPING -> LIFTING
        LIFTING -> PLACING (新增：如果是抓取動作，先放置物体)
        PLACING -> RETREATING (新增：放置后进入后退状态)
        LIFTING -> RETREATING
        RETREATING -> RETURNING_HOME
        RETURNING_HOME -> IDLE
    """
    IDLE = 0             # 空閒狀態，機器人在家位置等待
    APPROACH_TOP = 1     # 從上方接近目標位置，保持安全高度
    APPROACH_OBJECT = 2  # 從上方開始下降，接近目標物體
    PUSHING = 3          # 推動物體中，先下降後水平推動
    GRASPING = 4         # 抓取物體中，關閉夾爪
    LIFTING = 5          # 提起物體中，將物體提升到安全高度
    RETREATING = 6       # 後退中，推動後從接觸位置後退到安全高度
    RETURNING_HOME = 7   # 返回初始位置中，完成動作後的最後階段
    PLACING = 8          # 新增：放置物体到指定位置
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
                best_rotation_angle,
                flip_x=True,  # 与坐标转换一致，X轴方向翻转
                flip_y=True   # 与坐标转换一致，Y轴方向翻转
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
        x_center = ((workspace_limits[0][0] + workspace_limits[0][1]) / 2)-0.2
        y_center = ((workspace_limits[1][0] + workspace_limits[1][1]) / 2)-0.2
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
        self.wait_time_place = 1.0  # 新增：放置状态等待时间
        
        # 動作完成標誌
        self.action_complete = torch.full((self.num_envs,), True, dtype=torch.bool, device=self.device)
        
        # 新增：放置位置参数
        self.place_position = None
        self.place_stage = 0  # 放置阶段：0=未放置，1=移动到上方，2=下降，3=放开
        
        # 新增：设置默认放置位置（工作空间边缘）
        place_x = workspace_limits[0][0] + 0.1  # 工作空间x方向边缘向内10cm
        place_y = workspace_limits[1][0] + 0.1  # 工作空间y方向边缘向内10cm
        place_z = workspace_limits[2][0]  # 工作空间底部
        self.place_position = torch.tensor([place_x, place_y, place_z], device=self.device).repeat(num_envs, 1)
    
    def reset_idx(self, env_ids=None):
        """重置控制器。"""
        if env_ids is None:
            env_ids = slice(None)
        self.state[env_ids] = RobotState.IDLE
        self.wait_time[env_ids] = 0.0
        self.gripper_state[env_ids] = float(GripperState.OPEN)
        self.action_complete[env_ids] = True
        self.place_stage = 0  # 重置放置阶段
    
    def set_place_position(self, place_position=None):
        """设置抓取后物体的放置位置
        
        Args:
            place_position: 放置位置 [x, y, z]，如果为None则使用默认位置
        """
        if place_position is not None:
            # 使用自定义放置位置，并扩展到所有环境
            place_position_tensor = torch.tensor(place_position, device=self.device)
            # 确保输入是正确的形状
            if place_position_tensor.shape[-1] != 3:
                raise ValueError(f"放置位置必须是形状为[3]的数组，当前形状: {place_position_tensor.shape}")
            # 扩展到所有环境
            self.place_position = place_position_tensor.reshape(1, 3).repeat(self.num_envs, 1)
            print(f"已设置自定义放置位置: [{place_position[0]:.4f}, {place_position[1]:.4f}, {place_position[2]:.4f}]")
        else:
            # 使用默认放置位置（工作空间边缘）
            place_x = self.workspace_limits[0][0] + 0.1  # 工作空间x方向边缘向内10cm
            place_y = self.workspace_limits[1][0] + 0.1  # 工作空间y方向边缘向内10cm
            place_z = self.workspace_limits[2][0]        # 工作空间底部
            default_place_pos = [place_x, place_y, place_z]
            self.place_position = torch.tensor(default_place_pos, device=self.device).reshape(1, 3).repeat(self.num_envs, 1)
            print(f"使用默认放置位置: [{place_x:.4f}, {place_y:.4f}, {place_z:.4f}]")
    
    def _euler_to_quaternion(self, roll, pitch, yaw):
        """欧拉角转四元数，保持爪子朝下并绕Z轴旋转
        
        Args:
            roll: X轴旋转角度（弧度）
            pitch: Y轴旋转角度（弧度）
            yaw: Z轴旋转角度（弧度）
            
        Returns:
            quat: 四元数 [qw, qx, qy, qz]，适用于torch张量
        """
        # 基础朝下四元数 - 代表机械臂垂直向下的姿态
        base_qw = 0.0
        base_qx = 1.0
        base_qy = 0.0
        base_qz = 0.0
        
        # 重要：基于实际测试结果调整角度
        # 根据set_action中的角度调整逻辑 (2*pi - yaw)，这里也使用相同的变换
        # 这确保了机器人的旋转方向与预测图中的方向一致
        adjusted_yaw = 2*np.pi - yaw
        
        # Z轴旋转的四元数
        half_yaw = adjusted_yaw * 0.5
        z_qw = np.cos(half_yaw)
        z_qx = 0.0
        z_qy = 0.0
        z_qz = np.sin(half_yaw)
        
        # 组合两个四元数（先朝下，再绕Z轴旋转）
        # 四元数乘法公式
        qw = z_qw * base_qw - z_qx * base_qx - z_qy * base_qy - z_qz * base_qz
        qx = z_qw * base_qx + z_qx * base_qw + z_qy * base_qz - z_qz * base_qy
        qy = z_qw * base_qy - z_qx * base_qz + z_qy * base_qw + z_qz * base_qx
        qz = z_qw * base_qz + z_qx * base_qy - z_qy * base_qx + z_qz * base_qw
        
        # 返回适用于torch的格式
        return torch.tensor([qw, qx, qy, qz], device=self.device)
    
    def set_action(self, action_mode, target_position, target_rotation):
        """設置動作參數。
        
        Args:
            action_mode: 推動或抓取
            target_position: 目標位置 (x, y, z) 或 numpy數組
            target_rotation: 目標旋轉角度
        """
        self.action_mode = action_mode
        
        # 確保target_position是正確的形狀
        target_position_tensor = torch.tensor(target_position, device=self.device)
        
        # 如果輸入是單個位置，則擴展到所有環境
        if len(target_position_tensor.shape) == 1:
            self.target_position = target_position_tensor.reshape(1, 3).repeat(self.num_envs, 1)
        else:
            self.target_position = target_position_tensor.reshape(self.num_envs, 3)
            
        self.target_rotation = torch.tensor(target_rotation, device=self.device)
        
        # 重置狀態機
        self.state = torch.full((self.num_envs,), RobotState.APPROACH_TOP, dtype=torch.int32, device=self.device)
        self.wait_time = torch.zeros((self.num_envs,), device=self.device)
        self.action_complete = torch.full((self.num_envs,), False, dtype=torch.bool, device=self.device)
        
        # 計算推動的起點和終點
        if action_mode == ActionMode.PUSH:
            # 重要：调整旋转角度，完全反向，使得推动方向与预测图一致
            # 实测发现使用 pi + angle 导致方向反向，因此使用 angle 或 2*pi - angle 来纠正
            adjusted_rotation = 2*np.pi - target_rotation
            
            # DEBUG: 打印旋转角度信息
            print(f"原始旋转角度: {np.rad2deg(target_rotation):.1f}°")
            print(f"调整后旋转角度: {np.rad2deg(adjusted_rotation):.1f}°")
            
            # 计算推动方向向量（使用调整后的角度）
            push_dir_x = np.cos(adjusted_rotation)
            push_dir_y = np.sin(adjusted_rotation)
            
            # 打印推动方向向量
            print(f"推动方向向量: ({push_dir_x:.3f}, {push_dir_y:.3f})")
            
            # 让推动方向与预测图一致，设置正确的起点和终点
            # 起点位置在目标后方
            self.push_start_position[:, 0] = self.target_position[:, 0] - self.push_step_size * push_dir_x
            self.push_start_position[:, 1] = self.target_position[:, 1] - self.push_step_size * push_dir_y
            self.push_start_position[:, 2] = self.target_position[:, 2] + self.approach_height
            
            # 终点位置在目标前方
            self.push_end_position[:, 0] = self.target_position[:, 0] + self.push_step_size * push_dir_x
            self.push_end_position[:, 1] = self.target_position[:, 1] + self.push_step_size * push_dir_y
            self.push_end_position[:, 2] = self.target_position[:, 2] + 0.001  # 稍微高于表面以避免碰撞(1mm)
            
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
                
                # 根据目标旋转角度计算四元数，适用于推动和抓取
                rotation_rad = self.target_rotation.item() if isinstance(self.target_rotation, torch.Tensor) else float(self.target_rotation)
                rotation_quat = self._euler_to_quaternion(0, 0, rotation_rad)
                self.des_ee_pose[0, 3:] = rotation_quat
                
                # 推动时提前关闭夹爪，抓取时保持打开
                if self.action_mode == ActionMode.PUSH:
                    self.gripper_state[0] = float(GripperState.CLOSE)
                else:
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
            # 抓取专用：APPROACH_OBJECT状态：机器人从安全高度下降到物体位置
            if self.wait_time[0] < self.wait_time_approach:
                # 计算接近物体的进度（0.0到1.0之间的值）
                approach_progress = min(1.0, float(self.wait_time[0] / self.wait_time_approach))
                
                # 基于进度计算位置：保持XY不变，Z轴随进度下降
                pos_x = float(self.target_position[0, 0])
                pos_y = float(self.target_position[0, 1])
                pos_z = float(self.target_position[0, 2]) + self.approach_height * (1.0 - approach_progress)
                
                # 设置所需的末端效应器姿态
                self.des_ee_pose[0, :3] = torch.tensor([pos_x, pos_y, pos_z], device=self.device)
                
                # 根据目标旋转角度计算四元数
                rotation_rad = self.target_rotation.item() if isinstance(self.target_rotation, torch.Tensor) else float(self.target_rotation)
                rotation_quat = self._euler_to_quaternion(0, 0, rotation_rad)
                self.des_ee_pose[0, 3:] = rotation_quat
                
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
                    # 推动阶段 - 确保高度稍微高于表面但足够低以有效推动
                    pos_z = float(self.push_end_position[0, 2])
                
                # 设置所需的末端效应器姿态
                self.des_ee_pose[0, :3] = torch.tensor([pos_x, pos_y, pos_z], device=self.device)
                
                # 使用目标旋转角度计算四元数
                rotation_rad = self.target_rotation.item() if isinstance(self.target_rotation, torch.Tensor) else float(self.target_rotation)
                rotation_quat = self._euler_to_quaternion(0, 0, rotation_rad)
                self.des_ee_pose[0, 3:] = rotation_quat
                
                # 推动时保持夹爪关闭
                self.gripper_state[0] = float(GripperState.CLOSE)
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
                
                # 使用目标旋转角度计算四元数
                rotation_rad = self.target_rotation.item() if isinstance(self.target_rotation, torch.Tensor) else float(self.target_rotation)
                rotation_quat = self._euler_to_quaternion(0, 0, rotation_rad)
                self.des_ee_pose[0, 3:] = rotation_quat
                
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
                
                # 使用目标旋转角度计算四元数，保持抓取时的旋转
                rotation_rad = self.target_rotation.item() if isinstance(self.target_rotation, torch.Tensor) else float(self.target_rotation)
                rotation_quat = self._euler_to_quaternion(0, 0, rotation_rad)
                self.des_ee_pose[0, 3:] = rotation_quat
                
                # 保持夹爪关闭以继续抓住物体
                self.gripper_state[0] = float(GripperState.CLOSE)
            else:
                # 修改状态转换：抓取成功后进入放置状态，而不是直接后退
                if self.action_mode == ActionMode.GRASP:
                    # 重置放置阶段
                    self.place_stage = 1
                    # 进入放置状态
                    self.state[0] = RobotState.PLACING
                else:
                    # 对于推动动作，保持原有逻辑，直接后退
                    self.state[0] = RobotState.RETREATING
                
                self.wait_time[0] = 0.0
                
        elif self.state[0] == RobotState.PLACING:
            # 新增：PLACING状态 - 将抓取的物体放置在指定位置
            if self.wait_time[0] < self.wait_time_place:
                # 确保place_position已初始化
                if self.place_position is None:
                    print("警告：放置位置未设置，使用默认位置")
                    # 使用默认放置位置（工作空间边缘）
                    place_x = self.workspace_limits[0][0] + 0.1  # 工作空间x方向边缘向内10cm
                    place_y = self.workspace_limits[1][0] + 0.1  # 工作空间y方向边缘向内10cm
                    place_z = self.workspace_limits[2][0]        # 工作空间底部
                    default_place_pos = [place_x, place_y, place_z]
                    self.place_position = torch.tensor(default_place_pos, device=self.device).reshape(1, 3).repeat(self.num_envs, 1)
                
                # 分三个阶段实现放置：
                # 1. 移动到放置位置上方
                # 2. 下降到放置高度
                # 3. 打开夹爪释放物体
                
                if self.place_stage == 1:
                    # 阶段1：移动到放置位置上方
                    pos_x = float(self.place_position[0, 0])
                    pos_y = float(self.place_position[0, 1])
                    pos_z = float(self.place_position[0, 2]) + self.approach_height  # 放置位置上方
                    
                    # 当位置接近目标位置时，进入下一阶段
                    if self.wait_time[0] > self.wait_time_place * 0.4:
                        self.place_stage = 2
                
                elif self.place_stage == 2:
                    # 阶段2：下降到放置高度
                    pos_x = float(self.place_position[0, 0])
                    pos_y = float(self.place_position[0, 1])
                    
                    # 计算下降进度
                    place_progress = min(1.0, float((self.wait_time[0] - self.wait_time_place * 0.4) / (self.wait_time_place * 0.3)))
                    pos_z = float(self.place_position[0, 2]) + self.approach_height * (1.0 - place_progress)
                    
                    # 当下降接近完成时，进入下一阶段
                    if self.wait_time[0] > self.wait_time_place * 0.7:
                        self.place_stage = 3
                
                else:  # self.place_stage == 3
                    # 阶段3：打开夹爪释放物体
                    pos_x = float(self.place_position[0, 0])
                    pos_y = float(self.place_position[0, 1])
                    pos_z = float(self.place_position[0, 2])  # 直接放在指定高度
                    
                    # 打开夹爪
                    self.gripper_state[0] = float(GripperState.OPEN)
                
                # 设置末端执行器位置
                self.des_ee_pose[0, :3] = torch.tensor([pos_x, pos_y, pos_z], device=self.device)
                
                # 使用默认向下姿态
                self.des_ee_pose[0, 3:] = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device)
                
                # 根据阶段控制夹爪状态
                if self.place_stage < 3:
                    # 前两个阶段保持夹爪关闭
                    self.gripper_state[0] = float(GripperState.CLOSE)
                else:
                    # 最后阶段打开夹爪
                    self.gripper_state[0] = float(GripperState.OPEN)
                
            else:
                # 放置完成后，进入后退状态
                self.state[0] = RobotState.RETREATING
                self.wait_time[0] = 0.0
                # 重置放置阶段
                self.place_stage = 0
        
        elif self.state[0] == RobotState.RETREATING:
            # RETREATING状态：推动操作后从推动位置后退到安全高度
            if self.wait_time[0] < self.wait_time_retreat:
                # 计算后退进度
                retreat_progress = min(1.0, float(self.wait_time[0] / self.wait_time_retreat))
                
                if self.action_mode == ActionMode.PUSH:
                    # 推动后退：从推动结束位置后退
                    pos_x = float(self.push_end_position[0, 0])
                    pos_y = float(self.push_end_position[0, 1])
                    pos_z = float(self.push_end_position[0, 2]) + self.approach_height * retreat_progress
                else:  # GRASP
                    # 抓取后退：从放置位置后退
                    if self.place_position is None:
                        # 如果放置位置未设置，使用目标位置
                        pos_x = float(self.target_position[0, 0])
                        pos_y = float(self.target_position[0, 1])
                        pos_z = float(self.target_position[0, 2]) + self.approach_height * retreat_progress
                        print("警告：放置位置未设置，从抓取位置后退")
                    else:
                        # 正常从放置位置后退
                        pos_x = float(self.place_position[0, 0])
                        pos_y = float(self.place_position[0, 1])
                        pos_z = float(self.place_position[0, 2]) + self.approach_height * retreat_progress
                
                # 设置所需的末端效应器姿态
                self.des_ee_pose[0, :3] = torch.tensor([pos_x, pos_y, pos_z], device=self.device)
                
                # 使用目标旋转角度计算四元数
                rotation_rad = self.target_rotation.item() if isinstance(self.target_rotation, torch.Tensor) else float(self.target_rotation)
                rotation_quat = self._euler_to_quaternion(0, 0, rotation_rad)
                self.des_ee_pose[0, 3:] = rotation_quat
                
                # 后退时保持夹爪状态
                if self.action_mode == ActionMode.PUSH:
                    # 推动后退时保持夹爪关闭
                    self.gripper_state[0] = float(GripperState.CLOSE)
                else:
                    # 抓取后退时已释放物体，保持夹爪打开
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
                
                # 回归到默认姿态（垂直向下），但保持Z轴旋转
                rotation_rad = self.target_rotation.item() if isinstance(self.target_rotation, torch.Tensor) else float(self.target_rotation)
                rotation_quat = self._euler_to_quaternion(0, 0, rotation_rad)
                self.des_ee_pose[0, 3:] = rotation_quat
                
                # 夹爪保持打开状态
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
    
    # 打印環境配置中的相機設置信息
    if args_cli.debug:
        print("\n==== 相機配置信息 ====")
        print("相機設置將在環境創建後確認")
        print("========================\n")
    
    # Create environment
    env = gym.make("Isaac-Lift-Cube-UR5e-IK-Abs-v0", cfg=env_cfg)
    
    # Reset environment at start
    obs_dict = env.reset()
    
    # 安全地訪問unwrapped屬性
    env_unwrapped = env.unwrapped
    
    # Get device and action shape from environment
    device = getattr(env_unwrapped, "device", "cpu")  # 默認爲CPU如果沒有此屬性
    action_shape = env_unwrapped.action_space.shape
    
    # Create action buffers (position + quaternion + gripper)
    if action_shape is None:
        # 如果action_shape为None，使用默认形状 - [num_envs, 8]（位置3 + 四元数4 + 夹爪1）
        action_dim = 8
        actions = torch.zeros((args_cli.num_envs, action_dim), device=device)
    else:
        # 正常初始化actions
        actions = torch.zeros(action_shape, device=device)
    
    actions[:, 3] = 1.0  # quaternion w component
    
    # 安全訪問num_envs（或使用args_cli中的設置）
    num_envs = getattr(env_unwrapped, "num_envs", args_cli.num_envs)
    
    # Create controller
    controller = PushingGraspingController(
        env_cfg.sim.dt * env_cfg.decimation,
        num_envs,
        device,
        workspace_limits,
        heightmap_resolution
    )
    
    # 设置抓取后物体的放置位置 - 对所有环境使用相同的放置位置
    place_x = 0.2  # 工作空间左侧
    place_y = -0.5  # 工作空间前方
    place_z = workspace_limits[2][0] + 0.01  # 工作空间底部略高
    controller.set_place_position([place_x, place_y, place_z])
    print(f"已设置物体放置位置: [{place_x:.4f}, {place_y:.4f}, {place_z:.4f}]")
    
    # 初始化實驗跟蹤器
    experiment = Experiment()
    
    # 初始化訓練器和日誌記錄器
    if not args_cli.rule_based:
        # 創建日誌目錄
        if not os.path.exists(args_cli.logging_directory):
            os.makedirs(args_cli.logging_directory)
            
        # 初始化日誌記錄器
        logger = Logger(args_cli.continue_logging, args_cli.logging_directory)
        
        # 输出详细的GPU使用信息
        print("\n=================================================")
        print(f"GPU强制使用CPU: {args_cli.force_cpu}")
        print(f"CUDA是否可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available() and not args_cli.force_cpu:
            print(f"当前将使用GPU: {torch.cuda.get_device_name(0)}")
            # 显示GPU内存使用情况
            print(f"GPU总内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            print(f"当前已分配内存: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
            print(f"当前缓存内存: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        else:
            print("当前将使用CPU进行计算")
        print("=================================================\n")
        
        # 初始化訓練器
        trainer = Trainer(
            method=args_cli.method,
            push_rewards=True,
            future_reward_discount=0.5,
            is_testing=args_cli.is_testing,
            load_snapshot=args_cli.load_snapshot,
            snapshot_file=args_cli.snapshot_file,
            force_cpu=args_cli.force_cpu
        )
        
        # 如果繼續上一次的訓練，加載之前的轉換記錄
        if args_cli.continue_logging:
            trainer.preload(logger.transitions_directory)
            print(f"已從 {logger.transitions_directory} 加載訓練記錄")
            
        print(f"使用 {args_cli.method} 模型進行{'測試' if args_cli.is_testing else '訓練'}")
    else:
        # 使用規則式策略
                policy = RuleBasedPolicy(workspace_limits, heightmap_resolution)
                print("使用規則式策略")
    
    # 輸出相機ID說明
    print(f"使用相機ID: {args_cli.camera_id} ({'頂部相機' if args_cli.camera_id == 0 else '側面相機'})")
    
    # 安全地訪問scene屬性
    scene = getattr(env_unwrapped, "scene", None)
    if scene is None:
        print("警告：無法訪問場景對象！可能導致功能受限")
        simulation_app.close()
        return
    
    # Camera data - 確保使用正確的相機名稱
    camera_names = ["top_camera", "side_camera"]
    camera_name = camera_names[args_cli.camera_id]
    camera = scene[camera_name]
    
    # 多环境设置 - 确保为每个环境正确获取相机索引
    # 创建环境ID数组用于后续处理
    env_ids = np.arange(num_envs)
    print(f"环境数量: {num_envs}")
    
    if args_cli.debug:
        print(f"選擇的相機: {camera_name}")
        print(f"可用的場景物體: {list(scene.keys())}")
    
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
    
    # 保存相機參數（针对所有环境）
    if not args_cli.rule_based:
        # 对每个环境单独获取和处理相机数据
        for env_id in env_ids:
            # 獲取相機內參
            camera_intrinsics = camera.data.intrinsic_matrices[env_id].cpu().numpy()
            
            # 構建相機位姿矩陣（4x4）
            camera_pos = camera.data.pos_w[env_id].cpu().numpy()
            camera_quat = camera.data.quat_w_world[env_id].cpu().numpy()
            
            # 從四元數創建旋轉矩陣
            from scipy.spatial.transform import Rotation as R
            rot_matrix = R.from_quat([camera_quat[1], camera_quat[2], camera_quat[3], camera_quat[0]]).as_matrix()
            
            # 構建4x4位姿矩陣
            camera_pose = np.eye(4)
            camera_pose[:3, :3] = rot_matrix
            camera_pose[:3, 3] = camera_pos
            
            # 保存相機參數到日誌（仅针对环境0，避免重复记录）
            if env_id == 0:
                logger.save_camera_info(camera_intrinsics, camera_pose, 1.0)
                logger.save_heightmap_info(workspace_limits, heightmap_resolution)
                
                if args_cli.debug:
                    print(f"\n环境 {env_id} 相机位置: {camera_pos}")
                    print(f"环境 {env_id} 相机四元数: {camera_quat}")
                    print(f"环境 {env_id} 相机内参矩阵:\n{camera_intrinsics}")
    
    # 主循環迭代計數
    iteration = 0
    
    # 場景變化檢測相關變量
    prev_depth_heightmap = None
    
    # Main simulation loop
    while simulation_app.is_running():
        # Run everything in inference mode
        with torch.inference_mode():
            # Step environment
            obs_dict, reward, terminated, truncated, info = env.step(actions)
            dones = terminated
            
            # 安全獲取場景中的物體
            try:
                # Get observations
                # -- end-effector frame
                ee_frame_sensor = scene["ee_frame"]
                env_origins = scene.env_origins
                tcp_position = ee_frame_sensor.data.target_pos_w[..., 0, :].clone() - env_origins
                tcp_orientation = ee_frame_sensor.data.target_quat_w[..., 0, :].clone()
                ee_pose = torch.cat([tcp_position, tcp_orientation], dim=-1)
                
                # -- object frame (for visualization and debugging)
                object_data = scene["object1"].data
                object_position = object_data.root_pos_w - env_origins
            except (KeyError, AttributeError) as e:
                print(f"警告：無法訪問場景對象：{e}")
                # 使用默認值
                ee_pose = torch.zeros((num_envs, 7), device=device)
                ee_pose[:, 3] = 1.0  # quaternion w
            
            # 如果機器人當前不在執行動作，則進行感知和決策
            if not executing_action:
                try:
                    # 当前选中的环境ID（在多环境设置中，我们可以选择处理哪个环境）
                    # 在这里我们使用环境0进行演示，实际应用中可能需要处理所有环境
                    active_env_id = 0
                    
                    # Get camera data for the active environment
                    single_cam_data = convert_dict_to_backend(camera.data.output[active_env_id], backend="numpy")
                    color_img = single_cam_data["rgb"]
                    depth_img = single_cam_data["distance_to_camera"][:, :, 0]
                    
                    # 打印相機數據信息
                    if args_cli.debug:
                        print(f"\n==== 环境 {active_env_id} {camera_name} 相機數據 ====")
                        print(f"色彩圖像尺寸: {color_img.shape}")
                        print(f"深度圖像尺寸: {depth_img.shape}")
                        print(f"深度範圍: {np.nanmin(depth_img):.3f} 到 {np.nanmax(depth_img):.3f} 米")
                    if np.isnan(depth_img).any():
                        print(f"注意: 环境 {active_env_id} 深度圖包含 {np.sum(np.isnan(depth_img))} 個NaN值")
                
                    # Get camera parameters for the active environment
                    camera_intrinsics = camera.data.intrinsic_matrices[active_env_id].cpu().numpy()
                    
                    # 構建相機位姿矩陣（4x4）- 针对选中的环境
                    camera_pos = camera.data.pos_w[active_env_id].cpu().numpy()
                    camera_quat = camera.data.quat_w_world[active_env_id].cpu().numpy()
                    
                    # 從四元數創建旋轉矩陣
                    from scipy.spatial.transform import Rotation as R
                    rot_matrix = R.from_quat([camera_quat[1], camera_quat[2], camera_quat[3], camera_quat[0]]).as_matrix()
                    
                    # 構建4x4位姿矩陣
                    camera_pose = np.eye(4)
                    camera_pose[:3, :3] = rot_matrix
                    
                    camera_pose[:3, 3] = camera_pos
                    
                    # 打印相機位姿信息 - 仅针对活动环境
                    if args_cli.debug:
                        print(f"\n环境 {active_env_id} 相機位置: {camera_pos}")
                        print(f"环境 {active_env_id} 相機四元數: {camera_quat}")
                        print(f"环境 {active_env_id} 相機內參矩陣:\n{camera_intrinsics}")
                    
                    # 當前時間戳（用於保存文件名）
                    timestamp = int(time.time())
                    
                    # Generate heightmaps for active environment
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
                    
                    # 处理高度图，确保有有效数据用于训练和测试
                    color_heightmap, depth_heightmap = process_heightmaps(color_heightmap, depth_heightmap, workspace_limits)
                    
                except Exception as e:
                    print(f"錯誤：獲取相機數據失敗：{e}")
                    simulation_app.close()
                    return
                
                # 保存當前的RGB和深度圖以及高度圖
                if not args_cli.rule_based:
                    logger.save_images(experiment.iteration, color_img, depth_img, 'init')
                    logger.save_heightmaps(experiment.iteration, color_heightmap, depth_heightmap, 'init')
                
                # 打印高度图统计信息
                background_pixels = np.sum(depth_heightmap == 0)
                total_pixels = depth_heightmap.size
                object_pixels = np.sum(depth_heightmap > 0)
                print(f"\n===== 高度图统计信息 =====")
                print(f"高度图大小: {depth_heightmap.shape}, 总像素数: {total_pixels}")
                print(f"背景像素数(值=0): {background_pixels}, 占比: {background_pixels/total_pixels*100:.2f}%")
                print(f"物体像素数(值>0): {object_pixels}, 占比: {object_pixels/total_pixels*100:.2f}%")
                print(f"深度范围: {np.min(depth_heightmap):.6f} 到 {np.max(depth_heightmap):.6f}")
                if np.isnan(depth_heightmap).any():
                    print(f"警告: 高度图包含 {np.sum(np.isnan(depth_heightmap))} 个NaN值")
                print("===========================\n")
                
                # 檢測場景變化（如果不是第一次迭代）
                if prev_depth_heightmap is not None:
                    # 獲取場景中的物體
                    try:
                        # 获取物体位置信息
                        cube = scene["object1"]
                        cube_pos = cube.data.root_pos_w[active_env_id] - scene.env_origins[active_env_id]
                        
                        # 使用物体位置与之前位置比较
                        if experiment.prev_object_position is not None:
                            # 计算物体移动距离
                            position_diff = torch.norm(cube_pos - experiment.prev_object_position).item()
                            position_threshold = 0.005  # 5mm移动阈值
                            
                            # 判断是否有显著移动
                            change_detected = position_diff > position_threshold
                            experiment.change_detected = change_detected
                            
                            print(f"物体移动距离: {position_diff*1000:.2f}mm, 场景变化检测: {change_detected}")
                            
                            if change_detected:
                                print("檢測到場景變化!")
                                
                                # 如果上一次是推動動作，標記為推動成功
                                if experiment.primitive_action == ActionMode.PUSH:
                                    experiment.push_success = True
                                    print("推動成功！")
                        
                        # 保存当前物体位置用于下次比较
                        experiment.prev_object_position = cube_pos.clone()
                    except (KeyError, AttributeError) as e:
                        print(f"警告: 无法获取物体位置: {e}")
                        # 回退到使用深度图比较
                        depth_diff = np.abs(depth_heightmap - prev_depth_heightmap)
                        change_detected = np.sum(depth_diff > DEPTH_DIFF_THRESH) > MIN_PIXEL_DIFF_THRESH
                        experiment.change_detected = change_detected
                        
                        if change_detected:
                            print("使用深度图检测到场景变化")
                            
                            # 如果上一次是推动动作，标记为推动成功
                            if experiment.primitive_action == ActionMode.PUSH:
                                experiment.push_success = True
                                print("推动成功！")
                
                # 保存當前深度圖，用於下一次比較（作为备用）
                prev_depth_heightmap = depth_heightmap.copy()
                
                # 保存當前圖像和高度圖到實驗跟蹤器
                experiment.prev_color_img = color_img.copy()
                experiment.prev_depth_img = depth_img.copy()
                experiment.prev_color_heightmap = color_heightmap.copy()
                experiment.prev_depth_heightmap = depth_heightmap.copy()
                
                # 判斷上一次抓取是否成功（通過檢查物體高度變化）
                if experiment.primitive_action == ActionMode.GRASP and not experiment.grasp_success:
                    # 目前的簡化判斷: 如果工作區內的點明顯減少，則認為抓取成功
                    # 獲取有效的深度點數量
                    curr_valid_depth_pixels = np.sum(depth_heightmap > 0.005) # 5毫米以上的深度點
                    prev_valid_depth_pixels = np.sum(prev_depth_heightmap > 0.005)
                    
                    # 如果深度點減少超過一定比例，則認為抓取成功
                    if curr_valid_depth_pixels < prev_valid_depth_pixels * 0.8:
                        experiment.grasp_success = True
                        print("抓取成功！")
                        
                # 更新成功/失败状态
                if experiment.iteration > 0:
                    current_success = experiment.push_success if experiment.primitive_action == ActionMode.PUSH else experiment.grasp_success
                    experiment.update_failure_count(current_success)
                    
                    if not current_success:
                        print(f"动作失败！连续失败次数: {experiment.consecutive_failures}")
                
                # 決策：使用深度學習模型或基於規則的策略
                if not args_cli.rule_based:
                    # 使用模型做推理
                    push_predictions, grasp_predictions, state_feat = trainer.forward(
                        color_heightmap, 
                        depth_heightmap, 
                        is_volatile=True
                    )
                    
                    experiment.set_predictions(push_predictions, grasp_predictions)
                    
                    # 可視化預測結果
                    push_pred_vis = get_affordance_vis(push_predictions)
                    grasp_pred_vis = get_affordance_vis(grasp_predictions)
                    logger.save_visualizations(experiment.iteration, push_pred_vis, 'push')
                    logger.save_visualizations(experiment.iteration, grasp_pred_vis, 'grasp')
                    
                    # 如果上一個動作已經完成，進行反向傳播（如果不是測試模式）
                    if not args_cli.is_testing and experiment.primitive_action is not None:
                        # 計算標籤值（reward）- 直接使用整数作为action_type
                        label_value, current_reward = trainer.get_label_value(
                            experiment.primitive_action,  # 这已经是整数0或1
                            experiment.push_success,
                            experiment.grasp_success,
                            experiment.change_detected,
                            experiment.push_predictions,
                            experiment.grasp_predictions,
                            color_heightmap,
                            depth_heightmap
                        )
                        
                        # 更加突出地显示奖励信息
                        print("\n" + "="*50)
                        print(f"★★★ 奖励信息 ★★★")
                        print(f"动作类型: {'推动' if experiment.primitive_action == 0 else '抓取'}")
                        print(f"当前奖励: {current_reward:.6f}")
                        print(f"标签值(未来奖励): {label_value:.6f}")
                        print(f"推动成功: {experiment.push_success}")
                        print(f"抓取成功: {experiment.grasp_success}")
                        print(f"场景变化: {experiment.change_detected}")
                        print("="*50 + "\n")
                        
                        # 進行反向傳播
                        trainer.backprop(
                            experiment.prev_color_heightmap,
                            experiment.prev_depth_heightmap,
                            experiment.primitive_action,
                            experiment.best_pix_ind,
                            label_value
                        )
                        
                        # 經驗回放（如果啟用）
                        if args_cli.experience_replay:
                            trainer.experience_replay()
                            
                        # 每100次迭代保存模型
                        if experiment.iteration % 100 == 0:
                            logger.save_model(experiment.iteration, trainer.model, 'push-grasp')
                            logger.save_backup_model(trainer.model, 'push-grasp')
                            
                            # 保存訓練日誌
                            logger.write_to_log('executed-action', np.array(trainer.executed_action_log))
                            logger.write_to_log('label-value', np.array(trainer.label_value_log))
                            logger.write_to_log('reward-value', np.array(trainer.reward_value_log))
                            logger.write_to_log('predicted-value', np.array(trainer.predicted_value_log))
                            logger.write_to_log('use-heuristic', np.array(trainer.use_heuristic_log))
                            logger.write_to_log('is-exploit', np.array(trainer.is_exploit_log))
                    
                    # 计算探索提升率，如果连续失败则增加探索率
                    exploration_boost = experiment.get_exploration_boost()
                    
                    # 根据连续失败提升的探索率决定是否随机动作
                    if np.random.uniform() < exploration_boost:
                        print(f"启用探索模式（探索率提升：{exploration_boost:.2f}）")
                        
                        # 增强：优先选择未探索的区域
                        unexplored_regions = [i for i in range(experiment.region_grid_size**2) 
                                             if i not in experiment.explored_regions]
                        
                        # 如果有未探索区域，优先选择
                        if unexplored_regions and np.random.uniform() < 0.7:
                            # 随机选择一个未探索区域
                            target_region = np.random.choice(unexplored_regions)
                            
                            # 计算该区域的坐标范围
                            region_y = target_region // experiment.region_grid_size
                            region_x = target_region % experiment.region_grid_size
                            
                            h, w = depth_heightmap.shape
                            region_h = h // experiment.region_grid_size
                            region_w = w // experiment.region_grid_size
                            
                            y_min = region_y * region_h
                            y_max = min(h, (region_y + 1) * region_h)
                            x_min = region_x * region_w
                            x_max = min(w, (region_x + 1) * region_w)
                            
                            print(f"目标探索区域: {target_region}，坐标范围: ({y_min}-{y_max}, {x_min}-{x_max})")
                            
                            # 随机选择一个点和旋转
                            rand_idx = np.random.randint(0, args_cli.num_rotations)
                            rand_y = np.random.randint(y_min, y_max)
                            rand_x = np.random.randint(x_min, x_max)
                            
                            # 随机选择推动或抓取
                            if np.random.uniform() < 0.5:
                                experiment.set_primitive_action(ActionMode.PUSH)
                            else:
                                experiment.set_primitive_action(ActionMode.GRASP)
                            
                            experiment.set_best_pix_ind((rand_idx, rand_y, rand_x))
                            experiment.set_rotation_angle(np.deg2rad(rand_idx * (360.0 / args_cli.num_rotations)))
                            experiment.set_coords((rand_y, rand_x))
                            experiment.record_explored_region((rand_y, rand_x), depth_heightmap.shape)
                            
                            print(f"选择探索新区域 {target_region} 的{'推动' if experiment.primitive_action == ActionMode.PUSH else '抓取'}动作，位置：{experiment.best_pix_ind_coords}，旋转：{np.rad2deg(experiment.best_rotation_angle):.1f}度")
                        else:
                            # 原有随机策略
                            if np.random.uniform() < 0.5:
                                action_mode = ActionMode.PUSH
                                action_space = push_predictions.shape
                                # 随机选择一个点
                                rand_idx = np.random.randint(0, action_space[0])
                                rand_y = np.random.randint(0, action_space[1])
                                rand_x = np.random.randint(0, action_space[2])
                                experiment.set_primitive_action(ActionMode.PUSH)
                                experiment.set_best_pix_ind((rand_idx, rand_y, rand_x))
                                experiment.set_rotation_angle(np.deg2rad(rand_idx * (360.0 / args_cli.num_rotations)))
                                experiment.set_coords((rand_y, rand_x))
                                print(f"随机选择推动动作，位置：{experiment.best_pix_ind_coords}，旋转：{np.rad2deg(experiment.best_rotation_angle):.1f}度")
                            else:
                                action_mode = ActionMode.GRASP
                                action_space = grasp_predictions.shape
                                # 随机选择一个点
                                rand_idx = np.random.randint(0, action_space[0])
                                rand_y = np.random.randint(0, action_space[1])
                                rand_x = np.random.randint(0, action_space[2])
                                experiment.set_primitive_action(ActionMode.GRASP)
                                experiment.set_best_pix_ind((rand_idx, rand_y, rand_x))
                                experiment.set_rotation_angle(np.deg2rad(rand_idx * (360.0 / args_cli.num_rotations)))
                                experiment.set_coords((rand_y, rand_x))
                                print(f"随机选择抓取动作，位置：{experiment.best_pix_ind_coords}，旋转：{np.rad2deg(experiment.best_rotation_angle):.1f}度")
                    else:
                        # 正常决策逻辑
                        # 決定下一個動作（推動或抓取）
                        max_push_score = np.max(push_predictions)
                        max_grasp_score = np.max(grasp_predictions)
                            
                        # 获取最佳推动和抓取点
                        push_best_pix_ind = np.unravel_index(np.argmax(push_predictions), push_predictions.shape)
                        grasp_best_pix_ind = np.unravel_index(np.argmax(grasp_predictions), grasp_predictions.shape)
                        
                        # 简单检查：确保选择的点不在完全空白区域
                        push_y, push_x = push_best_pix_ind[1], push_best_pix_ind[2]
                        grasp_y, grasp_x = grasp_best_pix_ind[1], grasp_best_pix_ind[2]
                        
                        # 在选择动作点周围3x3区域检查是否有物体
                        def has_object_nearby(y, x, radius=2):
                            y_min = max(0, y - radius)
                            y_max = min(depth_heightmap.shape[0], y + radius + 1)
                            x_min = max(0, x - radius)
                            x_max = min(depth_heightmap.shape[1], x + radius + 1)
                            region = depth_heightmap[y_min:y_max, x_min:x_max]
                            return np.any(region > 0.001)  # 检查区域是否有物体
                        
                        # 如果推动点附近没有物体，打印警告
                        if not has_object_nearby(push_y, push_x):
                            print("警告：最佳推动点周围没有检测到物体，这可能导致无效动作")
                        
                        # 如果抓取点附近没有物体，打印警告
                        if not has_object_nearby(grasp_y, grasp_x):
                            print("警告：最佳抓取点周围没有检测到物体，这可能导致无效动作")
                        
                        # 检查最佳动作是否为重复动作
                        push_pixel_coords = (push_best_pix_ind[1], push_best_pix_ind[2])
                        grasp_pixel_coords = (grasp_best_pix_ind[1], grasp_best_pix_ind[2])
                        push_is_repeated = experiment.is_action_repeated(ActionMode.PUSH, push_pixel_coords)
                        grasp_is_repeated = experiment.is_action_repeated(ActionMode.GRASP, grasp_pixel_coords)
                        
                        # 如果两种动作都是重复动作，打印警告并尝试使用次优选择
                        if push_is_repeated and grasp_is_repeated and experiment.consecutive_failures > 1:
                            print("警告：最佳推动和抓取点都是重复的无效动作位置！")
                            # 尝试找到非重复的次优选择
                            temp_push_predictions = push_predictions.copy()
                            temp_grasp_predictions = grasp_predictions.copy()
                            
                            # 遮罩掉最佳点附近区域
                            mask_radius = experiment.location_similarity_threshold
                            y_indices, x_indices = np.indices(push_predictions.shape[1:])
                            
                            # 遮罩推动点附近区域
                            for rotation in range(push_predictions.shape[0]):
                                dist = np.sqrt((y_indices - push_best_pix_ind[1])**2 + (x_indices - push_best_pix_ind[2])**2)
                                temp_push_predictions[rotation][dist < mask_radius] = -np.inf
                            
                            # 遮罩抓取点附近区域
                            for rotation in range(grasp_predictions.shape[0]):
                                dist = np.sqrt((y_indices - grasp_best_pix_ind[1])**2 + (x_indices - grasp_best_pix_ind[2])**2)
                                temp_grasp_predictions[rotation][dist < mask_radius] = -np.inf
                            
                            # 重新找最佳点
                            if np.max(temp_push_predictions) != -np.inf:
                                push_best_pix_ind = np.unravel_index(np.argmax(temp_push_predictions), temp_push_predictions.shape)
                                max_push_score = temp_push_predictions[push_best_pix_ind]
                                push_is_repeated = False
                                print("已找到非重复推动位置")
                            
                            if np.max(temp_grasp_predictions) != -np.inf:
                                grasp_best_pix_ind = np.unravel_index(np.argmax(temp_grasp_predictions), temp_grasp_predictions.shape)
                                max_grasp_score = temp_grasp_predictions[grasp_best_pix_ind]
                                grasp_is_repeated = False
                                print("已找到非重复抓取位置")
                        
                        # 选择动作（考虑是否为重复动作）
                        if max_push_score > max_grasp_score and not push_is_repeated:
                            # 選擇推動
                            experiment.set_primitive_action(ActionMode.PUSH)
                            experiment.set_best_pix_ind(push_best_pix_ind)
                            # 最佳旋轉角度（基於離散化的旋轉）
                            experiment.set_rotation_angle(np.deg2rad(push_best_pix_ind[0] * (360.0 / args_cli.num_rotations)))
                            # 獲取像素xy坐標（與旋轉無關）
                            experiment.set_coords((push_best_pix_ind[1], push_best_pix_ind[2]))
                            print(f"選擇推動動作，位置：{experiment.best_pix_ind_coords}，旋轉：{np.rad2deg(experiment.best_rotation_angle):.1f}度，分數：{max_push_score:.3f}")
                        elif not grasp_is_repeated:
                            # 選擇抓取
                            experiment.set_primitive_action(ActionMode.GRASP)
                            experiment.set_best_pix_ind(grasp_best_pix_ind)
                            # 最佳旋轉角度（基於離散化的旋轉）
                            experiment.set_rotation_angle(np.deg2rad(grasp_best_pix_ind[0] * (360.0 / args_cli.num_rotations)))
                            # 獲取像素xy坐標（與旋轉無關）
                            experiment.set_coords((grasp_best_pix_ind[1], grasp_best_pix_ind[2]))
                            print(f"選擇抓取動作，位置：{experiment.best_pix_ind_coords}，旋轉：{np.rad2deg(experiment.best_rotation_angle):.1f}度，分數：{max_grasp_score:.3f}")
                        elif max_push_score > max_grasp_score:
                            # 不得不使用重复的推动动作
                            experiment.set_primitive_action(ActionMode.PUSH)
                            experiment.set_best_pix_ind(push_best_pix_ind)
                            experiment.set_rotation_angle(np.deg2rad(push_best_pix_ind[0] * (360.0 / args_cli.num_rotations)))
                            experiment.set_coords((push_best_pix_ind[1], push_best_pix_ind[2]))
                            print(f"选择重复推动动作（无更好选择），位置：{experiment.best_pix_ind_coords}，分数：{max_push_score:.3f}")
                        else:
                            # 不得不使用重复的抓取动作
                            experiment.set_primitive_action(ActionMode.GRASP)
                            experiment.set_best_pix_ind(grasp_best_pix_ind)
                            experiment.set_rotation_angle(np.deg2rad(grasp_best_pix_ind[0] * (360.0 / args_cli.num_rotations)))
                            experiment.set_coords((grasp_best_pix_ind[1], grasp_best_pix_ind[2]))
                            print(f"选择重复抓取动作（无更好选择），位置：{experiment.best_pix_ind_coords}，分数：{max_grasp_score:.3f}")
                else:
                    # 使用基於規則的策略
                    action_mode, best_pix_ind, rotation_angle = policy.get_action(color_heightmap, depth_heightmap)
                    experiment.set_primitive_action(action_mode)
                    experiment.set_best_pix_ind((0, best_pix_ind[0], best_pix_ind[1]))
                    experiment.set_rotation_angle(rotation_angle)
                    experiment.set_coords((best_pix_ind[1], best_pix_ind[0]))  # x, y順序
                
                # 可視化選擇的動作
                action_vis = get_action_visualization(
                    color_heightmap, 
                    depth_heightmap, 
                    experiment.primitive_action, 
                    experiment.best_pix_ind_coords[0], 
                    experiment.best_pix_ind_coords[1], 
                    experiment.best_rotation_angle,
                    scale_factor=1,
                    flip_x=True,  # 与坐标转换一致，X轴方向翻转
                    flip_y=True   # 与坐标转换一致，Y轴方向翻转
                )
                
                if not args_cli.rule_based:
                    # 添加保存预测可视化图像
                    # 创建推动和抓取预测的可视化
                    push_pred_vis = trainer.get_prediction_vis(push_predictions, color_heightmap, experiment.best_pix_ind)
                    grasp_pred_vis = trainer.get_prediction_vis(grasp_predictions, color_heightmap, experiment.best_pix_ind)
                    
                    # 保存预测可视化
                    logger.save_visualizations(experiment.iteration, push_pred_vis, 'push_prediction')
                    logger.save_visualizations(experiment.iteration, grasp_pred_vis, 'grasp_prediction')
                    
                    # 保存动作可视化
                    logger.save_visualizations(experiment.iteration, action_vis, 'action')
                elif args_cli.debug:
                    # 保存可視化結果
                    os.makedirs(os.path.join("output", "VPG", "viz"), exist_ok=True)
                    cv2.imwrite(os.path.join("output", "VPG", "viz", f"action_{timestamp}.png"), action_vis)
                    
                    # 如果是debug模式，在规则模式下也保存预测可视化
                    if hasattr(trainer, 'get_prediction_vis'):
                        try:
                            push_pred_vis = trainer.get_prediction_vis(push_predictions, color_heightmap, experiment.best_pix_ind)
                            grasp_pred_vis = trainer.get_prediction_vis(grasp_predictions, color_heightmap, experiment.best_pix_ind)
                            cv2.imwrite(os.path.join("output", "VPG", "viz", f"push_pred_{timestamp}.png"), push_pred_vis)
                            cv2.imwrite(os.path.join("output", "VPG", "viz", f"grasp_pred_{timestamp}.png"), grasp_pred_vis)
                        except Exception as e:
                            print(f"无法生成预测可视化: {e}")
                
                # 將像素坐標轉換為3D世界坐標
                pixel_x = experiment.best_pix_ind_coords[0]
                pixel_y = experiment.best_pix_ind_coords[1]
                
                # 計算3D位置（考虑X轴翻转，不考虑Y轴翻转）
                position_x = (depth_heightmap.shape[1] - 1 - pixel_x) * heightmap_resolution + workspace_limits[0][0]
                position_y = pixel_y * heightmap_resolution + workspace_limits[1][0]
                
                # 從深度高度圖獲取z位置
                if np.isnan(depth_heightmap[pixel_y, pixel_x]) or depth_heightmap[pixel_y, pixel_x] <= 0.0001:
                    position_z = workspace_limits[2][0] + 0.003  # 如果深度無效，使用工作空間的最小高度加上偏移(改為3mm)
                    print(f"使用默认Z高度: {position_z:.4f}，原始深度值: {depth_heightmap[pixel_y, pixel_x]}")
                else:
                    position_z = depth_heightmap[pixel_y, pixel_x] + workspace_limits[2][0]
                    print(f"使用计算Z高度: {position_z:.4f}，来自深度值: {depth_heightmap[pixel_y, pixel_x]}")
                
                target_position = np.array([position_x, position_y, position_z])
                
                # 记录动作到历史记录，用于检测重复动作
                experiment.add_action_to_memory(experiment.primitive_action, experiment.best_pix_ind_coords)
                
                # 設置機器人動作
                controller.set_action(experiment.primitive_action, target_position, experiment.best_rotation_angle)
                executing_action = True
                print(f"執行{'推動' if experiment.primitive_action == ActionMode.PUSH else '抓取'}動作，位置：{target_position}")
                
                # 遞增迭代計數
                experiment.iteration += 1
                iteration += 1
            
            # Compute actions from controller
            actions = controller.compute(ee_pose, env_cfg.sim.dt * env_cfg.decimation)
            
            # Check if action is complete
            if controller.state[0] == RobotState.IDLE and executing_action:
                executing_action = False
                print("動作執行完成")
            
            # Save camera data if requested
            if args_cli.save_camera_data:
                # Get camera data
                single_cam_data = convert_dict_to_backend(camera.data.output[active_env_id], backend="numpy")
                single_cam_info = camera.data.info[active_env_id]
                
                # Prepare camera data for saving
                timestamp = int(time.time())
                camera_data = {
                    "rgb": single_cam_data["rgb"],
                    "depth": single_cam_data["distance_to_camera"][:, :, 0],
                    "intrinsics": camera_intrinsics,
                    "pose": camera_pose,
                    "timestamp": timestamp
                }
                
                # 保存RGB图像
                rgb_path = os.path.join(output_dir, "rgb", f"rgb_{timestamp}.png")
                os.makedirs(os.path.dirname(rgb_path), exist_ok=True)
                cv2.imwrite(rgb_path, 
                            cv2.cvtColor(camera_data["rgb"], cv2.COLOR_RGB2BGR))
                
                # 保存深度图像
                depth_path = os.path.join(output_dir, "depth", f"depth_{timestamp}.png")
                os.makedirs(os.path.dirname(depth_path), exist_ok=True)
                depth_for_vis = camera_data["depth"].copy()
                
                # 处理NaN值
                depth_for_vis[np.isnan(depth_for_vis)] = 0
                
                # 归一化深度图以便可视化
                depth_min = np.min(depth_for_vis)
                depth_max = np.max(depth_for_vis)
                depth_range = depth_max - depth_min
                
                if depth_range > 0:
                    depth_normalized = (depth_for_vis - depth_min) / depth_range * 255
                    depth_normalized[np.isnan(depth_for_vis)] = 0
                    cv2.imwrite(depth_path, depth_normalized.astype(np.uint8))
            
            # Reset controller if environment is done
            if isinstance(dones, bool):
                dones_check = [dones]
            else:
                dones_check = dones
            if any(dones_check):
                # Find which environments are done
                done_env_ids = torch.nonzero(torch.tensor(dones, device=device), as_tuple=False).squeeze(-1)
                controller.reset_idx(done_env_ids)
    
    # 保存最終模型
    if not args_cli.rule_based and not args_cli.is_testing:
        logger.save_model(experiment.iteration, trainer.model, 'push-grasp-final')
        print(f"保存最終模型: iteration {experiment.iteration}")
    
    # Close the environment
    env.close()

if __name__ == "__main__":
    # Run the main function
    main()
    # Close sim app
    simulation_app.close() 