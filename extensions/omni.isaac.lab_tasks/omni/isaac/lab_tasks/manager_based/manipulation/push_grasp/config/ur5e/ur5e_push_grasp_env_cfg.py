# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.assets import RigidObjectCfg
from omni.isaac.lab.sensors import FrameTransformerCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
import omni.isaac.lab.sim as sim_utils
import numpy as np
import torch

from omni.isaac.lab_tasks.manager_based.manipulation.push_grasp import mdp
from omni.isaac.lab_tasks.manager_based.manipulation.push_grasp.push_grasp_env_cfg import PushGraspEnvCfg

##
# Pre-defined configs
##
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG  # isort: skip
from omni.isaac.lab_assets import UR5E_CFG_v2
from omni.isaac.lab.sensors import CameraCfg

@configclass
class UR5ePushGraspEnvCfg(PushGraspEnvCfg):
    """基于UR5e机器人的视觉推抓环境配置。"""
    
    def __post_init__(self):
        # 初始化父类
        super().__post_init__()
        
        # 设置UR5e为机器人
        self.scene.robot = UR5E_CFG_v2.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # 配置推动和抓取动作
        # 推动动作设置
        self.actions.push_action = mdp.PushActionCfg(
            asset_name="robot",
            ee_body_name="tool0",
            max_push_distance=0.15,  # 最大推动距离（米）
            push_height_offset=0.01,  # 推动高度偏移（距离物体表面）
            z_offset_before_pushing=0.05,  # 开始推动前的高度偏移
            push_speed=0.1,  # 推动速度（米/秒）
        )
        
        # 抓取动作设置（适用于Robotiq 85夹爪）
        self.actions.grasp_action = mdp.GraspActionCfg(
            asset_name="robot",
            ee_body_name="tool0",
            gripper_joint_names=["robotiq_85_left_knuckle_joint", "robotiq_85_right_knuckle_joint"],
            pre_grasp_height=0.1,  # 抓取前的高度（米）
            grasp_height=0.02,     # 抓取高度（米，相对于物体）
            post_grasp_height=0.2,  # 抓取后提升高度（米）
        )
        
        # 动作选择器
        self.actions.action_selector = mdp.ActionSelectorCfg(
            action_space_dim=8,  # 动作维度：1(选择器) + 3(位置x,y,z) + 1(角度) + 3(预留)
        )
        
        # 设置末端执行器的标记可视化
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        
        # 设置末端执行器的Frame Transformer
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=True,  # 开启可视化，方便调试
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/tool0",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.1034),  # 末端执行器偏移
                    ),
                ),
            ],
        )
        
        # 配置场景中的物体 - 使用不同形状和尺寸的小方块
        # 使用多种形状让机器人学习复杂的推抓策略
        
        # 物体1：红色方块
        self.scene.objects[0] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object_0",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, -0.1, 0.025), rot=(0.0, 0.0, 0.0, 1.0)),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.5, 0.5, 0.5),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )
        
        # 物体2：绿色长方体
        self.scene.objects[1] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object_1",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, 0.0, 0.025), rot=(0.0, 0.0, 0.3826834, 0.9238795)),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.6, 0.4, 0.4),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )
        
        # 物体3：蓝色圆柱体
        self.scene.objects[2] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object_2",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.1, 0.025), rot=(0.0, 0.0, 0.0, 1.0)),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Mustard/mustard_bottle.usd",
                scale=(0.7, 0.7, 0.7),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )
        
        # 物体4：黄色小方块
        self.scene.objects[3] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object_3",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, 0.025), rot=(0.0, 0.0, 0.0, 1.0)),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.4, 0.4, 0.4),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )
        
        # 物体5：红色小球
        self.scene.objects[4] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object_4",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.65, -0.1, 0.025), rot=(0.0, 0.0, 0.0, 1.0)),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Balls/Ball.usd",
                scale=(0.03, 0.03, 0.03),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )
        
        # 设置额外的物体（如果需要更多）
        for i in range(5, 10):
            self.scene.objects[i] = None
        
        # 设置相机
        
        # 顶部RGB-D相机（俯视）- 主要用于推抓决策
        self.scene.top_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/top_camera",
            offset=CameraCfg.OffsetCfg(
                pos=(0.5, 0.0, 0.7),  # 相机位置在桌子正上方
                rot=(0, 0, 1, 0),     # 相机朝下
                convention="ros",
            ),
            data_types=["rgb", "distance_to_camera"],  # 收集RGB和深度信息
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.01, 10.0)
            ),
            width=256,  # 图像尺寸
            height=256,
        )
        
        # 侧面相机（从一侧观察）- 辅助观察抓取
        self.scene.side_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/side_camera",
            offset=CameraCfg.OffsetCfg(
                pos=(0.7, 0.5, 0.4),  # 相机位置在桌子侧面
                rot=(-0.271, 0.652, 0.654, -0.271),  # 经典的45度俯视角度
                convention="ros",
            ),
            data_types=["rgb", "distance_to_camera"],  # 收集RGB和深度信息
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.01, 10.0)
            ),
            width=256,  # 图像尺寸
            height=256,
        )
        
        # 配置事件 - 物体随机初始化
        self.events.reset_objects = mdp.EventTermCfg(
            func=mdp.reset_multiple_objects_uniform,
            mode="reset",
            params={
                "num_objects": 5,  # 目前配置了5个物体
                "object_prefix": "Object_",
                "pose_range": {
                    "x": (0.3, 0.7),  # 工作区域X范围
                    "y": (-0.2, 0.2),  # 工作区域Y范围
                    "z": (0.025, 0.025)  # 物体初始Z高度
                },
                "rotation_range": {
                    "roll": (0, 0),
                    "pitch": (0, 0),
                    "yaw": (0, 6.28)  # 0-360度随机旋转
                },
                "min_distance": 0.05  # 物体之间的最小距离
            }
        )

@configclass
class UR5ePushGraspEnvCfg_TRAIN(UR5ePushGraspEnvCfg):
    """用于训练的UR5e视觉推抓环境配置。"""
    
    def __post_init__(self):
        # 初始化父类
        super().__post_init__()
        
        # 训练环境配置
        self.scene.num_envs = 1024  # 较多的并行环境加速训练
        self.scene.env_spacing = 2.5  # 环境之间的间距
        
        # 课程学习配置
        self.curriculum.increase_object_count = mdp.CurrTerm(
            func=mdp.adjust_object_count,
            params={
                "start_count": 2,    # 起始物体数量（简单场景）
                "target_count": 5,   # 目标物体数量（复杂场景）
                "num_steps": 5000    # 达到目标数量的步骤数
            }
        )
        
        # 探索率衰减以更多地利用已学到的策略
        self.curriculum.decrease_exploration = mdp.CurrTerm(
            func=mdp.modify_exploration_rate,
            params={
                "initial_rate": 0.7,  # 初始较高探索率
                "final_rate": 0.1,    # 最终较低探索率
                "num_steps": 10000    # 达到最终探索率的步骤数
            }
        )

@configclass
class UR5ePushGraspEnvCfg_EVAL(UR5ePushGraspEnvCfg):
    """用于评估的UR5e视觉推抓环境配置。"""
    
    def __post_init__(self):
        # 初始化父类
        super().__post_init__()
        
        # 评估环境配置
        self.scene.num_envs = 16  # 较少的并行环境，用于评估
        self.scene.env_spacing = 2.5
        
        # 关闭随机化和探索率
        self.exploration_rate = 0.0  # 纯利用模式
        
        # 固定物体数量为最大值
        self.events.reset_objects = mdp.EventTermCfg(
            func=mdp.reset_multiple_objects_uniform,
            mode="reset",
            params={
                "num_objects": 5,  # 固定使用5个物体
                "object_prefix": "Object_",
                "pose_range": {
                    "x": (0.3, 0.7),
                    "y": (-0.2, 0.2),
                    "z": (0.025, 0.025)
                },
                "rotation_range": {
                    "roll": (0, 0),
                    "pitch": (0, 0),
                    "yaw": (0, 6.28)
                },
                "min_distance": 0.05
            }
        )

@configclass
class UR5ePushGraspEnvCfg_PLAY(UR5ePushGraspEnvCfg):
    """用于演示的UR5e视觉推抓环境配置。"""
    
    def __post_init__(self):
        # 初始化父类
        super().__post_init__()
        
        # 演示环境配置
        self.scene.num_envs = 1  # 单个环境，便于观察
        self.scene.env_spacing = 2.5
        
        # 固定物体初始位置，便于演示
        self.events.reset_objects = mdp.EventTermCfg(
            func=mdp.reset_scene_to_default,
            mode="reset"
        )
        
        # 开启所有可视化和调试工具
        self.scene.ee_frame.debug_vis = True 