# # Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause

# from dataclasses import MISSING
# import numpy as np
# import omni.isaac.lab.sim as sim_utils
# from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
# from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
# from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
# from omni.isaac.lab.managers import EventTermCfg as EventTerm
# from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
# from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
# from omni.isaac.lab.managers import RewardTermCfg as RewTerm
# from omni.isaac.lab.managers import SceneEntityCfg
# from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
# from omni.isaac.lab.scene import InteractiveSceneCfg
# from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
# from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
# from omni.isaac.lab.utils import configclass
# from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
# from omni.isaac.lab.sensors import CameraCfg

# from . import mdp
# import torch

# ##
# # 场景定义
# ##

# @configclass
# class PushGraspSceneCfg(InteractiveSceneCfg):
#     """用于推抓任务的场景配置，包含机器人、多个相机和多个物体。
    
#     这是抽象基础实现，具体场景在派生类中定义，需要设置目标物体、机器人和末端执行器。
#     """
    
#     # 机器人配置
#     robot = ArticulationCfg(
#         prim_path="{ENV_REGEX_NS}/Robot",
#         spawn=None  # 将在子类中设置
#     )
    
#     # 末端执行器帧
#     ee_frame = FrameTransformerCfg(
#         prim_path="{ENV_REGEX_NS}/Robot/tool0",
#         target_frames=[]  # 将在子类中设置
#     )
    
#     # 物体配置（多个小物体）
#     objects = []
#     for i in range(10):  # 默认10个物体
#         objects.append(
#             RigidObjectCfg(
#                 prim_path=f"{{ENV_REGEX_NS}}/Object_{i}",
#                 spawn=None  # 将在子类中设置
#             )
#         )
    
#     # 顶部RGB-D相机（俯视）
#     top_camera = CameraCfg(
#         prim_path="{ENV_REGEX_NS}/top_camera",
#         offset=CameraCfg.OffsetCfg(
#             pos=(0.5, 0.0, 0.7),  # 相机位置
#             rot=(0, 0, 1, 0),     # 相机朝向
#         ),
#         data_types=["rgb", "distance_to_camera"],  # 收集RGB和深度信息
#         spawn=sim_utils.PinholeCameraCfg(
#             focal_length=24.0,
#             focus_distance=400.0,
#             horizontal_aperture=20.955,
#             clipping_range=(0.01, 10.0)
#         ),
#         width=256,  # 图像尺寸
#         height=256,
#     )
    
#     # 侧面相机（从一侧观察）
#     side_camera = CameraCfg(
#         prim_path="{ENV_REGEX_NS}/side_camera",
#         offset=CameraCfg.OffsetCfg(
#             pos=(0.7, 0.5, 0.4),  # 相机位置
#             rot=(0.7071, 0, 0.7071, 0),  # 相机朝向
#         ),
#         data_types=["rgb", "distance_to_camera"],  # 收集RGB和深度信息
#         spawn=sim_utils.PinholeCameraCfg(
#             focal_length=24.0,
#             focus_distance=400.0,
#             horizontal_aperture=20.955,
#             clipping_range=(0.01, 10.0)
#         ),
#         width=256,  # 图像尺寸
#         height=256,
#     )
    
#     # 工作台面
#     table = AssetBaseCfg(
#         prim_path="{ENV_REGEX_NS}/Table",
#         init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0, 0), rot=(0.707, 0, 0, 0.707)),
#         spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
#     )
    
#     # 地面
#     plane = AssetBaseCfg(
#         prim_path="/World/GroundPlane",
#         init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, -1.05)),
#         spawn=GroundPlaneCfg(),
#     )
    
#     # 灯光
#     light = AssetBaseCfg(
#         prim_path="/World/light",
#         spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
#     )

# ##
# # MDP设置
# ##

# @configclass
# class CommandsCfg:
#     """MDP的命令配置"""
#     pass  # 我们不需要命令，因为推抓任务基于视觉

# @configclass
# class ActionsCfg:
#     """MDP的动作配置"""
    
#     # 推动动作（选择方向和位置）
#     push_action = mdp.PushActionCfg(
#         asset_name="robot",
#         ee_body_name="tool0",
#         max_push_distance=0.15,  # 最大推动距离（米）
#         push_height_offset=0.01,  # 推动高度偏移（距离物体表面）
#         z_offset_before_pushing=0.05,  # 开始推动前的高度偏移
#         push_speed=0.1,  # 推动速度（米/秒）
#     )
    
#     # 抓取动作（选择位置和角度）
#     grasp_action = mdp.GraspActionCfg(
#         asset_name="robot",
#         ee_body_name="tool0",
#         gripper_joint_names=["robotiq_85_left_knuckle_joint", "robotiq_85_right_knuckle_joint"],
#         pre_grasp_height=0.1,  # 抓取前的高度（米）
#         grasp_height=0.02,     # 抓取高度（米，相对于物体）
#         post_grasp_height=0.2,  # 抓取后提升高度（米）
#     )
    
#     # 动作选择器（二选一：推或抓）
#     action_selector = mdp.ActionSelectorCfg(
#         action_space_dim=8,  # 动作维度：1(选择器) + 3(位置x,y,z) + 1(角度) + 3(预留)
#     )

# @configclass
# class ObservationsCfg:
#     """MDP的观察配置"""
    
#     @configclass
#     class PolicyCfg(ObsGroup):
#         """策略观察组"""
        
#         # 相机观察
#         top_rgb = ObsTerm(func=mdp.camera_rgb, params={"camera_name": "top_camera"})
#         top_depth = ObsTerm(func=mdp.camera_depth, params={"camera_name": "top_camera"})
#         side_rgb = ObsTerm(func=mdp.camera_rgb, params={"camera_name": "side_camera"})
#         side_depth = ObsTerm(func=mdp.camera_depth, params={"camera_name": "side_camera"})
        
#         # 机器人状态
#         ee_pose = ObsTerm(func=mdp.end_effector_pose, params={"asset_name": "robot", "body_name": "tool0"})
#         gripper_state = ObsTerm(func=mdp.gripper_joints_pos, params={"asset_name": "robot", "joint_names": ["robotiq_85_left_knuckle_joint"]})
        
#         # 上一个动作
#         last_action = ObsTerm(func=mdp.last_action)
    
#     # 观察组
#     policy: PolicyCfg = PolicyCfg()

# @configclass
# class EventCfg:
#     """事件配置"""
    
#     # 重置所有
#     reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    
#     # 随机重置所有物体位置
#     reset_objects = EventTerm(
#         func=mdp.reset_multiple_objects_uniform,
#         mode="reset",
#         params={
#             "num_objects": 10,
#             "object_prefix": "object_",
#             "pose_range": {
#                 "x": (0.3, 0.7),  # 工作区域X范围
#                 "y": (-0.2, 0.2),  # 工作区域Y范围
#                 "z": (0.05, 0.05)  # 物体初始Z高度
#             },
#             "rotation_range": {
#                 "roll": (0, 0),
#                 "pitch": (0, 0),
#                 "yaw": (0, 6.28)  # 0-360度随机旋转
#             },
#             "min_distance": 0.05  # 物体之间的最小距离
#         }
#     )

# @configclass
# class RewardsCfg:
#     """奖励配置"""
    
#     # 推动奖励（如果推动导致物体位置变化）
#     push_reward = RewTerm(
#         func=mdp.push_success_reward,
#         weight=0.5,
#         params={
#             "scene_change_threshold": 0.02,  # 场景变化检测阈值
#             "min_pixels_changed": 100,  # 最少像素变化数
#         }
#     )
    
#     # 抓取奖励（如果成功抓起物体）
#     grasp_reward = RewTerm(
#         func=mdp.grasp_success_reward,
#         weight=1.0,
#         params={
#             "height_threshold": 0.05,  # 物体被抓起的高度阈值
#             "gripper_close_threshold": 0.5,  # 判断爪子闭合的阈值
#         }
#     )
    
#     # 物体清除奖励（对工作区域的物体减少给予额外奖励）
#     clearance_reward = RewTerm(
#         func=mdp.workspace_clearance_reward, 
#         weight=0.2,
#         params={
#             "workspace_bounds": {
#                 "x": (0.3, 0.7),
#                 "y": (-0.2, 0.2),
#                 "z": (0.0, 0.1)
#             }
#         }
#     )
    
#     # 惩罚（摄像机视野没有检测到物体）
#     empty_workspace_penalty = RewTerm(
#         func=mdp.empty_workspace_penalty,
#         weight=-0.5,
#         params={
#             "min_object_count": 3  # 工作区最少物体数
#         }
#     )

# @configclass
# class TerminationsCfg:
#     """终止条件配置"""
    
#     # 时间限制
#     time_out = DoneTerm(
#         func=mdp.time_out,
#         time_out=True
#     )
    
#     # 工作区清空（成功完成任务）
#     workspace_cleared = DoneTerm(
#         func=mdp.workspace_cleared,
#         params={
#             "max_objects_left": 0,  # 工作区内最多剩余物体数
#             "workspace_bounds": {
#                 "x": (0.3, 0.7),
#                 "y": (-0.2, 0.2),
#                 "z": (0.0, 0.1)
#             }
#         }
#     )

# @configclass
# class CurriculumCfg:
#     """课程学习配置"""
    
#     # 增加物体数量的课程学习
#     increase_object_count = CurrTerm(
#         func=mdp.adjust_object_count,
#         params={
#             "start_count": 3,    # 起始物体数量
#             "target_count": 10,  # 目标物体数量
#             "num_steps": 5000    # 达到目标数量的步骤数
#         }
#     )
    
#     # 调整奖励权重
#     adjust_push_reward = CurrTerm(
#         func=mdp.modify_reward_weight,
#         params={
#             "term_name": "push_reward",
#             "weight": 0.3,      # 起始权重
#             "target_weight": 0.5,  # 目标权重
#             "num_steps": 3000
#         }
#     )
    
#     # 逐渐减少探索
#     decrease_exploration = CurrTerm(
#         func=mdp.modify_exploration_rate,
#         params={
#             "initial_rate": 0.5,  # 初始探索率
#             "final_rate": 0.1,    # 最终探索率
#             "num_steps": 10000    # 达到最终探索率的步骤数
#         }
#     )

# @configclass
# class PushGraspEnvCfg(ManagerBasedRLEnvCfg):
#     """推抓环境的主配置"""
    
#     # 场景设置
#     scene: PushGraspSceneCfg = PushGraspSceneCfg(num_envs=1024, env_spacing=2.5)
    
#     # 基本设置
#     observations: ObservationsCfg = ObservationsCfg()
#     actions: ActionsCfg = ActionsCfg()
#     commands: CommandsCfg = CommandsCfg()
    
#     # MDP设置
#     rewards: RewardsCfg = RewardsCfg()
#     terminations: TerminationsCfg = TerminationsCfg()
#     events: EventCfg = EventCfg()
#     curriculum: CurriculumCfg = CurriculumCfg()
    
#     def __post_init__(self):
#         """初始化后的设置"""
#         # 一般设置
#         self.decimation = 2  # 每两个仿真步骤对应一个控制步骤
#         self.episode_length_s = 120.0  # 每个回合120秒
        
#         # 仿真设置
#         self.sim.dt = 0.01  # 100Hz
#         self.sim.render_interval = self.decimation
        
#         # 物理设置
#         self.sim.physx.bounce_threshold_velocity = 0.2
#         self.sim.physx.friction_correlation_distance = 0.00625
#         self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
#         self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024 