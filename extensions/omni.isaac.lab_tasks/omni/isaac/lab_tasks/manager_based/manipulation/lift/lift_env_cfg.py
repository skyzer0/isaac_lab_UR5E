# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.sensors import TiledCameraCfg, CameraCfg

from . import mdp
import math
import torch

##
# Scene definition
##


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # 为这些属性提供默认值而不是MISSING
    # 注意：这些会在子类中被覆盖，这里只是为了通过linter检查提供默认值
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=None  # 会在子类中被正确设置
    )
    
    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        target_frames=[]  # 会在子类中被正确设置
    )
    
    # 多个物体定义
    object1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=None  # 会在子类中被正确设置
    )
    
    object2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object1",
        spawn=None  # 会在子类中被正确设置
    )
    
    object3 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object2",
        spawn=None  # 会在子类中被正确设置
    )
    
    object4 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object3",
        spawn=None  # 会在子类中被正确设置
    )
    
    object5 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object4",
        spawn=None  # 会在子类中被正确设置
    )
    
        # 顶部RGB-D相机（俯视）
    # top_camera = CameraCfg(
    #     prim_path="{ENV_REGEX_NS}/top_camera",
    #     offset=CameraCfg.OffsetCfg(
    #         pos=(0.5, 0.0, 0.7),  # 相机位置
    #         rot=(0, 0, 1, 0),     # 相机朝向
    #     ),
    #     data_types=["rgb", "distance_to_camera"],  # 收集RGB和深度信息
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=24.0,
    #         focus_distance=400.0,
    #         horizontal_aperture=20.955,
    #         clipping_range=(0.01, 10.0)
    #     ),
    #     width=256,  # 图像尺寸
    #     height=256,
    # )
    
    # # 左下角45度相机
    # side_camera = CameraCfg(
    #     prim_path="{ENV_REGEX_NS}/table_cam",
    #         offset=CameraCfg.OffsetCfg(
    #             pos=(1.0, 0.9, 1.5), 
    #             rot=(0.0, 0.437, 0.846, -0.306),
    #             # rot=(0,0,1,0),
    #             convention="ros"
    #         ),
    #         data_types=["rgb","distance_to_camera"],
    #         spawn=sim_utils.PinholeCameraCfg(
    #             focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
    #         ),
    #         width=244,
    #         height=244,
    #     )

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0, 0), rot=(0.707, 0, 0, 0.707)),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, -1.05)),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="tool0",  # 将MISSING改为具体值以解决错误
        resampling_time_range=(60, 60),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6), pos_y=(-0.25, 0.25), pos_z=(0.25, 0.5), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # 创建默认占位符动作配置，避免MISSING类型错误
    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*joint.*"], 
        scale=0.5, 
        use_default_offset=True
    )
    
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["robotiq_85_left_knuckle_joint", "robotiq_85_right_knuckle_joint"],
        open_command_expr={
            "robotiq_85_left_knuckle_joint": 0.00,
            "robotiq_85_right_knuckle_joint": 0.00,
        },
        close_command_expr={
            "robotiq_85_left_knuckle_joint": 1.0,
            "robotiq_85_right_knuckle_joint": 1.0,
        },
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        # joint_vel = ObsTerm(func=mdp.joint_vel_rel)

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    # pass
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # # 物体1重置
    # reset_object1_position = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("object1", body_names="Object1"),
    #     },
    # )
    
    # # 物体2重置
    # reset_object2_position = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {"x": (-0.1, 0.1), "y": (-0.15, 0.15), "z": (0.0, 0.0)},
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("object2", body_names="Object2"),
    #     },
    # )
    
    # # 物体3重置
    # reset_object3_position = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {"x": (-0.1, 0.1), "y": (-0.2, 0.2), "z": (0.0, 0.0)},
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("object3", body_names="Object3"),
    #     },
    # )
    
    # # 物体4重置
    # reset_object4_position = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {"x": (-0.1, 0.1), "y": (-0.18, 0.18), "z": (0.0, 0.0)},
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("object4", body_names="Object4"),
    #     },
    # )
    
    # # 物体5重置
    # reset_object5_position = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {"x": (-0.1, 0.1), "y": (-0.22, 0.22), "z": (0.0, 0.0)},
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("object5", body_names="Object5"),
    #     },
    # )


@configclass
class RewardsCfg:
    """自定义奖励函数，基于ActionStateMachine中的状态实现"""
    
    # 动作和状态惩罚
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-3)  # 惩罚频繁的动作变化
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.1)  # 惩罚关节过快运动
    alive = RewTerm(func=mdp.is_alive, weight=0.2)  # 奖励机械臂保持在活动状态


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # 时间超时终止
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    )


##
# Environment configuration
##


@configclass
class LiftEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 60.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625