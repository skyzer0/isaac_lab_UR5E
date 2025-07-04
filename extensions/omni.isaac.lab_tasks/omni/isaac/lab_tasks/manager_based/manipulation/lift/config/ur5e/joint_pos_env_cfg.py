# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#netron

from omni.isaac.lab.assets import RigidObjectCfg
from omni.isaac.lab.sensors import FrameTransformerCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

from omni.isaac.lab_tasks.manager_based.manipulation.lift import mdp
from omni.isaac.lab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg


##
# Pre-defined configs
##
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG  # isort: skip
from omni.isaac.lab_assets.UR5E_gripper import UR5E_CFG_v2_HIGH_PD_CFG  # isort: skip
from omni.isaac.lab_assets import UR5E_CFG_v2
from omni.isaac.lab.sensors import CameraCfg, RayCasterCameraCfg, TiledCameraCfg
import omni.isaac.lab.sim as sim_utils
import numpy as np
import torch


@configclass
class UR5eCubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # Post-initialize parent class
        super().__post_init__()

        # Set UR5e as robot
        self.scene.robot = UR5E_CFG_v2.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (UR5e with Robotiq gripper)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*joint.*"], scale=0.5, use_default_offset=True
        )

        # Gripper action configuration for Robotiq 85
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
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

        # Set the body name for the end effector
        self.commands.object_pose.body_name = "tool0"

        # Set Cube as object 
        self.scene.object1 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object1",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, -0.13, 0.055), rot=(0.0, 1.0, 0.0, 0.0)),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
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

        # 第一行中间的方块
        self.scene.object2 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object2",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.055), rot=(0.0, 1.0, 0.0, 0.0)),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
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

        # 第一行右边的方块
        self.scene.object3 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object3",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.03, 0.055), rot=(0.0, 1.0, 0.0, 0.0)),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
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

        # 第二行左边的方块
        self.scene.object4 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object4",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.53, -0.03, 0.055), rot=(0.0, 1.0, 0.0, 0.0)),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
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

        # 第二行中间的方块
        self.scene.object5 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object5",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.53, 0.0, 0.055), rot=(0.0, 1.0, 0.0, 0.0)),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
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

        # 第二行右边的方块
        self.scene.object6 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object6",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.53, 0.03, 0.055), rot=(0.0, 1.0, 0.0, 0.0)),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
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
       
       

            # 俯视角相机
        self.scene.top_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/table_cam",
            offset=CameraCfg.OffsetCfg(
                pos=(0.5, 0.0, 0.7),
                rot=(0, 0, 1, 0),  # 修改这里
                convention="ros"
            ),
            data_types=["rgb", "distance_to_camera"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, 
                focus_distance=400.0, 
                horizontal_aperture=20.955, 
                clipping_range=(0.1, 20.0)
            ),
            width=256,
            height=256,
        )


        #正面45
        # self.scene.top_camera = CameraCfg(
        #     prim_path="{ENV_REGEX_NS}/table_cam",
        #     offset=CameraCfg.OffsetCfg(
        #         pos=(1, 0, 0.3),  # 右侧靠近桌面，降低高度
 
        #         # rot=(-0.271,   0.652,  0.654, -0.271),#经典的右侧朝左，45度俯视
        #         rot=(0.5,-0.5,-0.5,0.5),
        #         # rot = ( -0.183,  0.683, 0.683, -0.183), # 右侧朝左，30度俯视
        #         convention="ros"
        #     ),
        #     data_types=["rgb", "distance_to_camera"],
        #     spawn=sim_utils.PinholeCameraCfg(
        #         focal_length=24.0, 
        #         focus_distance=400.0, 
        #         horizontal_aperture=20.955, 
        #         clipping_range=(0.1, 20.0)
        #     ),
        #     width=244,
        #     height=244,
        # )





        
        # # 左下角45度相机
        # self.scene.RGB_Camera = CameraCfg(
        #     prim_path="{ENV_REGEX_NS}/table_cam",
        #     offset=CameraCfg.OffsetCfg(
        #         pos=(1.0, 0.9, 1.5), 
        #         rot=(0.0, 0.437, 0.846, -0.306),
        #         # rot=(0,0,1,0),
        #         convention="ros"
        #     ),
        #     data_types=["rgb","distance_to_camera"],
        #     spawn=sim_utils.PinholeCameraCfg(
        #         focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        #     ),
        #     width=244,
        #     height=244,
        # )

        # camera
        # self.scene.RGB_Camera= CameraCfg(
        #     prim_path="{ENV_REGEX_NS}/Robot/tool0/rear_cam",
        #     offset=CameraCfg.OffsetCfg(
        #         pos=(0.0, 0.0, 0.02), 
        #         # rot=(0.0, 0.7071068, 0.0, 0.7071068),
        #         rot=(1,0,0,0),
        #         convention="ros"),
        #     data_types=["rgb","distance_to_camera"],
        #     spawn=sim_utils.PinholeCameraCfg(
        #         focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        #     ),
        #     width=640,
        #     height=480,
        # )


        # self.scene.RGB_Camera = CameraCfg(
        #     prim_path="{ENV_REGEX_NS}/table_cam",
        #     offset=CameraCfg.OffsetCfg(
        #         pos=(0.5, 0.0, 0.6),
        #         rot=(0, 0.7071, 0, 0.7071),  # 修改这里
        #         convention="ros"
        #     ),
        #     data_types=["rgb", "distance_to_camera"],
        #     spawn=sim_utils.PinholeCameraCfg(
        #         focal_length=24.0, 
        #         focus_distance=400.0, 
        #         horizontal_aperture=20.955, 
        #         clipping_range=(0.1, 20.0)
        #     ),
        #     width=244,
        #     height=244,
        # )
        

        # Frame Transformer configuration for the end effector
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/tool0",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.1034),
                    ),
                ),
            ],
        )


@configclass
class UR5eCubeLiftEnvCfg_PLAY(UR5eCubeLiftEnvCfg):
    def __post_init__(self):
        # Post-initialize parent class
        super().__post_init__()

        # Configure a smaller environment for play mode
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # Disable randomization for play mode
        self.observations.policy.enable_corruption = False

        desired_orientation = torch.zeros((self.scene.num_envs, 4), device=self.scene.device)
        desired_orientation[:, 1] = 1.0  # 设置默认朝向
