# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.assets import DeformableObjectCfg
from omni.isaac.lab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from omni.isaac.lab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sim.spawners import UsdFileCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR

import omni.isaac.lab_tasks.manager_based.manipulation.lift.mdp as mdp

from . import joint_pos_env_cfg

##
# Pre-defined configs
##
from omni.isaac.lab_assets.UR5E_gripper import UR5E_CFG_v2_HIGH_PD_CFG  # isort: skip

##
# Rigid object lift environment for UR5e with Robotiq gripper.
##

@configclass
class UR5eCubeLiftEnvCfg(joint_pos_env_cfg.UR5eCubeLiftEnvCfg):
    def __post_init__(self):
        # Post-initialize parent class
        super().__post_init__()

        # Set UR5e robot configuration with high PD gains for IK tracking
        self.scene.robot = UR5E_CFG_v2_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set the actions and joint control for UR5e with Robotiq gripper
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=[".*joint.*"],
            body_name="tool0",  # End-effector frame
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.14]),
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

##
# Deformable object lift environment for UR5e with Robotiq gripper.
##

@configclass
class UR5eTeddyBearLiftEnvCfg(UR5eCubeLiftEnvCfg):
    def __post_init__(self):
        # Post-initialize parent class
        super().__post_init__()
        print("Available actuators:", self.scene.robot.actuators.keys())

        self.scene.object = DeformableObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.5, 0, 0.05), rot=(0.707, 0, 0, 0.707)),
            spawn=UsdFileCfg(
                usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Objects/Teddy_Bear/teddy_bear.usd",
                scale=(0.01, 0.01, 0.01),
            ),
        )

        # Set the gripper's actuators to softer settings to avoid damaging the Teddy Bear
        self.scene.robot.actuators["ur5e_hand"].effort_limit = 50.0
        self.scene.robot.actuators["ur5e_hand"].stiffness = 40.0
        self.scene.robot.actuators["ur5e_hand"].damping = 10.0

        # Disable replicate physics due to limitations with deformable objects
        self.scene.replicate_physics = False

        # Configure events for the deformable object type
        self.events.reset_object_position = EventTerm(
            func=mdp.reset_nodal_state_uniform,
            mode="reset",
            params={
                "position_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("object"),
            },
        )

        # Disable state machine terms for performance
        self.terminations.object_dropping = None
        self.rewards.reaching_object = None
        self.rewards.lifting_object = None
        self.rewards.object_goal_tracking = None
        self.rewards.object_goal_tracking_fine_grained = None
        self.observations.policy.object_position = None



