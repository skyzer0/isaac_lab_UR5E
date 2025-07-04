######
##### to run this code run the following command in terminal
##./isaaclab.sh -p source/standalone/tutorials/06_own_testing/ik_cube_image.py --headless --save






from __future__ import annotations

"""Launch Isaac Sim Simulator first."""
import argparse
from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
# parser.add_argument("--cpu", action="store_true", default='cuda', help="Use CPU device for camera output.")
parser.add_argument("--robot", type=str, default="ur5e", help="Name of the robot.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument(
    "--camera_id",
    type=int,
    choices={0, 1},
    default=0,
    help=(
        "The camera ID to use for displaying points or saving the camera data. Default is 0."
        " The viewport will always initialize with the perspective of camera 0."
    ),
)

parser.add_argument(
    "--draw",
    action="store_true",
    default=False,
    help="Draw the pointcloud from camera at index specified by ``--camera_id``.",
)

parser.add_argument(
    "--save",
    action="store_true",
    default=False,
    help="Save the data from camera at index specified by ``--camera_id``.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# 启动 Omniverse 应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""其余部分代码在此之后编写。"""

from omni.isaac.lab.sim import SimulationCfg, SimulationContext
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, AssetBaseCfg
from omni.isaac.lab_assets import UR5E_CFG
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import subtract_frame_transforms
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg

#TESTING
from omni.isaac.lab_assets import FRANKA_PANDA_HIGH_PD_CFG
from omni.isaac.lab_assets import UR5E_CFG_v2
from omni.isaac.lab_assets import UR10_CFG
from omni.isaac.lab.sensors import FrameTransformerCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg

from omni.isaac.lab.markers.config import FRAME_MARKER_CFG  # isort: skip
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from omni.isaac.lab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns
import warp as wp
import torch
import numpy as np
from PIL import Image
import os
import omni.replicator.core as rep
from omni.isaac.lab.utils import convert_dict_to_backend
from omni.isaac.lab.utils.math import transform_points, unproject_depth
from omni.isaac.lab.sensors.camera.utils import create_pointcloud_from_depth
from omni.isaac.lab.markers.config import RAY_CASTER_MARKER_CFG
from omni.isaac.lab.utils.math import transform_points, unproject_depth
import open3d as o3d



# 配置类
@configclass
class ur5e_SceneCfg(InteractiveSceneCfg):
    """设计场景，通过 USD 文件生成地面、光源、对象和网格。"""

    # 地面
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # 灯光
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    # 设置 UR5e 机械臂
     # articulation
    if args_cli.robot == "franka_panda":
        robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    elif args_cli.robot == "ur10":
        robot = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    elif args_cli.robot == "ur5e":
        robot: AssetBaseCfg= (UR5E_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"))
    elif args_cli.robot == "ur5e_g":
        robot: AssetBaseCfg= (UR5E_CFG_v2.replace(prim_path="{ENV_REGEX_NS}/Robot"))
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10, ur5e")
    
    # cube = AssetBaseCfg(prim_path="World/Cube", spawn=sim_utils.Cubecfg())

    # Set Cube as object
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.6, 0.0, 0.3], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
        scale=(0.8, 0.8, 0.8), 
        rigid_props=RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,     
                max_depenetration_velocity=5.0, # Adjust this value if necessary
                disable_gravity=False, # Change this to True if you want to enable gravity
                ), 
            ), 
        ) 


    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/tool0/front_cam1",
        update_period=0.1,
        height=480,
        width=640,
        data_types=[
            "rgb",
            "distance_to_image_plane",
            # "normals",
            # "semantic_segmentation",
            # "instance_segmentation_fast",
            # "instance_id_segmentation_fast",
            ],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        ##default facing down the z-axis
        ##increase the z value to move the camera down
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.1), 
            rot=(1.0, 0.0, 0.0, 0.0), 
            convention="ros"
        ),
    )



def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    robot = scene["robot"]
    cube = scene["cube"]
    camera=scene["camera"]
    # camera: Camera = scene["camera"]

    # print("world_cam", world_cam)
    # camera=env.unwrapped.scene["camera"]
    camera_index = args_cli.camera_id
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "camera")
    rep_writer = rep.BasicWriter(
        output_dir=output_dir,
        frame_padding=0,
        colorize_instance_id_segmentation=camera.cfg.colorize_instance_id_segmentation,
        colorize_instance_segmentation=camera.cfg.colorize_instance_segmentation,
        colorize_semantic_segmentation=camera.cfg.colorize_semantic_segmentation,
    )

    cfg = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/CameraPointCloud")
    cfg.markers["hit"].radius = 0.002
    pc_markers = VisualizationMarkers(cfg)


    single_cam_data = convert_dict_to_backend(camera.data.output[camera_index], backend="numpy")
    single_cam_info = camera.data.info[camera_index]
    rep_output = {"annotators": {}}
    # Create controller
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    # ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    # goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))
    scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link", 
            debug_vis=False,
            visualizer_cfg=frame_marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/tool0",  # Update this to the end effector link of UR5e
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, -1.5],  # Adjust the offset if necessary
                    ),
                ),
            ],
        )
    
    
    # Specify robot-specific parameters
    if args_cli.robot == "franka_panda":
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    elif args_cli.robot == "ur10":
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*"], body_names=["ee_link"])
    elif args_cli.robot == "ur5e":
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*"], body_names=["tool0"])
    elif args_cli.robot == "ur5e_g":
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*"], body_names=["tool0"])
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10, ur5e")
    # Resolving the scene entities
    robot_entity_cfg.resolve(scene)
    # Obtain the frame index of the end-effector
    # For a fixed base robot, the frame index is one less than the body index. This is because
    # the root body is not included in the returned Jacobians.
    if robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0]

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    while simulation_app.is_running():

        # offset=torch.tensor([0.0, 0.0, 0.1], device=sim.device) 

        # offset_orientation=torch.tensor([0.0, 1.0, 0.0, 0.0], device=sim.device)
        cube_pos = cube.data.root_pos_w 
        object_position = cube_pos - scene.env_origins
        desired_orientation = cube.data.root_quat_w 
        desired_orientation[:, [0,1,2,3]] = desired_orientation[:, [1, 0,2,3]]


        # print ("designed",desired_orientation)
        object=torch.cat([object_position, desired_orientation], dim=-1)
        ee_goals = object
        ee_goals = torch.tensor(ee_goals, device=sim.device)


        # Track the given command
        current_goal_idx = 0


        # Create buffers to store actions
        ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=robot.device)
        ik_commands[:] = ee_goals[current_goal_idx]


        if args_cli.save:
            rep_output = {"annotators": {}}
            for key, data, info in zip(single_cam_data.keys(), single_cam_data.values(), single_cam_info.values()):
                if info is not None:
                    rep_output["annotators"][key] = {"render_product": {"data": data, **info}}
                else:
                    rep_output["annotators"][key] = {"render_product": {"data": data}}
                for key, data, info in zip(single_cam_data.keys(), single_cam_data.values(), single_cam_info.values()):
                    if info is not None:
                        rep_output["annotators"][key] = {"render_product": {"data": data, **info}}
                    else:
                        rep_output["annotators"][key] = {"render_product": {"data": data}}
                rep_output["trigger_outputs"] = {"on_time": camera.frame[camera_index]}
                rep_writer.write(rep_output)

                # Pointcloud in world frame
                points_3d_cam = unproject_depth(
                camera.data.output["distance_to_image_plane"], camera.data.intrinsic_matrices
                )
                points_np = points_3d_cam.numpy().reshape(-1, 3)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points_np)
                o3d.io.write_point_cloud("/home/simtech/IsaacLab/source/standalone/sky/output/point/output.pcd", pcd)
                print("points_3d_cam",points_3d_cam)


                # Create the markers for the --draw option outside of is_running() loop
            if args_cli.draw and "distance_to_image_plane" in camera.data.output.keys():
                # Derive pointcloud from camera at camera_index
                pointcloud = create_pointcloud_from_depth(
                    intrinsic_matrix=camera.data.intrinsic_matrices[camera_index],
                    depth=camera.data.output[camera_index]["distance_to_image_plane"],
                    position=camera.data.pos_w[camera_index],
                    orientation=camera.data.quat_w_ros[camera_index],
                    device='cuda:1',
                )
                if pointcloud.size()[0] > 0:
                    pc_markers.visualize(translations=pointcloud)
        # reset
        if count % 150 == 0:
            # reset time
            count = 0
            # reset joint state
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()
            # reset actions
            ik_commands[:] = ee_goals[current_goal_idx]
            joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()
            # reset controller
            diff_ik_controller.reset()
            diff_ik_controller.set_command(ik_commands)
            # change goal
            current_goal_idx = (current_goal_idx + 1) % len(ee_goals)
        else:
            # obtain quantities from simulation
            jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
            ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
            root_pose_w = robot.data.root_state_w[:, 0:7]
            joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
            # compute frame in root frame
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            # compute the joint commands
            joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

            ##obtain image from simulation
            # print information from the sensors
            sim_dt = sim.get_physics_dt()
            count = 0

        # apply actions
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        # print("joint_pos_des",joint_pos_des)
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        count += 1
        # update buffers
        scene.update(sim_dt)

        # obtain quantities from simulation
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        # update marker positions
        # ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        # goal_marker.visualize(ik_commands[:, 0:3] + scene.env_origins, ik_commands[:, 3:7])



def main():
    """主函数。"""
    # 初始化仿真上下文
    sim_cfg = sim_utils.SimulationCfg(device="cuda" if args_cli.cpu else "cpu")
    sim = SimulationContext(sim_cfg)

    # 设置主摄像头视角
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    # 创建场景
    scene_cfg = ur5e_SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    # 重置仿真
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # 运行主函数
    main()
    # 关闭仿真应用
    simulation_app.close()
