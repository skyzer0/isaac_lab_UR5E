import time
import datetime
import os
import numpy as np
import cv2
import torch 
# import h5py 

class Logger():

    def __init__(self, continue_logging, logging_directory):

        # Create directory to save data
        timestamp = time.time()
        timestamp_value = datetime.datetime.fromtimestamp(timestamp)
        self.continue_logging = continue_logging
        if self.continue_logging:
            self.base_directory = logging_directory
            print('Pre-loading data logging session: %s' % (self.base_directory))
        else:
            self.base_directory = os.path.join(logging_directory, timestamp_value.strftime('%Y-%m-%d.%H:%M:%S'))
            print('Creating data logging session: %s' % (self.base_directory))
        self.info_directory = os.path.join(self.base_directory, 'info')
        self.color_images_directory = os.path.join(self.base_directory, 'data', 'color-images')
        self.depth_images_directory = os.path.join(self.base_directory, 'data', 'depth-images')
        self.color_heightmaps_directory = os.path.join(self.base_directory, 'data', 'color-heightmaps')
        self.depth_heightmaps_directory = os.path.join(self.base_directory, 'data', 'depth-heightmaps')
        self.models_directory = os.path.join(self.base_directory, 'models')
        self.visualizations_directory = os.path.join(self.base_directory, 'visualizations')
        self.recordings_directory = os.path.join(self.base_directory, 'recordings')
        self.transitions_directory = os.path.join(self.base_directory, 'transitions')

        if not os.path.exists(self.info_directory):
            os.makedirs(self.info_directory)
        if not os.path.exists(self.color_images_directory):
            os.makedirs(self.color_images_directory)
        if not os.path.exists(self.depth_images_directory):
            os.makedirs(self.depth_images_directory)
        if not os.path.exists(self.color_heightmaps_directory):
            os.makedirs(self.color_heightmaps_directory)
        if not os.path.exists(self.depth_heightmaps_directory):
            os.makedirs(self.depth_heightmaps_directory)
        if not os.path.exists(self.models_directory):
            os.makedirs(self.models_directory)
        if not os.path.exists(self.visualizations_directory):
            os.makedirs(self.visualizations_directory)
        if not os.path.exists(self.recordings_directory):
            os.makedirs(self.recordings_directory)
        if not os.path.exists(self.transitions_directory):
            os.makedirs(os.path.join(self.transitions_directory, 'data'))

    def save_camera_info(self, intrinsics, pose, depth_scale):
        np.savetxt(os.path.join(self.info_directory, 'camera-intrinsics.txt'), intrinsics, delimiter=' ')
        np.savetxt(os.path.join(self.info_directory, 'camera-pose.txt'), pose, delimiter=' ')
        np.savetxt(os.path.join(self.info_directory, 'camera-depth-scale.txt'), [depth_scale], delimiter=' ')

    def save_heightmap_info(self, boundaries, resolution):
        np.savetxt(os.path.join(self.info_directory, 'heightmap-boundaries.txt'), boundaries, delimiter=' ')
        np.savetxt(os.path.join(self.info_directory, 'heightmap-resolution.txt'), [resolution], delimiter=' ')

    def save_images(self, iteration, color_image, depth_image, mode):
        if color_image is not None and color_image.size > 0:
            # 确保颜色图像是BGR格式（OpenCV默认格式）
            if color_image.shape[2] == 3:  # 检查是否是3通道图像
                color_image_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
            else:
                color_image_bgr = color_image
            
            # 保存颜色图像
            cv2.imwrite(os.path.join(self.color_images_directory, '%06d.%s.color.png' % (iteration, mode)), color_image_bgr)
        else:
            print(f"Warning: Color image is empty or invalid, skipping save")
        
        if depth_image is not None and depth_image.size > 0:
            # 处理深度图 - 处理无效值
            depth_image_clean = depth_image.copy()
            
            # 替换NaN和Inf值
            mask = np.isnan(depth_image_clean) | np.isinf(depth_image_clean) | (depth_image_clean < 0)
            if np.any(mask):
                depth_image_clean[mask] = 0
            
            # 检查是否有正值
            if np.any(depth_image_clean > 0):
                # 去除异常值（可选）
                valid_depths = depth_image_clean[depth_image_clean > 0]
                percentile_99 = np.percentile(valid_depths, 99) if len(valid_depths) > 0 else np.max(depth_image_clean)
                depth_image_clean = np.clip(depth_image_clean, 0, percentile_99)
                
                # 确保数据在有效范围内并转换为uint16
                depth_image_save = np.round(depth_image_clean * 10000).astype(np.uint16)  # Save depth in 1e-4 meters
                
                # 保存深度图
                cv2.imwrite(os.path.join(self.depth_images_directory, '%06d.%s.depth.png' % (iteration, mode)), depth_image_save)
                
                # 可视化版本（可选，用于调试）
                depth_norm = depth_image_clean / np.max(depth_image_clean) if np.max(depth_image_clean) > 0 else depth_image_clean
                depth_vis = (depth_norm * 255).astype(np.uint8)
                depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                cv2.imwrite(os.path.join(self.depth_images_directory, '%06d.%s.depth.vis.png' % (iteration, mode)), depth_colormap)
            else:
                # 如果没有有效深度，创建空白深度图
                print(f"Warning: Depth map has no valid values, saving blank depth map")
                depth_zeros = np.zeros_like(depth_image_clean, dtype=np.uint16)
                cv2.imwrite(os.path.join(self.depth_images_directory, '%06d.%s.depth.png' % (iteration, mode)), depth_zeros)
        else:
            print(f"Warning: Depth image is empty or invalid, skipping save")
    
    def save_heightmaps(self, iteration, color_heightmap, depth_heightmap, mode):
        color_heightmap = cv2.cvtColor(color_heightmap, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(self.color_heightmaps_directory, '%06d.%s.color.png' % (iteration, mode)), color_heightmap)
        
        # 确保深度图在有效范围内并正确处理
        if depth_heightmap is not None and depth_heightmap.size > 0:
            # 检查NaN和Inf值
            mask = np.isnan(depth_heightmap) | np.isinf(depth_heightmap)
            if np.any(mask):
                # 创建深度图副本并替换无效值
                depth_heightmap_clean = depth_heightmap.copy()
                depth_heightmap_clean[mask] = 0
            else:
                depth_heightmap_clean = depth_heightmap
            
            # 归一化并转换为16位格式 (乘以100000转换为1e-5米)
            depth_min = np.min(depth_heightmap_clean[depth_heightmap_clean > 0]) if np.any(depth_heightmap_clean > 0) else 0
            depth_max = np.max(depth_heightmap_clean) if np.any(depth_heightmap_clean > 0) else 1
            
            if depth_max > depth_min:
                # 清除极端值（可选）
                percentile_99 = np.percentile(depth_heightmap_clean[depth_heightmap_clean > 0], 99) if np.any(depth_heightmap_clean > 0) else depth_max
                depth_heightmap_clean = np.clip(depth_heightmap_clean, 0, percentile_99)
                
                # 转换为uint16格式保存
                depth_save = np.round(depth_heightmap_clean * 100000).astype(np.uint16)
            else:
                # 如果深度图全部相同，创建空白深度图
                depth_save = np.zeros_like(depth_heightmap_clean, dtype=np.uint16)
            
            # 保存深度图
            cv2.imwrite(os.path.join(self.depth_heightmaps_directory, '%06d.%s.depth.png' % (iteration, mode)), depth_save)
            
            # 可视化版本的深度图（用于调试）
            depth_vis = np.clip(depth_heightmap_clean, 0, 1)
            depth_vis = (depth_vis * 255).astype(np.uint8)
            depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(self.depth_heightmaps_directory, '%06d.%s.depth.vis.png' % (iteration, mode)), depth_colormap)
        else:
            print(f"Warning: Depth heightmap is empty or invalid, skipping save")
    
    def write_to_log(self, log_name, log):
        np.savetxt(os.path.join(self.transitions_directory, '%s.log.txt' % log_name), log, delimiter=' ')

    def save_model(self, iteration, model, name):
        torch.save(model.cpu().state_dict(), os.path.join(self.models_directory, 'snapshot-%06d.%s.pth' % (iteration, name)))

    def save_backup_model(self, model, name):
        torch.save(model.state_dict(), os.path.join(self.models_directory, 'snapshot-backup.%s.pth' % (name)))

    def save_visualizations(self, iteration, affordance_vis, name):
        try:
            # 检查输入是否有效
            if affordance_vis is None:
                print(f"警告: {name}可视化为空，跳过保存")
                return
            
            if isinstance(affordance_vis, np.ndarray):
                if affordance_vis.size == 0 or affordance_vis.shape[0] == 0 or affordance_vis.shape[1] == 0:
                    print(f"警告: {name}可视化形状无效: {affordance_vis.shape}，跳过保存")
                    return
                
                # 确保图像是uint8类型
                if affordance_vis.dtype != np.uint8:
                    affordance_vis = np.clip(affordance_vis, 0, 255).astype(np.uint8)
                
                # 确保图像是3通道（RGB）
                if len(affordance_vis.shape) == 2:
                    affordance_vis = cv2.cvtColor(affordance_vis, cv2.COLOR_GRAY2BGR)
                elif len(affordance_vis.shape) == 3 and affordance_vis.shape[2] == 1:
                    affordance_vis = cv2.cvtColor(affordance_vis, cv2.COLOR_GRAY2BGR)
                elif len(affordance_vis.shape) == 3 and affordance_vis.shape[2] == 3:
                    # 如果已经是3通道RGB，转换为BGR（OpenCV格式）
                    affordance_vis = cv2.cvtColor(affordance_vis, cv2.COLOR_RGB2BGR)
                elif len(affordance_vis.shape) == 3 and affordance_vis.shape[2] != 3:
                    print(f"警告: {name}可视化通道数异常({affordance_vis.shape[2]})，创建替代图像")
                    affordance_vis = np.ones((200, 200, 3), dtype=np.uint8) * 128
            
            # 保存图像
            cv2.imwrite(os.path.join(self.visualizations_directory, '%06d.%s.png' % (iteration, name)), affordance_vis)
            print(f"成功保存{name}可视化图像")
        except Exception as e:
            print(f"保存{name}可视化时出错: {e}")
            # 尝试保存一个替代图像
            try:
                fallback_img = np.ones((200, 200, 3), dtype=np.uint8) * 128
                cv2.putText(fallback_img, f"Error saving {name}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                fallback_path = os.path.join(self.visualizations_directory, '%06d.%s.fallback.png' % (iteration, name))
                cv2.imwrite(fallback_path, fallback_img)
                print(f"已保存替代图像到: {fallback_path}")
            except Exception as se:
                print(f"无法保存替代图像: {se}")

    # def save_state_features(self, iteration, state_feat):
    #     h5f = h5py.File(os.path.join(self.visualizations_directory, '%06d.state.h5' % (iteration)), 'w')
    #     h5f.create_dataset('state', data=state_feat.cpu().data.numpy())
    #     h5f.close()

    # Record RGB-D video while executing primitive
    # recording_directory = logger.make_new_recording_directory(iteration)
    # camera.start_recording(recording_directory)
    # camera.stop_recording()
    def make_new_recording_directory(self, iteration):
        recording_directory = os.path.join(self.recordings_directory, '%06d' % (iteration))
        if not os.path.exists(recording_directory):
            os.makedirs(recording_directory)
        return recording_directory

    def save_transition(self, iteration, transition):
        depth_heightmap = np.round(transition.state * 100000).astype(np.uint16) # Save depth in 1e-5 meters
        cv2.imwrite(os.path.join(self.transitions_directory, 'data', '%06d.0.depth.png' % (iteration)), depth_heightmap)
        next_depth_heightmap = np.round(transition.next_state * 100000).astype(np.uint16) # Save depth in 1e-5 meters
        cv2.imwrite(os.path.join(self.transitions_directory, 'data', '%06d.1.depth.png' % (iteration)), next_depth_heightmap)
        # np.savetxt(os.path.join(self.transitions_directory, '%06d.action.txt' % (iteration)), [1 if (transition.action == 'grasp') else 0], delimiter=' ')
        # np.savetxt(os.path.join(self.transitions_directory, '%06d.reward.txt' % (iteration)), [reward_value], delimiter=' ')

    def save_pointcloud_isaaclab(self, iteration, color_image, depth_image, camera_intrinsics, camera_pose, mode, device="cpu"):
        """使用IsaacLab风格的函数保存点云
        
        Args:
            iteration: 迭代次数
            color_image: RGB图像
            depth_image: 深度图像
            camera_intrinsics: 相机内参
            camera_pose: 相机位姿 (位置和方向)
            mode: 模式标识符
            device: 计算设备 ("cpu" 或 "cuda")
        """
        # 创建点云目录（如果不存在）
        pointcloud_directory = os.path.join(self.base_directory, 'data', 'pointclouds')
        if not os.path.exists(pointcloud_directory):
            os.makedirs(pointcloud_directory)
        
        # 处理深度图中的无效值
        depth_cleaned = depth_image.copy()
        mask = np.isnan(depth_cleaned) | np.isinf(depth_cleaned) | (depth_cleaned <= 0)
        if np.any(mask):
            depth_cleaned[mask] = 0.0
        
        # 提取相机位置和方向
        position = camera_pose[:3, 3]
        
        # 从旋转矩阵转换为四元数
        from scipy.spatial.transform import Rotation
        rot_matrix = camera_pose[:3, :3]
        r = Rotation.from_matrix(rot_matrix)
        orientation = r.as_quat()  # [x, y, z, w]
        # 转换为ROS约定 [w, x, y, z]
        orientation = np.array([orientation[3], orientation[0], orientation[1], orientation[2]])
        
        # 保存点云文件
        from utils import create_pointcloud_from_depth_isaaclab, save_pointcloud_as_ply_isaaclab
        
        # 生成点云
        try:
            print(f"Creating point cloud...")
            print(f"Depth map shape: {depth_cleaned.shape}")
            print(f"Intrinsic matrix shape: {camera_intrinsics.shape}")
            print(f"Camera position: {position}")
            print(f"Camera orientation: {orientation}")
            
            # 创建点云
            points, colors = create_pointcloud_from_depth_isaaclab(
                camera_intrinsics, 
                depth_cleaned, 
                position, 
                orientation, 
                device=device,
                rgb_image=color_image
            )
            
            # 如果是CUDA张量，移到CPU
            if isinstance(points, torch.Tensor):
                points = points.cpu().numpy()
            if isinstance(colors, torch.Tensor):
                colors = colors.cpu().numpy()
            
            # 过滤掉无效点（可选）
            if len(points) > 0:
                valid_mask = ~np.isnan(points).any(axis=1) & ~np.isinf(points).any(axis=1)
                if np.sum(valid_mask) > 0:
                    points = points[valid_mask]
                    if colors is not None and len(colors) == len(valid_mask):
                        colors = colors[valid_mask]
            
            # 设置保存路径
            pointcloud_filename = os.path.join(pointcloud_directory, '%06d.%s.ply' % (iteration, mode))
            
            # 保存点云
            print(f"Point cloud generation complete, saving to: {pointcloud_filename}")
            print(f"Point cloud contains {len(points)} points")
            
            if len(points) > 0:
                success = save_pointcloud_as_ply_isaaclab(points, pointcloud_filename, colors)
                
                if success:
                    print(f"✅ Point cloud successfully saved: {pointcloud_filename}")
                else:
                    print(f"❌ Point cloud save failed: {pointcloud_filename}")
            else:
                print(f"❌ Generated point cloud is empty, skipping save")
            
        except Exception as e:
            print(f"Error generating or saving point cloud: {str(e)}")
            import traceback
            traceback.print_exc()

    def save_debug_visualizations(self, iteration, color_heightmap, depth_heightmap, points=None, colors=None, workspace_limits=None, heightmap_resolution=None, mode="debug"):
        """保存调试可视化图像
        
        Args:
            iteration: 迭代次数
            color_heightmap: 颜色高度图
            depth_heightmap: 深度高度图
            points: 可选的点云数据
            colors: 可选的点云颜色
            workspace_limits: 工作空间边界
            heightmap_resolution: 高度图分辨率
            mode: 模式标识符
        """
        # 创建调试目录
        debug_directory = os.path.join(self.base_directory, 'debug')
        if not os.path.exists(debug_directory):
            os.makedirs(debug_directory)
        
        try:
            # 导入调试可视化函数
            from utils import visualize_heightmap, analyze_point_cloud, debug_heightmap_generation
            import cv2
            
            # 高度图可视化
            if depth_heightmap is not None and depth_heightmap.size > 0:
                # 使用matplotlib生成高度图可视化
                save_path = os.path.join(debug_directory, '%06d.%s.heightmap.vis.png' % (iteration, mode))
                vis_img = visualize_heightmap(depth_heightmap, color_heightmap, 
                                              title=f"高度图可视化 (迭代: {iteration}, 模式: {mode})",
                                              save_path=save_path)
                
                # 如果返回的是RGBA图像，转换为RGB
                if vis_img.shape[2] == 4:
                    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGBA2BGR)
                elif vis_img.shape[2] == 3:
                    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
                
                # 保存可视化结果
                cv2.imwrite(os.path.join(debug_directory, '%06d.%s.heightmap.debug.png' % (iteration, mode)), vis_img)
            
            # 点云分析（如果提供了点云数据）
            if points is not None and workspace_limits is not None and heightmap_resolution is not None:
                # 点云统计分析
                save_path = os.path.join(debug_directory, '%06d.%s.pointcloud.analysis.png' % (iteration, mode))
                pc_vis = analyze_point_cloud(points, colors, 
                                            title=f"点云分析 (迭代: {iteration}, 模式: {mode})",
                                            save_path=save_path)
                
                if pc_vis is not None:
                    # 如果返回的是RGBA图像，转换为RGB
                    if pc_vis.shape[2] == 4:
                        pc_vis = cv2.cvtColor(pc_vis, cv2.COLOR_RGBA2BGR)
                    elif pc_vis.shape[2] == 3:
                        pc_vis = cv2.cvtColor(pc_vis, cv2.COLOR_RGB2BGR)
                    
                    # 保存点云分析结果
                    cv2.imwrite(os.path.join(debug_directory, '%06d.%s.pointcloud.analysis.png' % (iteration, mode)), pc_vis)
                
                # 高度图生成调试
                save_path = os.path.join(debug_directory, '%06d.%s.heightmap.generation.png' % (iteration, mode))
                hm_debug_vis, _ = debug_heightmap_generation(points, workspace_limits, heightmap_resolution, save_path)
                
                if hm_debug_vis is not None:
                    # 如果返回的是RGBA图像，转换为RGB
                    if hm_debug_vis.shape[2] == 4:
                        hm_debug_vis = cv2.cvtColor(hm_debug_vis, cv2.COLOR_RGBA2BGR)
                    elif hm_debug_vis.shape[2] == 3:
                        hm_debug_vis = cv2.cvtColor(hm_debug_vis, cv2.COLOR_RGB2BGR)
                    
                    # 保存高度图生成调试结果
                    cv2.imwrite(os.path.join(debug_directory, '%06d.%s.heightmap.generation.png' % (iteration, mode)), hm_debug_vis)
            
            print(f"✅ Debug visualization saved to: {debug_directory}")
            
        except Exception as e:
            print(f"Error creating debug visualization: {str(e)}")
            import traceback
            traceback.print_exc()

