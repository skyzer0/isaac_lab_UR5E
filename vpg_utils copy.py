#!/usr/bin/env python

"""
工具函数库，用于RGB-D图像处理和高度图生成，
支持视觉推拉抓取策略的基础功能。
"""

import numpy as np
import cv2
import torch
import os
import time
from scipy.spatial.transform import Rotation as R
import torch.nn as nn
import torch.nn.functional as F
import sys

def save_pointcloud_to_ply(points, colors, filepath):
    """将点云数据保存为PLY格式文件
    
    Args:
        points: 点云坐标数组，形状为(N, 3)
        colors: 点云颜色数组，形状为(N, 3)，值范围为[0,1]
        filepath: 保存文件路径
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # 转换颜色值从[0,1]到[0,255]
    colors_uint8 = (colors * 255).astype(np.uint8)
    
    # 打开文件
    with open(filepath, 'w') as f:
        # 写入PLY头部
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {len(points)}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('end_header\n')
        
        # 写入点云数据
        for i in range(len(points)):
            x, y, z = points[i]
            r, g, b = colors_uint8[i]
            f.write(f'{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n')
    
    print(f"点云已保存为PLY格式: {filepath}")
    print(f"点云包含 {len(points)} 个点")

def create_pointcloud_from_depth(intrinsic_matrix, depth_img, camera_pose, workspace_limits, device="cpu", rgb_img=None):
    """从深度图和RGB图像生成点云，返回点云坐标和颜色
    
    Args:
        intrinsic_matrix: 相机内参矩阵
        depth_img: 深度图像
        camera_pose: 相机位姿 (4x4变换矩阵)
        workspace_limits: 工作空间边界
        device: 计算设备
        rgb_img: RGB图像，用于提取点云颜色
        
    Returns:
        cloud_world: 世界坐标系下的点云坐标
        colors: 点云颜色数组，值范围为[0,1]，如果没有提供RGB图像则返回None
    """
    # 从相机位姿矩阵中提取位置和旋转
    camera_pos = camera_pose[:3, 3]
    rot_matrix = camera_pose[:3, :3]
    
    # 获取深度图像大小
    img_h, img_w = depth_img.shape
    
    # 创建像素网格
    pix_x, pix_y = np.meshgrid(np.arange(img_w), np.arange(img_h))
    
    # 提取相机内参
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    
    # 计算相机坐标系下的点云
    z = depth_img
    x = (pix_x - cx) * z / fx
    y = (pix_y - cy) * z / fy
    
    # 创建点云数组
    cloud_cam = np.zeros((img_h, img_w, 3))
    cloud_cam[..., 0] = x
    cloud_cam[..., 1] = y
    cloud_cam[..., 2] = z
    
    # 重塑为(N, 3)
    cloud_cam = cloud_cam.reshape(-1, 3)
    
    # 准备颜色数组
    colors = None
    if rgb_img is not None:
        # 提取RGB图像颜色并归一化到[0,1]
        colors = rgb_img.reshape(-1, 3).astype(float) / 255.0
    
    # 过滤无效点
    valid_mask = ~np.isnan(cloud_cam[:, 2]) & (cloud_cam[:, 2] > 0)
    cloud_cam = cloud_cam[valid_mask]
    
    # 同时过滤颜色数组
    if colors is not None:
        colors = colors[valid_mask]
    
    # 注意：以下是关键的坐标系转换部分
    # 相机坐标系转换为世界坐标系需要一系列的旋转操作
    # 直接使用相机的旋转矩阵是不够的，因为存在坐标系约定差异
    
    # 1. 创建X轴旋转90度的矩阵 - 将Y轴指向后方，Z轴指向上方
    angle_x = np.pi/2  # 90度（弧度）
    rot_x = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(angle_x), -np.sin(angle_x)],
        [0.0, np.sin(angle_x), np.cos(angle_x)]
    ])
    
    # 2. 创建Y轴旋转180度的矩阵 - 将X轴指向后方，Z轴指向下方
    angle_y = np.pi  # 180度（弧度）
    rot_y = np.array([
        [np.cos(angle_y), 0.0, np.sin(angle_y)],
        [0.0, 1.0, 0.0],
        [-np.sin(angle_y), 0.0, np.cos(angle_y)]
    ])
    
    # 3. 创建Z轴旋转-90度的矩阵 - 将X轴指向右侧，Y轴指向后方
    angle_z = -np.pi/2  # -90度（弧度）
    rot_z = np.array([
        [np.cos(angle_z), -np.sin(angle_z), 0.0],
        [np.sin(angle_z), np.cos(angle_z), 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    # 4. 组合旋转矩阵 (Z * Y * X) - 变换顺序很重要
    combined_rot = np.matmul(rot_z, np.matmul(rot_y, rot_x))
    
    # 5. 获取相机旋转矩阵 (已经有了)
    camera_rot_matrix = rot_matrix
    
    # 6. 应用组合旋转和相机旋转 - 先应用坐标系变换，再应用相机旋转
    final_rot_matrix = np.matmul(camera_rot_matrix, combined_rot)
    
    # 7. 应用旋转和平移 - 将点从相机坐标系转换到世界坐标系
    cloud_world = np.matmul(cloud_cam, final_rot_matrix.T) + camera_pos
    
    # 8. 应用额外的Y轴翻转以修正上下颠倒问题
    # 这是因为在相机和世界坐标系之间可能存在Y轴方向的差异
    # 计算Y轴范围
    y_min, y_max = np.min(cloud_world[:, 1]), np.max(cloud_world[:, 1])
    # 计算Y轴中点
    y_center = (y_min + y_max) / 2
    # 绕Y轴中点进行翻转 - 保持点云的整体位置不变
    cloud_world[:, 1] = 2 * y_center - cloud_world[:, 1]
    
    return cloud_world, colors

def generate_heightmaps(color_img, depth_img, camera_intrinsics, camera_pose, workspace_limits, heightmap_resolution, device="cpu", save_ply=False, output_dir=None, timestamp=None):
    """从RGB-D图像生成高度图
    
    Args:
        color_img: RGB图像
        depth_img: 深度图像
        camera_intrinsics: 相机内参矩阵
        camera_pose: 相机位姿 (4x4变换矩阵)
        workspace_limits: 工作空间边界 [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        heightmap_resolution: 高度图分辨率 (米/像素)
        device: 计算设备
        save_ply: 是否保存点云为PLY格式
        output_dir: 输出目录
        timestamp: 时间戳，用于文件命名
        
    Returns:
        color_heightmap: 颜色高度图
        depth_heightmap: 深度高度图
    """
    # 打印相机位姿矩阵
    print("\n==== 相机位姿矩阵 ====")
    print(camera_pose)
    
    # 从相机位姿中提取位置和方向
    camera_pos = camera_pose[:3, 3]
    
    # 从旋转矩阵中提取四元数
    rot_matrix = camera_pose[:3, :3]
    r = R.from_matrix(rot_matrix)
    orientation = r.as_quat()  # [x, y, z, w]
    
    # 转换为[w, x, y, z]格式
    orientation = np.array([orientation[3], orientation[0], orientation[1], orientation[2]])
    
    # 打印相机外参信息
    print(f"相机位置: {camera_pos}")
    print(f"相机朝向(四元数[w,x,y,z]): {orientation}")
    
    # 检查工作空间与相机的相对关系
    print("\n工作空间范围检查:")
    print(f"  X: {workspace_limits[0][0]:.4f} 到 {workspace_limits[0][1]:.4f}")
    print(f"  Y: {workspace_limits[1][0]:.4f} 到 {workspace_limits[1][1]:.4f}")
    print(f"  Z: {workspace_limits[2][0]:.4f} 到 {workspace_limits[2][1]:.4f}")
    
    print(f"\n相机位置相对于工作空间:")
    print(f"  X方向: {'左侧' if camera_pos[0] < workspace_limits[0][0] else '右侧' if camera_pos[0] > workspace_limits[0][1] else '内部'}")
    print(f"  Y方向: {'前方' if camera_pos[1] < workspace_limits[1][0] else '后方' if camera_pos[1] > workspace_limits[1][1] else '内部'}")
    print(f"  Z方向: {'下方' if camera_pos[2] < workspace_limits[2][0] else '上方' if camera_pos[2] > workspace_limits[2][1] else '内部'}")
    
    # 生成点云
    points, colors = create_pointcloud_from_depth(
        camera_intrinsics, 
        depth_img, 
        camera_pose,
        workspace_limits,
        device=device,
        rgb_img=color_img
    )
    
    # 检查是否成功生成点云和颜色
    if points.shape[0] == 0:
        print("警告: 生成的点云为空，返回空高度图")
        heightmap_height = int(np.round((workspace_limits[1][1] - workspace_limits[1][0])/heightmap_resolution))
        heightmap_width = int(np.round((workspace_limits[0][1] - workspace_limits[0][0])/heightmap_resolution))
        return np.zeros((heightmap_height, heightmap_width, 3), dtype=np.uint8), np.zeros((heightmap_height, heightmap_width))
    
    # 如果颜色信息缺失，创建默认颜色（灰色）
    if colors is None:
        print("警告: 未提供RGB图像或无法提取颜色信息，使用默认灰色")
        colors = np.ones((points.shape[0], 3)) * 0.5  # 默认灰色
    
    # 保存原始点云为PLY格式（如果需要）
    if save_ply and output_dir is not None:
        # 创建点云保存目录
        ply_dir = os.path.join(output_dir, "pointcloud")
        os.makedirs(ply_dir, exist_ok=True)
        
        # 生成文件名
        ts = timestamp if timestamp is not None else int(time.time())
        ply_filepath = os.path.join(ply_dir, f"pointcloud_original_{ts}.ply")
        
        # 保存原始点云
        save_pointcloud_to_ply(points, colors, ply_filepath)
    
    # 保存原始点云副本
    points_orig = points.copy()
    colors_orig = colors.copy()
    
    # 过滤工作空间外的点
    print(f"\n应用Z轴范围过滤前的点数: {len(points)}")
    print(f"点云Z轴范围: {np.min(points[:, 2]):.4f} 到 {np.max(points[:, 2]):.4f}")
    print(f"工作空间Z轴范围: {workspace_limits[2][0]:.4f} 到 {workspace_limits[2][1]:.4f}")
    
    # 检查Z轴是否超出工作空间
    all_points_above = np.min(points[:, 2]) > workspace_limits[2][1]
    all_points_below = np.max(points[:, 2]) < workspace_limits[2][0]
    
    # 如果所有点都在工作空间Z轴范围外，需要进行调整
    if all_points_above or all_points_below:
        print("警告: 所有点的Z值都超出工作空间范围，尝试调整点云Z值...")
        
        # 计算需要的Z轴偏移量
        if all_points_above:
            z_offset = np.min(points[:, 2]) - workspace_limits[2][0]
            points[:, 2] -= z_offset
            print(f"应用Z轴向下偏移: {z_offset:.4f}米")
        elif all_points_below:
            z_offset = workspace_limits[2][1] - np.max(points[:, 2])
            points[:, 2] += z_offset
            print(f"应用Z轴向上偏移: {z_offset:.4f}米")
        
        # 另一种方法: 缩放点云Z值到工作空间范围
        z_min = np.min(points[:, 2])
        z_max = np.max(points[:, 2])
        
        # 防止除以零
        if z_max > z_min:
            z_scale = (workspace_limits[2][1] - workspace_limits[2][0]) / (z_max - z_min)
            
            # 应用缩放
            points[:, 2] = (points[:, 2] - z_min) * z_scale + workspace_limits[2][0]
            print(f"缩放点云Z值: {z_scale:.4f}")
            print(f"调整后点云Z轴范围: {np.min(points[:, 2]):.4f} 到 {np.max(points[:, 2]):.4f}")
    else:
        # 即使点云Z值在工作空间范围内，也进行一定的缩放以确保更好地适应工作空间
        z_min = np.min(points[:, 2])
        z_max = np.max(points[:, 2])
        
        # 只有当点云有一定厚度时才进行缩放
        if z_max - z_min > 0.005:  # 5mm以上的厚度
            # 将点云Z范围映射到工作空间Z范围的80%，保留一些余量
            z_scale = 0.8 * (workspace_limits[2][1] - workspace_limits[2][0]) / (z_max - z_min)
            z_offset = workspace_limits[2][0] + 0.1 * (workspace_limits[2][1] - workspace_limits[2][0])
            
            # 应用缩放和偏移
            points[:, 2] = (points[:, 2] - z_min) * z_scale + z_offset
            print(f"优化点云Z分布: 缩放因子 {z_scale:.4f}, 基础偏移 {z_offset:.4f}米")
            print(f"调整后点云Z轴范围: {np.min(points[:, 2]):.4f} 到 {np.max(points[:, 2]):.4f}")
    
    # 过滤工作空间外的点
    valid_pts_mask = np.logical_and.reduce((
        points[:, 0] >= workspace_limits[0][0],
        points[:, 0] < workspace_limits[0][1],
        points[:, 1] >= workspace_limits[1][0],
        points[:, 1] < workspace_limits[1][1],
        points[:, 2] >= workspace_limits[2][0],
        points[:, 2] < workspace_limits[2][1]
    ))
    
    print(f"有效点数: {np.sum(valid_pts_mask)}")
    
    # 应用过滤
    points = points[valid_pts_mask]
    colors = colors[valid_pts_mask]
    
    print(f"应用Z轴范围过滤后的点数: {len(points)}")
    
    # 保存过滤后的点云为PLY格式（如果需要）
    if save_ply and output_dir is not None and len(points) > 0:
        ply_dir = os.path.join(output_dir, "pointcloud")
        ts = timestamp if timestamp is not None else int(time.time())
        ply_filtered_filepath = os.path.join(ply_dir, f"pointcloud_filtered_{ts}.ply")
        save_pointcloud_to_ply(points, colors, ply_filtered_filepath)
    
    # 如果没有点，尝试更宽松的方法
    if len(points) == 0:
        print("警告: 过滤后没有有效点，尝试使用未过滤的点...")
        # 回退到使用原始点云，但强制Z值裁剪到工作空间范围
        points = points_orig.copy()
        colors = colors_orig.copy()
        # 裁剪Z值
        points[:, 2] = np.clip(points[:, 2], workspace_limits[2][0], workspace_limits[2][1])
        print(f"裁剪后点云Z轴范围: {np.min(points[:, 2]):.4f} 到 {np.max(points[:, 2]):.4f}")
        
        # 再次过滤XY方向上的点
        valid_pts_mask = np.logical_and.reduce((
            points[:, 0] >= workspace_limits[0][0],
            points[:, 0] < workspace_limits[0][1],
            points[:, 1] >= workspace_limits[1][0],
            points[:, 1] < workspace_limits[1][1]
        ))
        
        # 应用过滤
        points = points[valid_pts_mask]
        colors = colors[valid_pts_mask]
        
        print(f"XY方向过滤后的点数: {len(points)}")
        
        # 保存裁剪后的点云
        if save_ply and output_dir is not None and len(points) > 0:
            ply_dir = os.path.join(output_dir, "pointcloud")
            ts = timestamp if timestamp is not None else int(time.time())
            ply_clipped_filepath = os.path.join(ply_dir, f"pointcloud_clipped_{ts}.ply")
            save_pointcloud_to_ply(points, colors, ply_clipped_filepath)
    
    # 如果仍然没有点，返回空的高度图
    if len(points) == 0:
        print("警告: 仍然没有有效点，返回空的高度图")
        heightmap_height = int(np.round((workspace_limits[1][1] - workspace_limits[1][0])/heightmap_resolution))
        heightmap_width = int(np.round((workspace_limits[0][1] - workspace_limits[0][0])/heightmap_resolution))
        return np.zeros((heightmap_height, heightmap_width, 3), dtype=np.uint8), np.zeros((heightmap_height, heightmap_width))
    
    # 计算高度图尺寸
    heightmap_height = int(np.round((workspace_limits[1][1] - workspace_limits[1][0])/heightmap_resolution))
    heightmap_width = int(np.round((workspace_limits[0][1] - workspace_limits[0][0])/heightmap_resolution))
    
    print(f"\n=== 高度图信息 ===")
    print(f"工作空间边界: {workspace_limits}")
    print(f"高度图分辨率: {heightmap_resolution} 米/像素")
    print(f"高度图大小: 高度={heightmap_height}像素, 宽度={heightmap_width}像素")
    
    # 使用向量化的world_to_pixel函数计算像素坐标，提高效率
    heightmap_pix_x, heightmap_pix_y = world_to_pixel(points[:, 0], points[:, 1], workspace_limits, heightmap_resolution)
    
    # 筛选在高度图范围内的点
    in_bounds = np.logical_and.reduce((
        heightmap_pix_x >= 0,
        heightmap_pix_x < heightmap_width,
        heightmap_pix_y >= 0,
        heightmap_pix_y < heightmap_height
    ))
    
    # 统计有效点数量
    valid_points_count = np.sum(in_bounds)
    print(f"\n有效点数量: {valid_points_count}/{len(points)} ({valid_points_count/len(points)*100:.2f}%)")
    
    if valid_points_count == 0:
        print("警告: 没有有效点落在高度图范围内，返回空高度图")
        return np.zeros((heightmap_height, heightmap_width, 3), dtype=np.uint8), np.zeros((heightmap_height, heightmap_width))
    
    # 应用过滤
    heightmap_pix_x = heightmap_pix_x[in_bounds]
    heightmap_pix_y = heightmap_pix_y[in_bounds]
    points_z = points[in_bounds, 2]
    colors_rgb = colors[in_bounds] * 255.0
    
    # 初始化高度图
    color_heightmap = np.zeros((heightmap_height, heightmap_width, 3), dtype=np.uint8)
    depth_heightmap = np.zeros((heightmap_height, heightmap_width))
    heightmap_points_mask = np.zeros((heightmap_height, heightmap_width))
    
    print("\n开始生成高度图...")
    
    # 记录原始的深度范围
    z_min_orig = np.min(points_z)
    z_max_orig = np.max(points_z)
    
    # 按高度值排序点云（从上往下）
    # 因为我们已经翻转了Z轴，所以较大的Z值代表物体表面
    # 保持-points_z排序，确保顶部物体优先显示
    sort_indices = np.argsort(-points_z)
    heightmap_pix_x = heightmap_pix_x[sort_indices]
    heightmap_pix_y = heightmap_pix_y[sort_indices]
    points_z = points_z[sort_indices]
    colors_rgb = colors_rgb[sort_indices]
    
    # 填充高度图（高处的点优先，即采用最大高度值）
    for i in range(len(heightmap_pix_x)):
        # 如果该像素位置还没有点或当前点更高
        if heightmap_points_mask[heightmap_pix_y[i], heightmap_pix_x[i]] == 0:
            color_heightmap[heightmap_pix_y[i], heightmap_pix_x[i], :] = colors_rgb[i].astype(np.uint8)
            depth_heightmap[heightmap_pix_y[i], heightmap_pix_x[i]] = points_z[i]
            heightmap_points_mask[heightmap_pix_y[i], heightmap_pix_x[i]] = 1
    
    # 计算相对高度
    z_bottom = workspace_limits[2][0]
    depth_heightmap = depth_heightmap - z_bottom
    
    # 重置背景区域的值为0（背景区域在减去z_bottom后变为-z_bottom）
    depth_heightmap[depth_heightmap == -z_bottom] = 0
    
    # 过滤极小值，可能是噪声
    # 将阈值从0.0005降低到0.0003，提高对小物体的敏感度
    depth_heightmap[depth_heightmap < 0.0003] = 0  # 将小于0.3mm的高度视为噪声，设为0
    
    # 可选：应用形态学操作增强边缘
    if np.max(depth_heightmap) > 0:
        # 创建二值掩码图像用于形态学处理
        binary_mask = np.zeros_like(depth_heightmap, dtype=np.uint8)
        binary_mask[depth_heightmap > 0] = 1
        
        # 定义形态学处理的内核
        kernel = np.ones((3, 3), np.uint8)
        
        # 应用开运算去除噪点
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        
        # 应用闭运算填充小空洞
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        
        # 使用形态学处理后的掩码过滤原始深度图
        filtered_depth = np.zeros_like(depth_heightmap)
        filtered_depth[binary_mask > 0] = depth_heightmap[binary_mask > 0]
        depth_heightmap = filtered_depth
    
    # 确保深度值是正数，值越大表示越高
    if np.min(depth_heightmap[depth_heightmap > 0]) < 0:
        depth_heightmap = depth_heightmap * -1
        print("应用深度值反转以确保正确的高度表示")
    
    # 处理并可视化深度范围
    z_min = np.min(depth_heightmap[depth_heightmap > 0]) if np.any(depth_heightmap > 0) else 0
    z_max = np.max(depth_heightmap)
    print(f"深度范围调整: 原始 [{z_min_orig:.4f}, {z_max_orig:.4f}], 过滤后 [{z_min:.4f}, {z_max:.4f}]")
    
    # 可选：压缩深度范围，使高度更合理
    if z_max > 0.05:  # 如果最大高度超过5cm，可能需要压缩
        print(f"高度范围过大，执行压缩处理，原始最大高度: {z_max:.4f}m")
        # 只修改非零值
        non_zero_mask = depth_heightmap > 0
        if np.any(non_zero_mask):
            # 将非零高度值压缩到合理范围（最大值不超过3cm）
            depth_heightmap[non_zero_mask] = depth_heightmap[non_zero_mask] * min(1.0, 0.03 / z_max)
            print(f"压缩后最大高度: {np.max(depth_heightmap):.4f}m")
    
    # 注意：我们已经在create_pointcloud_from_depth函数中对点云坐标进行了Y轴翻转
    # 但高度图仍然需要翻转，因为从3D到2D的映射仍有方向问题
    color_heightmap = cv2.flip(color_heightmap, 0)  # 上下翻转
    depth_heightmap = cv2.flip(depth_heightmap, 0)  # 上下翻转
    print("应用最终图像翻转以确保高度图方向正确")
    
    print("\n高度图生成完成!")
    print(f"颜色高度图尺寸: {color_heightmap.shape}")
    print(f"深度高度图尺寸: {depth_heightmap.shape}")
    print(f"深度范围: {np.min(depth_heightmap)} 到 {np.max(depth_heightmap)}")
    
    # 验证坐标转换的一致性（可选）
    if '--debug' in sys.argv or any(arg == 'debug=true' for arg in sys.argv):
        test_coordinate_conversion(workspace_limits, heightmap_resolution)
    
    return color_heightmap, depth_heightmap

def get_affordance_vis(grasp_affordance, push_affordance=None, scale_factor=1):
    """生成抓取和推動可行性的可視化圖像
    
    Args:
        grasp_affordance: 抓取可行性地圖，形狀為(H,W)或(num_rotations,H,W)
        push_affordance: 推動可行性地圖，形狀為(H,W)或(num_rotations,H,W)
        scale_factor: 缩放因子
        
    Returns:
        affordance_vis: 可视化图像
    """
    # 判断输入数据类型
    if isinstance(grasp_affordance, torch.Tensor):
        grasp_affordance = grasp_affordance.cpu().numpy()
    
    if push_affordance is not None and isinstance(push_affordance, torch.Tensor):
        push_affordance = push_affordance.cpu().numpy()
    
    # 处理3D形状的affordance maps (num_rotations, H, W)
    if len(grasp_affordance.shape) == 3:
        # 使用最大值投影，获取每个像素位置最佳的可行性分数
        grasp_affordance = np.max(grasp_affordance, axis=0)
    
    if push_affordance is not None and len(push_affordance.shape) == 3:
        push_affordance = np.max(push_affordance, axis=0)
    
    # 创建空白图像
    if push_affordance is not None:
        affordance_vis = np.zeros((grasp_affordance.shape[0], grasp_affordance.shape[1], 3), dtype=np.uint8)
        
        # 归一化
        push_affordance_normalized = push_affordance.copy()
        grasp_affordance_normalized = grasp_affordance.copy()
        if np.max(push_affordance) > 0:
            push_affordance_normalized = push_affordance_normalized / np.max(push_affordance)
        if np.max(grasp_affordance) > 0:
            grasp_affordance_normalized = grasp_affordance_normalized / np.max(grasp_affordance)
        
        # 设置颜色：红色表示推动，绿色表示抓取
        affordance_vis[:,:,0] = (push_affordance_normalized * 255).astype(np.uint8)
        affordance_vis[:,:,1] = (grasp_affordance_normalized * 255).astype(np.uint8)
    else:
        affordance_vis = np.zeros((grasp_affordance.shape[0], grasp_affordance.shape[1], 3), dtype=np.uint8)
        
        # 归一化
        grasp_affordance_normalized = grasp_affordance.copy()
        if np.max(grasp_affordance) > 0:
            grasp_affordance_normalized = grasp_affordance_normalized / np.max(grasp_affordance)
        
        # 设置颜色：绿色表示抓取
        affordance_vis[:,:,1] = (grasp_affordance_normalized * 255).astype(np.uint8)
    
    # 缩放
    if scale_factor != 1:
        affordance_vis = cv2.resize(affordance_vis, 
                                   (int(grasp_affordance.shape[1] * scale_factor), 
                                    int(grasp_affordance.shape[0] * scale_factor)))
    
    return affordance_vis

def get_action_visualization(color_heightmap, depth_heightmap, action_mode, pixel_x, pixel_y, rotation_angle, scale_factor=1, flip_x=True, flip_y=True):
    """在高度图上可视化推动或抓取动作
    
    Args:
        color_heightmap: 颜色高度图
        depth_heightmap: 深度高度图
        action_mode: 0-推动，1-抓取
        pixel_x: 动作在X方向的像素位置
        pixel_y: 动作在Y方向的像素位置
        rotation_angle: 旋转角度（弧度）
        scale_factor: 缩放因子
        flip_x: 是否在X轴方向上翻转角度 (默认True，与主程序保持一致)
        flip_y: 是否在Y轴方向上翻转角度 (默认True，与主程序保持一致)
        
    Returns:
        vis_img: 可视化图像
    """
    vis_img = color_heightmap.copy()
    
    # 绘制动作
    center = (int(pixel_x), int(pixel_y))
    center_x, center_y = center  # 解包元组，避免使用索引
    
    # 调整角度，考虑坐标系翻转
    adjusted_angle = rotation_angle
    
    # X轴翻转时，角度变为 pi - angle
    if flip_x:
        adjusted_angle = np.pi - adjusted_angle
    
    # Y轴翻转时，角度变为 -angle
    if flip_y:
        adjusted_angle = -adjusted_angle
    
    # 标准化角度到 [0, 2*pi)
    adjusted_angle = adjusted_angle % (2 * np.pi)
    
    if action_mode == 0:  # Push
        # 计算推动起点和终点 - 使用调整后的角度
        push_length = 20  # 像素
        start_point = (
            int(pixel_x - push_length * np.cos(adjusted_angle)),
            int(pixel_y - push_length * np.sin(adjusted_angle))
        )
        end_point = (
            int(pixel_x + push_length * np.cos(adjusted_angle)),
            int(pixel_y + push_length * np.sin(adjusted_angle))
        )
        
        # 绘制推动方向箭头
        cv2.arrowedLine(vis_img, start_point, end_point, (255, 0, 0), 2)
        cv2.putText(vis_img, "PUSH", (center_x+5, center_y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # 添加角度标记，以便调试
        angle_deg = np.rad2deg(rotation_angle)
        adjusted_angle_deg = np.rad2deg(adjusted_angle)
        cv2.putText(vis_img, f"{angle_deg:.1f}°(世) {adjusted_angle_deg:.1f}°(图)", (center_x+5, center_y+15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
        
    else:  # Grasp
        # 计算夹爪方向 - 使用调整后的角度
        gripper_length = 15  # 像素
        # 对于抓取，夹爪是垂直于旋转方向的
        dx = int(gripper_length * np.cos(adjusted_angle))
        dy = int(gripper_length * np.sin(adjusted_angle))
        
        # 绘制夹爪 - 垂直于抓取方向
        cv2.line(vis_img, 
                (center_x - dy, center_y + dx), 
                (center_x + dy, center_y - dx), 
                (0, 255, 0), 2)
        
        # 绘制抓取方向 - 添加一个小箭头显示抓取方向
        arrow_length = 10  # 像素
        arrow_start = center
        arrow_end = (
            int(center_x + arrow_length * np.cos(adjusted_angle)),
            int(center_y + arrow_length * np.sin(adjusted_angle))
        )
        cv2.arrowedLine(vis_img, arrow_start, arrow_end, (0, 255, 0), 1)
        
        # 绘制夹爪中心点
        cv2.circle(vis_img, center, 5, (0, 255, 0), -1)
        cv2.putText(vis_img, "GRASP", (center_x+5, center_y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 添加角度标记，以便调试
        angle_deg = np.rad2deg(rotation_angle)
        adjusted_angle_deg = np.rad2deg(adjusted_angle)
        cv2.putText(vis_img, f"{angle_deg:.1f}°(世) {adjusted_angle_deg:.1f}°(图)", (center_x+5, center_y+15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
    
    # 缩放
    if scale_factor != 1:
        vis_img = cv2.resize(vis_img, 
                            (int(vis_img.shape[1] * scale_factor), 
                             int(vis_img.shape[0] * scale_factor)))
    
    return vis_img

class CrossEntropyLoss2d(nn.Module):
    """二维交叉熵损失函数"""

    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        # 使用现代化的API替代已废弃的nn.NLLLoss2d
        self.nll_loss = nn.NLLLoss(weight, size_average=size_average, reduction='mean' if size_average else 'none')

    def forward(self, inputs, targets):
        """前向传播
        
        Args:
            inputs: 预测输出 (N, C, H, W)
            targets: 目标标签 (N, H, W)
            
        Returns:
            loss: 损失值
        """
        # 应用log_softmax到通道维度，然后重塑为NLL损失所需的形状
        log_p = F.log_softmax(inputs, dim=1)
        # 调整目标形状到(N, H*W)
        N, C, H, W = log_p.size()
        log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
        log_p = log_p.view(-1, C)
        target = targets.view(-1)
        return self.nll_loss(log_p, target)


# 定义动作模式类型
class ActionMode:
    """Action modes for VPG."""
    PUSH = 0
    GRASP = 1


# 基于规则的推动与抓取策略
class RuleBasedPolicy:
    """基于规则的推动与抓取策略"""
    
    def __init__(self, workspace_limits, heightmap_resolution):
        """初始化策略
        
        Args:
            workspace_limits: 工作空间边界 [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
            heightmap_resolution: 高度图分辨率(米/像素)
        """
        self.workspace_limits = workspace_limits
        self.heightmap_resolution = heightmap_resolution
        self.push_step_size = 0.05  # 米
        
    def get_action(self, color_heightmap, depth_heightmap):
        """根据当前高度图获取下一步动作
        
        Args:
            color_heightmap: 颜色高度图 (H, W, 3)
            depth_heightmap: 深度高度图 (H, W)
            
        Returns:
            action_mode: 推动(0)或抓取(1)
            best_pix_ind: 动作的最佳像素位置 (y, x)
            best_rotation_angle: 动作的最佳旋转角度(弧度)
        """
        # 创建深度高度图副本
        depth_heightmap_copy = depth_heightmap.copy()
        
        # 过滤NaN值
        depth_heightmap_copy[np.isnan(depth_heightmap_copy)] = 0
        
        # 计算深度图的梯度
        Gx = cv2.Sobel(depth_heightmap_copy, cv2.CV_64F, 1, 0, ksize=5)
        Gy = cv2.Sobel(depth_heightmap_copy, cv2.CV_64F, 0, 1, ksize=5)
        
        # 计算边缘幅度
        edge_magnitude = np.sqrt(Gx**2 + Gy**2)
        
        # 计算边缘方向
        edge_orientation = np.arctan2(Gy, Gx)
        
        # 应用阈值查找边缘
        edge_mask = edge_magnitude > 0.01  # 阈值可调整
        
        # 沿边缘寻找抓取点
        grasp_candidates = np.zeros_like(depth_heightmap_copy)
        
        # 简单规则: 尝试在边缘幅度高的点抓取
        grasp_candidates[edge_mask] = edge_magnitude[edge_mask]
        
        # 应用距离变换查找远离边缘的点用于推动
        dist_transform = cv2.distanceTransform((~edge_mask).astype(np.uint8), cv2.DIST_L2, 5)
        push_candidates = dist_transform.copy()
        
        # 归一化候选项
        if np.max(grasp_candidates) > 0:
            grasp_candidates = grasp_candidates / np.max(grasp_candidates)
        if np.max(push_candidates) > 0:
            push_candidates = push_candidates / np.max(push_candidates)
        
        # 决定是推动还是抓取
        best_grasp_score = np.max(grasp_candidates) if np.max(grasp_candidates) > 0 else 0
        best_push_score = np.max(push_candidates) if np.max(push_candidates) > 0 else 0
        
        # 简单启发式规则: 如果有好的抓取点就抓取，否则推动
        if best_grasp_score > 0.5 and best_grasp_score > best_push_score:
            # 抓取
            action_mode = ActionMode.GRASP
            best_pix_ind = np.unravel_index(np.argmax(grasp_candidates), grasp_candidates.shape)
            best_rotation_angle = edge_orientation[best_pix_ind]
        else:
            # 推动
            action_mode = ActionMode.PUSH
            best_pix_ind = np.unravel_index(np.argmax(push_candidates), push_candidates.shape)
            
            # 为推动选择随机方向
            best_rotation_angle = np.random.uniform(0, np.pi)
        
        # 如果是调试模式，则可视化可行性图
        debug_mode = '--debug' in sys.argv or any(arg == 'debug=true' for arg in sys.argv)
        if debug_mode:
            affordance_vis = get_affordance_vis(grasp_candidates, push_candidates)
            action_vis = get_action_visualization(
                color_heightmap, depth_heightmap, 
                action_mode, best_pix_ind[1], best_pix_ind[0], 
                best_rotation_angle,
                flip_x=True,  # 与主程序保持一致
                flip_y=True   # 与主程序保持一致
            )
            
            # 保存可视化结果
            os.makedirs(os.path.join("output", "VPG", "viz"), exist_ok=True)
            timestamp = int(time.time())
            cv2.imwrite(os.path.join("output", "VPG", "viz", f"affordance_{timestamp}.png"), affordance_vis)
            cv2.imwrite(os.path.join("output", "VPG", "viz", f"action_{timestamp}.png"), action_vis)
        
        return action_mode, best_pix_ind, best_rotation_angle

def pixel_to_world(pixel_x, pixel_y, z, workspace_limits, heightmap_resolution, heightmap_shape):
    """将像素坐标转换为世界坐标，考虑Y轴翻转
    
    Args:
        pixel_x: 像素X坐标
        pixel_y: 像素Y坐标
        z: 高度值
        workspace_limits: 工作空间边界 [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        heightmap_resolution: 高度图分辨率(米/像素)
        heightmap_shape: 高度图的形状(高度,宽度)
        
    Returns:
        world_x, world_y, world_z: 世界坐标
    """
    world_x = pixel_x * heightmap_resolution + workspace_limits[0][0]
    # 考虑Y轴翻转 - 图像坐标系中Y向下，世界坐标系中Y可能向上或有不同方向
    world_y = (heightmap_shape[0] - 1 - pixel_y) * heightmap_resolution + workspace_limits[1][0]
    world_z = z + workspace_limits[2][0]  # z一般是相对于工作空间底部的高度
    
    return world_x, world_y, world_z


def world_to_pixel(world_x, world_y, workspace_limits, heightmap_resolution):
    """将世界坐标转换为像素坐标
    
    Args:
        world_x: 世界X坐标，可以是标量或数组
        world_y: 世界Y坐标，可以是标量或数组
        workspace_limits: 工作空间边界 [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        heightmap_resolution: 高度图分辨率(米/像素)
        
    Returns:
        tuple: (pixel_x, pixel_y) 像素坐标，如果输入是数组则输出也是数组
    """
    # 计算像素坐标
    if isinstance(world_x, np.ndarray):
        # 数组输入，使用向量化操作
        pixel_x = np.floor((world_x - workspace_limits[0][0]) / heightmap_resolution).astype(int)
        pixel_y = np.floor((world_y - workspace_limits[1][0]) / heightmap_resolution).astype(int)
    else:
        # 标量输入
        pixel_x = int(np.floor((world_x - workspace_limits[0][0]) / heightmap_resolution))
        pixel_y = int(np.floor((world_y - workspace_limits[1][0]) / heightmap_resolution))
    
    return pixel_x, pixel_y


def test_coordinate_conversion(workspace_limits, heightmap_resolution):
    """测试坐标转换的一致性
    
    Args:
        workspace_limits: 工作空间边界
        heightmap_resolution: 高度图分辨率
    
    Returns:
        是否通过测试
    """
    # 测试数据
    test_points = [
        # 工作空间中心
        (
            (workspace_limits[0][0] + workspace_limits[0][1]) / 2,
            (workspace_limits[1][0] + workspace_limits[1][1]) / 2,
            0.05  # 高度
        ),
        # 工作空间边缘
        (workspace_limits[0][0], workspace_limits[1][0], 0),
        (workspace_limits[0][1], workspace_limits[1][1], 0.1)
    ]
    
    # 假设的高度图尺寸（用于测试）
    # 计算高度图的大小（基于工作空间范围和分辨率）
    heightmap_height = int(np.round((workspace_limits[1][1] - workspace_limits[1][0])/heightmap_resolution))
    heightmap_width = int(np.round((workspace_limits[0][1] - workspace_limits[0][0])/heightmap_resolution))
    heightmap_shape = (heightmap_height, heightmap_width)
    
    passed = True
    for i, (x, y, z) in enumerate(test_points):
        # 世界坐标 -> 像素坐标
        pixel_x, pixel_y = world_to_pixel(x, y, workspace_limits, heightmap_resolution)
        
        # 像素坐标 -> 世界坐标
        world_x, world_y, _ = pixel_to_world(pixel_x, pixel_y, z, workspace_limits, heightmap_resolution, heightmap_shape)
        
        # 计算误差（考虑到取整导致的误差）
        error_x = abs(world_x - x)
        error_y = abs(world_y - y)
        max_allowed_error = heightmap_resolution  # 最大允许误差为一个像素
        
        if error_x > max_allowed_error or error_y > max_allowed_error:
            passed = False
            print(f"测试点 {i+1} 转换不一致:")
            print(f"  原始世界坐标: ({x:.4f}, {y:.4f}, {z:.4f})")
            print(f"  像素坐标: ({pixel_x}, {pixel_y})")
            print(f"  恢复的世界坐标: ({world_x:.4f}, {world_y:.4f})")
            print(f"  误差: ({error_x:.4f}, {error_y:.4f}), 最大允许: {max_allowed_error:.4f}")
        else:
            print(f"测试点 {i+1} 转换一致，误差在可接受范围内。")
    
    return passed