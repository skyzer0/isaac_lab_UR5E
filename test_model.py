#!/usr/bin/env python

"""
简单的测试脚本，用于验证模型在不同设备上的行为
"""

import os
import torch
import numpy as np
from models import reinforcement_net
import torch.nn.functional as F
import time

def test_model_device_consistency():
    """测试模型在不同设备间的一致性"""
    print("="*50)
    print("开始测试模型设备一致性")
    print("="*50)
    
    # 检查CUDA是否可用
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"当前设备: {device}")
    
    # 创建模型实例
    model = reinforcement_net(use_cuda=use_cuda)
    print(f"模型参数设备: {next(model.parameters()).device}")
    
    # 创建测试输入
    print("创建测试输入...")
    color_data = torch.rand(1, 3, 320, 320)  # 批次大小, 通道, 高, 宽
    depth_data = torch.rand(1, 3, 320, 320)  # 批次大小, 通道, 高, 宽
    
    # 测试不同设备输入的情况
    scenarios = []
    
    if use_cuda:
        scenarios = [
            ("CPU输入, 模型在GPU", color_data.cpu(), depth_data.cpu()),
            ("GPU输入, 模型在GPU", color_data.cuda(), depth_data.cuda()),
        ]
    else:
        scenarios = [
            ("CPU输入, 模型在CPU", color_data.cpu(), depth_data.cpu()),
        ]
    
    for name, color_input, depth_input in scenarios:
        print("\n" + "-"*30)
        print(f"测试场景: {name}")
        print(f"颜色输入设备: {color_input.device}")
        print(f"深度输入设备: {depth_input.device}")
        print(f"模型设备: {next(model.parameters()).device}")
        
        try:
            # 执行前向传播
            print("执行前向传播...")
            start_time = time.time()
            output_prob, interm_feat, state_feat = model(color_input, depth_input, is_volatile=True)
            elapsed = time.time() - start_time
            print(f"前向传播成功完成，耗时: {elapsed:.4f}秒")
            
            # 打印输出信息
            if isinstance(output_prob, list):
                for i, item in enumerate(output_prob):
                    push_output, grasp_output = item
                    print(f"旋转 {i} - 推动输出: {push_output.shape}, 设备: {push_output.device}")
                    print(f"旋转 {i} - 抓取输出: {grasp_output.shape}, 设备: {grasp_output.device}")
            else:
                print(f"输出形状: {output_prob.shape}, 设备: {output_prob.device}")
                
            print(f"中间特征数量: {len(interm_feat) if interm_feat else 'None'}")
            print(f"状态特征: {'有效' if state_feat is not None else '无'}")
            
            # 测试输出概率的访问
            print("\n检查输出概率访问...")
            if hasattr(model, 'output_prob') and len(model.output_prob) > 0:
                print(f"Model.output_prob存在，长度: {len(model.output_prob)}")
                for i, item in enumerate(model.output_prob[:2]):  # 只打印前2个用于演示
                    if isinstance(item, list) and len(item) >= 2:
                        print(f"Item {i}: Push shape: {item[0].shape}, Grasp shape: {item[1].shape}")
                    else:
                        print(f"Item {i}: 类型: {type(item)}")
            else:
                print("Model.output_prob不存在或为空")
            
            # 测试模型验证方法
            if hasattr(model, '_verify_device'):
                print("\n执行设备验证...")
                model._verify_device()
                
            print("\n测试成功!")
                
        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()
            
    print("\n" + "="*50)
    print("测试完成")
    print("="*50)

def test_rotation_mechanism():
    """测试旋转机制的正确性"""
    print("="*50)
    print("测试图像旋转机制")
    print("="*50)
    
    # 创建测试图像
    size = 16  # 小图像便于可视化
    # 创建一个简单的测试图像: 左上角是白色块
    test_img = torch.zeros(1, 3, size, size)
    test_img[0, :, 0:size//2, 0:size//2] = 1.0  # 左上角是白色
    
    # 显示原始图像
    print("原始图像:")
    img = test_img[0].permute(1, 2, 0).numpy()
    print(np.round(img[:, :, 0], 1))  # 只打印第一个通道
    
    # 创建旋转测试函数
    def apply_rotation(img, rotate_angle):
        # 计算旋转矩阵
        affine_mat = np.asarray([[np.cos(-rotate_angle), np.sin(-rotate_angle), 0],
                               [-np.sin(-rotate_angle), np.cos(-rotate_angle), 0]])
        affine_mat.shape = (2, 3, 1)
        affine_mat = torch.from_numpy(affine_mat).permute(2, 0, 1).float()
        
        # 创建流场
        flow_grid = F.affine_grid(affine_mat, list(img.size()), align_corners=True)
        
        # 应用旋转
        rotated = F.grid_sample(img, flow_grid, mode='nearest', align_corners=True)
        return rotated
    
    # 测试不同角度的旋转
    for angle_degrees in [0, 90, 180, 270]:
        angle_rad = np.radians(angle_degrees)
        rotated = apply_rotation(test_img, angle_rad)
        
        print(f"\n旋转 {angle_degrees} 度后:")
        img = rotated[0].permute(1, 2, 0).numpy()
        print(np.round(img[:, :, 0], 1))  # 只打印第一个通道

def test_backprop_on_random_data():
    """测试基于随机数据的反向传播"""
    print("="*50)
    print("测试随机数据上的反向传播")
    print("="*50)
    
    from trainer import Trainer
    
    # 检查CUDA是否可用
    use_cuda = torch.cuda.is_available()
    print(f"使用CUDA: {use_cuda}")
    
    # 创建Trainer实例
    trainer = Trainer(method='reinforcement', push_rewards=True, future_reward_discount=0.5, is_testing=False)
    
    # 创建随机输入
    color_heightmap = torch.rand(3, 320, 320)  # CHW格式
    depth_heightmap = torch.rand(320, 320)     # HW格式
    
    # 创建随机的最佳像素索引
    best_pix_ind = (np.random.randint(0, 16), np.random.randint(0, 224), np.random.randint(0, 224))
    
    # 创建随机标签值
    label_value = np.random.random()
    
    print(f"颜色高度图形状: {color_heightmap.shape}")
    print(f"深度高度图形状: {depth_heightmap.shape}")
    print(f"最佳像素索引: {best_pix_ind}")
    print(f"标签值: {label_value}")
    
    try:
        # 进行反向传播
        print("\n执行反向传播...")
        loss = trainer.backprop(color_heightmap, depth_heightmap, 'push', best_pix_ind, label_value)
        print(f"反向传播成功，损失值: {loss}")
        
    except Exception as e:
        print(f"反向传播失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n测试完成")

if __name__ == "__main__":
    # 测试模型设备一致性
    test_model_device_consistency()
    
    # 测试旋转机制
    # test_rotation_mechanism()
    
    # 测试反向传播
    # test_backprop_on_random_data() 