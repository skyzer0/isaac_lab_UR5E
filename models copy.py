#!/usr/bin/env python

from collections import OrderedDict
import numpy as np
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.models as models
import matplotlib.pyplot as plt
import time
import scipy.ndimage.filters as sf


class reactive_net(nn.Module):

    def __init__(self, use_cuda): # , snapshot=None
        super(reactive_net, self).__init__()
        self.use_cuda = use_cuda

        # 设定设备
        self.device = torch.device("cuda" if use_cuda else "cpu")
        print(f"初始化模型，使用设备: {self.device}")

        # 直接将模型创建在正确的设备上
        if use_cuda:
            # 使用map_location确保预训练权重直接加载到正确的设备
            try:
                # 使用新的API
                from torchvision.models import densenet121, DenseNet121_Weights
                print("使用最新的torchvision API加载DenseNet模型")
                self.push_color_trunk = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1).to(self.device)
                self.push_depth_trunk = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1).to(self.device)
                self.grasp_color_trunk = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1).to(self.device)
                self.grasp_depth_trunk = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1).to(self.device)
            except (ImportError, AttributeError):
                # 回退到旧版API
                print("回退到旧版torchvision API加载DenseNet模型")
                self.push_color_trunk = torchvision.models.densenet121(pretrained=True).to(self.device)
                self.push_depth_trunk = torchvision.models.densenet121(pretrained=True).to(self.device)
                self.grasp_color_trunk = torchvision.models.densenet121(pretrained=True).to(self.device)
                self.grasp_depth_trunk = torchvision.models.densenet121(pretrained=True).to(self.device)
        else:
            try:
                # 使用新的API
                from torchvision.models import densenet121, DenseNet121_Weights
                print("使用最新的torchvision API加载DenseNet模型")
                self.push_color_trunk = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
                self.push_depth_trunk = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
                self.grasp_color_trunk = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
                self.grasp_depth_trunk = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
            except (ImportError, AttributeError):
                # 回退到旧版API
                print("回退到旧版torchvision API加载DenseNet模型")
                self.push_color_trunk = torchvision.models.densenet121(pretrained=True)
                self.push_depth_trunk = torchvision.models.densenet121(pretrained=True)
                self.grasp_color_trunk = torchvision.models.densenet121(pretrained=True)
                self.grasp_depth_trunk = torchvision.models.densenet121(pretrained=True)

        self.num_rotations = 16

        # Construct network branches for pushing and grasping
        self.pushnet = nn.Sequential(OrderedDict([
            ('push-norm0', nn.BatchNorm2d(2048)),
            ('push-relu0', nn.ReLU(inplace=True)),
            ('push-conv0', nn.Conv2d(2048, 64, kernel_size=1, stride=1, bias=False)),
            ('push-norm1', nn.BatchNorm2d(64)),
            ('push-relu1', nn.ReLU(inplace=True)),
            ('push-conv1', nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False))
            # ('push-upsample2', nn.Upsample(scale_factor=4, mode='bilinear'))
        ]))
        self.graspnet = nn.Sequential(OrderedDict([
            ('grasp-norm0', nn.BatchNorm2d(2048)),
            ('grasp-relu0', nn.ReLU(inplace=True)),
            ('grasp-conv0', nn.Conv2d(2048, 64, kernel_size=1, stride=1, bias=False)),
            ('grasp-norm1', nn.BatchNorm2d(64)),
            ('grasp-relu1', nn.ReLU(inplace=True)),
            ('grasp-conv1', nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False))
            # ('grasp-upsample2', nn.Upsample(scale_factor=4, mode='bilinear'))
        ]))

        # 将pushnet和graspnet移到正确的设备
        if use_cuda:
            self.pushnet = self.pushnet.to(self.device)
            self.graspnet = self.graspnet.to(self.device)

        # Initialize network weights
        for m in self.named_modules():
            if 'push-' in m[0] or 'grasp-' in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    nn.init.kaiming_normal_(m[1].weight.data)
                elif isinstance(m[1], nn.BatchNorm2d):
                    m[1].weight.data.fill_(1)
                    m[1].bias.data.zero_()

        # Initialize output variable (for backprop)
        self.interm_feat = []
        self.output_prob = []

    def forward(self, input_color_data, input_depth_data, is_volatile=False, specific_rotation=-1):
        self.interm_feat = []
        self.output_prob = []
        
        # 确保数据在正确的设备上
        device = next(self.parameters()).device
        print(f"Forward函数中，当前设备：{device}")
        
        if input_color_data.device != device:
            input_color_data = input_color_data.to(device)
        if input_depth_data.device != device:
            input_depth_data = input_depth_data.to(device)

        if is_volatile:
            with torch.no_grad():
                output_prob = []
                interm_feat = []

                # Apply rotations to intermediate features
                # for rotate_idx in range(self.num_rotations):
                if specific_rotation >= 0:
                    rotate_idx = specific_rotation
                else:
                    rotate_idx = 0
                rotate_theta = np.radians(rotate_idx*(360/self.num_rotations))
                
                # Compute sample grid for rotation BEFORE branches
                affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],[-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
                affine_mat_before.shape = (2,3,1)
                affine_mat_before = torch.from_numpy(affine_mat_before).permute(2,0,1).float().to(device)
                
                # 將Size轉換為List[int]
                input_size = list(input_color_data.size())
                flow_grid_before = F.affine_grid(affine_mat_before, input_size, align_corners=True)
                flow_grid_before = flow_grid_before.to(device)

                # Rotate images clockwise
                try:
                    print(f"Input color data device: {input_color_data.device}")
                    print(f"Flow grid before device: {flow_grid_before.device}")
                    
                    # 修改grid_sample調用方式
                    rotate_color = F.grid_sample(input_color_data, flow_grid_before, mode='nearest', align_corners=True)
                    rotate_depth = F.grid_sample(input_depth_data, flow_grid_before, mode='nearest', align_corners=True)
                    
                    print("Grid sampling completed successfully")
                except Exception as e:
                    print(f"Error during grid sampling: {e}")
                    raise

                # Compute intermediate features
                try:
                    print(f"Rotate color shape: {rotate_color.shape}")
                    
                    # 確保輸入數據在當前設備上
                    rotate_color = rotate_color.to(device)
                    rotate_depth = rotate_depth.to(device)
                    
                    # 確保模型在當前設備上
                    self.push_color_trunk = self.push_color_trunk.to(device)
                    self.push_depth_trunk = self.push_depth_trunk.to(device)
                    self.grasp_color_trunk = self.grasp_color_trunk.to(device)
                    self.grasp_depth_trunk = self.grasp_depth_trunk.to(device)
                    
                    # 计算特征
                    interm_push_color_feat = self.push_color_trunk.features(rotate_color)
                    interm_push_depth_feat = self.push_depth_trunk.features(rotate_depth)
                    interm_push_feat = torch.cat((interm_push_color_feat, interm_push_depth_feat), dim=1)
                    
                    interm_grasp_color_feat = self.grasp_color_trunk.features(rotate_color)
                    interm_grasp_depth_feat = self.grasp_depth_trunk.features(rotate_depth)
                    interm_grasp_feat = torch.cat((interm_grasp_color_feat, interm_grasp_depth_feat), dim=1)
                    
                    # 存储中间特征
                    interm_feat.append([interm_push_feat, interm_grasp_feat])
                except Exception as e:
                    print(f"Error in computing intermediate features: {e}")
                    raise

                # Compute sample grid for rotation AFTER branches
                affine_mat_after = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0],[-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
                affine_mat_after.shape = (2,3,1)
                affine_mat_after = torch.from_numpy(affine_mat_after).permute(2,0,1).float().to(device)
                
                # 將Size轉換為List[int]
                feat_size = list(interm_push_feat.size())
                flow_grid_after = F.affine_grid(affine_mat_after, feat_size, align_corners=True)
                flow_grid_after = flow_grid_after.to(device)

                # Forward pass through branches, undo rotation on output predictions, upsample results
                self.pushnet = self.pushnet.to(device)
                self.graspnet = self.graspnet.to(device)

                # 创建上采样模块并移动到正确设备
                upsample = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True).to(device)
                
                # 计算push网络输出并上采样
                push_feat = self.pushnet(interm_push_feat)
                push_rotated = F.grid_sample(push_feat, flow_grid_after, mode='nearest', align_corners=True)
                push_output = upsample(push_rotated)
                
                # 计算grasp网络输出并上采样
                grasp_feat = self.graspnet(interm_grasp_feat)
                grasp_rotated = F.grid_sample(grasp_feat, flow_grid_after, mode='nearest', align_corners=True)
                grasp_output = upsample(grasp_rotated)
                
                # 添加到输出列表
                output_prob.append([push_output, grasp_output])

                return output_prob, interm_feat

        else:
            self.output_prob = []
            self.interm_feat = []

            # Apply rotations to intermediate features
            # for rotate_idx in range(self.num_rotations):
            if specific_rotation >= 0:
                rotate_idx = specific_rotation
            else:
                rotate_idx = 0
            rotate_theta = np.radians(rotate_idx*(360/self.num_rotations))

            # Compute sample grid for rotation BEFORE branches
            affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],[-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
            affine_mat_before.shape = (2,3,1)
            affine_mat_before = torch.from_numpy(affine_mat_before).permute(2,0,1).float().to(device)
            
            # 將Size轉換為List[int]
            input_size = list(input_color_data.size())
            flow_grid_before = F.affine_grid(affine_mat_before, input_size, align_corners=True)
            flow_grid_before = flow_grid_before.to(device)

            # Rotate images clockwise
            try:
                print(f"(Non-volatile) Input color data device: {input_color_data.device}")
                print(f"(Non-volatile) Flow grid before device: {flow_grid_before.device}")
                
                # 修改grid_sample調用方式
                rotate_color = F.grid_sample(input_color_data, flow_grid_before, mode='nearest', align_corners=True)
                rotate_depth = F.grid_sample(input_depth_data, flow_grid_before, mode='nearest', align_corners=True)
                
                print("(Non-volatile) Grid sampling completed successfully")
            except Exception as e:
                print(f"Error during non-volatile grid sampling: {e}")
                raise

            # Compute intermediate features
            try:
                print(f"(Non-volatile) Rotate color shape: {rotate_color.shape}")
                
                # 確保輸入數據在當前設備上
                rotate_color = rotate_color.to(device)
                rotate_depth = rotate_depth.to(device)
                
                # 確保模型在當前設備上
                self.push_color_trunk = self.push_color_trunk.to(device)
                self.push_depth_trunk = self.push_depth_trunk.to(device)
                self.grasp_color_trunk = self.grasp_color_trunk.to(device)
                self.grasp_depth_trunk = self.grasp_depth_trunk.to(device)
                
                # 计算特征
                interm_push_color_feat = self.push_color_trunk.features(rotate_color)
                interm_push_depth_feat = self.push_depth_trunk.features(rotate_depth)
                interm_push_feat = torch.cat((interm_push_color_feat, interm_push_depth_feat), dim=1)
                
                interm_grasp_color_feat = self.grasp_color_trunk.features(rotate_color)
                interm_grasp_depth_feat = self.grasp_depth_trunk.features(rotate_depth)
                interm_grasp_feat = torch.cat((interm_grasp_color_feat, interm_grasp_depth_feat), dim=1)
                
                # 存储中间特征
                self.interm_feat.append([interm_push_feat, interm_grasp_feat])
            except Exception as e:
                print(f"Error in computing non-volatile intermediate features: {e}")
                raise

            # Compute sample grid for rotation AFTER branches
            affine_mat_after = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0],[-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
            affine_mat_after.shape = (2,3,1)
            affine_mat_after = torch.from_numpy(affine_mat_after).permute(2,0,1).float().to(device)
            
            # 將Size轉換為List[int]
            feat_size = list(interm_push_feat.size())
            flow_grid_after = F.affine_grid(affine_mat_after, feat_size, align_corners=True)
            flow_grid_after = flow_grid_after.to(device)

            # Forward pass through branches, undo rotation on output predictions, upsample results
            self.pushnet = self.pushnet.to(device)
            self.graspnet = self.graspnet.to(device)
            
            # 创建上采样模块并移动到正确设备
            upsample = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True).to(device)
            
            # 计算push网络输出并上采样
            push_feat = self.pushnet(interm_push_feat)
            push_rotated = F.grid_sample(push_feat, flow_grid_after, mode='nearest', align_corners=True)
            push_output = upsample(push_rotated)
            
            # 计算grasp网络输出并上采样
            grasp_feat = self.graspnet(interm_grasp_feat)
            grasp_rotated = F.grid_sample(grasp_feat, flow_grid_after, mode='nearest', align_corners=True)
            grasp_output = upsample(grasp_rotated)
            
            # 添加到输出列表
            self.output_prob.append([push_output, grasp_output])
            
            # 确保forward方法返回state_feat，与volatile=True模式一致
            return self.output_prob, self.interm_feat, interm_push_feat


class reinforcement_net(nn.Module):

    def __init__(self, use_cuda): # , snapshot=None
        super(reinforcement_net, self).__init__()
        self.use_cuda = use_cuda
        
        # 确定设备并将其保存为实例变量，以便在整个类中使用
        self.device = torch.device("cuda" if use_cuda else "cpu")
        print(f"初始化reinforcement_net模型，使用设备: {self.device}")

        # 确保整个模型在指定设备上
        self.to(self.device)

        # 直接将模型创建在正确的设备上
        if use_cuda:
            # 使用map_location确保预训练权重直接加载到正确的设备
            try:
                # 使用新的API
                from torchvision.models import densenet121, DenseNet121_Weights
                print("使用最新的torchvision API加载DenseNet模型")
                self.push_color_trunk = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1).to(self.device)
                self.push_depth_trunk = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1).to(self.device)
                self.grasp_color_trunk = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1).to(self.device)
                self.grasp_depth_trunk = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1).to(self.device)
            except (ImportError, AttributeError):
                # 回退到旧版API
                print("回退到旧版torchvision API加载DenseNet模型")
                self.push_color_trunk = torchvision.models.densenet121(pretrained=True).to(self.device)
                self.push_depth_trunk = torchvision.models.densenet121(pretrained=True).to(self.device)
                self.grasp_color_trunk = torchvision.models.densenet121(pretrained=True).to(self.device)
                self.grasp_depth_trunk = torchvision.models.densenet121(pretrained=True).to(self.device)
        else:
            try:
                # 使用新的API
                from torchvision.models import densenet121, DenseNet121_Weights
                print("使用最新的torchvision API加载DenseNet模型")
                self.push_color_trunk = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
                self.push_depth_trunk = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
                self.grasp_color_trunk = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
                self.grasp_depth_trunk = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
            except (ImportError, AttributeError):
                # 回退到旧版API
                print("回退到旧版torchvision API加载DenseNet模型")
                self.push_color_trunk = torchvision.models.densenet121(pretrained=True)
                self.push_depth_trunk = torchvision.models.densenet121(pretrained=True)
                self.grasp_color_trunk = torchvision.models.densenet121(pretrained=True)
                self.grasp_depth_trunk = torchvision.models.densenet121(pretrained=True)

        self.num_rotations = 16

        # Construct network branches for pushing and grasping
        self.pushnet = nn.Sequential(OrderedDict([
            ('push-norm0', nn.BatchNorm2d(2048)),
            ('push-relu0', nn.ReLU(inplace=True)),
            ('push-conv0', nn.Conv2d(2048, 64, kernel_size=1, stride=1, bias=False)),
            ('push-norm1', nn.BatchNorm2d(64)),
            ('push-relu1', nn.ReLU(inplace=True)),
            ('push-conv1', nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False))
        ]))
        self.graspnet = nn.Sequential(OrderedDict([
            ('grasp-norm0', nn.BatchNorm2d(2048)),
            ('grasp-relu0', nn.ReLU(inplace=True)),
            ('grasp-conv0', nn.Conv2d(2048, 64, kernel_size=1, stride=1, bias=False)),
            ('grasp-norm1', nn.BatchNorm2d(64)),
            ('grasp-relu1', nn.ReLU(inplace=True)),
            ('grasp-conv1', nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False))
        ]))

        # 将网络分支也移动到相应的设备上
        self.pushnet = self.pushnet.to(self.device)
        self.graspnet = self.graspnet.to(self.device)
        print(f"网络分支已移动到{self.device}设备上")
        
        # 最后将整个模型移动到设备上，确保所有组件都在同一设备
        self.to(self.device)

        # Initialize network weights
        for m in self.named_modules():
            if 'push-' in m[0] or 'grasp-' in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    try:
                        nn.init.kaiming_normal_(m[1].weight.data)
                    except AttributeError:
                        nn.init.kaiming_normal(m[1].weight.data)
                elif isinstance(m[1], nn.BatchNorm2d):
                    m[1].weight.data.fill_(1)
                    m[1].bias.data.zero_()

        # Initialize output variable (for backprop)
        self.interm_feat = []
        self.output_prob = []
        
        # 确保所有模型参数都在正确的设备上
        self._verify_device()
    
    def _verify_device(self):
        """验证所有模型组件是否都在正确的设备上"""
        device = next(self.parameters()).device
        components = [
            (self.push_color_trunk, "push_color_trunk"),
            (self.push_depth_trunk, "push_depth_trunk"),
            (self.grasp_color_trunk, "grasp_color_trunk"),
            (self.grasp_depth_trunk, "grasp_depth_trunk"),
            (self.pushnet, "pushnet"),
            (self.graspnet, "graspnet")
        ]
        
        issues = []
        for component, name in components:
            component_device = next(component.parameters()).device
            if component_device != device:
                issues.append(f"{name} on {component_device}, should be on {device}")
                # 自动修复
                component.to(device)
                
        if issues:
            print(f"设备不一致问题已修复: {', '.join(issues)}")
        
        self.device = device  # 更新设备引用
        return device

    def forward(self, input_color_data, input_depth_data, is_volatile=False, specific_rotation=-1):
        # 首先检查并修复设备不一致问题
        device = self._verify_device()
        print(f"Forward方法中，当前设备: {device}")
        
        # 确保输入数据在正确的设备上
        if input_color_data.device != device:
            input_color_data = input_color_data.to(device)
        if input_depth_data.device != device:
            input_depth_data = input_depth_data.to(device)

        if is_volatile:
            with torch.no_grad():
                output_prob = []
                interm_feat = []
                state_feat_buffer = None  # 用于保存中间状态特征

                # Apply rotations to images
                for rotate_idx in range(self.num_rotations):
                    rotate_theta = np.radians(rotate_idx*(360/self.num_rotations))

                    # Compute sample grid for rotation BEFORE neural network
                    affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],[-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
                    affine_mat_before.shape = (2,3,1)
                    affine_mat_before = torch.from_numpy(affine_mat_before).permute(2,0,1).float()
                    # 确保在正确的设备上
                    affine_mat_before = affine_mat_before.to(device)
                    
                    try:
                        # 使用list(input_color_data.size())代替直接传入Size对象
                        flow_grid_before = F.affine_grid(affine_mat_before, list(input_color_data.size()), align_corners=True)
                        flow_grid_before = flow_grid_before.to(device)  # 确保在同一设备上
                    except Exception as e:
                        print(f"Error in affine_grid: {e}")
                        print(f"affine_mat_before shape: {affine_mat_before.shape}, device: {affine_mat_before.device}")
                        print(f"input_color_data shape: {input_color_data.shape}, device: {input_color_data.device}")
                        # 尝试修复
                        if len(input_color_data.shape) < 4:
                            print("Input shape too small, adding batch dimension")
                            input_color_data = input_color_data.unsqueeze(0)
                            input_depth_data = input_depth_data.unsqueeze(0)
                        flow_grid_before = F.affine_grid(affine_mat_before, list(input_color_data.size()), align_corners=True)
                        flow_grid_before = flow_grid_before.to(device)

                    # Rotate images clockwise
                    try:
                        rotate_color = F.grid_sample(input_color_data, flow_grid_before, mode='nearest', align_corners=True)
                        rotate_depth = F.grid_sample(input_depth_data, flow_grid_before, mode='nearest', align_corners=True)
                    except Exception as e:
                        print(f"Error in grid_sample: {e}")
                        print(f"input_color_data shape: {input_color_data.shape}, device: {input_color_data.device}")
                        print(f"flow_grid_before shape: {flow_grid_before.shape}, device: {flow_grid_before.device}")
                        raise

                    # 确保数据在正确的设备上
                    rotate_color = rotate_color.to(device)
                    rotate_depth = rotate_depth.to(device)

                    # Compute intermediate features
                    try:
                        interm_push_color_feat = self.push_color_trunk.features(rotate_color)
                        interm_push_depth_feat = self.push_depth_trunk.features(rotate_depth)
                        interm_push_feat = torch.cat((interm_push_color_feat, interm_push_depth_feat), dim=1)
                        interm_grasp_color_feat = self.grasp_color_trunk.features(rotate_color)
                        interm_grasp_depth_feat = self.grasp_depth_trunk.features(rotate_depth)
                        interm_grasp_feat = torch.cat((interm_grasp_color_feat, interm_grasp_depth_feat), dim=1)
                    except Exception as e:
                        print(f"Error in computing features: {e}")
                        print(f"rotate_color shape: {rotate_color.shape}, device: {rotate_color.device}")
                        print(f"push_color_trunk device: {next(self.push_color_trunk.parameters()).device}")
                        # 尝试再次确保所有组件在同一设备上
                        self._verify_device()
                        # 重新尝试
                        interm_push_color_feat = self.push_color_trunk.features(rotate_color)
                        interm_push_depth_feat = self.push_depth_trunk.features(rotate_depth)
                        interm_push_feat = torch.cat((interm_push_color_feat, interm_push_depth_feat), dim=1)
                        interm_grasp_color_feat = self.grasp_color_trunk.features(rotate_color)
                        interm_grasp_depth_feat = self.grasp_depth_trunk.features(rotate_depth)
                        interm_grasp_feat = torch.cat((interm_grasp_color_feat, interm_grasp_depth_feat), dim=1)
                    
                    # 保存中间特征
                    interm_feat.append([interm_push_feat, interm_grasp_feat])
                    
                    # 保存一个用于返回的状态特征
                    if state_feat_buffer is None:
                        state_feat_buffer = interm_push_feat

                    # Compute sample grid for rotation AFTER branches
                    affine_mat_after = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0],[-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
                    affine_mat_after.shape = (2,3,1)
                    affine_mat_after = torch.from_numpy(affine_mat_after).permute(2,0,1).float().to(device)
                    
                    try:
                        # 使用list代替直接传入Size对象
                        flow_grid_after = F.affine_grid(affine_mat_after, list(interm_push_feat.size()), align_corners=True)
                        flow_grid_after = flow_grid_after.to(device)  # 确保在同一设备上
                    except Exception as e:
                        print(f"Error in second affine_grid: {e}")
                        print(f"affine_mat_after shape: {affine_mat_after.shape}")
                        print(f"interm_push_feat shape: {interm_push_feat.shape}")
                        raise

                    # Forward pass through branches, undo rotation on output predictions, upsample results
                    upsample = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True).to(device)
                    
                    try:
                        push_feat = self.pushnet(interm_push_feat)
                        push_rotated = F.grid_sample(push_feat, flow_grid_after, mode='nearest', align_corners=True)
                        push_output = upsample(push_rotated)
                        
                        grasp_feat = self.graspnet(interm_grasp_feat)
                        grasp_rotated = F.grid_sample(grasp_feat, flow_grid_after, mode='nearest', align_corners=True)
                        grasp_output = upsample(grasp_rotated)
                    except Exception as e:
                        print(f"Error in forward pass through branches: {e}")
                        # 尝试再次确保所有组件在同一设备上
                        self._verify_device()
                        # 重新尝试
                        push_feat = self.pushnet(interm_push_feat)
                        push_rotated = F.grid_sample(push_feat, flow_grid_after, mode='nearest', align_corners=True)
                        push_output = upsample(push_rotated)
                        
                        grasp_feat = self.graspnet(interm_grasp_feat)
                        grasp_rotated = F.grid_sample(grasp_feat, flow_grid_after, mode='nearest', align_corners=True)
                        grasp_output = upsample(grasp_rotated)
                    
                    output_prob.append([push_output, grasp_output])

                # 保存在类变量中以便backprop时使用
                self.output_prob = output_prob
                self.interm_feat = interm_feat

                # 返回三个值：输出概率、中间特征、状态特征
                return output_prob, interm_feat, state_feat_buffer

        else:
            # 非推理模式，用于训练
            self.output_prob = []
            self.interm_feat = []
            
            # 使用指定的旋转角度
            rotate_idx = specific_rotation
            rotate_theta = np.radians(rotate_idx*(360/self.num_rotations))

            # Compute sample grid for rotation BEFORE branches
            affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],[-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
            affine_mat_before.shape = (2,3,1)
            affine_mat_before = torch.from_numpy(affine_mat_before).permute(2,0,1).float().to(device)
            
            try:
                # 使用list(input_color_data.size())代替直接传入Size对象
                flow_grid_before = F.affine_grid(affine_mat_before, list(input_color_data.size()), align_corners=True)
                flow_grid_before = flow_grid_before.to(device)  # 确保在同一设备上
            except Exception as e:
                print(f"训练模式中affine_grid错误: {e}")
                print(f"affine_mat_before shape: {affine_mat_before.shape}, device: {affine_mat_before.device}")
                print(f"input_color_data shape: {input_color_data.shape}, device: {input_color_data.device}")
                # 尝试修复
                if len(input_color_data.shape) < 4:
                    print("输入形状太小，添加批次维度")
                    input_color_data = input_color_data.unsqueeze(0)
                    input_depth_data = input_depth_data.unsqueeze(0)
                flow_grid_before = F.affine_grid(affine_mat_before, list(input_color_data.size()), align_corners=True)
                flow_grid_before = flow_grid_before.to(device)

            # Rotate images clockwise
            try:
                rotate_color = F.grid_sample(input_color_data, flow_grid_before, mode='nearest', align_corners=True)
                rotate_depth = F.grid_sample(input_depth_data, flow_grid_before, mode='nearest', align_corners=True)
            except Exception as e:
                print(f"训练模式中grid_sample错误: {e}")
                raise
            
            # 确保数据在正确的设备上
            rotate_color = rotate_color.to(device)
            rotate_depth = rotate_depth.to(device)

            # Compute intermediate features
            try:
                interm_push_color_feat = self.push_color_trunk.features(rotate_color)
                interm_push_depth_feat = self.push_depth_trunk.features(rotate_depth)
                interm_push_feat = torch.cat((interm_push_color_feat, interm_push_depth_feat), dim=1)
                interm_grasp_color_feat = self.grasp_color_trunk.features(rotate_color)
                interm_grasp_depth_feat = self.grasp_depth_trunk.features(rotate_depth)
                interm_grasp_feat = torch.cat((interm_grasp_color_feat, interm_grasp_depth_feat), dim=1)
            except Exception as e:
                print(f"训练模式中特征计算错误: {e}")
                # 尝试再次确保所有组件在同一设备上
                self._verify_device()
                # 重新尝试
                interm_push_color_feat = self.push_color_trunk.features(rotate_color)
                interm_push_depth_feat = self.push_depth_trunk.features(rotate_depth)
                interm_push_feat = torch.cat((interm_push_color_feat, interm_push_depth_feat), dim=1)
                interm_grasp_color_feat = self.grasp_color_trunk.features(rotate_color)
                interm_grasp_depth_feat = self.grasp_depth_trunk.features(rotate_depth)
                interm_grasp_feat = torch.cat((interm_grasp_color_feat, interm_grasp_depth_feat), dim=1)
            
            # 保存中间特征
            self.interm_feat.append([interm_push_feat, interm_grasp_feat])

            # Compute sample grid for rotation AFTER branches
            affine_mat_after = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0],[-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
            affine_mat_after.shape = (2,3,1)
            affine_mat_after = torch.from_numpy(affine_mat_after).permute(2,0,1).float().to(device)
            
            try:
                # 使用list代替直接传入Size对象
                flow_grid_after = F.affine_grid(affine_mat_after, list(interm_push_feat.size()), align_corners=True)
                flow_grid_after = flow_grid_after.to(device)  # 确保在同一设备上
            except Exception as e:
                print(f"训练模式中第二个affine_grid错误: {e}")
                raise

            # Forward pass through branches, undo rotation on output predictions, upsample results
            upsample = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True).to(device)
            
            try:
                push_feat = self.pushnet(interm_push_feat)
                push_rotated = F.grid_sample(push_feat, flow_grid_after, mode='nearest', align_corners=True)
                push_output = upsample(push_rotated)
                
                grasp_feat = self.graspnet(interm_grasp_feat)
                grasp_rotated = F.grid_sample(grasp_feat, flow_grid_after, mode='nearest', align_corners=True)
                grasp_output = upsample(grasp_rotated)
            except Exception as e:
                print(f"训练模式中分支前向传播错误: {e}")
                # 尝试再次确保所有组件在同一设备上
                self._verify_device()
                # 重新尝试
                push_feat = self.pushnet(interm_push_feat)
                push_rotated = F.grid_sample(push_feat, flow_grid_after, mode='nearest', align_corners=True)
                push_output = upsample(push_rotated)
                
                grasp_feat = self.graspnet(interm_grasp_feat)
                grasp_rotated = F.grid_sample(grasp_feat, flow_grid_after, mode='nearest', align_corners=True)
                grasp_output = upsample(grasp_rotated)
            
            # 保存输出概率
            self.output_prob.append([push_output, grasp_output])

            # 返回三个值：输出概率、中间特征、状态特征
            return self.output_prob, self.interm_feat, interm_push_feat

