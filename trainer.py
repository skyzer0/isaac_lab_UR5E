import os
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from vpg_utils import CrossEntropyLoss2d
from models import reactive_net, reinforcement_net
from scipy import ndimage
import matplotlib.pyplot as plt
from torch.optim import SGD  # 显式导入SGD
from logger import Logger  # 导入Logger类


class Trainer(object):
    def __init__(self, method, push_rewards=True, future_reward_discount=0.5,
                 is_testing=False, load_snapshot=False, snapshot_file=None, force_cpu=False):

        self.method = method
        self.is_testing = is_testing  # 明确保存测试模式标志
        print(f"Initializing trainer, method: {method}, testing mode: {is_testing}")

        # Check if CUDA can be used
        if torch.cuda.is_available() and not force_cpu:
            print("CUDA detected. Running with GPU acceleration.")
            self.use_cuda = True
        elif force_cpu:
            print("CUDA detected, but overriding with option '--cpu'. Running with only CPU.")
            self.use_cuda = False
        else:
            print("CUDA is *NOT* detected. Running with only CPU.")
            self.use_cuda = False

        # Fully convolutional classification network for supervised learning
        if self.method == 'reactive':
            self.model = reactive_net(self.use_cuda)

            # Initialize classification loss
            push_num_classes = 3 # 0 - push, 1 - no change push, 2 - no loss
            push_class_weights = torch.ones(push_num_classes)
            push_class_weights[push_num_classes - 1] = 0
            if self.use_cuda:
                self.push_criterion = CrossEntropyLoss2d(push_class_weights.cuda()).cuda()
            else:
                self.push_criterion = CrossEntropyLoss2d(push_class_weights)
            grasp_num_classes = 3 # 0 - grasp, 1 - failed grasp, 2 - no loss
            grasp_class_weights = torch.ones(grasp_num_classes)
            grasp_class_weights[grasp_num_classes - 1] = 0
            if self.use_cuda:
                self.grasp_criterion = CrossEntropyLoss2d(grasp_class_weights.cuda()).cuda()
            else:
                self.grasp_criterion = CrossEntropyLoss2d(grasp_class_weights)

        # Fully convolutional Q network for deep reinforcement learning
        elif self.method == 'reinforcement':
            self.model = reinforcement_net(self.use_cuda)
            self.push_rewards = push_rewards
            # 调整未来奖励折扣率
            if future_reward_discount is not None:
                self.future_reward_discount = future_reward_discount
            else:
                self.future_reward_discount = 0.5  # 默认值从0.65降低为0.5
            print(f"Reward settings - Push rewards: {push_rewards}, Future reward discount rate: {self.future_reward_discount}")

            # Initialize Huber loss
            self.criterion = torch.nn.SmoothL1Loss(reduce=False) # Huber loss
            if self.use_cuda:
                self.criterion = self.criterion.cuda()

        # Load pre-trained model
        if load_snapshot:
            self.model.load_state_dict(torch.load(snapshot_file))
            print('Pre-trained model snapshot loaded from: %s' % (snapshot_file))

        # Convert model from CPU to GPU
        if self.use_cuda:
            self.model = self.model.cuda()

        # Set model to training mode
        self.model.train()

        # Initialize optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-4, momentum=0.9, weight_decay=2e-5)
        self.iteration = 0

        # Initialize lists to save execution info and RL variables
        self.executed_action_log = []
        self.label_value_log = []
        self.reward_value_log = []
        self.predicted_value_log = []
        self.use_heuristic_log = []
        self.is_exploit_log = []
        self.clearance_log = []

        # 确保模型完全迁移到正确的设备上
        if self.use_cuda:
            # 获取当前设备
            device = torch.device("cuda")
            # 将整个模型显式迁移到设备上
            self.model = self.model.to(device)
            # 检查并确保所有参数都在CUDA上
            for name, param in self.model.named_parameters():
                if not param.is_cuda:
                    print(f"警告: 参数 {name} 不在CUDA上，正在移动...")
                    param.data = param.data.cuda()
            
            # 确保在每次前向传播前都检查模型是否完全在CUDA上
            print("模型已成功迁移到CUDA上，参数总数: ", sum(p.numel() for p in self.model.parameters()))
            
        # 记录加载信息
        self.logger = None  # 初始化logger属性

        # 定义损失函数
        if self.method == 'reactive':
            self.criterion = nn.CrossEntropyLoss(reduce=False)
        elif self.method == 'reinforcement':
            self.criterion = nn.MSELoss(reduce=False)
        if self.use_cuda:
            self.criterion = self.criterion.cuda()
            
        # 定义优化器
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-4, momentum=0.9, weight_decay=2e-5)
        
        # 记录执行日志
        self.iteration = 0
        self.executed_action_log = []
        self.label_value_log = []
        self.reward_value_log = []
        self.predicted_value_log = []
        self.use_heuristic_log = []
        self.is_exploit_log = []
        self.clearance_log = []
        
        # 将清理日志添加到类实例
        if isinstance(self.logger, Logger):
            self.logger.add_log('executed-action', np.zeros((0, 2)))
            self.logger.add_log('label-value', np.zeros((0, 2)))
            self.logger.add_log('reward-value', np.zeros((0, 2)))
            self.logger.add_log('predicted-value', np.zeros((0, 2)))
            self.logger.add_log('use-heuristic', np.zeros((0, 2)))
            self.logger.add_log('is-exploit', np.zeros((0, 2)))
            self.logger.add_log('clearance', np.zeros((0, 1)))

        # 在__init__方法中，添加对象级别的设备追踪（在self.initialized = True之前添加）
        self.device = torch.device("cuda") if self.use_cuda else torch.device("cpu")
        print(f"模型将运行在设备: {self.device}")


    # Pre-load execution info and RL variables
    def preload(self, transitions_directory):
        try:
            self.executed_action_log = np.loadtxt(os.path.join(transitions_directory, 'executed-action.log.txt'), delimiter=' ')
            self.iteration = self.executed_action_log.shape[0] - 2
            self.executed_action_log = self.executed_action_log[0:self.iteration,:]
            self.executed_action_log = self.executed_action_log.tolist()
            
            self.label_value_log = np.loadtxt(os.path.join(transitions_directory, 'label-value.log.txt'), delimiter=' ')
            self.label_value_log = self.label_value_log[0:self.iteration]
            self.label_value_log.shape = (self.iteration,1)
            self.label_value_log = self.label_value_log.tolist()
            
            self.predicted_value_log = np.loadtxt(os.path.join(transitions_directory, 'predicted-value.log.txt'), delimiter=' ')
            self.predicted_value_log = self.predicted_value_log[0:self.iteration]
            self.predicted_value_log.shape = (self.iteration,1)
            self.predicted_value_log = self.predicted_value_log.tolist()
            
            self.reward_value_log = np.loadtxt(os.path.join(transitions_directory, 'reward-value.log.txt'), delimiter=' ')
            self.reward_value_log = self.reward_value_log[0:self.iteration]
            self.reward_value_log.shape = (self.iteration,1)
            self.reward_value_log = self.reward_value_log.tolist()
            
            self.use_heuristic_log = np.loadtxt(os.path.join(transitions_directory, 'use-heuristic.log.txt'), delimiter=' ')
            self.use_heuristic_log = self.use_heuristic_log[0:self.iteration]
            self.use_heuristic_log.shape = (self.iteration,1)
            self.use_heuristic_log = self.use_heuristic_log.tolist()
            
            self.is_exploit_log = np.loadtxt(os.path.join(transitions_directory, 'is-exploit.log.txt'), delimiter=' ')
            self.is_exploit_log = self.is_exploit_log[0:self.iteration]
            self.is_exploit_log.shape = (self.iteration,1)
            self.is_exploit_log = self.is_exploit_log.tolist()
            
            self.clearance_log = np.loadtxt(os.path.join(transitions_directory, 'clearance.log.txt'), delimiter=' ')
            self.clearance_log.shape = (self.clearance_log.shape[0],1)
            self.clearance_log = self.clearance_log.tolist()
            
            print("Successfully loaded training logs, executed action records: %d, reward records: %d" % (len(self.executed_action_log), len(self.reward_value_log)))
        except Exception as e:
            print(f"Error loading training logs: {e}")
            # 重置列表，防止数据不一致
            self.executed_action_log = []
            self.label_value_log = []
            self.reward_value_log = []
            self.predicted_value_log = []
            self.use_heuristic_log = []
            self.is_exploit_log = []
            self.clearance_log = []


    # 在Trainer类中添加一个reset_device方法，在forward或其他任何方法调用前确保模型在正确的设备上
    def reset_device(self):
        """确保模型在正确的设备上，如果启用了CUDA则确保在GPU上"""
        # 首先确定目标设备
        if not hasattr(self, 'device'):
            self.device = torch.device("cuda" if self.use_cuda else "cpu")
        
        # 强制检查 - 如果use_cuda为True，则必须使用CUDA设备
        if self.use_cuda and 'cuda' not in str(self.device):
            print(f"强制将目标设备从 {self.device} 更改为 cuda")
            self.device = torch.device("cuda")
            
        target_device = self.device
        print(f"reset_device: 目标设备为 {target_device}")
            
        # 获取当前模型所在设备
        try:
            current_device = next(self.model.parameters()).device
        except StopIteration:
            print("信息：无法确定模型设备，模型可能没有参数")
            return target_device
            
        # 改进的设备比较逻辑
        devices_match = False
        if self.use_cuda:
            # 只要两个设备都是CUDA设备，就认为它们匹配
            devices_match = 'cuda' in str(current_device)
        else:
            # 非CUDA情况，需要完全匹配
            devices_match = (current_device == target_device)
            
        # 如果设备不一致，执行迁移
        if not devices_match:
            print(f"检测到设备不一致！将模型从 {current_device} 迁移到 {target_device}")
            
            # 1. 首先尝试使用模型自带的device管理方法
            if hasattr(self.model, '_verify_device'):
                print("使用模型内置的设备验证方法")
                try:
                    self.model._verify_device()
                    # 验证是否成功
                    current_device = next(self.model.parameters()).device
                    if self.use_cuda:
                        devices_match = 'cuda' in str(current_device)
                    else:
                        devices_match = (current_device == target_device)
                        
                    if devices_match:
                        print(f"内置方法成功迁移模型到 CUDA 设备")
                        return target_device
                    else:
                        print(f"内置方法迁移失败，当前设备仍为 {current_device}")
                except Exception as e:
                    print(f"内置方法出错: {e}")
            
            # 2. 尝试整体迁移模型
            try:
                # 使用to方法迁移模型
                self.model = self.model.to(target_device)
                print(f"模型已整体迁移到 {target_device}")
                
                # 立即验证迁移是否成功
                try:
                    current_device = next(self.model.parameters()).device
                    if self.use_cuda:
                        devices_match = 'cuda' in str(current_device)
                    else:
                        devices_match = (current_device == target_device)
                        
                    if not devices_match:
                        print(f"警告：整体迁移后设备仍不一致! 当前: {current_device}")
                    else:
                        print(f"验证通过：模型现在在 CUDA 设备上")
                        return target_device
                except StopIteration:
                    print("信息：无法验证迁移结果，模型可能没有参数")
            except Exception as e:
                print(f"整体迁移模型失败: {e}")
                
            # 3. 更彻底的解决方案：逐个迁移每个子模块和参数
            try:
                print("开始逐个迁移子模块和参数...")
                # 迁移所有子模块
                for name, module in self.model.named_children():
                    try:
                        # 尝试获取模块当前设备
                        try:
                            module_device = next(module.parameters()).device
                            
                            # 使用相同的设备比较逻辑
                            if self.use_cuda:
                                module_match = 'cuda' in str(module_device)
                            else:
                                module_match = (module_device == target_device)
                                
                            if module_match:
                                print(f"子模块 {name} 已在正确的设备上")
                                continue
                            else:
                                print(f"子模块 {name} 在 {module_device}，需要迁移")
                        except StopIteration:
                            print(f"子模块 {name} 没有参数，跳过")
                            continue
                            
                        # 迁移模块
                        module.to(target_device)
                        
                        # 验证迁移
                        try:
                            module_device = next(module.parameters()).device
                            if self.use_cuda:
                                module_match = 'cuda' in str(module_device)
                            else:
                                module_match = (module_device == target_device)
                                
                            if not module_match:
                                print(f"警告：子模块 {name} 迁移失败，仍在 {module_device}")
                            else:
                                print(f"子模块 {name} 成功迁移到目标设备")
                        except StopIteration:
                            pass
                    except Exception as e:
                        print(f"迁移子模块 {name} 时出错: {e}")
                
                # 迁移所有参数
                param_count = 0
                migrated_count = 0
                for name, param in self.model.named_parameters():
                    param_count += 1
                    
                    # 使用相同的设备比较逻辑
                    param_match = False
                    if self.use_cuda:
                        param_match = 'cuda' in str(param.device)
                    else:
                        param_match = (param.device == target_device)
                        
                    if not param_match:
                        try:
                            param.data = param.data.to(target_device)
                            migrated_count += 1
                        except Exception as e:
                            print(f"无法迁移参数 {name}: {e}")
                
                # 如果有参数迁移发生，记录日志
                if migrated_count > 0:
                    print(f"已手动迁移 {migrated_count}/{param_count} 个参数到目标设备")
            except Exception as e:
                print(f"逐个迁移过程中发生错误: {e}")
                
            # 4. 最后验证迁移结果
            try:
                current_device = next(self.model.parameters()).device
                
                # 使用相同的设备比较逻辑
                if self.use_cuda:
                    devices_match = 'cuda' in str(current_device)
                else:
                    devices_match = (current_device == target_device)
                    
                if not devices_match:
                    print(f"严重警告：所有迁移尝试后设备仍不一致! 当前: {current_device}")
                    
                    # 最后尝试：重新创建模型（如果有能力的话）
                    if hasattr(self, 'initialize_model'):
                        print("尝试重新初始化模型...")
                        try:
                            self.initialize_model()
                            print("模型重新初始化完成")
                        except Exception as e:
                            print(f"模型重新初始化失败: {e}")
                else:
                    print(f"所有迁移步骤完成，模型现在在目标设备上")
            except StopIteration:
                print("信息：无法验证最终迁移结果，模型可能没有参数")
        else:
            print(f"设备一致性检查通过: 模型运行在目标设备上")
                
        # 确保优化器的状态字典也在正确的设备上
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            for param_group in self.optimizer.param_groups:
                for param in param_group['params']:
                    # 使用相同的设备比较逻辑
                    param_match = False
                    if self.use_cuda:
                        param_match = 'cuda' in str(param.device)
                    else:
                        param_match = (param.device == target_device)
                        
                    if not param_match:
                        print(f"优化器参数在 {param.device}，移动到目标设备")
                        param.data = param.data.to(target_device)
        
        return target_device


    # Compute forward pass through model to compute affordances/Q
    def forward(self, color_heightmap, depth_heightmap, is_volatile=False, specific_rotation=-1):
        # 首先确保模型在正确的设备上
        target_device = self.reset_device()
        print(f"Trainer forward: 模型设备为 {'CUDA' if self.use_cuda else 'CPU'}")

        # 如果输入是 numpy 数组,转换为 tensor
        if isinstance(color_heightmap, np.ndarray):
            color_heightmap = torch.from_numpy(color_heightmap.astype(np.float32))
        if isinstance(depth_heightmap, np.ndarray):
            depth_heightmap = torch.from_numpy(depth_heightmap.astype(np.float32))
            
        # 确保预处理过程中用的张量都在CPU上
        color_heightmap_cpu = color_heightmap.cpu().numpy() if isinstance(color_heightmap, torch.Tensor) else color_heightmap.copy()
        depth_heightmap_cpu = depth_heightmap.cpu().numpy() if isinstance(depth_heightmap, torch.Tensor) else depth_heightmap.copy()

        # 确保输入维度正确
        if len(depth_heightmap_cpu.shape) == 3:
            depth_heightmap_cpu = depth_heightmap_cpu.squeeze(0)
        if len(color_heightmap_cpu.shape) == 4:
            color_heightmap_cpu = color_heightmap_cpu.squeeze(0)

        # Apply 2x scale to input heightmaps
        color_heightmap_2x = ndimage.zoom(color_heightmap_cpu, zoom=[2,2,1], order=0)
        depth_heightmap_2x = ndimage.zoom(depth_heightmap_cpu, zoom=[2,2], order=0)
        assert(color_heightmap_2x.shape[0:2] == depth_heightmap_2x.shape[0:2])

        # Add extra padding (to handle rotations inside network)
        diag_length = float(color_heightmap_2x.shape[0]) * np.sqrt(2)
        diag_length = np.ceil(diag_length/32)*32
        padding_width = int((diag_length - color_heightmap_2x.shape[0])/2)
        color_heightmap_2x_r =  np.pad(color_heightmap_2x[:,:,0], padding_width, 'constant', constant_values=0)
        color_heightmap_2x_r.shape = (color_heightmap_2x_r.shape[0], color_heightmap_2x_r.shape[1], 1)
        color_heightmap_2x_g =  np.pad(color_heightmap_2x[:,:,1], padding_width, 'constant', constant_values=0)
        color_heightmap_2x_g.shape = (color_heightmap_2x_g.shape[0], color_heightmap_2x_g.shape[1], 1)
        color_heightmap_2x_b =  np.pad(color_heightmap_2x[:,:,2], padding_width, 'constant', constant_values=0)
        color_heightmap_2x_b.shape = (color_heightmap_2x_b.shape[0], color_heightmap_2x_b.shape[1], 1)
        color_heightmap_2x = np.concatenate((color_heightmap_2x_r, color_heightmap_2x_g, color_heightmap_2x_b), axis=2)
        depth_heightmap_2x =  np.pad(depth_heightmap_2x, padding_width, 'constant', constant_values=0)

        # Pre-process color image (scale and normalize)
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        input_color_image = color_heightmap_2x.astype(float)/255
        for c in range(3):
            input_color_image[:,:,c] = (input_color_image[:,:,c] - image_mean[c])/image_std[c]

        # Pre-process depth image (normalize)
        image_mean = [0.01, 0.01, 0.01]
        image_std = [0.03, 0.03, 0.03]

        # 添加调试信息，查看深度图的范围
        depth_min = np.min(depth_heightmap_2x)
        depth_max = np.max(depth_heightmap_2x)
        print(f"Depth range: {depth_min:.4f} - {depth_max:.4f}")

        # 如果深度图范围异常，进行修正
        if depth_max > 0.5:  # 如果深度超过预期范围
            depth_scale = 0.3 / depth_max  # 将最大深度缩放到0.3
            depth_heightmap_2x = depth_heightmap_2x * depth_scale
            print(f"Abnormal depth map range, scaling factor applied: {depth_scale:.4f}")
            print(f"Adjusted depth range: {np.min(depth_heightmap_2x):.4f} - {np.max(depth_heightmap_2x):.4f}")

        depth_heightmap_2x.shape = (depth_heightmap_2x.shape[0], depth_heightmap_2x.shape[1], 1)
        input_depth_image = np.concatenate((depth_heightmap_2x, depth_heightmap_2x, depth_heightmap_2x), axis=2)
        for c in range(3):
            input_depth_image[:,:,c] = (input_depth_image[:,:,c] - image_mean[c])/image_std[c]

        # Construct minibatch of size 1 (b,c,h,w)
        input_color_image.shape = (input_color_image.shape[0], input_color_image.shape[1], input_color_image.shape[2], 1)
        input_depth_image.shape = (input_depth_image.shape[0], input_depth_image.shape[1], input_depth_image.shape[2], 1)
        input_color_data = torch.from_numpy(input_color_image.astype(np.float32)).permute(3,2,0,1)
        input_depth_data = torch.from_numpy(input_depth_image.astype(np.float32)).permute(3,2,0,1)

        # 将数据移到模型所在设备
        input_color_data = input_color_data.to(target_device)
        input_depth_data = input_depth_data.to(target_device)
        print(f"输入数据已移动到目标设备")
        
        # 再次确保模型在正确设备上，防止意外的设备切换
        self.model = self.model.to(target_device)
        print(f"Forward方法中，当前设备: {'CUDA' if self.use_cuda else 'CPU'}")

        # Pass input data through model
        try:
            # 使用正确的梯度设置方式
            if is_volatile:
                with torch.no_grad():
                    model_output = self.model.forward(input_color_data, input_depth_data, is_volatile, specific_rotation)
            else:
                model_output = self.model.forward(input_color_data, input_depth_data, is_volatile, specific_rotation)
            
            # 处理不同返回值情况
            if isinstance(model_output, tuple):
                if len(model_output) == 3:
                    output_prob, interm_feat, state_feat = model_output
                    print("模型返回3个值：output_prob, interm_feat, state_feat")
                elif len(model_output) == 2:
                    output_prob, interm_feat = model_output
                    state_feat = None
                    print("Warning: model.forward只返回了两个值，缺少state_feat")
                else:
                    output_prob = model_output[0]
                    interm_feat = None
                    state_feat = None
                    print("Warning: model.forward返回值数量异常")
            else:
                # 如果不是元组，直接使用作为output_prob
                output_prob = model_output
                interm_feat = None
                state_feat = None
                print("Warning: model.forward没有返回元组")

            # 验证输出在正确的设备上
            if isinstance(output_prob, list) and len(output_prob) > 0:
                if isinstance(output_prob[0], list) and len(output_prob[0]) > 0:
                    output_device = output_prob[0][0].device if isinstance(output_prob[0][0], torch.Tensor) else None
                else:
                    output_device = output_prob[0].device if isinstance(output_prob[0], torch.Tensor) else None
                    
                if output_device is not None:
                    print(f"模型输出在设备: {'CUDA' if 'cuda' in str(output_device) else 'CPU'}")
                    
                    # 使用改进的设备比较逻辑
                    device_match = False
                    if self.use_cuda:
                        device_match = 'cuda' in str(output_device)
                    else:
                        device_match = (output_device == target_device)
                        
                    if not device_match:
                        print(f"警告: 输出设备与目标设备不一致，可能影响性能")
                
        except Exception as e:
            print(f"前向传播发生错误: {e}")
            # 打印更多错误信息
            print(f"输入数据设备: {'CUDA' if 'cuda' in str(input_color_data.device) else 'CPU'}")
            
            try:
                param_device = next(self.model.parameters()).device
                print(f"模型参数设备: {'CUDA' if 'cuda' in str(param_device) else 'CPU'}")
                
                # 使用改进的设备比较逻辑
                device_match = False
                if self.use_cuda:
                    device_match = 'cuda' in str(param_device) and 'cuda' in str(input_color_data.device)
                else:
                    device_match = (param_device == input_color_data.device)
                    
                # 如果设备不匹配，尝试解决
                if not device_match:
                    print("设备不匹配，尝试修复...")
                    input_color_data = input_color_data.to(param_device)
                    input_depth_data = input_depth_data.to(param_device)
                    try:
                        # 再次尝试
                        if is_volatile:
                            with torch.no_grad():
                                model_output = self.model.forward(input_color_data, input_depth_data, is_volatile, specific_rotation)
                        else:
                            model_output = self.model.forward(input_color_data, input_depth_data, is_volatile, specific_rotation)
                        
                        # 处理不同返回值情况
                        if isinstance(model_output, tuple):
                            if len(model_output) == 3:
                                output_prob, interm_feat, state_feat = model_output
                            elif len(model_output) == 2:
                                output_prob, interm_feat = model_output
                                state_feat = None
                                print("Warning: model.forward只返回了两个值，缺少state_feat")
                            else:
                                output_prob = model_output[0]
                                interm_feat = None
                                state_feat = None
                                print("Warning: model.forward返回值数量异常")
                        else:
                            # 如果不是元组，直接使用作为output_prob
                            output_prob = model_output
                            interm_feat = None
                            state_feat = None
                            print("Warning: model.forward没有返回元组")
                            
                        print("修复成功，前向传播完成!")
                    except Exception as e2:
                        print(f"修复后再次失败: {e2}")
                        import traceback
                        traceback.print_exc()
                        raise e2
            except StopIteration:
                print("警告: 无法获取模型参数设备")
                import traceback
                traceback.print_exc()
                raise e
            except Exception:
                import traceback
                traceback.print_exc()
                raise e

        if self.method == 'reactive':
            # Return affordances (and remove extra padding)
            for rotate_idx in range(len(output_prob)):
                if rotate_idx == 0:
                    push_predictions = F.softmax(output_prob[rotate_idx][0], dim=1).cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]
                    grasp_predictions = F.softmax(output_prob[rotate_idx][1], dim=1).cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]
                else:
                    push_predictions = np.concatenate((push_predictions, F.softmax(output_prob[rotate_idx][0], dim=1).cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]), axis=0)
                    grasp_predictions = np.concatenate((grasp_predictions, F.softmax(output_prob[rotate_idx][1], dim=1).cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]), axis=0)

        elif self.method == 'reinforcement':
            # Return Q values (and remove extra padding)
            for rotate_idx in range(len(output_prob)):
                if rotate_idx == 0:
                    push_predictions = output_prob[rotate_idx][0].cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]
                    grasp_predictions = output_prob[rotate_idx][1].cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]
                else:
                    push_predictions = np.concatenate((push_predictions, output_prob[rotate_idx][0].cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]), axis=0)
                    grasp_predictions = np.concatenate((grasp_predictions, output_prob[rotate_idx][1].cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]), axis=0)

        return push_predictions, grasp_predictions, state_feat


    def get_label_value(self, primitive_action, push_success, grasp_success, change_detected, push_predictions, grasp_predictions, next_color_heightmap, next_depth_heightmap):
        # 確保預測值在CPU上
        if isinstance(push_predictions, torch.Tensor):
            push_predictions = push_predictions.detach().cpu().numpy()
        if isinstance(grasp_predictions, torch.Tensor):
            grasp_predictions = grasp_predictions.detach().cpu().numpy()

        # 輸出詳細的輸入參數信息，便於調試
        print("============ Reward Calculation Details ============")
        print(f"Action type: {primitive_action} ({'推动' if primitive_action == 0 or primitive_action == 'push' else '抓取'})")
        print(f"Push success: {push_success}")
        print(f"Grasp success: {grasp_success}")
        print(f"Scene change detected: {change_detected}")
        print("===================================================")

        # 计算当前奖励
        current_reward = 0
        
        # 转换为统一形式处理
        is_push = primitive_action == 0 or primitive_action == 'push'
        is_grasp = primitive_action == 1 or primitive_action == 'grasp'
        
        if is_push:
            # 推动动作奖励逻辑增强
            if push_success:
                print("推动成功，设置高奖励值")
                current_reward = 0.5  # 提高推动成功的奖励
            elif change_detected:
                print("检测到场景变化，设置基础奖励")
                current_reward = 0.2
            else:
                print("推动无效果，设置零奖励")
                current_reward = 0.0
        elif is_grasp:
            if grasp_success:
                print("抓取成功，设置高奖励值")
                current_reward = 1.0
                # 在测试模式下立即返回高奖励，不考虑未来奖励
                if hasattr(self, 'is_testing') and self.is_testing:
                    print("测试模式：直接返回抓取成功奖励: 1.0")
                    # 确保reward_value_log存在且为列表
                    if not hasattr(self, 'reward_value_log'):
                        self.reward_value_log = []
                    self.reward_value_log.append([float(current_reward)])
                    return 1.0, 1.0
            else:
                print("抓取失败，设置零奖励")
                current_reward = 0.0

        # 计算未来奖励
        if self.method == 'reinforcement':
            # 记录初始奖励，用于展示真实奖励值
            initial_reward = current_reward
            
            # 确保相关列表存在
            if not hasattr(self, 'reward_value_log'):
                self.reward_value_log = []
            if not hasattr(self, 'predicted_value_log'):
                self.predicted_value_log = []
            if not hasattr(self, 'label_value_log'):
                self.label_value_log = []
            
            # 检查模型是否需要重新初始化 - 改为调用可能存在的initialize_model方法
            if getattr(self, 'reinit_flag', False):
                print("模型需要重新初始化...")
                if hasattr(self, 'initialize_model'):
                    self.initialize_model()
                self.reinit_flag = False
                
            future_reward = 0.0
            
            # 对于成功动作，添加最小未来奖励基线
            min_future_reward = 0.0
            if push_success or grasp_success:
                min_future_reward = 0.1  # 成功动作的最小未来价值
                
            try:
                # 计算下一个状态的最大预期回报
                next_push_predictions, next_grasp_predictions, next_state_feat = self.forward(next_color_heightmap, next_depth_heightmap, is_volatile=True)
                
                if is_push:
                    future_reward = max(np.max(next_push_predictions), np.max(next_grasp_predictions))
                    predicted_value = np.max(next_push_predictions)
                else:
                    future_reward = max(np.max(next_push_predictions), np.max(next_grasp_predictions))
                    predicted_value = np.max(next_grasp_predictions)
                
                # 确保未来奖励不低于成功动作的最小基线
                future_reward = max(future_reward, min_future_reward)
                
                # Apply discount factor to future rewards
                future_reward = future_reward * self.future_reward_discount
                
                print(f"计算的未来预期回报: {future_reward:.6f}")
                
                # 对于强化学习方法，总回报是当前奖励加上折扣的未来奖励
                current_reward = current_reward + future_reward
                
                # 记录日志
                self.predicted_value_log.append([float(predicted_value)])
            except Exception as e:
                print(f"计算未来奖励时出错: {e}")
                traceback.print_exc()
                self.predicted_value_log.append([0.0])
            
            # 记录奖励值和标签值
            self.reward_value_log.append([float(initial_reward)])  # 记录实际奖励，不包括折扣未来回报
            
            # 输出详细奖励信息
            print(f"初始奖励: {initial_reward:.6f}, 未来折扣回报: {future_reward:.6f}, 总计标签值: {current_reward:.6f}")
        else:
            # 对于非强化学习方法，总回报就是当前奖励
            # 确保reward_value_log存在
            if not hasattr(self, 'reward_value_log'):
                self.reward_value_log = []
            self.reward_value_log.append([float(current_reward)])
            
        # 最终标签值等于总回报(当前+未来)
        label_value = current_reward
        
        # 记录日志
        if not hasattr(self, 'label_value_log'):
            self.label_value_log = []
        self.label_value_log.append([float(label_value)])
        
        return label_value, current_reward


    # Compute labels and backpropagate
    def backprop(self, color_heightmap, depth_heightmap, primitive_action, best_pix_ind, label_value):
        # 确保模型在正确设备上
        target_device = self.reset_device()
        print(f"backprop方法中，当前模型设备: {target_device}")
        
        # 确保模型所有部分都在同一设备上
        if hasattr(self.model, '_verify_device'):
            device = self.model._verify_device()
            print(f"验证并确保模型在统一设备上: {device}")
        
        # 确保输入数据在正确设备上并转换为正确格式
        if isinstance(color_heightmap, np.ndarray):
            color_heightmap = torch.from_numpy(color_heightmap.astype(np.float32))
        if isinstance(depth_heightmap, np.ndarray):
            depth_heightmap = torch.from_numpy(depth_heightmap.astype(np.float32))
            
        # 处理维度问题
        if len(color_heightmap.shape) == 3 and color_heightmap.shape[2] == 3:  # HWC格式
            color_heightmap = color_heightmap.permute(2, 0, 1)  # 转为CHW
        if len(depth_heightmap.shape) == 2:  # 单通道深度图
            depth_heightmap = depth_heightmap.unsqueeze(0)  # 添加通道维度
            
        # 添加批次维度（如果缺少）
        if len(color_heightmap.shape) == 3:  # CHW格式
            color_heightmap = color_heightmap.unsqueeze(0)  # 变为NCHW
        if len(depth_heightmap.shape) == 3 and depth_heightmap.shape[0] == 1:  # 单通道
            depth_heightmap = depth_heightmap.unsqueeze(0)  # 变为NCHW
            
        # 移动数据到正确设备
        if color_heightmap.device != target_device:
            color_heightmap = color_heightmap.to(target_device)
        if depth_heightmap.device != target_device:
            depth_heightmap = depth_heightmap.to(target_device)
            
        # 打印数据形状和设备以便调试
        print(f"输入数据形状 - 颜色: {color_heightmap.shape}, 深度: {depth_heightmap.shape}")
        print(f"输入数据设备 - 颜色: {color_heightmap.device}, 深度: {depth_heightmap.device}")
        
        # 转换primitive_action为字符串形式，确保与self.method == 'reinforcement'条件兼容
        action_type = 'push' if primitive_action == 0 or primitive_action == 'push' else 'grasp'
        print(f"动作类型: {action_type}")

        if self.method == 'reinforcement':
            try:
                # 计算标签
                label = np.zeros((1,320,320))
                action_area = np.zeros((224,224))
                action_area[best_pix_ind[1]][best_pix_ind[2]] = 1
                tmp_label = np.zeros((224,224))
                tmp_label[action_area > 0] = label_value
                label[0,48:(320-48),48:(320-48)] = tmp_label

                # 计算标签权重
                label_weights = np.zeros(label.shape)
                tmp_label_weights = np.zeros((224,224))
                tmp_label_weights[action_area > 0] = 1
                label_weights[0,48:(320-48),48:(320-48)] = tmp_label_weights

                # 将numpy数组转换为张量并移动到正确的设备
                label_tensor = torch.from_numpy(label).float().to(target_device)
                label_weights_tensor = torch.from_numpy(label_weights).float().to(target_device)
                
                print(f"标签张量形状: {label_tensor.shape}, 设备: {label_tensor.device}")
                print(f"权重张量形状: {label_weights_tensor.shape}, 设备: {label_weights_tensor.device}")

                # 清除梯度
                self.optimizer.zero_grad()
                loss_value = 0
                
                # 确保模型在训练模式
                self.model.train()
                
                print(f"执行前向传播，特定旋转角度索引: {best_pix_ind[0]}")
                
                # 对模型进行前向传播
                output_prob, interm_feat, state_feat = self.forward(
                    color_heightmap, 
                    depth_heightmap, 
                    is_volatile=False, 
                    specific_rotation=best_pix_ind[0]
                )
                
                print(f"前向传播完成")
                
                # 获取模型输出
                model_output = None
                # 首先尝试从类属性获取输出
                if hasattr(self.model, 'output_prob') and self.model.output_prob is not None and len(self.model.output_prob) > 0:
                    model_output = self.model.output_prob
                    print(f"使用模型的output_prob属性: {len(model_output)}个元素")
                # 如果类属性不可用，使用forward返回值
                elif output_prob is not None and len(output_prob) > 0:
                    model_output = output_prob
                    print(f"使用forward返回的output_prob: {len(model_output)}个元素")
                else:
                    raise ValueError("无法获取模型输出，model.output_prob和forward返回值均为空")
                
                # 确保模型输出不为空
                if model_output is None or len(model_output) == 0:
                    raise ValueError("模型输出为空，无法计算损失")
                
                # 根据动作类型计算损失并反向传播
                if action_type == 'push':
                    try:
                        # 确保索引有效
                        if not model_output or len(model_output[0]) < 1:
                            raise IndexError("模型输出索引无效，无法访问推动输出")
                        
                        # 确保张量在正确的设备上
                        push_output = model_output[0][0].to(target_device)
                        
                        # 打印输出形状以便调试
                        print(f"推动输出形状: {push_output.shape}")
                        
                        # 检查形状是否匹配
                        if push_output.shape != torch.Size([1, 1, 320, 320]):
                            print(f"警告: 推动输出形状 {push_output.shape} 与预期的 [1, 1, 320, 320] 不匹配")
                            # 尝试调整形状
                            if len(push_output.shape) == 4:
                                # 已经是4D，可能只需要调整大小
                                push_output = F.interpolate(push_output, size=(320, 320), mode='bilinear', align_corners=True)
                            elif len(push_output.shape) == 3:
                                # 可能缺少批次维度
                                push_output = push_output.unsqueeze(0)
                                if push_output.shape != torch.Size([1, 1, 320, 320]):
                                    push_output = F.interpolate(push_output, size=(320, 320), mode='bilinear', align_corners=True)
                        
                        # 计算损失
                        loss = self.criterion(
                            push_output.view(1,320,320), 
                            label_tensor
                        ) * label_weights_tensor
                        
                        # 求和并反向传播
                        loss = loss.sum()
                        loss.backward()
                        loss_value = loss.detach().cpu().numpy()
                        
                        print(f"推动动作损失: {loss_value}")
                    except Exception as e:
                        print(f"计算推动损失时出错: {e}")
                        import traceback
                        traceback.print_exc()
                        raise

                elif action_type == 'grasp':
                    try:
                        # 确保索引有效
                        if not model_output or len(model_output[0]) < 2:
                            raise IndexError("模型输出索引无效，无法访问抓取输出")
                        
                        # 确保张量在正确的设备上
                        grasp_output = model_output[0][1].to(target_device)
                        
                        # 打印输出形状以便调试
                        print(f"抓取输出形状: {grasp_output.shape}")
                        
                        # 检查形状是否匹配
                        if grasp_output.shape != torch.Size([1, 1, 320, 320]):
                            print(f"警告: 抓取输出形状 {grasp_output.shape} 与预期的 [1, 1, 320, 320] 不匹配")
                            # 尝试调整形状
                            if len(grasp_output.shape) == 4:
                                # 已经是4D，可能只需要调整大小
                                grasp_output = F.interpolate(grasp_output, size=(320, 320), mode='bilinear', align_corners=True)
                            elif len(grasp_output.shape) == 3:
                                # 可能缺少批次维度
                                grasp_output = grasp_output.unsqueeze(0)
                                if grasp_output.shape != torch.Size([1, 1, 320, 320]):
                                    grasp_output = F.interpolate(grasp_output, size=(320, 320), mode='bilinear', align_corners=True)
                        
                        # 计算损失
                        loss = self.criterion(
                            grasp_output.view(1,320,320), 
                            label_tensor
                        ) * label_weights_tensor
                        
                        # 求和并反向传播
                        loss = loss.sum()
                        loss.backward()
                        loss_value = loss.detach().cpu().numpy()
                        
                        print(f"抓取动作损失: {loss_value}")

                        # 由于抓取是对称的，使用相反的旋转角度进行另一次前向传播
                        opposite_rotate_idx = int((best_pix_ind[0] + self.model.num_rotations/2) % self.model.num_rotations)

                        print(f"执行对称抓取的前向传播，旋转角度索引: {opposite_rotate_idx}")
                        output_prob, interm_feat, state_feat = self.forward(
                            color_heightmap, 
                            depth_heightmap, 
                            is_volatile=False, 
                            specific_rotation=opposite_rotate_idx
                        )
                        
                        # 再次获取输出概率
                        if hasattr(self.model, 'output_prob') and len(self.model.output_prob) > 0:
                            model_output = self.model.output_prob
                        else:
                            model_output = output_prob
                        
                        # 确保输出有效
                        if model_output is None or len(model_output) == 0:
                            raise ValueError("对称抓取的输出为空，无法计算损失")
                        
                        # 确保张量在正确的设备上
                        grasp_output = model_output[0][1].to(target_device)
                        print(f"对称抓取输出形状: {grasp_output.shape}")

                        # 计算损失
                        loss = self.criterion(
                            grasp_output.view(1,320,320), 
                            label_tensor
                        ) * label_weights_tensor

                        # 求和并反向传播
                        loss = loss.sum()
                        loss.backward()
                        
                        # 计算平均损失
                        additional_loss_value = loss.detach().cpu().numpy()
                        loss_value = (loss_value + additional_loss_value) / 2
                        
                        print(f"对称抓取动作损失: {additional_loss_value}, 平均损失: {loss_value}")
                    except Exception as e:
                        print(f"计算抓取损失时出错: {e}")
                        import traceback
                        traceback.print_exc()
                        # 关键错误，重新抛出
                        raise

                print(f'训练损失: {loss_value}')
                
                # 更新模型参数
                self.optimizer.step()
                
                # 返回损失值以便调用者使用
                return loss_value
                
            except Exception as e:
                print(f"反向传播过程中发生错误: {e}")
                print(f"错误类型: {type(e).__name__}")
                import traceback
                traceback.print_exc()
                # 在出错时，尝试重置优化器状态
                self.optimizer.zero_grad()
                raise


    def get_prediction_vis(self, predictions, color_heightmap, best_pix_ind):
        # 确保模型在正确设备上
        self.reset_device()
        canvas = None
        num_rotations = predictions.shape[0]
        for canvas_row in range(int(num_rotations/4)):
            tmp_row_canvas = None
            for canvas_col in range(4):
                rotate_idx = canvas_row*4+canvas_col
                prediction_vis = predictions[rotate_idx,:,:].copy()
                prediction_vis = np.clip(prediction_vis, 0, 1)
                prediction_vis.shape = (predictions.shape[1], predictions.shape[2])
                prediction_vis = cv2.applyColorMap((prediction_vis*255).astype(np.uint8), cv2.COLORMAP_JET)
                if best_pix_ind is not None and rotate_idx == best_pix_ind[0]:
                    prediction_vis = cv2.circle(prediction_vis, (int(best_pix_ind[2]), int(best_pix_ind[1])), 7, (0,0,255), 2)
                prediction_vis = ndimage.rotate(prediction_vis, rotate_idx*(360.0/num_rotations), reshape=False, order=0)
                background_image = ndimage.rotate(color_heightmap, rotate_idx*(360.0/num_rotations), reshape=False, order=0)
                prediction_vis = (0.5*cv2.cvtColor(background_image, cv2.COLOR_RGB2BGR) + 0.5*prediction_vis).astype(np.uint8)
                if tmp_row_canvas is None:
                    tmp_row_canvas = prediction_vis
                else:
                    tmp_row_canvas = np.concatenate((tmp_row_canvas,prediction_vis), axis=1)
            if canvas is None:
                canvas = tmp_row_canvas
            else:
                canvas = np.concatenate((canvas,tmp_row_canvas), axis=0)

        return canvas


    def push_heuristic(self, depth_heightmap):
        # 确保模型在正确设备上
        self.reset_device()
        num_rotations = 16

        for rotate_idx in range(num_rotations):
            rotated_heightmap = ndimage.rotate(depth_heightmap, rotate_idx*(360.0/num_rotations), reshape=False, order=0)
            valid_areas = np.zeros(rotated_heightmap.shape)
            valid_areas[ndimage.interpolation.shift(rotated_heightmap, [0,-25], order=0) - rotated_heightmap > 0.02] = 1
            # valid_areas = np.multiply(valid_areas, rotated_heightmap)
            blur_kernel = np.ones((25,25),np.float32)/9
            valid_areas = cv2.filter2D(valid_areas, -1, blur_kernel)
            tmp_push_predictions = ndimage.rotate(valid_areas, -rotate_idx*(360.0/num_rotations), reshape=False, order=0)
            tmp_push_predictions.shape = (1, rotated_heightmap.shape[0], rotated_heightmap.shape[1])

            if rotate_idx == 0:
                push_predictions = tmp_push_predictions
            else:
                push_predictions = np.concatenate((push_predictions, tmp_push_predictions), axis=0)

        best_pix_ind = np.unravel_index(np.argmax(push_predictions), push_predictions.shape)
        return best_pix_ind


    def grasp_heuristic(self, depth_heightmap):
        # 确保模型在正确设备上
        self.reset_device()
        num_rotations = 16

        for rotate_idx in range(num_rotations):
            rotated_heightmap = ndimage.rotate(depth_heightmap, rotate_idx*(360.0/num_rotations), reshape=False, order=0)
            valid_areas = np.zeros(rotated_heightmap.shape)
            valid_areas[np.logical_and(rotated_heightmap - ndimage.interpolation.shift(rotated_heightmap, [0,-25], order=0) > 0.02, rotated_heightmap - ndimage.interpolation.shift(rotated_heightmap, [0,25], order=0) > 0.02)] = 1
            # valid_areas = np.multiply(valid_areas, rotated_heightmap)
            blur_kernel = np.ones((25,25),np.float32)/9
            valid_areas = cv2.filter2D(valid_areas, -1, blur_kernel)
            tmp_grasp_predictions = ndimage.rotate(valid_areas, -rotate_idx*(360.0/num_rotations), reshape=False, order=0)
            tmp_grasp_predictions.shape = (1, rotated_heightmap.shape[0], rotated_heightmap.shape[1])

            if rotate_idx == 0:
                grasp_predictions = tmp_grasp_predictions
            else:
                grasp_predictions = np.concatenate((grasp_predictions, tmp_grasp_predictions), axis=0)

        best_pix_ind = np.unravel_index(np.argmax(grasp_predictions), grasp_predictions.shape)
        return best_pix_ind


    def experience_replay(self):
        """经验回放功能的实现，基于保存的动作和奖励日志选择和学习历史经验"""
        # 确保模型在正确设备上
        self.reset_device()
        
        print("Executing experience replay...")
        
        # 确保logs是列表而非numpy数组
        if isinstance(self.executed_action_log, np.ndarray):
            self.executed_action_log = self.executed_action_log.tolist()
        if isinstance(self.reward_value_log, np.ndarray):
            self.reward_value_log = self.reward_value_log.tolist()
        
        # 确保长度一致
        if len(self.executed_action_log) != len(self.reward_value_log):
            print(f"Warning: Executed action log ({len(self.executed_action_log)}) and reward log ({len(self.reward_value_log)}) lengths do not match")
            # 对齐长度
            min_length = min(len(self.executed_action_log), len(self.reward_value_log))
            self.executed_action_log = self.executed_action_log[:min_length]
            self.reward_value_log = self.reward_value_log[:min_length]
        
        if self.iteration <= 1:
            print("Insufficient training iterations, skipping experience replay")
            return
        
        # 随机选择动作类型（push或grasp）作为样本
        if np.random.random() < 0.5:
            sample_primitive_action = 'push'
            sample_primitive_action_id = 0
        else:
            sample_primitive_action = 'grasp'
            sample_primitive_action_id = 1
        
        # 为选定的动作类型选择与当前结果相反的奖励值
        # 假设当前reward_value_log最后一个元素是最近一次动作的奖励
        if len(self.reward_value_log) > 0:
            prev_reward_value = self.reward_value_log[-1][0]
            sample_reward_value = 0 if prev_reward_value > 0.5 else 1
        else:
            sample_reward_value = np.random.randint(0, 2)
        
        print(f"Experience replay selection: Action={sample_primitive_action}, Target reward={sample_reward_value}")
        
        # 查找具有相同动作类型但不同结果的样本
        try:
            sample_ind = np.argwhere(np.logical_and(
                np.asarray(self.reward_value_log)[1:self.iteration,0] == sample_reward_value, 
                np.asarray(self.executed_action_log)[1:self.iteration,0] == sample_primitive_action_id
            ))
        except ValueError as e:
            print(f"Error: Cannot compare reward_value_log and executed_action_log, their shapes don't match: {e}")
            print("Skipping experience replay")
            return
        
        if sample_ind.size > 0:
            # 找到具有最高惊奇值的样本
            if self.method == 'reactive':
                sample_surprise_values = np.abs(np.asarray(self.predicted_value_log)[sample_ind[:,0]] - (1 - sample_reward_value))
            elif self.method == 'reinforcement':
                sample_surprise_values = np.abs(np.asarray(self.predicted_value_log)[sample_ind[:,0]] - np.asarray(self.label_value_log)[sample_ind[:,0]])
            
            # 基于惊奇值排序样本
            sorted_surprise_ind = np.argsort(sample_surprise_values[:,0])
            sorted_sample_ind = sample_ind[sorted_surprise_ind,0]
            
            # 使用幂律分布随机选择样本
            pow_law_exp = 2
            rand_sample_ind = int(np.round(np.random.power(pow_law_exp, 1)*(sample_ind.size-1)))
            sample_iteration = sorted_sample_ind[rand_sample_ind]
            
            print(f'Experience replay: Iteration {sample_iteration} (Surprise value: {sample_surprise_values[sorted_surprise_ind[rand_sample_ind]]})')
            
            # 跳过第一个样本(iteration 0)
            if sample_iteration == 0:
                print('Skipping first sample (iteration 0), as it may not have corresponding image files')
                return
            
            try:
                # 导入必要的模块
                import os
                import cv2
                
                # 获取Logger实例
                logger = self.logger if hasattr(self, 'logger') else None
                if logger is None:
                    from logger import Logger
                    import glob
                    # 寻找日志目录
                    log_dirs = glob.glob("logs-*")
                    if log_dirs:
                        logger = Logger(True, log_dirs[0])
                    else:
                        print("Cannot find log directory, skipping experience replay")
                        return
                
                # 加载样本RGB-D高度图
                color_heightmap_path = os.path.join(logger.color_heightmaps_directory, '%06d.0.color.png' % (sample_iteration))
                sample_color_heightmap = cv2.imread(color_heightmap_path)
                
                # 检查图像是否成功加载
                if sample_color_heightmap is None:
                    print(f'Warning: Cannot load image file for sample {sample_iteration}: {color_heightmap_path}')
                    return
                    
                sample_color_heightmap = cv2.cvtColor(sample_color_heightmap, cv2.COLOR_BGR2RGB)
                
                depth_heightmap_path = os.path.join(logger.depth_heightmaps_directory, '%06d.0.depth.png' % (sample_iteration))
                sample_depth_heightmap = cv2.imread(depth_heightmap_path, -1)
                
                # 检查深度图像是否成功加载
                if sample_depth_heightmap is None:
                    print(f'Warning: Cannot load depth image file for sample {sample_iteration}: {depth_heightmap_path}')
                    return
                    
                sample_depth_heightmap = sample_depth_heightmap.astype(np.float32)/100000
                
                # 获取当前样本的标签和动作信息
                sample_label_value = self.label_value_log[sample_iteration][0]
                sample_best_pix_ind = (np.asarray(self.executed_action_log)[sample_iteration,1:4]).astype(int)
                
                # 确定动作类型
                sample_primitive_action_id = self.executed_action_log[sample_iteration][0]
                sample_primitive_action = 'push' if sample_primitive_action_id == 0 else 'grasp'
                
                # 反向传播
                print(f"Performing backpropagation on sample {sample_iteration}, action type: {sample_primitive_action}")
                self.backprop(sample_color_heightmap, sample_depth_heightmap, 
                              sample_primitive_action, sample_best_pix_ind, sample_label_value)
                
                print(f"Experience replay completed: Sample {sample_iteration}")
                
            except Exception as e:
                print(f"Error during experience replay: {e}")
        else:
            print('Not enough previous training samples. Skipping experience replay.')
