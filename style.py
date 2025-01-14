import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights
import numpy as np
from PIL import Image
from Calculate_Gram import cal_gram


class ColorStyleTransfer(nn.Module):
    def __init__(self, content_weight=1.0, style_weight=1e6):
        super(ColorStyleTransfer, self).__init__()
        self.content_weight = content_weight  # 内容损失的权重
        self.style_weight = style_weight      # 风格损失的权重

        # 定义用于计算内容和风格损失的层
        self.content_layers = ['conv1_2', 'conv2_2', 'conv3_2', 'conv4_2', 'conv5_2']  # 内容损失层
        self.style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']  # 风格损失层

    def compute_content_loss(self, content_features, generated_features):
        """计算内容损失"""
        content_loss = 0
        for layer in self.content_layers:
            content_loss += F.mse_loss(content_features[layer], generated_features[layer])
        return content_loss

    def compute_style_loss(self, style_features, generated_features):
        """计算风格损失"""
        style_loss = 0
        for layer in self.style_layers:
            generated_gram = cal_gram(generated_features[layer])
            style_gram = cal_gram(style_features[layer])
            style_loss += F.mse_loss(generated_gram, style_gram)/(style_gram.shape[0]**2)
        return style_loss

    def forward(self, target_features, refer_features):
        """前向传播"""
        style_refer, content_refer = refer_features

        # 计算内容损失和风格损失
        content_loss = self.compute_content_loss(content_refer, target_features)
        style_loss = self.compute_style_loss(style_refer, target_features)

        # 总损失
        total_loss = self.content_weight * content_loss + self.style_weight * style_loss
        return total_loss

