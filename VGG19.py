import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
import time, sys



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class VGG19(nn.Module):
    '''
    VGG19 model with 16 conv layers in total, with maxpooling replaced by average pooling and without 3 FC layers,
    '''
    def __init__(self):
        super(VGG19, self).__init__()

        # the shape of each element of feature maps is [batch_size, channel, height, width] 
        self.features_maps = dict()
        
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu1_1 = nn.ReLU()
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu1_2 = nn.ReLU()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu2_1 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.relu2_2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.relu3_1 = nn.ReLU()
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu3_2 = nn.ReLU()
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu3_3 = nn.ReLU()
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu3_4 = nn.ReLU()
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.relu4_1 = nn.ReLU()
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.relu4_2 = nn.ReLU()
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.relu4_3 = nn.ReLU()
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.relu4_4 = nn.ReLU()
        self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.relu5_1 = nn.ReLU()
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.relu5_2 = nn.ReLU()
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.relu5_3 = nn.ReLU()
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.relu5_4 = nn.ReLU()
        self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)


    def forward(self, x: torch.Tensor):
        # reset feature maps at the beginning of forward pass
        self.features_maps = dict()
        temp = x.clone()

        def normalize_activations(activations):
            """
            Rescaling the weights in the network such that the mean activation of each filter over images and positions is equal to one. 
            """
            mean_activation = activations.mean(dim=[0, 2, 3], keepdim=True)  # Compute mean over batch, height, width
            return activations / (mean_activation + 1e-8)  # Normalize by mean activation

        x = self.conv1_1(temp)
        x = self.relu1_1(x)
        # x = normalize_activations(x) 
        self.features_maps["conv1_1"] = x
        x = self.conv1_2(x)
        x = self.relu1_2(x)
        # x = normalize_activations(x) 
        self.features_maps["conv1_2"] = x
        x = self.pool1(x)
        self.features_maps["pool1"] = x
        
        x = self.conv2_1(x)
        x = self.relu2_1(x)
        # x = normalize_activations(x) 
        self.features_maps["conv2_1"] = x
        x = self.conv2_2(x)
        x = self.relu2_2(x)
        # x = normalize_activations(x) 
        self.features_maps["conv2_2"] = x
        x = self.pool2(x)
        self.features_maps["pool2"] = x
        
        x = self.conv3_1(x)
        x = self.relu3_1(x)
        # x = normalize_activations(x)
        self.features_maps["conv3_1"] = x
        x = self.conv3_2(x)
        x = self.relu3_2(x)
        # x = normalize_activations(x)
        self.features_maps["conv3_2"] = x
        x = self.conv3_3(x)
        x = self.relu3_3(x)
        # x = normalize_activations(x)
        self.features_maps["conv3_3"] = x
        x = self.conv3_4(x)
        x = self.relu3_4(x)
        # x = normalize_activations(x)
        self.features_maps["conv3_4"] = x
        x = self.pool3(x)
        self.features_maps["pool3"] = x
        
        x = self.conv4_1(x)
        x = self.relu4_1(x)
        # x = normalize_activations(x)
        self.features_maps["conv4_1"] = x
        x = self.conv4_2(x)
        x = self.relu4_2(x)
        # x = normalize_activations(x)
        self.features_maps["conv4_2"] = x
        x = self.conv4_3(x)
        x = self.relu4_3(x)
        # x = normalize_activations(x)
        self.features_maps["conv4_3"] = x
        x = self.conv4_4(x)
        x = self.relu4_4(x)
        # x = normalize_activations(x)
        self.features_maps["conv4_4"] = x
        x = self.pool4(x)
        self.features_maps["pool4"] = x
        
        x = self.conv5_1(x)
        x = self.relu5_1(x)
        # x = normalize_activations(x)
        self.features_maps["conv5_1"] = x
        x = self.conv5_2(x)
        x = self.relu5_2(x)
        # x = normalize_activations(x)
        self.features_maps["conv5_2"] = x
        x = self.conv5_3(x)
        x = self.relu5_3(x)
        # x = normalize_activations(x)
        self.features_maps["conv5_3"] = x
        x = self.conv5_4(x)
        x = self.relu5_4(x)
        # x = normalize_activations(x)
        self.features_maps["conv5_4"] = x
        x = self.pool5(x)
        self.features_maps["pool5"] = x
        
        return x
    
def get_vgg19(pooling='avg'):

    layer_map = {
        'conv1_1.weight': 'features.0.weight',
        'conv1_1.bias': 'features.0.bias',
        'conv1_2.weight': 'features.2.weight',
        'conv1_2.bias': 'features.2.bias',
        'conv2_1.weight': 'features.5.weight',
        'conv2_1.bias': 'features.5.bias',
        'conv2_2.weight': 'features.7.weight',
        'conv2_2.bias': 'features.7.bias',
        'conv3_1.weight': 'features.10.weight',
        'conv3_1.bias': 'features.10.bias',
        'conv3_2.weight': 'features.12.weight',
        'conv3_2.bias': 'features.12.bias',
        'conv3_3.weight': 'features.14.weight',
        'conv3_3.bias': 'features.14.bias',
        'conv3_4.weight': 'features.16.weight',
        'conv3_4.bias': 'features.16.bias',
        'conv4_1.weight': 'features.19.weight',
        'conv4_1.bias': 'features.19.bias',
        'conv4_2.weight': 'features.21.weight',
        'conv4_2.bias': 'features.21.bias',
        'conv4_3.weight': 'features.23.weight',
        'conv4_3.bias': 'features.23.bias',
        'conv4_4.weight': 'features.25.weight',
        'conv4_4.bias': 'features.25.bias',
        'conv5_1.weight': 'features.28.weight',
        'conv5_1.bias': 'features.28.bias',
        'conv5_2.weight': 'features.30.weight',
        'conv5_2.bias': 'features.30.bias',
        'conv5_3.weight': 'features.32.weight',
        'conv5_3.bias': 'features.32.bias',
        'conv5_4.weight': 'features.34.weight',
        'conv5_4.bias': 'features.34.bias',
    }
    model = VGG19()
    pretrained_weights = models.vgg19(pretrained=True).state_dict()
    adapted_weights = {}

    for key in model.state_dict().keys():
        adapted_weights[key] = pretrained_weights[layer_map[key]]
    # print(adapted_weights.keys())
    model.load_state_dict(adapted_weights)

    if(pooling == 'max'):
        model.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        model.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        model.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        model.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        model.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
    return model

def rescale_weights(model,img_list:list,device=device):
    '''
    Rescaling the weights in the network such that the mean activation of each filter over images and positions is equal to one. 
    '''
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        layers=[]
        for name, module in model.named_modules():
            layers.append(module)
        layers.pop(0)
        for i,layer in enumerate(layers):
            if isinstance(layer, nn.Conv2d):
                total_activation = 0.0
                total_num = 0
                for img in img_list:
                    img = img.to(device)
                    for j in range(i+2):
                        img = layers[j](img)
                    total_activation += img.sum().item()
                    total_num += img.numel()
                    
                mean_activation = total_activation / total_num
                layer.weight.data /= mean_activation  
                layer.bias.data /= mean_activation
    return model

# if __name__ == '__main__':

#     model = get_vgg19('avg').to(device)
#     print(model)
#     # 随机生成一张图片
#     img = torch.rand(1,3,224,224).to(device)
#     model=rescale_weights(model,[img])
#     model(img)
#     feature_maps = model.features_maps
#     print(f'feature_maps in different layer: {feature_maps.keys()}')
#     for value in feature_maps.values():
#         print(value.mean())