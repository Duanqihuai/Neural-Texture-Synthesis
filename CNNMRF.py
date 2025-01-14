import torch
import torch.nn.functional as F
from style import ColorStyleTransfer


def feature_normalize(feature):
    feature_norm = torch.norm(feature, 2, 1, keepdim=True) + 1e-5 #L2范数
    feature_normalize = torch.div(feature, feature_norm)
    return feature_norm, feature_normalize


def L2_distance(x, y):
    """
    计算两个特征图之间的L2距离(欧氏距离)。

    参数:
        x (torch.Tensor): 第一个特征图，形状为 [N, C, H1, W1]。
        y (torch.Tensor): 第二个特征图，形状为 [N, C, H2, W2]。

    返回:
        distance (torch.Tensor): L2距离矩阵,形状为 [N, H1*W1, H2*W2]。
    """
    N, C, Hx, Wx = x.shape
    _, _, Hy, Wy = y.shape
    x_flat = x.view(N, C, -1)
    y_flat = y.view(N, C, -1)

    x_norm = torch.sum(x_flat**2, dim=1)
    y_norm = torch.sum(y_flat**2, dim=1)

    dot_product = torch.matmul(y_flat.transpose(1,2), x_flat)

    distance = (
        y_norm.unsqueeze(2) - 2 * dot_product + x_norm.unsqueeze(1)
    )

    distance = distance.transpose(1, 2).reshape(N, Hx * Wx, Hy * Wy)
    distance = torch.clamp(distance, min=0.0) / C

    return distance
def cosine_distance(x, y):
    """
    计算两个特征图之间的余弦距离。

    参数:
        x (torch.Tensor): 第一个特征图，形状为 [N, C, H, W]，其中：
            - N 是批量大小。
            - C 是通道数。
            - H 和 W 是高度和宽度。
        y (torch.Tensor): 第二个特征图，形状为 [N, C, H, W]。

    返回:
        distance (torch.Tensor): 余弦距离矩阵，形状为 [N, H*W, H*W]。
    """
    N, C, _, _= x.shape

    y_mean = y.view(N, C, -1).mean(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
    _, x_norm = feature_normalize(x - y_mean)  # batch_size * feature_depth * feature_size * feature_size
    _, y_norm = feature_normalize(y - y_mean)  
    x_norm = x_norm.view(N, C, -1)
    y_norm = y_norm.view(N, C, -1)
    x_T = x_norm.transpose(1, 2)  # batch_size * feature_size^2 * feature_depth

    # cosine distance = 1 - similarity
    cosine = x_T @ y_norm
    distance = (1 - cosine) / 2  # batch_size * feature_size^2 * feature_size^2
    distance = distance.clamp(min=0.)

    return distance
def orient_distance(x, y, patch_size):
    """
    计算两个方向图之间的方向距离。

    参数:
        x (torch.Tensor): 第一个方向图，形状为 [Nx, 2, Hx, Wx]，其中：
            - Nx 是批量大小。
            - 2 表示方向向量的两个分量（如 x 和 y 方向）。
            - Hx 和 Wx 是高度和宽度。
        y (torch.Tensor): 第二个方向图，形状为 [Ny, 2, Hy, Wy]。
        patch_size (int): 方向图的块大小。

    返回:
        distance (torch.Tensor): 方向距离矩阵。
    """
    Nx, _, Hx, Wx = x.shape
    Ny, _, Hy, Wy = y.shape

    x = x.view(Nx, 2, patch_size ** 2, Hx, Wx)
    y = y.view(Ny, 2, patch_size ** 2, Hy, Wy)

    distance = 0
    for i in range(patch_size ** 2):
        distance += torch.min(L2_distance(x[:, :, i], y[:, :, i]), L2_distance(x[:, :, i], -y[:, :, i]))
    distance /= patch_size ** 2

    return distance

def extract_image_patches(image, patch_size, stride):
    """
    从输入图像张量中提取固定大小的图像块。

    参数:
        image (torch.Tensor): 输入图像张量，形状为 [n, c, h, w]。
        patch_size (int): 图像块的大小（高度和宽度）。
        stride (int): 滑动窗口的步长。

    返回:
        patches (torch.Tensor): 提取的图像块张量，形状为 [n, c * patch_size * patch_size, h_out, w_out]。
    """
    batch_size, channels, height, width = image.shape

    h_out = (height - patch_size) // stride + 1
    w_out = (width - patch_size) // stride + 1

    patches = F.unfold(image, kernel_size=patch_size, stride=stride)

    # 调整形状
    patches = patches.contiguous().view(batch_size, channels * patch_size * patch_size, h_out, w_out)

    return patches

class Loss_forward(torch.nn.Module):
    def __init__(self, sample_size=100, h=0.5, patch_size=7, orientation_weight=0, occurrence_weight=0, colorstyle_weight=0):
        super(Loss_forward, self).__init__()
        self.sample_size = sample_size
        self.patch_size = patch_size
        self.stride = max(patch_size // 2, 2)
        self.h = h
        self.orientation_weight = orientation_weight
        self.occurrence_weight = occurrence_weight
        self.colorstyle_weight = colorstyle_weight

    def extract_features(self, feature, sample_field=None):
        if self.patch_size > 1:
            feature = extract_image_patches(feature, self.patch_size, self.stride) 
        N, C, H, W = feature.shape
        if N*H*W > self.sample_size**2:
            if sample_field==None:
                sample_field = torch.rand(N, self.sample_size, self.sample_size, 2, device=feature.device)*2 - 1
            feature = F.grid_sample(feature, sample_field, mode='nearest')
        return feature, sample_field

    def cal_distance(self, target_feature, refer_feature, style_content_feature=None, orientation=None):
        origin_target_size = target_feature.shape[-2:]
        origin_refer_size = refer_feature.shape[-2:]
        origin_target_feature = target_feature

        # feature
        target_feature, target_field = self.extract_features(target_feature)
        refer_feature, refer_field = self.extract_features(refer_feature)
        d_total = cosine_distance(target_feature, refer_feature)

        # colorstyle transfer
        if self.colorstyle_weight > 0 and not (style_content_feature==None):
            with torch.no_grad():
                color_style_transfer = ColorStyleTransfer()
                style_loss = color_style_transfer(origin_target_feature, style_content_feature)
                d_total += style_loss * self.colorstyle_weight

        # orientation
        if self.orientation_weight > 0 and not (orientation==None):
            with torch.no_grad():
                target_orient, refer_orient = orientation
                target_orient = F.interpolate(target_orient, origin_target_size)
                refer_orient = F.interpolate(refer_orient, origin_refer_size)

                target_orient = self.extract_features(target_orient, target_field)[0]
                refer_orient = self.extract_features(refer_orient, refer_field)[0]

                d_orient = orient_distance(target_orient, refer_orient, self.patch_size)
                
            d_total += d_orient * self.orientation_weight

        # Occurrence penalty
        if self.occurrence_weight > 0:
            with torch.no_grad():
                min_indices = torch.argmin(d_total, dim=-1, keepdim=True)
                normalization_factor = d_total.size(1) / d_total.size(2)
                penalty = torch.zeros(d_total.size(0), d_total.size(2), device=d_total.device)
                unique_indices, occurrence_counts = min_indices[0, :, 0].unique(return_counts=True)
                penalty[:, unique_indices] = occurrence_counts.float() / normalization_factor
                penalty = penalty.view(1, 1, -1)
                
                d_total += penalty * self.occurrence_weight

        return d_total
    def cal_loss(self, distance_matrix):
        """
        计算引导对应损失。

        参数:
            distance_matrix (torch.Tensor): 距离矩阵，形状为 [N, H*W, H*W]。

        返回:
            loss (torch.Tensor): 计算得到的损失值。
        """
        # 找到每个目标特征的最小距离
        min_distance = torch.min(distance_matrix, dim=-1, keepdim=True)[0]

        # 计算相对距离
        relative_distance = distance_matrix / (min_distance + 1e-5)

        # 计算权重
        weights = torch.exp((1 - relative_distance) / self.h)

        # 归一化权重
        normalized_weights = weights / torch.sum(weights, dim=-1, keepdim=True)

        # 计算每个样本的纹理损失
        max_weights = torch.max(normalized_weights, dim=-1)[0]
        loss = -torch.log(max_weights).mean()

        return loss

    def forward(self, target_features, reference_features, style_content_feature=None, orientation=None):
        """
        前向传播。

        参数:
            target_features (torch.Tensor): 目标特征图，形状为 [N, C, H, W]。
            reference_features (torch.Tensor): 参考特征图，形状为 [N, C, H, W]。
            orientations (tuple, optional): 目标方向图和参考方向图。

        返回:
            loss (torch.Tensor): 计算得到的损失值。
        """
        # 计算距离矩阵
        distance_matrix = self.cal_distance(target_features, reference_features, style_content_feature, orientation)

        # 计算损失
        loss = self.cal_loss(distance_matrix)

        return loss
