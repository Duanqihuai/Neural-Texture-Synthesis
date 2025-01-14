import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tf
import math
import PIL.Image as Image
import cv2


# def gradient_x(img):
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
#     img_ = cv2.GaussianBlur(img, (5,5), 1)
#     grad_x = cv2.Sobel(img_,cv2.CV_64F,1,0)

#     return grad_x

# def gradient_y(img):
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
#     img_ =cv2.GaussianBlur(img, (5,5), 1)
#     grad_y = cv2.Sobel(img_,cv2.CV_64F,0,1)
#     return grad_y

class DirectionFeatureExtractor(nn.Module):
    def __init__(self, num_bins=9, cell_size=16, min_angle=0, max_angle=180, eps=1e-9):
        super(DirectionFeatureExtractor, self).__init__()
        self.num_bins = num_bins
        self.cell_size = cell_size
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.eps = eps

        # Gray filter for RGB to grayscale conversion
        gray_filter = torch.FloatTensor([0.299, 0.587, 0.114])
        self.register_buffer('gray_filter', gray_filter[None, :, None, None])

        # Gradient filters for Sobel operator
        gradient_filter = torch.FloatTensor([-1, 0, 1])
        self.register_buffer('grad_x_filter', gradient_filter[None, None, None, :])
        self.register_buffer('grad_y_filter', gradient_filter[None, None, :, None].clone())

        # Binning parameters
        bin_interval = max_angle / num_bins
        bin_centers = torch.linspace(bin_interval / 2, max_angle - bin_interval / 2, num_bins)
        self.register_buffer('bin_centers', bin_centers[None, :, None, None])

        # Aggregate filter for HOG feature extraction
        self.register_buffer('aggregate_filter', self.create_aggregate_filter())

    def create_aggregate_filter(self):
        # Aggregate filter related to distance
        cell_size = self.cell_size
        aggregate_filter = torch.zeros(2 * cell_size + 1, 2 * cell_size + 1)
        filter_center = cell_size
        max_length = filter_center * 2 ** 0.5
        min_length = 0
        for i in range(2 * cell_size + 1):
            for j in range(2 * cell_size + 1):
                aggregate_filter[i, j] = ((i - filter_center) ** 2 + (j - filter_center) ** 2) ** 0.5
        aggregate_filter = 1 - (aggregate_filter - min_length) / (max_length - min_length)

        return aggregate_filter[None, None, :, :].expand(self.num_bins, 1, -1, -1)

    def compute_hog(self, input):
        # Calculate gradient
        grad_x = F.conv2d(input, self.grad_x_filter, padding=[0, 1])
        grad_y = F.conv2d(input, self.grad_y_filter, padding=[1, 0])
        intensity = torch.sqrt(grad_x ** 2 + grad_y ** 2)

        # Binning
        orientation = (torch.atan2(grad_y, grad_x) + math.pi) % math.pi / math.pi * self.max_angle
        orientation_bins = torch.clamp(
            1 - torch.abs(orientation - self.bin_centers) * self.num_bins / self.max_angle, 0, 1)

        # HOG extraction
        weighted_orientation_bins = orientation_bins * intensity
        padded_bins = F.pad(weighted_orientation_bins, [self.cell_size] * 4, mode='reflect')
        hog = F.conv2d(padded_bins, self.aggregate_filter, stride=self.cell_size, groups=self.num_bins)
        hog = hog / torch.norm(hog, dim=1).view(hog.shape[0], -1).max(-1)[0][:, None, None, None]

        return hog

    def extract_dominant_direction(self, hog):
        # Dominant orientation
        batch_size, _, hog_height, hog_width = hog.shape
        orientation_matrix = torch.stack([
            torch.sin(self.bin_centers / self.max_angle * math.pi) * hog,
            -torch.cos(self.bin_centers / self.max_angle * math.pi) * hog
        ], dim=1)
        orientation_matrix = torch.cat([orientation_matrix, -orientation_matrix], 2).permute(0, 3, 4, 1, 2).reshape(-1, 2, self.num_bins * 2)
        _, s, v = torch.svd(torch.bmm(orientation_matrix, orientation_matrix.permute(0, 2, 1)))
        v = v[:, :, 0]
        dominant_orientation = v.view(batch_size, hog_height, hog_width, 2).permute(0, 3, 1, 2)

        return dominant_orientation

    def resize_orientation(self, orientation, target_size):
        orientation_image = torch.cat([orientation, torch.ones_like(orientation[:, :1]) * -1], 1).cpu()

        output_orientations = []
        for index in range(orientation.shape[0]):
            current_image = (((orientation_image + 1) / 2) * 255)[index].permute(1, 2, 0).type(torch.uint8)
            current_image = Image.fromarray(current_image.numpy())
            current_image = tf.functional.resize(current_image, target_size, interpolation=tf.InterpolationMode.LANCZOS)
            output_orientation = torch.tensor(np.array(current_image), device=orientation.device)
            output_orientation = (output_orientation[:, :, :2] / 255 - 0.5) * 2
            output_orientation = output_orientation.permute(2, 0, 1)[None]
            output_orientation = output_orientation / output_orientation.norm(dim=1)
            output_orientations.append(output_orientation)

        return torch.cat(output_orientations, 0)

    def forward(self, input):
        input_size = input.shape[-2:]

        # Preprocess
        assert len(input.shape) == 4, 'Input data format is incorrect'
        if input.shape[1] == 3:
            input = F.conv2d(input, self.gray_filter)

        hog = self.compute_hog(input)
        orientation = self.extract_dominant_direction(hog)
        orientation = self.resize_orientation(orientation, input_size)

        return [orientation]

