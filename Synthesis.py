import torch
import cv2
import numpy as np
import argparse
import torchvision.transforms.functional as TF
from torchvision import transforms
from tqdm import tqdm
from VGG19 import get_vgg19











if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural texture synthesis')
    parser.add_argument('--model', type=str, default='vgg19')
    args = parser.parse_args()

