import random

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms import InterpolationMode
import numpy as np

# --- Параметры ---
IMG_SIZE = 224
PAD_SIZE = 40
CROP_SIZE = IMG_SIZE

# --- Кастомный шум ---
class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.02):
        self.mean = mean
        self.std = std
    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std + self.mean
        return tensor + noise


class RandomRotationWithAngle:
    def __init__(self, degrees, interpolation=InterpolationMode.BILINEAR, expand=False):
        self.degrees = degrees
        self.interpolation = interpolation
        self.expand = expand

    def __call__(self, img):
        rotate = random.uniform(-self.degrees, self.degrees)

        # Поворачиваем изображение
        img_rotated = T.functional.rotate(img, rotate, interpolation=self.interpolation, expand=self.expand)

        return {
            'image': img_rotated,
            'rotation_angle': img_rotated
        }

class DictTransform:
    """Обёртка, которая применяет трансформацию только к 'image' в словаре"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, data):
        # data = {'image': PIL/tensor, 'rotation_angle': float}
        img = data['image']
        angle = data['rotation_angle']

        # Применяем трансформацию к изображению
        img_transformed = self.transform(img)

        return {
            'image': img_transformed,
            'rotation_angle': angle
        }

    def __repr__(self):
        return f"DictTransform({self.transform})"

# # --- Аугментация ---
# --- Параметры ---
IMG_SIZE = 224

# --- Аугментация для обучения ---
train_transformer = T.Compose([
    # Начинаем с PIL Image
    T.ToTensor(),
    T.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BILINEAR),
    # Кастомный поворот — уже работает с dict
    # DictTransform(T.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BILINEAR)),
    # DictTransform(T.RandomHorizontalFlip(p=0.5)),
    #
    # # Остальные аугментации — оборачиваем в DictTransform
    # DictTransform(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)),
    # DictTransform(T.RandomResizedCrop(
    #     IMG_SIZE,
    #     scale=(0.8, 1.0),
    #     ratio=(0.9, 1.1),
    #     interpolation=InterpolationMode.BILINEAR
    # )),
    # DictTransform(T.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3), value='random')),
    # DictTransform(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])),
])
