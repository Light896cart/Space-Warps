"""
model_architecture.py

Описание:
    Базовая архитектура нейронной сети, реализованная на PyTorch.
    Модель можно наследовать или модифицировать под конкретную задачу
    (классификация, регрессия, семантическая сегментация и т.д.).

Автор: Senior ML Engineer
Дата: 2025-04-05
"""
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from torchvision import transforms

# Простой transforms: изменение размера и преобразование в тензор
transform = transforms.Compose([
    transforms.ToTensor(),          # Преобразовать в тензор PyTorch
])

class ProgressiveModel(nn.Module):
    def __init__(self, extra_input_dim=2,class_data=2):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = None

        self.class_data = class_data
        self.extra_input_dim = extra_input_dim

        self.extra_proj = nn.Sequential(
            nn.Linear(self.extra_input_dim, 32),  # расширяем скаляр в вектор 32
            nn.ReLU(),
            nn.Dropout(0.3)  # можно убрать, если не нужно
        )


    def add_block(self, in_channels, out_channels,check_add=0):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(2)
        )
        if check_add == 0:
            self.blocks.append(block)
        else:
            del self.blocks[-1]
            self.blocks.append(block)
        self._update_classfic(out_channels)

    # def _update_classfic(self,output_line):
    #     self.classifier = nn.Sequential(
    #         nn.Dropout(0.2),
    #         nn.Linear(output_line + (32 if self.extra_input_dim > 0 else 0), self.class_data)
    #     )

    def _update_classfic(self,output_line):
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(output_line + (32 if self.extra_input_dim > 0 else 0), self.class_data)
        )
    def forward(self, x, extra=None):
        # Обработка изображения
        for block in self.blocks:
            x = block(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)

        if extra is not None:
            extra_features = self.extra_proj(extra.float())  # [B, 32]
            x = torch.cat([x, extra_features], dim=1)
        x = self.classifier(x)
        return x
