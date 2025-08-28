import random

import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split

from src.model.model_architecture import ProgressiveModel
from src.preprocessing.augmentation import train_transformer
from src.utils.dataset import Space_Galaxi
from torchvision import transforms

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Простой transforms: изменение размера и преобразование в тензор
transform = transforms.Compose([
    transforms.ToTensor(),          # Преобразовать в тензор PyTorch
    transforms.Normalize(mean=[0.1], std=[0.15])
])

# Корректные пути
csv_path = r'D:\Code\Space_Warps\data\reg\balanced_2001_by_class_cycle.csv'
img_dir_path = r'D:\Code\Space_Warps\data\image_data\img_csv'

# 🟢 Создаём датасет ДВАЖДЫ — с разными трансформациями
train_dataset_full = Space_Galaxi(csv_path, img_dir_path=img_dir_path, transform=train_transformer)
val_dataset_full = Space_Galaxi(csv_path, img_dir_path=img_dir_path, transform=transform)

# Общий размер
total_size = len(train_dataset_full)

# Берём 100% данных (или 5%, если хочешь маленький сет)
subset_size = int(1.0 * total_size)  # или int(0.05 * total_size)
remaining = total_size - subset_size

# Создаём индексы для подмножества
indices = list(range(total_size))
random.shuffle(indices)
subset_indices = indices[:subset_size]

# Разбиваем индексы на train/val
train_val_split = int(0.9 * len(subset_indices))
train_indices = subset_indices[:train_val_split]
val_indices = subset_indices[train_val_split:]

# --- Создаём подмножества с разными transform ---
train_subset = torch.utils.data.Subset(train_dataset_full, train_indices)
val_subset = torch.utils.data.Subset(val_dataset_full, val_indices)

train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)
# --- Параметры ---
num_models = 19  # c от 1 до 19
epochs = 20

# --- Списки для хранения результатов по моделям ---
best_val_accs = []     # лучшая val_acc для каждой модели
best_val_losses = []   # соответствующий val_loss
arch_configs = []      # 3*c — количество каналов

# --- Предполагается, что train_loader и val_loader уже определены ---
# Пример:
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# --- Функции обучения и валидации (без device) ---
def train_and_evaluate_model(model, train_loader, val_loader, epochs=10, lr=1e-3):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(epochs):
        # Обучение
        for images, labels,extra in train_loader:
            optimizer.zero_grad()
            outputs = model(images,extra)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Валидация
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels,extra in val_loader:
                outputs = model(images,extra)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_acc = correct / total

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        model.train()

    return best_val_acc

print("🚀 Начинаем прогрессивное построение архитектуры")

model = ProgressiveModel()

best_val_acc = 0.0
in_channels = 1  # RGB
history = []     # для логов: (глубина, k, каналы, acc)

print("🔍 Пошаговое наращивание модели...")

layer_idx = 0
max_layers = 10  # защита от бесконечного роста

while layer_idx < max_layers:
    print(f"\n--- Попытка добавить слой {layer_idx + 1} ---")
    best_k = None
    best_c = None
    best_acc_for_layer = 0.0
    c = 3
    # Перебираем k от 1 до 20 → каналы = 3*k
    for k in range(3):
        c = 3 * c
        # Копируем текущую модель + добавляем блок
        candidate = ProgressiveModel()
        # Воссоздаём все предыдущие слои
        prev_out = in_channels
        for block in model.blocks:
            in_ch = block[0].in_channels
            out_ch = block[0].out_channels
            candidate.add_block(in_ch, out_ch)
            prev_out = out_ch
        # Добавляем новый блок
        candidate.add_block(prev_out, c)

        # Оцениваем
        acc = train_and_evaluate_model(candidate, train_loader, val_loader, epochs=8)
        print(f"  Слой {layer_idx+1}, k={k} → {c} каналов: val_acc = {acc:.4f}")

        if acc > best_acc_for_layer:
            best_acc_for_layer = acc
            best_k = k
            best_c = c

    # Проверяем, улучшилось ли
    if best_acc_for_layer > best_val_acc:
        # Добавляем лучший блок в финальную модель
        model.add_block(in_channels if layer_idx == 0 else model.blocks[-1][0].out_channels, best_c)
        best_val_acc = best_acc_for_layer
        history.append((layer_idx + 1, best_k, best_c, best_val_acc))
        print(f"✅ Добавлен слой {layer_idx + 1}: k={best_k} → {best_c} каналов, acc = {best_val_acc:.4f}")
        layer_idx += 1
    else:
        print("❌ Новый слой не улучшил результат — останавливаемся.")
        break

# ===================================================
# 6. Итог
# ===================================================
if layer_idx == 0:
    print("❌ Не удалось обучить даже один слой.")
else:
    print("\n" + "="*60)
    print("🏆 ФИНАЛЬНАЯ МОДЕЛЬ")
    print(f"🔹 Глубина: {len(model.blocks)} слоя")
    print(f"🔹 Лучшая точность: {best_val_acc:.4f}")
    print("🔹 Архитектура:")
    for depth, k, c, acc in history:
        print(f"   Слой {depth}: k={k} → {c} каналов, val_acc = {acc:.4f}")
    print("="*60)

    # График
    depths = [h[0] for h in history]
    channels = [h[2] for h in history]
    plt.figure(figsize=(10, 4))
    plt.bar(depths, channels, color='lightgreen', edgecolor='k')
    plt.title('Прогрессивная архитектура (каналы = 3*k)')
    plt.xlabel('Номер слоя')
    plt.ylabel('Число каналов')
    plt.xticks(depths)
    plt.grid(axis='y', alpha=0.3)
    plt.show()