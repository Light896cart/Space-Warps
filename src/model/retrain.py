import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from src.model.model_architecture import ProgressiveModel
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
])

# Корректные пути
csv_path = r'D:\Code\Space_Warps\data\reg\new.csv'
img_dir_path = r'D:\Code\Space_Warps\data\image_data\img_csv_0001'

dataset = Space_Galaxi(csv_path, img_dir_path=img_dir_path,transform=transform)
# Задаём размеры: например, 80% на обучение, 20% на валидацию
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

# Разделяем случайным образом
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Создаём DataLoader'ы
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# model = ProgressiveModel()
# model.add_block(1,3)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# criterion = nn.CrossEntropyLoss()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# Симуляция метрики
prev_metric = 0.0
patience = 5
no_improve = 0

# Списки для хранения метрик
train_losses = []
val_losses = []
train_accs = []
val_accs = []



for c in range(1,20):
    model = ProgressiveModel()
    model.add_block(1, 3*c)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_val_acc = 0.0
    for epoch in range(20):
        # --- Обучение ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()










            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total

        # --- Валидация ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total

        # # Сохраняем метрики
        # train_losses.append(avg_train_loss)
        # val_losses.append(avg_val_loss)
        # train_accs.append(train_acc)
        # val_accs.append(val_acc)
        # Сохраняем лучшую модель
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # torch.save(model.state_dict(), "best_model.pth")
            print(f"✅ Сохранена лучшая модель: val_acc = {best_val_acc}")
            print(f"**** Сохранена лучшая модель: loss = {avg_val_loss}")

    val_accs.append(best_val_acc)
    val_losses.append(avg_val_loss)
    print("НОВАЯ МОДЕЛЬ")
        # # Лог
        # print(f"Epoch [{epoch+1}/100]")
        # print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}")
        # print(f"  Val Loss:   {avg_val_loss:.4f}   | Val Acc:   {val_acc:.4f}")
        # print("-" * 50)


plt.figure(figsize=(12, 5))

# График 1: Валидационный лосс
plt.subplot(1, 2, 1)
plt.plot(val_losses, label='Validation Loss', color='red', linewidth=2)
plt.title('Validation Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)
plt.legend()

# График 2: Валидационная точность
plt.subplot(1, 2, 2)
plt.plot(val_accs, label='Validation Accuracy', color='green', linewidth=2)
plt.title('Validation Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.grid(True, alpha=0.3)
plt.legend()

# Оформление
plt.tight_layout()
plt.show()

#
# # === Построение графиков ===
# epochs_range = range(1, len(train_losses) + 1)
#
# plt.figure(figsize=(12, 5))
#
# # График Loss
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, train_losses, label='Train Loss', marker='o')
# plt.plot(epochs_range, val_losses, label='Val Loss', marker='s')
# plt.title('Loss по эпохам')
# plt.xlabel('Эпоха')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(True)
#
# # График Accuracy
# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, train_accs, label='Train Accuracy', marker='o')
# plt.plot(epochs_range, val_accs, label='Val Accuracy', marker='s')
# plt.title('Accuracy по эпохам')
# plt.xlabel('Эпоха')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.grid(True)
#
# plt.tight_layout()
# plt.show()