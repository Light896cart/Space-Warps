from typing import Optional, Tuple

import pandas as pd
import torch
from pathlib import Path
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import Subset
from src.preprocessing.augmentation import train_transformer
from torchvision import transforms


def create_train_val_dataloaders(
    csv_path: Optional[str] = None,
    img_dir_path: Optional[str] = None,
    train_transform: Optional[transforms.Compose] = None,
    val_transform: Optional[transforms.Compose] = None,
    fraction: float = 1.0,
    train_ratio: float = 0.9,
    batch_size: int = 64,
    seed: int = 42,
    num_workers: int = 3,
    pin_memory: bool = False,
    dataset: Dataset = None,
) -> Tuple[DataLoader, DataLoader]:
    """
        Создаёт DataLoader'ы для обучения и валидации с разными трансформациями,
        поддержкой доли данных, воспроизводимым разбиением.

        Args:
            csv_path: Путь к CSV с метками. Если None — используется дефолт.
            img_dir_path: Путь к папке с изображениями.
            train_transform: Трансформации для train (augmentations). Если None — используется дефолт.
            val_transform: Трансформации для val (без аугментаций). Если None — используется дефолт.
            fraction: Доля данных для использования (например, 0.1 для 10%).
            train_ratio: Доля train в подмножестве (остальное — val).
            batch_size: Размер батча.
            seed: Seed для воспроизводимости.
            num_workers: Количество процессов для загрузки данных.
            pin_memory: Использовать ли pinned memory (ускоряет передачу на GPU).

        Returns:
            Кортеж (train_loader, val_loader)

        Raises:
            FileNotFoundError: Если не найден CSV или папка с изображениями.
            ValueError: При некорректных аргументах.
        """
    # --- Валидация аргументов ---
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    # --- 📁 Определение путей ---

    if csv_path is None:
        csv_path = r'D:\Code\Space_Warps\data\reg\balanced_2001_by_class_cycle.csv'
    else:
        csv_path = Path(csv_path)

    if img_dir_path is None:
        img_dir_path = r'D:\Code\Space_Warps\data\image_data\img_csv'
    else:
        img_dir_path = Path(img_dir_path)

    if not csv_path:
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if not img_dir_path:
        raise FileNotFoundError(f"Image directory not found: {img_dir_path}")

    # --- Дефолтные трансформации ---
    if val_transform is None:
        val_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    if train_transform is None:
        train_transform = train_transformer

    # --- 🧩 Создаём один датасет без transform (будем применять через collate_fn или в Subset?) ---
    # Но лучше: использовать один датасет, а transform задавать в Subset через обёртку
    # Однако PyTorch не позволяет менять transform у Subset напрямую → обход: держать два датасета, но с shared data

    # 💡 Оптимизация: используем один раз чтение CSV, кэшируем метки
    df = pd.read_csv(csv_path)
    # total_size = len(df)
    total_size = 25000

    subset_size = int(fraction * total_size)
    if subset_size == 0:
        raise ValueError("Fraction is too small, subset_size = 0")

    # --- 🔁 Воспроизводимое разбиение ---
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total_size, generator=generator).tolist()
    subset_indices = indices[:subset_size]

    train_split = int(train_ratio * len(subset_indices))
    train_indices = subset_indices[:train_split]
    val_indices = subset_indices[train_split:]

    if len(train_indices) == 0:
        raise ValueError("Train split is empty after applying fraction and ratio.")
    if len(val_indices) == 0:
        raise ValueError("Validation split is empty after applying fraction and ratio.")

# --- 📦 Два экземпляра датасета с разными трансформациями ---
    # Это нормально: они делят данные (если датасет не грузит всё в память сразу)
    train_dataset = dataset(csv_path=str(csv_path), img_dir_path=str(img_dir_path), transform=train_transform)
    val_dataset = dataset(csv_path=str(csv_path), img_dir_path=str(img_dir_path), transform=val_transform)

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    # --- 🚚 DataLoader ---
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        generator=generator,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        generator=generator,
    )

    print(f"✅ DataLoader created:")
    print(f"   Total dataset size: {total_size}")
    print(f"   Used fraction: {fraction:.1%} → {subset_size} samples")
    print(f"   Train: {len(train_indices)} | Val: {len(val_indices)}")
    print(f"   Batch size: {batch_size}, Workers: {num_workers}")

    return train_loader, val_loader