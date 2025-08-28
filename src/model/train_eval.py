from typing import Optional, Dict, Any

import torch.optim as optim
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

def train_and_evaluate_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-3,
    device: str = None,
    save_best_path: str = "best_model.pth",
    verbose: bool = True,
    seed: int = 42,
    weight_model: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """
    Обучает модель и оценивает её на валидации. Сохраняет лучшую по accuracy модель.
    Поддерживает прогрессивное обучение: использует текущие веса модели (warm-start).

    Args:
        model: PyTorch модель (уже может иметь предобученные слои)
        train_loader: DataLoader для обучения
        val_loader: DataLoader для валидации
        epochs: Количество эпох обучения
        lr: Скорость обучения
        device: 'cuda' или 'cpu'. Если None — определяется автоматически.
        save_best_path: Путь для сохранения лучшей модели
        verbose: Показывать ли прогресс и логи
        seed: Seed для воспроизводимости внутри обучения

    Returns:
        Словарь с результатами:
            - best_val_acc: лучшая accuracy на валидации
            - train_losses: loss по эпохам
            - val_accuracies: accuracy по эпохам
            - model: модель с лучшими весами
            - device: используемое устройство
    """
    # --- 🌱 Воспроизводимость ---
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # --- 🖥️ Device setup ---
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # --- ⚙️ Оптимизатор и лосс ---
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    # --- 📈 Логи ---
    train_losses = []
    val_accuracies = []
    best_val_acc = 0.0
    best_model_wts = None
    weight_model_new = None
    bias_model_new = None

    # --- 🔁 Цикл обучения ---
    for epoch in range(epochs):
        # --- 🟠 Обучение ---
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{epochs}",
            unit="batch",
            disable=not verbose,
            leave=False
        )

        for batch in progress_bar:
            # Поддержка разных форматов: (img, label, extra) или (img, label)
            if len(batch) == 3:
                images, labels, extra = batch
            else:
                images, labels = batch
                extra = None

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Обработка extra
            if extra is not None:
                extra = extra.to(device, non_blocking=True)

            optimizer.zero_grad()
            if extra is not None:
                outputs = model(images, extra)
            else:
                outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # --- 🟢 Валидация ---
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    images, labels, extra = batch
                else:
                    images, labels = batch
                    extra = None

                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                if extra is not None:
                    extra = extra.to(device, non_blocking=True)
                    outputs = model(images, extra)
                else:
                    outputs = model(images)

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_acc = correct / total
        val_accuracies.append(val_acc)
        # --- 🏆 Сохранение лучшей модели ---
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_model_wts = model.state_dict().copy()

            last_conv = model.blocks[-1][0]
            weight_model_new = last_conv.weight  # [C_out, C_in, k, k]
            bias_model_new = last_conv.bias if last_conv.bias is not None else None

            if save_best_path:
                torch.save(best_model_wts, save_best_path)
                if verbose:
                    print(f"🔥 Best model updated: val_acc = {val_acc:.4f}")

        # --- 📢 Логирование ---
        if verbose:
            print(f"Epoch [{epoch+1:2d}/{epochs}] | Loss: {epoch_loss:6.4f} | Val Acc: {val_acc:6.4f} | Best: {best_val_acc:6.4f}")

    # --- ✅ Восстанавливаем лучшую модель ---
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    # --- 📦 Возвращаем результаты ---
    return {
        "best_val_acc": best_val_acc,
        "train_losses": train_losses,
        "val_accuracies": val_accuracies,
        "model": model,
        "new_weight": weight_model_new,
        "new_bias": bias_model_new,  # Добавлено
        "device": device,
    }
