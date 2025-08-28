from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model.model_architecture import ProgressiveModel
from src.model.train_eval import train_and_evaluate_model



def progressive_architecture_search(
    train_loader: DataLoader,
    val_loader: DataLoader,
    max_layers: int = 10,
    k_candidates: List[int] = None,
    epochs_per_eval: int = 5,
    lr: float = 1e-3,
    device: str = None,
    seed: int = 42,
    copy_weights:bool = True
):
    """
    Пошагово наращивает модель, выбирая наилучший блок (по val_acc) на каждом шаге.

    Args:
        train_loader, val_loader: данные
        max_layers: макс. число блоков
        k_candidates: список значений k (каналы = base_channels * k)
        epochs_per_eval: сколько эпох учить каждый кандидат
        lr: learning rate
        device: cuda/cpu
        seed: воспроизводимость

    Returns:
        Лучшая модель и история поиска
    """
    if k_candidates is None:
        k_candidates = [[3, 9, 27, 81]]  # например

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 🌱 Воспроизводимость
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 🧱 Начинаем с пустой модели
    model = ProgressiveModel(extra_input_dim=0,class_data=2).to(device)
    best_candidate_state = None
    channel_image = 3
    best_val_acc = 0.0
    best_k = None
    best_c = None
    history = []
    print("🔍 Запуск прогрессивного поиска архитектуры...")
    for quantity_blocks in range(len(k_candidates)):
        print(f"\n--- [Шаг {quantity_blocks + 1}] Подбор следующего блока ---")
        # log_system_usage(f"Начало шага {quantity_blocks + 1}")
        weight_model = None
        best_acc_for_layer = 0.0
        bias_model = None
        for i,candidat in enumerate(k_candidates[quantity_blocks]):
            # log_system_usage(f"Перед add_block {i+1}")
            out_channels = channel_image if quantity_blocks == 0 else model.blocks[-1][0].out_channels if i==0 else model.blocks[-2][0].out_channels
            model.add_block(out_channels,candidat,check_add=i)

            # log_system_usage(f"После add_block {i + 1}")
            # 🔁 Переносим веса, если есть предыдущие
            if weight_model is not None and len(model.blocks) > 0:
                target_conv = model.blocks[-1][0]  # последний добавленный слой
                src_shape = weight_model.shape[1]  # [N, C, K, K] → C = in_channels
                tgt_in_ch = target_conv.weight.shape[1]

                # Если входные каналы не совпадают — это проблема, но предположим, что совпадают
                # Копируем веса с учётом числа выходных каналов
                num_copy = min(target_conv.out_channels, weight_model.shape[0])

                # Копируем первые num_copy фильтров
                with torch.no_grad():
                    target_conv.weight.data[:num_copy, :, :, :].copy_(weight_model[:num_copy, :, :, :])

                    # Если целевой слой больше — инициализируем остаток
                    if target_conv.out_channels > num_copy:
                        # Инициализируем новые фильтры
                        new_filters = target_conv.out_channels - num_copy
                        # Используем Kaiming или малый шум
                        nn.init.kaiming_normal_(
                            target_conv.weight.data[num_copy:, :, :, :],
                            mode='fan_in',
                            nonlinearity='relu'
                        )

            # 🔬 Оцениваем (с теми же параметрами!)
            result = train_and_evaluate_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=epochs_per_eval,
                lr=lr,
                weight_model=weight_model,
                device=device,
                save_best_path=None,  # не сохраняем каждый кандидат
                verbose=True,
            )
            # log_system_usage(f"После обучения кандидата {i + 1}")
            acc = result["best_val_acc"]
            if copy_weights:
                new_weight = result["new_weight"]
                new_bias = result["new_bias"]
                max_index = len(k_candidates[quantity_blocks]) - 1
                next_idx = min(i + 1, max_index)
                next_candidat = k_candidates[quantity_blocks][next_idx]
                number_repeat = next_candidat // candidat
                print(f'🔁 number_repeat = {number_repeat} (candidat={candidat}, next={next_candidat})')
                weight_model = new_weight.detach().repeat(number_repeat, 1, 1, 1)  # обязательно detach()
                bias_model = new_bias.detach().repeat(number_repeat) if new_bias is not None else None
                # 🔥 ДОБАВЛЯЕМ СЛУЧАЙНОСТЬ: weight noise (jitter)
                # Шум ~ N(0, σ), где σ = 0.1 * std(весов)
                noise_std = 0.03 * weight_model.std()
                weight_model += torch.randn_like(weight_model) * noise_std

                # Аналогично для bias (если есть)
                if bias_model is not None:
                    bias_noise_std = 0.1 * bias_model.std()
                    bias_model += torch.randn_like(bias_model) * bias_noise_std
            else:
                weight_model = None
                bias_model = None
            print(f"    → val_acc = {acc:.4f}")
            if acc > best_acc_for_layer:
                best_acc_for_layer = acc
                best_k = candidat
                best_c = out_channels
                best_candidate_state = {k: v.detach().clone() for k, v in model.state_dict().items()} # сохраняем веса
        # 🏁 Проверяем, стоит ли расти
        weight_model = None
        # log_system_usage(f"После очистки")
        if best_acc_for_layer > best_val_acc:
            model.add_block(best_c, best_k, check_add=1)
            model.load_state_dict(best_candidate_state, strict=False)
            best_val_acc = best_acc_for_layer

            history.append({
                "layer": quantity_blocks + 1,
                "k": best_k,
                "channels": best_c,
                "val_acc": best_val_acc
            })

            print(f"✅ Добавлен блок {quantity_blocks + 1}: k={best_k} → {best_c} каналов, acc={best_val_acc:.4f}")
        else:
            print("❌ Нет улучшения — останов.")
            break

    print(f"\n✅ Поиск завершён. Лучшая модель: {len(model.blocks)} блоков, acc={best_val_acc:.4f}")
    return model, history
