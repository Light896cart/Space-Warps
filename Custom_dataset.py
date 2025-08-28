# def progressive_architecture_search(
#     train_loader: DataLoader,
#     val_loader: DataLoader,
#     max_layers: int = 10,
#     k_candidates: List[int] = None,
#     epochs_per_eval: int = 5,
#     lr: float = 1e-3,
#     device: str = None,
#     seed: int = 42,
#     copy_weights:bool = True
#
# ) -> Tuple[ProgressiveModel, List[dict]]:
#     """
#     Пошагово наращивает модель, выбирая наилучший блок (по val_acc) на каждом шаге.
#
#     Args:
#         train_loader, val_loader: данные
#         max_layers: макс. число блоков
#         k_candidates: список значений k (каналы = base_channels * k)
#         epochs_per_eval: сколько эпох учить каждый кандидат
#         lr: learning rate
#         device: cuda/cpu
#         seed: воспроизводимость
#
#     Returns:
#         Лучшая модель и история поиска
#     """
#     if k_candidates is None:
#         k_candidates = [1, 3, 9, 27]  # например
#
#     if device is None:
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#
#     # 🌱 Воспроизводимость
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#
#     # 🧱 Начинаем с пустой модели
#     model = ProgressiveModel(extra_input_dim=0,class_data=2).to(device)
#     base_channels = 3  # для RGB
#     best_val_acc = 0.0
#     history = []
#
#     print("🔍 Запуск прогрессивного поиска архитектуры...")
#
#     for layer_idx in range(max_layers):
#         print(f"\n--- [Шаг {layer_idx + 1}] Подбор следующего блока ---")
#         best_k = None
#         best_c = None
#         best_acc_for_layer = 0.0
#         best_candidate_state = None
#         weight_model = None
#         bias_model = None
#
#         prev_out_channels = 3 if len(model.blocks) == 0 else model.blocks[-1][0].out_channels
#
#         # 💡 Перебираем возможные k
#         for k in k_candidates:
#
#             channels = base_channels * k
#             print(f"  Оценка: блок → {prev_out_channels} → {channels} каналов (k={k})")
#
#             # 🔁 Копируем ТЕКУЩУЮ архитектуру
#             candidate = ProgressiveModel(extra_input_dim=0,class_data=2)
#             # candidate = torch.compile(candidate, mode="max-autotune")
#             # Воссоздаём все существующие блоки
#             for block in model.blocks:
#                 in_ch = block[0].in_channels
#                 out_ch = block[0].out_channels
#                 candidate.add_block(in_ch, out_ch)
#             # Добавляем новый
#             candidate.add_block(prev_out_channels, channels)
#             print(f'Это кандидат {candidate}')
#             print(f'Это модель {model.blocks.weight}')
#             if weight_model is not None and bias_model is not None:
#                 with torch.no_grad():
#                     candidate.blocks[-1][0].weight.data.copy_(weight_model)
#                     if candidate.blocks[-1][0].bias is not None:
#                         candidate.blocks[-1][0].bias.data.copy_(bias_model)
#             candidate.to(device)
#
#             # 🔬 Оцениваем (с теми же параметрами!)
#             result = train_and_evaluate_model(
#                 model=candidate,
#                 train_loader=train_loader,
#                 val_loader=val_loader,
#                 epochs=epochs_per_eval,
#                 lr=lr,
#                 weight_model=weight_model,
#                 device=device,
#                 save_best_path=None,  # не сохраняем каждый кандидат
#                 verbose=True,
#             )
#             acc = result["best_val_acc"]
#             if copy_weights:
#                 new_weight = result["new_weight"]
#                 new_bias = result["new_bias"]
#                 weight_model = new_weight.detach().repeat(3, 1, 1, 1)  # обязательно detach()
#                 bias_model = new_bias.detach().repeat(3) if new_bias is not None else None
#                 # 🔥 ДОБАВЛЯЕМ СЛУЧАЙНОСТЬ: weight noise (jitter)
#                 # Шум ~ N(0, σ), где σ = 0.1 * std(весов)
#                 noise_std = 0.03 * weight_model.std()
#                 weight_model += torch.randn_like(weight_model) * noise_std
#
#                 # Аналогично для bias (если есть)
#                 if bias_model is not None:
#                     bias_noise_std = 0.1 * bias_model.std()
#                     bias_model += torch.randn_like(bias_model) * bias_noise_std
#             else:
#                 weight_model = None
#                 bias_model = None
#             print(f"    → val_acc = {acc:.4f}")
#             if acc > best_acc_for_layer:
#                 best_acc_for_layer = acc
#                 best_k = k
#                 best_c = channels
#                 best_candidate_state = {k: v.detach().clone() for k, v in candidate.state_dict().items()} # сохраняем веса
#         # 🏁 Проверяем, стоит ли расти
#         weight_model = None
#         if best_acc_for_layer > best_val_acc:
#             # ✅ Добавляем лучший блок в ОСНОВНУЮ модель
#             model.add_block(prev_out_channels, best_c)
#             model.load_state_dict(best_candidate_state)  # ← переносим обученные веса!
#             best_val_acc = best_acc_for_layer
#
#             history.append({
#                 "layer": layer_idx + 1,
#                 "k": best_k,
#                 "channels": best_c,
#                 "val_acc": best_val_acc
#             })
#
#             print(f"✅ Добавлен блок {layer_idx + 1}: k={best_k} → {best_c} каналов, acc={best_val_acc:.4f}")
#         else:
#             print("❌ Нет улучшения — останов.")
#             break
#
#     print(f"\n✅ Поиск завершён. Лучшая модель: {len(model.blocks)} блоков, acc={best_val_acc:.4f}")
#     return model, history
import random
ver = 50
reg = random.uniform(-ver,ver)
print(reg)
