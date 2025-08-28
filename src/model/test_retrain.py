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
        k_candidates = [3, 9, 27, 81]  # например

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 🌱 Воспроизводимость
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 🧱 Начинаем с пустой модели
    model = ProgressiveModel(extra_input_dim=0,class_data=2).to(device)
    channel_image = 3
    for quantity_blocks in range(max_layers):
        weight_model = None
        best_k = None
        best_c = None
        best_candidate_state = None
        best_acc_for_layer = 0.0
        bias_model = None
        for i,candidat in enumerate(k_candidates):
            model.add_block(channel_image,candidat,check_add=i)

            if weight_model is not None:
                model.blocks[-1][0].weight.data.copy_(weight_model)
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
            acc = result["best_val_acc"]
            if copy_weights:
                new_weight = result["new_weight"]
                new_bias = result["new_bias"]
                weight_model = new_weight.detach().repeat(3, 1, 1, 1)  # обязательно detach()
                bias_model = new_bias.detach().repeat(3) if new_bias is not None else None
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
                best_c = channel_image
                best_candidate_state = {k: v.detach().clone() for k, v in model.state_dict().items()} # сохраняем веса
                print(f'НУ ЭТО СОХРАНЕНКА {best_candidate_state}')
                print(f'НУ ЭТО СОХРАНЕНКА shav{best_candidate_state.shape}')
