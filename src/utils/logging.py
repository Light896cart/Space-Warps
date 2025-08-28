import json
from typing import Optional, Dict, Any, List
import matplotlib.pyplot as plt
import torch

def summarize_progressive_growth(
    history: List[Dict[str, Any]],  # ✅ Исправлено: ожидаем словари
    model: torch.nn.Module,
    best_val_acc: float,
    save_path: Optional[str] = "progressive_architecture.json",
    show_plot: bool = True,
):
    if not history or best_val_acc == 0.0:
        print("❌ Не удалось обучить ни один слой. Проверь:")
        print("   - Корректность данных и трансформаций")
        print("   - Скорость обучения и оптимизатор")
        print("   - Реализацию ProgressiveModel.add_block()")
        return

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "=" * 60)
    print("🏆 ФИНАЛЬНАЯ МОДЕЛЬ УСПЕШНО ПОСТРОЕНА")
    print("=" * 60)
    print(f"🔹 Глубина модели: {len(history)} слоёв")
    print(f"🔹 Лучшая val accuracy: {best_val_acc:.4f}")
    print(f"🔹 Общее число обучаемых параметров: {total_params:,}")

    print("\n🔹 Архитектура по слоям:")
    print(f"{'Слой':<6} {'k':<4} {'Каналы':<8} {'Accuracy':<10}")
    print("-" * 40)
    for h in history:
        depth = h["layer"]
        k = h["k"]
        c = h["channels"]
        acc = h["val_acc"]
        print(f"{depth:<6} {k:<4} {c:<8} {acc:<10.4f}")  # ✅ Теперь acc — float

    print("=" * 60)

    # 📊 Визуализация
    if show_plot:
        fig, ax1 = plt.subplots(figsize=(10, 5))
        depths = [h["layer"] for h in history]
        channels = [h["channels"] for h in history]
        accuracies = [h["val_acc"] for h in history]

        bars = ax1.bar(depths, channels, color='skyblue', edgecolor='navy', alpha=0.8, label='Каналы')
        ax1.set_xlabel('Номер слоя', fontsize=12)
        ax1.set_ylabel('Число каналов', color='blue', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_xticks(depths)
        ax1.set_ylim(0, max(channels) * 1.1)

        for bar, ch in zip(bars, channels):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(channels)*0.02,
                     f'{ch}', ha='center', fontsize=10, color='navy')

        ax2 = ax1.twinx()
        ax2.plot(depths, accuracies, color='darkred', marker='o', linewidth=2, markersize=6, label='Accuracy')
        ax2.set_ylabel('Validation Accuracy', color='red', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim(min(accuracies) * 0.9, max(accuracies) * 1.05)

        plt.title('Прогрессивный рост модели: архитектура и качество', fontsize=14, pad=20)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')

        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legend_elements = [
            Patch(facecolor='skyblue', edgecolor='navy', label='Число каналов'),
            Line2D([0], [0], color='darkred', marker='o', label='Val Accuracy')
        ]
        ax1.legend(handles=legend_elements, loc='upper left')

        fig.tight_layout()
        plt.show()

    # 📦 Сохранение
    summary = {
        "final_val_accuracy": round(best_val_acc, 6),
        "num_layers": len(history),
        "total_trainable_params": total_params,
        "architecture": [
            {
                "layer": h["layer"],
                "k": h["k"],
                "channels": h["channels"],
                "val_accuracy": round(h["val_acc"], 6)
            }
            for h in history
        ],
        "model_str": str(model)
    }

    if save_path:
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=4, ensure_ascii=False)
            print(f"✅ Результаты сохранены в: {save_path}")
        except Exception as e:
            print(f"❌ Не удалось сохранить JSON: {e}")

    return summary