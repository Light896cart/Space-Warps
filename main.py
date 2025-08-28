import torch

from src.data.dataloader import create_train_val_dataloaders
from src.data.dataset import Space_Galaxi
from src.model.progressive_search import progressive_architecture_search
from src.utils.logging import summarize_progressive_growth
from src.utils.seeding import set_seed


def main():
    # 🌱 1. Воспроизводимость
    print("🔧 Устанавливаем seed для воспроизводимости...")
    set_seed(42)
    print('hi')
    ver = r'D:\Code\Space_Warps\train'
    # 📥 2. Подготовка данных
    print("\n📥 Загружаем и разбиваем данные...")
    train_loader, val_loader = create_train_val_dataloaders(
        csv_path=None,                # использовать дефолтный путь
        img_dir_path=ver,
        dataset=Space_Galaxi,
        train_ratio=0.9,
        fraction=0.02,                 # использовать 100% данных
        batch_size=32,
        num_workers=0,
        seed=42                       # для DataLoader
    )
    # 🧠 3. Прогрессивный поиск архитектуры
    print("\n🚀 Запуск прогрессивного поиска архитектуры...")
    model, history = progressive_architecture_search(
        train_loader=train_loader,
        val_loader=val_loader,
        max_layers=10,
        k_candidates=[[9, 21, 54,120],[21,50,86,140],[20,40,60,80],[9,21,54,90]],    # k: множитель для числа каналов (каналы = 3 * k)
        epochs_per_eval=1,            # сколько эпох учить каждый кандидат
        lr=1e-3,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=42,
        copy_weights=True
    )

    # 📊 4. Итоговый отчёт
    if len(history) > 0:
        best_val_acc = max(h["val_acc"] for h in history)  # ✅ исправлено
    else:
        best_val_acc = 0.0

    print("\n📊 Генерация финального отчёта...")
    summary = summarize_progressive_growth(
        history=history,
        model=model,
        best_val_acc=best_val_acc,
        save_path=r"D:\Code\Space_Warps\result\progressive_summary.json",
        show_plot=True
    )

    # ✅ 5. Сохранение лучшей модели
    if summary:
        model_save_path = "results/best_progressive_model.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"✅ Лучшая модель сохранена: {model_save_path}")

    print("\n🎉 Обучение завершено.")


# Запуск
if __name__ == "__main__":
    main()
