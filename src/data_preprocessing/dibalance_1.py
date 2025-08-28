import pandas as pd
import os
from itertools import cycle

# Пути
input_dir = r'D:\Code\Space_Warps\data\raw_data'
output_file = r'D:\Code\Space_Warps\data\reg\balanced_2001_by_class_cycle.csv'

# Создаём папку для результата
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Параметры
target_total = 2001
chunk_prefix = "chunk_"
chunk_ext = ".csv"

# Получаем список файлов в порядке: spall_chunk_0001.csv, 0002, ...
files = sorted([
    f for f in os.listdir(input_dir)
    if f.startswith(chunk_prefix) and f.endswith(chunk_ext)
])

if not files:
    raise FileNotFoundError(f"Не найдено файлов с префиксом '{chunk_prefix}' в директории {input_dir}")

print(f"Найдено {len(files)} файлов для обработки.")

# Проверка наличия нужных столбцов
required_columns = {'id', 'class'}

# Итератор по файлам
file_iter = iter(files)
result_rows = []  # Список для итоговых строк (каждая — pd.Series или dict)
all_classes = set()

# Функция загрузки следующего файла
def load_next_file():
    try:
        filename = next(file_iter)
        path = os.path.join(input_dir, filename)
        print(f"Читаю файл: {filename}")
        df = pd.read_csv(path)
        if not required_columns.issubset(df.columns):
            print(f"Пропускаю {filename}: отсутствуют столбцы {required_columns - set(df.columns)}")
            return None
        return df.copy()  # Возвращаем весь DataFrame со всеми столбцами
    except StopIteration:
        return None

# Загружаем первый файл
current_df = load_next_file()
if current_df is None:
    raise ValueError("Нет подходящих данных в файлах.")

# Обновляем список классов
all_classes.update(current_df['class'].unique())

# Основной цикл: собираем ровно 2001 строк
while len(result_rows) < target_total:
    # Сортируем классы для предсказуемого порядка (можно перемешать при желании)
    ordered_classes = sorted(all_classes)
    class_cycle = cycle(ordered_classes)

    while len(result_rows) < target_total:
        target_class = next(class_cycle)

        found = False

        # Попробуем найти строку с нужным классом в текущем DataFrame
        if current_df is not None and len(current_df) > 0:
            class_mask = current_df['class'] == target_class
            if class_mask.any():
                # Берём первую попавшуюся строку этого класса
                row = current_df.loc[class_mask].iloc[0]
                result_rows.append(row)  # Сохраняем всю строку
                # Удаляем её из текущего DataFrame
                current_df = current_df.drop(class_mask.idxmax())  # idxmax() — первый True индекс
                found = True

        # Если не нашли — пробуем загрузить следующий файл
        if not found:
            current_df = load_next_file()
            if current_df is None:
                print("⚠️ Все файлы обработаны, но не набрано 2001 строк.")
                break
            all_classes.update(current_df['class'].unique())

    # Если файлы закончились, выходим
    if current_df is None:
        break

# Обрезаем, если вдруг собрали больше (маловероятно, но на всякий случай)
if len(result_rows) > target_total:
    result_rows = result_rows[:target_total]

# Создаём итоговый DataFrame — все столбцы сохраняются
result = pd.DataFrame(result_rows)

# Приводим id и class к строкам (для безопасности при экспорте)
result['id'] = result['id'].astype(str)
result['class'] = result['class'].astype(str)

# Сохраняем все столбцы
result.to_csv(
    output_file,
    index=False,
    quoting=1,          # Кавычки вокруг всех полей
    quotechar='"',
    escapechar='\\'
)

# Статистика
print(f"\n✅ Готово! Сохранено {len(result)} строк в файл:")
print(output_file)

print(f"\nРаспределение по классам:")
print(result['class'].value_counts().sort_index())

print(f"\nДоступные столбцы: {list(result.columns)}")