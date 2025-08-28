import pandas as pd
import os
#
#
# def create_label_mapping_from_first_n_chunks(data_dir, n=5, column='class'):
#     """
#     Собирает уникальные значения из столбца `obj_type` первых n чанков,
#     создаёт маппинг в цифровые метки и применяет его ко всем чанкам в папке.
#
#     data_dir: путь к директории с чанками (например, 'spall_csv_chunks_lazy/')
#     n: количество первых файлов для анализа уникальных значений
#     column: имя столбца для обработки
#     """
#     # Получаем первые n файлов
#     chunk_files = sorted(
#         [f for f in os.listdir(data_dir) if f.startswith("chunk_") and f.endswith(".csv")]
#     )[:n]
#
#     # Собираем все уникальные значения
#     unique_labels = set()
#     for filename in chunk_files:
#         file_path = os.path.join(data_dir, filename)
#         df = pd.read_csv(file_path)
#         if column in df.columns:
#             unique_labels.update(df[column].dropna().astype(str))
#
#     # Создаём маппинг метка -> цифра
#     label_to_id = {label: idx for idx, label in enumerate(sorted(unique_labels))}
#     print("Маппинг классов:")
#     for label, idx in label_to_id.items():
#         print(f"{label} -> {idx}")
#
#     # Применяем маппинг ко всем файлам в директории
#     all_files = [f for f in os.listdir(data_dir) if f.startswith("chunk_") and f.endswith(".csv")]
#     for filename in all_files:
#         file_path = os.path.join(data_dir, filename)
#         df = pd.read_csv(file_path)
#
#         if column in df.columns:
#             df[column] = df[column].astype(str).map(label_to_id)
#             df.to_csv(file_path, index=False)  # Перезаписываем файл
#
#     print(f"Обработка завершена. Всего уникальных классов: {len(label_to_id)}")
#
# dir_path = r'D:\Code\Space_Warps\data\raw_data'
#
# create_label_mapping_from_first_n_chunks(dir_path)

import pandas as pd


# def count_and_list_unique_obj_types(csv_path):
#     """
#     Считывает CSV файл и выводит количество уникальных значений в столбце 'obj_type',
#     а также список этих уникальных значений.
#     """
#     df = pd.read_csv(csv_path)
#     unique_values = df['subclass'].dropna().unique()
#     unique_count = len(unique_values)
#
#     print(f"Количество уникальных объектов в 'subclass': {unique_count}")
#     print("Уникальные значения в 'subclass':")
#     for value in sorted(unique_values):
#         print(value)
#
#     return unique_values



# count_and_list_unique_obj_types(r'D:\Code\Space_Warps\data\reg\balanced_2001_by_class_cycle.csv')
#
# rqqew = r'D:\Code\Space_Warps\data\reg\balanced_2001_by_class_cycle.csv'
#
# import pandas as pd
#
# # Читаем файл: табуляция как разделитель, кавычки вокруг строк
# df = pd.read_csv(
#     rqqew,          # ← замени на имя твоего файла
#     sep='\t',
#     quotechar='"',
#     engine='python',
#     header=0              # первая строка — заголовок
# )
#
# # Убираем возможные пробелы в названиях столбцов
# df.columns = df.columns.str.strip()
#
# # Сохраняем как обычный CSV: запятые, без кавычек, без индекса
# df.to_csv(rqqew, sep=',', index=False, quoting=None)
#
# print("Готово! Сохранено в 'output.csv'")