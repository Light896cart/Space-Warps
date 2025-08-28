import pandas as pd
import os


def process_chunks_inplace(input_dir):
    """
    Последовательно обрабатывает каждый CSV-файл в input_dir.
    Удаляет строки:
        - где 'id' отсутствует (NaN или пустой)
        - где 'obj_type' == 'Not found'
    Изменяет файлы на месте, перезаписывая их.
    Выводит статус для каждого файла.

    input_dir: путь к директории с чанками (например, 'spall_csv_chunks_lazy/')
    """
    # Получаем и сортируем список файлов
    chunk_files = sorted(
        [f for f in os.listdir(input_dir) if f.startswith("spall_chunk_") and f.endswith(".csv")]
    )

    print(f"Найдено {len(chunk_files)} файлов для обработки.")

    for filename in chunk_files:
        file_path = os.path.join(input_dir, filename)

        print(f"Обработка: {filename}...", end="")

        # Читаем чанк, сохраняя id как строку
        df = pd.read_csv(file_path, dtype={'id': 'object'})
        initial_count = len(df)

        # Фильтрация: удаляем строки без id
        df = df[df['id'].notna()]
        df = df[df['id'].astype(str).str.strip() != '']

        # Если есть столбец obj_type, удаляем 'Not found'
        if 'class' in df.columns:
            df = df[df['class'] != 'Not found']

        final_count = len(df)

        # Перезаписываем файл
        df.to_csv(file_path, index=False)

        print(f" строки: {initial_count} → {final_count} ({final_count - initial_count:+d})")

    print("Обработка завершена.")

process_chunks_inplace(r'/data/raw_data')

# def count_and_list_unique_obj_types(csv_path):
#     """
#     Считывает CSV файл и выводит количество уникальных значений в столбце 'obj_type',
#     а также список этих уникальных значений.
#     """
#     df = pd.read_csv(csv_path)
#     unique_values = df['obj_type'].dropna().unique()
#     unique_count = len(unique_values)
#
#     print(f"Количество уникальных объектов в 'obj_type': {unique_count}")
#     print("Уникальные значения в 'obj_type':")
#     for value in sorted(unique_values):
#         print(value)
#
#     return unique_values
#
# count_and_list_unique_obj_types('spall_csv_chunks_lazy/spall_chunk_0002.csv')
