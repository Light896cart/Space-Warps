# -*- coding: utf-8 -*-
"""
Ленивое чтение spAll-v6_1_3.fits и создание CSV файлов по N строк.
Извлекаются столбцы: specObjID, RA, DEC.
"""

import os
from astropy.io import fits
import pandas as pd


def lazy_create_csv_chunks(fits_filename, chunk_size=1000, output_dir='spall_csv_chunks'):
    """
    Ленивое чтение spAll FITS файла и создание CSV файлов по N строк.
    Поля: id, ra, dec, class, subclass, z, z_err.

    Параметры:
    fits_filename (str): Путь к файлу spAll-v6_1_3.fits.
    chunk_size (int): Количество строк в каждом CSV файле.
    output_dir (str): Директория для сохранения CSV файлов.
    """

    import os
    import pandas as pd
    from astropy.io import fits

    # Создаем директорию для выходных файлов
    os.makedirs(output_dir, exist_ok=True)

    print(f"Открываем {fits_filename}...")

    with fits.open(fits_filename, memmap=True) as hdul:
        hdu = hdul[1]  # BinTableHDU

        try:
            total_rows = len(hdu.data)
        except TypeError:
            total_rows = hdu.data.shape[0]

        print(f"Общее количество строк: {total_rows}")

        # --- Поиск нужных колонок ---
        colnames = [col.name for col in hdu.columns]

        print(colnames)

        col_mapping = {
            'id': ['SPECOBJID', 'specobjid'],
            'ra': ['PLUG_RA', 'RA', 'plug_ra', 'ra'],
            'dec': ['PLUG_DEC', 'DEC', 'plug_dec', 'dec', 'DECCAT'],
            'class': ['CLASS', 'class'],
            'subclass': ['SUBCLASS', 'subclass'],
            'z': ['Z', 'z'],
            'z_err': ['Z_ERR', 'z_err', 'ZERROR']
        }

        col_idx = {}
        for key, names in col_mapping.items():
            col_idx[key] = None
            for name in names:
                if name in colnames:
                    col_idx[key] = colnames.index(name)
                    print(f"✅ {key} -> '{name}' (индекс {col_idx[key]})")
                    break
            if col_idx[key] is None:
                print(f"❌ {key} не найден. Проверьте имя колонки.")
                return  # Прерываем, если не найдено критическое поле

        # Проверим обязательные поля
        required = ['id', 'ra', 'dec', 'class', 'z', 'z_err']
        for key in required:
            if col_idx[key] is None:
                print(f"Критическая ошибка: отсутствует поле '{key}'")
                return

        # --- Чтение и запись чанков ---
        file_counter = 1
        buffer = []

        for i in range(total_rows):
            try:
                record = hdu.data[i]
            except Exception as e:
                continue  # пропускаем битые строки

            try:
                row = {
                    'id': record[col_idx['id']],
                    'ra': record[col_idx['ra']],
                    'dec': record[col_idx['dec']],
                    'class': record[col_idx['class']],
                    'subclass': record[col_idx['subclass']],
                    'z': record[col_idx['z']],
                    'z_err': record[col_idx['z_err']]
                }
                buffer.append(row)
            except Exception as e:
                print(f"Ошибка при обработке строки {i}: {e}")
                continue

            if len(buffer) >= chunk_size:
                filename = os.path.join(output_dir, f"chunk_{file_counter:04d}.csv")
                pd.DataFrame(buffer).to_csv(filename, index=False)
                print(f"💾 Сохранено: {filename}")
                buffer = []
                file_counter += 1

        # Последний чанк
        if buffer:
            filename = os.path.join(output_dir, f"chunk_{file_counter:04d}.csv")
            pd.DataFrame(buffer).to_csv(filename, index=False)
            print(f"💾 Сохранено (остаток): {filename}")

    print(f"✅ Готово: {file_counter - 1} файлов создано.")

output_dir = r'D:\Code\Space_Warps\data\raw_data'
# --- Основная часть ---
if __name__ == "__main__":
    fits_file_path = r'D:\Code\Space_Warps\spAll-v6_1_3.fits'

    if not os.path.exists(fits_file_path):
        print(f"Ошибка: Файл '{fits_file_path}' не найден.")
    else:
        lazy_create_csv_chunks(fits_file_path, chunk_size=1000, output_dir=output_dir)
